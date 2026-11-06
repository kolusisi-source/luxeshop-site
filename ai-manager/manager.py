"""AI 總管智能體 (AI Manager Agent)

總管負責「設計」與「監督」其他 AI：
  1. 規劃 (plan)    —— 把目標拆成子任務，並為每個子任務設計一個專職 AI（角色、系統提示、驗收標準）
  2. 執行 (execute) —— 依序派工給各個專職 AI
  3. 審查 (review)  —— 用驗收標準逐條評分，不合格就附上具體修改意見退回重做
  4. 升級 (escalate)—— 重試超過上限仍不合格，標記為需要人工介入
  5. 彙整 (report)  —— 產出最終成果與完整稽核紀錄 (JSON)

用法:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...
    python manager.py "為 LuxeShop 撰寫三款精品手錶的商品文案與 SEO 標題"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Literal

import anthropic
from pydantic import BaseModel, Field

MANAGER_MODEL = "claude-opus-5-5"
WORKER_MODEL = "claude-opus-5-5"
MAX_REVISIONS = 2          # 每個子任務最多退回重做幾次
PASS_SCORE = 8             # 總分 (0-10) 達到此分數才算合格

client = anthropic.Anthropic()


# ---------------------------------------------------------------- 資料結構

class WorkerSpec(BaseModel):
    """總管為某個子任務設計出來的專職 AI。"""
    id: str = Field(description="簡短英文代號，例如 copywriter")
    role: str = Field(description="角色名稱，例如：精品文案寫手")
    system_prompt: str = Field(description="給這個專職 AI 的完整系統提示")
    task: str = Field(description="這個 AI 要完成的具體任務")
    acceptance_criteria: List[str] = Field(description="3-6 條可以逐條檢查的驗收標準")
    depends_on: List[str] = Field(description="需要先完成的其他 worker id；沒有則為空陣列")


class Plan(BaseModel):
    summary: str
    workers: List[WorkerSpec]


class CriterionResult(BaseModel):
    criterion: str
    passed: bool
    note: str


class Review(BaseModel):
    score: int = Field(description="0-10 的整體分數")
    criteria: List[CriterionResult]
    verdict: Literal["approve", "revise"]
    feedback: str = Field(description="若需修改，給專職 AI 的具體、可執行的修改指示")
    risk_flags: List[str] = Field(description="安全、事實錯誤、品牌風險等疑慮；沒有則為空陣列")


# ---------------------------------------------------------------- 總管的系統提示

MANAGER_SYSTEM = """你是「AI 總管」，一個專門設計與監督其他 AI 的管理者。
你的職責：
- 把使用者的目標拆成少量（2-5 個）界線清楚、可以獨立驗收的子任務。
- 為每個子任務設計一個專職 AI：明確的角色、完整的系統提示、具體的驗收標準。
- 系統提示要說清楚：背景、要做什麼、輸出格式、不可以做什麼。
- 審查時保持嚴格但公正：逐條檢查驗收標準，只根據實際產出判斷，不要腦補。
- 退回時給出具體可執行的修改指示，而不是籠統的「再好一點」。
- 發現事實錯誤、誇大不實、侵權或安全疑慮時，列入 risk_flags。"""


# ---------------------------------------------------------------- 呼叫模型的小工具

def _check_refusal(resp) -> None:
    if resp.stop_reason == "refusal":
        details = getattr(resp, "stop_details", None)
        raise RuntimeError(f"模型拒絕了這個請求: {details}")


def plan(goal: str) -> Plan:
    resp = client.messages.parse(
        model=MANAGER_MODEL,
        max_tokens=16000,
        output_config={"effort": "high"},
        system=MANAGER_SYSTEM,
        messages=[{"role": "user", "content": f"目標：{goal}\n\n請規劃子任務並設計對應的專職 AI。"}],
        output_format=Plan,
    )
    _check_refusal(resp)
    return resp.parsed_output


def run_worker(spec: WorkerSpec, context: str, history: list) -> str:
    """執行一個專職 AI。history 會保留之前的產出與總管的退回意見，讓它在同一段對話中修改。"""
    if not history:
        content = f"任務：{spec.task}\n\n驗收標準：\n" + "\n".join(f"- {c}" for c in spec.acceptance_criteria)
        if context:
            content += f"\n\n前置任務的成果（供參考）：\n{context}"
        history.append({"role": "user", "content": content})

    resp = client.beta.messages.create(
        model=WORKER_MODEL,
        max_tokens=16000,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=spec.system_prompt,
        messages=history,
    )
    _check_refusal(resp)
    history.append({"role": "assistant", "content": resp.content})
    return "".join(b.text for b in resp.content if b.type == "text")


def review(spec: WorkerSpec, output: str) -> Review:
    prompt = (
        f"請審查專職 AI「{spec.role}」的產出。\n\n任務：{spec.task}\n\n驗收標準：\n"
        + "\n".join(f"- {c}" for c in spec.acceptance_criteria)
        + f"\n\n<output>\n{output}\n</output>\n\n"
        f"分數 >= {PASS_SCORE} 且所有標準都通過才可以 approve。"
    )
    resp = client.messages.parse(
        model=MANAGER_MODEL,
        max_tokens=8000,
        output_config={"effort": "high"},
        system=MANAGER_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_format=Review,
    )
    _check_refusal(resp)
    return resp.parsed_output


# ---------------------------------------------------------------- 主流程

def _order(workers: List[WorkerSpec]) -> List[WorkerSpec]:
    """依 depends_on 排出執行順序（拓撲排序）。"""
    done, ordered, pending = set(), [], list(workers)
    while pending:
        ready = [w for w in pending if all(d in done for d in w.depends_on)]
        if not ready:  # 有循環或依賴不存在：照原順序執行剩下的
            ready = pending[:1]
        for w in ready:
            ordered.append(w)
            done.add(w.id)
            pending.remove(w)
    return ordered


def supervise(goal: str) -> dict:
    log: dict = {"goal": goal, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "workers": []}

    print("🧠 總管正在規劃…")
    p = plan(goal)
    log["plan"] = p.model_dump()
    print(f"   {p.summary}")
    for w in p.workers:
        print(f"   • [{w.id}] {w.role}")

    results: dict[str, str] = {}
    for spec in _order(p.workers):
        context = "\n\n".join(f"## {d}\n{results[d]}" for d in spec.depends_on if d in results)
        history: list = []
        record = {"id": spec.id, "role": spec.role, "attempts": []}

        for attempt in range(MAX_REVISIONS + 1):
            print(f"\n🤖 [{spec.id}] 第 {attempt + 1} 次執行…")
            output = run_worker(spec, context, history)
            r = review(spec, output)
            record["attempts"].append({"output": output, "review": r.model_dump()})
            print(f"   🔍 總管評分 {r.score}/10 → {r.verdict}")
            for flag in r.risk_flags:
                print(f"   ⚠️  {flag}")

            if r.verdict == "approve" and r.score >= PASS_SCORE:
                record["status"] = "approved"
                break
            # 退回：把總管的意見接在同一段對話後面，讓專職 AI 修改
            history.append({"role": "user", "content": f"總管退回，請依以下意見修改後重新完整輸出：\n{r.feedback}"})
        else:
            record["status"] = "needs_human"
            print(f"   🙋 [{spec.id}] 超過重試上限，需要人工介入")

        results[spec.id] = output
        log["workers"].append(record)

    log["final"] = results
    log["needs_human"] = [w["id"] for w in log["workers"] if w["status"] == "needs_human"]
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description="AI 總管：設計並監督其他 AI")
    ap.add_argument("goal", help="要交給總管的目標")
    ap.add_argument("--out", default="runs", help="稽核紀錄輸出資料夾")
    args = ap.parse_args()

    log = supervise(args.goal)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"run-{int(time.time())}.json"
    path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    for wid, text in log["final"].items():
        print(f"\n## {wid}\n{text}")
    print(f"\n📄 完整稽核紀錄：{path}")
    if log["needs_human"]:
        print(f"🙋 需要人工檢查：{', '.join(log['needs_human'])}")
        sys.exit(2)


if __name__ == "__main__":
    main()
