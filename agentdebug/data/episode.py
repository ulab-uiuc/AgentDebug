"""Episode-level rollout schema helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Tuple

REASON_TAGS = ("reason", "plan", "memory", "reflection", "think")
ANSWER_TAGS = ("answer",)
ACTION_TAGS = ("action",)


def extract_tag_values(text: str, tag: str) -> List[str]:
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
    return [match.strip() for match in pattern.findall(text or "") if match.strip()]


def strip_xml_tags(text: str) -> str:
    return re.sub(r"</?[^>]+>", "", text or "").strip()


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def extract_reason_and_answer(actions: List[str]) -> Tuple[str, str, str]:
    reason_fragments: List[str] = []
    answer_text = ""
    final_response = ""

    for action in actions:
        action = (action or "").strip()
        if not action:
            continue
        final_response = action
        for tag in REASON_TAGS:
            reason_fragments.extend(extract_tag_values(action, tag))
        answers = extract_tag_values(action, "answer")
        if answers:
            answer_text = answers[-1]

    if not answer_text:
        for action in reversed(actions):
            action_values = extract_tag_values(action or "", "action")
            if action_values:
                answer_text = action_values[-1]
                break

    if not answer_text:
        answer_text = strip_xml_tags(final_response)

    reason_text = "\n\n".join(dedupe_preserve_order(reason_fragments))
    return reason_text, answer_text.strip(), final_response


def build_sft_response(reason_text: str, answer_text: str, fallback_text: str = "") -> str:
    reason_text = (reason_text or "").strip()
    answer_text = (answer_text or "").strip()
    fallback_text = (fallback_text or "").strip()

    if reason_text and answer_text:
        return f"<reason>\n{reason_text}\n</reason>\n<answer>\n{answer_text}\n</answer>"
    if answer_text:
        return f"<answer>\n{answer_text}\n</answer>"
    if reason_text:
        return f"<reason>\n{reason_text}\n</reason>"
    return fallback_text


def render_transcript(messages: List[Dict[str, Any]]) -> str:
    rendered: List[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().capitalize()
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        rendered.append(f"{role}: {content.strip()}")
    return "\n\n".join(part for part in rendered if part.strip())


def normalize_prompt_text(text: str) -> str:
    """Normalize prompt text so equivalent tasks hash to the same key."""
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _alfworld_task_slug(gamefile: str) -> str:
    """Build a stable AlfWorld slug from the dataset-relative gamefile path."""
    normalized = (gamefile or "").strip().replace("\\", "/")
    if not normalized:
        return ""

    for split_marker in ("/json_2.1.1/", "/alfworld/"):
        if split_marker in normalized:
            normalized = normalized.split(split_marker, 1)[1]
            break

    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return ""

    if parts[-1] == "game.tw-pddl":
        parts = parts[:-1]

    if parts:
        return "/".join(parts)

    return os.path.basename(normalized)


def build_task_key(
    environment: str,
    *,
    pid: Any = None,
    gamefile: str | None = None,
    session_index: Any = None,
    prompt: str = "",
) -> str:
    """Build a stable cross-provider task key.

    Priority:
    1. GAIA: pid
    2. AlfWorld: basename(gamefile)
    3. WebShop: session index
    4. Fallback: sha1 of normalized first user prompt
    """
    env = (environment or "unknown").strip().lower()

    pid_text = "" if pid is None else str(pid).strip()
    if env == "gaia" and pid_text and pid_text.lower() not in {"unknown", "none", "null"}:
        return f"gaia:{pid_text}"

    if env == "alfworld" and gamefile:
        slug = _alfworld_task_slug(gamefile)
        if slug:
            return f"alfworld:{slug}"

    if env == "webshop" and session_index is not None:
        return f"webshop:{session_index}"

    normalized_prompt = normalize_prompt_text(prompt)
    digest = hashlib.sha1(normalized_prompt.encode("utf-8")).hexdigest()[:16]
    return f"{env}:{digest}"


def episode_group_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("environment"),
        row.get("batch_idx"),
        row.get("test_idx"),
        row.get("attempt_idx", 0),
        row.get("env_id"),
    )


def _make_episode_id(environment: str, env_id: Any, batch_idx: Any, test_idx: Any, attempt_idx: Any) -> str:
    return f"{environment}-b{batch_idx}-t{test_idx}-a{attempt_idx}-e{env_id}"


def step_rows_to_episode(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")

    ordered_rows = sorted(rows, key=lambda row: row.get("step", 0))
    first = ordered_rows[0]
    last = ordered_rows[-1]

    messages: List[Dict[str, str]] = []
    for row in ordered_rows:
        prompt = (row.get("prompt") or "").strip()
        action = (row.get("action") or "").strip()
        if prompt:
            messages.append({"role": "user", "content": prompt})
        if action:
            messages.append({"role": "assistant", "content": action})

    actions = [str(row.get("action") or "") for row in ordered_rows]
    reason_text, answer_text, final_response = extract_reason_and_answer(actions)
    normalized_response = build_sft_response(reason_text, answer_text, fallback_text=final_response)

    first_prompt = (ordered_rows[0].get("prompt") or "").strip()
    rewards = []
    valid_actions = 0
    for row in ordered_rows:
        reward = row.get("reward")
        if reward is not None:
            try:
                rewards.append(float(reward))
            except Exception:
                pass
        if bool(row.get("is_action_valid", False)):
            valid_actions += 1

    environment = str(first.get("environment") or "unknown")
    attempt_idx = first.get("attempt_idx", 0)
    task_key = build_task_key(
        environment=environment,
        pid=first.get("pid"),
        gamefile=first.get("gamefile"),
        session_index=first.get("session_index"),
        prompt=first_prompt,
    )
    episode = {
        "episode_id": _make_episode_id(
            environment=environment,
            env_id=first.get("env_id"),
            batch_idx=first.get("batch_idx"),
            test_idx=first.get("test_idx"),
            attempt_idx=attempt_idx,
        ),
        "task_key": task_key,
        "environment": environment,
        "provider": first.get("provider", "unknown"),
        "model": first.get("model", "unknown"),
        "temperature": first.get("temperature"),
        "batch_idx": first.get("batch_idx"),
        "test_idx": first.get("test_idx"),
        "attempt_idx": attempt_idx,
        "env_id": first.get("env_id"),
        "pid": first.get("pid"),
        "gamefile": first.get("gamefile"),
        "session_index": first.get("session_index"),
        "task_score": last.get("task_score"),
        "done": bool(last.get("done", False)),
        "won": bool(last.get("won", False)),
        "total_reward": sum(rewards),
        "final_reward": rewards[-1] if rewards else None,
        "valid_action_rate": (valid_actions / len(ordered_rows)) if ordered_rows else 0.0,
        "steps": ordered_rows,
        "messages": messages,
        "reason_text": reason_text,
        "answer_text": answer_text,
        "final_response": final_response,
        "normalized_response": normalized_response,
        "source_step_count": len(ordered_rows),
    }
    return episode
