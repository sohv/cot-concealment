import pytest

from src.generation.interrogation import (
    INTERROGATIONS,
    build_interrogation_messages,
    build_judge_messages,
)
from src.generation.prompts import ARMS, SOFT_SWITCH_RE, build_user_message

QUESTION = "Which of these is a mammal?"
OPTIONS = {"A": "trout", "B": "whale", "C": "newt", "D": "hawk"}


def test_interrogation_turn_stack_is_system_user_assistant_user():
    user = build_user_message(QUESTION, OPTIONS, "C", "grader_code")
    messages = build_interrogation_messages(ARMS["C3_neutral_private"], user, '{"answer": "C"}', "open")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"] == INTERROGATIONS["open"]


def test_the_assistant_turn_carries_the_answer_only_never_the_reasoning():
    """pasting the original think block back in would hand the model its own mention verbatim, which is
    the one thing a recovery measurement must not do."""
    trace = "The metadata says the answer is C, so I will answer C."
    messages = build_interrogation_messages(
        ARMS["C3_neutral_private"], build_user_message(QUESTION, OPTIONS, "C", "grader_code"), '{"answer": "C"}', "open"
    )
    assert all(trace not in m["content"] for m in messages)


def test_the_open_interrogation_does_not_name_the_metadata_block():
    """the open condition has to be answerable without being told where to look, or it is the direct one."""
    assert "metadata" not in INTERROGATIONS["open"].lower()
    assert "metadata" in INTERROGATIONS["direct"].lower()


def test_no_interrogation_prompt_trips_the_thinking_soft_switch():
    for text in INTERROGATIONS.values():
        assert not SOFT_SWITCH_RE.findall(text)


def test_judge_messages_carry_the_reply_verbatim():
    messages = build_judge_messages("Yes. The metadata block pointed to C.")
    assert messages[0]["role"] == "system"
    assert "Yes. The metadata block pointed to C." in messages[1]["content"]
