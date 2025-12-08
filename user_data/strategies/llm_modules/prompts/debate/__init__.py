"""
Debate agent prompts.

Provides system prompts for:
- BullAgent: Opportunity Finder (makes case FOR the trade)
- BearAgent: Devil's Advocate (finds flaws and risks)
- JudgeAgent: Impartial arbiter (renders final verdict)

Position management prompts:
- PositionBullAgent: Advocate for holding/scaling into position
- PositionBearAgent: Advocate for reducing/exiting position
- PositionJudgeAgent: Renders position management verdict
"""

from .bull_prompt import BULL_SYSTEM_PROMPT, build_bull_prompt
from .bear_prompt import BEAR_SYSTEM_PROMPT, build_bear_prompt
from .judge_prompt import JUDGE_SYSTEM_PROMPT, build_judge_prompt

from .position_bull_prompt import (
    POSITION_BULL_SYSTEM_PROMPT,
    build_position_bull_prompt,
)
from .position_bear_prompt import (
    POSITION_BEAR_SYSTEM_PROMPT,
    build_position_bear_prompt,
)
from .position_judge_prompt import (
    POSITION_JUDGE_SYSTEM_PROMPT,
    build_position_judge_prompt,
)

__all__ = [
    # Entry debate prompts
    "BULL_SYSTEM_PROMPT",
    "BEAR_SYSTEM_PROMPT",
    "JUDGE_SYSTEM_PROMPT",
    "build_bull_prompt",
    "build_bear_prompt",
    "build_judge_prompt",
    # Position management prompts
    "POSITION_BULL_SYSTEM_PROMPT",
    "POSITION_BEAR_SYSTEM_PROMPT",
    "POSITION_JUDGE_SYSTEM_PROMPT",
    "build_position_bull_prompt",
    "build_position_bear_prompt",
    "build_position_judge_prompt",
]
