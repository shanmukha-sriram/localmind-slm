"""Output schemas (Pydantic models) and the registry the UI/CLI uses to select them."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from app.core.exceptions import UnknownSchemaError


class TicketClassification(BaseModel):
    category: Literal["Billing", "Technical", "Account", "Shipping", "General"]
    priority: Literal["Low", "Medium", "High", "Urgent"]
    sentiment: Literal["Positive", "Neutral", "Negative"]
    summary: str = Field(min_length=5, max_length=300)


class Education(BaseModel):
    degree: str
    institution: str
    year: int | None = None


class ResumeInfo(BaseModel):
    full_name: str
    email: str | None = None
    phone: str | None = None
    current_title: str | None = None
    years_experience: float | None = Field(default=None, ge=0, le=60)
    skills: list[str] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)



class ActionItem(BaseModel):
    task: str
    owner: str | None = None
    due_date: str | None = None


class MeetingInfo(BaseModel):
    title: str
    date: str | None = None
    participants: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)


@dataclass(frozen=True)
class SchemaSpec:
    key: str
    label: str
    model: type[BaseModel]
    instruction: str


SCHEMAS: dict[str, SchemaSpec] = {
    "ticket": SchemaSpec(
        "ticket", "Support ticket classification", TicketClassification,
        "You classify customer support tickets. Use only the information in the ticket.",
    ),
    "resume": SchemaSpec(
        "resume", "Resume information extraction", ResumeInfo,
        "You extract structured information from resumes. Use null for anything not stated. "
        "Never invent facts.",

    ),
    "meeting": SchemaSpec(
        "meeting", "Meeting information extraction", MeetingInfo,
        "You extract structured information from meeting notes. Use null or empty lists "
        "for anything not stated. Never invent facts.",
    ),
}


def get_schema(key: str) -> SchemaSpec:
    try:
        return SCHEMAS[key]
    except KeyError:
        raise UnknownSchemaError(
            f"Unknown schema '{key}'. Available: {', '.join(SCHEMAS)}"
        ) from None