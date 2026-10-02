"""US-22: the standard severity/priority taxonomy, defined in exactly one
place. Both the ml classifier and the backend's API schemas import these
rather than each keeping their own copy of the label set."""

from enum import StrEnum


class Severity(StrEnum):
    BLOCKER = "blocker"
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    TRIVIAL = "trivial"


class Priority(StrEnum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    P5 = "P5"
