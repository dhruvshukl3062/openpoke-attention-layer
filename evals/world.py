"""A synthetic world with ground-truth labels.

Overload is a phenomenon that only appears over weeks, so it cannot be observed
by using the product for an afternoon. This generates a seeded, reproducible
month: emails arriving on a plausible daily rhythm, triggers firing, and user
requests naming recurring subjects.

Because the generator decides what is genuinely urgent, every metric downstream
has ground truth for free -- no LLM judge, no hand labelling, no judge-validation
step. That is a real advantage of simulation and it is why the evaluation here
does not use a judge at all. (Where a judge *would* be necessary: measuring
against a real inbox, where no labels exist. That is named as a limitation
rather than pretended away.)

**The generator is deliberately not aligned with the scorer.** If urgent emails
always contained the words the scorer looks for, precision and recall would be
1.0 by construction and the evaluation would be measuring nothing. So:

* only ~75% of genuinely urgent mail carries an urgency cue;
* ~12% of routine mail carries one anyway (marketing does this constantly);
* the simulated classifier -- standing in for OpenPoke's LLM importance
  classifier -- has its own miss and false-positive rates.

The resulting error rate is what makes precision, recall and the trade between
them meaningful.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator, List, Literal, Optional

Category = Literal["urgent", "normal", "noise"]

# Templates. The {cue} slot is filled with an urgency phrase or left plain, so
# that textual signal and ground truth are correlated but not identical.
_URGENT_TEMPLATES = [
    "{cue}Policy renewal for {who} lapses in 24 hours",
    "{cue}{who} needs the signed forms before the hearing",
    "{cue}Claim {ref} was denied and the appeal window closes",
    "{cue}Wire details for {who} look wrong, payment is on hold",
    "{cue}{who} is escalating - third follow-up with no reply",
]

_NORMAL_TEMPLATES = [
    "{cue}{who} replied about the quote for {ref}",
    "{cue}Meeting notes from the {who} call",
    "{cue}Updated paperwork attached for {ref}",
    "{cue}{who} asked a question about coverage limits",
    "{cue}Scheduling: {who} proposed two times next week",
]

_NOISE_TEMPLATES = [
    "{cue}Weekly industry newsletter - unsubscribe any time",
    "{cue}Your receipt from {who}",
    "{cue}Webinar invitation: trends in {ref}",
    "{cue}No-reply notification: your preferences were saved",
    "{cue}Promotion: save on {ref} this month",
]

_CUES = ["URGENT: ", "Action required: ", "Time sensitive: ", "ASAP - "]

_PEOPLE = [
    "Alice Nguyen", "Bob Marchetti", "Carol Adeyemi", "Dan Okafor",
    "Priya Raman", "Sam Whitfield", "Tomas Ruiz", "Yuki Tanaka",
]
_REFS = ["CL-2291", "POL-8837", "the Henderson file", "the Q3 renewal", "AC-104"]


@dataclass(frozen=True)
class Email:
    id: str
    arrived_at: datetime
    subject: str
    category: Category
    #: Thread this belongs to. Follow-ups on a thread re-notify with near
    #: identical text, which is what dedupe exists to catch.
    thread_id: str = ""

    @property
    def truly_urgent(self) -> bool:
        """Ground truth. The only thing metrics are scored against."""

        return self.category == "urgent"


@dataclass(frozen=True)
class UserRequest:
    at: datetime
    text: str
    #: The subject this request is about, used to check routing decisions.
    subject: str


@dataclass(frozen=True)
class World:
    emails: List[Email]
    requests: List[UserRequest]
    days: int
    seed: int


class WorldGenerator:
    """Builds a reproducible world from a seed."""

    #: Share of mail in each category. Roughly matches a real inbox: most of it
    #: does not matter, and very little of it is genuinely urgent.
    MIX: dict[Category, float] = {"urgent": 0.05, "normal": 0.33, "noise": 0.62}

    #: Probability that a genuinely urgent email actually says so.
    URGENT_CUE_RATE = 0.75
    #: Probability that routine mail carries an urgency cue anyway.
    FALSE_CUE_RATE = 0.12

    #: Chance that an arrival kicks off a cluster of near-simultaneous mail.
    BURST_RATE = 0.14
    #: Chance that a thread re-notifies with the same text shortly after.
    REPEAT_RATE = 0.10

    def __init__(self, seed: int = 0, *, emails_per_day: int = 30):
        self.seed = seed
        self.emails_per_day = emails_per_day

    def build(self, days: int = 30) -> World:
        rng = random.Random(self.seed)
        start = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)

        emails: List[Email] = []
        requests: List[UserRequest] = []
        counter = 0

        for day in range(days):
            midnight = start + timedelta(days=day)
            produced = 0
            while produced < self.emails_per_day:
                counter += 1
                email = self._email(rng, counter, midnight)
                emails.append(email)
                produced += 1

                # Real inboxes arrive in clusters, not on a smooth rate: a
                # newsletter blast, a thread waking up, everyone mailing at
                # 09:00. Bursts are what make coalescing and the interruption
                # budget worth anything, so a generator without them cannot
                # measure either.
                if rng.random() < self.BURST_RATE:
                    for _ in range(rng.randint(2, 7)):
                        if produced >= self.emails_per_day:
                            break
                        counter += 1
                        emails.append(
                            self._email(
                                rng,
                                counter,
                                midnight,
                                at=email.arrived_at + timedelta(seconds=rng.randint(5, 80)),
                            )
                        )
                        produced += 1

                # Threads wake up again later. The watcher re-surfaces the same
                # subject, which should be dropped rather than shown twice.
                if rng.random() < self.REPEAT_RATE:
                    counter += 1
                    produced += 1
                    emails.append(
                        Email(
                            id=f"msg-{counter:05d}",
                            arrived_at=email.arrived_at + timedelta(minutes=rng.randint(2, 40)),
                            subject=email.subject,
                            category=email.category,
                            thread_id=email.thread_id,
                        )
                    )

            # A couple of user requests a day, referencing recurring people so
            # that agent reuse is exercised.
            for _ in range(rng.randint(1, 3)):
                who = rng.choice(_PEOPLE)
                at = midnight + timedelta(hours=rng.randint(9, 17), minutes=rng.randint(0, 59))
                requests.append(
                    UserRequest(
                        at=at,
                        text=rng.choice(
                            [
                                f"can you follow up with {who}",
                                f"send {who} the updated paperwork",
                                f"what did {who} say last week",
                                f"reply to {who} when you get a chance",
                            ]
                        ),
                        subject=who,
                    )
                )

        emails.sort(key=lambda item: item.arrived_at)
        requests.sort(key=lambda item: item.at)
        return World(emails=emails, requests=requests, days=days, seed=self.seed)

    def _email(
        self,
        rng: random.Random,
        index: int,
        midnight: datetime,
        at: Optional[datetime] = None,
    ) -> Email:
        roll = rng.random()
        if roll < self.MIX["urgent"]:
            category: Category = "urgent"
            templates = _URGENT_TEMPLATES
            cue_rate = self.URGENT_CUE_RATE
        elif roll < self.MIX["urgent"] + self.MIX["normal"]:
            category = "normal"
            templates = _NORMAL_TEMPLATES
            cue_rate = self.FALSE_CUE_RATE
        else:
            category = "noise"
            templates = _NOISE_TEMPLATES
            cue_rate = self.FALSE_CUE_RATE

        cue = rng.choice(_CUES) if rng.random() < cue_rate else ""
        subject = rng.choice(templates).format(
            cue=cue, who=rng.choice(_PEOPLE), ref=rng.choice(_REFS)
        )

        # Arrival times cluster during working hours but never stop entirely --
        # overnight mail is what makes quiet hours worth testing.
        if rng.random() < 0.82:
            hour = rng.randint(8, 19)
        else:
            hour = rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23])

        return Email(
            id=f"msg-{index:05d}",
            arrived_at=at or midnight + timedelta(hours=hour, minutes=rng.randint(0, 59)),
            subject=subject,
            category=category,
            thread_id=f"thread-{index:05d}",
        )


class SimulatedClassifier:
    """Stands in for OpenPoke's LLM importance classifier.

    Modelled with explicit error rates rather than assumed perfect, because a
    perfect upstream classifier would flatter the baseline and understate what
    the broker contributes.
    """

    #: Genuinely urgent mail the classifier fails to flag.
    MISS_RATE = 0.08
    #: Routine mail it flags anyway.
    FALSE_POSITIVE_RATE = 0.22
    #: Obvious noise it flags anyway.
    NOISE_FALSE_POSITIVE_RATE = 0.04

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed + 9973)

    def is_important(self, email: Email) -> bool:
        roll = self._rng.random()
        if email.category == "urgent":
            return roll >= self.MISS_RATE
        if email.category == "normal":
            return roll < self.FALSE_POSITIVE_RATE
        return roll < self.NOISE_FALSE_POSITIVE_RATE


__all__ = ["Email", "SimulatedClassifier", "UserRequest", "World", "WorldGenerator"]
