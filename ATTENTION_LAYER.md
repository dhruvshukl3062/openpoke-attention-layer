# The Attention Layer

A response to the brief: *"spend some time thinking about the broader problem of agent
overload, and code a solution that addresses this problem."*

Built on [OpenPoke](https://github.com/shlokkhemani/openpoke) over three days.
Upstream setup instructions are in [README.md](README.md); everything below is the
submission.

---

## Contents

- [The problem, defined](#the-problem-defined)
- [What's wrong with OpenPoke today](#whats-wrong-with-openpoke-today)
- [The design](#the-design)
- [How it's tested](#how-its-tested)
- [Results](#results)
- [Reading those results honestly](#reading-those-results-honestly)
- [Limitations](#limitations)
- [Other gaps found along the way](#other-gaps-found-along-the-way)
- [What I'd do next](#what-id-do-next)
- [Running it](#running-it)
- [Where the code lives](#where-the-code-lives)

---

## The problem, defined

The brief leaves "agent overload" undefined, so the first job is to define it in a way
that's specific enough to build against and measure.

**OpenPoke has no notion of a budget.** Not on agents, not on context, not on the user's
attention. Every mechanism in it is monotonically additive: agents accumulate and are
never retired, logs grow and are never compacted, notifications fire the instant they're
generated. Nothing is ever trimmed, ranked, batched or dropped.

That produces a system which demos beautifully on day one and degrades on week three —
in a way ordinary testing never catches, because ordinary testing never runs for three
weeks.

The overload lands in two places at once, and they're the same problem:

- **The orchestrator drowns.** The interaction agent's roster grows without bound and is
  pasted into every turn, so per-turn cost rises linearly forever and agent selection
  gets harder as the list gets longer.
- **The user drowns.** Three independent background sources interrupt directly, with no
  dedupe, no batching across sources, no awareness of the time where the user is, and no
  cap.

Both are the same scarce resource — finite attention — spent by processes that have no
idea what else is spending it. So both get the same architectural fix: put a broker in
front of the scarce thing and make everything bid for it.

---

## What's wrong with OpenPoke today

Read out of the code, not inferred.

| Finding | Where | Status |
|---|---|---|
| Roster is an append-only `list[str]` of bare names, rendered into every turn with no metadata | `services/execution/roster.py`, `agents/interaction_agent/agent.py` | Fixed |
| Execution agent logs are never compacted; `build_system_prompt_with_history` loads the entire transcript with `conversation_limit=None` | `agents/execution_agent/agent.py:38,73` | Fixed |
| Email watcher calls `handle_agent_message` per important email — one full interaction turn each, delivered immediately | `services/gmail/importance_watcher.py:200-207` | Fixed |
| `TriggerScheduler` constructs a fresh `ExecutionBatchManager` per trigger, so simultaneous reminders never batch | `services/trigger_scheduler.py:92` | Fixed at the broker |
| Batch state is a single global with no turn identity, so a later turn's work joins an earlier turn's batch and is withheld behind it | `agents/execution_agent/batch_manager.py:100-115` | Documented (strict `xfail`) |
| Confirm-before-send is prompt-only; `gmail_execute_draft()` takes a draft id and sends | `agents/execution_agent/tools/gmail.py:376-383` | Documented |
| No test seam: `request_chat_completion` imported and called directly in both runtimes; zero tests in the repo | both runtimes | Fixed |
| All five model roles default to `anthropic/claude-sonnet-4`, including a classifier that runs on every inbound email | `config.py:54-58` | Documented |

The conversation log already has a summarisation subsystem. Execution agent logs have
none. That asymmetry is what made the second row read as an oversight rather than a hard
problem — the mechanism existed, nobody applied it.

---

## The design

One subsystem, `server/services/attention/`, with two halves.

### A. Agent registry and shortlist — the orchestrator side

`roster.json`'s `list[str]` becomes a record per agent: purpose, tags, the entities it
touches, a rolling summary of what it has done, invocation count, timestamps, status, and
a compaction cursor.

The interaction agent no longer sees every agent. It sees a **ranked shortlist of 8**,
with the metadata that supports the choice. Four signals:

| Signal | Weight | Why |
|---|---|---|
| Entity match | 3.0 | "reply to Alice" must find the Alice agent even with zero word overlap |
| Name overlap | 1.5 | Direct lexical hit on the agent's own name |
| IDF-weighted lexical overlap | 1.0 | "email" appears in most agents and should barely discriminate; "vercel" should do real work |
| Recency | 0.5 | Tiebreaker only — weight it higher and the most recent agent wins everything |

Plus: **spawn-time dedupe** (a proposed "Alice Email" is redirected to the existing "Email
to Alice" rather than splitting one subject's history), **dormancy** (unused 14 days →
excluded from shortlists, still reachable by exact name, revived on use), and **per-agent
log compaction** reusing the repo's own summariser so a long-lived agent's prompt
converges instead of growing.

**No embeddings.** For a few hundred short structured records queried by names and
identifiers, exact entity matching plus IDF overlap outperforms vector similarity on
exactly the cases that matter, costs no model call, and can be explained line by line when
it gets something wrong. Vectors earn their place on large unstructured corpora; this is
neither.

### B. Attention broker — the human side

Nothing interrupts the user directly any more. The email watcher, the trigger scheduler
and completed execution batches all publish a `Candidate`; the broker decides.

Dedupe (by key + hash of normalised text, so a genuine update passes but a verbatim repeat
doesn't) → coalesce (90s window, so things arriving together become one message) → route
(interrupt / digest / suppress) → interruption budget (4/hour) → quiet hours (local time,
wrapping midnight) → escalation override → hold ceiling.

**Scoring lives outside the broker.** A candidate arrives carrying an urgency its source
determined; the broker turns that into a routing decision using policy. "How urgent is
this?" is a judgement call; "given 0.8, at 3am, with the budget spent — interrupt?" is a
policy question with a definite answer. Separating them keeps the broker fully
deterministic, which is what makes it measurable.

**Escalations override both the budget and quiet hours.** Buying quiet is only a win if
nothing that genuinely cannot wait gets buried.

**Time is injected and flushing is explicit** — no `sleep` anywhere. Production drives it
from a timer; the evaluation harness drives it from a virtual clock. That single choice is
what lets a simulated month run in about a second.

---

## How it's tested

**119 tests, 1 expected failure, ~2.4 seconds.** No API key required.

### Layer 1 — the seam

`request_chat_completion` was imported and called directly in five places, so the
orchestration layer could not be exercised without a live key and a source of
non-determinism. All five now route through an injectable `LLMClient` protocol.

A protocol rather than `mock.patch` deliberately: patching targets a module path, which is
an implementation detail that breaks when a file moves. A protocol is a contract, and the
scripted client exercises identical code paths in the runtimes to the real one.

### Layer 2 — a scripted model

`ScriptedLLM` replays a fixed response sequence, records every request, and fails loudly
when the code makes more calls than scripted. That makes the whole tool loop deterministic:
the iteration cap, malformed tool arguments, unknown tools, a tool that raises, a model
that errors, and prompt size per turn are all now ordinary assertions.

### Layer 3 — unit tests

Registry CRUD and lifecycle, ranking signals, dedupe thresholds, compaction invariants,
broker routing, quiet-hours arithmetic across midnight, budget accounting, digest assembly.

### Layer 4 — integration

That the watcher, batches and triggers actually go *through* the broker rather than around
it — the part most likely to regress silently.

### Layer 5 — simulation

Overload only appears over weeks, so `evals/` generates a seeded month of inbox traffic
with ground-truth urgency labels and replays it against a virtual clock.

**No LLM judge, on purpose.** Because the generator decides what is genuinely urgent,
ground truth is free — a judge would add noise, cost, and a judge-validation problem of its
own. A judge becomes necessary the moment you evaluate against a real inbox, where no
labels exist. That's named as a limitation rather than skipped.

**The generator is deliberately misaligned with the scorer.** If urgent mail always
contained the words the scorer looks for, precision and recall would be 1.0 by
construction and the evaluation would measure nothing. So only ~75% of genuinely urgent
mail carries an urgency cue, ~12% of routine mail carries one anyway, and the simulated
classifier has an 8% miss rate and a 22% false-positive rate. Two tests assert that
misalignment still holds, so nobody can accidentally "improve" the generator into
uselessness.

**The baseline is not a second implementation.** It's the same broker with every policy
disabled, which reproduces upstream exactly. One code path, config-driven — so ablations
fall out for free and the comparison cannot drift.

---

## Results

30 simulated days × 5 seeds.

| Config | Intr/day | Prec@intr | Recall | Timely recall | Ctx chars | Dup agents |
|---|---|---|---|---|---|---|
| baseline (upstream) | 4.13 | 0.304 | 0.887 | 0.887 | 1035 | 20.0 |
| **attention layer** | **2.17** | **0.367** | **0.887** | **0.472** | **647** | **0.0** |
| – no coalescing | 2.23 | 0.367 | 0.887 | 0.472 | 647 | 0.0 |
| – no dedupe | 2.22 | 0.369 | 0.887 | 0.477 | 647 | 20.0 |
| – no budget | 2.17 | 0.367 | 0.887 | 0.472 | 647 | 0.0 |
| – no quiet hours | 2.37 | 0.408 | 0.887 | 0.662 | 647 | 0.0 |

Context cost is also **flat in roster size** — measured at 10 agents and 400, the marginal
per-turn cost differs by ≤4 characters, which are the digits in `total="N"`. Upstream grows
linearly and without bound. `test_roster_cost_per_turn_is_flat_regardless_of_roster_size`
pins it.

---

## Reading those results honestly

Three things in that table deserve to be said out loud rather than left for a reader to
notice.

**1. The headline recall number was flattering me.** Plain recall counts a digest delivered
the next morning as a success. For something genuinely urgent, twenty hours late is a miss.
So I added *timely* recall — surfaced within an hour — and it read 0.472 against a
baseline of 0.887. Half the urgent mail was arriving too late to matter. The apparent free
win was an artefact of the wrong metric.

**2. Fixing it revealed the real bottleneck, which wasn't the policy.** I added a hold
ceiling so nothing sits in the digest indefinitely, then swept it from 1h to 24h:

| max_hold | Intr/day | Timely recall |
|---|---|---|
| 1h | 2.53 | 0.476 |
| 3h | 2.17 | 0.472 |
| 24h | 0.83 | 0.472 |

Interruptions move a lot; timely recall barely moves at all. That rules out the hold
ceiling and points at the scorer: of 54 urgent emails, the classifier flags 51, but the
keyword scorer rates only 42 of those above the interrupt threshold. **The rule-based
scorer caps timely recall at 0.78 by construction** — no policy tuning recovers mail it
never rated urgent.

An ablation that rules something *out* is as useful as one that confirms something.

**3. Most of the ablations are inside the noise.** Standard deviation across seeds on
interruptions/day is ~0.26. Coalescing (2.17→2.23), dedupe (2.17→2.22) and the budget
(2.17→2.17) all fall inside that — **I cannot claim any of them reduced interruptions.**

They're not worthless; they're just measured on the wrong axis in that column. Dedupe is
the only thing taking duplicate agents from 20 to 0. Quiet hours is what costs the most
timely recall (0.472 → 0.662 when removed), so it's paying the largest share of the
recall bill. The headline drop from 4.13 to 2.17 is almost entirely the **routing
thresholds** — urgency below 0.7 goes to the digest, below 0.25 is suppressed.

The budget in particular is a safety rail rather than a lever: at 30 emails/day a 4/hour
cap almost never binds, and even at 120 emails/day it barely does. It exists for the bad
day, and I'd keep it for that reason while being clear it isn't doing work in these
numbers.

---

## Limitations

- The synthetic inbox is not a real inbox. Category mix, cue rates, burst rate and
  classifier error rates are chosen, not observed.
- Ground truth is free because I generated it — which is also why it cannot tell me
  whether my notion of "urgent" matches a real broker's.
- No online signal. Nothing measures whether a person acted on what was surfaced, which is
  the only measure that finally matters.
- The urgency scorer is keyword-based and caps timely recall at 0.78.
- The batch-state defect is documented, not fixed.
- Confirmation-before-send remains prompt-only. Identified, not fixed.
- Agent-routing quality is measured only through the duplicate-agent count and the
  bounded-context property. There's no labelled dataset of "which agent should have
  handled this request", which is the eval I'd add next on that half.

---

## Other gaps found along the way

**Indirect prompt injection is the most serious thing in the repo.** The watcher pulls
email bodies — arbitrary text written by strangers — cleans them, and feeds them to an
agent holding `gmail_create_draft`, `gmail_execute_draft`, `gmail_forward_email` and
`gmail_reply_to_thread`. There is no reliable in-band boundary between instructions and
data inside a prompt, so this can't be fixed by prompting. It needs privilege separation
(the component reading untrusted content shouldn't hold send capability), recipient
allowlists for automated sends, human gates enforced in code, and an injection test set so
resistance becomes a measured property rather than a hope.

**Safety by prompt rather than policy.** Same root cause, worth naming separately.
Confirmation-before-send exists only as English. A confirmation token issued by an explicit
user action and checked in code makes it a property of the system rather than a hope about
the model.

**No observability.** No trace across the interaction agent, batch manager and *n*
execution agents; no token or cost accounting per turn. Every other problem here is more
expensive to diagnose because of it, and evaluation without instrumentation is guesswork.

**No shared user model.** Agent memory is per-agent and siloed, so every new agent
re-derives the same facts about the user's relationships and preferences. There's no
consolidation from episodic logs into durable semantic memory, and no forgetting.

**Fire-and-forget error swallowing.** Execution agents are launched with
`asyncio.create_task` and never awaited, so an agent that raises fails silently.

**Single-tenant assumptions in the module layer.** Import-time singletons against
hardcoded paths, one Gmail account, no auth. The `fcntl` locking on `roster.json` hints at
multi-process awareness the rest of the design doesn't support. Fine for a demo; it's the
first thing that has to change for a real deployment. It's also why the test fixtures have
to swap singletons by hand.

---

## What I'd do next

1. **Replace the keyword scorer with a small model.** It's the measured bottleneck and the
   headroom is quantified: 0.47 today, 0.78 ceiling with perfect policy, 0.89 if scoring
   matched the classifier. The interface is `str -> float`, so it's a one-line change per
   call site.
2. **Add an online feedback signal.** Did the user act on what was surfaced? That replaces
   my hand-written importance rules with something learned, and it's the only way to know
   whether the synthetic labels resemble reality.
3. **Make confirmation-before-send an enforced policy**, and add the injection test set.
4. **Fix the batch-state defect properly** — key batches by turn, add a concurrency
   semaphore and a per-agent circuit breaker.
5. **Shared semantic memory**, consolidated from agent logs, so agents stop re-deriving the
   same facts.

---

## Running it

No API key needed for either of these.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt -r server/requirements-dev.txt

pytest -rx                                  # 119 passed, 1 xfailed
python -m evals.run --days 30 --seeds 5     # baseline vs treatment + ablations
python -m evals.run --sweep                 # + the hold-ceiling sensitivity sweep
```

The one expected failure is the documented batch-state defect, marked `xfail(strict=True)`
so it converts to a suite failure if anyone fixes the bug without removing the marker.

---

## Where the code lives

```
server/services/llm/           the test seam
  base.py                      LLMClient protocol, ContextVar, use_llm_client()
  openrouter.py                production adapter (deliberately logic-free)
  scripted.py                  ScriptedLLM — replay, record, fail loudly

server/services/attention/     the Attention Layer
  registry.py                  agent records, lifecycle, migration, atomic writes
  ranking.py                   shortlist scoring + spawn-time dedupe
  compaction.py                rolling summary + tail for agent history
  broker.py                    dedupe, coalesce, route, budget, quiet hours
  scoring.py                   where urgency comes from (rule-based, swappable)
  service.py                   process-wide broker + the drain loop

evals/
  world.py                     seeded synthetic inbox with ground-truth labels
  simulate.py                  virtual-clock replay + metrics
  run.py                       baseline vs treatment, ablations, sweep

tests/                         119 tests, ~2.4s, no API key
```

Integration points touched in upstream code, kept deliberately small: both agent runtimes
and three services now call `get_llm_client()`; `_render_active_agents()` renders the
shortlist; `send_message_to_agent` records to the registry and dedupes; the watcher and
batch manager publish to the broker; `app.py` starts the drain loop.
