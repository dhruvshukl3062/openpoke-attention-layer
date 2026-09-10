# The Attention Layer

An attention budget for OpenPoke. It bounds what the orchestrator sends the model, and
puts a single gate in front of everything that wants to interrupt the user.

I built this on [OpenPoke](https://github.com/shlokkhemani/openpoke) over three days.
Upstream setup instructions are still in [README.md](README.md). Everything below is my
work.

---

## Contents

- [How I defined the problem](#how-i-defined-the-problem)
- [What I found in the code](#what-i-found-in-the-code)
- [The design](#the-design)
- [How I tested it](#how-i-tested-it)
- [Results](#results)
- [Reading those results honestly](#reading-those-results-honestly)
- [Limitations](#limitations)
- [Other gaps I found along the way](#other-gaps-i-found-along-the-way)
- [What I'd do next](#what-id-do-next)
- [Running it](#running-it)
- [Where the code lives](#where-the-code-lives)

---

## How I defined the problem

"Agent overload" isn't defined for me anywhere, so the first thing I had to do was pin it
down tightly enough to build against and measure.

Here's where I landed. **OpenPoke has no notion of a budget.** Not on agents, not on
context, not on the user's attention. Every mechanism in it only ever adds: agents
accumulate and are never retired, logs grow and are never compacted, notifications fire
the moment they're generated. Nothing is trimmed, ranked, batched or dropped.

The result is a system that demos beautifully on day one and degrades by week three.
Ordinary testing never catches it, because ordinary testing never runs for three weeks.

The overload lands in two places at once:

- **The orchestrator drowns.** The interaction agent's roster grows without bound and
  gets pasted into every turn, so per-turn cost rises linearly forever and picking the
  right agent gets harder as the list gets longer.
- **The user drowns.** Three independent background sources interrupt directly, with no
  dedupe, no batching across sources, no awareness of what time it is where the user is,
  and no cap.

Those are the same problem. Both spend a finite resource, attention, from processes that
have no idea what else is spending it. So I gave both the same fix: put a broker in front
of the scarce thing and make everything bid for it.

---

## What I found in the code

All of this is read out of the source, not inferred.

| Finding | Where | Status |
|---|---|---|
| Roster is an append-only `list[str]` of bare names, rendered into every turn with no metadata | `services/execution/roster.py`, `agents/interaction_agent/agent.py` | Fixed |
| Execution agent logs are never compacted; `build_system_prompt_with_history` loads the entire transcript with `conversation_limit=None` | `agents/execution_agent/agent.py:38,73` | Fixed |
| Email watcher calls `handle_agent_message` per important email, one full interaction turn each, delivered immediately | `services/gmail/importance_watcher.py:200-207` | Fixed |
| `TriggerScheduler` constructs a fresh `ExecutionBatchManager` per trigger, so simultaneous reminders never batch | `services/trigger_scheduler.py:92` | Fixed at the broker |
| Batch state is a single global with no turn identity, so a later turn's work joins an earlier turn's batch and is withheld behind it | `agents/execution_agent/batch_manager.py:100-115` | Documented (strict `xfail`) |
| Confirm-before-send is prompt-only; `gmail_execute_draft()` takes a draft id and sends | `agents/execution_agent/tools/gmail.py:376-383` | Documented |
| No test seam: `request_chat_completion` imported and called directly in both runtimes; zero tests in the repo | both runtimes | Fixed |
| All five model roles default to `anthropic/claude-sonnet-4`, including a classifier that runs on every inbound email | `config.py:54-58` | Documented |

One thing convinced me the second row was an oversight rather than a hard problem. The
conversation log already has a full summarisation subsystem. Execution agent logs have
none. The mechanism was sitting right there and nobody applied it.

---

## The design

Everything new lives in one subsystem, `server/services/attention/`, and it has two
halves.

### A. Agent registry and shortlist, for the orchestrator side

I replaced the `list[str]` in `roster.json` with a record per agent: purpose, tags, the
entities it touches, a rolling summary of what it has done, invocation count, timestamps,
status, and a compaction cursor.

The interaction agent no longer sees every agent. It sees a **ranked shortlist of 8**,
carrying the metadata that supports the choice. I score on four signals:

| Signal | Weight | Why I chose that weight |
|---|---|---|
| Entity match | 3.0 | "reply to Alice" has to find the Alice agent even with zero word overlap |
| Name overlap | 1.5 | Direct lexical hit on the agent's own name |
| IDF-weighted lexical overlap | 1.0 | "email" appears in most agents and should barely discriminate; "vercel" should do real work |
| Recency | 0.5 | A tiebreaker. Weight it any higher and the most recent agent wins everything |

On top of the ranking there are three smaller pieces. **Spawn-time dedupe** redirects a
proposed "Alice Email" to the existing "Email to Alice" instead of splitting one subject's
history across two agents. **Dormancy** drops an agent from shortlists after 14 days
unused, while keeping it reachable by exact name and reviving it on use. And **per-agent
log compaction** reuses the repo's own summariser, so a long-lived agent's prompt
converges instead of growing.

I did not use embeddings, and that was a considered choice. For a few hundred short
structured records queried by names and identifiers, exact entity matching plus IDF
overlap beats vector similarity on exactly the cases that matter, costs no model call, and
can be explained line by line when it gets something wrong. Vectors earn their place on
large unstructured corpora. This is neither.

### B. Attention broker, for the human side

Nothing interrupts the user directly any more. The email watcher, the trigger scheduler
and completed execution batches all publish a `Candidate`, and the broker decides what
happens to it.

The pipeline runs in this order: dedupe (by key plus a hash of the normalised text, so a
real update passes but a verbatim repeat doesn't), coalesce (a 90 second window, so things
arriving together become one message), route (interrupt, digest or suppress), interruption
budget (4 per hour), quiet hours (local time, wrapping midnight), escalation override, and
finally a hold ceiling.

Three decisions in there are worth explaining.

**Scoring lives outside the broker.** A candidate arrives already carrying an urgency its
source worked out, and the broker only turns that into a routing decision. "How urgent is
this?" is a judgement call with no provably right answer. "Given 0.8, at 3am, with the
budget spent, do I interrupt?" is a policy question with a definite one. Keeping them
apart makes the broker fully deterministic, which is what makes it measurable, and it
means I can swap the scorer for a model later without touching any policy.

**Escalations override both the budget and quiet hours.** Buying quiet is only a win if
nothing that genuinely can't wait gets buried, and that's the failure mode I'd care most
about in production.

**Time is injected and flushing is explicit.** There is no `sleep` anywhere in the broker.
Production drives it from a timer and the evaluation harness drives it from a virtual
clock. That one choice is what lets a simulated month run in about a second.

---

## How I tested it

**119 tests, 1 expected failure, roughly 2.4 seconds.** No API key required.

### Layer 1: the seam

`request_chat_completion` was imported and called directly in five places, so none of the
orchestration layer could be exercised without a live key and a source of
non-determinism. That's why the repo had no tests. All five now route through an
injectable `LLMClient` protocol.

I used a protocol rather than `mock.patch` on purpose. Patching targets a module path,
which is an implementation detail that silently breaks when a file moves. A protocol is a
contract, and the scripted client exercises the same code paths in the runtimes that the
real one does.

### Layer 2: a scripted model

`ScriptedLLM` replays a fixed response sequence, records every request, and fails loudly
when the code makes more calls than I scripted. That makes the whole tool loop
deterministic, so the iteration cap, malformed tool arguments, unknown tools, a tool that
raises, a model that errors, and prompt size per turn all become ordinary assertions.

### Layer 3: unit tests

Registry CRUD and lifecycle, ranking signals, dedupe thresholds, compaction invariants,
broker routing, quiet-hours arithmetic across midnight, budget accounting, digest
assembly.

### Layer 4: integration

These check that the watcher, the batches and the triggers actually go *through* the
broker rather than around it. That's the part most likely to regress silently, since every
component would still pass its own unit tests.

### Layer 5: simulation

Overload only shows up over weeks, so `evals/` generates a seeded month of inbox traffic
with ground-truth urgency labels and replays it against a virtual clock.

**There's no LLM judge, and that's intentional.** Because the generator decides what counts
as genuinely urgent, ground truth is free. A judge would add noise, cost, and a
judge-validation problem of its own. A judge does become necessary the moment you evaluate
against a real inbox, where no labels exist, and I've named that as a limitation rather
than skipping past it.

**The generator is deliberately misaligned with the scorer.** If urgent mail always
contained the words my scorer looks for, precision and recall would be 1.0 by construction
and the evaluation would be measuring nothing. So only about 75% of genuinely urgent mail
carries an urgency cue, about 12% of routine mail carries one anyway, and the simulated
classifier has an 8% miss rate and a 22% false-positive rate. Two tests assert that the
misalignment still holds, so nobody can accidentally "improve" the generator into
uselessness.

**The baseline is not a second implementation.** It's the same broker with every policy
disabled, which reproduces upstream exactly. One code path, driven by config, so the
ablations fall out for free and the comparison can't drift as I edit.

---

## Results

30 simulated days, 5 seeds.

| Config | Intr/day | Prec@intr | Recall | Timely recall | Ctx chars | Dup agents |
|---|---|---|---|---|---|---|
| baseline (upstream) | 4.13 | 0.304 | 0.887 | 0.887 | 1035 | 20.0 |
| **attention layer** | **2.17** | **0.367** | **0.887** | **0.472** | **647** | **0.0** |
| – no coalescing | 2.23 | 0.367 | 0.887 | 0.472 | 647 | 0.0 |
| – no dedupe | 2.22 | 0.369 | 0.887 | 0.477 | 647 | 20.0 |
| – no budget | 2.17 | 0.367 | 0.887 | 0.472 | 647 | 0.0 |
| – no quiet hours | 2.37 | 0.408 | 0.887 | 0.662 | 647 | 0.0 |

Context cost is also **flat in roster size**. Measured at 10 agents and at 400, the
marginal per-turn cost differs by 4 characters or fewer, and those characters are the
digits in `total="N"`. Upstream grows linearly and without bound.
`test_roster_cost_per_turn_is_flat_regardless_of_roster_size` pins that down.

---

## Reading those results honestly

Three things in that table deserve to be said out loud rather than left for a reader to
find.

**1. The headline recall number was flattering me.** Plain recall counts a digest
delivered the next morning as a success. For something genuinely urgent, twenty hours late
is a miss. So I added *timely* recall, meaning surfaced within an hour, and it came back
at 0.472 against a baseline of 0.887. Half the urgent mail was arriving too late to
matter. The apparent free win was an artefact of measuring the wrong thing.

**2. Fixing that revealed the real bottleneck, and it wasn't the policy.** I added a hold
ceiling so nothing sits in the digest indefinitely, then swept it from 1h to 24h to check
whether the fix was doing what I thought:

| max_hold | Intr/day | Timely recall |
|---|---|---|
| 1h | 2.53 | 0.476 |
| 3h | 2.17 | 0.472 |
| 24h | 0.83 | 0.472 |

Interruptions move a lot. Timely recall barely moves at all. That rules out the hold
ceiling and points at the scorer instead: of 54 urgent emails, the classifier flags 51,
but the keyword scorer rates only 42 of those above the interrupt threshold. **The
rule-based scorer caps timely recall at 0.78 by construction.** No amount of policy tuning
recovers mail it never rated urgent in the first place.

An ablation that rules something *out* is as useful as one that confirms something.

**3. Most of the ablations are inside the noise.** Standard deviation across seeds on
interruptions per day is about 0.26. Coalescing (2.17 to 2.23), dedupe (2.17 to 2.22) and
the budget (2.17 to 2.17) all fall inside that, so **I can't claim any of them reduced
interruptions.**

That doesn't make them worthless, it means that column is the wrong axis to judge them on.
Dedupe is the only thing taking duplicate agents from 20 to 0. Quiet hours costs the most
timely recall (0.472 rises to 0.662 when I remove it), so it's paying the largest share of
the recall bill. The headline drop from 4.13 to 2.17 is almost entirely the **routing
thresholds**: urgency below 0.7 goes to the digest, below 0.25 is suppressed.

The budget in particular is a safety rail rather than a lever. At 30 emails a day a 4 per
hour cap almost never binds, and even at 120 emails a day it barely does. It exists for
the bad day, and I'd keep it for that reason while being clear that it isn't doing work in
these numbers.

---

## Limitations

- The synthetic inbox is not a real inbox. Category mix, cue rates, burst rate and
  classifier error rates are all chosen by me, not observed.
- Ground truth is free because I generated it, which is also why it can't tell me whether
  my notion of "urgent" matches a real user's.
- There's no online signal. Nothing measures whether a person acted on what was surfaced,
  and that's the only measure that finally matters.
- The urgency scorer is keyword-based and caps timely recall at 0.78.
- The batch-state defect is documented, not fixed.
- Confirmation-before-send is still prompt-only. I identified it and did not fix it.
- I only measure agent-routing quality through the duplicate-agent count and the
  bounded-context property. There's no labelled dataset of "which agent should have
  handled this request", and that's the eval I'd add next on that half.

---

## Other gaps I found along the way

**Indirect prompt injection is the most serious thing in this repo.** The watcher pulls
email bodies, which are arbitrary text written by strangers, cleans them, and feeds them to
an agent holding `gmail_create_draft`, `gmail_execute_draft`, `gmail_forward_email` and
`gmail_reply_to_thread`. There's no reliable in-band boundary between instructions and data
inside a prompt, so this cannot be fixed by prompting. It needs privilege separation, where
the component reading untrusted content doesn't hold send capability, plus recipient
allowlists for automated sends, human gates enforced in code, and an injection test set so
resistance becomes a measured property rather than a hope.

**Safety by prompt rather than by policy.** Same root cause, but worth naming on its own.
Confirmation-before-send exists only as English in a system prompt. A confirmation token
issued by an explicit user action and checked in code would make it a property of the
system instead of a hope about the model.

**No observability.** There's no trace across the interaction agent, the batch manager and
*n* execution agents, and no token or cost accounting per turn. Every other problem in this
list is more expensive to diagnose because of it, and evaluating a system you can't
instrument is guesswork.

**No shared user model.** Agent memory is per-agent and siloed, so every new agent
re-derives the same facts about the user's relationships and preferences. There's no
consolidation from episodic logs into durable semantic memory, and no forgetting.

**Fire-and-forget error swallowing.** Execution agents are launched with
`asyncio.create_task` and never awaited, so an agent that raises fails silently.

**Single-tenant assumptions in the module layer.** Import-time singletons against hardcoded
paths, one Gmail account, no auth. The `fcntl` locking on `roster.json` hints at
multi-process awareness that the rest of the design doesn't actually support. That's fine
for a demo, but it's the first thing that would have to change for a real deployment, and
it's also why my test fixtures have to swap singletons by hand.

---

## What I'd do next

1. **Replace the keyword scorer with a small model.** It's the measured bottleneck and I
   can quantify the headroom: 0.47 today, 0.78 as the ceiling with perfect policy, 0.89 if
   scoring matched the classifier. The interface is `str -> float`, so it's a one-line
   change per call site.
2. **Add an online feedback signal.** Did the user act on what was surfaced? That would
   replace my hand-written importance rules with something learned, and it's the only way
   to find out whether my synthetic labels resemble reality.
3. **Make confirmation-before-send an enforced policy**, and add the injection test set.
4. **Fix the batch-state defect properly**, by keying batches on turn and adding a
   concurrency semaphore and a per-agent circuit breaker.
5. **Build shared semantic memory**, consolidated from agent logs, so agents stop
   re-deriving the same facts about the user.

---

## Running it

Neither of these needs an API key.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt -r server/requirements-dev.txt

pytest -rx                                  # 119 passed, 1 xfailed
python -m evals.run --days 30 --seeds 5     # baseline vs treatment + ablations
python -m evals.run --sweep                 # + the hold-ceiling sensitivity sweep
```

The one expected failure is the batch-state defect described above. I marked it
`xfail(strict=True)`, so it converts into a suite failure if anyone fixes the bug without
removing the marker.

---

## Where the code lives

```
server/services/llm/           the test seam
  base.py                      LLMClient protocol, ContextVar, use_llm_client()
  openrouter.py                production adapter (deliberately logic-free)
  scripted.py                  ScriptedLLM: replay, record, fail loudly

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

I kept the integration points in upstream code deliberately small. Both agent runtimes and
three services now call `get_llm_client()`. `_render_active_agents()` renders the
shortlist. `send_message_to_agent` records to the registry and dedupes. The watcher and
batch manager publish to the broker. `app.py` starts the drain loop. That's the whole
footprint: 2,195 lines added in new packages against 57 lines changed in existing ones.
