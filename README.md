# VoiceVibeCoder

**Say what you want. It gets written, run, and committed.**

VoiceVibeCoder is a program that writes programs from your voice. You speak;
it transcribes locally, works out whether you gave it a *command* or a
*specification*, writes real files, runs them, fixes its own mistakes, commits
the result, and then tells you the one idea that would make what you just built
remarkable.

```
◈  What do you want your program to do?
🎙  um, make me a, uh, program that totals the numbers i give it
⇢  make me a program that totals the numbers i give it
✎  main.py
◈  Wrote main.py, which totals the numbers it is given.
▶  main.py
✖  NameError: name 'numbrs' is not defined
◈  Fixing it.
✎  main.py
▶  main.py
   total: 31
◈  main.py ran cleanly. It printed: total: 31
⑂  8f2a1c3  Wrote main.py, which totals the numbers it is given.
◈  One idea: Sparkline Output. Print the numbers as a one-line sparkline above
   the total, so the shape of the data is visible before the arithmetic.
   Say build the first one to do it.
```

---

## The idea

Dictating code is a bad idea. Dictating *intent* is a very good one — and the
gap between them is where all the engineering lives.

    microphone
      → endpointer          when did a sentence start and stop?
      → transcription       words, locally, offline
      → NLP                 what was meant, not what was heard
      → command grammar     control phrase, or specification?
      ├─ control            undo / run / show / delete — instant, free, exact
      └─ specification      Claude writes the files
      → runner              does it actually work?
      → git                 one commit per accepted change
      → improvement         what would make this remarkable?

Two rules hold the design together:

**Control is a grammar, code is a conversation.** "undo that" must undo, every
time, in microseconds, for nothing. It is a closed set of anchored patterns, so
it never reaches a model. Everything the grammar does *not* claim is an
open-ended specification, and that goes to Claude with the full session
context. `run it` is a command; `run the simulation ten thousand times` is a
specification, and the grammar knows the difference.

**Speech is lossy, so everything is reversible.** Every change is wrapped in a
snapshot before it happens and committed after it lands. "undo that" is a
directory restore, not an apology.

---

## It has its own brain

VoiceVibeCoder is not a front-end for one company's API. The model layer is one
small protocol — *take a turn* and *answer in this schema* — and everything
above it is ordinary software that does not care who is thinking.

| Brain | What it is | Needs |
| --- | --- | --- |
| `claude` | Claude, streaming, with adaptive thinking and cached prompts | an API key |
| `local` | a model on this machine — Ollama, llama.cpp, LM Studio, vLLM | no key, no network |
| `auto` | Claude when credentials exist, local when they do not | whichever is there |

```bash
# Ollama
ollama pull qwen2.5-coder:7b && voicevibe --brain local

# llama.cpp — the same flag, nothing else to configure
llama-server -m qwen2.5-coder-7b-q4.gguf --port 11434
voicevibe --brain local

# any model, any machine
voicevibe --brain local --local-model llama3.1:8b
voicevibe --local-url http://192.168.1.20:11434
```

**The server is detected, not configured.** Whichever model-listing endpoint
answers — Ollama's `/api/tags` or the OpenAI-compatible `/v1/models` — decides
which dialect it speaks, including how tool results are addressed and how JSON
schemas are requested. Pin it with `local_api = "ollama"` or `"openai"` if you
would rather it not probe.

For this program's job — writing whole files that run first time — a
**coder-tuned** model beats a general one of the same size. `qwen2.5-coder:7b`
follows the file protocol more reliably than `llama3.1:8b`.

The hard part is not the transport, it is that **small local models are bad at
tool calling**. A 7B coder will describe the file it would write instead of
calling `write_file`, and a brain that only understands tool calls comes back
empty every time. So the local brain speaks both dialects: native tool calls
when the model manages them, and otherwise a plain-prose convention —

    FILE: main.py
    ```python
    print("hello")
    ```

— parsed out of the reply and handed upward *as tool calls anyway*, with the
`SUMMARY:` line becoming what gets spoken. The build loop never learns which
happened. That is what makes a small local model a first-class citizen here
rather than a degraded mode.

The transport is `urllib` from the standard library: a program whose selling
point is working offline should not need a package index to start.

---

## Install

The core program is **standard library only** — the grammar, the NLP pipeline,
the workspace, the runner and the local brain need nothing from PyPI. Each
capability beyond that is an extra, so no machine builds a wheel for a code
path it will not run.

```bash
./voicevibe                 # no install at all: run it straight from the clone
pip install -e '.[all]'     # the laptop install: Claude, microphone, speech
pip install -e '.[claude]'  # + Claude only (you have a key, no microphone)
pip install -e '.[voice]'   # + microphone and local speech recognition
pip install -e .            # nothing compiled: local brain + typed input
```

On Debian/Ubuntu the microphone also needs PortAudio: `sudo apt install libportaudio2`.

For the Claude brain, authentication follows the Anthropic SDK: export
`ANTHROPIC_API_KEY`, or run `ant auth login` once and let the SDK find the
profile. For the local brain there is nothing to authenticate.

---

## Use

```bash
voicevibe                                  # listen on the microphone
voicevibe --text                           # type instead of speaking
voicevibe --say "make a maze generator"    # one instruction, then exit
voicevibe --workspace ~/lab --type-commits # typed commit messages
voicevibe --wake "hey vibe"                # only act when addressed
voicevibe --list-devices                   # which microphone is which
```

**No microphone? It is still the same program.** `--text` reads typed lines and
feeds them through the identical pipeline — NLP, grammar, workspace, runner and
all. That is also how the tests and the demo run:

```bash
python examples/offline_demo.py    # the whole loop, no API key, no network
```

### On Android

The desktop chain does not port: there is no PortAudio on a phone and no
CTranslate2 wheel for Android, so `sounddevice` and `faster-whisper` are both
out. Android already does capture, endpointing and recognition behind one
system service, so VoiceVibeCoder uses that instead. Everything downstream is
unchanged.

Install **Termux** *and* **Termux:API** from F-Droid — the Play Store builds are
stale and cannot talk to each other — then:

```bash
pkg install python git termux-api
git clone https://github.com/samus0123/voicevibecoder && cd voicevibecoder
./voicevibe --brain local --local-url http://<your-desktop>:11434
```

**There is no install step.** `./voicevibe` is a shell script that points Python
at the source next to it and runs the program — nothing to compile, nothing to
download, nothing that can fail with `No module named voicevibecoder`. To call
it from anywhere, link it onto your PATH:

```bash
ln -s "$PWD/voicevibe" "$PREFIX/bin/voicevibe"
voicevibe                             # now works from any directory
```

`pip install -e .` still works if you prefer a real install, and gives you the
same `voicevibe` command. It is simply not required.

That install is deliberately empty of dependencies: on a phone, listening goes
through Termux:API rather than PortAudio, and the local brain speaks HTTP from
the standard library, so neither numpy nor a Rust toolchain is involved.
`--android` is implied inside Termux.

To use Claude from the phone instead, add the extra — budget time for it, since
`pydantic-core` has no Termux wheel and builds from source:

```bash
pkg install rust binutils
pip install -e '.[claude]'
export ANTHROPIC_API_KEY=...
voicevibe
```

Listening is one Android recognition session per utterance: it blocks until you
stop talking, hands over the text, and listens again. Responses are spoken
through `termux-tts-speak`. A silent session is ignored; three failed sessions
in a row stop with a setup hint rather than spinning on a missing permission.

**On Android 14 and 15**, three settings decide whether a session survives more
than a minute. Android's phantom-process killer reaps long-running child
processes, which is exactly what a Python session and a model server are:

1. Settings → Developer options → **Disable child process restrictions**
2. Settings → Apps → Termux → Battery → **Unrestricted**
3. Pull down the Termux notification → **Acquire wakelock** (survives screen-off)

Android 15 refuses to install apps targeting API < 24; the F-Droid Termux build
targets 28, so it installs normally. Grant the microphone permission when the
first `termux-speech-to-text` prompts for it — pick "While using the app"
rather than "Only this time", or you will re-grant it every session.

Two honest caveats. **Recognition may not be local** — `termux-speech-to-text`
uses Android's recognition service, usually Google's, which sends audio to the
cloud unless you have installed offline recognition for your language. The
desktop path keeps dictated code on the machine; this one does not. And **the
`anthropic` install pulls `pydantic-core`**, which is Rust: if there is no
prebuilt wheel for your Termux target, `pkg install rust` first and expect a
long build.

**Fully off the cloud on a phone:** point the local brain at a model server on
your desktop and nothing leaves your LAN except over your own wire.

```bash
voicevibe --brain local --local-url http://192.168.1.20:11434
```

A phone can also host the model itself, but a handset realistically tops out
around a 1–3B model — usable for small edits, slow for a whole program.

**The alternative that needs nothing:** run the session on a real machine and
use the phone as a voice front-end over SSH.

```bash
# in Termux, talking to a session running on your laptop
while text=$(termux-speech-to-text); do echo "$text"; done | ssh laptop 'voicevibe --text'
```

Recognition happens on the phone, the build happens where your toolchain is,
and nothing about the pipeline changes — `--text` is the same path speech
takes, minus the microphone.

### When something is not working

```bash
./voicevibe --doctor
```

One command, every way the setup can be half-finished: Python version, which
packages are present, whether either brain is reachable, how listening will
happen, and whether the workspace is writable — ending in a verdict that says
what to do next.

The session itself starts whether or not a brain is reachable. Listing files,
reading them back, running a program, undoing, dictating and quitting need no
model at all; only building does, and if nothing is reachable it says so when
you ask rather than refusing to open.

### What you can say

Everything not in this table is a specification: describe the program you want,
in whatever words you would use to describe it to a person.

| Say | What happens |
| --- | --- |
| *make a command line to-do list that saves to JSON* | it writes the files |
| *now add due dates* | follow-ups keep the context |
| **run it** / **stop it** | run the entry point, or interrupt it |
| **undo that** / *scratch that* | restore the previous snapshot |
| **list files** | inventory of the workspace |
| **show me main dot pie** | reads the file back, on screen |
| **delete main.py** | asks first, then deletes |
| **the main file is app dot pie** | sets what "run it" means |
| **new project called radar** | fresh workspace, cleared context |
| **dictate into notes dot m d** | literal text until "end dictation" |
| **explain the parser** | answers, changes nothing |
| **give me ideas** / *what should I build* | genius-only proposals |
| **how could this be better** | improvements to what exists |
| **build the second one** | builds a proposal by ordinal |
| **repeat that** / **help** / **quit** | |

---

## The parts

### Natural language processing (`intent/nlp.py`)

Speech recognisers are trained on prose, and prose has no `def`. Between the
transcript and the grammar sits a pipeline of pure token functions:

| Stage | Heard | Understood |
| --- | --- | --- |
| self-repair | "make it red, actually no, make it blue" | "make it blue" |
| disfluency removal | "um, uh, make it faster" | "make it faster" |
| domain lexicon | "write a deaf that uses numb pie" | "write a def that uses numpy" |
| number folding | "run it ten thousand times" | "run it 10000 times" |
| sentence typing | "what does the parser do" | *question, not instruction* |

The lexicon is small and curated on purpose — every entry is a phrase that is
not a plausible thing to say to a coding assistant, so correcting it is safe. A
big fuzzy-matching table would "fix" words you actually meant. `Utterance`
carries the edits it made, so the console can show `heard → understood` and you
can tell a bad idea from a bad transcript at a glance.

Spoken paths get the same treatment: "source slash web app dot j s" becomes
`source/web_app.js`, and a file slot that contains prose rather than a path is
rejected — which is why "delete the duplicate entries from the list" is a
programming instruction and not a deletion.

### Endpointing (`audio/vad.py`)

Energy-based VAD with three properties that matter in a real room: an adaptive
noise floor re-estimated from silent frames only (a fan spinning up does not
trip the gate), hysteresis (the pause between words does not chop the sentence
in half), and a pre-roll buffer (the plosive at the start of "print this" is
never clipped). The minimum-duration test counts *voiced* frames, so a door
slam is not mistaken for a third of a second of speech.

### The workspace (`workspace/`)

Every path is resolved against the workspace root and rejected if it lands
outside — absolute paths, `..` walks, and symlinks pointing out of the tree all
fail identically. The model proposes paths; the workspace decides. There is no
shell tool: running code is *your* verb, so a spoken sentence can never become
an unreviewed command.

Programs run in a child process, rooted at the workspace, with a wall-clock
limit and captured output. When one fails, the traceback goes straight back to
the model that wrote it, up to `self_heal_attempts` times.

### Ideas with a bar (`codegen/ideas.py`)

Ask any model for app ideas and you get a to-do list with tags. So the request
is structured instead: every idea must name the *mechanism* — the specific
trick, inversion or constraint that makes it work — and score itself against a
rubric. To-do lists, weather dashboards, habit trackers and chat wrappers are
disqualified by name. The model scores; the client filters at `idea_bar`
(default 80), so tightening the standard needs no prompt changes.

The same machinery, pointed at the code you just built, answers *"what would
make this remarkable?"* — with chores explicitly disqualified: add tests, add
error handling, add type hints, add a README, split into modules. Those you can
ask for by name. It is looking for the thing you did not think of, and
returning nothing is a valid answer.

### Version control (`workspace/versioning.py`)

Snapshots exist for "undo that": fast, in-process, disposable. Git exists for
what you want after an hour of talking — a readable history. Commits are cheap
and unlimited: every accepted change is one commit, each carrying the spoken
instruction that caused it and an AI-assisted trailer. Commit messages are the
model's spoken summary by default, or **typed** (`--type-commits`), because
dictating prose is pleasant and spelling out a conventional-commit subject is
not.

### Nothing is dropped

Every utterance is journalled to `.voicevibe/transcript.jsonl` with what was
heard, what it was understood as, and the intent it resolved to. Unrecognised
input becomes a build instruction rather than an error; an utterance that
raises is reported and the loop continues. A pending question that gets a new
instruction instead of an answer drops the question and obeys the instruction —
the way a person would.

---

## Configuration

Defaults → `voicevibe.toml` in the workspace → `VVC_*` environment variables →
command line flags. See [`voicevibe.example.toml`](voicevibe.example.toml) for
every key.

```toml
[voicevibe]
model = "claude-opus-5"
effort = "high"
whisper_model = "base.en"
auto_run = true
self_heal_attempts = 2
idea_bar = 80
commit_mode = "auto"        # auto | typed | off
```

---

## Development

```bash
pip install -e '.[dev]'
pytest                       # 191 tests, no microphone or API key required
ruff check voicevibecoder
```

The test suite runs the entire application against a scripted generator: the
NLP pipeline, path resolution, endpointing with synthetic audio, workspace
containment, snapshots and undo, the runner, the git journal, idea filtering,
and the session state machine end to end.

## Layout

```
voicevibecoder/
  config.py          layered configuration
  session.py         the state machine everything else serves
  console.py         heard vs. understood, on screen
  cli.py             microphone or keyboard, same path
  audio/             capture + endpointing
  speech/            transcription in, speech out, Android via Termux:API
  intent/            NLP pipeline + command grammar + spoken paths
  codegen/           the brain protocol, Claude + local brains, the idea bar
  workspace/         containment, snapshots, runner, git
```

## Licence

MIT — see [LICENSE](LICENSE).
