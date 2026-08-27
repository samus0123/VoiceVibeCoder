"""Command line entry point.

Two input modes, one session. ``--text`` reads typed lines from stdin and is
the honest default when there is no microphone (it is also how the tests and
the demo run); without it, the microphone is opened and every utterance is
endpointed and transcribed locally before taking exactly the same path.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from voicevibecoder import config as config_module
from voicevibecoder.codegen.generator import CodeGenerator
from voicevibecoder.console import Console
from voicevibecoder.session import Session
from voicevibecoder.speech.speak import NullSpeaker, build_speaker
from voicevibecoder.workspace.project import Workspace
from voicevibecoder.workspace.versioning import GitJournal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voicevibe",
        description="Speak a program into existence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  voicevibe                                  # listen on the microphone\n"
            "  voicevibe --text                           # type instead of speaking\n"
            "  voicevibe --say 'make a maze generator'    # one instruction, then exit\n"
            "  voicevibe --type-commits --workspace ~/lab # typed commit messages\n"
        ),
    )
    parser.add_argument("--workspace", "-w", type=Path, help="where generated code lives")
    parser.add_argument("--say", metavar="TEXT", help="run one instruction and exit")
    parser.add_argument(
        "--text", action="store_true", help="read typed lines instead of listening"
    )
    parser.add_argument(
        "--brain",
        choices=("auto", "claude", "local"),
        help="which model does the thinking (default: auto)",
    )
    parser.add_argument("--model", help="Claude model id (default: claude-opus-5)")
    parser.add_argument(
        "--local-model", help="local model name (default: qwen2.5-coder:7b)"
    )
    parser.add_argument(
        "--local-url", help="local model server (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        help="how hard the model thinks (default: high)",
    )
    parser.add_argument("--wake", metavar="PHRASE", help="only act on 'PHRASE ...'")
    parser.add_argument("--device", type=int, help="input device index")
    parser.add_argument(
        "--list-devices", action="store_true", help="show input devices and exit"
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="report what is and is not working, then exit",
    )
    parser.add_argument("--whisper-model", help="faster-whisper model (default: base.en)")
    parser.add_argument(
        "--android",
        action="store_true",
        help="listen through Termux:API instead of PortAudio (auto on Termux)",
    )
    parser.add_argument("--quiet", action="store_true", help="do not speak responses")
    parser.add_argument(
        "--no-run", action="store_true", help="do not run programs automatically"
    )
    parser.add_argument(
        "--no-confirm", action="store_true", help="never ask before destructive actions"
    )
    parser.add_argument(
        "--no-suggest",
        action="store_true",
        help="do not volunteer improvement ideas after a build",
    )
    parser.add_argument(
        "--idea-bar",
        type=int,
        metavar="0-100",
        help="minimum score an idea needs to be spoken (default: 80)",
    )
    parser.add_argument(
        "--type-commits",
        action="store_true",
        help="type each commit message instead of using the spoken summary",
    )
    parser.add_argument("--no-git", action="store_true", help="do not commit changes")
    parser.add_argument("--no-color", action="store_true", help="plain output")
    return parser


def config_from_args(args: argparse.Namespace) -> config_module.Config:
    overrides: dict[str, object] = {
        "brain": args.brain,
        "model": args.model,
        "local_model": args.local_model,
        "local_url": args.local_url,
        "effort": args.effort,
        "wake_phrase": args.wake,
        "input_device": args.device,
        "whisper_model": args.whisper_model,
        "idea_bar": args.idea_bar,
    }
    if args.quiet:
        overrides["tts_backend"] = "off"
    if args.no_run:
        overrides["auto_run"] = False
    if args.no_confirm:
        overrides["confirm_destructive"] = False
    if args.no_suggest:
        overrides["suggest_improvements"] = False
    if args.no_git:
        overrides["git"] = False
    if args.type_commits:
        overrides["commit_mode"] = "typed"
    return config_module.load(workspace=args.workspace, **overrides)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        from voicevibecoder.audio.capture import list_devices  # noqa: PLC0415

        print(list_devices())
        return 0

    config = config_from_args(args)
    console = Console(color=not args.no_color)

    if args.doctor:
        from voicevibecoder.doctor import brain_reachable, report  # noqa: PLC0415

        print(report(config))
        # Exit status is the machine-readable half: 0 means a build would
        # work right now, so a script can ask without parsing prose.
        return 0 if brain_reachable(config) else 1

    workspace = Workspace(config.workspace)

    # Constructing this connects to nothing; the brain is dialled on the first
    # build, so a missing model server cannot stop the session from opening.
    generator = CodeGenerator(config, on_text=console.chunk)

    speaker = NullSpeaker() if args.quiet else build_speaker(config)
    session = Session(
        config=config,
        workspace=workspace,
        generator=generator,
        speaker=speaker,
        console=console,
        journal=GitJournal(
            workspace.root, enabled=config.git, trailer=config.ai_trailer
        ),
    )

    console.banner()
    console.write(f"  workspace: {workspace.root}", "dim")
    console.write(f"  brain: {_brain_label(config)}", "dim")

    if args.say:
        session.handle(args.say)  # a one-shot needs no greeting
        return 0

    session.greet()

    try:
        source = pick_source(args, config, console)
        session.run(source)
    except RuntimeError as exc:  # missing microphone or ASR model
        console.error(str(exc))
        return 2
    except KeyboardInterrupt:
        console.write("")
        console.vibe("Stopped listening.")
    return 0


def _brain_label(config: config_module.Config) -> str:
    if config.brain == "local":
        return f"{config.local_model} (local, offline)"
    if config.brain == "claude":
        return config.model
    return f"auto — {config.model} if a key is set, else {config.local_model}"


def typed_lines(console: Console) -> Iterator[str]:
    """Keyboard input, presented exactly like speech to the session."""
    console.write("  type an instruction, or 'help'. ctrl-d to quit.", "dim")
    while True:
        try:
            sys.stdout.write("\n  ⌨  ")
            sys.stdout.flush()
            line = input()
        except (EOFError, KeyboardInterrupt):
            console.write("")
            return
        if line.strip():
            yield line


def pick_source(
    args: argparse.Namespace, config: config_module.Config, console: Console
) -> Iterator[str]:
    """Keyboard, Android speech services, or microphone — in that order."""
    from voicevibecoder.speech.android import is_termux  # noqa: PLC0415

    if args.text:
        return typed_lines(console)
    if args.android or config.asr_backend == "termux" or is_termux():
        return android_lines(config, console)
    return spoken_lines(config, console)


def android_lines(config: config_module.Config, console: Console) -> Iterator[str]:
    """Android's own speech recognition, one utterance per invocation."""
    from voicevibecoder.speech.android import utterances  # noqa: PLC0415

    console.write("  listening through Termux:API. say 'help' for commands.", "dim")
    return utterances(config, on_status=console.warn)


def spoken_lines(config: config_module.Config, console: Console) -> Iterator[str]:
    """Microphone input: endpoint, transcribe, hand over the text."""
    from voicevibecoder.audio.capture import utterances  # noqa: PLC0415
    from voicevibecoder.speech.listen import build_transcriber  # noqa: PLC0415

    console.write(f"  loading {config.whisper_model}...", "dim")
    transcriber = build_transcriber(config)
    console.write("  listening. say 'help' for the command list.", "dim")

    for segment in utterances(config):
        text = transcriber.transcribe(segment.audio, segment.sample_rate)
        if text.strip():
            yield text


if __name__ == "__main__":
    raise SystemExit(main())
