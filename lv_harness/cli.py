"""
LV-Harness command-line entry point.

Usage:
    python -m lv_harness run --config configs/tasks/videomme_streaming.yaml
    python -m lv_harness run --config configs/tasks/videomme_streaming.yaml --reasoning.model gemini-2.5-flash
"""
import argparse
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


def main():
    parser = argparse.ArgumentParser(description="LV-Harness: a framework for streaming long-video reasoning agents")
    subparsers = parser.add_subparsers(dest="command", help="subcommands")

    # run subcommand
    run_parser = subparsers.add_parser("run", help="run evaluation")
    run_parser.add_argument("--config", type=str, default=None, help="path to the YAML config file")
    run_parser.add_argument("--data_file", type=str, default=None, help="path to the data file")
    run_parser.add_argument("--eventgraph_dir", type=str, default=None, help="EventGraph directory")
    run_parser.add_argument("--backend", type=str, default=None, choices=["openai", "vllm"])
    run_parser.add_argument("--model", type=str, default=None, help="reasoning model")
    run_parser.add_argument("--max_rounds", type=int, default=None, help="max number of reasoning rounds")
    run_parser.add_argument("--workers", type=int, default=None, help="concurrency")
    run_parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    run_parser.add_argument("--output_dir", type=str, default=None, help="output directory")
    run_parser.add_argument("--batch_mode", action="store_true", help="use batch processing mode")
    run_parser.add_argument("--streaming", action="store_true", help="enable streaming reasoning mode (build memory and answer while watching the video)")
    run_parser.add_argument("--strategy", type=str, default=None,
                            choices=["hierarchical", "no_graph_walk", "videograph_only",
                                     "eventgraph_only", "sliding_window", "compressed"],
                            help="memory strategy")
    run_parser.add_argument("--eventgraph_incremental", action="store_true",
                            help="enable EventGraph incremental update (only effective in streaming mode)")
    run_parser.add_argument("--eventgraph_update_interval", type=int, default=None,
                            help="EventGraph incremental update interval (number of clips)")
    run_parser.add_argument("--eventgraph_model", type=str, default=None,
                            help="LLM model used for EventGraph incremental update")
    # Self-evolution parameters
    run_parser.add_argument("--evolution", action="store_true",
                            help="enable the self-evolution system (experience capture -> skill distillation -> skill injection -> wisdom distillation)")
    run_parser.add_argument("--evolution_dir", type=str, default=None,
                            help="root directory for self-evolution data storage")
    run_parser.add_argument("--promote_threshold", type=int, default=None,
                            help="trigger skill promotion when N similar experiences accumulate (default 3)")
    run_parser.add_argument("--route_threshold", type=float, default=None,
                            help="skill routing match score threshold (default 0.3)")
    run_parser.add_argument("--skill_use_llm_instructions", action="store_true",
                            help="whether to call the LLM to generate special_instructions for each new skill (off by default; uses a template placeholder when off)")
    run_parser.add_argument("--skill_instructions_llm_model", type=str, default=None,
                            help="LLM model for generating skill special_instructions (default gemini-2.5-flash)")
    run_parser.add_argument("--wisdom_use_llm", action="store_true",
                            help="whether WisdomDistiller calls the LLM to produce strategy-level insights")
    run_parser.add_argument("--wisdom_model", type=str, default=None,
                            help="LLM model used by WisdomDistiller (default gemini-2.5-flash)")
    run_parser.add_argument("--reflection_llm_max_tokens", type=int, default=None,
                            help="max output tokens for LLM-generated reflection (default 8192)")
    run_parser.add_argument("--load_prior_skills", action="store_true",
                            help="whether to load historical skills from evolution_dir/skills for cross-run reuse")

    args = parser.parse_args()

    if args.command == "run":
        _run(args)
    else:
        parser.print_help()


def _run(args):
    """Run the evaluation."""
    from .orchestrator import HarnessOrchestrator

    # Build override parameters
    overrides = {}
    if args.data_file:
        overrides["data.annotation_file"] = args.data_file
    if args.eventgraph_dir:
        overrides["memory.eventgraph_dir"] = args.eventgraph_dir
    if args.backend:
        overrides["reasoning.backend"] = args.backend
    if args.model:
        overrides["reasoning.model"] = args.model
    if args.max_rounds:
        overrides["reasoning.max_rounds"] = args.max_rounds
    if args.workers:
        overrides["reasoning.workers"] = args.workers
    if args.output_dir:
        overrides["output.dir"] = args.output_dir
    if args.strategy:
        overrides["memory.strategy"] = args.strategy
    if args.eventgraph_incremental:
        overrides["memory.eventgraph_incremental"] = True
    if args.eventgraph_update_interval:
        overrides["memory.eventgraph_update_interval"] = args.eventgraph_update_interval
    if args.eventgraph_model:
        overrides["memory.eventgraph_model"] = args.eventgraph_model
    if args.evolution:
        overrides["evolution.enabled"] = True
    if args.evolution_dir:
        overrides["evolution.dir"] = args.evolution_dir
        overrides["evolution.learnings_dir"] = f"{args.evolution_dir}/learnings"
        overrides["evolution.skills_dir"] = f"{args.evolution_dir}/skills"
        overrides["evolution.wisdom_path"] = f"{args.evolution_dir}/WISDOM.md"
        overrides["evolution.reflections_dir"] = f"{args.evolution_dir}/reflections"
    if args.promote_threshold is not None:
        overrides["evolution.promote_threshold"] = args.promote_threshold
    if args.route_threshold is not None:
        overrides["evolution.route_threshold"] = args.route_threshold
    if getattr(args, "skill_use_llm_instructions", False):
        overrides["evolution.skill_use_llm_instructions"] = True
    if getattr(args, "skill_instructions_llm_model", None):
        overrides["evolution.skill_instructions_llm_model"] = args.skill_instructions_llm_model
    if getattr(args, "wisdom_use_llm", False):
        overrides["evolution.wisdom_use_llm"] = True
    if getattr(args, "wisdom_model", None):
        overrides["evolution.wisdom_llm_model"] = args.wisdom_model
    if getattr(args, "reflection_llm_max_tokens", None):
        overrides["evolution.reflection_llm_max_tokens"] = args.reflection_llm_max_tokens
    if getattr(args, "load_prior_skills", False):
        overrides["evolution.load_prior_skills"] = True

    harness = HarnessOrchestrator(config_path=args.config, overrides=overrides)

    if args.streaming:
        results = harness.run_streaming()
    elif args.batch_mode:
        results = harness.run_batch(batch_size=args.batch_size)
    else:
        results = harness.run()

    print(f"\nEvaluation complete. Results: {results}")


if __name__ == "__main__":
    main()
