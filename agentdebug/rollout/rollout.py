#!/usr/bin/env python3
"""
Unified rollout script for AlfWorld, GAIA, and WebShop environments.
Provides a single interface for running rollouts across all three environments.
"""

import os
import time
import json
import logging
import argparse
from types import SimpleNamespace
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
import random
import hashlib
import sys
from agentdebug.environments.env_manager import *
from agentdebug.engines.factory import create_chat_model

try:
    import dotenv
    dotenv.load_dotenv()
except Exception:
    pass


class UnifiedAgent:
    """OpenManus agent that can work with all environments."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        base_url: str | None = None,
        env_type: str = "alfworld",
        provider: str = "auto",
        api_key: str | None = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.env_type = env_type
        self.client = create_chat_model(
            model_string=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            provider=provider,
        )
        self.provider = getattr(self.client, "provider", provider)
        # Defer to environment-provided prompts; no extra system prompt injection.
        self.system_prompts: dict[str, Optional[str]] = {}

    def get_action_from_llm(self, obs: str) -> str:
        """Get action from LLM for a single observation."""
        messages = []

        # Add system prompt if available for this environment
        system_prompt = self.system_prompts.get(self.env_type)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": obs})
        return self.client.generate(messages)

    def clone_with_temperature(self, temperature: float) -> "UnifiedAgent":
        return UnifiedAgent(
            model_name=self.model_name,
            temperature=temperature,
            base_url=getattr(self.client, "base_url", None),
            env_type=self.env_type,
            provider=self.provider,
            api_key=getattr(self.client, "api_key", None),
        )

    def get_actions_batch(
        self,
        prompts: List[str],
        concurrency: int = 4,
        retries: int = 3,
        backoff: float = 0.5,
    ) -> List[str]:
        """Get actions for multiple observations in parallel."""
        actions = [None] * len(prompts)

        def _one(idx_prompt):
            idx, prompt = idx_prompt
            delay = backoff
            for attempt in range(retries):
                try:
                    act = self.get_action_from_llm(prompt)
                    return idx, act
                except Exception as e:
                    if attempt == retries - 1:
                        # Return a default action based on environment
                        default_actions = {
                            "webshop": "<think>error</think><action>search[product]</action>",
                            "gaia": "None",
                            "alfworld": "None"
                        }
                        return idx, default_actions.get(self.env_type, "None")
                    time.sleep(delay)
                    delay *= 2
        
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futures = [ex.submit(_one, (i, p)) for i, p in enumerate(prompts)]
            for fut in as_completed(futures):
                i, act = fut.result()
                actions[i] = act
        
        return actions


OpenManusAgent = UnifiedAgent


class EnvironmentFactory:
    """Factory for creating different environment types"""
    
    @staticmethod
    def build_env(env_type: str, **kwargs) -> Any:
        """Build environment based on type"""
        
        if env_type == "alfworld":
            return EnvironmentFactory._build_alfworld(**kwargs)
        elif env_type == "gaia":
            return EnvironmentFactory._build_gaia(**kwargs)
        elif env_type == "webshop":
            return EnvironmentFactory._build_webshop(**kwargs)
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")
    
    @staticmethod
    def _build_alfworld(env_num: int = 1, seed: int = 1, history_length: int = 2,
                       alf_env_type: str = "alfworld/AlfredTWEnv", 
                       game_files: Optional[List[str]] = None, **kwargs):
        """Build AlfWorld environment"""
        from agentdebug.environments.alfworld import alfworld_projection
        from agentdebug.environments.alfworld import build_alfworld_envs
        
        alf_config_path = os.path.join(
            os.path.dirname(__file__), 
            '../../agentdebug.environments.alfworld/configs/config_tw.yaml'
        )
        
        envs = build_alfworld_envs(
            alf_config_path, 
            seed=seed, 
            env_num=env_num, 
            group_n=1, 
            is_train=True, 
            env_kwargs={}, 
            game_files=game_files
        )
        
        cfg = SimpleNamespace(
            env=SimpleNamespace(
                env_name=alf_env_type, 
                history_length=history_length
            )
        )
        
        return AlfWorldEnvironmentManager(envs, alfworld_projection, cfg)
    
    @staticmethod
    def _build_gaia(
        tasks_data: List[Dict],
        available_tools: List[str],
        env_num: int = 1,
        seed: int = 1,
        history_length: int = 2,
        max_steps: int = 30,
        tool_llm_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Build GAIA/Tool Use environment"""

        from agentdebug.environments.gaia.projection import tool_use_projection
        from agentdebug.environments.gaia.envs import build_tool_use_envs
        from agentdebug.environments.gaia.manager import ToolUseEnvironmentManager
        
        envs = build_tool_use_envs(
            tasks_data=tasks_data,
            available_tools=available_tools,
            seed=seed,
            env_num=env_num,
            group_n=1,
            is_train=True,
            tool_llm_config=tool_llm_config,
        )
        
        cfg = SimpleNamespace(
            env=SimpleNamespace(
                env_name="tool_use",
                history_length=history_length,
                max_steps=max_steps
            )
        )
        
        return ToolUseEnvironmentManager(envs, tool_use_projection, cfg)
    
    @staticmethod
    def _build_webshop(env_num: int = 1, seed: int = 1, history_length: int = 2,
                       use_train_set: bool = False, **kwargs):
        """Build WebShop environment"""
        from agentdebug.environments.webshop import build_webshop_envs, webshop_projection
        
        env_kwargs = {"observation_mode": "text"}
        
        envs = build_webshop_envs(
            seed=seed,
            env_num=env_num,
            group_n=1,
            is_train=use_train_set,
            env_kwargs=env_kwargs,
        )
        
        cfg = SimpleNamespace(
            env=SimpleNamespace(
                env_name="webshop/WebAgentTextEnv",
                history_length=history_length
            )
        )
        
        return WebshopEnvironmentManager(envs, webshop_projection, cfg)


def load_gaia_tasks(data_path: str, max_tasks: Optional[int] = None) -> List[Dict]:
    """Load GAIA tasks from JSON file"""
    with open(data_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    if max_tasks:
        tasks = tasks[:max_tasks]
    
    return tasks


def select_task_window(items: List[Any], total_envs: int, offset: int, label: str) -> Optional[List[Any]]:
    """Select a contiguous window from a shuffled task pool."""
    if offset < 0:
        logging.error(f"{label} offset must be non-negative, got {offset}")
        return None
    if offset + total_envs > len(items):
        logging.error(
            f"Not enough {label}: need window [{offset}, {offset + total_envs}), have {len(items)} total"
        )
        return None
    return items[offset: offset + total_envs]


def prepare_alfworld_game_files(env_type: str, total_envs: int, seed: int) -> Optional[List[str]]:
    """Prepare unique game files for AlfWorld if requested"""
    if env_type != "alfworld":
        return None
        
    from agentdebug.environments.alfworld.envs import load_config_file
    from agentdebug.environments.alfworld.alfworld.agents.environment import get_environment
    
    alf_config_path = os.path.join(
        os.path.dirname(__file__),
        '../../agentdebug.environments.alfworld/configs/config_tw.yaml'
    )
    
    try:
        cfg = load_config_file(alf_config_path)
        env_type = cfg['env']['type']
        BaseEnvCls = get_environment(env_type)
        tmp_env = BaseEnvCls(cfg, train_eval='train')
        tmp_env.collect_game_files()
        all_game_files = list(getattr(tmp_env, 'game_files', []))
        
        rng = random.Random(seed)
        rng.shuffle(all_game_files)
        return all_game_files
        
    except Exception as e:
        logging.error(f"Failed to collect game files: {e}")
        return None


def prepare_webshop_goal_indices(total_envs: int, seed: int, use_train_set: bool) -> Optional[List[int]]:
    """Prepare unique WebShop goal indices for one rollout run."""
    try:
        from agentdebug.environments.webshop import build_webshop_envs

        tmp_env = build_webshop_envs(
            seed=seed,
            env_num=1,
            group_n=1,
            is_train=use_train_set,
        )
        try:
            all_goal_indices = list(getattr(tmp_env, "goal_idxs", []))
        finally:
            tmp_env.close()

        rng = random.Random(seed)
        rng.shuffle(all_goal_indices)
        return all_goal_indices
    except Exception as e:
        logging.error(f"Failed to collect WebShop goal indices: {e}")
        return None


def run_single_attempt(env_manager, agent, args, batch_idx, test_idx, attempt_idx, 
                       global_env_counter, run_ts, chat_base_dir, dump_fp, 
                       current_batch_size) -> Dict:
    """Run a single attempt for best-of-N sampling"""
    
    logging.info(f"  Attempt {attempt_idx + 1}/{args.best_of_n}")
    start_time = time.time()
    
    obs, infos = env_manager.reset()
    env_dones = [False] * current_batch_size
    
    # Per-env chat buffers
    chats = [[] for _ in range(current_batch_size)]
    saved_flags = [False] * current_batch_size
    last_infos = infos
    
    # Statistics for single attempt
    overall_success_this_attempt = np.zeros(current_batch_size, dtype=bool)
    task_success_cnt = defaultdict(int)
    task_total_cnt = defaultdict(int)
    
    for step_idx in range(args.max_steps):
        logging.debug(f"  Attempt {attempt_idx + 1} Step {step_idx}; Dones ({np.array(env_dones).sum()}/{current_batch_size}); SR {overall_success_this_attempt.mean():.3f}")
        
        # Assemble actions
        prompts = []
        idx_map = []
        for i in range(current_batch_size):
            if not env_dones[i]:
                prompts.append(obs["text"][i])
                idx_map.append(i)
        
        if not prompts:
            break
        
        batch_actions = agent.get_actions_batch(
            prompts, 
            concurrency=args.concurrency, 
            retries=args.retries
        )
        
        actions = ["None"] * current_batch_size
        for k, i in enumerate(idx_map):
            actions[i] = batch_actions[k]
        
        # Environment stepping
        prev_prompts = obs["text"]
        raw_actions = actions.copy()
        obs, rewards, dones, infos = env_manager.step(actions.copy())
        last_infos = infos
        
        # Process results
        for i in range(current_batch_size):
            if env_dones[i]:
                continue
            
            # Append chat history
            if prev_prompts and i < len(prev_prompts):
                chats[i].append({"role": "user", "content": prev_prompts[i]})
            chats[i].append({"role": "assistant", "content": raw_actions[i]})
            
            # Dump trajectory
            if args.dump_path and (i in idx_map):
                try:
                    row = {
                        "batch_idx": batch_idx,
                        "test_idx": test_idx,
                        "attempt_idx": attempt_idx,  # Add attempt index for best-of-N
                        "step": step_idx,
                        "env_id": global_env_counter + i,
                        "environment": args.env,
                        "provider": agent.provider,
                        "model": args.model,
                        "temperature": agent.temperature,
                        "prompt": prev_prompts[i],
                        "action": raw_actions[i],
                        "reward": float(rewards[i]) if i < len(rewards) else None,
                        "done": bool(dones[i]) if i < len(dones) else None,
                        "won": bool(infos[i].get("won", False)),
                        "is_action_valid": bool(infos[i].get("is_action_valid", False)),
                    }
                    
                    # Add environment-specific fields
                    if args.env == "gaia":
                        row["pid"] = infos[i].get("pid", "unknown")
                    elif args.env == "alfworld":
                        row["gamefile"] = infos[i].get("extra.gamefile", "")
                    elif args.env == "webshop":
                        row["task_score"] = float(infos[i].get("task_score", 0))
                    
                    dump_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                    dump_fp.flush()
                except Exception as e:
                    logging.debug(f"Dump error: {e}")
            
            # Check if done
            if dones[i]:
                env_dones[i] = True
                won = bool(infos[i].get("won", False))
                overall_success_this_attempt[i] = won
                
                # Track task success
                # Use environment index as part of task ID to ensure each env is tracked separately
                if args.env == "gaia":
                    task_id = f"env_{i}_{infos[i].get('pid', f'task_{i}')}"
                elif args.env == "alfworld":
                    gamefile = infos[i].get("extra.gamefile", "")
                    # Extract task type from gamefile
                    task_types = ["pick_and_place", "pick_two_obj_and_place", 
                                 "look_at_obj_in_light", "pick_heat_then_place_in_recep",
                                 "pick_cool_then_place_in_recep", "pick_clean_then_place_in_recep"]
                    task_type = "other"
                    for t in task_types:
                        if t in gamefile:
                            task_type = t
                            break
                    # Include env index to track each environment separately
                    task_id = f"env_{i}_{task_type}"
                else:  # webshop
                    task_id = f"env_{i}_task"
                
                # Set to 1 to indicate this task was encountered
                task_total_cnt[task_id] = 1
                if won:
                    task_success_cnt[task_id] = 1
                
                # Save chat history (only for successful attempts or if it's the last attempt)
                if chat_base_dir and not saved_flags[i] and (won or attempt_idx == args.best_of_n - 1):
                    try:
                        task_hash = hashlib.sha1(str(task_id).encode()).hexdigest()[:8]
                        unique_id = f"b{batch_idx:03d}_t{test_idx:02d}_e{i:02d}_a{attempt_idx:02d}-{task_hash}"
                        out_path = os.path.join(chat_base_dir, f"chat_{unique_id}.json")
                        
                        meta = {
                            "batch_idx": batch_idx,
                            "env_id": global_env_counter + i,
                            "test_idx": test_idx,
                            "attempt_idx": attempt_idx,
                            "provider": agent.provider,
                            "model": args.model,
                            "steps": step_idx + 1,
                            "won": won,
                            "timestamp": run_ts,
                            "environment": args.env,
                            "best_of_n": args.best_of_n,
                            "temperature": agent.temperature,
                        }
                        
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump({"messages": chats[i], "metadata": meta}, f, ensure_ascii=False, indent=2)
                        saved_flags[i] = True
                    except Exception as e:
                        logging.debug(f"Failed to save chat: {e}")
        
        if all(env_dones):
            break
    
    # Save any unfinished chats (for last attempt only)
    if chat_base_dir and attempt_idx == args.best_of_n - 1:
        for i in range(current_batch_size):
            if not saved_flags[i]:
                try:
                    task_hash = hashlib.sha1(f"unfinished_{i}".encode()).hexdigest()[:8]
                    unique_id = f"b{batch_idx:03d}_t{test_idx:02d}_e{i:02d}_a{attempt_idx:02d}-{task_hash}"
                    out_path = os.path.join(chat_base_dir, f"chat_{unique_id}.json")
                    
                    meta = {
                        "batch_idx": batch_idx,
                        "env_id": global_env_counter + i,
                        "test_idx": test_idx,
                        "attempt_idx": attempt_idx,
                        "provider": agent.provider,
                        "model": args.model,
                        "steps": len(chats[i]) // 2,
                        "won": False,
                        "timestamp": run_ts,
                        "environment": args.env,
                        "best_of_n": args.best_of_n,
                        "temperature": agent.temperature,
                    }
                    
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump({"messages": chats[i], "metadata": meta}, f, ensure_ascii=False, indent=2)
                    saved_flags[i] = True
                except Exception as e:
                    logging.debug(f"Failed to save unfinished chat: {e}")
    
    elapsed_time = time.time() - start_time
    attempt_success_rate = overall_success_this_attempt.mean()
    
    logging.info(f"  Attempt {attempt_idx + 1} success: {attempt_success_rate:.4f}, time: {elapsed_time:.2f}s")
    
    return {
        "success": overall_success_this_attempt,
        "task_success_cnt": task_success_cnt,
        "task_total_cnt": task_total_cnt,
        "elapsed_time": elapsed_time
    }


def main():
    parser = argparse.ArgumentParser(description="Unified rollout script for multiple environments")
    
    # Environment selection
    parser.add_argument("--env", choices=["alfworld", "gaia", "webshop"], required=True,
                       help="Environment to run")
    
    # Common parameters
    parser.add_argument("--batch_size", type=int, default=10, 
                       help="Number of envs to process per batch")
    parser.add_argument("--total_envs", type=int, default=100, 
                       help="Total number of environments to rollout")
    parser.add_argument("--test_times", type=int, default=1,
                       help="Number of test runs per batch")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Maximum steps per episode (default: 30)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--history_length", type=int, default=3)
    
    # Model parameters
    parser.add_argument("--model", default="gpt-4o",
                       help="Model name (OpenAI: gpt-4o, gpt-4o-mini; Together: Qwen/Qwen2.5-7B-Instruct-Turbo, etc.)")
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "openai", "anthropic", "gemini", "together", "openai-compatible"],
        help="Model provider routing mode",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--base_url", default=None,
                       help="OpenAI-compatible base URL (e.g., vLLM http://127.0.0.1:8000/v1)")
    
    # Execution parameters
    parser.add_argument("--concurrency", type=int, default=4,
                       help="Max concurrent LLM requests per step")
    parser.add_argument("--retries", type=int, default=3,
                       help="Retries per request on failure")
    parser.add_argument("--pool_offset", type=int, default=0,
                       help="Offset into the shuffled task/environment pool before taking total_envs items")
    
    # Output parameters
    parser.add_argument("--dump_path", default=None,
                       help="If set, write JSONL trajectory to this file")
    parser.add_argument(
        "--chat_root",
        default=os.getcwd(),
        help=(
            "Root directory to save per-episode chat histories. "
            "Defaults to the current working directory."
        ),
    )
    
    # Environment-specific parameters
    parser.add_argument("--alf_env_type", default="alfworld/AlfredTWEnv",
                       help="AlfWorld environment type")
    parser.add_argument("--gaia_data_path", default="data/gaia/val.json",
                       help="Path to GAIA dataset")
    parser.add_argument("--gaia_tools", nargs='+', 
                       default=['google_search', 'wikipedia_knowledge_searcher', 'python_code_generator'],
                       help="List of available tools for GAIA")
    parser.add_argument("--webshop_train", action="store_true",
                       help="Use WebShop training set instead of test set")
    
    # Best-of-N sampling options
    parser.add_argument("--best_of_n", type=int, default=1,
                       help="Number of independent attempts per task (best-of-N sampling)")
    parser.add_argument("--best_of_n_temp", type=float, default=1.2,
                       help="Temperature for best-of-N sampling (default: 1.2)")
    
    # Other options
    parser.add_argument("--unique_envs", action="store_true",
                       help="Ensure unique tasks/games across all environments")
    parser.add_argument("--dry_run", action="store_true",
                       help="Only print batch allocation without running")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    # Summary-related options  
    parser.add_argument("--use_summary", action="store_true",
                       help="Enable memory summarization instead of sliding window")
    parser.add_argument("--summary_api_key", default=None,
                       help="API key for summary LLM (defaults to environment variables)")
    parser.add_argument("--summary_endpoint", default=None, 
                       help="API endpoint for summary LLM (defaults to environment variables)")
    
    args = parser.parse_args()
    
    # Set default max_steps based on environment
    if args.max_steps is None:
        args.max_steps = {
            "alfworld": 30,
            "gaia": 30,
            "webshop": 30
        }[args.env]
    
    # Setup logging
    os.makedirs(f"logs/{args.env}", exist_ok=True)
    log_fp = os.path.join(
        f"logs/{args.env}", 
        f"unified_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )
    
    logging.info(f"Starting unified rollout for {args.env}")
    logging.info(f"Model: {args.model}, Provider: {args.provider}, Temperature: {args.temperature}")
    logging.info(f"Total envs: {args.total_envs}, Batch size: {args.batch_size}, Max steps: {args.max_steps}")
    
    # Calculate number of batches
    num_batches = (args.total_envs + args.batch_size - 1) // args.batch_size
    logging.info(f"Running {args.total_envs} envs in {num_batches} batches")
    
    # Prepare output files
    dump_fp = None
    if args.dump_path:
        os.makedirs(os.path.dirname(args.dump_path) or ".", exist_ok=True)
        dump_fp = open(args.dump_path, "a", encoding="utf-8")
        logging.info(f"Dumping trajectories to: {args.dump_path}")
    
    # Prepare chat history directories
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    chat_base_dir = None
    if args.chat_root:
        chat_ts_root = os.path.join(args.chat_root, 'trajectories', run_ts)
        chat_base_dir = os.path.join(chat_ts_root, args.env, args.model.replace('/', '_'))
        os.makedirs(chat_base_dir, exist_ok=True)
        logging.info(f"Saving chats to: {chat_base_dir}")
    
    def _sanitize(s: str) -> str:
        """Sanitize string for filename"""
        return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '-' for c in str(s))[:200]
    
    # Prepare environment-specific data
    gaia_tasks = None
    alfworld_game_files = None
    
    if args.env == "gaia":
        logging.info(f"Loading GAIA tasks from {args.gaia_data_path}")
        gaia_tasks = load_gaia_tasks(args.gaia_data_path)
        logging.info(f"Loaded {len(gaia_tasks)} tasks")

        # Shuffle tasks for random sampling
        rng = random.Random(args.seed)
        rng.shuffle(gaia_tasks)
        selected_tasks = select_task_window(gaia_tasks, args.total_envs, args.pool_offset, "GAIA tasks")
        if selected_tasks is None:
            sys.exit(1)
        gaia_tasks = selected_tasks
        logging.info(f"Selected GAIA task window offset={args.pool_offset} size={len(gaia_tasks)}")
        
    elif args.env == "alfworld" and args.unique_envs:
        alfworld_game_files = prepare_alfworld_game_files(args.env, args.total_envs, args.seed)
        if alfworld_game_files:
            selected_game_files = select_task_window(
                alfworld_game_files, args.total_envs, args.pool_offset, "AlfWorld game files"
            )
            if selected_game_files is None:
                sys.exit(1)
            alfworld_game_files = selected_game_files
            logging.info(
                f"Prepared {len(alfworld_game_files)} unique game files "
                f"(offset={args.pool_offset})"
            )
    elif args.env == "webshop" and args.unique_envs:
        webshop_goal_indices = prepare_webshop_goal_indices(args.total_envs, args.seed, args.webshop_train)
        if webshop_goal_indices:
            selected_goal_indices = select_task_window(
                webshop_goal_indices, args.total_envs, args.pool_offset, "WebShop goal indices"
            )
            if selected_goal_indices is None:
                sys.exit(1)
            webshop_goal_indices = selected_goal_indices
            logging.info(
                f"Prepared {len(webshop_goal_indices)} unique WebShop goal indices "
                f"(offset={args.pool_offset})"
            )
    
    # Dry run mode
    if args.dry_run:
        logging.info(f"[Dry-Run] Environment: {args.env}")
        logging.info(f"[Dry-Run] Total envs: {args.total_envs}, Batches: {num_batches}")
        
        for b in range(num_batches):
            start = b * args.batch_size
            end = min(start + args.batch_size, args.total_envs)
            batch_size = end - start
            
            if args.env == "gaia" and gaia_tasks:
                batch_tasks = gaia_tasks[start:end]
                pids = [t.get('pid', f'task_{i}') for i, t in enumerate(batch_tasks)]
                logging.info(f"[Dry-Run] Batch {b+1:02d}: {batch_size} tasks; PIDs: {', '.join(pids[:3])}...")
            elif args.env == "alfworld" and alfworld_game_files:
                batch_files = alfworld_game_files[start:end]
                examples = [os.path.basename(f) for f in batch_files[:3]]
                logging.info(f"[Dry-Run] Batch {b+1:02d}: {batch_size} files; Examples: {', '.join(examples)}")
            elif args.env == "webshop" and args.unique_envs and webshop_goal_indices:
                batch_goal_indices = webshop_goal_indices[start:end]
                examples = [str(idx) for idx in batch_goal_indices[:3]]
                logging.info(f"[Dry-Run] Batch {b+1:02d}: {batch_size} goal indices; Examples: {', '.join(examples)}")
            else:
                logging.info(f"[Dry-Run] Batch {b+1:02d}: {batch_size} environments")
        
        sys.exit(0)
    
    # Initialize agent (defer until after potential dry-run exit to avoid requiring API keys)
    agent = UnifiedAgent(
        model_name=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
        env_type=args.env,
        provider=args.provider,
    )
    
    if args.best_of_n > 1:
        logging.info(f"Best-of-N sampling enabled: {args.best_of_n} attempts per task with temperature {args.best_of_n_temp}")

    # Statistics tracking
    all_overall_success_rates = []
    all_task_success_history = defaultdict(list)
    global_env_counter = 0
    
    # Main rollout loop
    try:
        for batch_idx in range(num_batches):
            # Calculate actual batch size
            current_batch_size = min(args.batch_size, args.total_envs - batch_idx * args.batch_size)
            logging.info(f"\n========== Starting Batch {batch_idx + 1}/{num_batches} with {current_batch_size} envs ==========")
            
            # Prepare environment-specific kwargs
            env_kwargs = {
                "env_num": current_batch_size,
                "seed": args.seed + batch_idx,
                "history_length": args.history_length,
                # Summary configuration
                "use_summary": args.use_summary,
                "summary_api_key": args.summary_api_key or os.getenv("OAI_KEY") or os.getenv("OPENAI_API_KEY"),
                "summary_endpoint": args.summary_endpoint or os.getenv("OAI_ENDPOINT") or os.getenv("OPENAI_ENDPOINT"),
            }
            
            if args.env == "gaia":
                start = batch_idx * args.batch_size
                end = start + current_batch_size
                env_kwargs["tasks_data"] = gaia_tasks[start:end]
                env_kwargs["available_tools"] = args.gaia_tools
                env_kwargs["max_steps"] = args.max_steps
                
            elif args.env == "alfworld":
                env_kwargs["alf_env_type"] = args.alf_env_type
                if alfworld_game_files:
                    start = batch_idx * args.batch_size
                    end = start + current_batch_size
                    env_kwargs["game_files"] = alfworld_game_files[start:end]
                    
            elif args.env == "webshop":
                env_kwargs["use_train_set"] = args.webshop_train
            
            # Create environment
            env_manager = EnvironmentFactory.build_env(args.env, **env_kwargs)
            batch_session_indices = None
            if args.env == "webshop" and args.unique_envs and webshop_goal_indices:
                start = batch_idx * args.batch_size
                end = start + current_batch_size
                batch_session_indices = webshop_goal_indices[start:end]
            
            # Batch-level statistics
            batch_overall_success_rates = []
            batch_task_success_history = defaultdict(list)
            
            try:
                # Test loop for this batch
                for test_idx in range(args.test_times):
                    logging.info(f"\n========== Start Batch {batch_idx + 1} Test {test_idx} ==========")
                    start_time = time.time()
                    
                    if batch_session_indices is not None:
                        obs, infos = env_manager.reset(session_indices=batch_session_indices)
                    else:
                        obs, infos = env_manager.reset()
                    env_dones = [False] * current_batch_size
                    
                    # Per-env chat buffers
                    chats = [[] for _ in range(current_batch_size)]
                    saved_flags = [False] * current_batch_size
                    last_infos = infos
                    
                    # Statistics for single round
                    overall_success_this_round = np.zeros(current_batch_size, dtype=bool)
                    task_success_cnt = defaultdict(int)
                    task_total_cnt = defaultdict(int)
                    
                    for step_idx in range(args.max_steps):
                        logging.info(f"Batch {batch_idx + 1} Step {step_idx}; Dones ({np.array(env_dones).sum()}/{current_batch_size}); SR {overall_success_this_round.mean():.3f}")
                        
                        # Assemble actions
                        prompts = []
                        idx_map = []
                        for i in range(current_batch_size):
                            if not env_dones[i]:
                                prompts.append(obs["text"][i])
                                idx_map.append(i)
                        
                        if not prompts:
                            break
                        
                        batch_actions = agent.get_actions_batch(
                            prompts, 
                            concurrency=args.concurrency, 
                            retries=args.retries
                        )
                        
                        actions = ["None"] * current_batch_size
                        for k, i in enumerate(idx_map):
                            actions[i] = batch_actions[k]
                        
                        # Environment stepping
                        prev_prompts = obs["text"]
                        raw_actions = actions.copy()
                        obs, rewards, dones, infos = env_manager.step(actions.copy())
                        last_infos = infos
                        
                        # Process results
                        for i in range(current_batch_size):
                            if env_dones[i]:
                                continue
                            
                            # Append chat history
                            if prev_prompts and i < len(prev_prompts):
                                chats[i].append({"role": "user", "content": prev_prompts[i]})
                            chats[i].append({"role": "assistant", "content": raw_actions[i]})
                            
                            # Dump trajectory
                            if args.dump_path and (i in idx_map):
                                try:
                                    row = {
                                        "batch_idx": batch_idx,
                                        "test_idx": test_idx,
                                        "step": step_idx,
                                        "env_id": global_env_counter + i,
                        "environment": args.env,
                        "provider": agent.provider,
                        "model": args.model,
                        "temperature": agent.temperature,
                                        "prompt": prev_prompts[i],
                                        "action": raw_actions[i],
                                        "reward": float(rewards[i]) if i < len(rewards) else None,
                                        "done": bool(dones[i]) if i < len(dones) else None,
                                        "won": bool(infos[i].get("won", False)),
                                        "is_action_valid": bool(infos[i].get("is_action_valid", False)),
                                    }
                                    
                                    # Add environment-specific fields
                                    if args.env == "gaia":
                                        row["pid"] = infos[i].get("pid", "unknown")
                                    elif args.env == "alfworld":
                                        row["gamefile"] = infos[i].get("extra.gamefile", "")
                                    elif args.env == "webshop":
                                        row["session_index"] = infos[i].get("session_index")
                                        row["task_score"] = float(infos[i].get("task_score", 0))
                                    
                                    dump_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                                except Exception as e:
                                    logging.debug(f"Dump error: {e}")
                            
                            # Check if done
                            if dones[i]:
                                env_dones[i] = True
                                won = bool(infos[i].get("won", False))
                                overall_success_this_round[i] = won
                                
                                # Track task success
                                if args.env == "gaia":
                                    task_id = infos[i].get("pid", f"task_{i}")
                                elif args.env == "alfworld":
                                    gamefile = infos[i].get("extra.gamefile", "")
                                    # Extract task type from gamefile
                                    task_types = ["pick_and_place", "pick_two_obj_and_place", 
                                                 "look_at_obj_in_light", "pick_heat_then_place_in_recep",
                                                 "pick_cool_then_place_in_recep", "pick_clean_then_place_in_recep"]
                                    task_id = "other"
                                    for t in task_types:
                                        if t in gamefile:
                                            task_id = t
                                            break
                                else:  # webshop
                                    task_id = f"task_{i}"
                                
                                task_total_cnt[task_id] = 1
                                if won:
                                    task_success_cnt[task_id] = 1
                                
                                # Save chat history
                                if chat_base_dir and not saved_flags[i]:
                                    try:
                                        task_hash = hashlib.sha1(str(task_id).encode()).hexdigest()[:8]
                                        unique_id = f"b{batch_idx:03d}_t{test_idx:02d}_e{i:02d}-{task_hash}"
                                        out_path = os.path.join(chat_base_dir, f"chat_{unique_id}.json")
                                        
                                        meta = {
                                            "batch_idx": batch_idx,
                                            "env_id": global_env_counter + i,
                                            "test_idx": test_idx,
                                            "provider": agent.provider,
                                            "model": args.model,
                                            "steps": step_idx + 1,
                                            "won": won,
                                            "timestamp": run_ts,
                                            "environment": args.env,
                                        }
                                        
                                        with open(out_path, "w", encoding="utf-8") as f:
                                            json.dump({"messages": chats[i], "metadata": meta}, f, ensure_ascii=False, indent=2)
                                        saved_flags[i] = True
                                    except Exception as e:
                                        logging.debug(f"Failed to save chat: {e}")
                        
                        if all(env_dones):
                            logging.info("All environments finished early!")
                            break
                    
                    # Save any unfinished chats
                    if chat_base_dir:
                        for i in range(current_batch_size):
                            if not saved_flags[i]:
                                try:
                                    task_hash = hashlib.sha1(f"unfinished_{i}".encode()).hexdigest()[:8]
                                    unique_id = f"b{batch_idx:03d}_t{test_idx:02d}_e{i:02d}-{task_hash}"
                                    out_path = os.path.join(chat_base_dir, f"chat_{unique_id}.json")
                                    
                                    meta = {
                                        "batch_idx": batch_idx,
                                        "env_id": global_env_counter + i,
                                        "test_idx": test_idx,
                                        "provider": agent.provider,
                                        "model": args.model,
                                        "steps": len(chats[i]) // 2,
                                        "won": False,
                                        "timestamp": run_ts,
                                        "environment": args.env,
                                    }
                                    
                                    with open(out_path, "w", encoding="utf-8") as f:
                                        json.dump({"messages": chats[i], "metadata": meta}, f, ensure_ascii=False, indent=2)
                                    saved_flags[i] = True
                                except Exception as e:
                                    logging.debug(f"Failed to save unfinished chat: {e}")
                    
                    # Round statistics
                    round_success_rate = overall_success_this_round.mean()
                    batch_overall_success_rates.append(round_success_rate)
                    
                    if args.best_of_n > 1:
                        logging.info(f"Batch {batch_idx + 1} Test {test_idx} final success (best-of-{args.best_of_n}): {round_success_rate:.4f}")
                    else:
                        logging.info(f"Batch {batch_idx + 1} Test {test_idx} overall success: {round_success_rate:.4f}")
                    
                    # Calculate and store per-task success rates for this test
                    # Aggregate by task type (remove env_ prefix for statistics)
                    task_type_success = defaultdict(int)
                    task_type_total = defaultdict(int)
                    
                    for task, total in task_total_cnt.items():
                        # Extract task type from task_id (format: "env_{i}_{task_type}")
                        if task.startswith("env_"):
                            parts = task.split("_", 2)
                            if len(parts) >= 3:
                                task_type = parts[2]
                            else:
                                task_type = "unknown"
                        else:
                            task_type = task
                        
                        task_type_total[task_type] += total
                        task_type_success[task_type] += task_success_cnt.get(task, 0)
                    
                    # Log and store statistics by task type
                    for task_type, total in task_type_total.items():
                        if total > 0:
                            rate = task_type_success[task_type] / total
                            batch_task_success_history[task_type].append(rate)
                            
                            # Log task-specific results for alfworld
                            if args.env == "alfworld":
                                logging.info(f"    {task_type:<35s}: {rate:.4f} ({task_type_success[task_type]}/{total})")
                    
                    total_elapsed_time = time.time() - start_time
                    logging.info(f"Batch {batch_idx + 1} Test {test_idx} time elapsed: {total_elapsed_time:.2f}s\n")
                
            finally:
                # Accumulate batch results
                all_overall_success_rates.extend(batch_overall_success_rates)
                for task, rates in batch_task_success_history.items():
                    all_task_success_history[task].extend(rates)
                
                # Update global counter
                global_env_counter += current_batch_size
                
                # Clean up resources
                try:
                    env_manager.envs.close()
                    logging.info(f"Released resources for Batch {batch_idx + 1}")
                except Exception as e:
                    logging.warning(f"Failed to release resources: {e}")
                
                logging.info(f"========== Finished Batch {batch_idx + 1}/{num_batches}, processed {global_env_counter}/{args.total_envs} envs ==========\n")
        
    finally:
        if dump_fp is not None:
            dump_fp.flush()
            dump_fp.close()
            logging.info(f"Trajectories saved to: {args.dump_path}")
    
    # Final summary
    logging.info("=============== Final Summary ===============")
    logging.info(f"Environment: {args.env}")
    logging.info(f"Total batches: {num_batches} | Batch size: {args.batch_size} | Total envs processed: {global_env_counter}")
    
    if args.best_of_n > 1:
        logging.info(f"Best-of-N sampling: {args.best_of_n} attempts per task with temperature {args.best_of_n_temp}")
    
    # Echo final artifact locations for convenience
    if args.dump_path:
        logging.info(f"Trajectory file: {args.dump_path}")
    if chat_base_dir:
        logging.info(f"Chats directory: {chat_base_dir}")
    
    if all_overall_success_rates:
        success_label = f"Overall success (best-of-{args.best_of_n})" if args.best_of_n > 1 else "Overall success"
        logging.info(
            f"{success_label} avg ± std: "
            f"{np.mean(all_overall_success_rates):.4f} ± {np.std(all_overall_success_rates):.4f}"
        )
    
    # Environment-specific summaries
    if args.env == "alfworld":
        task_types = ["pick_and_place", "pick_two_obj_and_place", "look_at_obj_in_light",
                     "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", 
                     "pick_clean_then_place_in_recep", "other"]
        for task in task_types:
            if task in all_task_success_history and all_task_success_history[task]:
                rates = [r for r in all_task_success_history[task] if r is not None]
                if rates:
                    logging.info(f"{task:<35s}: {np.mean(rates):.4f} ± {np.std(rates):.4f}")
    
    elif args.env == "gaia":
        successful_tasks = sum(1 for rates in all_task_success_history.values() if any(r > 0 for r in rates))
        logging.info(f"Successfully completed {successful_tasks} out of {len(all_task_success_history)} unique tasks")


if __name__ == "__main__":
    main()
