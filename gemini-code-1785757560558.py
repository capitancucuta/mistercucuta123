"""Store Challenge Pipeline: Classifies products via YOLO + FoundationPose, 
routes misplaced items to the return tray, and valid items to matching stock shelves.
"""

from __future__ import annotations

import argparse
from typing import Any

from foundation_pose import DEFAULT_CAMERA, DEFAULT_TIMEOUT_S
from foundation_pose_ioai_sim_scene import (
    DEFAULT_BRIDGE_DIR,
    IoaiSimSceneFPPickAgent,
    make_perception_env,
)
from ioailab.agents import TaskFlowAgent, TaskFlowSpec
from ioailab.tasks.ioai_sim_scene import GALBOT_G1_IOAI_SIM_SCENE_TASK_ID
from ioailab.tasks.ioai_sim_scene.policy_pipeline import (
    CachedPolicyAgent,
    PolicyAgentCache,
    ScenarioMatchedNavAgent,
    load_policy_manifest,
)
from ioailab.tasks.ioai_sim_scene.runtime_state import activate_pipeline_product
from ioailab.tasks.ioai_sim_scene.table_layout import load_ioai_table_layout
from ioailab.utils.log_utils import configure, get_logger

logger = get_logger(__name__)

# Color and Shelf Rules for Store Tasks
STOCK_COLOR_MAPPING = {
    "gum": "shelf_red",
    "cocacola": "shelf_red",
    "water": "shelf_blue",
    "pepsichips": "shelf_blue",
    "coffee": "shelf_yellow",
    "beefnoodle": "shelf_yellow",
}


def is_item_misplaced(product_id: str, current_slot: str) -> bool:
    """Check if an item is misplaced in its current slot based on color/type rules."""
    # Example logic: Slot A1/A2 expect red items, B1/B2 expect blue/yellow
    expected_zone = "A" if product_id in ["gum", "cocacola"] else "B"
    return not current_slot.startswith(expected_zone)


def build_dynamic_product_agent(
    env: Any,
    product_id: str,
    current_slot: str,
    *,
    yolo_model: str,
    bridge_dir: str,
    manifest: Any,
    cache: PolicyAgentCache,
) -> TaskFlowAgent:
    """Builds a TaskFlowAgent dynamically:
    - If misplaced: Route to Return Tray (Cocoa single-phase Pick flow)
    - If correct stock: Route to Pick -> Nav -> Place policy flow
    """
    root_flow = env.unwrapped.cfg.task_flow
    misplaced = is_item_misplaced(product_id, current_slot)

    # 1. Perception-backed Pick Agent (Perception Seam)
    fp_pick_agent = IoaiSimSceneFPPickAgent(
        product_id=product_id,
        yolo_model=yolo_model,
        bridge_dir=bridge_dir,
    )

    if misplaced or product_id == "cocoa":
        logger.info(f"Item {product_id} in {current_slot} is MISPLACED. Routing to Return Tray.")
        # Cocoa trick: Pick phase deposits misplaced items into tray_2
        pick_only_flow = TaskFlowSpec(
            phases=(root_flow.phase("pick"),),
            final_phase="pick",
            phase_state_getter=root_flow.phase_state_getter,
        )
        return TaskFlowAgent(pick_only_flow, agents={"pick": fp_pick_agent}, env=env)

    logger.info(f"Item {product_id} is MATCHING STOCK. Routing to Shelf Place.")
    # Standard continuous Pick -> Nav -> Place flow
    bundle = manifest.require(product_id)
    phase_agents = {
        "pick": fp_pick_agent,
        "nav": ScenarioMatchedNavAgent(
            product_id=product_id,
            references=bundle.place_start_references,
        ),
        "place": CachedPolicyAgent(
            cache,
            product_id=product_id,
            phase="place",
            checkpoint=bundle.place_checkpoint,
        ),
    }
    return TaskFlowAgent.from_env(env, agents=phase_agents)


def execute_store_task(
    table_layout_path: str,
    policy_manifest_path: str,
    yolo_model: str,
    bridge_dir: str,
    headless: bool = False,
):
    configure()
    layout = load_ioai_table_layout(table_layout_path)
    manifest = load_policy_manifest(policy_manifest_path, required_products=layout.products)
    
    env = make_perception_env(
        GALBOT_G1_IOAI_SIM_SCENE_TASK_ID,
        num_envs=1,
        headless=headless,
        task_options={
            "pick_product": "gum",
            "table_layout": table_layout_path,
            "defer_success_termination": True,
        },
    )

    cache = PolicyAgentCache(capacity=2)

    try:
        env.reset()
        # Process objects 1 by 1 according to layout slots
        for slot, product_id in zip(layout.slots, layout.products):
            logger.info(f"--- Processing slot {slot} with product {product_id} ---")
            activate_pipeline_product(env, product_id)
            
            agent = build_dynamic_product_agent(
                env=env,
                product_id=product_id,
                current_slot=slot,
                yolo_model=yolo_model,
                bridge_dir=bridge_dir,
                manifest=manifest,
                cache=cache,
            )
            
            agent.reset(env)
            steps = 0
            while steps < 3000:
                action = agent.act(env)
                _, _, terminated, truncated, _ = env.raw_env.step(action)
                steps += 1
                if agent.done(env) or terminated or truncated:
                    break
            agent.close()
            
    finally:
        cache.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-layout", default="outputs/ioai_table_layout.yaml")
    parser.add_argument("--policy-manifest", default="outputs/ioai_policy_baseline.yaml")
    parser.add_argument("--yolo-model", required=True)
    parser.add_argument("--bridge-dir", default=DEFAULT_BRIDGE_DIR)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    execute_store_task(
        table_layout_path=args.table_layout,
        policy_manifest_path=args.policy_manifest,
        yolo_model=args.yolo_model,
        bridge_dir=args.bridge_dir,
        headless=args.headless,
    )