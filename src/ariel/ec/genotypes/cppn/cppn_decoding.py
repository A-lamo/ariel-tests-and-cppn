import numpy as np
import numpy.typing as npt
from pathlib import Path
import mujoco
from mujoco import viewer
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.decoders.cppn_breadth_first import MorphologyDecoderBFS
from ariel.body_phenotypes.robogen_lite.decoders.cppn_best_first import MorphologyDecoderBestFirst

from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
)
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer
from src.ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome

SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)
SEED = 42
console = Console()
RNG = np.random.default_rng(SEED)
np.set_printoptions(precision=3, suppress=True)

def softmax(raw_scores: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    e_x = np.exp(raw_scores - np.max(raw_scores))
    return e_x / e_x.sum()


def run(robot: CoreModule, *, with_viewer: bool = False) -> None:
    world = SimpleFlatWorld()
    world.spawn(robot.spec)
    model = world.spec.compile()
    data = mujoco.MjData(model)
    xml = world.spec.to_xml()
    with (DATA / f"{SCRIPT_NAME}.xml").open("w", encoding="utf-8") as f:
        f.write(xml)
    console.log(f"DoF (model.nv): {model.nv}, Actuators (model.nu): {model.nu}")
    mujoco.mj_resetData(model, data)
    single_frame_renderer(model, data, steps=10)
    if with_viewer:
        viewer.launch(model=model, data=data)


if __name__ == "__main__":
    console.rule("[bold green]CPPN Morphology Generation with Improved Growth[/bold green]")
    
    #Choose the decoder type and parameters
    DECODER_TYPE = "bfs"  # "best_first" or "bfs"
    CORE_BIAS_VALUE = 5.0  # Try values like 0.5, 1.0, 5.0. (Only for BFS though, but I don't know how much it helps yet lol)
    
    MAX_MODULES = 20
    MAX_ATTEMPTS = 10 # Number of attempts to find a morphology that's not just a core lol
    
    T, R = NUM_OF_TYPES_OF_MODULES, NUM_OF_ROTATIONS
    NUM_CPPN_INPUTS, NUM_CPPN_OUTPUTS = 6, 1 + T + R
    
    final_robot_graph = None
    attempt = 0

    while attempt < MAX_ATTEMPTS:
        attempt += 1
        console.log(f"\n--- Attempt {attempt}/{MAX_ATTEMPTS} ---")

        my_cppn_genome = Genome.random(
            num_inputs=NUM_CPPN_INPUTS, num_outputs=NUM_CPPN_OUTPUTS,
            next_node_id=(NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS),
            next_innov_id=(NUM_CPPN_INPUTS * NUM_CPPN_OUTPUTS),
        )

        console.log(f"Decoding with [bold cyan]{DECODER_TYPE}[/bold cyan]...")
        if DECODER_TYPE == "bfs":
            decoder = MorphologyDecoderBFS(
                cppn_genome=my_cppn_genome, 
                max_modules=MAX_MODULES, 
                core_bias=CORE_BIAS_VALUE
            )
        else:
            decoder = MorphologyDecoderBestFirst(
                cppn_genome=my_cppn_genome, 
                max_modules=MAX_MODULES
            )
        
        decoded_robot_graph = decoder.decode()
        
        num_nodes = decoded_robot_graph.number_of_nodes() if decoded_robot_graph else 0
        if num_nodes >= MAX_MODULES:
            console.log(f"[bold green]Success! Found a morphology with {num_nodes} modules.[/bold green]")
            final_robot_graph = decoded_robot_graph
            break
        else:
            console.log(f"[yellow]Attempt failed. Generated only {num_nodes} modules. Retrying...[/yellow]")
    
    if final_robot_graph:
        console.log("\nConstructing final robot...")
        try:
            core = construct_mjspec_from_graph(final_robot_graph)
            console.log("[bold green]Robot constructed! Starting simulation...[/bold green]")
            run(core, with_viewer=True)
        except Exception as e:
            console.log(f"[bold red]Failed to construct or run the robot: {e}[/bold red]")
    else:
        console.log(f"[bold red]Failed to generate a valid robot after {MAX_ATTEMPTS} attempts.[/bold red]")