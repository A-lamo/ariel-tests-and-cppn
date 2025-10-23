import numpy as np
from pathlib import Path
import mujoco
from mujoco import viewer
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.cppn_neat.id_manager import IdManager
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
    DECODER_TYPE = "best_first"
    MAX_MODULES = 20
    num_initial_mutations = 0
    
    T, R = NUM_OF_TYPES_OF_MODULES, NUM_OF_ROTATIONS
    NUM_CPPN_INPUTS, NUM_CPPN_OUTPUTS = 6, 1 + T + R
    
    # 1. Define the starting innovation ID for the first genome.
    initial_innov_id = NUM_CPPN_INPUTS * NUM_CPPN_OUTPUTS

    # 2. Calculate the next available ID AFTER the first genome is created.
    next_available_innov_id = initial_innov_id + (NUM_CPPN_INPUTS * NUM_CPPN_OUTPUTS)
    
    # 3. Initialize the IdManager correctly.
    id_manager = IdManager(
        node_start=NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS - 1,
        innov_start=next_available_innov_id - 1
    )
    # 4. Create the initial random genome.
    my_cppn_genome = Genome.random(
        num_inputs=NUM_CPPN_INPUTS, num_outputs=NUM_CPPN_OUTPUTS,
        next_node_id=(NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS),
        next_innov_id=initial_innov_id, 
    )

    # 5. Apply initial mutations to the genome.
    #  (this was done in old revolve2, and it makes sense to keep it but it's optional)
    for i in range(num_initial_mutations):
        print(f"Applying mutation {i+1}/{num_initial_mutations}...")
        my_cppn_genome.mutate(
                                1.0, # Use floats for rates
                                1.0, # Use floats for rates
                                id_manager.get_next_innov_id,
                                id_manager.get_next_node_id
                                )
        print("Number of Nodes: ", len(my_cppn_genome.nodes))
        print("Number of Connections: ", len(my_cppn_genome.connections))

    # 6. Decode the genome into a robot morphology.    
    console.log(f"Decoding with [bold cyan]{DECODER_TYPE}[/bold cyan]...")
    
    decoder = MorphologyDecoderBestFirst(
        cppn_genome=my_cppn_genome, 
        max_modules=MAX_MODULES
    )
    decoded_robot_graph = decoder.decode()


    console.log(f"[bold green]Success! Found a morphology with {decoded_robot_graph.number_of_nodes()} modules.[/bold green]")
    final_robot_graph = decoded_robot_graph

    # 7. Construct and run the final robot.
    if final_robot_graph:
        console.log("\nConstructing final robot...")
        try:
            core = construct_mjspec_from_graph(final_robot_graph)
            console.log("[bold green]Robot constructed! Starting simulation...[/bold green]")
            run(core, with_viewer=True)
        except Exception as e:
            console.log(f"[bold red]Failed to construct or run the robot: {e}[/bold red]")
    else:
        console.log(f"[bold red]Failed to generate a valid robot.[/bold red]")