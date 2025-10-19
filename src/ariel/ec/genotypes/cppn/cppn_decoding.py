import numpy as np
import numpy.typing as npt
from pathlib import Path
import mujoco
from mujoco import viewer
from rich.console import Console
import networkx as nx

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
    NUM_OF_FACES,
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

class MorphologyDecoderBestFirst:
    """Decodes a CPPN using a true greedy, best-first search strategy."""
    def __init__(self, cppn_genome: Genome, max_modules: int = 20):
        self.cppn_genome = cppn_genome
        self.max_modules = max_modules
        self.face_deltas = {
            ModuleFaces.FRONT: (1, 0, 0), ModuleFaces.BACK: (-1, 0, 0),
            ModuleFaces.TOP: (0, 1, 0), ModuleFaces.BOTTOM: (0, -1, 0),
            ModuleFaces.RIGHT: (0, 0, 1), ModuleFaces.LEFT: (0, 0, -1),
        }

    def _get_child_coords(self, parent_pos: tuple, face: ModuleFaces) -> tuple:
        delta = self.face_deltas[face]
        return (parent_pos[0] + delta[0], parent_pos[1] + delta[1], parent_pos[2] + delta[2])

    def decode(self) -> nx.DiGraph:
        robot_graph = nx.DiGraph()
        occupied_coords = {}
        module_data = {}
        
        core_id, core_pos, core_type, core_rot = IDX_OF_CORE, (0, 0, 0), ModuleType.CORE, ModuleRotationsIdx.DEG_0
        robot_graph.add_node(core_id, type=core_type.name, rotation=core_rot.name)
        occupied_coords[core_pos] = core_id
        module_data[core_id] = {'pos': core_pos, 'type': core_type, 'rot': core_rot}
        
        # The frontier now contains ALL modules with potential open faces.
        frontier = [core_id]
        next_module_id = 1

        # Check for the max module count.
        while len(robot_graph) < self.max_modules:
            potential_connections = []
            
            # At each step, we check the ENTIRE frontier of all existing modules.
            for parent_id in frontier:
                parent_pos = module_data[parent_id]['pos']
                parent_type = module_data[parent_id]['type']
                for face in ModuleFaces:
                    
                    if face not in ALLOWED_FACES[parent_type]:
                        continue
                    
                    child_pos = self._get_child_coords(parent_pos, face)
                    
                    if child_pos in occupied_coords:
                        continue

                    cppn_inputs = list(parent_pos) + list(child_pos)
                    raw_outputs = self.cppn_genome.activate(cppn_inputs)
                    
                    conn_score = raw_outputs[0]
                    type_scores = np.array(raw_outputs[1:1+NUM_OF_TYPES_OF_MODULES])
                    rot_scores = np.array(raw_outputs[1+NUM_OF_TYPES_OF_MODULES:])
                    
                    child_type = ModuleType(np.argmax(softmax(type_scores)))
                    child_rot = ModuleRotationsIdx(np.argmax(softmax(rot_scores)))
                    
                    if child_type not in (ModuleType.NONE, ModuleType.CORE) and \
                       face in ALLOWED_FACES[child_type] and child_rot in ALLOWED_ROTATIONS[child_type]:
                        potential_connections.append({
                            'score': conn_score, 'parent_id': parent_id, 'child_pos': child_pos,
                            'child_type': child_type, 'child_rot': child_rot, 'face': face,
                        })

            if not potential_connections:
                # If there are no possible moves anywhere, we have to stop.
                console.log("[yellow]Decoder stalled: No valid connections found anywhere on the robot.[/yellow]")
                break
            
            best_conn = max(potential_connections, key=lambda x: x['score'])
            
            child_id = next_module_id
            robot_graph.add_node(child_id, type=best_conn['child_type'].name, rotation=best_conn['child_rot'].name)
            robot_graph.add_edge(best_conn['parent_id'], child_id, face=best_conn['face'].name)
            
            occupied_coords[best_conn['child_pos']] = child_id
            module_data[child_id] = {'pos': best_conn['child_pos'], 'type': best_conn['child_type'], 'rot': best_conn['child_rot']}
            
            # I no longer remove the parent, I just add the new child.
            # (I think this makes snakes less likely)
            frontier.append(child_id)
            next_module_id += 1

        return robot_graph


class MorphologyDecoderBFS:
    """Decodes a CPPN using BFS with an adjustable 'core_bias'."""
    def __init__(self, cppn_genome: Genome, max_modules: int = 20, core_bias: float = 0.0):
        self.cppn_genome = cppn_genome
        self.max_modules = max_modules
        self.core_bias = core_bias # New parameter to prioritize the core
        self.face_deltas = {
            ModuleFaces.FRONT: (1, 0, 0), ModuleFaces.BACK: (-1, 0, 0),
            ModuleFaces.TOP: (0, 1, 0), ModuleFaces.BOTTOM: (0, -1, 0),
            ModuleFaces.RIGHT: (0, 0, 1), ModuleFaces.LEFT: (0, 0, -1),
        }

    def _get_child_coords(self, parent_pos: tuple, face: ModuleFaces) -> tuple:
        delta = self.face_deltas[face]
        return (parent_pos[0] + delta[0], parent_pos[1] + delta[1], parent_pos[2] + delta[2])

    def decode(self) -> nx.DiGraph:
        robot_graph = nx.DiGraph()
        occupied_coords = {}
        module_data = {}
        core_id, core_pos, core_type, core_rot = IDX_OF_CORE, (0, 0, 0), ModuleType.CORE, ModuleRotationsIdx.DEG_0
        robot_graph.add_node(core_id, type=core_type.name, rotation=core_rot.name)
        occupied_coords[core_pos] = core_id
        module_data[core_id] = {'pos': core_pos, 'type': core_type, 'rot': core_rot}
        next_module_id = 1
        current_layer = [core_id]

        while current_layer and len(robot_graph) < self.max_modules:
            next_layer = []
            
            for parent_id in current_layer:
                if len(robot_graph) >= self.max_modules: break

                potential_children = []
                parent_pos = module_data[parent_id]['pos']
                parent_type = module_data[parent_id]['type']

                for face in ModuleFaces:
                    if face not in ALLOWED_FACES[parent_type]:
                        continue
                    
                    child_pos = self._get_child_coords(parent_pos, face)
                    
                    if child_pos in occupied_coords:
                        continue

                    cppn_inputs = list(parent_pos) + list(child_pos)
                    raw_outputs = self.cppn_genome.activate(cppn_inputs)
                    
                    conn_score = raw_outputs[0]

                    # this is where I add a bias towards connecting to the core 
                    if parent_id == IDX_OF_CORE:
                        conn_score += self.core_bias

                    type_scores = np.array(raw_outputs[1:1+NUM_OF_TYPES_OF_MODULES])
                    rot_scores = np.array(raw_outputs[1+NUM_OF_TYPES_OF_MODULES:])

                    child_type = ModuleType(np.argmax(softmax(type_scores)))
                    child_rot = ModuleRotationsIdx(np.argmax(softmax(rot_scores)))
                    
                    if child_type not in (ModuleType.NONE, ModuleType.CORE) and \
                       face in ALLOWED_FACES[child_type] and child_rot in ALLOWED_ROTATIONS[child_type]:
                        potential_children.append({
                            'score': conn_score, 'child_pos': child_pos,
                            'child_type': child_type, 'child_rot': child_rot, 'face': face,
                        })
                
                if potential_children:
                    best_child = max(potential_children, key=lambda x: x['score'])

                    if best_child['child_pos'] in occupied_coords:
                        continue
                    
                    child_id = next_module_id
                    robot_graph.add_node(child_id, type=best_child['child_type'].name, rotation=best_child['child_rot'].name)
                    robot_graph.add_edge(parent_id, child_id, face=best_child['face'].name)
                    
                    occupied_coords[best_child['child_pos']] = child_id
                    
                    module_data[child_id] = {'pos': best_child['child_pos'], 'type': best_child['child_type'], 'rot': best_child['child_rot']}
                    next_layer.append(child_id)
                    next_module_id += 1

            current_layer = next_layer

        return robot_graph

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
    DECODER_TYPE = "best_first"  # "best_first" or "bfs"
    CORE_BIAS_VALUE = 0.0  # Try values like 0.5, 1.0, 5.0. (Only for BFS though, but I don't know how much it helps yet lol)
    
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