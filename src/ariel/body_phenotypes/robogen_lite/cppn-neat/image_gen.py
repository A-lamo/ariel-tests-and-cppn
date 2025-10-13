import numpy as np
from PIL import Image
# Assuming the files are in a package structure, use relative imports:
from .genome import Genome 
from .activations import ACTIVATION_FUNCTIONS # To check available functions

# --- Global ID Management (Should be defined globally or in a class) ---
# Start tracking IDs from where the initial nodes/connections would end
global_innov_id = 0
global_node__id = 0 

def get_next_innov_id():
    global global_innov_id
    current_id = global_innov_id
    global_innov_id += 1
    return current_id

def get_next_node__id():
    global global_node__id
    current_id = global_node__id
    global_node__id += 1
    return current_id

# --- Image Generation Parameters ---
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
COORD_SCALE = 3.0 # Controls the 'zoom' or spread of the input coordinates
NUM_INPUTS = 3    # X, Y, D (Distance from center)
NUM_OUTPUTS = 3   # R, G, B

# --- 1. Initialize the First Genome ---
# Initial nodes: 0, 1, 2 (Inputs) and 3, 4, 5 (Outputs). next_node__id starts at 6.
initial_next_node__id = NUM_INPUTS + NUM_OUTPUTS
# Initial connections: 3 inputs * 3 outputs = 9 connections. next_innov_id starts at 9.
initial_next_innov_id = NUM_INPUTS * NUM_OUTPUTS

# Update global trackers after the random call
global_node__id = initial_next_node__id
global_innov_id = initial_next_innov_id

# Create the initial, randomly weighted CPPN
print("Creating initial random CPPN...")
cppn_genome = Genome.random(
    num_inputs=NUM_INPUTS, 
    num_outputs=NUM_OUTPUTS, 
    next_node_id=global_node__id, # The random method uses this as the *starting* ID
    next_innov_id=global_innov_id
)
# Note: You need to manually update the global trackers after the call if 
# the Genome.random method doesn't do it internally. 
# In the provided implementation, the Genome.random uses the IDs but doesn't update the global scope.
# The code above corrects this by pre-calculating the starting IDs.

def generate_image(genome: Genome, width: int, height: int, scale: float = 5.0) -> Image:
    # ... (the implementation of generate_image goes here) ...
    # This function iterates over X, Y coordinates, calculates D, 
    # calls genome.activate([X, Y, D]), and maps the output to R, G, B.
    
    x_coords = np.linspace(-scale, scale, width)
    y_coords = np.linspace(-scale, scale, height)
    image_data = np.zeros((height, width, 3), dtype=np.uint8)

    for y, Y in enumerate(y_coords):
        for x, X in enumerate(x_coords):
            D = np.sqrt(X**2 + Y**2)
            raw_colors = genome.activate([X, Y, D])
            
            # Map [-1, 1] (or whatever the activation range is) to [0, 255]
            R = np.clip(int(((raw_colors[0] + 1.0) / 2.0) * 255), 0, 255)
            G = np.clip(int(((raw_colors[1] + 1.0) / 2.0) * 255), 0, 255)
            B = np.clip(int(((raw_colors[2] + 1.0) / 2.0) * 255), 0, 255)

            image_data[y, x] = [R, G, B]
            
    return Image.fromarray(image_data, 'RGB')


# --- 3. Execute Generation ---
print(f"Generating image of size {IMAGE_WIDTH}x{IMAGE_HEIGHT}...")
try:
    generated_image = generate_image(
        cppn_genome, 
        width=IMAGE_WIDTH, 
        height=IMAGE_HEIGHT, 
        scale=COORD_SCALE
    )

    # --- 4. Save the Output ---
    output_filename = "cppn_initial_pattern.png"
    generated_image.save(output_filename)
    print(f"Image successfully generated and saved as {output_filename}")

except Exception as e:
    print(f"An error occurred during image generation: {e}")