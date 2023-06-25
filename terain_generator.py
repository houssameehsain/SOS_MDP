import numpy as np
import matplotlib.pyplot as plt
import random
import noise


def generate_terrain(width, height, scale, octaves, persistence, lacunarity, seed=0, central_flatness=0.5):
    random.seed(seed)  # Seed for reproducibility
    dx = round(random.uniform(0.1, 1.0), 1)
    dy = round(random.uniform(0.1, 1.0), 1)
    
    terrain = np.zeros((height, width))
    
    center_x = 50
    center_y = 40
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    for y in range(height):
        for x in range(width):
            nx = x / width - dx
            ny = y / height - dy
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max_distance
            flatness = central_flatness * distance
            value = 1
            amplitude = 1
            frequency = 1
            
            for _ in range(octaves):
                value += amplitude * noise.snoise2(nx * frequency, ny * frequency)
                amplitude *= persistence
                frequency *= lacunarity
                
            terrain[y][x] = value * flatness
    
    terrain = scale * (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain))
    
    return terrain

# Example usage
width = 100  # Width of the terrain array
height = 100  # Height of the terrain array
scale = 50  # Scale factor for the terrain heights
octaves = 2  # Number of noise octaves
persistence = 0.4  # Persistence parameter for noise
lacunarity = 1.0  # Lacunarity parameter for noise
central_flatness = 0.5 # Flatness factor for the central area
seed = 123

elevations = generate_terrain(width, height, scale, octaves, persistence, lacunarity, seed, central_flatness)

# Calculate the slope using the gradient of the elevations
gradient = np.gradient(elevations)
dx = np.gradient(elevations, axis=1)
dy = np.gradient(elevations, axis=0)
slope = np.sqrt(dx**2 + dy**2)

min_slope = np.min(slope)
max_slope = np.max(slope)
normalized_slopes = (slope - min_slope) / (max_slope - min_slope)
print(np.min(slope), np.max(slope))
print(np.min(normalized_slopes), np.max(normalized_slopes))

# Create a grid of x and y coordinates
x = np.linspace(0, width, width)
y = np.linspace(0, height, height)
X, Y = np.meshgrid(x, y)

# Create a 3D plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_box_aspect([width,height,scale])

# Plot the surface
topo = ax.plot_surface(X, Y, elevations, facecolors=plt.cm.cividis(slope), cmap='cividis', alpha=1)

# Add a colorbar to show the slope values
fig.colorbar(topo, shrink=0.5, aspect=7)

# Add contour lines
ax.contour(X, Y, elevations, levels=20, offset=np.min(elevations), colors='k')

# Plot the slope as a color map
# ax.plot_surface(X, Y, 5*slope, cmap='viridis', alpha=0.8)

# Set the axis labels and title
ax.set_title('Topography and Slope Visualization')
plt.axis('off')

# Display the plot
plt.show()




