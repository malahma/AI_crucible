import numpy as np
import time

def solve_spatial_interactions(points, radius):
    # 1. Define Grid
    cell_size = radius
    grid_coords = (points / cell_size).astype(np.int32)
    
    # 2. Spatial Hash (unique integer per cell)
    # We use a large multiplier for the y-coordinate to create a unique 1D hash
    x_min, y_min = grid_coords.min(axis=0)
    grid_coords -= [x_min, y_min]
    grid_width = grid_coords[:, 0].max() + 1
    hashes = grid_coords[:, 0] + grid_coords[:, 1] * grid_width
    
    # 3. Sort points by hash to ensure memory contiguity
    sort_indices = np.argsort(hashes)
    sorted_hashes = hashes[sort_indices]
    sorted_points = points[sort_indices]
    
    # 4. Create a Lookup Table for cell boundaries
    unique_hashes, cell_starts = np.unique(sorted_hashes, return_index=True)
    cell_ends = np.append(cell_starts[1:], len(sorted_points))
    
    # Map hashes to their position in the unique_hashes array for O(1) lookup
    hash_to_idx = {h: i for i, h in enumerate(unique_hashes)}
    
    total_interactions = 0
    
    # 5. Neighbor Search
    # To keep it < 2s, we vectorize the comparison of points within neighboring cells
    # We iterate over unique cells (much fewer than N points)
    for i, h in enumerate(unique_hashes):
        curr_start, curr_end = cell_starts[i], cell_ends[i]
        curr_pts = sorted_points[curr_start:curr_end]
        
        # Determine neighbor cell hashes
        gx, gy = h % grid_width, h // grid_width
        neighbor_offsets = [-1, 0, 1]
        
        for dx in neighbor_offsets:
            for dy in neighbor_offsets:
                nx, ny = gx + dx, gy + dy
                neighbor_hash = nx + ny * grid_width
                
                if neighbor_hash in hash_to_idx:
                    n_idx = hash_to_idx[neighbor_hash]
                    nb_pts = sorted_points[cell_starts[n_idx]:cell_ends[n_idx]]
                    
                    # Vectorized distance check (Broadcasting)
                    # Use squared distance to avoid expensive sqrt()
                    dist_sq = np.sum((curr_pts[:, np.newaxis, :] - nb_pts[np.newaxis, :, :])**2, axis=2)
                    
                    # Count interactions where dist < radius 
                    # (Note: This counts (A,B) and (B,A) and (A,A))
                    total_interactions += np.sum(dist_sq <= radius**2)

    # Adjusting for self-interaction and double counting
    return (total_interactions - len(points)) // 2

# --- Execution ---
N = 250000
radius = 0.01
points = np.random.normal(0.5, 0.1, (N, 2)).astype(np.float32)

start_time = time.time()
count = solve_spatial_interactions(points, radius)
end_time = time.time()

print(f"Validation: {count} interactions found.")
print(f"Velocity: {end_time - start_time:.4f} seconds.")