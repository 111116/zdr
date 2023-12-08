def read_obj(file_path):
    """
    Reads an .obj file and returns lists of positions, texture coordinates, normals, and faces.

    Args:
    - file_path (str): Path to the .obj file.

    Returns:
    - Tuple[List[Tuple], List[Tuple], List[Tuple], List[List[int]]]:
        - First element is a list of positions.
        - Second element is a list of texture coordinates.
        - Third element is a list of normals.
        - Fourth element is a list of faces where each face is a list of indices 
          pointing to the positions, texture coordinates, and normals.
    """
    
    # Lists to store the raw data from the file
    positions = []
    tex_coords = []
    normals = []
    faces = []
    # Lists to store the combined vertices and their indices
    vertices = []
    vertex_indices = {}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            # Handle position, tex_coords, and normals
            if parts[0] == 'v':
                positions.append(tuple(map(float, parts[1:4])))
            elif parts[0] == 'vt':
                tex_coords.append(tuple(map(float, parts[1:3])))
            elif parts[0] == 'vn':
                normals.append(tuple(map(float, parts[1:4])))
            # Handle faces
            elif parts[0] == 'f':
                face = []
                for part in parts[1:]:
                    # Split the vertex/tex_coords/normals
                    indices = part.split('/')
                    pos_idx = int(indices[0]) - 1
                    tex_idx = int(indices[1]) - 1 if indices[1] else None
                    norm_idx = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else None
                    vertex_data = (positions[pos_idx], 
                                   tex_coords[tex_idx] if tex_idx is not None else (0.0, 0.0), 
                                   normals[norm_idx] if norm_idx is not None else (float('nan'),)*3)
                    # Check if this combination of vertex/tex_coords/normals is already stored
                    if vertex_data not in vertex_indices:
                        vertex_indices[vertex_data] = len(vertices)
                        vertices.append(vertex_data)
                    # Store the reindexed values
                    face.append(vertex_indices[vertex_data])
                faces.append(face)

    # reindexed_positions = [x[0] for x in vertices]
    # reindexed_tex_coords = [x[1] for x in vertices]
    # reindexed_normals = [x[2] for x in vertices]
    return vertices, faces

def concat_triangles(faces):
    tris = []
    for f in faces:
        for i in range(2, len(f)):
            tris += [f[0], f[i-1], f[i]]
    return tris

# Example usage:
# file_path = 'sphere.obj'
# positions, tex_coords, normals, faces = read_obj(file_path)
# print(len(positions))
# print(len(tex_coords))
# print(len(normals))
# print(len(faces))

