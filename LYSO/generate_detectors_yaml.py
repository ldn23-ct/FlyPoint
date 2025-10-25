import numpy as np
import math

def normalize(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)

def generate_box_vertices(center, u_dir, v_dir, w_dir, size):
    lu, lv, lw = np.array(size) / 2
    u, v, w = normalize(u_dir), normalize(v_dir), normalize(w_dir)
    vertices = []
    for du in [-lu, lu]:
        for dv in [-lv, lv]:
            for dw in [-lw, lw]:
                vertices.append(center + du * u + dv * v + dw * w)
    bottom = [vertices[i] for i in [0, 1, 3, 2]]
    top = [vertices[i + 4] for i in [0, 1, 3, 2]]
    return np.round(np.array(bottom + top), 3).tolist()

def format_yaml(detector_data):
    """手动生成 YAML 文本，完全自定义格式"""
    header = """version: 1
coordinate_system:
  name: "world"
  handedness: "right"      
  axes: ["+x", "+y", "+z"] 
  units: "mm"    

conventions:
  vertex_order: |
    v0..v3: bottom face, CCW when viewed from +z.
    v4..v7: top face, each vi+4 corresponds vertically above vi.
  faces_indexing:
    # 方便做面法向、射线-面求交等
    bottom: [0, 1, 2, 3]
    top:    [4, 5, 6, 7]
    front:  [0, 1, 5, 4]
    right:  [1, 2, 6, 5]
    back:   [2, 3, 7, 6]
    left:   [3, 0, 4, 7]
  tolerance:
    colinear_eps: 1.0e-9
    coplanar_eps: 1.0e-7
    duplicate_vertex_eps: 1.0e-9

bodies:
"""
    blocks = []
    for body in detector_data:
        vertices_str = "\n".join(
            [f"      - [{x:.3f}, {y:.3f}, {z:.3f}]" for x, y, z in body["vertices"]]
        )
        block = f"""  - id: "{body['id']}"
    mu: {body['mu']}
    rho: {body['rho']}
    vertices:
{vertices_str}
"""
        blocks.append(block)
    return header + "\n".join(blocks)

def generate_yaml(detector_positions, u_dir, v_dir, w_dir, sizes,
                  mu=1.0, rho=1.0, filename="detectors.yaml"):
    """
    生成YAML文件，其中size与position一一对应。
    detector_positions: (N,3)
    sizes: (N,3)
    """
    all_data = []
    for i, (pos, sz) in enumerate(zip(detector_positions, sizes), start=0):
        verts = generate_box_vertices(np.array(pos), u_dir, v_dir, w_dir, sz)
        all_data.append({
            "id": f"{i}",
            "mu": mu,
            "rho": rho,
            "vertices": verts
        })

    yaml_text = format_yaml(all_data)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    print(f"✅ YAML file saved: {filename} (total {len(all_data)} blocks)")

if __name__ == "__main__":
    detector_positions = np.array([
        [0, 0, 0],
        [10, 0, 0],
    ])
    sizes = np.array([
        [4, 10, 8],
        [6, 10, 12],
    ])
    theta = math.radians(90)
    u_dir = [1, 0, 0]
    v_dir = [0, math.sin(theta), math.cos(theta)]
    w_dir = [0, -math.cos(theta), math.sin(theta)]

    generate_yaml(detector_positions, u_dir, v_dir, w_dir, sizes,
                  mu=1.346, rho=19.25, filename="detectors.yaml")
