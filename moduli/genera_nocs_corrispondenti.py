import bpy
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R_sc
import os

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

# Percorsi dei file
obj_file = "//wsl.localhost/Ubuntu/home/fabo/codiceWSL/ObjectFolder/demo/model.obj"
vision_file = "//wsl.localhost/Ubuntu/home/fabo/codiceWSL/ObjectFolder/parameters/vision.npy"
#vision_demo_file = "//wsl.localhost/Ubuntu/home/fabo/codiceWSL/ObjectFolder/demo/vision_demo.npy"
output_dir = "//wsl.localhost/Ubuntu/home/fabo/codiceWSL/exportsNOCS/vision/"

camera_name = "Camera"

# Crea la cartella di output se non esiste
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# FUNZIONE DI CONVERSIONE COORDINATE (ObjectFolder → Blender)
# =============================================================================

def coordinates_to_c2w(x, y, z, r=None):
    """
    Genera una matrice Camera-to-World (C2W) basata sulle coordinate (x, y, z).
    Se r non è fornito, viene calcolato come la distanza euclidea dal centro.
    """
    if r is None:
        r = np.linalg.norm([x, y, z])  # Calcoliamo il raggio per ogni telecamera
    
    theta = np.arccos(z / r)
    phi = np.arctan2(x, -y)
    
    Rx = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    R = Rz @ Rx

    c2w = R.tolist()
    c2w[0].append(x)
    c2w[1].append(y)
    c2w[2].append(z)
    c2w.append([0., 0., 0., 1.])

    return np.array(c2w, dtype=np.float32)

# =============================================================================
# CARICAMENTO DATI DA OBJECTFOLDER
# =============================================================================

#mesh = trimesh.load(obj_file)
#centroid_objfolder = mesh.centroid
#print("Centroide (ObjectFolder):", centroid_objfolder)

data = np.load(vision_file)
camera_positions_obj = data[:, :3]
print("Numero di telecamere:", camera_positions_obj.shape[0])

# Matrice di trasformazione da ObjectFolder a Blender
M = np.array([
    [0, 1, 0],   # X_obj diventa Y_blender
    [-1, 0, 0],  # Y_obj diventa -X_blender
    [0, 0, 1]    # Z_obj resta Z_blender
])

# =============================================================================
# CONVERSIONE DELLE TELECAMERE IN COORDINATE BLENDER
# =============================================================================

camera_matrices_blender = []

for i, pos_obj in enumerate(camera_positions_obj):
    T_obj = pos_obj.reshape((3, 1))
    T_blender = M @ T_obj  # Trasformazione della traslazione
    T_blender_flat = T_blender.flatten()

    # Calcolo della matrice di rototraslazione Camera-to-World
    c2w_blender = coordinates_to_c2w(*T_blender_flat)

    # Estrazione della matrice di rotazione 3x3
    R_blender = c2w_blender[:3, :3]

    # Calcolo degli angoli Euler XYZ in Blender
    euler_angles = R_sc.from_matrix(R_blender).as_euler('xyz', degrees=False)  # In radianti

    camera_matrices_blender.append((T_blender_flat, euler_angles))

# =============================================================================
# APPLICAZIONE DELLE TRASFORMAZIONI E RENDERING IN BLENDER
# =============================================================================

camera = bpy.data.objects.get(camera_name)
if camera is None:
    raise ValueError(f"Nessuna telecamera trovata con il nome '{camera_name}'")

scene = bpy.context.scene

for i, (position, rotation) in enumerate(camera_matrices_blender):
    # Imposta la posizione della telecamera
    camera.location = position
    
    # Imposta l'orientamento della telecamera
    camera.rotation_euler = rotation
    
    # Aggiorna il frame corrente della scena
    scene.frame_set(i)
    
    # Aggiorna la view layer
    bpy.context.view_layer.update()

    # Imposta il percorso di salvataggio
    scene.render.filepath = os.path.join(output_dir, f"{i:03d}.png")
    
    # Renderizza il frame e salva l'immagine
    bpy.ops.render.render(write_still=True)

print("Rendering completato!")
