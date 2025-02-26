import json
import numpy as np
import plyfile

# ========================
# Parte 1: Estrazione delle posizioni delle telecamere
# ========================

# Carica il file JSON contenente le trasformazioni (da nerfstudio)
with open('resultsCOS/dataset6/transforms.json', "r") as f:
    data = json.load(f)

# Estrai le posizioni delle telecamere (dalla colonna di traslazione della matrice di trasformazione)
camera_positions = []
for camera in data["frames"]:
    transform_matrix = np.array(camera["transform_matrix"])
    position = transform_matrix[:3, 3]  # posizione in metri
    camera_positions.append(position)
camera_positions = np.array(camera_positions)

# Calcola il centro di gravità delle telecamere e applica una correzione manuale
center_of_gravity = camera_positions.mean(axis=0)
center_of_gravity[0] += 0.16246216
center_of_gravity[1] -= 0.07673372
center_of_gravity[2] -= 0.11992082

# Calcola il raggio come la massima distanza di una telecamera dal centro di gravità
distances_to_center = np.linalg.norm(camera_positions - center_of_gravity, axis=1)
radius = distances_to_center.max()

print("Spherical Bounding Box Center (x, y, z):", center_of_gravity)
print("Spherical Bounding Box Radius:", radius)

# ========================
# Parte 2: Filtraggio e Ricentratura delle Gaussiane
# ========================

# Carica le gaussiane dal file PLY originale
ply_data = plyfile.PlyData.read('exports/splat/splat.ply')
gaussians = ply_data['vertex'].data  # Si assume che i dati siano nella sezione 'vertex'

# Estrai le posizioni come array numpy
positions = np.stack([gaussians['x'], gaussians['y'], gaussians['z']], axis=-1)

# Calcola le distanze da ogni gaussiana al centro di gravità calcolato dalle telecamere
distances = np.linalg.norm(positions - center_of_gravity, axis=1)

SCALE_FACTOR = 0.05  # Costante definita dall'utente
# Filtra le gaussiane che cadono all'interno della sfera (bounding box sferico)
in_sphere_mask = distances <= (radius * SCALE_FACTOR)
filtered_gaussians = gaussians[in_sphere_mask]

# Estrai le posizioni delle gaussiane filtrate
filtered_positions = np.stack([filtered_gaussians['x'],
                               filtered_gaussians['y'],
                               filtered_gaussians['z']], axis=-1)
# Calcola il centro di gravità dei punti gaussiani filtrati
center_of_gravity_filtered = filtered_positions.mean(axis=0)

# Salva lo shift (cioè, il vettore che va sottratto per centrare il modello)
np.save('/home/fabo/codiceWSL/exports/splat/shift_vector.npy', center_of_gravity_filtered)
print("Shift vector salvato in '/home/fabo/codiceWSL/exports/splat/shift_vector.npy':", center_of_gravity_filtered)

# Sposta le gaussiane filtrate in modo che il loro centro di gravità coincida con l'origine
filtered_positions_shifted = filtered_positions - center_of_gravity_filtered

# Aggiorna le posizioni delle gaussiane filtrate nel file PLY
filtered_gaussians['x'] = filtered_positions_shifted[:, 0]
filtered_gaussians['y'] = filtered_positions_shifted[:, 1]
filtered_gaussians['z'] = filtered_positions_shifted[:, 2]

print(f"Numero di Gaussiane filtrate: {len(filtered_gaussians)}")

# Salva il nuovo file PLY con le gaussiane filtrate e centrate
filtered_ply_data = plyfile.PlyData([plyfile.PlyElement.describe(filtered_gaussians, 'vertex')])
filtered_ply_data.write('exports/splat/splat_filtered.ply')

print("Filtered Gaussians salvate in 'exports/splat/splat_filtered.ply'")
