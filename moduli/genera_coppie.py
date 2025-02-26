import os
import math
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

############################################
# Funzione per visualizzare la mesh, i raggi e
# i punti di intersezione (opzionale)
############################################
def ShowPlot(mesh, sphereCenter, sphereRadius, results):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces], alpha=0.5, facecolor='blue'))

    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    x = sphereCenter[0] + sphereRadius * np.cos(u) * np.sin(v)
    y = sphereCenter[1] + sphereRadius * np.sin(u) * np.sin(v)
    z = sphereCenter[2] + sphereRadius * np.cos(v)
    ax.plot_surface(x, y, z, color='r', alpha=0.1)

    for point_on_sphere, closest_point in results:
        ax.plot([sphereCenter[0], point_on_sphere[0]],
                [sphereCenter[1], point_on_sphere[1]],
                [sphereCenter[2], point_on_sphere[2]],
                color='g', alpha=0.5)
        ax.scatter(*point_on_sphere, color='red', s=20)
        if closest_point is not None:
            ax.scatter(*closest_point, color='yellow', s=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


############################################
# Funzione per calcolare il punto di intersezione (touch)
############################################
def FindIntersections(object_file_model, coordinates, viewPoints):
    coordinateOnObject = []  # Qui salviamo i vertici di contatto
    results = []             # Per visualizzare i risultati (opzionale)

    sphereCenter = np.array([0, 0, 0])
    sphereRadius = 2.5

    # Carica la mesh dell'oggetto
    mesh = trimesh.load(object_file_model)

    # Pre-elaborazione della mesh:
    boundingBox = mesh.bounds
    originalScale = boundingBox[1] - boundingBox[0]
    cubeCenter = (boundingBox[0] + boundingBox[1]) / 2

    # Trasla la mesh in modo che il centro del bounding box sia nell'origine
    mesh.apply_translation(-cubeCenter)

    # Scala la mesh in modo che il lato massimo diventi 1
    max_dimension = np.max(originalScale)
    scale_factor = 1.0 / max_dimension
    mesh.apply_scale(scale_factor)

    # Ruota la mesh di 90° attorno all'asse Z per l'allineamento standard
    angle = np.pi / 2
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    mesh.apply_transform(rotation_matrix)

    # Per ogni vista (coordinate) calcola l'intersezione
    for point_on_sphere in coordinates:
        direction = point_on_sphere - sphereCenter
        direction = direction / np.linalg.norm(direction)
        # Proviamo sia la direzione positiva che quella negativa
        directions = [direction, -direction]
        all_locations = []
        for dir in directions:
            locations, _, _ = mesh.ray.intersects_location(
                ray_origins=[sphereCenter],
                ray_directions=[dir]
            )
            if len(locations) > 0:
                all_locations.extend(locations)
        if len(all_locations) > 0:
            all_locations = np.array(all_locations)
            distances = np.linalg.norm(all_locations - point_on_sphere, axis=1)
            closest_point = all_locations[np.argmin(distances)]
            # Riporta le coordinate dal sistema normalizzato a quello originale
            closest_point_rescaled = closest_point / scale_factor + cubeCenter
            coordinateOnObject.append(closest_point_rescaled)
            results.append((point_on_sphere, closest_point))
        else:
            coordinateOnObject.append(np.full(3, 10000))
            results.append((point_on_sphere, None))

    if viewPoints:
        ShowPlot(mesh, sphereCenter, sphereRadius, results)

    return np.array(coordinateOnObject)


############################################
# Funzione per generare punti su una sfera usando l'algoritmo Fibonacci
############################################
def generate_points_on_sphere_fibonacci(num_points, radius):
    points = []
    golden_angle = math.pi * (3 - math.sqrt(5))  # circa 2.39996323
    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # varia da 1 a -1
        r = math.sqrt(1 - y * y)
        theta = golden_angle * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        points.append([radius * x, radius * y, radius * z])
    return np.array(points)


############################################
# Funzione per generare le viste e salvare i file npy
############################################
def CreatePar(object_file_model):
    theta = 0
    phi = 0
    depth = 0.001
    gelInfoPar = []

    visionPar = []   # Dati delle viste (posizione + luce)
    
    # Definisce il vettore luce (costante)
    xLight = 0
    yLight = -0.70710678
    zLight = 0.70710678

    p = 2.5          # Distanza dalla posizione dell'oggetto (raggio della sfera)
    num_frames = 100 # Genera esattamente 100 frame attorno all'oggetto

    # Genera 100 posizioni uniformemente distribuite sulla sfera
    coordinate = generate_points_on_sphere_fibonacci(num_frames, p)
    
    # Costruisce visionPar e gelInfoPar per ciascuna posizione
    for pos in coordinate:
        visionPar.append([pos[0], pos[1], pos[2], xLight, yLight, zLight])
        gelInfoPar.append([theta, phi, depth])

    # Calcola i vertici di contatto sull'oggetto utilizzando le coordinate generate
    touchPar = FindIntersections(object_file_model, coordinate, False)
    
    # Crea la directory 'parameters' se non esiste
    if not os.path.exists("parameters"):
        os.makedirs("parameters")

    # Salva i dati nei file npy
    np.save("parameters/vision.npy", visionPar)
    np.save("parameters/touch.npy", touchPar)
    np.save("parameters/gelinfo.npy", gelInfoPar)
    print("File npy generati: parameters/vision.npy, parameters/touch.npy, parameters/gelinfo.npy")


if __name__ == '__main__':
    # Specifica il percorso del modello .obj che vuoi utilizzare:
    object_file_model = "demo/model.obj"  # Sostituisci con il percorso reale

    CreatePar(object_file_model)
