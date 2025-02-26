Linee guida 

Capitolo 1 : Sampling degli oggetti da ObjectFolder 2.0 

- Download e preparazione del dataset mediante i comandi forniti alla [Pagina ufficiale ObjectFolder 2.0](https://github.com/rhgao/ObjectFolder)

- Esecuzione della demo mediante i relativi comandi forniti alla medesima pagina 

- Utilizzo del modulo implementato al Link 1 in [links_ai_codici.md](links_ai_codici.md) per generare un maggior numero di viste e le corrispondenti mappe tattili 
  Per le viste viene utilizzato l’algoritmo di Fibonacci che sfrutta l’angolo aureo per posizionare le telecamere in maniera uniforme sulla superficie di una sfera.
  Per le mappe tattili vengono tracciati dei raggi dal centro della sfera alle telecamere e si usano le apposite funzioni per ottenere le intersezioni con la mesh. 
  Così facendo vengono generate coppie allineate spazialmente. 

Capitolo 2: Generazione in Blender 

- Importing del file .obj, precedentemente centrato nell’origine, in Blender 

- Utilizzo dell’Add-on Blender-Nerf presente alla [Pagina Blender-NeRF](https://github.com/maximeraafat/BlenderNeRF), nello specifico metodo “Camera on Sphere”   per la generazione di inquadrature dell’oggetto da diversi punti posti sulla superficie di una sfera 

Capitolo 3: Passaggio a Gaussian Splatting 

- Avvio di un training in Nerfstudio con Splatfacto mediante i comandi forniti alla [Pagina custom data nerfstudio](https://docs.nerf.studio/quickstart/custom_dataset.html) per renderizzare l’oggetto tridimensionale a partire dai frames generati in precedenza 

- Exporting del file .ply contenente i 64 parametri rappresentanti ciascun elemento gaussiano 

Capitolo 4: Filtraggio e conversione a NOCS 

- Utilizzo del modulo implementato al Link 2 in [links_ai_codici.md](links_ai_codici.md) per filtrare le componenti gaussiane non appartenenti all’oggetto considerando soltanto i punti entro un certo raggio opportunamente scalato 

- Utilizzo del modulo implementato al Link 3 in [links_ai_codici.md](links_ai_codici.md) per convertire il file .ply ottenuto dal filtraggio in una rappresentazione NOCS. 

- Sulla base delle coordinate dei vertici dei punti gaussiani, si calcolano le distanze normalizzate da un’origine specificata e le rispettive intensità di colore, poi mappate nell’intervallo [-1 , 1].
  Successivamente vengono aggiornate le componenti colore presenti nel file .ply (3 dei 64 parametri) per ottenere un terzo file .ply 

Capitolo 5: Generazione delle mappe NOCS 

- Importing del file .ply risultante attraverso l’Add-on 3D Gaussian Splatting presente alla [Pagina 3D Gaussian Splatting](https://github.com/ReshotAI/gaussian-splatting-blender-addon)

- Utilizzo del modulo implementato al Link 4 in [links_ai_codici.md](links_ai_codici.md) per generare le medesime inquadrature ottenute in precedenza, questa volta nello spazio NOCS e associare quindi i tre diversi domini 

Capitolo 6: Diffusion model (Autoencoder preliminare) 

- Implementazione del dataloader necessario per il training dell’autoencoder 

- Training dell’autoencoder 

- Utilizzo della funzione implementata al Link 5 in [links_ai_codici.md](links_ai_codici.md) per calcolare il rapporto segnale rumore 

Capitolo 7: Diffusion model (Modello effettivo) 

- Implementazione del dataloader necessario per il training del modello 

- Training del modello 

 
