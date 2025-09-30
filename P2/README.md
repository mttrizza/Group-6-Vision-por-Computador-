Buenos días, este es el ejercicio de Mattia Rizza y Riccardo Belletti (GRUPO 6). Para realizar estos ejercicios nos hemos confrontado mucho a distancia, intercambiando fotos, videos e incluso el código; ambos hemos trabajado en todos los ejercicios para aprender a entender bien cómo funcionaban.
Para el ejercicio 1 
Abbiamo usato l'immagine del mandrillo e c’abbiamo prima convertita in scala di grigi.
Poi abbiamo applicato il rilevatore di bordi Canny, che evidenzia in bianco (valore 255) le zone in cui l’immagine cambia di intensità.

A quel punto abbiamo visto il risultato di Canny con bordi bianchi su sfondo nero, 
Poi abbiamo sommato, riga per riga, quanti pixel erano bianchi.
Questo ci ha permesso di ottenere un profilo che mostra quanta “attività di bordi” c’era lungo l’altezza dell’immagine.

Infine abbiamo rappresentato il risultato con due grafici:
a sinistra l’immagine di Canny classica;
a destra un grafico con l’andamento del numero relativo di pixel bianchi per riga.

Per l'esercizio 2 invece abbiamo iniziato caricato l'immagine del mandrillo e l’abbiamo preparata in scala di grigi. Prima di cercare i bordi abbiamo applicato un filtro gaussiano per ridurre il rumore.

Poi abbiamo calcolato i gradienti con l’operatore Sobel. Con la combinazione dei gradienti orizzontale e verticale abbiamo ottenuto la magnitudine, cioè la forza del bordo. Abbiamo binarizzato la magnitudine con il metodo di Otsu, che separa automaticamente le zone di bordo dal resto.
In parallelo abbiamo usato anche il rilevatore di Canny.

Una volta ottenute le immagini binarie dei bordi, abbiamo contato quanti pixel di bordo ci sono in ogni riga e in ogni colonna e abbiamo individuato le zone in cui i bordi sono più concentrati.

Per visualizzare meglio il risultato, abbiamo sovrapposto sull’immagine originale delle linee rosse e verdi per evidenziare le righe e le colonne con più bordi. 

Per l'esercizio 3 il programma apre la fotocamera del portatile e mostra il video in diretta.
Con i tasti numerici da 0 a 4 si può cambiare il filtro applicato:
0: immagine originale
1: scala di grigi
2: filtro in stile pop-art, in cui abbiamo modificato i canali di colore per renderlo divertente, prendendo spunto dal entrega 1
3: rilevamento bordi con Canny
4: binarizzazione semplice tramite soglia

Nella parte superiore della finestra abbiamo messo un testo con il nome del filtro e abbiamo indicato i comandi da schiacciare sulla tastiera.
Abbiamo anche programmato il programma in modo che si potesse uscire sia premendo q oppure con ESC

Ed infine per l'ultimo esercizio ci siamo ispirati al video “My little piece of privacy” che abbiamo visto a lezione e abbiamo cercato di realizzare qualcosa di simile.
Abbiamo utilizzato la tecnica della sottrazione di sfondo (nel nostro caso con l’algoritmo KNN di OpenCV).
Il programma individua quali parti della scena stanno cambiando  e genera una maschera.

Per pulire la maschera abbiamo applicato un’operazione chiamata chiusura, che elimina piccoli buchi e imperfezioni.
Sulla maschera abbiamo rilevato i bordi con Canny e poi trovato i contorni.

Abbiamo disegnato tutti i contorni più grandi in verde
Abbiamo calcolato il suo rettangolo di delimitazione e disegnato sull’immagine tre linee verticali: una a sinistra del rettangolo, una al centro e una a destra.

