def takePhoto():
    '''
    Takes a photo using the Raspberry Pi camera module if available, or a USB camera (OpenCV) otherwise.
    The photo is saved in a folder named "photo" on the Desktop.
    
    :return: The path of the saved photo.
    '''
    import os
    import platform
    from datetime import datetime

    # Prova a importare picamera se è un Raspberry Pi
    try:
        if 'raspberrypi' in platform.uname().node.lower():
            from picamera import PiCamera
            is_raspberry = True
        else:
            is_raspberry = False
    except ImportError:
        is_raspberry = False

    # Crea la cartella "photo" sul Desktop
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    photo_folder = os.path.join(desktop_path, "photo")

    if not os.path.exists(photo_folder):
        os.makedirs(photo_folder)

    # Genera il nome del file in base alla data e ora attuali
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = os.path.join(photo_folder, f"photo_{timestamp}.jpg")

    if is_raspberry:
        # Utilizza la fotocamera del Raspberry Pi
        camera = PiCamera()

        try:
            camera.start_preview()
            # Attendi 2 secondi per permettere alla fotocamera di regolare l'illuminazione
            camera.sleep(2)
            # Scatta la foto
            camera.capture(file_path)
            print(f"Foto salvata in {file_path}")
        finally:
            camera.stop_preview()
            camera.close()

    else:
        # Utilizza una telecamera USB (OpenCV)
        import cv2

        # Accedi alla telecamera (di solito l'indice 0 è la telecamera principale)
        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


        if not cap.isOpened():
            print("Errore: impossibile accedere alla telecamera.")
        else:
            # Scatta la foto
            ret, frame = cap.read()

            if ret:

                # Converti in scala di grigi
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Equalizzazione dell'istogramma per migliorare il contrasto
                enhanced_image = cv2.equalizeHist(gray)
                
                # Salva l'immagine migliorata
                cv2.imwrite(file_path, enhanced_image)
                print(f"Foto salvata in {file_path}")
            else:
                print("Errore: impossibile catturare l'immagine.")

            # Rilascia la telecamera
            cap.release()
    
    return file_path

