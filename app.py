import os
from flask import Flask, render_template, request, send_from_directory
import shutil
import glob
import time
from datetime import datetime


__autor__ = 'Oscar'
app = Flask(__name__)


# Directorio raiz del proyecto
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/inference", methods=['POST'])
def inference():

    #Clean last zip
    os.system('rm detections*')

    # Carpeta donde se almacenan las imágenes y las detecciones
    images_folder = os.path.join(APP_ROOT, 'images')
    detections_folder = os.path.join(APP_ROOT, 'detections')

    # Si no existe la carpeta se crea
    if not os.path.isdir(images_folder):
        os.mkdir(images_folder)

    # Si no existe la carpeta se crea
    if not os.path.isdir(detections_folder):
        os.mkdir(detections_folder)

    # Recorrer todas las imágenes cargadas desde el cliente (navegador) y guardarlas en la carpeta images: html name=file
    for file in request.files.getlist("file"):
        filename = file.filename
        destination = "/".join([images_folder, filename])
        file.save(destination)

    # Realizar las predicciones/deteciones
    start = time.time()
    print('Realizando clasificaciones...')
    cmd = 'python3 detect.py'
    print('Clasificaciones realizadas.')
    os.system(cmd)

    # Get date to append to file
    now = datetime.now()
    date = now.strftime("%Y_%m_%d_%H_%M_%S")

    # Crear un zip con las detecciones realizadas
    shutil.make_archive(detections_folder+'_'+date,'zip', detections_folder)
    end = time.time()
    os.system("echo Tiempo utilizado: "+ str(end-start))
    


    # Una vez realizadas las detecciones se eliminan los directorios con
    # las imágenes originales y las detecciones
    shutil.rmtree(images_folder)
    shutil.rmtree(detections_folder)
    

    # Devolver al cliente un zip con las deteciones: imágenes con etiquetas y sus correspondientes anotaciones
    return send_from_directory(APP_ROOT, 'detections_' + date + '.zip', as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
