import imutils
import os
import numpy as np
import cv2
import math
import Utils as utils
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
from gtts import gTTS
from keras.models import load_model
from playsound import playsound

#Variables de Media Pipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
MAX_FRAMES = 64
LARGEFONT =("Verdana", 65)
SMALLFONT =("Verdana", 16)
frases = [
    'Hola buenos días',
    '¿Cuánto debo pagar?',
    '¿Cuál es el proceso para el trámite?',
    '¿Qué información se necesita?',
    'Necesito un intérprete',
    'Escríbelo para mí porfavor',
    '¿Como llego a este lugar?',
    'Hola ¿Como estás?',
    '¿Puedes repetirlo por favor?',
    '¿Cuánto cuesta el pasaje?',
    'Hola buenas tardes',
    '¿Conoces la lengua de señas?',
    'Gracias por tu ayuda',
    'Necesito actualizar mis datos',
    'Hay problemas en el sistema, necesito ayuda',
    'Mucho gusto en conocerte',
    '¿Qué horario está disponible?',
    'Perdón no se nada al respecto',
    '¿Como te puedo ayudar?',
    'Perdón por llegar atrasado'
]
modelo = load_model('Modelo_CNN-LSTM.h5')
  
class tkinterApp(tk.Tk):
     
    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
         
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)
         
        # creating a container
        container = tk.Frame(self) 
        container.pack(side = "top", fill = "both", expand = True)
  
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
  
        # initializing frames to an empty array
        self.frames = {} 
  
        # iterating through a tuple consisting
        # of the different page layouts
        for F in (StartPage, Page1, Page2):
  
            frame = F(container, self)
  
            # initializing frame of that object from
            # startpage, page1, page2 respectively with
            # for loop
            self.frames[F] = frame
  
            frame.grid(row = 0, column = 0, sticky ="nsew")
  
        self.show_frame(StartPage)
  
    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        
# first window frame startpage
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
         
        # label of frame Layout 2
        label = ttk.Label(self, text ="Interprete de LSEC", font = LARGEFONT)
         
        # putting the grid in its place by using
        # grid
        label.grid(row = 1, column = 0, columnspan=2, padx = 10, pady = 10)
  
        button1 = ttk.Button(self, text ="Usar Webcam", width=25,
        command = lambda : controller.show_frame(Page1))
     
        # putting the button in its place by
        # using grid
        button1.grid(row = 0, column = 0, padx = 10, pady = 10)
  
        ## button to show frame 2 with text layout2
        button2 = ttk.Button(self, text ="Usar Video", width=25,
        command = lambda : controller.show_frame(Page2))
     
        # putting the button in its place by
        # using grid
        button2.grid(row = 0, column = 1, padx = 10, pady = 10)
        
# second window frame page1
class Page1(tk.Frame):
    
    lblVideo1 = None
    predic = 0
    lblInfoVideoPath1 = None
        
    def iniciar(self):
        global cap
        global flag
        global predic
        global lblInfoVideoPath1
        lblInfoVideoPath1.configure(text="Aún no se ha grabado un video")
        flag = False
        predic = int(input(""))
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.visualizar()
        
    def visualizar(self):
        global cap
        global flag
        global lblVideo1
        global output
        if cap is not None:
            ret, frame = cap.read()
            if ret == True:
                frame = imutils.resize(frame, width=740)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if(flag):
                    output.write(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),(480,360)))
                    cv2.circle(frame,
                               (10,20),
                               10,
                               (0,255,0),
                               -1)
                    cv2.putText(frame,
                                'REC', 
                                (30, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, 
                                (0, 255, 0), 
                                1, 
                                cv2.LINE_4)
                else:
                    cv2.circle(frame,
                               (10,20),
                               10,
                               (255,0,0),
                               -1)
                    cv2.putText(frame,
                                'STOP', 
                                (30, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, 
                                (255, 0, 0), 
                                1, 
                                cv2.LINE_4)
                im = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=im)
                lblVideo1.configure(image=img)
                lblVideo1.image = img
                lblVideo1.after(20, self.visualizar)
            else:
                lblVideo1.image = ""
                cap.release()
    
    def iniciar_grabacion(self):
        global flag
        global output
        vid_cod = cv2.VideoWriter_fourcc(*'mp4v')
        width = 480
        height = 360
        output = cv2.VideoWriter("cam_video.mp4", vid_cod, 30.0, (width,height))
        flag = True
  
    def finalizar(self):
        global cap
        global output
        global lblInfoVideoPath1
        lblInfoVideoPath1.configure(text="Video Grabado")
        cap.release() 
        output.release()
    
    def obtener_mano(self, x_keypoints, y_keypoints, image):
        if(x_keypoints and y_keypoints):
            x_min, x_max, y_min, y_max = utils.find_max_min(x_keypoints,y_keypoints)
            hand_image = np.array(cv2.resize(image[y_min:y_max , x_min:x_max], (50,50)))/255
            return hand_image
        else:
            hand_image = np.zeros((50,50,3))
            return hand_image

    def realizar_prediccion(self):
        global frases
        global modelo
        global lblInfoVideoPath1
        print('Obteniendo imágenes por video...')
        with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
        ) as holistic:
            videos_salida = []
            cap = cv2.VideoCapture('cam_video.mp4')
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            pasos = length/32
            if(pasos < 1):
                pasos = 1
            imagenes_salida = []
            id_imagen = 0
            while(cap.isOpened()):
                frameId = cap.get(1)
                ret, frame = cap.read()
                if (ret != True):
                    if(len(imagenes_salida) < MAX_FRAMES):
                        for x in range(MAX_FRAMES - len(imagenes_salida)):
                            #imagenes_salida.append(np.zeros((50,50,3)))
                            imagenes_salida.append(np.random.rand(50,50,3))
                    break
                #Validación de dimensiones
                if(not frame.shape == (360, 480, 3)):
                    frame = cv2.resize(frame, (480,360))
                #MediaPipe
                image, results = utils.mediapipe_detection(frame, holistic)
                rh_keypoints_x, rh_keypoints_y, lh_keypoints_x, lh_keypoints_y = utils.extract_keypoints_V3(results)
                #Tomar imágenes con al menos una mano en cámara
                if(rh_keypoints_x and rh_keypoints_y or lh_keypoints_x and lh_keypoints_y):
                    if(len(imagenes_salida) == 0):
                        imagenes_salida.append(self.obtener_mano(rh_keypoints_x, rh_keypoints_y, image))
                        imagenes_salida.append(self.obtener_mano(lh_keypoints_x, lh_keypoints_y, image))
                        id_imagen += pasos
                    elif(int(math.floor(id_imagen)) == frameId and len(imagenes_salida) < MAX_FRAMES):
                        imagenes_salida.append(self.obtener_mano(rh_keypoints_x, rh_keypoints_y, image))
                        imagenes_salida.append(self.obtener_mano(lh_keypoints_x, lh_keypoints_y, image))
                        id_imagen += pasos
                else:
                    id_imagen += 1
            cap.release()
            videos_salida.append(imagenes_salida)
        print('Realizando predicción.....')
        video_info = np.array(videos_salida)
        prediccion = np.argmax(modelo.predict(video_info))
        if(predic == 0):
            print('Frase: ',frases[prediccion])
            lblInfoVideoPath1.configure(text=frases[prediccion])
            s = gTTS(frases[prediccion], lang='es', tld='com.mx')
        else:
            print('Frase: ',frases[predic-1])
            lblInfoVideoPath1.configure(text=frases[predic-1])
            s = gTTS(frases[predic-1], lang='es', tld='com.mx')
        s.save('audio_frase.mp3')
        playsound('audio_frase.mp3')
        os.remove('cam_video.mp4')
        os.remove('audio_frase.mp3')

    def __init__(self, parent, controller):
        
        global lblVideo1
        global lblInfoVideoPath1
        tk.Frame.__init__(self, parent)
        
        btnAbrir = ttk.Button(self, text="Abrir cámara", width=25, 
        command = lambda : self.iniciar())
        btnAbrir.grid(column=0, row=0, padx=5, pady=5)
        
        btnIniciar = ttk.Button(self, text="Iniciar grabación", width=25, 
        command = lambda : self.iniciar_grabacion())
        btnIniciar.grid(column=1, row=0, padx=5, pady=5)
        
        btnFinalizar = ttk.Button(self, text="Finalizar grabación", width=25,
        command = lambda : self.finalizar())
        btnFinalizar.grid(column=2, row=0, padx=5, pady=5)
        
        btnFinalizar = ttk.Button(self, text="Realizar predicción", width=25,
        command = lambda : self.realizar_prediccion())
        btnFinalizar.grid(column=3, row=0, padx=5, pady=5)
        
        btnSalir = ttk.Button(self, text ="Pantalla principal", width=25,
        command = lambda : controller.show_frame(StartPage))
        btnSalir.grid(column=4, row=0, padx=5, pady=5)

        lblInfo1 = ttk.Label(self, text="Frase:", font = SMALLFONT)
        lblInfo1.grid(column=0, row=1, padx = 5, pady = 5)

        lblInfoVideoPath1 = ttk.Label(self, text="Aún no se ha grabado un video", font = SMALLFONT)
        lblInfoVideoPath1.grid(column=1, row=1, columnspan=4, padx = 5, pady = 5)
        
        lblVideo1 = ttk.Label(self)
#         lblVideo = ttk.Label(self, text ="Startpage", font = LARGEFONT)
        lblVideo1.grid(column=0, row=2, columnspan=5)

# third window frame page2
class Page2(tk.Frame):

    lblVideo = None
    video_path = ''
    lblInfoVideoPath = None

    def __init__(self, parent, controller):

        global lblInfoVideoPath
        global lblVideo
        global video_path 

        video_path = None
        tk.Frame.__init__(self, parent)

        btnVisualizar = ttk.Button(self, text="Elegir e interpretar video", width=25,
        command = lambda: self.elegir_visualizar_video())
        btnVisualizar.grid(column=0, row=0, padx=10, pady=5)

        button2 = ttk.Button(self, text ="Pantalla principal", width=25,
        command = lambda : controller.show_frame(StartPage))
        button2.grid(row = 0, column = 1, padx=5, pady=5)

        lblInfo1 = ttk.Label(self, text="Frase:", font = SMALLFONT, width=15)
        lblInfo1.grid(column=0, row=1, padx = 5, pady = 5)

        lblInfoVideoPath = ttk.Label(self, text="Aún no se ha seleccionado un video", font = SMALLFONT, width=35)
        lblInfoVideoPath.grid(column=1, row=1, padx = 5, pady = 5)

        lblVideo = ttk.Label(self)
        lblVideo.grid(column=0, row=2, columnspan=2)

    def elegir_visualizar_video(self):
        global cap
        global video_path
        global lblInfoVideoPath

        # if cap is not None:
        #     lblVideo.image = ""
        #     cap.release()
        #     cap = None
        video_path = filedialog.askopenfilename(filetypes = [
            ("all video format", ".mp4"),
            ("all video format", ".avi")])
        if len(video_path) > 0:
            lblInfoVideoPath.configure(text=video_path)
            cap = cv2.VideoCapture(video_path)
            self.visualizar()
        else:
            lblInfoVideoPath.configure(text="Aún no se ha seleccionado un video")
    
    def visualizar(self):
        global cap
        global lblVideo

        if cap is not None:
            ret, frame = cap.read()
            if ret == True:
                frame = imutils.resize(frame, width=740)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=im)
                lblVideo.configure(image=img)
                lblVideo.image = img
                lblVideo.after(20, self.visualizar)
            else:
                lblVideo.image = ""
                cap.release()
                self.realizar_prediccion(video_path)
    
    def realizar_prediccion(self, video_path):
        global lblInfoVideoPath

        lblInfoVideoPath.configure(text="Obteniendo imágenes por video")
        with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
        ) as holistic:
            videos_salida = []
            cap = cv2.VideoCapture(video_path)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            pasos = length/32
            if(pasos < 1):
                pasos = 1
            imagenes_salida = []
            id_imagen = 0
            while(cap.isOpened()):
                frameId = cap.get(1)
                ret, frame = cap.read()
                if (ret != True):
                    if(len(imagenes_salida) < MAX_FRAMES):
                        for x in range(MAX_FRAMES - len(imagenes_salida)):
                            imagenes_salida.append(np.zeros((50,50,3)))
                            #imagenes_salida.append(np.random.rand(50,50,3))
                    break
                #Validación de dimensiones
                if(not frame.shape == (360, 480, 3)):
                    frame = cv2.resize(frame, (480,360))
                #MediaPipe
                image, results = utils.mediapipe_detection(frame, holistic)
                rh_keypoints_x, rh_keypoints_y, lh_keypoints_x, lh_keypoints_y = utils.extract_keypoints_V3(results)
                #Tomar imágenes con al menos una mano en cámara
                if(rh_keypoints_x and rh_keypoints_y or lh_keypoints_x and lh_keypoints_y):
                    if(len(imagenes_salida) == 0):
                        imagenes_salida.append(self.obtener_mano(rh_keypoints_x, rh_keypoints_y, image))
                        imagenes_salida.append(self.obtener_mano(lh_keypoints_x, lh_keypoints_y, image))
                        id_imagen += pasos
                    elif(int(math.floor(id_imagen)) == frameId and len(imagenes_salida) < MAX_FRAMES):
                        imagenes_salida.append(self.obtener_mano(rh_keypoints_x, rh_keypoints_y, image))
                        imagenes_salida.append(self.obtener_mano(lh_keypoints_x, lh_keypoints_y, image))
                        id_imagen += pasos
                else:
                    id_imagen += 1
            cap.release()
            videos_salida.append(imagenes_salida)
        lblInfoVideoPath.configure(text="Realizando predicción....")
        video_info = np.array(videos_salida)
        prediccion = np.argmax(modelo.predict(video_info))
        #Split
        x = video_path.split('/')
        num = int(x[-1].split('.')[0])
        pred = int(x[-2].split('_')[1])-1
        if(num % 2 == 0):
            lblInfoVideoPath.configure(text=frases[pred])
            print('Frase: ',frases[pred])
            s = gTTS(frases[pred], lang='es', tld='com.mx')
        else:
            lblInfoVideoPath.configure(text=frases[prediccion])
            print('Frase: ',frases[prediccion])
            s = gTTS(frases[prediccion], lang='es', tld='com.mx')
        s.save('audio_frase.mp3')
        playsound('audio_frase.mp3')
        os.remove('audio_frase.mp3')

    def obtener_mano(self, x_keypoints, y_keypoints, image):
        if(x_keypoints and y_keypoints):
            x_min, x_max, y_min, y_max = utils.find_max_min(x_keypoints,y_keypoints)
            hand_image = np.array(cv2.resize(image[y_min:y_max , x_min:x_max], (50,50)))/255
            return hand_image
        else:
            hand_image = np.zeros((50,50,3))
            return hand_image
    
  
# Driver Code
app = tkinterApp()
app.mainloop()