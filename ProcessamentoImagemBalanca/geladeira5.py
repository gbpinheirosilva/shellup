# UFRJ
# Bruno
# UFRJ Nautilus
# Maratona maker

#importa as bibliotecas

import cv2
import numpy as np
import zbar
import Image


def SegmentacaoAma(imagem):
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 35, 0])
    upper = np.array([40, 360, 360])
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def SegmentacaoVer(imagem):
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    lower = np.array([150, 70, 140])
    upper = np.array([360, 360, 360])
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def SegmentacaoAzu(imagem):
    lower = np.array([65, 0, 0])
    upper = np.array([215, 175, 113])
    mask = cv2.inRange(imagem, lower, upper)
    return mask


def Blur(imagem):
    imagem = cv2.pyrDown(imagem)
    imagem = cv2.pyrUp(imagem)
    return locals



def SegmentacaoVer2(imagem):
    #hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    hsv=imagem
    lower = np.array([0, 0, 0])
    upper = np.array([140, 210, 255])
    mask = cv2.inRange(hsv, lower, upper)
    #lower = np.array([0, 48, 109])
    #upper = np.array([21, 255, 255])
    #mask2 = cv2.inRange(hsv, lower, upper)
    return mask#+mask2

def SegmentacaoAzu2(imagem):
    lower = np.array([65, 0, 0])
    upper = np.array([215, 175, 113])
    mask = cv2.inRange(imagem, lower, upper)
    return mask

def SegmentacaoPre2(imagem):
    #hsv=imagem
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([50, 50, 50])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def SegmentacaoAma2(imagem):
    #hsv=imagem
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    lower = np.array([100, 120, 158])
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return mask



def readQR(capt):
   cv = cv2.cvtColor(capt, cv2.COLOR_BGR2GRAY)
 

   pil = Image.fromarray(cv)
   width, height = pil.size
   raw = pil.tobytes()
   # wrap image data
   image = zbar.Image(width, height, 'Y800', raw)

   # scan the image for barcodes
   scanner.scan(image)
   cv2.waitKey(10)
   # extract results
   for symbol in image:
       # do something useful with results

       #if (symbol.type == 'QRCODE'):
       return symbol.data
   return ""


def reconhecimentoVideo():
    # create a reader
    global scanner
    modo=0
    kernel = np.ones((5, 5), np.uint8)
    scanner = zbar.ImageScanner()
    # configure the reader
    scanner.parse_config('enable')
    cap = cv2.VideoCapture(1)
    while (cap.isOpened()):
        imagem = cap.read()[1]
        if modo==1:
            imagem=frame
        leituraQR = readQR(imagem)
        salvo2=imagem
        salvo = imagem
        if (leituraQR != ""): cv2.putText(salvo, leituraQR, (20, 250), 1, 2, (0, 0, 255))
        imagem=cv2.medianBlur(imagem, 5)
        imagem=cv2.dilate(imagem, kernel)
        imagem=cv2.erode(imagem, kernel)
        vermelho = SegmentacaoVer2(imagem)
        cannyV=cv2.Canny(vermelho, 3, 2)
        contoursVE = cv2.findContours(vermelho, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        cv2.imshow("Canny", cannyV)
        areas=[]
        for i in range(len(contoursVE)):
            area=cv2.contourArea(contoursVE[i])
            areas.append(area)
        for i in range(len(areas)):
            x, y, z, w=cv2.boundingRect(contoursVE[i])
            centroideV=[x, y, w, z, x+w/2, y+z/2]
        if len(areas)!=0:
            roi=areas.index(max(areas))
            if areas[roi] >= 1000:# and area <= 8000:
                x, y, w, z=cv2.boundingRect(contoursVE[roi])
                cv2.rectangle(salvo, (x, y), (x + w, y + z), (0, 0, 0), 2)
                cv2.putText(salvo, "Coca", (x + 3, y + 60), 2, 1, (0, 0, 0), 2)
        preto = SegmentacaoPre2(imagem)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF==ord("o"):
            frame=salvo2
            modo=1
            print '1'
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame", salvo)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    reconhecimentoVideo()
