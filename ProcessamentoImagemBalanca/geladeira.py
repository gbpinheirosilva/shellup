# UFRJ
# Bruno
# UFRJ Nautilus
# Maratona maker

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


def reconhecimento(imagem):
#    endereco = "/home/zarjitsky/Imagens/reconhecimentoGeladeira/"
    imagem = cv2.imread(imagem)
    salvo = imagem
    cv2.medianBlur(imagem, 5)
    amarelo = SegmentacaoAma(imagem)
    contoursAM = cv2.findContours(amarelo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    for i in range(len(contoursAM)):
        area = cv2.contourArea(contoursAM[i])
        x, y, w, z = cv2.boundingRect(contoursAM[i])
        if area >= 3000 and area <= 8000:
            cv2.rectangle(salvo, (x, y), (x + w, y + z), (0, 0, 0), 2)
            cv2.putText(salvo, "Leite", (x + 3, y - 6), 1, 1, (0, 0, 0))
    vermelho = SegmentacaoVer(imagem)
    contoursVE = cv2.findContours(vermelho, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    for i in range(len(contoursVE)):
        area = cv2.contourArea(contoursVE[i])
        x, y, w, z = cv2.boundingRect(contoursVE[i])
        if area >= 3000 and area <= 8000:
            cv2.rectangle(salvo, (x, y), (x + w, y + z), (0, 0, 0), 2)
            cv2.putText(salvo, "Coca", (x + 3, y - 6), 1, 1, (0, 0, 0))
    azul = SegmentacaoAzu(imagem)
    contoursAZ = cv2.findContours(azul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    for i in range(len(contoursAZ)):
        area = cv2.contourArea(contoursAZ[i])
        x, y, w, z = cv2.boundingRect(contoursAZ[i])
        if area >= 1000 and area <= 8000:
            cv2.rectangle(salvo, (x, y), (x + w, y + z), (0, 0, 0), 2)
            cv2.putText(salvo, "Requeijao", (x + 3, y - 6), 1, 1, (0, 0, 0))
    imagem = vermelho + azul + amarelo
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", salvo)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



lista = ["geladeiraRequeijaoLeite.jpg", "geladeiraRequeijao.jpg", "geladeiraRequeijaoLeite.jpg", "geladeiraLeite.jpg", \
         "geladeiraCoca.jpg", "geladeiraCheia.jpg", "geladeiraCocaRequeijao.jpg", \
         "geladeiraCocaLeite.jpg", "geladeiraCheia2.jpg", "geladeiraTeste.jpg"]
if __name__ == "__main__":
    for i in lista:
        reconhecimento(i)
