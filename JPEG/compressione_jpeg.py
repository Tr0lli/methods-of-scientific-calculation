import numpy as np
from JPEG.dct2D import dct_2D, idct_2D
import matplotlib.pyplot as plt

def jpeg_demo_block(q=95):
    # === MATRICE DI QUANTIZZAZIONE ===
    if q > 50:
        qf = (100 - q) / 50
    else:
        qf = 50 / q

    Q = qf * np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109,103,77],
        [24, 35, 55, 64, 81, 104,113,92],
        [49, 64, 78, 87, 103,121,120,101],
        [72, 92, 95, 98, 112,100,103,99]
    ])

    Q = np.round(Q).astype(np.int32)
    Q = np.maximum(Q, 1)

    print("Matrice di quantizzazione Q:")
    print(Q)
    input("Premi invio per continuare...\n")

    # === MATRICE DI INPUT (immagine farlocca 8x8) ===
    A = np.array([
        [52, 55, 61, 66, 70, 61, 64, 73],
        [63, 59, 55, 90, 109, 85, 69, 72],
        [62, 59, 68, 113, 144, 104, 66, 73],
        [63, 58, 71, 122, 154, 106, 70, 69],
        [67, 61, 68, 104, 126, 88, 68, 70],
        [79, 65, 60, 70, 77, 68, 58, 75],
        [85, 71, 64, 59, 55, 61, 65, 83],
        [87, 79, 69, 68, 65, 76, 78, 94]
    ], dtype=np.float64)

    print("Matrice originale A:")
    print(A)
    # input("Premi invio per continuare...\n")

    # === COMPRESSIONE ===
    print("Sottraggo 128")
    AA2 = A - 128
    print(AA2)
    # input("Premi invio per continuare...\n")

    print("Applico la DCT 2D")
    AA2 = dct_2D(AA2)
    print(np.round(AA2, 2))
    # input("Premi invio per continuare...\n")

    print("Divido per Q e arrotondo")
    AA2 = np.round(AA2 / Q).astype(np.int32)
    print(AA2)
    # input("Premi invio per continuare...\n")

    # === DECOMPRESSIONE ===
    print("Rimoltiplico per Q")
    A2 = AA2 * Q
    print(A2)
    # input("Premi invio per continuare...\n")

    print("Applico la IDCT 2D")
    A2 = idct_2D(A2)
    print(np.round(A2, 2))
    # input("Premi invio per continuare...\n")

    print("Riaggiungo 128, arrotondo e taglio in [0, 255]")
    A2 = np.round(A2 + 128)
    A2 = np.clip(A2, 0, 255).astype(np.uint8)
    print(A2)
    # input("Premi invio per continuare...\n")

    print("Confronto con A originale:")
    print(A.astype(np.uint8))

    # Visualizza la differenza assoluta
    diff = np.abs(A - A2)
    plt.imshow(diff, cmap='gray')
    plt.title('Differenza Assoluta tra A e A2')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    jpeg_demo_block(q=95)
