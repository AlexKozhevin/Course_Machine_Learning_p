from skimage.io import imread
import skimage
import pylab
import numpy as np
from sklearn.cluster import KMeans
from math import log10, sqrt
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 1
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


"""
  Загрузите картинку parrots.jpg. Преобразуйте изображение, приведя все значения в интервал от 0 до 1.
ля этого можно воспользоваться функцией img_as_float из модуля skimage. Обратите внимание на этот шаг,
так как при работе с исходным изображением вы получите некорректный результат.
"""

image = imread('static/parrots.jpg')
image = skimage.img_as_float(image)

"""
  Создайте матрицу объекты-признаки: характеризуйте каждый пиксель тремя координатами - значениями интенсивности
в пространстве RGB.
"""

w, h, d = original_shape = tuple(image.shape)
assert d == 3
matrix = np.reshape(image, (w * h, d))

n = 1
psnr1 = 0
psnr2 = 0

"""
  Запустите алгоритм K-Means с параметрами init='k-means++' и random_state=241. После выделения кластеров
все пиксели, отнесенные в один кластер, попробуйте заполнить двумя способами: медианным и средним цветом
по кластеру.
  Измерьте качество получившейся сегментации с помощью метрики PSNR.
"""

for clusters in range(1, 20):
    kmn = KMeans(init='k-means++', random_state=241, n_clusters=clusters)
    kmn.fit(matrix)
    pred = kmn.predict(matrix)

    X_median = matrix.copy()
    for i in range(kmn.n_clusters):
        X_median[pred == i] = np.median(matrix[pred == i], axis=0)

    X_mean = matrix.copy()
    for i in range(kmn.n_clusters):
        X_mean[pred == i] = np.mean(matrix[pred == i], axis=0)

    psnr1 = peak_signal_noise_ratio(matrix, X_median, data_range=1)
    psnr2 = peak_signal_noise_ratio(matrix, X_mean, data_range=1)

    if psnr1 < 20 and psnr2 < 20:
        n += 1
    else:
        break

"""
  Найдите минимальное количество кластеров, при котором значение PSNR выше 20 (можно рассмотреть
не более 20 кластеров, но не забудьте рассмотреть оба способа заполнения пикселей одного кластера).
"""

# print(n)
# print(PSNR(matrix, X_median))
# print(PSNR(matrix, X_mean))
# print(psnr1)
# print(psnr2)

plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image')
plt.imshow(image)

plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('K-Median')
plt.imshow(recreate_image(kmn.cluster_centers_, kmn.predict(X_median), w, h))

plt.figure(3)
plt.clf()
plt.axis('off')
plt.title('K-Means')
plt.imshow(recreate_image(kmn.cluster_centers_, kmn.predict(X_mean), w, h))
plt.show()
