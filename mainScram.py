import numpy as np
import imageio.v2 as imageio
import os
import cv2
from Scrambling import *


def block_scramble(image, block_size, iterations):
    # Определяем размеры изображения
    height, width = image.shape
    scrambled_image = np.copy(image)

    # Перебираем блоки
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # Извлечение блока
            block = scrambled_image[i:i + block_size, j:j + block_size]
            # Скрамблируем блок
            block = Encrypt_Arnold_transform(block, iterations)
            # Возвращаем скрамблированный блок в изображение
            scrambled_image[i:i + block_size, j:j + block_size] = block

    return scrambled_image


def main():
    current_directory = os.path.dirname(__file__)
    img_path = os.path.join(current_directory, 'Images/Airplane.tiff')
    original_image = imageio.imread(img_path)

    # Делаем изображение квадратным в случае несоответствия
    if original_image.shape[0] != original_image.shape[1]:
        size = max(original_image.shape[0], original_image.shape[1])
        original_image = cv2.resize(original_image, (size, size))

    # Убедимся, что изображение чёрно-белое
    if len(original_image.shape) == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Параметры для скрамблирования
    block_size = 64  # Размер блока
    K1 = 2  # Количество итераций для блочного скрамблирования
    K2 = 10  # Количество итераций для скрамблирования всего изображения

    # Первое скрамблирование — блочное
    scrambled_blocks_image = block_scramble(original_image, block_size, K1)

    # Второе скрамблирование — всего изображения
    final_scrambled_image = Encrypt_Arnold_transform(scrambled_blocks_image, K2)

    # Дескрамблирование для проверки (обратный порядок операций)
    descrambled_image = Decrypt_Arnold_transform(final_scrambled_image, K2)
    descrambled_image = block_scramble(descrambled_image, block_size, K1)

    # Отображение результатов
    Display_images(original_image, scrambled_blocks_image, final_scrambled_image,
                   'Two-level Arnold transformation scrambling')


if __name__ == "__main__":
    main()
