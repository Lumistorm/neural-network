import numpy as np
import pygame
import sys

from neural_network import NeuralNetwork
from mnist import preprocess_mnist_train, preprocess_mnist_test


def train_model(path):
    model = NeuralNetwork()
    x_train, y_train = preprocess_mnist_train()

    model.train(x_train, y_train, 21, 10, 0.1)
    # model.train(x_train, y_train, 21, 10, 0.1) 137
    model.save(path)


def test_model(path):
    model = NeuralNetwork()
    model.load(path
               )
    x_test, y_test = preprocess_mnist_test()
    correct = 0
    for index, inputs in enumerate(x_test):
        target = y_test[index]
        inputs.shape = inputs.shape + (1, )
        inputs = inputs.T

        prediction = model.forward_prop(inputs)

        loss = model.cross_entropy_loss(prediction, target)
        predicted_class = np.argmax(prediction)
        target_value = np.argmax(target)

        is_correct = predicted_class == target_value
        print(f'{is_correct}: Prediction: {predicted_class}, Target: {target_value}, Loss: {loss}')

        correct += is_correct

    print(f'Correct: {correct}, Wrong: {len(x_test) - correct}')


def run(model_path):
    model = NeuralNetwork()
    model.load(model_path)

    scale = 30
    display = pygame.display.set_mode((840, 840))
    clock = pygame.time.Clock()
    image_array = np.zeros((1, 784))
    lossy = np.zeros((1, 784))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_just_pressed()

        # clear drawing
        if keys[pygame.K_q]:
            display.fill((0, 0, 0))
            image_array.fill(0)
            lossy.fill(0)

        # draw
        if pygame.mouse.get_pressed()[0]:
            mouse_pos = pygame.mouse.get_pos()
            grid_x = mouse_pos[0] // scale
            grid_y = mouse_pos[1] // scale
            image_array[0, grid_y * 28 + grid_x] = 1
            image_array[0, (grid_y - 1) * 28 + grid_x] = 1
            image_array[0, (grid_y + 1) * 28 + grid_x] = 1
            # image_array[0, grid_y * 28 + grid_x + 1] = 1
            image_array[0, grid_y * 28 + grid_x - 1] = 1

            # add lossy
            lossy = image_array.reshape((28, 28))
            lossy = (
                lossy * 4 +
                np.roll(lossy, 1, axis=0) +
                np.roll(lossy, -1, axis=0) +
                np.roll(lossy, 1, axis=1) +
                np.roll(lossy, -1, axis=1)
            ) / 8

            # convert to pygame surface
            rgb = (lossy * 255).astype(np.uint8)
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
            surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            surf = pygame.transform.scale(surf, (840, 840))
            display.blit(surf, (0, 0))

        # predict
        if keys[pygame.K_SPACE]:
            inputs = lossy.reshape((1, 784))
            prediction = model.forward_prop(inputs)
            predicted_class = np.argmax(prediction)
            print(f'{predicted_class}     {(np.max(prediction) * 100):.1f} % certainty')

        pygame.display.flip()
        clock.tick(120)


if __name__ == '__main__':
    run('model_with_tuning.npz')
