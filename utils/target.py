import matplotlib.pyplot as plt
import numpy as np
import time
import os


def get_current_time():
    """获取已使用时间"""
    current_time = time.strftime('%Y-%m-%d %H：%M：%S', time.localtime(time.time()))
    return current_time

def draw_target(train_acc, test_acc, train_f1, test_f1, name, message):
    current_time = str(get_current_time())
    save_path = f"result/{name}"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path = save_path + "/" + current_time
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    draw_acc(train_acc, test_acc, message, save_path)
    draw_f1(train_f1, test_f1, message, save_path)

    return save_path


def draw_f1(train_f1, test_f1, message, save_path):
    plt.figure(figsize=(16, 8))
    plt.title('f1_score')
    plt.plot(train_f1, label='train f1', linewidth=1, color='red')
    plt.plot(test_f1, label='test f1', linewidth=1, color='black')
    plt.xlabel(message)

    train_max = np.argmax(train_f1)
    test_max = np.argmax(test_f1)

    show_train = '[' + str(train_max) + '  ' + str(train_f1[train_max]) + ']'
    plt.plot(train_max, train_f1[train_max], 'ro')
    plt.annotate(show_train, xy=(train_max, train_f1[train_max]),
                 xytext=(train_max, train_f1[train_max]))

    show_test = '[' + str(test_max) + '  ' + str(test_f1[test_max]) + ']'
    plt.plot(test_max, test_f1[test_max], 'ko')
    plt.annotate(show_test, xy=(test_max, test_f1[test_max]), xytext=(test_max, test_f1[test_max]))

    plt.legend()

    plt.savefig(save_path + '/' + 'f1_score.jpg')


def draw_acc(train_acc, test_acc, message, save_path):
    plt.figure(figsize=(16, 8))
    plt.title('acc')
    plt.plot(train_acc, label='train acc', linewidth=1, color='red')
    plt.plot(test_acc, label='test acc', linewidth=1, color='black')
    plt.xlabel(message)

    train_max = np.argmax(train_acc)
    test_max = np.argmax(test_acc)

    show_train = '[' + str(train_max) + '  ' + str(train_acc[train_max]) + ']'
    plt.plot(train_max, train_acc[train_max], 'ro')
    plt.annotate(show_train, xy=(train_max, train_acc[train_max]),
                 xytext=(train_max, train_acc[train_max]))

    show_test = '[' + str(test_max) + '  ' + str(test_acc[test_max]) + ']'
    plt.plot(test_max, test_acc[test_max], 'ko')
    plt.annotate(show_test, xy=(test_max, test_acc[test_max]), xytext=(test_max, test_acc[test_max]))

    plt.legend()

    plt.savefig(save_path + '/' + 'acc.jpg')


if __name__ == "__main__":
    a = np.random.randint(0, 100, 100)
    b = np.random.randint(0, 100, 100)

    draw_acc(a, b, 'rnn')
