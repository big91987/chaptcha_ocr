from captcha.image import ImageCaptcha
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from PIL import Image
import random
import tensorflow as tf
from tensorflow.python import keras


number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = [
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h',
    'i',
    'j',
    'k',
    'l',
    'm',
    'n',
    'o',
    'p',
    'q',
    'r',
    's',
    't',
    'u',
    'v',
    'w',
    'x',
    'y',
    'z']
Alphabet = [
    'A',
    'B',
    'C',
    'D',
    'E',
    'F',
    'G',
    'H',
    'I',
    'J',
    'K',
    'L',
    'M',
    'N',
    'O',
    'P',
    'Q',
    'R',
    'S',
    'T',
    'U',
    'V',
    'W',
    'X',
    'Y',
    'Z']

char_bag = number + alphabet + Alphabet
char_set = number * 4 + alphabet + Alphabet


# char_set : 字符集合
# size: 验证码的长度
# 在线生成验证码
def gen_captcha(char_set=char_set, n_char=4, show_img=False):

    tmp = map(lambda x: np.random.choice(char_set), [0] * n_char)
    text = reduce(lambda x, y: str(x) + str(y), tmp)
    gen = ImageCaptcha()
    captcha = gen.generate(text)
    captcha_img = Image.open(captcha)
    captcha_array = np.array(captcha_img)
    if show_img:
        print(text)
        captcha_img.show()
    # print(text)

    return captcha_array, text
    # print(type(captcha_img))
    # print(str(captcha_img))
    # print(captcha.__dict__)


def make_one_hot(data1, max_num):
    return (np.arange(max_num)==data1[:,None]).astype(np.integer)

def one_hot(input, max_num):
    if isinstance(input, int):
        if input >= max_num:
            return None
        a = np.zeros(shape=(max_num))
        a[input] = 1
        return a
    elif isinstance(input, np.ndarray):
        #a = np.zeros_like(input)
        shape = list(input.shape)
        shape.append(max_num)
        print('shape ===' + str(shape))
        a = np.zeros(shape=shape, dtype=int)
        # a = a[np.newaxis, : ]
        print('a ====='+str(a))
        a[input] = 1


def text2vec(text, char_bag = char_bag):

    tmp = list(map(lambda c: char_bag.index(c) if c in char_bag else None,
               list(text)))
    if None in tmp:
        return None

    a = np.array(tmp)
    max_num = len(char_bag)

    return a, (np.arange(max_num) == a[:,None]).astype(np.integer)


def vec2text(vec, char_bag = char_bag):

    if isinstance(vec, np.ndarray):
        if len(vec.shape) == 1:
            return vec2text(list(vec), char_bag)
        if len(vec.shape) == 2:
            a = np.argmax(vec, axis=1)
            print('a === ' + str(a))
            return vec2text(list(a), char_bag)
        else:
            print('shape of ndarray does not match in vec2text(instance of np.array)')
            return None
    if isinstance(vec, list):
        try:
            return ''.join(list(map(lambda x: char_bag[x], vec)))
        except Exception as e:
            print('err in transfer list to text ...')
            print(e)
            return None

        #if None in tmp:

        #return ''.join(list(map(lambda x:char_bag[x] ,vec)))


#def generate_batch(char_set = char_set, n_char = 4, batch_size = 64):
#    for _ in batch_size:


def generate_dataset(size, ):
    pass



def test_captcha_gen():
    gen_captcha(show_img=True, size=4)

# def test_one_hot():
#     a = np.array([[1,2,3],[3,5,1]])
#     #print(type(a))
#     print(np.arange(10))
#     a = np.array([[1,3,5], [4,3,1]])
#     print(a[:,None])
#     #print(np.array([1,3,5]))
#     print((np.arange(10) == a[:,:]))
#
#
#     a = one_hot(4, 5).astype(np.integer)
#     #b = make_one_hot(4, 5)
#     print('a ====' + str(a))

def test_text2vec():
    a = 'S82x'
    print('a === ' + str(a))
    print('text2vec ing ...')
    vec, vec_one_hot = text2vec(text=a,
                   char_bag = char_bag)
    print('vec === ')
    print(str(vec))
    print('vec_one_hot === ')
    print(vec_one_hot)
    print('one_hot 2 1dim ...')
    print(vec_one_hot.reshape(-1))
    print(vec_one_hot.reshape(-1).reshape(-1,len(char_bag)))

    print('vec2text ing ...')
    aaa = vec2text(vec, char_bag)
    print('text === ' + str(aaa))

    print('vec_one_hot 2 text ing ...')
    bbb = vec2text(vec_one_hot, char_bag)
    print('text = ' + str(bbb))





if __name__ == '__main__':
    # test_captcha_gen()
    # test_one_hot()
    test_text2vec()
