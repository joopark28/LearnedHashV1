# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np

char_table = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
reverse_char_dict = {char_table[i]: i for i in range(len(char_table))}

#根据不同的分布产生数字
def generate_digit(size, distribute='uniform', min=0.0, max=1.0, mean=0.0, sigma=1.0, plot=False):
    data = []
    if distribute == 'uniform':
        data = np.random.rand(size) * (max - min) + min
    elif distribute == 'normal':
        data = np.random.randn(size)
        min = data.min()
        max = data.max()
    elif distribute == 'lognormal':
        data = np.random.lognormal(mean, sigma, size=size)
        min = data.min()
        max = data.max()
    elif distribute == 'int':
        data = np.random.randint(int(min), int(max), size)
    if plot:
        data.sort()
        # plt.figure(1)
        # plt.plot([i for i in range(size)], data)
        # plt.hist(data, 1000, normed=True, histtype='step')

        # plt.plot()
        # plt.show()
        # plt.figure(2)
        # plt.hist(data, 1000, cumulative=True)
        # plt.show()
        print(max, min)
        pdf_data = np.floor((data - min) / (max - min) * 1000)
        print(pdf_data)
        pdf = [0 for i in range(1001)]
        for p in pdf_data:
            pdf[int(p)] += 1
        pdf = list(map(lambda x: x / size, pdf))
        print(pdf)
        cdf = np.cumsum(pdf)
        plt.figure(2)
        index = [i * (max - min) / 1001 + min for i in range(1001)]
        print(index)
        plt.subplot(121)
        plt.plot(index, pdf)
        # plt.ylim(0, 0.002)
        # plt.show()
        # plt.figure(3)
        plt.subplot(122)
        plt.plot(index, cdf)
        plt.show()
    return data

#根据分布产生字符串
def generate_strings(size, distribute='uniform', padding=False, padding_size=20, return_type='char',
                     char_table=char_table,
                     min_length=1, max_length=20, mean_length=4.0,
                     sigma_length=0.3, mean_char=8.0, sigma_char=0.03):
    table_size = len(char_table)
    random_min = 0.0
    random_max = 0.0
    if distribute == 'uniform':
        return [generate_chars(generate_randint(1, distribute, min_length, max_length), distribute, padding=padding,
                               padding_size=padding_size, return_type=return_type, char_table=char_table) for i in
                range(size)]
    if distribute == 'normal':
        normal_data = np.random.randn(10000000)
        random_min = normal_data.min()
        random_max = normal_data.max()
        return [generate_chars(
            generate_randint(1, distribute, min_length, max_length, random_min=random_min, random_max=random_max),
            distribute,
            padding=padding, padding_size=padding_size, return_type=return_type, char_table=char_table,
            random_min=random_min,
            random_max=random_max) for i in range(size)]
    elif distribute == 'lognormal':
        log_normal_length = np.random.lognormal(mean_length, sigma_length, 10000000)
        log_normal_char = np.random.lognormal(mean_char, sigma_char, 10000000)
        random_min_length = log_normal_length.min()
        random_max_length = log_normal_length.max()
        random_min_char = log_normal_char.min()
        random_max_char = log_normal_char.max()
        return [generate_chars(
            generate_randint(1, distribute, min_length, max_length, mean_length, sigma_length, random_min_length,
                             random_max_length), distribute, padding, padding_size, return_type, char_table, mean_char,
            sigma_char, random_min_char, random_max_char) for i in range(size)]

#生成字符串分两步 上一个函数随机生成字符串的长度 着一个是针对每一个长度按分部生成字符
def generate_chars(size, distribute='uniform', padding=False, padding_size=20, return_type='char',
                   char_table=char_table,
                   mean=10.0, sigma=0.1,
                   random_min=0.0, random_max=20.0):
    table_size = len(char_table)
    min = 0
    max = table_size - 1
    if return_type == 'char':
        chars = [char_table[i] for i in
                 generate_randint(size, distribute, min, max, mean, sigma, random_min, random_max)]
        np.random.shuffle(chars)
        if padding:
            chars += ['\0' for i in range(padding_size - size)]
    else:
        chars = generate_randint(size, distribute, min, max, mean, sigma, random_min, random_max)
        np.random.shuffle(chars)
        if padding:
            chars = np.concatenate((chars, [0 for i in range(padding_size - size)]))
    # print(chars)
    return (chars, size)

#随机的生成一个整型数字
def generate_randint(size=1, distribute='uniform', min=0, max=20, mean=0.0, sigma=1.0, random_min=0.0, random_max=20.0):
    if distribute == 'uniform':
        if size == 1:
            return np.random.randint(min, max + 1)
        else:
            return np.random.randint(min, max + 1, size)
    elif distribute == 'normal':
        if size == 1:
            num = np.int32(np.floor((np.random.randn() - random_min) / (random_max - random_min) * (max - min) + min))
        else:
            num = np.int32(
                np.floor((np.random.randn(size) - random_min) / (random_max - random_min) * (max - min) + min))
        return check_num(num, min, max)
    elif distribute == 'lognormal':
        if size == 1:
            num = np.int32(np.floor(
                (np.random.lognormal(mean, sigma) - random_min) / (random_max - random_min) * (max - min) + min))
        else:
            num = np.int32(np.floor(
                (np.random.lognormal(mean, sigma, size) - random_min) / (random_max - random_min) * (max - min) + min))
        return check_num(num, min, max)

#正太分布只能 指定标准差，  将字符串长度 规范到 限定范围   本案例用的 4-10
def check_num(num, min, max):
    num1 = np.where(num < min, min, num)
    num2 = np.where(num1 >= max, max, num1)
    return num2

#画出字符串的分布
def plot():
    np.random.seed(12345)
    method = 'uniform'
    size = 1000000
    # generate_digit(1000000, method='normal', plot=True)
    # generate_digit(1000000, method='lognormal', mean=0.0, sigma=0.6, plot=True)
    # generate_digit(1000000, method='uniform', min=1, max=2, plot=True)  # 长尾分布

    #
    # strings = generate_strings(size, 'lognormal', char_table, 4, 10, mean_length=7.0, sigma_length=0.08,
    #                            mean_char=8.0, sigma_char=0.2) # long tail
    # strings = generate_strings(size, 'uniform', char_table, 4, 10)
    strings = generate_strings(size, 'normal', char_table, 4, 10)
    # print(strings)
    l_pdf = [0 for i in range(7)]
    c_pdf = {}
    c_count = 0
    for string in strings:
        length = len(string)
        l_pdf[length - 4] += 1
        for s in string:
            if c_pdf.__contains__(s):
                c_pdf[s] += 1
            else:
                c_pdf[s] = 1
            c_count += 1
    print(c_pdf)
    print(len(c_pdf))

    tmp = [0 for i in range(len(char_table))]
    for k in c_pdf:
        tmp[reverse_char_dict[k]] = c_pdf[k] / c_count
    c_pdf = tmp
    print(c_pdf)

    l_pdf = list(map(lambda x: x / size, l_pdf))

    l_cdf = np.cumsum(l_pdf)
    c_cdf = np.cumsum(c_pdf)
    plt.figure()
    plt.subplot(221)
    l_index = [4, 5, 6, 7, 8, 9, 10]
    plt.plot(l_index, l_pdf)
    # plt.ylim(0, 0.2)
    plt.subplot(222)
    plt.plot(l_index, l_cdf)
    plt.subplot(223)
    c_index = range(len(char_table))
    # plt.ylim(0, 0.05)
    plt.xticks(c_index, char_table)
    plt.plot(c_index, c_pdf)
    plt.subplot(224)
    plt.xticks(c_index, char_table)
    plt.plot(c_index, c_cdf)

    plt.show()

#集成之前的接口 提供对外接口
def generate_string_data(method='save', distribute='uniform'):
    total_size = 100000
    keys_list = []
    sequence_length_list = []
    if method == 'save':
        for i in range(10):
            data = generate_strings(total_size, distribute='uniform', padding=True, padding_size=10, return_type='int',
                                    char_table=char_table, min_length=4, max_length=10, mean_length=4.0,
                                    sigma_length=0.3, mean_char=8.0, sigma_char=0.03)
            keys = np.expand_dims(list(map(lambda x: x[0], data)), -1)
            sequence_length = list(map(lambda x: x[1], data))
            np.save("data/chars/keys{}.npy".format(i), keys)
            np.save("data/chars/length{}.npy".format(i), sequence_length)
            keys_list.append(keys)
            sequence_length_list.append(sequence_length)
    elif method == 'load':
        for i in range(10):
            keys = np.load("data/chars/keys{}.npy".format(i))
            sequence_length = np.load("data/chars/length{}.npy".format(i))
            keys_list.append(keys)
            sequence_length_list.append(sequence_length)
    return keys_list, sequence_length_list

#集成之前的api 对外提供接口
def generate_digit_data(method='save', distribute='uniform'):
    total_size = 100000
    keys_list = []
    sequence_length_list = []
    if method == 'save':
        for i in range(10):
            keys = generate_digit(total_size, distribute='uniform', min=0.0, max=10.0, mean=0.0, sigma=1.0)
            np.save("data/digit/keys{}.npy".format(i), keys)
            keys_list.append(keys)
    elif method == 'load':
        for i in range(10):
            keys = np.load("data/digit/keys{}.npy".format(i))
            keys_list.append(keys)
    return keys_list


if __name__ == '__main__':
    for distribute in ['uniform', 'normal', 'lognormal']:
        keys_list, sequence_length_list = generate_string_data("save", distribute)
        generate_digit_data("save", distribute)
    print('data generation finished.')
    # print(generate_digit_data("load")[0][1])
    # print(generate_digit_data("load")[0][1])
    # print(generate_digit_data("load")[0][1])
    # print(generate_digit_data("load")[0][1])
