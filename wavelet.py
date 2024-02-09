import numpy as np

def haar_wavelet(data):
    transformed = haar_transform(data)
    reconstructed = haar_inverse(transformed)
    return reconstructed

def haar_transform(data):
    data = np.array(data)
    n = len(data)
    output = np.zeros(n)

    for step in range(int(np.log2(n))):
        for i in range(0, n-1, 2 ** step):
            avg = (data[i] + data[i + 1]) / 2
            diff = (data[i] - data[i + 1]) / 2
            output[i // 2] = avg
            output[n // 2 + i // 2] = diff
        data[:n] = output[:n]

    return output

def haar_inverse(data):
    data = np.array(data)
    n = len(data)
    output = np.zeros(n)

    for step in reversed(range(int(np.log2(n)))):
        for i in range(0, n-1, 2 ** step):
            avg = data[i // 2]
            diff = data[n // 2 + i // 2]
            output[i] = avg + diff
            output[i + 1] = avg - diff
        data[:n] = output[:n]

    return output

def bior3_3_wavelet(data):
    transformed = bior3_3_transform(data)
    reconstructed = bior3_3_inverse(transformed)
    return reconstructed

def bior3_3_transform(data):
    h0 = [1/16, 1/4, 3/8, 1/4, 1/16]
    h1 = [1/2, 1, 1/2]
    g0 = [-1/4, -1/2, 3/4, -1/2, 1/4]
    g1 = [1/2, -1, 1/2]

    data = np.array(data)
    n = len(data)
    output = np.zeros(n)

    for step in range(int(np.log2(n))):
        for i in range(0, n-1, 2 ** step):
            conv_h = np.convolve(data[i:i+2**step], h0, mode='valid')
            conv_g = np.convolve(data[i:i+2**step], g0, mode='valid')
            output[i:i+len(conv_h)] = conv_h

            output[n//2 + i//2:n//2 + i//2 + len(conv_g)] += conv_g

        data[:n] = output[:n]

    return output

def bior3_3_inverse(data):
    h0 = [1/16, 1/4, 3/8, 1/4, 1/16]
    h1 = [1/2, 1, 1/2]
    g0 = [-1/4, -1/2, 3/4, -1/2, 1/4]
    g1 = [1/2, -1, 1/2]

    data = np.array(data)
    n = len(data)
    output = np.zeros(n)

    for step in reversed(range(int(np.log2(n)))):
        for i in range(0, n-1, 2 ** step):
            conv_h = np.convolve(data[i:i+2**step], h1, mode='valid')
            conv_g = np.convolve(data[i:i+2**step], g1, mode='valid')
            output[i:i+len(conv_h)] = conv_h
            output[n//2 + i//2:n//2 + i//2 + len(conv_g)] += conv_g

        data[:n] = output[:n]

    return output

def coif3_wavelet(data):
    transformed = coif3_transform(data)
    reconstructed = coif3_inverse(transformed)
    return reconstructed

def coif3_transform(data):
    h0 = [-0.0156557281, -0.0727326213, 0.384864856, 0.8525720202, 0.3378976625, -0.0727326195, -0.0210602925]
    h1 = [0.0210602925, -0.0727326195, -0.3378976625, 0.8525720202, -0.384864856, -0.0727326213, 0.0156557281]

    data = np.array(data)
    n = len(data)
    output = np.zeros(n)

    for step in range(int(np.log2(n))):
        for i in range(0, n-1, 2 ** step):
            conv_h = np.convolve(data[i:i+2**step], h0, mode='valid')
            conv_g = np.convolve(data[i:i+2**step], h1, mode='valid')
            output[i:i+len(conv_h)] = conv_h

            output[n//2 + i//2:n//2 + i//2 + len(conv_g)] += conv_g

        data[:n] = output[:n]

    return output

def coif3_inverse(data):
    h0 = [-0.0156557281, -0.0727326213, 0.384864856, 0.8525720202, 0.3378976625, -0.0727326195, -0.0210602925]
    h1 = [0.0210602925, -0.0727326195, -0.3378976625, 0.8525720202, -0.384864856, -0.0727326213, 0.0156557281]

    data = np.array(data)
    n = len(data)
    output = np.zeros(n)

    for step in reversed(range(int(np.log2(n)))):
        for i in range(0, n-1, 2 ** step):
            conv_h = np.convolve(data[i:i+2**step], h0, mode='valid')
            conv_g = np.convolve(data[i:i+2**step], h1, mode='valid')
            output[i:i+len(conv_h)] = conv_h
            output[n//2 + i//2:n//2 + i//2 + len(conv_g)] += conv_g

        data[:n] = output[:n]

    return output

def sym15_wavelet(data):
    transformed = sym15_transform(data)
    reconstructed = sym15_inverse(transformed)
    return reconstructed

def sym15_transform(data):
    h0 = [0.000334, -0.001528, 0.000410, 0.003545, -0.000938, -0.008233, 0.002173, 0.019120, -0.004293, -0.044412,
          0.011961, 0.099922, -0.032885, -0.266088, 0.745338, 0.745338, -0.266088, -0.032885, 0.099922, 0.011961,
         -0.044412, -0.004293, 0.019120, 0.002173, -0.008233, -0.000938, 0.003545, 0.000410, -0.001528, 0.000334]

    g0 = h0[::-1]  # Меняем порядок коэффициентов для обратного преобразования

    data = np.array(data)
    n = len(data)
    output = np.zeros(n)

    for step in range(int(np.log2(n))):
        for i in range(0, n-1, 2 ** step):
            conv_h = np.convolve(data[i:i+2**step], h0, mode='valid')
            conv_g = np.convolve(data[i:i+2**step], g0, mode='valid')
            output[i:i+len(conv_h)] = conv_h

            output[n//2 + i//2:n//2 + i//2 + len(conv_g)] += conv_g

        data[:n] = output[:n]

    return output

def sym15_inverse(data):
    h0 = [0.000334, -0.001528, 0.000410, 0.003545, -0.000938, -0.008233, 0.002173, 0.019120, -0.004293, -0.044412,
          0.011961, 0.099922, -0.032885, -0.266088, 0.745338, 0.745338, -0.266088, -0.032885, 0.099922, 0.011961,
         -0.044412, -0.004293, 0.019120, 0.002173, -0.008233, -0.000938, 0.003545, 0.000410, -0.001528, 0.000334]

    g0 = h0[::-1]  # Меняем порядок коэффициентов для обратного преобразования

    data = np.array(data)
    n = len(data)
    output = np.zeros(n)

    for step in reversed(range(int(np.log2(n)))):
        for i in range(0, n-1, 2 ** step):
            conv_h = np.convolve(data[i:i+2**step], h0, mode='valid')
            conv_g = np.convolve(data[i:i+2**step], g0, mode='valid')
            output[i:i+len(conv_h)] = conv_h
            output[n//2 + i//2:n//2 + i//2 + len(conv_g)] += conv_g

        data[:n] = output[:n]

    return output