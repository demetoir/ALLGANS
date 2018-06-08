from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from util.deco import deco_timeit
from util.tensor_ops import *


@deco_timeit
def rnn_mnist_clf():
    mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

    # hyper
    learning_rate = 0.001
    total_epoch = 30
    batch_size = 128

    # input shape
    n_input = 28
    n_step = 28
    n_hidden = 128
    n_class = 10
    X_shape = [None, n_step, n_input]
    Y_shape = [None, n_class]

    X = tf.placeholder(tf.float32, X_shape)
    Y = tf.placeholder(tf.float32, Y_shape)

    def get_cells(X):
        basic_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
        lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
        gru_cell = tf.nn.rnn_cell.GRUCell(n_hidden)

        cell = gru_cell
        output, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        return output, state

    outputs, states = get_cells(X)

    outputs = flatten(outputs)
    model = linear(outputs, n_class)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        total_batch = int(mnist.train.num_examples / batch_size)

        for epoch in range(total_epoch):
            total_cost = 0

            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                batch_xs = batch_xs.reshape((batch_size, n_step, n_input))

                _, cost_val = sess.run([optimizer, cost],
                                       feed_dict={X: batch_xs, Y: batch_ys})
                total_cost += cost_val

            print('Epoch:', '%04d' % (epoch + 1),
                  'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

        print('최적화 완료!')

        #########
        # 결과 확인
        ######

        test_batch_size = len(mnist.test.images)
        test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
        test_ys = mnist.test.labels

        print('정확도:', sess.run(acc, feed_dict={X: test_xs, Y: test_ys}))


@deco_timeit
def rnn_autocomplete():
    # 자연어 처리나 음성 처리 분야에 많이 사용되는 RNN 의 기본적인 사용법을 익힙니다.
    # 4개의 글자를 가진 단어를 학습시켜, 3글자만 주어지면 나머지 한 글자를 추천하여 단어를 완성하는 프로그램입니다.

    char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
                'h', 'i', 'j', 'k', 'l', 'm', 'n',
                'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']

    # one-hot 인코딩 사용 및 디코딩을 하기 위해 연관 배열을 만듭니다.
    # {'a': 0, 'b': 1, 'c': 2, ..., 'j': 9, 'k', 10, ...}
    num_dic = {n: i for i, n in enumerate(char_arr)}
    dic_len = len(num_dic)

    # 다음 배열은 입력값과 출력값으로 다음처럼 사용할 것 입니다.
    # wor -> X, d -> Y
    # woo -> X, d -> Y
    seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']

    def make_batch(seq_data):
        input_batch = []
        target_batch = []

        for seq in seq_data:
            # 여기서 생성하는 input_batch 와 target_batch 는
            # 알파벳 배열의 인덱스 번호 입니다.
            # [22, 14, 17] [22, 14, 14] [3, 4, 4] [3, 8, 21] ...
            input = [num_dic[n] for n in seq[:-1]]
            # 3, 3, 15, 4, 3 ...
            target = num_dic[seq[-1]]
            # one-hot 인코딩을 합니다.
            # if input is [0, 1, 2]:
            # [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
            #  [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
            #  [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]]
            input_batch.append(np.eye(dic_len)[input])
            # 지금까지 손실함수로 사용하던 softmax_cross_entropy_with_logits 함수는
            # label 값을 one-hot 인코딩으로 넘겨줘야 하지만,
            # 이 예제에서 사용할 손실 함수인 sparse_softmax_cross_entropy_with_logits 는
            # one-hot 인코딩을 사용하지 않으므로 index 를 그냥 넘겨주면 됩니다.
            target_batch.append(target)

        return input_batch, target_batch

    # hyper
    learning_rate = 0.01
    n_hidden = 128
    total_epoch = 300

    # 타입 스텝: [1 2 3] => 3
    # RNN 을 구성하는 시퀀스의 갯수입니다.
    n_step = 3
    # 입력값 크기. 알파벳에 대한 one-hot 인코딩이므로 26개가 됩니다.
    # 예) c => [0 0 1 0 0 0 0 0 0 0 0 ... 0]
    # 출력값도 입력값과 마찬가지로 26개의 알파벳으로 분류합니다.
    n_input = n_class = dic_len
    X_shape = [None, n_step, n_input]
    Y_shape = [None]

    #########
    # 신경망 모델 구성
    ######
    X = tf.placeholder(tf.float32, X_shape)
    # 비용함수에 sparse_softmax_cross_entropy_with_logits 을 사용하므로
    # 출력값과의 계산을 위한 원본값의 형태는 one-hot vector가 아니라 인덱스 숫자를 그대로 사용하기 때문에
    # 다음처럼 하나의 값만 있는 1차원 배열을 입력값으로 받습니다.
    # [3] [3] [15] [4] ...
    # 기존처럼 one-hot 인코딩을 사용한다면 입력값의 형태는 [None, n_class] 여야합니다.
    Y = tf.placeholder(tf.int32, Y_shape)

    def get_cell(X):
        # RNN 셀을 생성합니다.
        cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
        # 과적합 방지를 위한 Dropout 기법을 사용합니다.
        cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
        # 여러개의 셀을 조합해서 사용하기 위해 셀을 추가로 생성합니다.
        cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

        # 여러개의 셀을 조합한 RNN 셀을 생성합니다.
        multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

        # tf.nn.dynamic_rnn 함수를 이용해 순환 신경망을 만듭니다.
        # time_major=True
        outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
        return outputs, states

    outputs, states = get_cell(X)
    outputs = flatten(outputs)
    model = linear(outputs, n_class)

    cost = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=model, labels=Y))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    prediction = tf.cast(tf.argmax(model, 1), tf.int32)
    # one-hot 인코딩이 아니므로 입력값을 그대로 비교합니다.
    prediction_check = tf.equal(prediction, Y)
    accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

    with tf.Session() as sess:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        input_batch, target_batch = make_batch(seq_data)

        for epoch in range(total_epoch):
            _, loss = sess.run([optimizer, cost],
                               feed_dict={X: input_batch, Y: target_batch})

            print('Epoch:', '%04d' % (epoch + 1),
                  'cost =', '{:.6f}'.format(loss))

        print('최적화 완료!')

        #########
        # 결과 확인
        ######
        # 레이블값이 정수이므로 예측값도 정수로 변경해줍니다.

        input_batch, target_batch = make_batch(seq_data)
        predict, accuracy_val = sess.run([prediction, accuracy],
                                         feed_dict={X: input_batch, Y: target_batch})

        predict_words = []
        for idx, val in enumerate(seq_data):
            last_char = char_arr[predict[idx]]
            predict_words.append(val[:3] + last_char)

        print('=== 예측 결과 ===')
        print('입력값:', [w[:3] + ' ' for w in seq_data])
        print('예측값:', predict_words)
        print('정확도:', accuracy_val)


@deco_timeit
def rnn_seq2seq():
    # S: 디코딩 입력의 시작을 나타내는 심볼
    # E: 디코딩 출력을 끝을 나타내는 심볼
    # P: 현재 배치 데이터의 time step 크기보다 작은 경우 빈 시퀀스를 채우는 심볼
    #    예) 현재 배치 데이터의 최대 크기가 4 인 경우
    #       word -> ['w', 'o', 'r', 'd']
    #       to   -> ['t', 'o', 'P', 'P']
    char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
    num_dic = {n: i for i, n in enumerate(char_arr)}
    dic_len = len(num_dic)

    # 영어를 한글로 번역하기 위한 학습 데이터
    seq_data = [['word', '단어'], ['wood', '나무'],
                ['game', '놀이'], ['girl', '소녀'],
                ['kiss', '키스'], ['love', '사랑']]

    def make_batch(seq_data):
        input_batch = []
        output_batch = []
        target_batch = []

        for seq in seq_data:
            # 인코더 셀의 입력값. 입력단어의 글자들을 한글자씩 떼어 배열로 만든다.
            input = [num_dic[n] for n in seq[0]]
            # 디코더 셀의 입력값. 시작을 나타내는 S 심볼을 맨 앞에 붙여준다.
            output = [num_dic[n] for n in ('S' + seq[1])]
            # 학습을 위해 비교할 디코더 셀의 출력값. 끝나는 것을 알려주기 위해 마지막에 E 를 붙인다.
            target = [num_dic[n] for n in (seq[1] + 'E')]

            input_batch.append(np.eye(dic_len)[input])
            output_batch.append(np.eye(dic_len)[output])
            # 출력값만 one-hot 인코딩이 아님 (sparse_softmax_cross_entropy_with_logits 사용)
            target_batch.append(target)

        return input_batch, output_batch, target_batch

    #########
    # 옵션 설정
    ######
    learning_rate = 0.01
    n_hidden = 128
    total_epoch = 100
    # 입력과 출력의 형태가 one-hot 인코딩으로 같으므로 크기도 같다.
    n_class = n_input = dic_len

    #########
    # 신경망 모델 구성
    ######
    # Seq2Seq 모델은 인코더의 입력과 디코더의 입력의 형식이 같다.
    # [batch size, time steps, input size]
    enc_input = tf.placeholder(tf.float32, [None, None, n_input])
    dec_input = tf.placeholder(tf.float32, [None, None, n_input])
    # [batch size, time steps]
    targets = tf.placeholder(tf.int64, [None, None])

    # 인코더 셀을 구성한다.
    def encoder(input_):
        with tf.variable_scope('encode'):
            enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
            enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)

            outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, input_,
                                                    dtype=tf.float32)
        return outputs, enc_states

    def decoder(input_, state):
        # 디코더 셀을 구성한다.
        with tf.variable_scope('decode'):
            dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
            dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

            # Seq2Seq 모델은 인코더 셀의 최종 상태값을
            # 디코더 셀의 초기 상태값으로 넣어주는 것이 핵심.
            outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, input_,
                                                    initial_state=state,
                                                    dtype=tf.float32)
        return outputs, dec_states

    enc_output, enc_state = encoder(enc_input)
    dec_output, dec_state = decoder(dec_input, enc_state)

    model = tf.layers.dense(dec_output, n_class, activation=None)

    cost = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=model, labels=targets))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # 결과가 [batch size, time step, input] 으로 나오기 때문에,
    # 2번째 차원인 input 차원을 argmax 로 취해 가장 확률이 높은 글자를 예측 값으로 만든다.
    prediction = tf.argmax(model, 2)

    #########
    # 신경망 모델 학습
    ######
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    input_batch, output_batch, target_batch = make_batch(seq_data)

    for epoch in range(total_epoch):
        _, loss = sess.run([optimizer, cost],
                           feed_dict={enc_input: input_batch,
                                      dec_input: output_batch,
                                      targets: target_batch})

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.6f}'.format(loss))

    print('최적화 완료!')

    #########
    # 번역 테스트
    ######
    # 단어를 입력받아 번역 단어를 예측하고 디코딩하는 함수
    def translate(word):
        # 이 모델은 입력값과 출력값 데이터로 [영어단어, 한글단어] 사용하지만,
        # 예측시에는 한글단어를 알지 못하므로, 디코더의 입출력값을 의미 없는 값인 P 값으로 채운다.
        # ['word', 'PPPP']
        seq_data = [word, 'P' * len(word)]

        input_batch, output_batch, target_batch = make_batch([seq_data])

        result = sess.run(prediction,
                          feed_dict={enc_input: input_batch,
                                     dec_input: output_batch,
                                     targets: target_batch})

        # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
        decoded = [char_arr[i] for i in result[0]]

        # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
        end = decoded.index('E')
        translated = ''.join(decoded[:end])

        return translated

    print('\n=== 번역 테스트 ===')

    print('word ->', translate('word'))
    print('wodr ->', translate('wodr'))
    print('love ->', translate('love'))
    print('loev ->', translate('loev'))
    print('game ->', translate('game'))
    print('girl ->', translate('girl'))

    print()

    alphabet_set = 'asdfghjklzxcvbnmqwertyuiop'
    alpha_list = [i for i in alphabet_set]
    for _ in range(10):
        word = ""
        for i in range(4):
            word += np.random.choice(alpha_list)

        print('{} ->'.format(word), translate(word))


def rnn_cells():
    pass
    # core rnn cell
    n_unit = 10
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_unit)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_unit, name='name', reuse=False)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_unit, name='name', reuse=False)
    gru_cell = tf.nn.rnn_cell.GRUCell(n_unit)

    # Classes storing split `RNNCell` state
    tf.nn.rnn_cell.LSTMStateTuple()

    # core wrapper...

    mullti_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell, gru_cell])
    dropout_cell = tf.contrib.rnn.DropoutWrapper([lstm_cell])

    # tf.contrib.rnn.LSTMBlockWrapper()
    # tf.contrib.rnn.EmbeddingWrapper()
    # tf.contrib.rnn.InputProjectionWrapper()
    # tf.contrib.rnn.OutputProjectionWrapper()
    # tf.contrib.rnn.DeviceWrapper()
    # tf.contrib.rnn.ResidualWrapper()
    #
    # # contrib
    # tf.contrib.nn.BidirectionalGridLSTMCell()
    #
    # # Block RNNCells
    # tf.contrib.rnn.LSTMBlockCell()
    # tf.contrib.rnn.GRUBlockCell()
    # # Fused RNNCells
    # tf.contrib.rnn.FusedRNNCell()
    # tf.contrib.rnn.FusedRNNCellAdaptor()
    # tf.contrib.rnn.TimeReversedFusedRNN()
    # tf.contrib.rnn.LSTMBlockFusedCell()
    #
    # # LSTM - like cells
    # tf.contrib.rnn.CoupledInputForgetGateLSTMCell()
    # tf.contrib.rnn.TimeFreqLSTMCell()
    # tf.contrib.rnn.GridLSTMCell()
    #
    # # RNNCell wrappers
    # tf.contrib.rnn.AttentionCellWrapper()
    # tf.contrib.rnn.CompiledWrapper()
