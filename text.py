from keras.utils import to_categorical

a = [[1],[5],[4], [21], [1]]
r = to_categorical(a)
print(r)