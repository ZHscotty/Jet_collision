from ADNet_6 import Model
import process_data


x_train, x_dev, y_train, y_dev = process_data.load_data('train')
x_test, id = process_data.load_data('test')
model = Model()
model.train(x_train, y_train, x_dev, y_dev, batch_size=64, epoch=20)
predict = model.predict(x_test)
print('result ==> result.csv')
process_data.output_data(predict, id)