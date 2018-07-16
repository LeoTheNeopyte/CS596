import numpy as np
from conf_matrix import func_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf
import time
import data_helpers
import func_two_layer_fc
import os.path
from datetime import datetime



data_sets = np.load("faces2.npy").item()
images_train = data_sets["images_train"]
labels_train = data_sets["labels_train"]
images_test = data_sets["images_test"]
labels_test = data_sets["labels_test"]

PIXELS = 80*53

# transform each image 
x_train = images_train.reshape(len(images_train), PIXELS).astype('float32')
x_test = images_test.reshape(len(images_test), PIXELS).astype('float32')
y_train = labels_train
y_test = labels_test
test_x = x_test



# normalize inputs from gray scale of 0-255 to values between 0-1
x_train = x_train / 255
x_test = x_test / 255


#Split Training set  into training and validation sets
x_training = x_train[:175,:]
y_training = y_train[:175]
x_validation = x_train[175:,:]
y_validation = y_train[175:]



# Model parameters as external flags
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for the training.')
flags.DEFINE_integer('max_steps', 500, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1',500, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('batch_size', 400,
  'Batch size. Must divide dataset sizes without remainder.')
flags.DEFINE_string('train_dir', 'tf_logs',
  'Directory to put the training data.')
flags.DEFINE_float('reg_constant', 0.1, 'Regularization constant.')

FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
  print('{} = {}'.format(attr, value))
print()



CLASSES = 2

beginTime = time.time()

logdir = FLAGS.train_dir + '/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'



# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, PIXELS],
  name='images')
labels_placeholder = tf.placeholder(tf.int64, shape=[None], name='image-labels')

# Operation for the classifier's result
logits = func_two_layer_fc.inference(images_placeholder, PIXELS,
  FLAGS.hidden1, CLASSES, reg_constant=FLAGS.reg_constant)

# Operation for the loss function
loss = func_two_layer_fc.loss(logits, labels_placeholder)

# Operation for the training step
train_step = func_two_layer_fc.training(loss, FLAGS.learning_rate)

# Operation calculating the accuracy of our predictions
accuracy = func_two_layer_fc.evaluation(logits, labels_placeholder)

#conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int)

# Operation merging summary data for TensorBoard
summary = tf.summary.merge_all()

# Define saver to save model state at checkpoints
saver = tf.train.Saver()

# -----------------------------------------------------------------------------
# Run the TensorFlow graph
# -----------------------------------------------------------------------------

with tf.Session() as sess:
  # Initialize variables and create summary-writer
  sess.run(tf.global_variables_initializer())
  summary_writer = tf.summary.FileWriter(logdir, sess.graph)

  #Generate input data batches
  zipped_data = zip(x_train,y_train)
  batches = data_helpers.gen_batch(list(zipped_data), FLAGS.batch_size,
    FLAGS.max_steps)
   
  for i in range(FLAGS.max_steps):
  
    # Get next input data batch
    batch = next(batches)
    images_batch, labels_batch  = zip(*batch)
               
    feed_dict = {
        images_placeholder: images_batch,
        labels_placeholder: labels_batch,
        
      }
 
    # Periodically print out the model's current accuracy
    if i % 100 == 0:
         train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
         print('Step {:d}, training accuracy {:g}'.format(i, train_accuracy))
         summary_str = sess.run(summary, feed_dict=feed_dict)
         summary_writer.add_summary(summary_str, i)

    # Perform a single training step
    sess.run([train_step, loss], feed_dict=feed_dict)
    
    # Periodically save checkpoint
    if (i + 1) % 1000 == 0:
         checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
         saver.save(sess, checkpoint_file, global_step=i)
         print('Saved checkpoint')
       
  # After finishing the training, evaluate on the test set
  test_accuracy = sess.run(accuracy, feed_dict={
    images_placeholder: x_test,
    labels_placeholder: y_test})
 
    #Prediction based on test set      
  prediction = sess.run(logits, feed_dict ={
    images_placeholder: x_test,
    labels_placeholder: y_test})
  prediction = tf.argmax(prediction, 1)
  y_pred = prediction.eval(feed_dict = {images_placeholder: x_validation})
  conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(labels_test, y_pred)
  print("\n""Confusion Matrix: \n" ,conf_matrix,"\n""\n accuracy: ",
    accuracy,"\n""\n recall array: ", recall_array,"\n""\n precision_array: ", precision_array)
  
  
  correct_pred = (y_pred == labels_test)
  count = 0
  while (True):
    x = np.random.randint(0,len(correct_pred))
    if(not(correct_pred[x])):
        count+=1
        print("Predicted to: ",y_pred[x]," - Correct value: ",labels_test[x])
        plt.imshow(images_test[x,:],interpolation = 'nearest')
        plt.show()
    if (count == 10):
        break
  
  print("\n"'Test accuracy {:g}'.format(test_accuracy))
  
endTime = time.time()
print("\n"'Total time: {:5.2f}s'.format(endTime - beginTime))





