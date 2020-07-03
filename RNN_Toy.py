import tensorflow as tf
import tensorflow.keras.preprocessing.sequence as keras_prep_sq

from tensorflow.keras.preprocessing.text import Tokenizer

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.model_selection import train_test_split

import re
import os
import time
import numpy as np
import io




def data_to_arr(f_name):
  el_file = open(f_name + ".txt", 'r', encoding='utf-8' )
  text = el_file.read().lower()
  text = text.replace("\n", "")
  text = text.split("<spl>")
  
  t_arr = np.array(text)

  return t_arr


def tokenize(corpus):

  tokenizer = Tokenizer(num_words=3000, filters='', oov_token=1)#Just creates tokenzier
  tokenizer.fit_on_texts(corpus) # Actually applies Tokenizer

  tensor = tokenizer.texts_to_sequences(corpus)
  tensor = keras_prep_sq.pad_sequences(tensor,padding='post')

  return tensor, tokenizer

def string_to_tensors(data_volume):

  tr_input, tr_output = data_to_arr("train_BD"), data_to_arr("train_TI")
  ginor_index = []

  # Limiting Data size 
  for i, v in enumerate(tr_input):
    if (len(v) > 750):
      ginor_index.append(i)
    else:
      continue

  print("bad inputs: ", len(ginor_index))
  tr_input = np.delete(tr_input, ginor_index)
  tr_output = np.delete(tr_output, ginor_index)
  print("good inputs: ", len(tr_input))

  sz = min(data_volume, len(tr_input))

  tr_input = tr_input[0:(sz-1)]
  tr_output = tr_output[0:(sz-1)]
  
  input_tensor, input_tok = tokenize(tr_input)
  output_tensor, output_tok = tokenize(tr_output)

  return input_tensor, output_tensor, input_tok, output_tok

def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))

titles = data_to_arr("test_TI")
print(titles[2])



bd_tensor, tl_tensor, bd_tknz, tl_tknz = string_to_tensors(2000)

max_length_targ, max_length_inp = tl_tensor.shape[1], bd_tensor.shape[1]

bdts_tr, bdts_val, tlts_tr, tlts_val = train_test_split(bd_tensor, tl_tensor, test_size=0.2)

print(bdts_tr.shape)

'''
print ("Input Language; index to word mapping")
convert(bd_tknz, bdts_tr[765])
print ()
print ("Target Language; index to word mapping")
convert(tl_tknz, tlts_tr[765])
'''

####################################################################
####################################################################
####////////////////////////////////////////////////////////////####
####////////////////////////////////////////////////////////////####
####################################################################
####################################################################

BUFFER_SIZE = len(bdts_tr)
BATCH_SIZE = 32 #Originally 64
steps_per_epoch = len(bdts_tr)//BATCH_SIZE
embedding_dim = 512 #Originally 256
units = 512 #originally 1024

vocab_inp_size = len(bd_tknz.word_index)+1
vocab_tar_size = len(tl_tknz.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((bdts_tr, tlts_tr)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
print(example_input_batch.shape, example_target_batch.shape)
print("Vocab Size (Body)\": ",  vocab_inp_size)
print("Vocab Size (title)\": ",  vocab_tar_size)

'''
/////////////
Encoder/////
////////////
'''
class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)


# sample input
sample_hidden = encoder.initialize_hidden_state()


sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))


'''
/////////////
Attention/////
////////////
'''
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

  

attention_layer = BahdanauAttention(256)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

'''
/////////////
Decoder/////
////////////
'''
class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the LSTM
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

####################################################################
####################################################################
####////////////////////////////////////////////////////////////####
####////////////////////////////////////////////////////////////####
####################################################################
####################################################################

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


####################################################################
####################################################################
####////////////////////////////////////////////////////////////####
####/////////////TRAINING///////////////////////////////////////####
####################################################################
####################################################################

@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([tl_tknz.word_index['<title>']] * BATCH_SIZE, 1)
    print("opened gradient tape")

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
      print("Created predictoins and dec_hidden")

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss




EPOCHS = 40

losslist = []

for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss

    if batch % 10 == 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
  # saving (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  losslist.append(total_loss / steps_per_epoch)
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))



####################################################################
####################################################################
####////////////////////////////////////////////////////////////####
####/////////////Evaluating/////////////////////////////////////####
####################################################################
####################################################################

def evaluate(recipe):
  recipe = recipe.strip()

  inputs = []
  
  for i in recipe.split(' '):
    try:
      inputs.append(bd_tknz.word_index[i])
    except KeyError:
      inputs.append(1)
      
      
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  init_hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, init_hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([tl_tknz.word_index['<title>']], 0)

  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input,dec_hidden, enc_out)


    predicted_id = tf.argmax(predictions[0]).numpy()

    result += tl_tknz.index_word[predicted_id] + ' '

    if tl_tknz.index_word[predicted_id] == '</title>':
      return result, recipe

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, recipe



def title_generation(recipe):
  result, recipe = evaluate(recipe)

  return result

  


####################################################################
####################################################################
####////////////////////////////////////////////////////////////####
####/////////////My inputs /////////////////////////////////////####
####################################################################
####################################################################
'''
#plotting

losslist = np.array(losslist)

figure, ax = plt.subplots(figsize=(16,12))
figure.suptitle("Loss over time").set_fontsize(30)

ax.plot(np.arange(EPOCHS)+1, losslist, color='r')

ax.set(xlabel= "Epoch", ylabel="Loss")

figure.patch.set_facecolor('w')

plt.savefig("epork.png")
'''

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))



recips = data_to_arr("test_BD")
recips = recips[300:400]
titles = data_to_arr("test_TI")
titles = titles[300:400]

final_result = open("result_j.txt", 'w', encoding='utf-8')

for i, x in enumerate(recips):
  predct = title_generation(x)
  actual = titles[i]
  final_result.write("Predicted Title: " + predct + "\n")
  final_result.write("Actual Title: " + actual + "\n\n")

final_result.close()






