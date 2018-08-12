# Sound Variational Autoencoder Design

---
## Compression
In order to reduce the computational load, I have decided that
only the first $1/8^{th}$ of the Discrete Cosine Transform
components will be used to train the model. Also, I will take
the DCT over series containing 11025 samples ($1/4^{th}$ seconds).


## Model Design
### Overview
This needs to be a generative model that is capable of producing
music, so I decided to go with a Variational Autoencoder. Why did
I not make a Generative Adversarial Network? PixelRNN? PixelCNN? RBM?
I did not choose an RBM because I don't have much of an idea how to
binarize my data, and I don't know how to train an RBM on non-binary
data. PixelCNN and PixelRNN are out of the question because they take
*way* too long to produce an output since it *must* be done sequentially.
I had to make a tough decision between a GAN and VAE. There are two
reasons why I chose the latter:
1. Logical Reason: GANs are more difficult to train than VAEs, and
VAEs allow precise control over the characteristics of the output.
2. Real reason: I just happened to study about VAEs first.

While most people train their music-generating neural networks on
MIDI files of piano pieces (there are *way* too many of them but I'm
not hatin' on y'all), I decided to take the road not taken and train
mine on raw audio data...well, kinda.

See, we human beings are *way* too good at hearing compared to how
good we are at seeing. Some of us can hear sounds in the range of
approx. 20-20,000 Hz. In order to be able to capture all those
possible frequencies, sound is commonly sampled at 44,100 Hz or
48,000 Hz. In my case, all my files are sampled at 44,100 Hz. This
means that there are 44,100 samples in *each second* of an audio file.
And there are two channels in stereo audio (which is what we have).
This means that in a song that lasts four minutes, there are
$2\cdot4\cdot60\cdot44,100=21,168,000$ samples! That is *way* too much
resolution to train a model on. Even though I have access to a
GTX 1070 GPU (thanks, dad). We can make out an image even if we remove,
let's say, three-fourths of all its pixels (resize by 0.5 along each
dimension), but sounds can get pretty distorted if we do the same
with raw samples.

So I decided to do something similar to what JPEG does for images and
MP3 does for audio: take the Discrete Cosine Transform and remove the
higher frequencies (that is not *exactly* what JPEG and MP3 do, but
that is the gist of it), simply because the higher frequencies don't
contribute as much to our perception of a song/image as do lower
frequencies. So I have decided to take the DCT of the songs and remove
all but the one-eighth lowest frequencies. In order not to lose temporal
information, I ran the DCT on spans covering quarter-second durations.
The output of the VAE will be passed through an inverse-DCT in order to
get a song back. This will allow us to reduce the size of our inputs and
outputs to just an eighth of the original audio files, making the model
much more manageable and resistant to overfitting. And my GPU will have
a somewhat easier time training.

I then needed to choose between an RNN and a CNN for the encoder and
decoder (a dense network is but the last thing you'd want for time-series
data like music). I chose to go with a CNN instead of the more obvious RNN
for a few reasons:
1. RNNs generate series sequentially, one time step at a time, while CNNs
generate series in parallel, producing all the time steps in one go. This
means inference using the decoder will be much faster with a CNN than with
an RNN.
2. CNNs are faster to train compared to RNNs due to reason 1.
3. CNNs don't suffer from the vanishing gradient problem unlike RNNs,
because a deep enough CNN with large enough kernels (and adequate dilation)
can easily include the entire input/output sequence in its receptive field,
while a similarly powerful LSTM network would need *way* more parameters
and would take *much* longer to train.

So let me explain a little detail about my data format. Since I am
(a) taking the DCT of my input song over intervals, and
(b) using a CNN, it only makes sense that my inputs and outputs
be like images. More precisely, the "images" are going to contain
$\left\lfloor11025/8\right\rfloor=1378$ columns from the output of
the DCT, and $128\cdot4=512$ rows from 4 time steps in each second, of
which there are going to be 128, because I plan to train on 128-second
long audio samples. Finally, there will be two channels in the image,
one for the right audio channel and one for the left audio channel, for
a grand total of $2\cdot1378\cdot512=1,411,072$ 32-bit floats per
training sample. That is *way* better than the 21,168,000 32-bit
floats we would've had if we had trained on unprocessed audio data,
more than 15 times smaller to be precise. These will be the inputs and
outputs for our convolutional VAE. Now let's talk a little about the
structure of the VAE itself.


### Structure
I will make it a fully convolutional VAE, so we will only use the
[`Conv2D`](https://keras.io/layers/convolutional/#conv2d)
and [`Conv2DTranspose`](https://keras.io/layers/convolutional/#conv2dtranspose)
layers from the
[Keras Functional API](https://keras.io/getting-started/functional-api-guide/)
to build the actual neural network, with the encoder comprising of
`Conv2D` layers and the decoder comprising of `Conv2DTranspose` layers.
There will also be a `Lambda` layer to sample the latent variables. Now
let's agree upon a set of hyperparameters for the network.

1. We will train the network for 200 epochs, with each epoch consisting of
40 128-second long training samples.
2. The input and output layers will be of shape $(1378\times512\times2)$.
3. The latent dimension will be of shape $(1\times1\times64)$. Since this VAE
is fully convolutional, the hidden layer needs to have 3 dimensions, not 1.
