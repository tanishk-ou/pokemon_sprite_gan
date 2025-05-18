# Pokemon Sprites GAN
Training a GAN model to generate pokemon sprites

In Pokémon games, "sprites" are 2D images used to represent Pokémon and other characters in the game. They are the visual representation of these elements on the screen, particularly in older generations of Pokémon games.

I am trying to train a GAN (Generative Adverserial Network) to generate images similar to the official Pokémon sprites used in Pokémon games.

This project is on-going.

Augmented Pokémon Sprites Images - [Google Drive Link](https://drive.google.com/file/d/1zlUBxeFKxT7EYi0UkAJIlE-VfGgBgaFM/) <br>
.pth file from recent GAN Training - [Google Drive Link](https://drive.google.com/file/d/1vS29NHQScIsEfqyh0nSqDD5ZGjhU9qkZ/)

GAN is currently in very early stages of training and regular updates and improvements are being made to GAN model architecture according to the generator outputs.

## GAN Architecture
### Initial Model
Conv2d layers for discriminator with ReLU and BatchNorm2d. <br>
ConvTranspose2d layers for generator with ReLU and BatchNorm2d.<br>
Generator outputs - Fully Gray

### Update 1
Replaced ConvTranspose2d layers with Upsample combined with Conv2d layers.<br>
Replaced ReLU with LeakyReLU activation.<br>
Generator outputs - Gray with random shapes

### Update 2
Added Dropout layers in generator.<br>
This finally introduced color in generator outputs.<br>
gan-epoch1.png showcases output from this update.

### Update 3
Restarted training with new models.<br>
Replaced BatchNorm2d with PixelNorm. Added Noise Injection in generator blocks, similar to StyleGAN. Used Residual blocks and skip connections in generator blocks.<br>
Removed BatchNorm2d from critic. Added Dropout layers in initial critic blocks. Added Residual blocks and skip connections in critic blocks which didn't have Dropout layers.<br>
Used different initialization method.<br>



Progress will gradually be updated.
