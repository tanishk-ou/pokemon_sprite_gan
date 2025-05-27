# Pokemon Sprites GAN
Training a GAN model to generate pokemon sprites

In Pokémon games, "sprites" are 2D images used to represent Pokémon and other characters in the game. They are the visual representation of these elements on the screen, particularly in older generations of Pokémon games.

I am going to train a GAN (Generative Adverserial Network) to generate images similar to the official Pokémon sprites used in Pokémon games.

This project is on-going.

Augmented Pokémon Sprites Images - [Google Drive Link](https://drive.google.com/file/d/1zlUBxeFKxT7EYi0UkAJIlE-VfGgBgaFM/) <br>
.pth file from recent GAN Training - [Google Drive Link](https://drive.google.com/file/d/1vS29NHQScIsEfqyh0nSqDD5ZGjhU9qkZ/)

GAN is currently in very early stages of training and regular updates and improvements are being made to GAN model architecture according to the generator outputs.
## Vision for this project
Creating a GAN model to generate Pokémon sprites. These are 96x96 colored png files.<br>
The end goal for the GAN model is to create new Pokémon sprites which are similar to the original sprites from the games.<br>
Also, the GAN should generate images with transparency similar to the original sprites.<br>
Therefore, first I will train the GAN to produce good enough sprites in RGB format. Then, use the weights from that model for a new model which generates RGBA format images.<br>
The reason for this is that if the GAN is trained from scratch in RGBA format, it is complicated and it is also extremely easy for generator to enter mode collapse by creating transparent images.

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

### Update 4
Restarted training with a different custom Dataset class that changes the background of RGBA images into blurry background or opaque mean color from the pixels of that Pokémon.<br>
These images are then converted into RGB format and fed for training. Hopefully, this encourages GAN to focus on the Pokémon rather than the background.<br>
Also, started using gradient accumulation, so that batches of 128 size are effectively accumulated into batches of size 2048.

### Update 5
The GAN is still focusing on backgrounds since the randomness of backgrounds is much easier to capture than the randomness of different Pokémons. <br>
Added a Mini-Batch Discrimination layer in Critic. Added a mask which forces critic to focus more near the centre of the images in every residual block of critic.<br>
Since the critic was focusing too much to backgrounds and all Pokémon are centered in the image, a mask to prioritise centre part of the image should help critic focus on the foreground.<br>
GAN will now be trained on RGBA images and to force the models to not focus on alpha, the following changes have been made:-<br>
Generator and Critic losses are changed as well.<br>

```python
fg_mask = alpha
bg_mask = 1 - alpha

real_loss = fg_mask * critic_real_score + bg_mask * bg_weight * critic_real_score
fake_loss = fg_mask * critic_fake_score + bg_mask * bg_weight * critic_fake_score

critic_loss = fake_loss - real_loss + c_lambda * gp

gen_loss = -fake_loss + bg_lambda * background_supression_loss
```
here, both losses are essentially the same as the original WGAN-GP losses.<br>
the changes in the losses are-<br>
1. `fake_loss` and `real_loss` are combinations of the original critic scores and the alpha of original images. This reduces gradients of the background.
2. `background_supression_loss` forces generator to focus more on foreground and suppresses all pixel values in background.
This should help the GAN to finally focus on the actual Pokémon.
<br><br><br><br>
Progress will gradually be updated.
