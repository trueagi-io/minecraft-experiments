* ae_0_stack_z_learn.py
   - Layer-by-layer training of stacked autoencoders
   - 2x features for 2x compression over H and over W (2x compression overall)
   - `z` SGD for decoder training alternated with whole AE layer training
   - (the idea is to train decoder and latent representation amenable for `z` SGD)
   - One layer reconstruction loss (L-0): 0.000141 (`z` optimization), 0.000440 (AE)
   - Layer-1: 0.000310/0.000345, L-2: 0.000315/0.000335, L-3: 0.000465/0.000493
   - ae_0_stack_z_test.py : optimizes all `z` simultaneouly:
   - Bottom Reconstruction Loss: ~0.0003 | Total Loss: ~0.0007
   - Straight Reconstruction Loss: 0.00327

* ae_1_sparse_learn.py
   - Layer-by-layer training of stacked autoencoders
   - `z` is not optimized
   - (optimization of `z` is possible, but requires applying constraints)
   - One layer reconstruction loss (L-0): 0.000020
   - L-1: 0.000059, L-2: 0.000007, L-3: 0.000056
   - Straight Reconstruction Loss: 0.00192

* ae_2_allz_learn.py
   - Simultaneous training of all decoders with gradient optimization of `z`
   - Encoders are trained to produce optimized `z`
   - No need for layer-by-layer pretraining, but slower.
   - Prone to unstable training, because `z` SGD, decoder, encoder training have their own speeds
   - Sparse version can be trained at least from pretrained checkpoints
   - Loss greatly depends on setup, but typical/achievable is
   - Loss: (0.000256445 0.000185261 8.95173e-05 1.98128e-05) | Encoders: 0.000319

* ae_3_3d_learn.py (not too successful)
   - Layer-by-layer training of stacked 3D autoencoders
   - 4x features for 2x compression over H/W/T (2x compression overall)
     One layer reconstruction loss (L-0): 0.0010 (AE) is not too promising
   - 2x features for 2x H/W compression without T compression
     One layer reconstruction loss (L-0): 0.0009 (AE), which worse than 2D AE.
     It's strange, because 3D kernels could degrade to 2D, and shouldn't be worse.
     The training process is slow, so there is a chance that the model underlearns.
   - Also, side effects have considerable impact.
   - Training multiple levels with `z` optimization works very badly, because it
     seems that different layers should have very different gradient steps.
   - `z`-optimized training with per layer pretraining (worse than 2D AE)
     Loss: (0.0005756 0.000161053 2.24812e-05 5.48212e-05) | Encoders: 0.000052

 * ae_3t_3d_learn.py
   - Causal convolution: no padding, deconvolution results which get not enough
     inputs are cut off. One layer `z` lacks `kernel_size-1` elements, and
     reconstruction contains `(kernel_size-1)*2`, which get less than
     `kernel_size-1` inputs.
   - Faster and more stable training than non-causal 3D AE
   - 2x features for 2x H/W compression without T compression
     One layer reconstruction loss (L-0): 0.00071 (AE), which
     better than non-causal 3D AE, but still worse than 2D AE.
   - May still be underlearned.
   - ------------------
   - Causal AE can be used for prediction. To do this, one just needs to
     pass shifted input to reconstruction loss
   - 4x features for 2x H/W (no compression), because it's reconstruction
   - One layer prediction loss (L-0): 0.0081
   - 4 layers straight prediction loss: 0.0023
   - The issue with predictive 3D AE is that they predict latent codes of
     each next frame not using information from it. This helps to extract
     useful features. But they don't construct descriptions of image frames
     (although, one may think of a decoder as a generative model of current
     frames, which can be pretrained with a normal encoder).

  * ae_4_fixing_learn.py
   - Incomplete attempt to replace SGD `z` optimization with fixing encoders
     trained to predict dz based on dx. Hierarchical optimization is not
     implemented yet, and the overall training procedure may require
     refinements (e.g. whether the decoder should be optimized for final z,
     or just together with the first-guess encoder, etc.).
   - One layer reconstruction loss (L-0): 0.000186 (`z` optimization - somewhat
     worse than SGD optimization, but faster and hopefully more stable) /
     0.00048 (main AE only - can be improved but makes z optimization worse
     for some reason)
