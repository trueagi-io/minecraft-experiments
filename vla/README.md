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
   - Non-causal 3D convolution
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
   - 4 layers straight reconstruction loss: 0.0023
   - May still be undertrained.
   - ------------------
   - Causal AE can be used for prediction. To do this, one just needs to
     pass shifted input to reconstruction loss
   - 4x features for 2x H/W (no compression), because it's reconstruction
   - One layer prediction loss (L-0): 0.0081
   - 4 layers straight prediction loss: 0.0023 (too close to reconstruction?)
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


  * vt_0_learn.py
    - Factorized spatial/temporal transformer predictor on top of AE features.
    - 3 spatial + 4 temporal attention layers are tested.
    - Prediction loss of dense layer-4 AE features - 0.0020
    - Prediction loss of sparse layer-4 AE features - 0.00031 (0.00026 in another run)
      (losses on sparse and dense latents cannot be compared, however, reconstruction
      from prediction of dense latents looks just like blurring, while in the case
      of sparse latents motion is in general predicted)
    - Sparse latents prediction by Transformer is similar in quality to dense
      3D causal convolution-based prediction. The former produces more stable
      results, which degrade slower over time, and which can remain unblurred for
      static camera. The latter can reproduce, what was outside the current frame
      several frames ago, but its results degrade due to increasing artefacts
      even for a static camera.
    - 3D AE features could be helpful for better prediction, but it didn't
      happen in experiments

  * vt_0e_learn.py (kinda negative despite reasonable loss)
      - end-to-end Transformer+AE fine-tuning on pixel-level next frame prediction.
      - Pixel-level prediction loss - 0.00228 - very close to causal convolution.
      - Quality is worse than for latent code prediction by Transformer, because
        Transformer still produces the predicted latent code, which is worsened
        by decoding-encoding. If one tries to do autoregressive prediction of
        the latent code without decoding-encoding, this makes things even worse,
        because encoder and decoder are not in correspondence anymore. Transformer
        predicts the latent code, which is good for decoder, but this is not the same
        latent code, which is constructed by encoder.
      - This end-to-end fine-tuning doesn't seem like a good idea in such settings.
