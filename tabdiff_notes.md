Some notes one where stuff happens in tabdiff

Tabdiff is trained using a "trainer" object, which will take care of the optimizer, # iter, etc. (basically what `transformers.Trainer` does). So to make our custom changes, we need to modify the training loop inside the trainer.

Files of interest:

- `ppsdg/models/tabdiff/trainer.py`
  - `run_loop`
  - `_run_step` ( this will be where the grad clip/noise injection should happen)
- `ppsdg/models/tabdiff/models/unified_ctime_diffusion.py`
  - `mixed_loss` (this is the top level loss function)
  - `_edm_loss` (continuous loss).


things of note:

- The loss function might need some modifications.
  - contiuous: problematic? it's an unbounded L2 based loss. 
    - But if we can scale our input numerical features with something that sets hard boundaries (like minmax), then it might be ok.
  - discrete: probably safe, looks like cross-entropy

- The final model is not actually directly updated by the optimizer. The optimizer updates a "surrogate" model. This updated surrogate model is then used to compute a weighted average with the state of the real model to update the state of the real model (exponential moving average).
  - [This is what the Adam paper author has to say](https://www.reddit.com/r/MachineLearning/comments/ucflc2/d_understanding_the_use_of_ema_in_diffusion_models/i6a82p3/)
  - interesting that this was originally used to boost SGD performance (_kind_ of simlar to momentum) but now it's used in Adam and diffusion models.
  - But this is fine for DP, since we **only** look at the surrogate model for the new weights (and not the data directly). This can be verified by looking at where they call `update_ema` in `trainer.py`.
