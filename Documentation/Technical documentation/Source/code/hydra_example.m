_target_: src.models.audio_model_lightning.AudioLitModule

net:
  _target_: src.models.components.cnn_spectogram_net.CNNSpectogramNet
  no_classes: ${data.general.no_classes}

optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.0002
  weight_decay: 0.00001

scheduler:
  _target_: transformers.get_polynomial_decay_schedule_with_warmup
  _partial_: true
  lr_end: 0.000002
  power: 0.7

scheduler_warmup_percentage: 0.05
no_classes: ${data.general.no_classes}
threshold_value: 0.5
aggregation_function: S2
