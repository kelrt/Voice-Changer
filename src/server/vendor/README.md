# Vendored: Applio RVC model code

`rvc/lib/algorithm/` is copied unmodified from [IAHispano/Applio](https://github.com/IAHispano/Applio)
at commit `59c113106c0dfff647324ab1a80c286f17b51929` (MIT, see `LICENSE-Applio`), minus the
training-only `discriminators.py`.

It defines the RVC v2 `Synthesizer` network (text encoder, normalizing flow, HiFi-GAN NSF vocoder).
`server.py` loads trained `.pth` checkpoints into it. Using the exact architecture the model was
trained with is what makes every checkpoint weight load; nothing is guessed.
