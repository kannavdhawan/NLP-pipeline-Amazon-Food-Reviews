import keras
reconstructed_model = keras.models.load_model("data/ae.model")
print(reconstructed_model.summary())


