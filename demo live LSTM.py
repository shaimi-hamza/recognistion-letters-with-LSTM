import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


# Charger votre modèle
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("./modele_Version_Finale4.keras")
    return model
characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

model = load_model()

# Configuration de l'application
st.title("Reconnaissance de lettres manuscrites")
st.write("Dessinez une lettre dans la zone ci-dessous et laissez le modèle la prédire.")

# Zone de dessin
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  # Couleur de remplissage
    stroke_width=10,                    # Épaisseur du pinceau
    stroke_color="black",               # Couleur du pinceau
    background_color="white",           # Couleur de fond
    width=200,                          # Largeur du canevas
    height=200,                         # Hauteur du canevas
    drawing_mode="freedraw",            # Mode dessin
    key="canvas",
)

# Si un dessin est fait
if canvas_result.image_data is not None:
    # Prétraitement de l'image
    image = Image.fromarray((canvas_result.image_data).astype("uint8"))
    image = image.convert("L")  # Convertir en niveau de gris
    image = ImageOps.invert(image)  # Inverser les couleurs
    image = image.resize((28, 28))# Redimensionner pour correspondre à l'entrée du modèle
    image2 = image
    image = ImageOps.mirror(image) 
    image = image.rotate(90)
    image_array = np.array(image) / 255.0  # Normaliser
    image_array = image_array.reshape(1, 28, 28, 1)  # Ajouter les dimensions nécessaires

    # Afficher l'image prétraitée
    st.image(image2, caption="Image prétraitée", width=150)

    # Prédire la lettre
    
    prediction = model.predict(image_array)
    confidence_score = np.max(prediction)  # Score de confiance (probabilité maximale)
    # predicted_letter = chr(np.argmax(prediction) + 65)  # Convertir l'index en lettre
    predicted_index = np.argmax(prediction)  # Trouver l'indice de la classe prédite
    predicted_character = characters[predicted_index]  # Trouver le caractère correspondant

    # Afficher la prédiction
    st.subheader(f"(Lettre ou chiffre) prédite : {predicted_character}")
    # Ajouter la précision
    st.write(f"Précision du modèle : {confidence_score:.2%}")

