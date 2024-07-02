import streamlit as st
import streamlit.components.v1 as components
import torch
import io
import base64
from PIL import Image
from torchvision import transforms
from gtts import gTTS
from utils import EncoderDecoder, dataset


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to generate captions from image
def generate_caption(image):
    # Load the saved PyTorch model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EncoderDecoder(
        embed_size=300,
        vocab_size=len(dataset.vocab), 
        attention_dim=256,
        encoder_dim=2048,
        decoder_dim=512
    ).to(device)
    model_state = torch.load('./attention_model_state.pth', map_location=torch.device('cpu'))
    model.load_state_dict(model_state['state_dict'])
    model.eval()
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encoder(image_tensor.to(device))
        captions, alphas = model.decoder.generate_caption(features, vocab=dataset.vocab)
        caption = ' '.join(captions)
    return caption, alphas

# Function to generate audio from caption
def generate_audio(caption):
    print(caption)
    tts = gTTS(text=caption, lang='en', slow=False)
    audio_path = "output.mp3"
    tts.save(audio_path)
    return audio_path

def main():
    st.title("Image Captioning and Speech Generation")
    def generate_caption_and_audio(image):
        # Generate and display caption
        caption, _ = generate_caption(image)
        caption = caption.replace('<EOS>', '').strip()
        st.write("Generated Caption:", caption)

        # Generate audio from the caption
        audio_file = generate_audio(caption)

        if st.button("Play Audio"):
                st.audio(audio_file, format='audio/mp3')

    # Function to upload image from device
    def upload_image():
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            generate_caption_and_audio(image)

    # Function to capture live photo
    def capture_photo():
        image_file = st.camera_input("Capture your surroundings:")
        if image_file is not None:
            image = Image.open(image_file)
            st.image(image, caption='Captured Image', use_column_width=True)
            generate_caption_and_audio(image)


    option = st.radio("Choose an option:", ("Upload Image", "Capture Photo"))

    if option == "Upload Image":
        upload_image()
    elif option == "Capture Photo":
        capture_photo()

if __name__ == "__main__":
    main()