import streamlit as st
import cv2
import numpy as np
import os

# Define the save folder
SAVE_FOLDER = 'V:/NIS/'
os.makedirs(SAVE_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist

def xor_encrypt_decrypt(image, key):
    # Convert image and key to NumPy arrays
    image_np = np.array(image)
    key_np = np.array(key)
    
    # Resize the key to match the shape of the image
    key_np = np.resize(key_np, image_np.shape)
    
    # Perform XOR operation between image and key
    result = np.bitwise_xor(image_np, key_np)
    return result

def save_image(image, filename):
    # Save the image with the specified filename
    cv2.imwrite(filename, image)
    st.write(f"Image saved as {filename}")

def save_key(key, filename):
    # Save the key
    np.save(filename, key)
    st.write(f"Key saved as {filename}")

def load_key(filename):
    # Load the key from file
    return np.load(filename)

def main():
    # Customizing Streamlit's theme
    st.markdown(
        """
        <style>
            .sidebar .sidebar-content {
                background-color: #f0f2f6;
            }
            .sidebar .sidebar-content .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .sidebar .sidebar-content .block-container .block-element {
                margin-bottom: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Image Encryption and Decryption")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("", ("Encrypt", "Decrypt"))

    if selected_page == "Encrypt":
        st.header("Encryption")

        # Upload the image for encryption
        uploaded_file = st.file_uploader("Upload an image for encryption", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Read the uploaded image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
            
            # Convert BGR to RGB color space
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Define a secret key (should be the same length as the number of pixels in the image)
            key = np.random.randint(0, 256, size=image.shape, dtype=np.uint8)

            # Save the key
            key_path = os.path.join(SAVE_FOLDER, 'key.npy')
            save_key(key, key_path)

            # Encrypt the image
            encrypted_image = xor_encrypt_decrypt(image, key)

            # Convert encrypted image to RGB format
            encrypted_image_rgb = cv2.cvtColor(encrypted_image, cv2.COLOR_BGR2RGB)

            # Display original and encrypted images side by side
            st.subheader("Encryption Result")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_rgb, caption="Original Image")
            with col2:
                st.image(encrypted_image_rgb, caption="Encrypted Image")

            # Save the encrypted image
            encrypted_image_path = os.path.join(SAVE_FOLDER, 'encrypted_image.png')
            save_image(encrypted_image, encrypted_image_path)

    elif selected_page == "Decrypt":
        st.header("Decryption")

        # Upload the encrypted image for decryption
        uploaded_encrypted_image = st.file_uploader("Upload the encrypted image for decryption", type=["jpg", "jpeg", "png"])
        if uploaded_encrypted_image is not None:
            # Read the uploaded encrypted image
            encrypted_image = cv2.imdecode(np.frombuffer(uploaded_encrypted_image.read(), np.uint8), 1)

            # Ask for the key to decrypt the image
            uploaded_key = st.file_uploader("Upload the key for decryption", type=["npy"])
            if uploaded_key is not None:
                key = load_key(uploaded_key)
                decrypted_image = xor_encrypt_decrypt(encrypted_image, key)

                # Convert images to RGB format
                encrypted_image_rgb = cv2.cvtColor(encrypted_image, cv2.COLOR_BGR2RGB)
                decrypted_image_rgb = cv2.cvtColor(decrypted_image, cv2.COLOR_BGR2RGB)

                # Display images side by side
                st.subheader("Decryption Result")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(encrypted_image_rgb, caption="Encrypted Image")
                with col2:
                    st.image(decrypted_image_rgb, caption="Decrypted Image")

                # Save the decrypted image with the specified filename
                decrypted_image_path = os.path.join(SAVE_FOLDER, 'decrypted_image.png')
                save_image(decrypted_image, decrypted_image_path)

if __name__ == "__main__":
    main()
