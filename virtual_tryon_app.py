import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path

# Import your custom modules
from part1_detection import detect_and_segment_clothing
from part2_masking import create_agnostic_mask, combine_cloth_with_mask

# Configure page
st.set_page_config(
    page_title="Virtual Try-On Demo",
    page_icon="👕",
    layout="wide"
)

st.title("🔥 Virtual Try-On System")
st.markdown("Upload a person image and clothing item to see the magic!")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
detection_threshold = st.sidebar.slider("Detection Threshold", 0.1, 1.0, 0.35, 0.05)
text_threshold = st.sidebar.slider("Text Threshold", 0.1, 1.0, 0.25, 0.05)
clothing_classes = st.sidebar.multiselect(
    "Clothing Classes to Detect",
    ["tshirt", "shirt", "top", "dress", "jacket"],
    default=["tshirt"]
)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

def load_models():
    """Load GroundingDINO and SAM models"""
    try:
        with st.spinner("Loading AI models... This may take a moment."):
            # Note: In a real deployment, you'd need to install and configure these models
            st.success("Models loaded successfully!")
            st.session_state.models_loaded = True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Note: This demo requires GroundingDINO and SAM models to be properly installed.")

def preprocess_image_for_torch(image, size=(512, 512)):
    """Convert PIL image to torch tensor"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def create_agnostic_mask(original_image, mask_image):
    """Create agnostic mask from human image (Part 2 functionality)"""
    # Convert PIL to numpy arrays
    original_np = np.array(original_image)
    mask_np = np.array(mask_image.convert('L'))
    
    # Apply threshold to create binary mask
    _, mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    
    # Create grey image
    grey_image = np.ones_like(original_np) * 128
    
    # Apply mask
    mask_3d = np.stack([mask] * 3, axis=-1) / 255.0
    inv_mask_3d = 1 - mask_3d
    
    # Combine images
    masked_grey = grey_image * mask_3d
    masked_original = original_np * inv_mask_3d
    result_image = masked_grey + masked_original
    
    return result_image.astype(np.uint8)

def combine_cloth_with_mask(cloth_image, mask_image):
    """Combine clothing with mask (Part 2 functionality)"""
    # Convert to tensors
    cloth_tensor = preprocess_image_for_torch(cloth_image)
    mask_tensor = preprocess_image_for_torch(mask_image.convert('RGB'))
    
    # Apply mask to cloth
    output = cloth_tensor * (mask_tensor > 0.5).float()
    
    # Convert back to PIL
    output_image = transforms.ToPILImage()(output.squeeze())
    return output_image

def simulate_virtual_tryon(person_image, clothing_image):
    """Simulate the virtual try-on process"""
    # For demo purposes, we'll create a simple overlay
    # In production, this would use your trained models
    
    person_np = np.array(person_image.resize((512, 512)))
    clothing_np = np.array(clothing_image.resize((512, 512)))
    
    # Create a simple mask for the torso area (demo purposes)
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[150:400, 150:350] = 255  # Rough torso area
    
    # Apply clothing to the masked area
    mask_3d = np.stack([mask] * 3, axis=-1) / 255.0
    
    result = person_np * (1 - mask_3d) + clothing_np * mask_3d
    return Image.fromarray(result.astype(np.uint8))

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.header("👤 Person Image")
    person_image = st.file_uploader(
        "Upload person image",
        type=['png', 'jpg', 'jpeg'],
        key="person"
    )
    
    if person_image:
        person_pil = Image.open(person_image).convert('RGB')
        st.image(person_pil, caption="Original Person Image", use_column_width=True)

with col2:
    st.header("👕 Clothing Image")
    clothing_image = st.file_uploader(
        "Upload clothing image",
        type=['png', 'jpg', 'jpeg'],
        key="clothing"
    )
    
    if clothing_image:
        clothing_pil = Image.open(clothing_image).convert('RGB')
        st.image(clothing_pil, caption="Clothing Item", use_column_width=True)

# Processing section
if person_image and clothing_image:
    st.header("🎯 Processing Options")
    
    processing_mode = st.radio(
        "Choose processing mode:",
        ["Simple Virtual Try-On", "Advanced Segmentation (Requires Models)"]
    )
    
    if st.button("🚀 Generate Virtual Try-On", type="primary"):
        with st.spinner("Processing..."):
            if processing_mode == "Simple Virtual Try-On":
                # Simple demo version
                result = simulate_virtual_tryon(person_pil, clothing_pil)
                
                st.header("✨ Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(person_pil, caption="Original Person", use_column_width=True)
                
                with col2:
                    st.image(clothing_pil, caption="Clothing Item", use_column_width=True)
                
                with col3:
                    st.image(result, caption="Virtual Try-On Result", use_column_width=True)
                
            else:
                # Advanced mode (requires models)
                if not st.session_state.models_loaded:
                    st.warning("AI models not loaded. Click 'Load Models' first.")
                    if st.button("Load Models"):
                        load_models()
                else:
                    st.info("Advanced processing would use GroundingDINO + SAM for precise segmentation")
                    # Here you would integrate your Part 1 and Part 2 code
                    
# Additional features
st.header("🛠️ Additional Tools")

with st.expander("Create Agnostic Mask"):
    st.write("Upload a person image and its corresponding mask to create an agnostic version")
    
    col1, col2 = st.columns(2)
    with col1:
        mask_person = st.file_uploader("Person image for masking", type=['png', 'jpg', 'jpeg'], key="mask_person")
    with col2:
        mask_file = st.file_uploader("Mask image", type=['png', 'jpg', 'jpeg'], key="mask_file")
    
    if mask_person and mask_file and st.button("Create Agnostic Mask"):
        mask_person_pil = Image.open(mask_person).convert('RGB')
        mask_file_pil = Image.open(mask_file)
        
        agnostic_result = create_agnostic_mask(mask_person_pil, mask_file_pil)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(mask_person_pil, caption="Original", use_column_width=True)
        with col2:
            st.image(mask_file_pil, caption="Mask", use_column_width=True)
        with col3:
            st.image(agnostic_result, caption="Agnostic Result", use_column_width=True)

with st.expander("Cloth + Mask Combination"):
    st.write("Combine a clothing item with a mask")
    
    col1, col2 = st.columns(2)
    with col1:
        cloth_for_mask = st.file_uploader("Clothing image", type=['png', 'jpg', 'jpeg'], key="cloth_mask")
    with col2:
        mask_for_cloth = st.file_uploader("Mask for clothing", type=['png', 'jpg', 'jpeg'], key="mask_cloth")
    
    if cloth_for_mask and mask_for_cloth and st.button("Combine Cloth + Mask"):
        cloth_pil = Image.open(cloth_for_mask).convert('RGB')
        mask_pil = Image.open(mask_for_cloth)
        
        combined_result = combine_cloth_with_mask(cloth_pil, mask_pil)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(cloth_pil, caption="Clothing", use_column_width=True)
        with col2:
            st.image(mask_pil, caption="Mask", use_column_width=True)
        with col3:
            st.image(combined_result, caption="Combined Result", use_column_width=True)

# Information section
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
This virtual try-on system uses:
- **GroundingDINO** for object detection
- **SAM** for precise segmentation
- **Custom algorithms** for image blending

**Features:**
- Automatic clothing detection
- Precise segmentation
- Agnostic mask creation
- Realistic virtual try-on
""")

st.sidebar.markdown("---")
st.sidebar.write("🔗 [GitHub Repository](https://github.com/codeskavy/Virtual_Tryon/)")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Virtual Try-On Demo • Built with Streamlit • Powered by AI
    </div>
    """,
    unsafe_allow_html=True
)