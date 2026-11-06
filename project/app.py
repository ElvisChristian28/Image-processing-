"""
Foreground Object Extraction Tool - Streamlit App

A web-based tool for extracting foreground objects from images using
thresholding + GrabCut segmentation.
"""

import streamlit as st
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
import json

# Import local modules
from src.utils import (
    bytes_to_image,
    resize_if_needed,
    scale_mask_to_original,
    image_to_bytes,
    create_comparison_grid,
    ensure_bgr,
)
from src.threshold import threshold_pipeline
from src.grabcut_seg import grabcut_pipeline
from src.extract import (
    extract_foreground,
    crop_to_foreground,
    composite_on_background,
    create_transparent_from_alpha,
)
from src import ai_segment


# Page configuration
st.set_page_config(
    page_title="Foreground Object Extraction Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        "original_image": None,
        "processed_image": None,
        "scale_factor": 1.0,
        "thresh_mask": None,
        "grabcut_mask": None,
        "final_mask": None,
        "outputs": None,
        "alpha": None,  # soft alpha matte (AI mode)
        "rect": None,
        "canvas_key": 0,  # For forcing canvas reset
        "last_uploaded_file_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def parse_canvas_rect(canvas_data: Optional[Dict]) -> Optional[Tuple[int, int, int, int]]:
    """
    Parse rectangle from streamlit-drawable-canvas JSON data.
    
    Returns:
        (x, y, w, h) tuple or None if no rectangle drawn
    """
    if canvas_data is None:
        return None
    
    try:
        objects = canvas_data.get("objects", [])
        for obj in objects:
            if obj.get("type") == "rect":
                left = int(obj.get("left", 0))
                top = int(obj.get("top", 0))
                width = int(obj.get("width", 0))
                height = int(obj.get("height", 0))
                
                # Only return valid rectangles
                if width > 10 and height > 10:
                    return (left, top, width, height)
    except Exception:
        pass
    
    return None


def request_reprocess():
    """Widget callback: flag a re-run (the heavy work happens in main, not in the callback)."""
    st.session_state["needs_reprocess"] = True


def is_ai_mode() -> bool:
    """True when the AI (rembg) mode is selected in the sidebar."""
    default = "AI (neural network)" if ai_segment.is_available() else "Classical (Threshold + GrabCut)"
    return st.session_state.get("mode", default) == "AI (neural network)"


def process_image_ai():
    """AI pipeline: neural-network alpha matte -> outputs (no threshold / GrabCut)."""
    original = st.session_state.original_image
    with st.spinner("Running AI background removal (first run downloads the model)..."):
        alpha, mask = ai_segment.ai_pipeline(
            original,
            model=st.session_state.get("ai_model", "people"),
        )
    st.session_state.alpha = alpha
    st.session_state.thresh_mask = None
    st.session_state.grabcut_mask = None
    st.session_state.final_mask = mask

    outputs = extract_foreground(original, mask)
    # Use the soft matte for the transparent PNG (smoother hair/edges than feathering)
    outputs["transparent"] = create_transparent_from_alpha(original, alpha)
    st.session_state.outputs = outputs


def process_image():
    """Run the full extraction pipeline on the current processed image."""
    if st.session_state.processed_image is None:
        return

    try:
        if is_ai_mode():
            process_image_ai()
            return
        st.session_state.alpha = None

        # Step 1: Thresholding
        with st.spinner("Running thresholding..."):
            thresh_mask = threshold_pipeline(
                st.session_state.processed_image,
                method=st.session_state.get("threshold_method", "otsu"),
                blur_ksize=st.session_state.get("blur_ksize", 5),
                adaptive_block_size=st.session_state.get("adaptive_block_size", 11),
                adaptive_c=st.session_state.get("adaptive_c", 2),
            )
        st.session_state.thresh_mask = thresh_mask

        # Step 2: GrabCut
        with st.spinner("Running GrabCut segmentation..."):
            rect = st.session_state.get("rect")
            grabcut_mask = grabcut_pipeline(
                st.session_state.processed_image,
                thresh_mask,
                rect=rect,
                iterations=st.session_state.get("grabcut_iterations", 5),
                use_rect_fallback=st.session_state.get("use_rect_fallback", True),
            )
        st.session_state.grabcut_mask = grabcut_mask

        # Step 3: Scale mask to original resolution
        if st.session_state.scale_factor > 1.0:
            with st.spinner("Scaling mask to original resolution..."):
                final_mask = scale_mask_to_original(
                    grabcut_mask,
                    st.session_state.original_image.shape[:2],
                )
        else:
            final_mask = grabcut_mask
        st.session_state.final_mask = final_mask

        # Step 4: Extract foreground
        with st.spinner("Generating outputs..."):
            outputs = extract_foreground(st.session_state.original_image, final_mask)
        st.session_state.outputs = outputs

    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        st.exception(e)


def handle_file_upload(uploaded_file):
    """Handle new file upload and reset state."""
    if uploaded_file is None:
        return
    
    # Check if this is a new file
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.last_uploaded_file_id == file_id:
        return  # Same file, don't reprocess
    
    st.session_state.last_uploaded_file_id = file_id
    
    # Reset state for new image
    st.session_state.original_image = None
    st.session_state.processed_image = None
    st.session_state.scale_factor = 1.0
    st.session_state.thresh_mask = None
    st.session_state.grabcut_mask = None
    st.session_state.final_mask = None
    st.session_state.outputs = None
    st.session_state.alpha = None
    st.session_state.rect = None
    st.session_state.canvas_key += 1  # Force canvas reset
    
    try:
        # Load and validate image
        file_bytes = uploaded_file.read()
        image = bytes_to_image(file_bytes)
        image = ensure_bgr(image)  # Ensure 3-channel BGR
        
        # Resize if needed for processing
        processed, scale = resize_if_needed(image, max_dimension=1024)
        
        st.session_state.original_image = image
        st.session_state.processed_image = processed
        st.session_state.scale_factor = scale
        
        # Auto-run pipeline
        process_image()
        
        st.success(f"Image loaded: {image.shape[1]}x{image.shape[0]} "
                   f"({'resized' if scale > 1.0 else 'original size'})")
        
    except Exception as e:
        st.error(f"Failed to load image: {str(e)}")
        st.session_state.original_image = None


def render_sidebar():
    """Render sidebar controls."""
    with st.sidebar:
        st.header("🧠 Mode")
        ai_ok = ai_segment.is_available()
        options = ["AI (neural network)", "Classical (Threshold + GrabCut)"]
        if not ai_ok:
            # Must be set BEFORE the widget is created
            st.session_state["mode"] = options[1]
        st.radio(
            "Segmentation method",
            options,
            key="mode",
            index=0 if ai_ok else 1,
            on_change=request_reprocess,
            help="AI works on real photos (people, objects, busy backgrounds). "
                 "Classical works best on simple, high-contrast images.",
        )
        if not ai_ok:
            st.warning("AI mode needs `rembg`: `pip install rembg onnxruntime`. Using Classical.")
        if is_ai_mode():
            st.selectbox(
                "AI model",
                list(ai_segment.AI_MODELS.keys()),
                key="ai_model",
                format_func=lambda k: {"people": "People / portraits", "general": "General objects"}[k],
                on_change=request_reprocess,
                help="Runs locally. No API key needed; the model downloads once (~170 MB).",
            )
        st.divider()

        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            help="Upload an image to extract foreground objects",
        )
        handle_file_upload(uploaded_file)
        
        if st.session_state.original_image is None:
            st.info("👆 Upload an image to get started")
            return
        
        if is_ai_mode():
            st.divider()
            st.header("▶️ Actions")
            if st.button("🔄 Re-run AI", type="primary", use_container_width=True):
                process_image()
                st.rerun()
            return

        st.divider()
        st.header("🔧 Threshold Settings")
        
        # Threshold method
        threshold_method = st.selectbox(
            "Method",
            ["otsu", "adaptive_mean", "adaptive_gaussian"],
            format_func=lambda x: {
                "otsu": "Otsu's Method",
                "adaptive_mean": "Adaptive Mean",
                "adaptive_gaussian": "Adaptive Gaussian",
            }[x],
            key="threshold_method",
            help="Thresholding algorithm to create initial mask",
        )
        
        # Blur kernel size (odd values only)
        blur_ksize = st.slider(
            "Blur Kernel Size",
            min_value=3,
            max_value=31,
            value=5,
            step=2,
            key="blur_ksize",
            help="Gaussian blur kernel size (must be odd)",
        )
        
        # Adaptive threshold parameters (only shown for adaptive methods)
        if threshold_method in ["adaptive_mean", "adaptive_gaussian"]:
            adaptive_block_size = st.slider(
                "Adaptive Block Size",
                min_value=3,
                max_value=51,
                value=11,
                step=2,
                key="adaptive_block_size",
                help="Neighborhood size for adaptive threshold (must be odd)",
            )
            adaptive_c = st.slider(
                "Adaptive C",
                min_value=0,
                max_value=20,
                value=2,
                key="adaptive_c",
                help="Constant subtracted from mean/weighted mean",
            )
        else:
            # Still need keys for session state
            st.session_state.adaptive_block_size = 11
            st.session_state.adaptive_c = 2
        
        st.divider()
        st.header("✂️ GrabCut Settings")
        
        grabcut_iterations = st.slider(
            "GrabCut Iterations",
            min_value=1,
            max_value=20,
            value=5,
            key="grabcut_iterations",
            help="Number of GrabCut refinement iterations",
        )
        
        use_rect_fallback = st.checkbox(
            "Use Rectangle Fallback",
            value=True,
            key="use_rect_fallback",
            help="If threshold mask is empty/full, use drawn rectangle instead",
        )
        
        st.divider()
        st.header("▶️ Actions")
        
        if st.button("🔄 Re-run Pipeline", type="primary", use_container_width=True):
            process_image()
            st.rerun()
        
        if st.button("🗑️ Clear Rectangle", use_container_width=True):
            st.session_state.rect = None
            st.session_state.canvas_key += 1
            process_image()
            st.rerun()
        
        if st.button("🔄 Reset All", use_container_width=True):
            for key in ["original_image", "processed_image", "scale_factor",
                       "thresh_mask", "grabcut_mask", "final_mask", "outputs", "alpha", "rect"]:
                st.session_state[key] = None if key != "scale_factor" else 1.0
            st.session_state.last_uploaded_file_id = None
            st.session_state.canvas_key += 1
            st.rerun()


def render_canvas():
    """Render the rectangle drawing canvas."""
    if st.session_state.processed_image is None:
        return None
    
    # Import streamlit-drawable-canvas
    try:
        from streamlit_drawable_canvas import st_canvas
    except ImportError:
        st.error("⚠️ `streamlit-drawable-canvas` not installed. Run: `pip install streamlit-drawable-canvas`")
        return None
    
    # Convert processed image to RGB for canvas (it expects RGB)
    canvas_image = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
    h, w = canvas_image.shape[:2]
    
    # Create canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.1)",  # Semi-transparent red fill
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=None,
        background_color=None,
        update_streamlit=True,
        height=h,
        width=w,
        drawing_mode="rect",
        point_display_radius=0,
        key=f"canvas_{st.session_state.canvas_key}",
        initial_drawing=None,
    )
    
    return canvas_result


def get_foreground_only() -> Optional[np.ndarray]:
    """Cropped BGRA foreground (transparent background); uses the soft AI matte when available."""
    if st.session_state.outputs is None or st.session_state.final_mask is None:
        return None
    return crop_to_foreground(
        st.session_state.original_image,
        st.session_state.final_mask,
        alpha=st.session_state.alpha,
    )


def render_results():
    """Render the side-by-side comparison and download buttons."""
    if st.session_state.original_image is None:
        return
    
    st.header("📊 Results")
    
    # Prepare images for comparison grid
    # Original (resized for display if needed)
    display_orig = st.session_state.processed_image
    
    # Threshold mask visualization (3-channel)
    thresh_vis = None
    if st.session_state.thresh_mask is not None:
        thresh_vis = cv2.cvtColor(st.session_state.thresh_mask, cv2.COLOR_GRAY2BGR)
    
    # GrabCut mask visualization
    grabcut_vis = None
    if st.session_state.grabcut_mask is not None:
        grabcut_vis = cv2.cvtColor(st.session_state.grabcut_mask, cv2.COLOR_GRAY2BGR)
    
    # AI alpha matte visualization (AI mode only)
    alpha_vis = None
    if st.session_state.alpha is not None:
        alpha_vis = cv2.cvtColor(st.session_state.alpha, cv2.COLOR_GRAY2BGR)

    # Final image: ONLY the foreground (tight crop, transparent background)
    fg_final = get_foreground_only()

    # Build comparison grid
    images = []
    titles = []

    if display_orig is not None:
        images.append(display_orig)
        titles.append("Original")
    if alpha_vis is not None:
        images.append(alpha_vis)
        titles.append("AI Alpha Matte")
    if thresh_vis is not None:
        images.append(thresh_vis)
        titles.append("Threshold Mask")
    if grabcut_vis is not None:
        images.append(grabcut_vis)
        titles.append("GrabCut Mask")
    if fg_final is not None and fg_final.size > 0:
        images.append(composite_on_background(fg_final, (255, 255, 255)))
        titles.append("Final (Foreground Only)")

    if images:
        try:
            grid = create_comparison_grid(images, titles, cols=len(images), figsize=(4 * len(images), 4))
            st.image(grid, caption="Comparison: " + " | ".join(titles), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create comparison grid: {e}")
            # Fallback: show individual images
            cols = st.columns(len(images))
            for i, (img, title) in enumerate(zip(images, titles)):
                with cols[i]:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=title, use_container_width=True)
    
    # Download buttons
    if st.session_state.outputs is not None:
        st.subheader("💾 Download Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Transparent PNG
            transparent_bytes = image_to_bytes(st.session_state.outputs["transparent"], ".png")
            st.download_button(
                label="📥 Transparent PNG",
                data=transparent_bytes,
                file_name="foreground_transparent.png",
                mime="image/png",
                use_container_width=True,
                help="RGBA image with transparent background",
            )
        
        with col2:
            # Black Background
            black_bg_bytes = image_to_bytes(st.session_state.outputs["black_bg"], ".png")
            st.download_button(
                label="📥 Black Background",
                data=black_bg_bytes,
                file_name="foreground_black_bg.png",
                mime="image/png",
                use_container_width=True,
                help="BGR image with black background",
            )
        
        with col3:
            # Binary Mask
            mask_bytes = image_to_bytes(st.session_state.outputs["binary_mask"], ".png")
            st.download_button(
                label="📥 Binary Mask",
                data=mask_bytes,
                file_name="foreground_mask.png",
                mime="image/png",
                use_container_width=True,
                help="Binary mask visualization (white=foreground)",
            )
        
        with col4:
            # Foreground Only (cropped, no background)
            fg_only_bytes = image_to_bytes(get_foreground_only(), ".png")
            st.download_button(
                label="📥 Foreground Only",
                data=fg_only_bytes,
                file_name="foreground_only.png",
                mime="image/png",
                use_container_width=True,
                help="Cropped foreground object, transparent background",
            )
    
    # Mask statistics
    if st.session_state.final_mask is not None:
        with st.expander("📈 Mask Statistics"):
            mask = st.session_state.final_mask
            fg_pixels = np.count_nonzero(mask)
            total_pixels = mask.size
            fg_percent = (fg_pixels / total_pixels) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Foreground Pixels", f"{fg_pixels:,}")
            col2.metric("Background Pixels", f"{total_pixels - fg_pixels:,}")
            col3.metric("Foreground %", f"{fg_percent:.1f}%")


def main():
    """Main Streamlit app entry point."""
    # Custom CSS
    st.markdown("""
        <style>
        .stButton > button {
            width: 100%;
        }
        .main > div {
            padding-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("🎯 Foreground Object Extraction Tool")
    st.caption("AI background removal, or classical Thresholding + GrabCut segmentation")
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()

    # Mode / model changed in the sidebar -> re-run on the current image
    if st.session_state.pop("needs_reprocess", False) and st.session_state.original_image is not None:
        process_image()
    
    # Main content area
    if st.session_state.original_image is None:
        st.info("👈 Upload an image from the sidebar to begin")
        
        # Show usage instructions
        with st.expander("📖 How to Use"):
            st.markdown("""
            **AI mode (default, best for photos):** upload an image and the background is removed
            automatically. Choose *People* for portraits or *General* for objects. Runs locally,
            no API key needed.

            **Classical mode:**

            1. **Upload an image** using the file uploader in the sidebar
            2. **Adjust threshold settings** to create a good initial mask:
               - Try **Otsu's method** first (automatic)
               - For uneven lighting, use **Adaptive Mean/Gaussian**
               - Increase **Blur Kernel** to reduce noise
            3. **Draw a rectangle** on the image to guide GrabCut (optional but recommended)
            4. **Adjust GrabCut iterations** for refinement quality
            5. Click **Re-run Pipeline** to update results
            6. **Download** your results in three formats:
               - Transparent PNG (for compositing)
               - Black Background (for viewing)
               - Binary Mask (for further processing)
            """)
        return
    
    if is_ai_mode():
        # AI mode needs no rectangle or threshold tuning
        render_results()
        return

    # Classical mode: canvas for the guiding rectangle + results
    tab1, tab2 = st.tabs(["🎨 Draw Rectangle", "📊 Results"])

    with tab1:
        st.subheader("Draw Rectangle (Optional)")
        st.caption("Draw a rectangle around the foreground object to guide GrabCut segmentation.")

        canvas_result = render_canvas()

        # Parse rectangle from canvas
        if canvas_result is not None:
            rect = parse_canvas_rect(canvas_result.json_data)
            if rect is not None:
                st.session_state.rect = rect
                st.success(f"Rectangle: x={rect[0]}, y={rect[1]}, w={rect[2]}, h={rect[3]}")
            elif st.session_state.rect is not None:
                # Keep existing rect if no new one drawn
                pass

    with tab2:
        render_results()


if __name__ == "__main__":
    main()