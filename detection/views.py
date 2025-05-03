import os
import base64
from io import BytesIO
import numpy as np
from django.conf import settings
from django.shortcuts import render
from .forms import ImageUploadForm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Cache for already‑loaded Keras models
MODEL_CACHE = {}


def get_model(model_filename: str):
    """Load and cache a Keras model by filename."""
    if model_filename not in MODEL_CACHE:
        path = os.path.join(settings.BASE_DIR, model_filename)
        MODEL_CACHE[model_filename] = load_model(path, compile=False)
    return MODEL_CACHE[model_filename]


def preprocess_image_from_file(file_obj, target_size=(224, 224)):
    """Resize to target_size, normalize to 0‑1 and add batch dimension."""
    img = Image.open(file_obj).convert("RGB")
    img = img.resize(target_size)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def index(request):
    result = None
    image_data_uri = None
    uploaded_name = None  # what will be shown next to “Selected:”
    prev_image_b64 = ""
    prev_filename = ""  # NEW

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            if form.cleaned_data["image"]:
                # New file picked
                uploaded_file = form.cleaned_data["image"]
                uploaded_name = uploaded_file.name
                file_bytes = uploaded_file.read()
            else:
                # Re‑use previous file
                uploaded_name = form.cleaned_data["prev_filename"] or "previous_image"
                file_bytes = base64.b64decode(form.cleaned_data["prev_image"])

            prev_filename = uploaded_name  # keep for hidden field
            prev_image_b64 = base64.b64encode(file_bytes).decode()
            image_data_uri = f"data:image/jpeg;base64,{prev_image_b64}"

            x = preprocess_image_from_file(BytesIO(file_bytes))
            model = get_model(form.cleaned_data["model"])
            result = float(model.predict(x)[0][0]) * 100
    else:
        form = ImageUploadForm()

    return render(
        request,
        "detection/index.html",
        {
            "form": form,
            "result": result,
            "image_data_uri": image_data_uri,
            "uploaded_name": uploaded_name,
            "prev_image_b64": prev_image_b64,
            "prev_filename": prev_filename,
        },
    )
