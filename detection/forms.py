# detection/forms.py
from django import forms

MODEL_CHOICES = [
    ("models/cnn_model_v1.keras", "v1"),
    ("models/cnn_model_v2.keras", "v2"),
    ("models/cnn_model_v3.keras", "v3"),
]


# Form for image upload
class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        label="Eye image",
        required=False,
        widget=forms.ClearableFileInput(attrs={"class": "form-control"})
    )

    prev_image = forms.CharField(required=False, widget=forms.HiddenInput())

    # NEW ─ stores original filename across submits
    prev_filename = forms.CharField(required=False, widget=forms.HiddenInput())

    model = forms.ChoiceField(
        label="Select model",
        choices=MODEL_CHOICES,
        initial="models/cnn_model_v3.keras",
        widget=forms.Select(attrs={"class": "form-select"})
    )

    def clean(self):
        """Either a new file or a previous one must be present."""
        cleaned = super().clean()
        if not cleaned.get("image") and not cleaned.get("prev_image"):
            raise forms.ValidationError("Please choose an image file.")
        return cleaned
