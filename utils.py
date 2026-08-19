"""Image validation, preprocessing, and presentation helpers."""

from io import BytesIO

from PIL import Image, UnidentifiedImageError
from torchvision import transforms


IMAGE_SIZE = 224
MAX_UPLOAD_MB = 10
DEFAULT_CONFIDENCE_THRESHOLD = 70.0
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

DISEASE_INFO = {
    "benign": {
        "summary": "The model classifies this image as benign, meaning it does not match the malignant training category.",
        "guidance": "Continue routine skin checks and seek clinical advice for a lesion that changes, bleeds, or does not heal.",
    },
    "malignant": {
        "summary": "The model identifies visual patterns associated with the malignant training category.",
        "guidance": "Arrange prompt evaluation by a qualified dermatologist. This model output is not a diagnosis.",
    },
    "non-neoplastic": {
        "summary": "The model classifies this image as non-neoplastic, meaning it does not match the benign or malignant training categories.",
        "guidance": "A clinician should assess persistent, painful, or changing skin findings even when model confidence is high.",
    },
}


def load_image(uploaded_file) -> Image.Image:
    """Decode an uploaded file and return an RGB PIL image."""
    if uploaded_file is None:
        raise ValueError("Choose an image before starting the analysis.")
    if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Images must be smaller than {MAX_UPLOAD_MB} MB.")
    try:
        image = Image.open(BytesIO(uploaded_file.getvalue()))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The selected file is not a valid JPG, JPEG, or PNG image.") from exc


def preprocess_image(image: Image.Image):
    return IMAGE_TRANSFORM(image).unsqueeze(0)


def display_name(label: str) -> str:
    return label.replace("-", " ").title()


def disease_info(label: str) -> dict[str, str]:
    return DISEASE_INFO.get(label, {
        "summary": "No description is available for this model category.",
        "guidance": "Discuss any concerning skin finding with a qualified clinician.",
    })
