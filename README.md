# DermaGuard AI

DermaGuard AI is a research prototype for three-class skin lesion image classification. It combines a PyTorch ResNet18 model with Grad-CAM visual explanations so a reviewer can see both the predicted category and the image regions that influenced the prediction.

The project is designed for a hackathon demo and academic evaluation. It is not a medical diagnostic device.

## Features

- FastAPI inference API with CPU fallback and CUDA support.
- ResNet18 classification for `benign`, `malignant`, and `non-neoplastic` classes.
- Image validation for JPG, JPEG, and PNG uploads up to 10 MB.
- Grad-CAM heatmaps and overlays served as browser-accessible images.
- Responsive HTML/CSS/JavaScript demo interface.
- Class-balanced Improved V2 training with `WeightedRandomSampler`.
- Accuracy, precision, recall, F1, confusion matrices, and Fitzpatrick group analysis.
- Streamlit interface for an alternative local demo workflow.

## Dataset

Training uses Fitzpatrick17k-derived CSV splits:

- `datasets/fitzpatrick17k/train.csv`
- `datasets/fitzpatrick17k/validation.csv`
- `datasets/fitzpatrick17k/test.csv`

The dataset contains skin-tone metadata through `fitzpatrick_scale`. The project reports group-level metrics to expose performance differences, but these results should not be interpreted as clinical fairness certification.

## Model Architecture

The classifier is a torchvision ResNet18 initialized from ImageNet weights during training. Its final classifier is replaced with a three-output layer. Images are resized to `224 x 224`, converted to tensors, and normalized with ImageNet mean and standard deviation.

The production checkpoint is:

```text
models/resnet18_best.pth
```

Improved V2 training uses a `WeightedRandomSampler` to increase minority-class exposure during training. Validation and test splits remain deterministic and are not resampled.

## Grad-CAM Explanation

Grad-CAM uses gradients from the final convolutional layer to produce a class-specific activation map. The CAM is normalized to `0-255`, resized to the original image dimensions, colored with OpenCV `COLORMAP_JET`, and blended with the original image at `alpha=0.45`.

Warm colors indicate regions that contributed more strongly to the model prediction. Cooler colors contributed less. Grad-CAM is an interpretability aid, not proof that the model uses clinically valid reasoning.

## Screenshots

Generated visual examples are stored in:

```text
reports/gradcam_examples/
```

The repository also contains generated training and fairness plots under `reports/improved_v2/`.

## Local Installation

Use Python 3.11 and place the model checkpoint in `models/resnet18_best.pth`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### FastAPI backend

```powershell
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

The API is available at `http://127.0.0.1:8000`. Swagger documentation is available at `/docs`.

### Frontend

Serve `frontend/` with any static file server, for example:

```powershell
python -m http.server 5500 --directory frontend
```

Open `http://127.0.0.1:5500` after starting the backend.

To point the frontend at a deployed API, define `window.DERMAGUARD_API_URL` before `script.js` loads, or update the default in `frontend/script.js`:

```html
<script>window.DERMAGUARD_API_URL = "https://your-api.onrender.com";</script>
<script src="script.js"></script>
```

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | API status message |
| `GET` | `/health` | Health check |
| `GET` | `/verify-static` | Verify generated report directory and sample files |
| `POST` | `/predict` | Classify an uploaded image |
| `POST` | `/gradcam` | Classify an image and generate Grad-CAM assets |
| `GET` | `/reports/<filename>` | Serve generated Grad-CAM images |

Example `/gradcam` response:

```json
{
  "prediction": "benign",
  "confidence": 97.72,
  "heatmap": "/reports/example_heatmap.jpg",
  "overlay": "/reports/example_overlay.jpg"
}
```

## Render Deployment

Deploy the FastAPI service as a Render Web Service:

1. Push the repository to GitHub.
2. Create a Render Web Service from the repository.
3. Use Python 3.11, or let Render use `runtime.txt`.
4. Set the build command:

	```text
	pip install -r requirements.txt
	```

5. Set the start command:

	```text
	uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
	```

6. Ensure `models/resnet18_best.pth` is available in the deployment image or download it during the build from a protected artifact store.
7. Host the `frontend/` directory using a static host and set `window.DERMAGUARD_API_URL` to the Render API URL.

The frontend and API must allow cross-origin requests. The current API has permissive CORS for hackathon use; restrict `allow_origins` to the deployed frontend domain before production use.

## Docker

The included Dockerfile runs the Streamlit interface:

```powershell
docker build -t dermaguard-ai .
docker run --rm -p 8501:8501 -v "${PWD}\models:/app/models:ro" dermaguard-ai
```

For the FastAPI service, use:

```powershell
docker run --rm -p 8000:8000 dermaguard-ai uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

## Limitations

- Model confidence is not calibrated clinical probability.
- Dataset labels and metadata may contain noise and incomplete skin-tone annotations.
- Fitzpatrick group performance is limited by the number and distribution of samples in each group.
- Grad-CAM highlights model sensitivity, not verified clinical evidence.
- The system has not been approved for diagnosis, triage, or treatment decisions.
- Uploaded images and generated reports require an explicit retention policy for real deployment.

## Future Work

- Calibrate confidence scores and report uncertainty.
- Evaluate with dermatologist-reviewed external datasets.
- Improve performance for underrepresented skin-tone groups.
- Add automated model and data drift monitoring.
- Add authenticated storage with controlled report retention.
- Serve the frontend and API through a unified production origin.

## License and Responsible Use

Use this project for research, education, and controlled demonstrations. Do not use its output as a substitute for evaluation by a qualified healthcare professional.
