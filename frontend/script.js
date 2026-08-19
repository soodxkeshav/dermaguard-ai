const API_BASE_URL = window.DERMAGUARD_API_URL || "http://127.0.0.1:8000";
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ALLOWED_EXTENSIONS = new Set(["jpg", "jpeg", "png"]);
const ALLOWED_MIME_TYPES = new Set(["image/jpeg", "image/png"]);

const fileInput = document.getElementById("imageInput");
const fileName = document.getElementById("fileName");
const analyzeButton = document.getElementById("analyzeButton");
const resetButton = document.getElementById("resetButton");
const statusMessage = document.getElementById("statusMessage");
const results = document.getElementById("result");
const originalImage = document.getElementById("originalImage");
const heatmapImage = document.getElementById("heatmapImage");
const overlayImage = document.getElementById("overlayImage");
const predictionValue = document.getElementById("predictionValue");
const confidenceValue = document.getElementById("confidenceValue");

let previewUrl = null;
let generationTimer = null;

function setStatus(message, type = "") {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`.trim();
}

function validateFile(file) {
    const extension = file.name.split(".").pop().toLowerCase();

    if (!ALLOWED_EXTENSIONS.has(extension) ||
        (file.type && !ALLOWED_MIME_TYPES.has(file.type))) {
        return "Please choose a JPG, JPEG, or PNG image.";
    }

    if (file.size > MAX_FILE_SIZE) {
        return "The image must be smaller than 10 MB.";
    }

    return "";
}

function showPreview(file) {
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }
    previewUrl = URL.createObjectURL(file);
    originalImage.src = previewUrl;
    originalImage.classList.remove("is-hidden");
    originalImage.alt = `Original uploaded image: ${file.name}`;
    fileName.textContent = file.name;
}

function clearImage(image) {
    image.removeAttribute("src");
    image.classList.add("is-hidden");
}

function handleImageLoadFailure(event) {
    const image = event.currentTarget;
    image.classList.add("is-hidden");
    if (image !== originalImage) {
        setStatus("Visualization could not be loaded.", "error");
    } else {
        setStatus("The uploaded image could not be loaded.", "error");
    }
}

function resetAnalysis() {
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }
    if (generationTimer) {
        clearTimeout(generationTimer);
        generationTimer = null;
    }
    fileInput.value = "";
    fileName.textContent = "No image selected";
    predictionValue.textContent = "—";
    confidenceValue.textContent = "—";
    clearImage(originalImage);
    clearImage(heatmapImage);
    clearImage(overlayImage);
    results.hidden = true;
    results.setAttribute("aria-busy", "false");
    analyzeButton.disabled = false;
    setStatus("");
}

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) {
        fileName.textContent = "No image selected";
        return;
    }

    const validationError = validateFile(file);
    if (validationError) {
        fileInput.value = "";
        fileName.textContent = "No image selected";
        results.hidden = true;
        setStatus(validationError, "error");
        return;
    }

    showPreview(file);
    results.hidden = true;
    setStatus("");
});

async function predictImage() {
    const file = fileInput.files[0];
    if (!file) {
        setStatus("Please select an image before analyzing.", "error");
        return;
    }

    const validationError = validateFile(file);
    if (validationError) {
        setStatus(validationError, "error");
        return;
    }

    analyzeButton.disabled = true;
    results.hidden = true;
    results.setAttribute("aria-busy", "true");
    setStatus("Analyzing image...", "loading");

    generationTimer = setTimeout(() => {
        setStatus("Generating Grad-CAM...", "loading");
    }, 700);

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${API_BASE_URL}/gradcam`, {
            method: "POST",
            body: formData,
        });
        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error("The backend returned an invalid response.");
        }
        if (!response.ok) {
            throw new Error(data.detail || "The image analysis request failed.");
        }

        predictionValue.textContent = String(data.prediction).replaceAll("-", " ");
        confidenceValue.textContent = `${Number(data.confidence).toFixed(2)}%`;
        heatmapImage.src = `${API_BASE_URL}${data.heatmap}`;
        heatmapImage.classList.remove("is-hidden");
        overlayImage.src = `${API_BASE_URL}${data.overlay}`;
        overlayImage.classList.remove("is-hidden");
        results.hidden = false;
        setStatus("Analysis complete.");
    } catch (error) {
        const message = error instanceof TypeError
            ? "Unable to reach the backend. Make sure the FastAPI server is running."
            : error.message;
        setStatus(message || "The image analysis could not be completed.", "error");
    } finally {
        if (generationTimer) {
            clearTimeout(generationTimer);
            generationTimer = null;
        }
        results.setAttribute("aria-busy", "false");
        analyzeButton.disabled = false;
    }
}

originalImage.addEventListener("error", handleImageLoadFailure);
heatmapImage.addEventListener("error", handleImageLoadFailure);
overlayImage.addEventListener("error", handleImageLoadFailure);
analyzeButton.addEventListener("click", predictImage);
resetButton.addEventListener("click", resetAnalysis);
