const API_URL = "http://127.0.0.1:8000/api/v1/predict";

const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const uploadText = document.getElementById("uploadText");
const predictButton = document.getElementById("predictButton");
const loading = document.getElementById("loading");
const error = document.getElementById("error");
const result = document.getElementById("result");
const plantName = document.getElementById("plantName");
const confidence = document.getElementById("confidence");

let selectedFile = null;

imageInput.addEventListener("change", () => {
  selectedFile = imageInput.files[0];

  result.classList.add("hidden");
  error.classList.add("hidden");

  if (!selectedFile) {
    preview.classList.add("hidden");
    predictButton.disabled = true;
    uploadText.textContent = "Choose an image";
    return;
  }

  uploadText.textContent = selectedFile.name;
  preview.src = URL.createObjectURL(selectedFile);
  preview.classList.remove("hidden");
  predictButton.disabled = false;
});

predictButton.addEventListener("click", async () => {
  if (!selectedFile) return;

  const formData = new FormData();
  formData.append("file", selectedFile);

  loading.classList.remove("hidden");
  error.classList.add("hidden");
  result.classList.add("hidden");
  predictButton.disabled = true;

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Prediction request failed.");
    }

    plantName.textContent = data.plant;
    confidence.textContent = `${(data.confidence * 100).toFixed(2)}%`;
    result.classList.remove("hidden");
  } catch (err) {
    error.textContent = err.message;
    error.classList.remove("hidden");
  } finally {
    loading.classList.add("hidden");
    predictButton.disabled = false;
  }
});
