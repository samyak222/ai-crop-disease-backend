import { useState } from "react";
import axios from "axios";

function UploadCard() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const response = await axios.post(
      "http://127.0.0.1:5000/predict",
      formData
    );

    setResult(response.data);
  };

  return (
    <div>
      <input
        type="file"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <button onClick={handleUpload}>
        Predict Disease
      </button>

      {result && (
        <div>
          <h2>{result.disease}</h2>
          <p>Confidence: {result.confidence}%</p>
          <p>{result.treatment}</p>
        </div>
      )}
    </div>
  );
}

export default UploadCard;