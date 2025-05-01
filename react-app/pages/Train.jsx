// src/pages/Train.jsx
import React, { useState } from "react";

export default function Train() {
  const [loading, setLoading] = useState(false);

  const handleTrain = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:8000/train", {
        method: "POST",
      });
      const data = await response.json();
      alert("Training completed!");
    } catch (error) {
      console.error("Training failed:", error);
      alert("Training failed.");
    }
    setLoading(false);
  };

  return (
    <div className="bg-white p-6 rounded shadow">
      <h2 className="text-xl font-semibold mb-4 text-blue-600">Train Model</h2>
      <button
        className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded"
        onClick={handleTrain}
        disabled={loading}
      >
        {loading ? "Training..." : "Start Training"}
      </button>
    </div>
  );
}
