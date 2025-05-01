// src/pages/Predict.jsx
import React, { useState } from "react";

export default function Predict() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Prediction failed:", error);
    }
    setLoading(false);
  };

  return (
    <div className="bg-white p-6 rounded shadow">
      <h2 className="text-xl font-semibold mb-4 text-blue-600">Run Prediction</h2>
      <button
        className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded"
        onClick={handlePredict}
        disabled={loading}
      >
        {loading ? "Predicting..." : "Run Prediction"}
      </button>

      {result && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold text-green-600">Prediction Results</h3>
          <p className="mt-2">Daily Return: <strong>{result.returns}%</strong></p>
          <p>New Capital: <strong>{Math.round(result.new_capital)}</strong></p>
          <div className="mt-4">
            <h4 className="text-md font-semibold mb-2">Actions:</h4>
            <table className="table-auto w-full text-left border">
              <thead>
                <tr className="bg-gray-200">
                  {Object.keys(result.actions).map((asset) => (
                    <th key={asset} className="px-2 py-1 border">{asset}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  {Object.values(result.actions).map((val, idx) => (
                    <td key={idx} className="px-2 py-1 border">{val.toFixed(4)}</td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
