// src/pages/ResetCapital.jsx
import React, { useState } from "react";

export default function ResetCapital() {
  const [initialCapital, setInitialCapital] = useState("");
  const [loading, setLoading] = useState(false);

  const handleReset = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:8000/reset-capital", {
        method: "POST",
        body: new URLSearchParams({ initial_capital: initialCapital }),
      });
      const data = await response.json();
      alert(`Capital reset to ${initialCapital}`);
    } catch (error) {
      console.error("Reset failed:", error);
      alert("Reset failed.");
    }
    setLoading(false);
  };

  return (
    <div className="bg-white p-6 rounded shadow">
      <h2 className="text-xl font-semibold mb-4 text-red-600">Reset Capital</h2>
      <form onSubmit={handleReset} className="space-y-4">
        <input
          type="number"
          value={initialCapital}
          onChange={(e) => setInitialCapital(e.target.value)}
          className="w-full p-2 border rounded"
          placeholder="Enter new initial capital"
          required
        />
        <button
          type="submit"
          className="w-full bg-red-500 hover:bg-red-600 text-white py-2 px-4 rounded"
          disabled={loading}
        >
          {loading ? "Resetting..." : "Reset Capital"}
        </button>
      </form>
    </div>
  );
}
