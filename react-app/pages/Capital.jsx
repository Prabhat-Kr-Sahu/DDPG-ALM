// src/pages/Capital.jsx
import React, { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

export default function Capital() {
  const [capitalData, setCapitalData] = useState([]);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/capital-json")
      .then((res) => res.json())
      .then((data) => {
        const formatted = data.map((capital, index) => ({ day: index + 1, capital }));
        setCapitalData(formatted);
      });
  }, []);

  return (
    <div className="bg-white p-6 rounded shadow">
      <h2 className="text-xl font-semibold mb-4">Capital Over Time</h2>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={capitalData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="day" label={{ value: 'Day', position: 'insideBottomRight', offset: -5 }} />
          <YAxis label={{ value: 'Capital', angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          <Line type="monotone" dataKey="capital" stroke="#3b82f6" strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}