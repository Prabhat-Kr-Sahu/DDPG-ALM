// src/App.jsx
import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Home from "../pages/Home";
import Predict from "../pages/Predict";
import Capital from "../pages/Capital";
import ResetCapital from "../pages/ResetCapital";
import Train from "../pages/Train";

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <nav className="bg-white shadow p-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-blue-600">Portfolio Optimizer</h1>
          <div className="space-x-4">
            <Link to="/" className="text-blue-500 hover:underline">Home</Link>
            <Link to="/predict" className="text-blue-500 hover:underline">Predict</Link>
            <Link to="/train" className="text-blue-500 hover:underline">Train</Link>
            <Link to="/capital" className="text-blue-500 hover:underline">Capital</Link>
            <Link to="/reset" className="text-red-500 hover:underline">Reset</Link>
          </div>
        </nav>
        <main className="p-6">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/train" element={<Train />} />
            <Route path="/capital" element={<Capital />} />
            <Route path="/reset" element={<ResetCapital />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}
