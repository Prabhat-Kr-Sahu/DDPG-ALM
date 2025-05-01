import React from "react";
import { useNavigate } from "react-router-dom";

export default function Home() {
  const navigate = useNavigate();
  
  const menuItems = [
    { 
      title: "Run Prediction", 
      description: "Generate portfolio predictions using our trained model",
      path: "/predict", 
      color: "bg-blue-600 hover:bg-blue-700",
      icon: "📈" 
    },
    { 
      title: "View Capital Chart", 
      description: "Track your portfolio's performance over time",
      path: "/capital", 
      color: "bg-green-600 hover:bg-green-700",
      icon: "💰" 
    },
    { 
      title: "Train Model", 
      description: "Improve predictions with new training data",
      path: "/train", 
      color: "bg-purple-600 hover:bg-purple-700",
      icon: "🧠" 
    },
    { 
      title: "Reset Capital", 
      description: "Start fresh with a new capital baseline",
      path: "/reset", 
      color: "bg-red-600 hover:bg-red-700",
      icon: "🔄" 
    }
  ];

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 to-gray-200 p-4">
      <div className="max-w-4xl w-full">
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-800">
            <span className="text-blue-700">Portfolio</span> Optimizer
          </h1>
          <p className="mt-4 text-xl text-gray-600 max-w-2xl mx-auto">
            Optimize your investment strategy with AI-powered predictions and analytics
          </p>
        </div>
        
        <div className="grid gap-6 md:grid-cols-2">
          {menuItems.map((item, index) => (
            <div 
              key={index}
              className="bg-white rounded-xl shadow-lg overflow-hidden transform transition duration-300 hover:scale-105 hover:shadow-xl"
            >
              <div className={`${item.color} h-2`}></div>
              <div className="p-6">
                <div className="flex items-center mb-4">
                  <span className="text-2xl mr-3">{item.icon}</span>
                  <h3 className="text-xl font-semibold text-gray-800">{item.title}</h3>
                </div>
                <p className="text-gray-600 mb-6">{item.description}</p>
                <button
                  className={`w-full ${item.color} transition-colors duration-200 text-white font-medium py-3 px-4 rounded-lg shadow`}
                  onClick={() => navigate(item.path)}
                >
                  {item.title}
                </button>
              </div>
            </div>
          ))}
        </div>
        
        <div className="mt-12 text-center text-gray-500 text-sm">
          <p>© {new Date().getFullYear()} Portfolio Optimizer. All rights reserved.</p>
        </div>
      </div>
    </div>
  );
}