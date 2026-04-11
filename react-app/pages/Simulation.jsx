import React, { useEffect, useState } from "react";
import {
  LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

const COLORS = {
  ddpg: "#3b82f6",
  buy_and_hold: "#10b981",
  equal_weight: "#f59e0b",
  benchmark: "#ef4444",
};

const LABELS = {
  ddpg: "DDPG-ALM",
  buy_and_hold: "Buy & Hold",
  equal_weight: "Equal Weight",
  benchmark: "Nifty 50",
};

const DASHES = {
  ddpg: "0",
  buy_and_hold: "5 5",
  equal_weight: "3 3",
  benchmark: "8 4",
};

function formatPct(v) {
  if (v == null) return "—";
  return `${(v * 100).toFixed(2)}%`;
}

function buildRows(dates, series) {
  return dates.map((date, i) => {
    const row = { date };
    for (const key of Object.keys(series)) {
      row[key] = series[key] != null ? series[key][i] : null;
    }
    return row;
  });
}

export default function Simulation() {
  const [cumData, setCumData] = useState([]);
  const [ddData, setDdData] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [hasBenchmark, setHasBenchmark] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch("http://127.0.0.1:8000/simulation/cumulative-returns").then((r) => r.json()),
      fetch("http://127.0.0.1:8000/simulation/drawdowns").then((r) => r.json()),
      fetch("http://127.0.0.1:8000/simulation/metrics").then((r) => r.json()),
    ])
      .then(([cum, dd, met]) => {
        setCumData(buildRows(cum.dates, cum.series));
        setDdData(buildRows(dd.dates, dd.series));
        setMetrics(met);
        setHasBenchmark(cum.series.benchmark != null);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div className="p-6 text-gray-500">Loading simulation data…</div>;
  if (error) return <div className="p-6 text-red-500">Error: {error}</div>;

  const seriesKeys = hasBenchmark
    ? ["ddpg", "buy_and_hold", "equal_weight", "benchmark"]
    : ["ddpg", "buy_and_hold", "equal_weight"];

  return (
    <div className="space-y-8">
      {/* Cumulative Returns */}
      <div className="bg-white p-6 rounded shadow">
        <h2 className="text-xl font-semibold mb-4">Cumulative Returns</h2>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={cumData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} interval="preserveStartEnd" />
            <YAxis tickFormatter={(v) => `${(v * 100).toFixed(1)}%`} />
            <Tooltip formatter={(v) => formatPct(v)} />
            <Legend />
            {seriesKeys.map((key) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                name={LABELS[key]}
                stroke={COLORS[key]}
                strokeWidth={key === "ddpg" ? 2.5 : 1.5}
                strokeDasharray={DASHES[key]}
                dot={false}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Drawdown */}
      <div className="bg-white p-6 rounded shadow">
        <h2 className="text-xl font-semibold mb-4">Drawdown</h2>
        <ResponsiveContainer width="100%" height={350}>
          <AreaChart data={ddData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} interval="preserveStartEnd" />
            <YAxis tickFormatter={(v) => `${(v * 100).toFixed(1)}%`} />
            <Tooltip formatter={(v) => formatPct(v)} />
            <Legend />
            {seriesKeys.map((key) => (
              <Area
                key={key}
                type="monotone"
                dataKey={key}
                name={LABELS[key]}
                stroke={COLORS[key]}
                fill={COLORS[key] + "22"}
                strokeWidth={key === "ddpg" ? 2.5 : 1.5}
                strokeDasharray={DASHES[key]}
                dot={false}
                connectNulls
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Metrics Table */}
      <div className="bg-white p-6 rounded shadow">
        <h2 className="text-xl font-semibold mb-4">Performance Metrics</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-sm">
            <thead>
              <tr className="bg-gray-100">
                <th className="px-4 py-2 border border-gray-200">Strategy</th>
                <th className="px-4 py-2 border border-gray-200">Ann. Return</th>
                <th className="px-4 py-2 border border-gray-200">Ann. Volatility</th>
                <th className="px-4 py-2 border border-gray-200">Sharpe Ratio</th>
                <th className="px-4 py-2 border border-gray-200">Max Drawdown</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((row) => (
                <tr
                  key={row.strategy}
                  className={row.strategy === "DDPG-ALM" ? "bg-blue-50 font-semibold" : ""}
                >
                  <td className="px-4 py-2 border border-gray-200">{row.strategy}</td>
                  <td className="px-4 py-2 border border-gray-200">{formatPct(row.annualized_return)}</td>
                  <td className="px-4 py-2 border border-gray-200">{formatPct(row.annualized_vol)}</td>
                  <td className="px-4 py-2 border border-gray-200">{row.sharpe.toFixed(3)}</td>
                  <td className="px-4 py-2 border border-gray-200 text-red-600">
                    {formatPct(row.max_drawdown)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
