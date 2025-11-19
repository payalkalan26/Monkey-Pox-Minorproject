/* Interactive Time-Series Forecasting (client-side)
 - Loads a CSV of daily country-wise confirmed cases (wide format) OR a two-column (date,total) CSV.
 - Computes Total_cases over time, renders Plotly chart, and produces a simple moving-average forecast.
 - No server required.
*/

// CDN-loaded globals expected in index.html:
// - Plotly (plotly-latest.min.js)
// - Papa (papaparse.min.js)

const state = {
  series: [], // [{date: Date, value: number}]
  window: 7,
  horizon: 14
};

function parseWideFormat(data) {
  // First column is 'Country', remaining columns are dates
  // Sum per date across rows
  if (!data || data.length === 0) return [];
  const header = data[0];
  const dateCols = header.slice(1);
  const totals = new Array(dateCols.length).fill(0);
  for (let r = 1; r < data.length; r++) {
    const row = data[r];
    for (let c = 1; c < row.length; c++) {
      const v = Number(row[c] || 0);
      if (!isNaN(v)) totals[c - 1] += v;
    }
  }
  return dateCols.map((d, i) => ({ date: new Date(d), value: totals[i] }));
}

function parseTwoColumn(data) {
  // Expect header: [date, total]
  const out = [];
  for (let r = 1; r < data.length; r++) {
    const row = data[r];
    if (!row || row.length < 2) continue;
    const d = new Date(row[0]);
    const v = Number(row[1]);
    if (!isNaN(d.getTime()) && !isNaN(v)) out.push({ date: d, value: v });
  }
  return out;
}

function detectFormatAndParse(results) {
  const data = results.data;
  if (!data || data.length === 0) return [];
  const header = data[0] || [];
  // Wide format heuristic: header[0] == 'Country' and many date-like columns
  const looksWide = header[0] && header[0].toLowerCase().includes('country');
  return looksWide ? parseWideFormat(data) : parseTwoColumn(data);
}

function sortSeries(series) {
  return [...series].filter(d => !isNaN(d.date)).sort((a, b) => a.date - b.date);
}

function movingAverageForecast(series, window, horizon) {
  const vals = series.map(d => d.value);
  const ma = (arr, w) => {
    if (arr.length < w) return arr.length ? arr[arr.length - 1] : 0;
    let sum = 0;
    for (let i = arr.length - w; i < arr.length; i++) sum += arr[i];
    return sum / w;
  };
  // Simple rolling std as uncertainty proxy
  const rollingStd = (arr, w) => {
    if (arr.length < w) return 0;
    const sub = arr.slice(-w);
    const mean = sub.reduce((a, b) => a + b, 0) / sub.length;
    const variance = sub.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / sub.length;
    return Math.sqrt(variance);
  };

  const points = [];
  const lastDate = series[series.length - 1]?.date || new Date();
  let cur = new Date(lastDate);
  let hist = vals.slice();
  const std = rollingStd(hist, window);
  for (let i = 1; i <= horizon; i++) {
    cur = new Date(cur.getTime());
    cur.setDate(cur.getDate() + 1);
    const pred = ma(hist, window);
    points.push({ date: new Date(cur), value: pred, lo: Math.max(0, pred - 1.96 * std), hi: pred + 1.96 * std });
    hist.push(pred);
  }
  return points;
}

function toISODate(d) {
  return d.toISOString().slice(0, 10);
}

async function plot(series, forecast) {
  const xHist = series.map(d => toISODate(d.date));
  const yHist = series.map(d => d.value);
  const xFor = forecast.map(d => toISODate(d.date));
  const yFor = forecast.map(d => d.value);
  const yLo = forecast.map(d => d.lo);
  const yHi = forecast.map(d => d.hi);

  const histTrace = { x: xHist, y: yHist, mode: 'lines+markers', name: 'Total cases', line: { color: '#2563eb' } };
  const forTrace = { x: xFor, y: yFor, mode: 'lines+markers', name: 'Forecast', line: { color: '#16a34a' } };
  const bandTrace = {
    x: [...xFor, ...xFor.slice().reverse()],
    y: [...yHi, ...yLo.slice().reverse()],
    fill: 'toself', type: 'scatter', name: '95% interval', line: { color: 'rgba(16,185,129,0.2)' }, fillcolor: 'rgba(16,185,129,0.15)'
  };

  const layout = {
    title: 'Monkeypox Daily Total Cases and Forecast',
    xaxis: { title: 'Date' },
    yaxis: { title: 'Cases' },
    legend: { orientation: 'h' },
    margin: { t: 40, r: 20, b: 40, l: 50 }
  };

  Plotly.newPlot('chart', [histTrace, bandTrace, forTrace], layout, { responsive: true });
}

function handleCSVFile(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      complete: (results) => {
        try {
          const parsed = detectFormatAndParse(results);
          resolve(sortSeries(parsed));
        } catch (e) {
          reject(e);
        }
      },
      error: (err) => reject(err)
    });
  });
}

async function runForecast() {
  const horizon = Number(document.getElementById('horizon').value) || state.horizon;
  const window = Number(document.getElementById('window').value) || state.window;
  if (!state.series.length) {
    alert('Please load a CSV first.');
    return;
  }
  const fc = movingAverageForecast(state.series, window, horizon);
  plot(state.series, fc);
}

function wireUI() {
  const fileInput = document.getElementById('csvInput');
  const loadBtn = document.getElementById('loadSample');
  const runBtn = document.getElementById('runForecast');

  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    document.getElementById('status').textContent = 'Parsing CSV...';
    try {
      state.series = await handleCSVFile(file);
      document.getElementById('status').textContent = `Loaded ${state.series.length} days.`;
      await runForecast();
    } catch (e) {
      document.getElementById('status').textContent = 'Failed to parse CSV.';
      console.error(e);
    }
  });

  loadBtn.addEventListener('click', async () => {
    // Try to fetch sample from site/assets/Daily_Country_Wise_Confirmed_Cases.csv if present
    document.getElementById('status').textContent = 'Loading sample CSV...';
    try {
      const res = await fetch('assets/Daily_Country_Wise_Confirmed_Cases.csv');
      if (!res.ok) throw new Error('Sample CSV not found under site/assets/.');
      const text = await res.text();
      Papa.parse(text, {
        complete: async (results) => {
          state.series = sortSeries(detectFormatAndParse(results));
          document.getElementById('status').textContent = `Loaded ${state.series.length} days from sample.`;
          await runForecast();
        }
      });
    } catch (e) {
      document.getElementById('status').textContent = 'Sample missing. Upload your CSV.';
    }
  });

  runBtn.addEventListener('click', runForecast);
}

window.addEventListener('DOMContentLoaded', wireUI);
