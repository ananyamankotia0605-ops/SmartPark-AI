import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SMARTPARK_DIR = path.resolve(__dirname, 'SmartPark-AI');
const DATA_DIR = path.join(SMARTPARK_DIR, 'data');
const SAMPLE_CSV_PATH = path.join(DATA_DIR, 'sample_parking_data.csv');
const ACTIVE_CSV_PATH = path.join(DATA_DIR, 'active_dataset.csv');
const CONFIG_JSON_PATH = path.join(DATA_DIR, 'app_config.json');
const PREDICTIONS_CSV_PATH = path.join(DATA_DIR, 'predictions.csv');
const LOGO_PNG_PATH = path.join(SMARTPARK_DIR, 'assets', 'logo.png');

const PREDICTION_HEADER =
  'Timestamp,Location,Day,Entry_Time,Hour,Total_Capacity,Previous_Occupied,Previous_Occupancy_Pct,Predicted_Status,Confidence_Pct,Estimated_Occupied,Estimated_Available,Threshold_Pct';

const SAMPLE_DISCLAIMER =
  'DEMONSTRATION / SYNTHETIC DATA: Generated for academic ML training and simulation across 4 campus parking zones. Does not represent live sensors, CCTV, or real campus telemetry.';

interface AppConfig {
  threshold_pct: number;
  data_source_mode: 'SYNTHETIC' | 'CUSTOM';
  selected_location: string;
}

function ensureDataDir() {
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }
}

function readAppConfig(): AppConfig {
  const defaults: AppConfig = {
    threshold_pct: 30,
    data_source_mode: 'SYNTHETIC',
    selected_location: 'All Campus Zones',
  };
  if (!fs.existsSync(CONFIG_JSON_PATH)) return defaults;
  try {
    const parsed = JSON.parse(fs.readFileSync(CONFIG_JSON_PATH, 'utf-8'));
    return {
      threshold_pct:
        typeof parsed.threshold_pct === 'number' && parsed.threshold_pct >= 10 && parsed.threshold_pct <= 60
          ? parsed.threshold_pct
          : 30,
      data_source_mode: parsed.data_source_mode === 'CUSTOM' ? 'CUSTOM' : 'SYNTHETIC',
      selected_location: typeof parsed.selected_location === 'string' && parsed.selected_location ? parsed.selected_location : 'All Campus Zones',
    };
  } catch {
    return defaults;
  }
}

function writeAppConfig(cfg: AppConfig) {
  ensureDataDir();
  fs.writeFileSync(CONFIG_JSON_PATH, JSON.stringify(cfg, null, 2), 'utf-8');
}

function ensurePredictionCsv() {
  ensureDataDir();
  if (!fs.existsSync(PREDICTIONS_CSV_PATH)) {
    fs.writeFileSync(PREDICTIONS_CSV_PATH, `${PREDICTION_HEADER}\n`, 'utf-8');
    return;
  }
  const content = fs.readFileSync(PREDICTIONS_CSV_PATH, 'utf-8');
  const firstLine = content.split('\n')[0] || '';
  if (firstLine && !firstLine.includes('Location')) {
    fs.writeFileSync(PREDICTIONS_CSV_PATH, `${PREDICTION_HEADER}\n`, 'utf-8');
  }
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '10mb' }));

  app.get('/api/health', (_req, res) => {
    const cfg = readAppConfig();
    res.json({
      status: 'ok',
      hasSampleDataset: fs.existsSync(SAMPLE_CSV_PATH),
      hasCustomDataset: fs.existsSync(ACTIVE_CSV_PATH),
      dataSourceMode: cfg.data_source_mode,
      thresholdPct: cfg.threshold_pct,
    });
  });

  app.get('/api/logo.png', (_req, res) => {
    if (fs.existsSync(LOGO_PNG_PATH)) {
      res.sendFile(LOGO_PNG_PATH);
    } else {
      res.status(404).end();
    }
  });

  app.get('/api/state', (_req, res) => {
    ensurePredictionCsv();
    const cfg = readAppConfig();
    const useCustom = cfg.data_source_mode === 'CUSTOM' && fs.existsSync(ACTIVE_CSV_PATH);
    const csvPath = useCustom ? ACTIVE_CSV_PATH : SAMPLE_CSV_PATH;

    if (!fs.existsSync(csvPath)) {
      res.status(404).json({ error: 'Dataset file not found on disk.' });
      return;
    }

    const csvText = fs.readFileSync(csvPath, 'utf-8');
    const predictionsCsv = fs.readFileSync(PREDICTIONS_CSV_PATH, 'utf-8');

    res.json({
      sourceMode: useCustom ? 'CUSTOM' : 'SYNTHETIC',
      thresholdPct: cfg.threshold_pct,
      selectedLocation: cfg.selected_location,
      disclaimer: SAMPLE_DISCLAIMER,
      csvText,
      predictionsCsv,
    });
  });

  app.get('/api/dataset', (_req, res) => {
    const cfg = readAppConfig();
    const useCustom = cfg.data_source_mode === 'CUSTOM' && fs.existsSync(ACTIVE_CSV_PATH);
    const csvPath = useCustom ? ACTIVE_CSV_PATH : SAMPLE_CSV_PATH;
    if (!fs.existsSync(csvPath)) {
      res.status(404).json({ error: 'Dataset not found.' });
      return;
    }
    const csvText = fs.readFileSync(csvPath, 'utf-8');
    res.json({
      sourceMode: useCustom ? 'CUSTOM' : 'SYNTHETIC',
      disclaimer: SAMPLE_DISCLAIMER,
      csvText,
    });
  });

  app.post('/api/custom-dataset', (req, res) => {
    const { csvText } = req.body ?? {};
    if (typeof csvText !== 'string' || csvText.trim().split('\n').length < 2) {
      res.status(400).json({ error: 'Custom CSV payload must contain a header and at least one data row.' });
      return;
    }
    ensureDataDir();
    fs.writeFileSync(ACTIVE_CSV_PATH, csvText, 'utf-8');
    const prev = readAppConfig();
    const nextCfg: AppConfig = {
      ...prev,
      data_source_mode: 'CUSTOM',
      selected_location: 'All Campus Zones',
    };
    writeAppConfig(nextCfg);
    res.json({ ok: true, sourceMode: 'CUSTOM' });
  });

  app.post('/api/reset-dataset', (_req, res) => {
    if (fs.existsSync(ACTIVE_CSV_PATH)) {
      fs.unlinkSync(ACTIVE_CSV_PATH);
    }
    const prev = readAppConfig();
    const nextCfg: AppConfig = {
      ...prev,
      data_source_mode: 'SYNTHETIC',
      selected_location: 'All Campus Zones',
    };
    writeAppConfig(nextCfg);
    const csvText = fs.existsSync(SAMPLE_CSV_PATH) ? fs.readFileSync(SAMPLE_CSV_PATH, 'utf-8') : '';
    res.json({ ok: true, sourceMode: 'SYNTHETIC', csvText, disclaimer: SAMPLE_DISCLAIMER });
  });

  app.post('/api/config', (req, res) => {
    const { thresholdPct, selectedLocation } = req.body ?? {};
    const prev = readAppConfig();
    const nextThreshold =
      typeof thresholdPct === 'number' && thresholdPct >= 10 && thresholdPct <= 60
        ? thresholdPct
        : prev.threshold_pct;
    const nextLocation =
      typeof selectedLocation === 'string' && selectedLocation.trim()
        ? selectedLocation.trim()
        : prev.selected_location;

    const nextCfg: AppConfig = {
      ...prev,
      threshold_pct: nextThreshold,
      selected_location: nextLocation,
    };
    writeAppConfig(nextCfg);
    res.json({ ok: true, thresholdPct: nextThreshold, selectedLocation: nextLocation });
  });

  app.get('/api/predictions', (_req, res) => {
    ensurePredictionCsv();
    const csvText = fs.readFileSync(PREDICTIONS_CSV_PATH, 'utf-8');
    res.json({ csvText });
  });

  app.post('/api/predictions', (req, res) => {
    ensurePredictionCsv();
    const row = req.body;
    if (!row || !row.Day || !row.Predicted_Status) {
      res.status(400).json({ error: 'Invalid prediction log payload.' });
      return;
    }

    const safeCsv = (v: unknown) => {
      const str = String(v ?? '').replace(/"/g, '""');
      return str.includes(',') ? `"${str}"` : str;
    };

    const line = [
      safeCsv(row.Timestamp),
      safeCsv(row.Location || 'Main Gate Lot'),
      safeCsv(row.Day),
      safeCsv(row.Entry_Time),
      safeCsv(row.Hour),
      safeCsv(row.Total_Capacity),
      safeCsv(row.Previous_Occupied),
      safeCsv(row.Previous_Occupancy_Pct),
      safeCsv(row.Predicted_Status),
      safeCsv(row.Confidence_Pct),
      safeCsv(row.Estimated_Occupied),
      safeCsv(row.Estimated_Available),
      safeCsv(row.Threshold_Pct),
    ].join(',');

    const existing = fs.readFileSync(PREDICTIONS_CSV_PATH, 'utf-8').trimEnd().split('\n');
    const header = existing[0] || PREDICTION_HEADER;
    const body = existing.slice(1).filter((l) => l.trim().length > 0);
    const updated = [header, line, ...body].join('\n') + '\n';
    fs.writeFileSync(PREDICTIONS_CSV_PATH, updated, 'utf-8');

    res.json({ ok: true, csvText: updated });
  });

  app.get('/api/source-files', (_req, res) => {
    const fileSpecs = [
      { rel: 'app.py', name: 'app.py', language: 'python' },
      { rel: 'model.py', name: 'model.py', language: 'python' },
      { rel: 'data_generator.py', name: 'data_generator.py', language: 'python' },
      { rel: 'data_processor.py', name: 'data_processor.py', language: 'python' },
      { rel: 'analytics.py', name: 'analytics.py', language: 'python' },
      { rel: 'tests/test_model.py', name: 'tests/test_model.py', language: 'python' },
      { rel: 'requirements.txt', name: 'requirements.txt', language: 'text' },
      { rel: 'README.md', name: 'README.md', language: 'markdown' },
    ];

    const files = fileSpecs
      .map((spec) => {
        const full = path.join(SMARTPARK_DIR, spec.rel);
        if (!fs.existsSync(full)) return null;
        return {
          path: `SmartPark-AI/${spec.rel}`,
          name: spec.name,
          language: spec.language,
          content: fs.readFileSync(full, 'utf-8'),
        };
      })
      .filter(Boolean);

    res.json({ files });
  });

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  const PORT = 3000;
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`SmartPark AI server listening on http://localhost:${PORT}`);
  });
}

startServer();
