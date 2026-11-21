import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  BarChart3,
  BrainCircuit,
  CarFront,
  Compass,
  Database,
  Info,
  LayoutDashboard,
  MapPin,
  Menu,
  X,
} from 'lucide-react';
import { AnalyticsView, DashboardView } from './components/DashboardAndAnalytics';
import { AboutProjectView, ModelPerformanceView } from './components/ModelAndAbout';
import { ParkingFinderView } from './components/ParkingFinderView';
import { HistoricalDataView, PredictView } from './components/PredictAndHistory';
import {
  parseCsvText,
  trainAndEvaluateEngine,
  validateAndCleanDataset,
} from './ml/parkingEngine';
import type {
  ParkingRecord,
  PredictionHistoryRow,
  ValidationSummary,
} from './types/parking';

type NavSection =
  | 'finder'
  | 'dashboard'
  | 'predict'
  | 'analytics'
  | 'historical'
  | 'performance'
  | 'about';

const NAV_ITEMS: { id: NavSection; label: string; icon: React.FC<{ className?: string }> }[] = [
  { id: 'finder', label: 'Parking Finder', icon: MapPin },
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'predict', label: 'Predict Availability', icon: Compass },
  { id: 'analytics', label: 'Parking Analytics', icon: BarChart3 },
  { id: 'historical', label: 'Historical Data', icon: Database },
  { id: 'performance', label: 'Model Performance', icon: BrainCircuit },
  { id: 'about', label: 'About Project', icon: Info },
];

export default function App() {
  const [activePage, setActivePage] = useState<NavSection>('finder');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [thresholdPct, setThresholdPct] = useState<number>(30);
  const [selectedLocation, setSelectedLocation] = useState<string>('All Campus Zones');
  const [finderLocation, setFinderLocation] = useState<string>('Main Gate Lot');

  const [activeRawRows, setActiveRawRows] = useState<Record<string, string>[]>([]);
  const [dataSourceMode, setDataSourceMode] = useState<'SYNTHETIC' | 'CUSTOM'>('SYNTHETIC');
  const [disclaimer, setDisclaimer] = useState(
    'DEMONSTRATION / SYNTHETIC DATA: Generated for academic ML training and simulation across 4 campus parking zones. Does not represent live sensors, CCTV, or real campus telemetry.'
  );
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState('');
  const [history, setHistory] = useState<PredictionHistoryRow[]>([]);

  useEffect(() => {
    let mounted = true;
    async function init() {
      setLoading(true);
      setLoadError('');
      try {
        const res = await fetch('/api/state');
        if (!res.ok) {
          throw new Error('Unable to load persisted SmartPark AI state.');
        }
        const stateJson = await res.json();
        const parsed = parseCsvText(stateJson.csvText || '');
        const histRaw = parseCsvText(stateJson.predictionsCsv || '');
        const parsedHistory: PredictionHistoryRow[] = histRaw.map((r) => ({
          Timestamp: r.Timestamp || '',
          Location: r.Location || 'Main Gate Lot',
          Day: r.Day || '',
          Entry_Time: r.Entry_Time || '',
          Hour: Number(r.Hour || 0),
          Total_Capacity: Number(r.Total_Capacity || 0),
          Previous_Occupied: Number(r.Previous_Occupied || 0),
          Previous_Occupancy_Pct: Number(r.Previous_Occupancy_Pct || 0),
          Predicted_Status: r.Predicted_Status === 'LOW' ? 'LOW' : 'HIGH',
          Confidence_Pct: Number(r.Confidence_Pct || 0),
          Estimated_Occupied: Number(r.Estimated_Occupied || 0),
          Estimated_Available: Number(r.Estimated_Available || 0),
          Threshold_Pct: Number(r.Threshold_Pct || 30),
        }));

        if (mounted) {
          if (stateJson.disclaimer) setDisclaimer(stateJson.disclaimer);
          if (typeof stateJson.thresholdPct === 'number') setThresholdPct(stateJson.thresholdPct);
          if (typeof stateJson.selectedLocation === 'string') setSelectedLocation(stateJson.selectedLocation);
          setDataSourceMode(stateJson.sourceMode === 'CUSTOM' ? 'CUSTOM' : 'SYNTHETIC');
          setActiveRawRows(parsed);
          setHistory(parsedHistory);
        }
      } catch (err) {
        if (mounted) {
          setLoadError(err instanceof Error ? err.message : 'Failed to initialize SmartPark AI.');
        }
      } finally {
        if (mounted) setLoading(false);
      }
    }
    init();
    return () => {
      mounted = false;
    };
  }, []);

  const { records, validationSummary, modelBundle, trainingError } = useMemo(() => {
    if (activeRawRows.length === 0) {
      return {
        records: [] as ParkingRecord[],
        validationSummary: null as ValidationSummary | null,
        modelBundle: null,
        trainingError: '',
      };
    }
    try {
      const { records: cleaned, summary } = validateAndCleanDataset(activeRawRows, thresholdPct);
      const bundle = trainAndEvaluateEngine(cleaned, thresholdPct);
      return {
        records: cleaned,
        validationSummary: summary,
        modelBundle: bundle,
        trainingError: '',
      };
    } catch (err) {
      return {
        records: [] as ParkingRecord[],
        validationSummary: null,
        modelBundle: null,
        trainingError: err instanceof Error ? err.message : 'Model training error.',
      };
    }
  }, [activeRawRows, thresholdPct]);

  const handleThresholdChange = useCallback(async (nextVal: number) => {
    setThresholdPct(nextVal);
    try {
      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thresholdPct: nextVal }),
      });
    } catch {
      // ignore offline config write
    }
  }, []);

  const handleLocationChange = useCallback(async (nextLoc: string) => {
    setSelectedLocation(nextLoc);
    try {
      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ selectedLocation: nextLoc }),
      });
    } catch {
      // ignore offline config write
    }
  }, []);

  const handleRecordPrediction = useCallback(async (row: PredictionHistoryRow) => {
    setHistory((prev) => [row, ...prev]);
    try {
      await fetch('/api/predictions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(row),
      });
    } catch {
      // Optimistic local state preserved
    }
  }, []);

  const handleApplyCustomDataset = useCallback(
    async (rawText: string, rawRows: Record<string, string>[]) => {
      setActiveRawRows(rawRows);
      setDataSourceMode('CUSTOM');
      setSelectedLocation('All Campus Zones');
      try {
        await fetch('/api/custom-dataset', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ csvText: rawText }),
        });
      } catch {
        // local state remains active
      }
    },
    []
  );

  const handleResetToSample = useCallback(async () => {
    try {
      const res = await fetch('/api/reset-dataset', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setActiveRawRows(parseCsvText(data.csvText || ''));
        setDataSourceMode('SYNTHETIC');
        setSelectedLocation('All Campus Zones');
      }
    } catch {
      // ignore
    }
  }, []);

  const sidebarContent = (
    <div className="flex flex-col h-full bg-[#0B192C] text-white w-64 shrink-0">
      <div className="px-5 py-6 border-b border-white/10 flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg overflow-hidden bg-[#152C4E] border border-white/15 shrink-0 flex items-center justify-center">
          <img
            src="/api/logo.png"
            alt="SmartPark AI Logo"
            referrerPolicy="no-referrer"
            className="w-full h-full object-cover"
            onError={(e) => {
              e.currentTarget.style.display = 'none';
            }}
          />
          <CarFront className="w-5 h-5 text-sky-400 hidden" />
        </div>
        <div className="min-w-0">
          <div className="text-base font-bold tracking-tight whitespace-nowrap">SMARTPARK AI</div>
          <div className="text-[11px] text-sky-200/80 whitespace-nowrap">Campus Parking ML</div>
        </div>
      </div>

      <nav className="flex-1 px-3 py-5 space-y-1 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const active = activePage === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                setActivePage(item.id);
                setMobileMenuOpen(false);
              }}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                active
                  ? 'bg-white text-[#0B192C]'
                  : 'text-slate-200 hover:bg-white/10 hover:text-white'
              }`}
            >
              <Icon className="w-4 h-4 shrink-0" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="p-4 border-t border-white/10 space-y-3">
        <div>
          <div className="flex items-center justify-between text-xs text-slate-200 mb-1.5">
            <span>HIGH Threshold</span>
            <span className="font-mono-num font-semibold text-sky-300">&gt;{thresholdPct}% open</span>
          </div>
          <input
            type="range"
            min={10}
            max={60}
            step={5}
            value={thresholdPct}
            onChange={(e) => handleThresholdChange(Number(e.target.value))}
            className="w-full accent-sky-400 cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-slate-400 font-mono-num">
            <span>10%</span>
            <span>30%</span>
            <span>60%</span>
          </div>
        </div>

        <div className="text-[11px] text-slate-300 leading-relaxed pt-2 border-t border-white/10">
          Dataset: {dataSourceMode === 'SYNTHETIC' ? 'Demonstration / Synthetic' : 'Custom Uploaded CSV'}
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen flex bg-slate-50 text-slate-900">
      <aside className="hidden lg:block sticky top-0 h-screen">{sidebarContent}</aside>

      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div className="relative z-10 h-full">{sidebarContent}</div>
          <button
            type="button"
            aria-label="Close navigation"
            onClick={() => setMobileMenuOpen(false)}
            className="flex-1 bg-slate-950/60"
          />
        </div>
      )}

      <div className="flex-1 min-w-0 flex flex-col">
        <header className="bg-[#0B192C] text-white px-5 sm:px-8 py-5 border-b border-slate-800">
          <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 min-w-0">
              <button
                type="button"
                aria-label="Toggle navigation menu"
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="lg:hidden p-2 rounded-lg bg-white/10 hover:bg-white/20 text-white shrink-0 cursor-pointer"
              >
                {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
              <div className="min-w-0">
                <h1 className="text-lg sm:text-2xl font-bold tracking-tight text-white">
                  SmartPark AI – Intelligent Campus Parking System
                </h1>
                <p className="text-xs sm:text-sm text-sky-200 mt-0.5">
                  Predict smarter. Park faster. Save time.
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setActivePage(activePage === 'finder' ? 'dashboard' : 'finder')}
              className="hidden sm:inline-flex items-center px-4 py-2 bg-white text-[#0B192C] text-xs font-semibold rounded-lg hover:bg-slate-100 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              {activePage === 'finder' ? 'Analytics Dashboard' : 'Parking Finder'}
            </button>
          </div>
        </header>

        <main className="flex-1 px-5 sm:px-8 py-6">
          <div className="max-w-7xl mx-auto space-y-5">
            {dataSourceMode === 'SYNTHETIC' && (
              <div className="bg-sky-50 border border-sky-200 rounded-xl px-4 py-3 text-xs text-[#0B192C]">
                <span className="font-semibold">Dataset Notice:</span> {disclaimer}
              </div>
            )}

            {loading ? (
              <div className="bg-white border border-slate-200 rounded-xl p-10 text-center text-sm text-slate-600">
                Loading persisted parking records and training Random Forest classifier...
              </div>
            ) : loadError ? (
              <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-sm text-red-800">
                {loadError}
              </div>
            ) : trainingError || !modelBundle ? (
              <div className="bg-amber-50 border border-amber-200 rounded-xl p-6 space-y-3">
                <div className="text-sm font-semibold text-amber-950">Model Training Warning</div>
                <p className="text-xs text-amber-900">{trainingError}</p>
                {dataSourceMode === 'CUSTOM' && (
                  <button
                    type="button"
                    onClick={handleResetToSample}
                    className="px-3.5 py-2 bg-[#0B192C] text-white text-xs font-semibold rounded-lg cursor-pointer"
                  >
                    Restore Sample Synthetic Dataset
                  </button>
                )}
              </div>
            ) : (
              <>
                {activePage === 'finder' && (
                  <ParkingFinderView
                    records={records}
                    modelBundle={modelBundle}
                    selectedLocation={finderLocation}
                    onSelectLocation={setFinderLocation}
                    onOpenPredictWorkbench={() => setActivePage('predict')}
                  />
                )}
                {activePage === 'dashboard' && (
                  <DashboardView
                    records={records}
                    modelBundle={modelBundle}
                    thresholdPct={thresholdPct}
                    selectedLocation={selectedLocation}
                    onSelectLocation={handleLocationChange}
                    onNavigate={(p) => setActivePage(p)}
                  />
                )}
                {activePage === 'predict' && (
                  <PredictView
                    records={records}
                    modelBundle={modelBundle}
                    thresholdPct={thresholdPct}
                    history={history}
                    onRecordPrediction={handleRecordPrediction}
                  />
                )}
                {activePage === 'analytics' && (
                  <AnalyticsView
                    records={records}
                    selectedLocation={selectedLocation}
                    onSelectLocation={handleLocationChange}
                  />
                )}
                {activePage === 'historical' && (
                  <HistoricalDataView
                    records={records}
                    thresholdPct={thresholdPct}
                    dataSourceMode={dataSourceMode}
                    validationSummary={validationSummary}
                    onApplyCustomDataset={handleApplyCustomDataset}
                    onResetToSample={handleResetToSample}
                  />
                )}
                {activePage === 'performance' && (
                  <ModelPerformanceView modelBundle={modelBundle} />
                )}
                {activePage === 'about' && <AboutProjectView />}
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
