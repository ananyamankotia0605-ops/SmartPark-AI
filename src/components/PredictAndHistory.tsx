import React, { useEffect, useMemo, useState } from 'react';
import { ArrowUpDown, Download, RotateCcw, Search, Upload } from 'lucide-react';
import type { TrainedModelsBundle } from '../ml/parkingEngine';
import {
  CSV_UPLOAD_TEMPLATE,
  VALID_DAYS,
  getLocationSnapshots,
  parseCsvText,
  predictParkingAvailability,
  validateAndCleanDataset,
  validatePredictionInput,
} from '../ml/parkingEngine';
import type {
  DayName,
  ParkingRecord,
  PredictionHistoryRow,
  PredictionResult,
  ValidationSummary,
} from '../types/parking';
import { AvailabilityGauge } from './Charts';

export const PredictView: React.FC<{
  records: ParkingRecord[];
  modelBundle: TrainedModelsBundle;
  thresholdPct: number;
  history: PredictionHistoryRow[];
  onRecordPrediction: (row: PredictionHistoryRow) => Promise<void>;
}> = ({ records, modelBundle, thresholdPct, history, onRecordPrediction }) => {
  const snapshots = useMemo(() => getLocationSnapshots(records), [records]);
  const locationNames = useMemo(() => snapshots.map((s) => s.location), [snapshots]);

  const [location, setLocation] = useState<string>(locationNames[0] ?? 'Main Gate Lot');
  const [day, setDay] = useState<DayName>('Monday');
  const [entryTime, setEntryTime] = useState('10:00');
  const [totalCapacity, setTotalCapacity] = useState(
    String(snapshots[0]?.totalCapacity ?? 300)
  );
  const [previousOccupied, setPreviousOccupied] = useState(
    String(snapshots[0]?.latestOccupied ?? 180)
  );
  const [error, setError] = useState('');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (locationNames.length > 0 && !locationNames.includes(location)) {
      setLocation(locationNames[0]);
    }
  }, [locationNames, location]);

  const handleLocationChange = (nextLoc: string) => {
    setLocation(nextLoc);
    const snap = snapshots.find((s) => s.location === nextLoc);
    if (snap) {
      setTotalCapacity(String(snap.totalCapacity));
      setPreviousOccupied(String(snap.latestOccupied));
    }
  };

  const capNum = Number(totalCapacity);
  const occNum = Number(previousOccupied);
  const liveOccupancyPct =
    Number.isFinite(capNum) && capNum > 0 && Number.isFinite(occNum) && occNum >= 0 && occNum <= capNum
      ? ((occNum / capNum) * 100).toFixed(2)
      : null;

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    const cap = Number(totalCapacity);
    const prev = Number(previousOccupied);
    const check = validatePredictionInput(day, entryTime, cap, prev, location, locationNames);
    if (!check.valid) {
      setError(check.error);
      setResult(null);
      return;
    }

    setSubmitting(true);
    try {
      const pred = predictParkingAvailability(modelBundle, day, check.hour, prev, cap, location);
      setResult(pred);

      const newRow: PredictionHistoryRow = {
        Timestamp: new Date().toISOString().replace('T', ' ').slice(0, 19),
        Location: location,
        Day: day,
        Entry_Time: entryTime,
        Hour: check.hour,
        Total_Capacity: cap,
        Previous_Occupied: prev,
        Previous_Occupancy_Pct: pred.previousOccupancyPct,
        Predicted_Status: pred.predictedStatus,
        Confidence_Pct: pred.confidencePct,
        Estimated_Occupied: pred.estimatedOccupied,
        Estimated_Available: pred.estimatedAvailable,
        Threshold_Pct: thresholdPct,
      };
      await onRecordPrediction(newRow);
    } finally {
      setSubmitting(false);
    }
  };

  const downloadHistoryCsv = () => {
    const header =
      'Timestamp,Location,Day,Entry_Time,Hour,Total_Capacity,Previous_Occupied,Previous_Occupancy_Pct,Predicted_Status,Confidence_Pct,Estimated_Occupied,Estimated_Available,Threshold_Pct';
    const lines = history.map((r) =>
      [
        r.Timestamp,
        r.Location,
        r.Day,
        r.Entry_Time,
        r.Hour,
        r.Total_Capacity,
        r.Previous_Occupied,
        r.Previous_Occupancy_Pct,
        r.Predicted_Status,
        r.Confidence_Pct,
        r.Estimated_Occupied,
        r.Estimated_Available,
        r.Threshold_Pct,
      ].join(',')
    );
    const blob = new Blob([[header, ...lines].join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'smartpark_prediction_history.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        <form
          onSubmit={handlePredict}
          className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6 space-y-4"
        >
          <div>
            <h2 className="text-base font-bold text-slate-900">Predict Parking Availability</h2>
            <p className="text-xs text-slate-500 mt-1">
              Predictions are computed by the trained Random Forest classifier using historical patterns and your prior occupancy input.
            </p>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1.5" htmlFor="pred-loc">
              Parking Location
            </label>
            <select
              id="pred-loc"
              value={location}
              onChange={(e) => handleLocationChange(e.target.value)}
              className="w-full px-3 py-2 text-sm bg-white border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#0B192C]"
            >
              {locationNames.map((loc) => (
                <option key={loc} value={loc}>
                  {loc}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5" htmlFor="pred-day">
                Day of the Week
              </label>
              <select
                id="pred-day"
                value={day}
                onChange={(e) => setDay(e.target.value as DayName)}
                className="w-full px-3 py-2 text-sm bg-white border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#0B192C]"
              >
                {VALID_DAYS.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5" htmlFor="pred-time">
                Time of Entry
              </label>
              <input
                id="pred-time"
                type="time"
                value={entryTime}
                onChange={(e) => setEntryTime(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-white border border-slate-300 rounded-lg font-mono-num focus:outline-none focus:ring-2 focus:ring-[#0B192C]"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5" htmlFor="pred-cap">
                Total Parking Capacity
              </label>
              <input
                id="pred-cap"
                type="number"
                min={1}
                step={1}
                value={totalCapacity}
                onChange={(e) => setTotalCapacity(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-white border border-slate-300 rounded-lg font-mono-num focus:outline-none focus:ring-2 focus:ring-[#0B192C]"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5" htmlFor="pred-occ">
                Previous Occupied
              </label>
              <input
                id="pred-occ"
                type="number"
                min={0}
                step={1}
                value={previousOccupied}
                onChange={(e) => setPreviousOccupied(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-white border border-slate-300 rounded-lg font-mono-num focus:outline-none focus:ring-2 focus:ring-[#0B192C]"
              />
            </div>
          </div>

          <div className="px-3 py-2.5 bg-slate-50 border border-slate-200 rounded-lg flex items-center justify-between text-xs">
            <span className="text-slate-600">Auto-calculated Previous Occupancy %</span>
            <span className="font-mono-num font-bold text-slate-900">
              {liveOccupancyPct !== null ? `${liveOccupancyPct}%` : 'Invalid ratio'}
            </span>
          </div>

          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-xs text-red-800 font-medium">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting}
            className="w-full py-2.5 px-4 bg-[#0B192C] text-white text-sm font-semibold rounded-lg hover:bg-[#152C4E] transition-colors disabled:opacity-60 cursor-pointer"
          >
            {submitting ? 'Computing Prediction...' : 'Predict Parking Availability'}
          </button>
        </form>

        <div className="lg:col-span-7 space-y-4">
          {!result ? (
            <div className="bg-white border border-slate-200 rounded-xl p-8 text-center space-y-2">
              <div className="text-base font-semibold text-slate-900">Ready to Predict Entry Availability</div>
              <p className="text-sm text-slate-500 max-w-md mx-auto">
                Select a parking zone, day of entry, arrival time, lot capacity, and prior occupied count, then click Predict Parking Availability.
              </p>
            </div>
          ) : (
            <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-5">
              <div
                className={`rounded-xl p-4 border ${
                  result.predictedStatus === 'HIGH'
                    ? 'bg-emerald-50 border-emerald-200 text-emerald-950'
                    : 'bg-red-50 border-red-200 text-red-950'
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-lg font-bold tracking-tight">
                    {result.location} · {result.predictedStatus} AVAILABILITY
                  </span>
                  <span className="text-xs font-mono-num font-semibold">
                    Confidence: {result.confidencePct.toFixed(1)}% (HIGH {result.probHighPct.toFixed(1)}% · LOW{' '}
                    {result.probLowPct.toFixed(1)}%)
                  </span>
                </div>
                <p className="text-sm mt-1.5">{result.recommendation}</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
                <div className="md:col-span-2 grid grid-cols-2 gap-3">
                  <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg">
                    <div className="text-xs text-slate-500">Prediction Confidence</div>
                    <div className="text-xl font-bold text-slate-900 font-mono-num mt-0.5">
                      {result.confidencePct.toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg">
                    <div className="text-xs text-slate-500">Estimated Available</div>
                    <div className="text-xl font-bold text-emerald-700 font-mono-num mt-0.5">
                      {result.estimatedAvailable} spaces
                    </div>
                  </div>
                  <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg">
                    <div className="text-xs text-slate-500">Estimated Occupied</div>
                    <div className="text-xl font-bold text-slate-900 font-mono-num mt-0.5">
                      {result.estimatedOccupied} spaces
                    </div>
                  </div>
                  <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg">
                    <div className="text-xs text-slate-500">Estimated Occupancy %</div>
                    <div className="text-xl font-bold text-slate-900 font-mono-num mt-0.5">
                      {result.estimatedOccupancyPct.toFixed(1)}%
                    </div>
                  </div>
                </div>

                <AvailabilityGauge
                  availablePct={result.estimatedAvailabilityPct}
                  status={result.predictedStatus}
                />
              </div>

              <div className="pt-3 border-t border-slate-200">
                <h3 className="text-xs font-semibold text-slate-700 mb-2">Prediction Explanation</h3>
                <ul className="space-y-1.5 text-xs text-slate-600 list-disc pl-4">
                  {result.explanations.map((line, i) => (
                    <li key={i}>{line}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-bold text-slate-900">Saved Prediction History</h3>
            <p className="text-xs text-slate-500">Persisted on disk in SmartPark-AI/data/predictions.csv</p>
          </div>
          {history.length > 0 && (
            <button
              type="button"
              onClick={downloadHistoryCsv}
              className="inline-flex items-center gap-2 px-3.5 py-2 bg-[#0B192C] text-white text-xs font-semibold rounded-lg hover:bg-[#152C4E] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              Download Prediction History
            </button>
          )}
        </div>

        {history.length === 0 ? (
          <div className="py-6 text-center text-sm text-slate-500 border border-dashed border-slate-200 rounded-lg">
            No predictions logged yet. Run your first prediction above to populate history.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2 pr-4 font-semibold">Timestamp</th>
                  <th className="py-2 pr-4 font-semibold">Location</th>
                  <th className="py-2 pr-4 font-semibold">Day</th>
                  <th className="py-2 pr-4 font-semibold">Time</th>
                  <th className="py-2 pr-4 font-semibold text-right">Capacity</th>
                  <th className="py-2 pr-4 font-semibold text-right">Prev Occ</th>
                  <th className="py-2 pr-4 font-semibold">Prediction</th>
                  <th className="py-2 pr-4 font-semibold text-right">Confidence</th>
                  <th className="py-2 font-semibold text-right">Est. Available</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {history.slice(0, 20).map((row, idx) => (
                  <tr key={`${row.Timestamp}-${idx}`} className="hover:bg-slate-50">
                    <td className="py-2.5 pr-4 font-mono-num text-slate-600">{row.Timestamp}</td>
                    <td className="py-2.5 pr-4 text-slate-800">{row.Location}</td>
                    <td className="py-2.5 pr-4 text-slate-800">{row.Day}</td>
                    <td className="py-2.5 pr-4 font-mono-num text-slate-700">{row.Entry_Time}</td>
                    <td className="py-2.5 pr-4 font-mono-num text-right text-slate-800">{row.Total_Capacity}</td>
                    <td className="py-2.5 pr-4 font-mono-num text-right text-slate-800">
                      {row.Previous_Occupied} ({row.Previous_Occupancy_Pct}%)
                    </td>
                    <td
                      className={`py-2.5 pr-4 font-semibold ${
                        row.Predicted_Status === 'HIGH' ? 'text-emerald-700' : 'text-red-700'
                      }`}
                    >
                      {row.Predicted_Status}
                    </td>
                    <td className="py-2.5 pr-4 font-mono-num text-right text-slate-800">{row.Confidence_Pct}%</td>
                    <td className="py-2.5 font-mono-num text-right text-slate-800">{row.Estimated_Available}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export const HistoricalDataView: React.FC<{
  records: ParkingRecord[];
  thresholdPct: number;
  dataSourceMode: 'SYNTHETIC' | 'CUSTOM';
  validationSummary: ValidationSummary | null;
  onApplyCustomDataset: (rawText: string, rawRows: Record<string, string>[]) => Promise<void>;
  onResetToSample: () => Promise<void>;
}> = ({
  records,
  thresholdPct,
  dataSourceMode,
  validationSummary,
  onApplyCustomDataset,
  onResetToSample,
}) => {
  const [search, setSearch] = useState('');
  const [locationFilter, setLocationFilter] = useState('ALL');
  const [dayFilter, setDayFilter] = useState('ALL');
  const [dateFilter, setDateFilter] = useState('ALL');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [sortCol, setSortCol] = useState<keyof ParkingRecord>('Date');
  const [sortAsc, setSortAsc] = useState(false);
  const [page, setPage] = useState(1);

  const [uploadPreview, setUploadPreview] = useState<{
    fileName: string;
    rawText: string;
    rawRows: Record<string, string>[];
    cleanedCount: number;
    summary: ValidationSummary;
  } | null>(null);
  const [uploadError, setUploadError] = useState('');

  const uniqueLocations = useMemo(
    () => Array.from(new Set(records.map((r) => r.Location))).sort(),
    [records]
  );

  const uniqueDates = useMemo(
    () => Array.from(new Set(records.map((r) => r.Date))).sort(),
    [records]
  );

  const filteredRecords = useMemo(() => {
    const q = search.trim().toLowerCase();
    const list = records.filter((r) => {
      if (locationFilter !== 'ALL' && r.Location !== locationFilter) return false;
      if (dayFilter !== 'ALL' && r.Day !== dayFilter) return false;
      if (dateFilter !== 'ALL' && r.Date !== dateFilter) return false;
      if (statusFilter !== 'ALL' && r.Availability_Status !== statusFilter) return false;
      if (q) {
        const hay = `${r.Date} ${r.Time} ${r.Location} ${r.Day} ${r.Availability_Status}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });

    list.sort((a, b) => {
      const av = a[sortCol];
      const bv = b[sortCol];
      if (typeof av === 'number' && typeof bv === 'number') {
        return sortAsc ? av - bv : bv - av;
      }
      return sortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    });
    return list;
  }, [records, search, locationFilter, dayFilter, dateFilter, statusFilter, sortCol, sortAsc]);

  const pageSize = 25;
  const totalPages = Math.max(1, Math.ceil(filteredRecords.length / pageSize));
  const currentPage = Math.min(page, totalPages);
  const pageRows = filteredRecords.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  const toggleSort = (col: keyof ParkingRecord) => {
    if (sortCol === col) setSortAsc(!sortAsc);
    else {
      setSortCol(col);
      setSortAsc(true);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    setUploadError('');
    setUploadPreview(null);
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const rawRows = parseCsvText(text);
      const { records: cleaned, summary } = validateAndCleanDataset(rawRows, thresholdPct);
      setUploadPreview({
        fileName: file.name,
        rawText: text,
        rawRows,
        cleanedCount: cleaned.length,
        summary,
      });
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Unable to parse CSV file.');
    }
  };

  const downloadCsv = (content: string, filename: string) => {
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const recordsToCsv = (rows: ParkingRecord[]) => {
    const header =
      'Date,Time,Location,Day,Hour,Total_Capacity,Previous_Occupied,Occupancy_Percentage,Available_Spaces,Availability_Status';
    const body = rows.map((r) =>
      [
        r.Date,
        r.Time,
        r.Location,
        r.Day,
        r.Hour,
        r.Total_Capacity,
        r.Previous_Occupied,
        r.Occupancy_Percentage,
        r.Available_Spaces,
        r.Availability_Status,
      ].join(',')
    );
    return [header, ...body].join('\n') + '\n';
  };

  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-bold text-slate-900">Dataset Management & CSV Upload</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Upload your own campus parking CSV to validate, preview, persist across restarts, and retrain the ML models.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => downloadCsv(CSV_UPLOAD_TEMPLATE, 'smartpark_upload_template.csv')}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-800 text-xs font-medium rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              CSV Template
            </button>
            <button
              type="button"
              onClick={() => downloadCsv(recordsToCsv(records), 'smartpark_active_dataset.csv')}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-800 text-xs font-medium rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              Active Dataset
            </button>
            {dataSourceMode === 'CUSTOM' && (
              <button
                type="button"
                onClick={onResetToSample}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-amber-50 hover:bg-amber-100 text-amber-900 border border-amber-200 text-xs font-medium rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Restore Sample Dataset
              </button>
            )}
          </div>
        </div>

        <div className="flex flex-col sm:flex-row sm:items-center gap-3">
          <label className="inline-flex items-center gap-2 px-4 py-2 bg-[#0B192C] text-white text-xs font-semibold rounded-lg hover:bg-[#152C4E] transition-colors cursor-pointer whitespace-nowrap shrink-0 w-fit">
            <Upload className="w-3.5 h-3.5" />
            Choose CSV File
            <input type="file" accept=".csv,text/csv" onChange={handleFileChange} className="hidden" />
          </label>
          <span className="text-xs text-slate-500">
            Required: Day, Total_Capacity, Previous_Occupied, and Hour or Time (optional: Location, Date, Available_Spaces).
          </span>
        </div>

        {uploadError && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-xs text-red-800 font-medium">
            {uploadError}
          </div>
        )}

        {uploadPreview && (
          <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="text-xs text-slate-800">
                <span className="font-semibold">{uploadPreview.fileName}</span> · {uploadPreview.summary.validRows} valid rows
                ({uploadPreview.summary.droppedRows} dropped) · HIGH {uploadPreview.summary.highCount} · LOW{' '}
                {uploadPreview.summary.lowCount}
              </div>
              <button
                type="button"
                onClick={async () => {
                  await onApplyCustomDataset(uploadPreview.rawText, uploadPreview.rawRows);
                  setUploadPreview(null);
                }}
                className="px-4 py-2 bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-semibold rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                Save & Train Model on Uploaded Data
              </button>
            </div>
            {uploadPreview.summary.issues.length > 0 && (
              <ul className="text-xs text-amber-800 list-disc pl-4 space-y-0.5">
                {uploadPreview.summary.issues.map((iss, i) => (
                  <li key={i}>{iss}</li>
                ))}
              </ul>
            )}
          </div>
        )}

        {validationSummary && validationSummary.droppedRows > 0 && (
          <div className="text-xs text-slate-600">
            Validation note: {validationSummary.droppedRows} invalid/incomplete row(s) excluded from active training.
          </div>
        )}
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h3 className="text-sm font-bold text-slate-900">Historical Parking Records</h3>
          <button
            type="button"
            onClick={() => downloadCsv(recordsToCsv(filteredRecords), 'smartpark_filtered_records.csv')}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 bg-[#0B192C] text-white text-xs font-semibold rounded-lg hover:bg-[#152C4E] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            Download Filtered CSV
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search date, location, day..."
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(1);
              }}
              className="w-full pl-8 pr-3 py-2 text-xs bg-white border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#0B192C]"
            />
          </div>

          <select
            value={locationFilter}
            onChange={(e) => {
              setLocationFilter(e.target.value);
              setPage(1);
            }}
            className="px-3 py-2 text-xs bg-white border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#0B192C]"
          >
            <option value="ALL">All Locations</option>
            {uniqueLocations.map((loc) => (
              <option key={loc} value={loc}>
                {loc}
              </option>
            ))}
          </select>

          <select
            value={dayFilter}
            onChange={(e) => {
              setDayFilter(e.target.value);
              setPage(1);
            }}
            className="px-3 py-2 text-xs bg-white border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#0B192C]"
          >
            <option value="ALL">All Days</option>
            {VALID_DAYS.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>

          <select
            value={dateFilter}
            onChange={(e) => {
              setDateFilter(e.target.value);
              setPage(1);
            }}
            className="px-3 py-2 text-xs bg-white border border-slate-300 rounded-lg font-mono-num focus:outline-none focus:ring-2 focus:ring-[#0B192C]"
          >
            <option value="ALL">All Dates ({uniqueDates.length})</option>
            {uniqueDates.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>

          <select
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value);
              setPage(1);
            }}
            className="px-3 py-2 text-xs bg-white border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#0B192C]"
          >
            <option value="ALL">All Statuses</option>
            <option value="HIGH">HIGH Availability</option>
            <option value="LOW">LOW Availability</option>
          </select>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500">
                {(
                  [
                    ['Date', 'Date'],
                    ['Time', 'Time'],
                    ['Location', 'Location'],
                    ['Day', 'Day'],
                    ['Hour', 'Hour'],
                    ['Total_Capacity', 'Capacity'],
                    ['Previous_Occupied', 'Prev Occupied'],
                    ['Occupancy_Percentage', 'Prev Occ %'],
                    ['Available_Spaces', 'Available'],
                    ['Availability_Status', 'Status'],
                  ] as [keyof ParkingRecord, string][]
                ).map(([key, label]) => (
                  <th key={key} className="py-2.5 pr-4 font-semibold">
                    <button
                      type="button"
                      onClick={() => toggleSort(key)}
                      className="inline-flex items-center gap-1 hover:text-slate-900 cursor-pointer whitespace-nowrap"
                    >
                      {label}
                      <ArrowUpDown className="w-3 h-3" />
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {pageRows.map((r, i) => (
                <tr key={`${r.Date}-${r.Location}-${r.Time}-${i}`} className="hover:bg-slate-50">
                  <td className="py-2.5 pr-4 font-mono-num text-slate-700">{r.Date}</td>
                  <td className="py-2.5 pr-4 font-mono-num text-slate-700">{r.Time}</td>
                  <td className="py-2.5 pr-4 text-slate-800">{r.Location}</td>
                  <td className="py-2.5 pr-4 text-slate-800">{r.Day}</td>
                  <td className="py-2.5 pr-4 font-mono-num text-slate-700">{r.Hour}</td>
                  <td className="py-2.5 pr-4 font-mono-num text-slate-800">{r.Total_Capacity}</td>
                  <td className="py-2.5 pr-4 font-mono-num text-slate-800">{r.Previous_Occupied}</td>
                  <td className="py-2.5 pr-4 font-mono-num text-slate-800">{r.Occupancy_Percentage.toFixed(1)}%</td>
                  <td className="py-2.5 pr-4 font-mono-num text-slate-800">{r.Available_Spaces}</td>
                  <td
                    className={`py-2.5 font-semibold ${
                      r.Availability_Status === 'HIGH' ? 'text-emerald-700' : 'text-red-700'
                    }`}
                  >
                    {r.Availability_Status}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-100 text-xs text-slate-600">
          <span className="tabular-nums">
            Showing {filteredRecords.length === 0 ? 0 : (currentPage - 1) * pageSize + 1}–
            {Math.min(currentPage * pageSize, filteredRecords.length)} of{' '}
            {filteredRecords.length.toLocaleString()} records
          </span>
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={currentPage <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className="px-3 py-1.5 border border-slate-300 rounded-md disabled:opacity-40 hover:bg-slate-50 cursor-pointer"
            >
              Previous
            </button>
            <span className="font-mono-num">
              Page {currentPage} / {totalPages}
            </span>
            <button
              type="button"
              disabled={currentPage >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              className="px-3 py-1.5 border border-slate-300 rounded-md disabled:opacity-40 hover:bg-slate-50 cursor-pointer"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
