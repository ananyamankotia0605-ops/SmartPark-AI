import React, { useEffect, useMemo, useState } from 'react';
import { MapPin } from 'lucide-react';
import type { TrainedModelsBundle } from '../ml/parkingEngine';
import {
  VALID_DAYS,
  forecastParkingHours,
  getLocationSnapshots,
  validatePredictionInput,
} from '../ml/parkingEngine';
import type { DayName, ParkingRecord } from '../types/parking';
import { AvailabilityGauge } from './Charts';

export const ParkingFinderView: React.FC<{
  records: ParkingRecord[];
  modelBundle: TrainedModelsBundle;
  selectedLocation: string;
  onSelectLocation: (loc: string) => void;
  onOpenPredictWorkbench: () => void;
}> = ({
  records,
  modelBundle,
  selectedLocation,
  onSelectLocation,
  onOpenPredictWorkbench,
}) => {
  const snapshots = useMemo(() => getLocationSnapshots(records), [records]);
  const activeSnap =
    snapshots.find((s) => s.location === selectedLocation) ?? snapshots[0];

  const [day, setDay] = useState<DayName>(activeSnap?.latestDay ?? 'Monday');
  const [startHour, setStartHour] = useState<number>(9);
  const [observedOccupied, setObservedOccupied] = useState<string>(
    String(activeSnap?.latestOccupied ?? 150)
  );

  // When location changes, update only that location's capacity-bounded occupancy default
  useEffect(() => {
    if (activeSnap) {
      setObservedOccupied(String(activeSnap.latestOccupied));
    }
  }, [activeSnap?.location, activeSnap?.latestOccupied]);

  const parsedOcc = Number(observedOccupied);
  const cap = activeSnap?.totalCapacity ?? 250;
  const inputCheck = validatePredictionInput(
    day,
    `${String(startHour).padStart(2, '0')}:00`,
    cap,
    parsedOcc,
    activeSnap?.location ?? 'Main Gate Lot'
  );

  const forecastSteps = useMemo(() => {
    if (!activeSnap || !inputCheck.valid) return [];
    return forecastParkingHours(
      modelBundle,
      activeSnap.location,
      day,
      startHour,
      parsedOcc,
      activeSnap.totalCapacity,
      4
    );
  }, [activeSnap, inputCheck.valid, modelBundle, day, startHour, parsedOcc]);

  const firstStep = forecastSteps[0];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-bold text-slate-900">Campus Parking Finder</h2>
          <p className="text-xs sm:text-sm text-slate-600">
            Select a campus parking zone to inspect its dedicated capacity and run a 4-hour ML arrival forecast.
          </p>
        </div>
        <button
          type="button"
          onClick={onOpenPredictWorkbench}
          className="px-4 py-2 bg-[#0B192C] text-white text-xs font-semibold rounded-lg hover:bg-[#152C4E] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          Open Single-Slot ML Workbench
        </button>
      </div>

      {/* 4 Campus Location Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {snapshots.map((snap) => {
          const isSelected = activeSnap?.location === snap.location;
          const isHigh = snap.latestStatus === 'HIGH';
          return (
            <button
              key={snap.location}
              type="button"
              onClick={() => onSelectLocation(snap.location)}
              className={`text-left rounded-xl p-4 border transition-colors cursor-pointer ${
                isSelected
                  ? 'bg-[#0B192C] text-white border-[#0B192C]'
                  : 'bg-white text-slate-900 border-slate-200 hover:border-slate-300'
              }`}
            >
              <div className="flex items-center justify-between gap-2 mb-2">
                <span className="inline-flex items-center gap-1.5 text-xs font-semibold truncate">
                  <MapPin className={`w-3.5 h-3.5 shrink-0 ${isSelected ? 'text-sky-300' : 'text-slate-500'}`} />
                  {snap.location}
                </span>
                <span
                  className={`text-[11px] font-bold ${
                    isSelected
                      ? isHigh
                        ? 'text-emerald-300'
                        : 'text-red-300'
                      : isHigh
                        ? 'text-emerald-700'
                        : 'text-red-700'
                  }`}
                >
                  {snap.latestStatus}
                </span>
              </div>

              <div className="text-2xl font-bold font-mono-num">
                {snap.latestAvailable} <span className="text-xs font-normal opacity-80">/ {snap.totalCapacity} open</span>
              </div>

              <div className={`text-xs mt-2 font-mono-num ${isSelected ? 'text-slate-300' : 'text-slate-500'}`}>
                Occupied: {snap.latestOccupied} ({snap.latestOccupancyPct.toFixed(1)}%)
              </div>
              <div className={`text-[11px] mt-0.5 ${isSelected ? 'text-sky-200' : 'text-slate-400'}`}>
                Latest: {snap.latestDay} {snap.latestTime} · {snap.recordCount.toLocaleString()} records
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Location Detail & Chained 4-Hour Forecast */}
      {activeSnap && (
        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-200">
            <div>
              <h3 className="text-base font-bold text-slate-900">{activeSnap.location} — Availability Forecast</h3>
              <p className="text-xs text-slate-500 mt-0.5">
                Dedicated capacity: {activeSnap.totalCapacity} spaces · Forecast chains only prior ML-estimated occupancy (zero future actuals).
              </p>
            </div>
            <div className="w-full sm:w-64">
              <label className="sr-only" htmlFor="finder-location-select">
                Parking Location
              </label>
              <select
                id="finder-location-select"
                value={activeSnap.location}
                onChange={(e) => onSelectLocation(e.target.value)}
                className="w-full px-3 py-2 text-xs font-medium bg-slate-50 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#0B192C]"
              >
                {snapshots.map((s) => (
                  <option key={s.location} value={s.location}>
                    {s.location} ({s.totalCapacity} spaces)
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5" htmlFor="finder-day">
                Arrival Day
              </label>
              <select
                id="finder-day"
                value={day}
                onChange={(e) => setDay(e.target.value as DayName)}
                className="w-full px-3 py-2 text-xs bg-white border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#0B192C]"
              >
                {VALID_DAYS.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5" htmlFor="finder-hour">
                Starting Hour (24h)
              </label>
              <select
                id="finder-hour"
                value={startHour}
                onChange={(e) => setStartHour(Number(e.target.value))}
                className="w-full px-3 py-2 text-xs bg-white border border-slate-300 rounded-lg font-mono-num focus:outline-none focus:ring-2 focus:ring-[#0B192C]"
              >
                {Array.from({ length: 17 }, (_, i) => i + 6).map((hr) => (
                  <option key={hr} value={hr}>
                    {String(hr).padStart(2, '0')}:00
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5" htmlFor="finder-occ">
                Current Observed Occupied (Max {activeSnap.totalCapacity})
              </label>
              <input
                id="finder-occ"
                type="number"
                min={0}
                max={activeSnap.totalCapacity}
                step={1}
                value={observedOccupied}
                onChange={(e) => setObservedOccupied(e.target.value)}
                className="w-full px-3 py-2 text-xs bg-white border border-slate-300 rounded-lg font-mono-num focus:outline-none focus:ring-2 focus:ring-[#0B192C]"
              />
            </div>
          </div>

          {!inputCheck.valid ? (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-xs text-red-800 font-medium">
              {inputCheck.error}
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-center">
              <div className="lg:col-span-8 overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500">
                      <th className="py-2 pr-3 font-semibold">Forecast Window</th>
                      <th className="py-2 pr-3 font-semibold text-right">Chained Prior Input</th>
                      <th className="py-2 pr-3 font-semibold">ML Status</th>
                      <th className="py-2 pr-3 font-semibold text-right">Confidence</th>
                      <th className="py-2 pr-3 font-semibold text-right">Est. Occupied</th>
                      <th className="py-2 font-semibold text-right">Est. Available</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono-num">
                    {forecastSteps.map((step) => (
                      <tr key={`${step.stepOffset}-${step.timeLabel}`} className="hover:bg-slate-50">
                        <td className="py-2.5 pr-3 font-sans text-slate-800">
                          {step.day.slice(0, 3)} {step.timeLabel}
                        </td>
                        <td className="py-2.5 pr-3 text-right text-slate-500">
                          {step.inputPrevOccupied} / {activeSnap.totalCapacity}
                        </td>
                        <td
                          className={`py-2.5 pr-3 font-sans font-bold ${
                            step.predictedStatus === 'HIGH' ? 'text-emerald-700' : 'text-red-700'
                          }`}
                        >
                          {step.predictedStatus}
                        </td>
                        <td className="py-2.5 pr-3 text-right text-slate-800">
                          {step.confidencePct.toFixed(1)}%
                        </td>
                        <td className="py-2.5 pr-3 text-right text-slate-800">
                          {step.estimatedOccupied}
                        </td>
                        <td className="py-2.5 text-right font-semibold text-slate-900">
                          {step.estimatedAvailable} ({step.estimatedAvailabilityPct.toFixed(1)}%)
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {firstStep && (
                <div className="lg:col-span-4 space-y-3">
                  <AvailabilityGauge
                    availablePct={firstStep.estimatedAvailabilityPct}
                    status={firstStep.predictedStatus}
                  />
                  <div className="text-xs text-slate-600 leading-relaxed">
                    {firstStep.recommendation}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
