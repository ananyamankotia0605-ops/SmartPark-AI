import React, { useMemo } from 'react';
import { Download } from 'lucide-react';
import type { TrainedModelsBundle } from '../ml/parkingEngine';
import {
  computeAnalyticsSummary,
  filterRecordsByLocation,
  getLocationSnapshots,
  predictParkingAvailability,
} from '../ml/parkingEngine';
import type { ParkingRecord } from '../types/parking';
import {
  AvailabilityByDayChart,
  AvailabilitySplitDonut,
  HourlyOccupancyChart,
  OccupancyDistributionChart,
  WeekdayOccupancyBarChart,
  WeeklyTrendChart,
} from './Charts';

const MetricCard: React.FC<{
  label: string;
  value: string;
  subtext: string;
  accent?: 'default' | 'high' | 'low';
}> = ({ label, value, subtext, accent = 'default' }) => {
  const valueColor =
    accent === 'high' ? 'text-emerald-700' : accent === 'low' ? 'text-red-700' : 'text-[#0B192C]';
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4">
      <div className="text-xs font-medium text-slate-500 mb-1">{label}</div>
      <div className={`text-2xl font-bold tabular-nums tracking-tight ${valueColor}`}>{value}</div>
      <div className="text-xs text-slate-500 mt-1 tabular-nums">{subtext}</div>
    </div>
  );
};

export const DashboardView: React.FC<{
  records: ParkingRecord[];
  modelBundle: TrainedModelsBundle;
  thresholdPct: number;
  selectedLocation: string;
  onSelectLocation: (loc: string) => void;
  onNavigate: (page: 'finder' | 'predict') => void;
}> = ({ records, modelBundle, thresholdPct, selectedLocation, onSelectLocation, onNavigate }) => {
  const locationOptions = useMemo(
    () => ['All Campus Zones', ...Array.from(new Set(records.map((r) => r.Location))).sort()],
    [records]
  );

  const effectiveLocation = locationOptions.includes(selectedLocation)
    ? selectedLocation
    : 'All Campus Zones';

  const scopedRecords = useMemo(
    () => filterRecordsByLocation(records, effectiveLocation),
    [records, effectiveLocation]
  );
  const analytics = useMemo(() => computeAnalyticsSummary(scopedRecords), [scopedRecords]);
  const snapshots = useMemo(() => getLocationSnapshots(scopedRecords), [scopedRecords]);
  const repSnap = snapshots[0];

  const latestScenario = repSnap
    ? predictParkingAvailability(
        modelBundle,
        repSnap.latestDay,
        repSnap.latestHour,
        repSnap.latestOccupied,
        repSnap.totalCapacity,
        repSnap.location
      )
    : null;

  const peakHoursText = analytics.busiestHours
    .map((h) => `${String(h.hour).padStart(2, '0')}:00 (${h.avgOccPct.toFixed(1)}%)`)
    .join(' · ');

  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200 rounded-xl px-5 py-3.5 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-bold text-slate-900">Analytics Dashboard Scope</h2>
          <p className="text-xs text-slate-500">
            Selecting a parking location updates all capacity totals, occupancy metrics, and charts strictly for that zone.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label className="text-xs font-semibold text-slate-700 whitespace-nowrap" htmlFor="dash-loc-select">
            Location:
          </label>
          <select
            id="dash-loc-select"
            value={effectiveLocation}
            onChange={(e) => onSelectLocation(e.target.value)}
            className="px-3 py-1.5 text-xs font-medium bg-slate-50 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#0B192C]"
          >
            {locationOptions.map((loc) => (
              <option key={loc} value={loc}>
                {loc}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Total Parking Capacity"
          value={`${analytics.totalCapacity.toLocaleString()} spaces`}
          subtext={`Scope: ${effectiveLocation}`}
        />
        <MetricCard
          label="Currently Occupied (Latest)"
          value={`${analytics.latestOccupied.toLocaleString()} spaces`}
          subtext={`Historical Avg: ${analytics.avgOccupied} / zone`}
        />
        <MetricCard
          label="Available Parking Spaces"
          value={`${analytics.latestAvailable.toLocaleString()} spaces`}
          subtext={`Historical Avg: ${analytics.avgAvailable} open / zone`}
        />
        <MetricCard
          label="Current Availability Percentage"
          value={`${analytics.latestAvailabilityPct.toFixed(1)}%`}
          subtext={`Historical Avg: ${analytics.avgAvailabilityPct.toFixed(1)}% open`}
        />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <MetricCard
          label="Predicted Availability Status"
          value={latestScenario ? `${latestScenario.predictedStatus} AVAILABILITY` : 'N/A'}
          subtext={
            latestScenario && repSnap
              ? `Model confidence: ${latestScenario.confidencePct.toFixed(1)}% (${repSnap.location})`
              : ''
          }
          accent={latestScenario?.predictedStatus === 'HIGH' ? 'high' : 'low'}
        />
        <MetricCard
          label="Total Historical Records"
          value={analytics.totalRecords.toLocaleString()}
          subtext={`Rule: >${thresholdPct}% open = HIGH`}
        />
        <MetricCard
          label="Model Accuracy (Random Forest)"
          value={`${(modelBundle.primaryMetrics.accuracy * 100).toFixed(2)}%`}
          subtext={`Evaluated on ${modelBundle.testRecords.toLocaleString()} held-out records`}
        />
      </div>

      <div className="bg-[#0B192C] text-white rounded-xl p-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1 text-sm">
          <div className="font-semibold text-sky-300">
            Peak Hour Detection & Arrival Insight ({effectiveLocation})
          </div>
          <div className="text-slate-200 tabular-nums">
            Busiest hours: {peakHoursText || 'N/A'} · {analytics.bestTimeRecommendation}
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={() => onNavigate('finder')}
            className="px-3.5 py-2 bg-white text-[#0B192C] text-xs font-semibold rounded-lg hover:bg-slate-100 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            Parking Finder
          </button>
          <button
            type="button"
            onClick={() => onNavigate('predict')}
            className="px-3.5 py-2 bg-sky-600 text-white text-xs font-semibold rounded-lg hover:bg-sky-500 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            Predict Slot
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <OccupancyDistributionChart histogram={analytics.occupancyHistogram} />
        <AvailabilityByDayChart daySeries={analytics.daySeries} />
        <HourlyOccupancyChart hourlySeries={analytics.hourlySeries} />
        <WeeklyTrendChart weeklySeries={analytics.weeklySeries} />
      </div>

      <AvailabilitySplitDonut highCount={analytics.highCount} lowCount={analytics.lowCount} />
    </div>
  );
};

export const AnalyticsView: React.FC<{
  records: ParkingRecord[];
  selectedLocation: string;
  onSelectLocation: (loc: string) => void;
}> = ({ records, selectedLocation, onSelectLocation }) => {
  const locationOptions = useMemo(
    () => ['All Campus Zones', ...Array.from(new Set(records.map((r) => r.Location))).sort()],
    [records]
  );

  const effectiveLocation = locationOptions.includes(selectedLocation)
    ? selectedLocation
    : 'All Campus Zones';

  const scopedRecords = useMemo(
    () => filterRecordsByLocation(records, effectiveLocation),
    [records, effectiveLocation]
  );
  const analytics = useMemo(() => computeAnalyticsSummary(scopedRecords), [scopedRecords]);

  const downloadAnalyticsCsv = () => {
    const rows = [
      ['Scope', effectiveLocation],
      ['Total Records', String(analytics.totalRecords)],
      ['Total Zone Capacity', String(analytics.totalCapacity)],
      ['Average Available Spaces', String(analytics.avgAvailable)],
      ['Average Occupancy (%)', String(analytics.avgOccupancyPct)],
      ['HIGH Availability Count', String(analytics.highCount)],
      ['HIGH Availability (%)', String(analytics.highPct)],
      ['LOW Availability Count', String(analytics.lowCount)],
      ['LOW Availability (%)', String(analytics.lowPct)],
      ['Best Time Recommendation', `"${analytics.bestTimeRecommendation}"`],
    ];
    const csv = rows.map((r) => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `smartpark_analytics_${effectiveLocation.toLowerCase().replace(/\s+/g, '_')}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const busiestDays = [...analytics.daySeries].sort((a, b) => b.avgOccPct - a.avgOccPct);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-bold text-slate-900">Parking Analytics & Demand Patterns</h2>
          <p className="text-sm text-slate-600">
            Computed dynamically from {analytics.totalRecords.toLocaleString()} records ({effectiveLocation}).
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <select
            aria-label="Filter Analytics Location"
            value={effectiveLocation}
            onChange={(e) => onSelectLocation(e.target.value)}
            className="px-3 py-2 text-xs font-medium bg-white border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#0B192C]"
          >
            {locationOptions.map((loc) => (
              <option key={loc} value={loc}>
                {loc}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={downloadAnalyticsCsv}
            className="inline-flex items-center gap-2 px-4 py-2 bg-[#0B192C] text-white text-xs font-semibold rounded-lg hover:bg-[#152C4E] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            Download Analytics CSV
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Average Occupancy Percentage"
          value={`${analytics.avgOccupancyPct.toFixed(1)}%`}
          subtext={`Scope: ${effectiveLocation}`}
        />
        <MetricCard
          label="Average Available Spaces"
          value={`${analytics.avgAvailable} spaces`}
          subtext={`Avg lot size: ${analytics.avgCapacity}`}
        />
        <MetricCard
          label="HIGH Availability Frequency"
          value={`${analytics.highPct.toFixed(1)}%`}
          subtext={`${analytics.highCount.toLocaleString()} intervals`}
          accent="high"
        />
        <MetricCard
          label="LOW Availability Frequency"
          value={`${analytics.lowPct.toFixed(1)}%`}
          subtext={`${analytics.lowCount.toLocaleString()} intervals`}
          accent="low"
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-slate-900 mb-3">Busiest Parking Hours</h3>
          <div className="space-y-2.5">
            {analytics.busiestHours.map((h) => (
              <div key={h.hour} className="flex items-center justify-between text-sm border-b border-slate-100 pb-2 last:border-0 last:pb-0">
                <span className="font-medium text-slate-700 font-mono-num">{String(h.hour).padStart(2, '0')}:00</span>
                <span className="font-semibold text-red-700 font-mono-num">{h.avgOccPct.toFixed(1)}% occupied</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-slate-900 mb-3">Least Busy Parking Hours</h3>
          <div className="space-y-2.5">
            {analytics.leastBusyHours.map((h) => (
              <div key={h.hour} className="flex items-center justify-between text-sm border-b border-slate-100 pb-2 last:border-0 last:pb-0">
                <span className="font-medium text-slate-700 font-mono-num">{String(h.hour).padStart(2, '0')}:00</span>
                <span className="font-semibold text-emerald-700 font-mono-num">{h.avgOccPct.toFixed(1)}% occupied</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-slate-900 mb-3">Busiest Weekdays</h3>
          <div className="space-y-2.5">
            {busiestDays.slice(0, 3).map((d) => (
              <div key={d.day} className="flex items-center justify-between text-sm border-b border-slate-100 pb-2 last:border-0 last:pb-0">
                <span className="font-medium text-slate-700">{d.day}</span>
                <span className="font-semibold text-slate-900 font-mono-num">{d.avgOccPct.toFixed(1)}% occupied</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-4 text-sm text-emerald-900">
        <span className="font-semibold">Best Time Recommendation:</span> {analytics.bestTimeRecommendation}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <WeekdayOccupancyBarChart daySeries={analytics.daySeries} />
        <HourlyOccupancyChart hourlySeries={analytics.hourlySeries} />
        <AvailabilityByDayChart daySeries={analytics.daySeries} />
        <WeeklyTrendChart weeklySeries={analytics.weeklySeries} />
      </div>
    </div>
  );
};
