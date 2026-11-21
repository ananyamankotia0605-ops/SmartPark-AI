import React, { useState } from 'react';
import type {
  AnalyticsSummary,
  AvailabilityStatus,
  FeatureImportanceItem,
  ModelMetrics,
} from '../types/parking';

const NAVY = '#0B192C';
const NAVY_MID = '#1E3A8A';
const SKY = '#0284C7';
const GREEN = '#16A34A';
const RED = '#DC2626';

export const OccupancyDistributionChart: React.FC<{
  histogram: AnalyticsSummary['occupancyHistogram'];
}> = ({ histogram }) => {
  const [hovered, setHovered] = useState<number | null>(null);
  const maxVal = Math.max(1, ...histogram.map((b) => b.high + b.low));
  const width = 560;
  const height = 240;
  const pad = { top: 20, right: 16, bottom: 38, left: 42 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const barW = plotW / Math.max(1, histogram.length);

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5">
      <div className="flex items-center justify-between gap-4 mb-3">
        <h3 className="text-sm font-semibold text-slate-900">Parking Occupancy Distribution</h3>
        <div className="flex items-center gap-4 text-xs text-slate-600">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-emerald-600 inline-block" />
            HIGH
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-red-600 inline-block" />
            LOW
          </span>
        </div>
      </div>

      <div className="relative">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-56 overflow-visible">
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const y = pad.top + plotH * (1 - t);
            const val = Math.round(maxVal * t);
            return (
              <g key={t}>
                <line x1={pad.left} x2={width - pad.right} y1={y} y2={y} stroke="#E2E8F0" strokeDasharray="3 3" />
                <text x={pad.left - 8} y={y + 4} textAnchor="end" fontSize="10" fill="#64748B" className="tabular-nums">
                  {val}
                </text>
              </g>
            );
          })}

          {histogram.map((b, idx) => {
            const x = pad.left + idx * barW + 4;
            const innerW = Math.max(8, barW - 8);
            const lowH = (b.low / maxVal) * plotH;
            const highH = (b.high / maxVal) * plotH;
            const yLow = pad.top + plotH - lowH;
            const yHigh = yLow - highH;

            return (
              <g
                key={b.bucket}
                onMouseEnter={() => setHovered(idx)}
                onMouseLeave={() => setHovered(null)}
                className="cursor-pointer"
              >
                {hovered === idx && (
                  <rect
                    x={pad.left + idx * barW}
                    y={pad.top}
                    width={barW}
                    height={plotH}
                    fill="#F1F5F9"
                    rx={4}
                  />
                )}
                <rect x={x} y={yHigh} width={innerW} height={Math.max(0, highH)} fill={GREEN} rx={2} />
                <rect x={x} y={yLow} width={innerW} height={Math.max(0, lowH)} fill={RED} rx={2} />
                <text
                  x={x + innerW / 2}
                  y={height - 14}
                  textAnchor="middle"
                  fontSize="9.5"
                  fill="#475569"
                  className="tabular-nums"
                >
                  {b.min}–{b.max}%
                </text>
              </g>
            );
          })}
        </svg>

        {hovered !== null && histogram[hovered] && (
          <div className="mt-2 px-3 py-1.5 bg-slate-900 text-white text-xs rounded-lg flex items-center justify-between tabular-nums">
            <span>Occupancy Bin: {histogram[hovered].bucket}</span>
            <span>
              HIGH: {histogram[hovered].high} · LOW: {histogram[hovered].low}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export const AvailabilityByDayChart: React.FC<{
  daySeries: AnalyticsSummary['daySeries'];
}> = ({ daySeries }) => {
  const [hovered, setHovered] = useState<number | null>(null);
  const maxVal = Math.max(1, ...daySeries.flatMap((d) => [d.highCount, d.lowCount]));
  const width = 560;
  const height = 240;
  const pad = { top: 20, right: 16, bottom: 38, left: 42 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const groupW = plotW / Math.max(1, daySeries.length);

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5">
      <div className="flex items-center justify-between gap-4 mb-3">
        <h3 className="text-sm font-semibold text-slate-900">Parking Availability by Day</h3>
        <div className="flex items-center gap-4 text-xs text-slate-600">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-emerald-600 inline-block" />
            HIGH
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-red-600 inline-block" />
            LOW
          </span>
        </div>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-56 overflow-visible">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const y = pad.top + plotH * (1 - t);
          return (
            <g key={t}>
              <line x1={pad.left} x2={width - pad.right} y1={y} y2={y} stroke="#E2E8F0" strokeDasharray="3 3" />
              <text x={pad.left - 8} y={y + 4} textAnchor="end" fontSize="10" fill="#64748B" className="tabular-nums">
                {Math.round(maxVal * t)}
              </text>
            </g>
          );
        })}

        {daySeries.map((d, idx) => {
          const gx = pad.left + idx * groupW;
          const barW = Math.min(20, (groupW - 14) / 2);
          const xHigh = gx + (groupW - barW * 2 - 4) / 2;
          const xLow = xHigh + barW + 4;
          const hHigh = (d.highCount / maxVal) * plotH;
          const hLow = (d.lowCount / maxVal) * plotH;

          return (
            <g
              key={d.day}
              onMouseEnter={() => setHovered(idx)}
              onMouseLeave={() => setHovered(null)}
              className="cursor-pointer"
            >
              {hovered === idx && (
                <rect x={gx + 2} y={pad.top} width={groupW - 4} height={plotH} fill="#F1F5F9" rx={4} />
              )}
              <rect x={xHigh} y={pad.top + plotH - hHigh} width={barW} height={hHigh} fill={GREEN} rx={2} />
              <rect x={xLow} y={pad.top + plotH - hLow} width={barW} height={hLow} fill={RED} rx={2} />
              <text
                x={gx + groupW / 2}
                y={height - 14}
                textAnchor="middle"
                fontSize="10.5"
                fill="#475569"
              >
                {d.day.slice(0, 3)}
              </text>
            </g>
          );
        })}
      </svg>

      {hovered !== null && daySeries[hovered] && (
        <div className="mt-2 px-3 py-1.5 bg-slate-900 text-white text-xs rounded-lg flex items-center justify-between tabular-nums">
          <span>{daySeries[hovered].day}</span>
          <span>
            HIGH: {daySeries[hovered].highCount} · LOW: {daySeries[hovered].lowCount} · Avg Occ: {daySeries[hovered].avgOccPct.toFixed(1)}%
          </span>
        </div>
      )}
    </div>
  );
};

export const HourlyOccupancyChart: React.FC<{
  hourlySeries: AnalyticsSummary['hourlySeries'];
}> = ({ hourlySeries }) => {
  const [hovered, setHovered] = useState<number | null>(null);
  const width = 560;
  const height = 240;
  const pad = { top: 20, right: 20, bottom: 38, left: 42 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  if (hourlySeries.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-5 text-sm text-slate-500">
        No hourly occupancy records available.
      </div>
    );
  }

  const pts = hourlySeries.map((pt, idx) => {
    const x =
      hourlySeries.length === 1
        ? pad.left + plotW / 2
        : pad.left + (idx / (hourlySeries.length - 1)) * plotW;
    const y = pad.top + plotH * (1 - pt.avgOccPct / 100);
    return { ...pt, x, y };
  });

  const linePath = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');
  const areaPath = `${linePath} L ${pts[pts.length - 1].x.toFixed(1)} ${(pad.top + plotH).toFixed(1)} L ${pts[0].x.toFixed(1)} ${(pad.top + plotH).toFixed(1)} Z`;

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5">
      <div className="flex items-center justify-between gap-4 mb-3">
        <h3 className="text-sm font-semibold text-slate-900">Parking Occupancy by Hour</h3>
        <span className="text-xs text-slate-500">24-hour campus schedule</span>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-56 overflow-visible">
        {[0, 25, 50, 75, 100].map((val) => {
          const y = pad.top + plotH * (1 - val / 100);
          return (
            <g key={val}>
              <line x1={pad.left} x2={width - pad.right} y1={y} y2={y} stroke="#E2E8F0" strokeDasharray="3 3" />
              <text x={pad.left - 8} y={y + 4} textAnchor="end" fontSize="10" fill="#64748B" className="tabular-nums">
                {val}%
              </text>
            </g>
          );
        })}

        <path d={areaPath} fill="rgba(2, 132, 199, 0.12)" />
        <path d={linePath} fill="none" stroke={NAVY_MID} strokeWidth="2.5" strokeLinejoin="round" />

        {pts.map((p, idx) => (
          <g
            key={p.hour}
            onMouseEnter={() => setHovered(idx)}
            onMouseLeave={() => setHovered(null)}
            className="cursor-pointer"
          >
            <circle cx={p.x} cy={p.y} r={hovered === idx ? 6 : 4} fill={hovered === idx ? NAVY : SKY} stroke="#FFFFFF" strokeWidth="1.5" />
            <text x={p.x} y={height - 14} textAnchor="middle" fontSize="10" fill="#475569" className="tabular-nums">
              {String(p.hour).padStart(2, '0')}h
            </text>
          </g>
        ))}
      </svg>

      {hovered !== null && pts[hovered] && (
        <div className="mt-2 px-3 py-1.5 bg-slate-900 text-white text-xs rounded-lg flex items-center justify-between tabular-nums">
          <span>Hour: {String(pts[hovered].hour).padStart(2, '0')}:00</span>
          <span>Avg Occupancy: {pts[hovered].avgOccPct.toFixed(1)}% ({pts[hovered].records} records)</span>
        </div>
      )}
    </div>
  );
};

export const WeeklyTrendChart: React.FC<{
  weeklySeries: AnalyticsSummary['weeklySeries'];
}> = ({ weeklySeries }) => {
  const [hovered, setHovered] = useState<number | null>(null);
  const width = 560;
  const height = 240;
  const pad = { top: 20, right: 20, bottom: 38, left: 42 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  if (weeklySeries.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-5 text-sm text-slate-500">
        No date-indexed records available for weekly trends.
      </div>
    );
  }

  const pts = weeklySeries.map((pt, idx) => {
    const x =
      weeklySeries.length === 1
        ? pad.left + plotW / 2
        : pad.left + (idx / (weeklySeries.length - 1)) * plotW;
    const y = pad.top + plotH * (1 - pt.avgOccPct / 100);
    return { ...pt, x, y };
  });

  const linePath = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5">
      <div className="flex items-center justify-between gap-4 mb-3">
        <h3 className="text-sm font-semibold text-slate-900">Weekly Parking Trends</h3>
        <span className="text-xs text-slate-500 tabular-nums">{weeklySeries.length} weeks tracked</span>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-56 overflow-visible">
        {[0, 25, 50, 75, 100].map((val) => {
          const y = pad.top + plotH * (1 - val / 100);
          return (
            <g key={val}>
              <line x1={pad.left} x2={width - pad.right} y1={y} y2={y} stroke="#E2E8F0" strokeDasharray="3 3" />
              <text x={pad.left - 8} y={y + 4} textAnchor="end" fontSize="10" fill="#64748B" className="tabular-nums">
                {val}%
              </text>
            </g>
          );
        })}

        <path d={linePath} fill="none" stroke={NAVY} strokeWidth="2.5" strokeLinejoin="round" />

        {pts.map((p, idx) => (
          <g
            key={p.weekStart}
            onMouseEnter={() => setHovered(idx)}
            onMouseLeave={() => setHovered(null)}
            className="cursor-pointer"
          >
            <circle cx={p.x} cy={p.y} r={hovered === idx ? 6 : 4} fill={hovered === idx ? NAVY : SKY} stroke="#FFFFFF" strokeWidth="1.5" />
            {(idx % Math.max(1, Math.ceil(pts.length / 6)) === 0 || idx === pts.length - 1) && (
              <text x={p.x} y={height - 14} textAnchor="middle" fontSize="9.5" fill="#475569" className="tabular-nums">
                {p.weekStart.slice(5)}
              </text>
            )}
          </g>
        ))}
      </svg>

      {hovered !== null && pts[hovered] && (
        <div className="mt-2 px-3 py-1.5 bg-slate-900 text-white text-xs rounded-lg flex items-center justify-between tabular-nums">
          <span>Week of {pts[hovered].weekStart}</span>
          <span>Avg Occupancy: {pts[hovered].avgOccPct.toFixed(1)}% ({pts[hovered].records} records)</span>
        </div>
      )}
    </div>
  );
};

export const AvailabilitySplitDonut: React.FC<{
  highCount: number;
  lowCount: number;
}> = ({ highCount, lowCount }) => {
  const total = Math.max(1, highCount + lowCount);
  const highPct = (highCount / total) * 100;
  const lowPct = (lowCount / total) * 100;
  const radius = 56;
  const circumference = 2 * Math.PI * radius;
  const highArc = (highPct / 100) * circumference;

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between">
      <h3 className="text-sm font-semibold text-slate-900 mb-3">Low vs High Availability Distribution</h3>
      <div className="flex flex-col sm:flex-row items-center gap-6 my-auto">
        <div className="relative w-40 h-40 shrink-0 flex items-center justify-center">
          <svg viewBox="0 0 150 150" className="w-full h-full -rotate-90">
            <circle cx="75" cy="75" r={radius} fill="none" stroke={RED} strokeWidth="20" />
            <circle
              cx="75"
              cy="75"
              r={radius}
              fill="none"
              stroke={GREEN}
              strokeWidth="20"
              strokeDasharray={`${highArc} ${circumference}`}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
            <span className="text-xl font-bold text-slate-900 tabular-nums">{total.toLocaleString()}</span>
            <span className="text-[11px] text-slate-500">Total Records</span>
          </div>
        </div>

        <div className="flex-1 w-full space-y-3">
          <div className="flex items-center justify-between py-2 border-b border-slate-100">
            <div className="flex items-center gap-2 text-sm font-medium text-slate-800">
              <span className="w-3 h-3 rounded-sm bg-emerald-600 inline-block" />
              HIGH Availability
            </div>
            <div className="text-sm font-semibold text-slate-900 tabular-nums">
              {highCount.toLocaleString()} ({highPct.toFixed(1)}%)
            </div>
          </div>
          <div className="flex items-center justify-between py-2">
            <div className="flex items-center gap-2 text-sm font-medium text-slate-800">
              <span className="w-3 h-3 rounded-sm bg-red-600 inline-block" />
              LOW Availability
            </div>
            <div className="text-sm font-semibold text-slate-900 tabular-nums">
              {lowCount.toLocaleString()} ({lowPct.toFixed(1)}%)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export const WeekdayOccupancyBarChart: React.FC<{
  daySeries: AnalyticsSummary['daySeries'];
}> = ({ daySeries }) => {
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5">
      <h3 className="text-sm font-semibold text-slate-900 mb-4">Weekly Occupancy Comparison</h3>
      <div className="space-y-3">
        {daySeries.map((d) => {
          const barColor = d.avgOccPct >= 70 ? 'bg-red-600' : d.avgOccPct >= 50 ? 'bg-amber-500' : 'bg-emerald-600';
          return (
            <div key={d.day} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="font-medium text-slate-700">{d.day}</span>
                <span className="font-mono-num text-slate-900 font-semibold">{d.avgOccPct.toFixed(1)}%</span>
              </div>
              <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
                <div className={`h-full ${barColor} rounded-full`} style={{ width: `${Math.min(100, d.avgOccPct)}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export const FeatureImportanceChart: React.FC<{
  importances: FeatureImportanceItem[];
}> = ({ importances }) => {
  const maxImp = Math.max(1, ...importances.map((i) => i.importancePct));
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5">
      <h3 className="text-sm font-semibold text-slate-900 mb-4">Feature Importance (Random Forest)</h3>
      <div className="space-y-3.5">
        {importances.map((item) => (
          <div key={item.feature} className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="font-medium text-slate-800">{item.label}</span>
              <span className="font-mono-num font-semibold text-slate-900">{item.importancePct.toFixed(1)}%</span>
            </div>
            <div className="w-full h-3 bg-slate-100 rounded-md overflow-hidden">
              <div
                className="h-full bg-[#0B192C] rounded-md transition-all duration-300"
                style={{ width: `${(item.importancePct / maxImp) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export const ModelComparisonChart: React.FC<{
  rf: ModelMetrics;
  dt: ModelMetrics;
}> = ({ rf, dt }) => {
  const items = [
    { label: 'Accuracy', rf: rf.accuracy * 100, dt: dt.accuracy * 100 },
    { label: 'Precision', rf: rf.precision * 100, dt: dt.precision * 100 },
    { label: 'Recall', rf: rf.recall * 100, dt: dt.recall * 100 },
    { label: 'F1-Score', rf: rf.f1Score * 100, dt: dt.f1Score * 100 },
  ];

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-slate-900">Random Forest vs Decision Tree</h3>
        <div className="flex items-center gap-4 text-xs text-slate-600">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-[#0B192C] inline-block" />
            Random Forest
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-sky-600 inline-block" />
            Decision Tree
          </span>
        </div>
      </div>

      <div className="space-y-4">
        {items.map((m) => (
          <div key={m.label} className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="font-medium text-slate-700">{m.label}</span>
              <span className="font-mono-num text-slate-700">
                RF {m.rf.toFixed(1)}% · DT {m.dt.toFixed(1)}%
              </span>
            </div>
            <div className="space-y-1">
              <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
                <div className="h-full bg-[#0B192C] rounded-full" style={{ width: `${m.rf}%` }} />
              </div>
              <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
                <div className="h-full bg-sky-600 rounded-full" style={{ width: `${m.dt}%` }} />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export const ConfusionMatrixCard: React.FC<{
  metrics: ModelMetrics;
}> = ({ metrics }) => {
  const [[hh, hl], [lh, ll]] = metrics.confusionMatrix;
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5">
      <h3 className="text-sm font-semibold text-slate-900 mb-3">{metrics.algorithm} Confusion Matrix</h3>
      <div className="grid grid-cols-3 gap-2 text-center text-xs">
        <div />
        <div className="font-semibold text-slate-500 py-1">Pred HIGH</div>
        <div className="font-semibold text-slate-500 py-1">Pred LOW</div>

        <div className="font-semibold text-slate-500 flex items-center justify-end pr-2">Actual HIGH</div>
        <div className="bg-[#0B192C] text-white rounded-lg py-4 font-mono-num">
          <div className="text-lg font-bold">{hh}</div>
          <div className="text-[11px] text-sky-200">True HIGH</div>
        </div>
        <div className="bg-slate-100 text-slate-800 rounded-lg py-4 font-mono-num">
          <div className="text-lg font-bold">{hl}</div>
          <div className="text-[11px] text-slate-500">False LOW</div>
        </div>

        <div className="font-semibold text-slate-500 flex items-center justify-end pr-2">Actual LOW</div>
        <div className="bg-slate-100 text-slate-800 rounded-lg py-4 font-mono-num">
          <div className="text-lg font-bold">{lh}</div>
          <div className="text-[11px] text-slate-500">False HIGH</div>
        </div>
        <div className="bg-[#0B192C] text-white rounded-lg py-4 font-mono-num">
          <div className="text-lg font-bold">{ll}</div>
          <div className="text-[11px] text-sky-200">True LOW</div>
        </div>
      </div>
    </div>
  );
};

export const AvailabilityGauge: React.FC<{
  availablePct: number;
  status: AvailabilityStatus;
}> = ({ availablePct, status }) => {
  const clamped = Math.max(0, Math.min(100, availablePct));
  const angle = -90 + (clamped / 100) * 180;
  const strokeColor = status === 'HIGH' ? GREEN : RED;

  return (
    <div className="flex flex-col items-center justify-center p-4 bg-slate-50 border border-slate-200 rounded-xl">
      <svg viewBox="0 0 200 120" className="w-48 h-28">
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke="#E2E8F0"
          strokeWidth="16"
          strokeLinecap="round"
        />
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke={strokeColor}
          strokeWidth="16"
          strokeLinecap="round"
          strokeDasharray={`${(clamped / 100) * 251.3} 251.3`}
        />
        <g transform={`translate(100, 100) rotate(${angle})`}>
          <line x1="0" y1="0" x2="0" y2="-62" stroke={NAVY} strokeWidth="3" strokeLinecap="round" />
          <circle cx="0" cy="0" r="5" fill={NAVY} />
        </g>
        <text x="100" y="90" textAnchor="middle" fontSize="18" fontWeight="700" fill={NAVY} className="tabular-nums">
          {clamped.toFixed(1)}%
        </text>
      </svg>
      <span className="text-xs font-medium text-slate-600 mt-1">Estimated Open Spaces</span>
    </div>
  );
};
