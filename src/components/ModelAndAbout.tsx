import React, { useEffect, useState } from 'react';
import { Check, Copy, Download } from 'lucide-react';
import type { TrainedModelsBundle } from '../ml/parkingEngine';
import type { SourceFileItem } from '../types/parking';
import { ConfusionMatrixCard, FeatureImportanceChart, ModelComparisonChart } from './Charts';

export const ModelPerformanceView: React.FC<{
  modelBundle: TrainedModelsBundle;
}> = ({ modelBundle }) => {
  const rf = modelBundle.primaryMetrics;
  const dt = modelBundle.comparisonMetrics;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <div className="text-xs text-slate-500">Algorithm</div>
          <div className="text-sm font-bold text-[#0B192C] mt-1">Random Forest</div>
          <div className="text-xs text-slate-500 mt-1 tabular-nums">Trained {modelBundle.trainedAt}</div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <div className="text-xs text-slate-500">Training Records</div>
          <div className="text-xl font-bold text-[#0B192C] font-mono-num mt-1">
            {modelBundle.trainRecords.toLocaleString()}
          </div>
          <div className="text-xs text-slate-500 mt-1">80% Stratified Split</div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <div className="text-xs text-slate-500">Testing Records</div>
          <div className="text-xl font-bold text-[#0B192C] font-mono-num mt-1">
            {modelBundle.testRecords.toLocaleString()}
          </div>
          <div className="text-xs text-slate-500 mt-1">20% Held-Out Set</div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <div className="text-xs text-slate-500">Accuracy</div>
          <div className="text-xl font-bold text-[#0B192C] font-mono-num mt-1">
            {(rf.accuracy * 100).toFixed(2)}%
          </div>
          <div className="text-xs text-slate-500 mt-1 font-mono-num">DT: {(dt.accuracy * 100).toFixed(2)}%</div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <div className="text-xs text-slate-500">Precision / Recall</div>
          <div className="text-lg font-bold text-[#0B192C] font-mono-num mt-1">
            {(rf.precision * 100).toFixed(1)}% / {(rf.recall * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-slate-500 mt-1">Weighted Test Avg</div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <div className="text-xs text-slate-500">F1-Score</div>
          <div className="text-xl font-bold text-[#0B192C] font-mono-num mt-1">
            {(rf.f1Score * 100).toFixed(2)}%
          </div>
          <div className="text-xs text-slate-500 mt-1 font-mono-num">DT: {(dt.f1Score * 100).toFixed(2)}%</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <ModelComparisonChart rf={rf} dt={dt} />
        <FeatureImportanceChart importances={modelBundle.featureImportances} />
        <ConfusionMatrixCard metrics={rf} />
        <ConfusionMatrixCard metrics={dt} />
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-bold text-slate-900">Evaluation Comparison & Classification Breakdown</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500">
                <th className="py-2 pr-4 font-semibold">Model</th>
                <th className="py-2 pr-4 font-semibold text-right">Accuracy</th>
                <th className="py-2 pr-4 font-semibold text-right">Precision</th>
                <th className="py-2 pr-4 font-semibold text-right">Recall</th>
                <th className="py-2 pr-4 font-semibold text-right">F1-Score</th>
                <th className="py-2 font-semibold text-right">HIGH F1 / LOW F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono-num">
              <tr>
                <td className="py-2.5 pr-4 font-sans font-semibold text-slate-900">{rf.algorithm}</td>
                <td className="py-2.5 pr-4 text-right">{(rf.accuracy * 100).toFixed(2)}%</td>
                <td className="py-2.5 pr-4 text-right">{(rf.precision * 100).toFixed(2)}%</td>
                <td className="py-2.5 pr-4 text-right">{(rf.recall * 100).toFixed(2)}%</td>
                <td className="py-2.5 pr-4 text-right">{(rf.f1Score * 100).toFixed(2)}%</td>
                <td className="py-2.5 text-right">
                  {(rf.perClass.HIGH.f1Score * 100).toFixed(1)}% / {(rf.perClass.LOW.f1Score * 100).toFixed(1)}%
                </td>
              </tr>
              <tr>
                <td className="py-2.5 pr-4 font-sans font-medium text-slate-700">{dt.algorithm}</td>
                <td className="py-2.5 pr-4 text-right">{(dt.accuracy * 100).toFixed(2)}%</td>
                <td className="py-2.5 pr-4 text-right">{(dt.precision * 100).toFixed(2)}%</td>
                <td className="py-2.5 pr-4 text-right">{(dt.recall * 100).toFixed(2)}%</td>
                <td className="py-2.5 pr-4 text-right">{(dt.f1Score * 100).toFixed(2)}%</td>
                <td className="py-2.5 text-right">
                  {(dt.perClass.HIGH.f1Score * 100).toFixed(1)}% / {(dt.perClass.LOW.f1Score * 100).toFixed(1)}%
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="pt-3 border-t border-slate-100">
          <h4 className="text-xs font-semibold text-slate-800 mb-1">
            Why Random Forest Performs Better Than a Single Decision Tree
          </h4>
          <p className="text-xs text-slate-600 leading-relaxed">
            A single Decision Tree relies on one greedy hierarchy of splits, making it sensitive to small variations or noise in historical campus parking intervals. Random Forest builds an ensemble of bootstrapped trees using random feature subsets at each split and averages their probability predictions. This ensemble averaging reduces variance, resists overfitting, and delivers better-calibrated probability confidence scores on unseen arrival scenarios.
          </p>
        </div>
      </div>
    </div>
  );
};

export const AboutProjectView: React.FC = () => {
  const [sourceFiles, setSourceFiles] = useState<SourceFileItem[]>([]);
  const [activeFile, setActiveFile] = useState<string>('SmartPark-AI/app.py');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    fetch('/api/source-files')
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data.files) && data.files.length > 0) {
          setSourceFiles(data.files);
          setActiveFile(data.files[0].path);
        }
      })
      .catch(() => {});
  }, []);

  const currentFile = sourceFiles.find((f) => f.path === activeFile);

  const handleCopy = () => {
    if (!currentFile) return;
    navigator.clipboard.writeText(currentFile.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const handleDownloadFile = () => {
    if (!currentFile) return;
    const blob = new Blob([currentFile.content], { type: 'text/plain;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = currentFile.name.split('/').pop() || currentFile.name;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
        <h2 className="text-lg font-bold text-slate-900">
          SmartPark AI – Intelligent Campus Parking Availability Predictor
        </h2>
        <p className="text-sm text-slate-600 leading-relaxed">
          Built as a complete B.Tech Computer Science (AI/ML) project, SmartPark AI predicts whether campus parking availability will be HIGH or LOW before commuters enter the parking area. Predictions are grounded strictly in historical parking occupancy records and user-entered prior occupancy—without claiming live CCTV, IoT hardware, or real-time GPS feeds.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
          <div>
            <h3 className="text-sm font-semibold text-slate-900 mb-2">Machine Learning Methodology</h3>
            <ul className="space-y-1.5 text-xs text-slate-600 list-disc pl-4">
              <li>Inputs: Hour of day, Day of week, Previous occupied spaces, Total capacity, Previous occupancy %.</li>
              <li>Target labels: HIGH (&gt;30% open, configurable) vs LOW (&le;30% open).</li>
              <li>Leak-free pipeline: 80/20 stratified train/test split performed before encoding or fitting.</li>
              <li>Primary Random Forest Classifier evaluated against Decision Tree Classifier baseline.</li>
              <li>Confidence scores derived from real ensemble class probabilities.</li>
            </ul>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-slate-900 mb-2">Future Improvements</h3>
            <ul className="space-y-1.5 text-xs text-slate-600 list-disc pl-4">
              <li>IoT ultrasonic/magnetic parking bay sensors for automated counts.</li>
              <li>CCTV-based computer vision vehicle entry/exit counting.</li>
              <li>Live parking occupancy streaming and mobile commuter alerts.</li>
              <li>GPS-based turn-by-turn campus parking bay navigation across multiple zones.</li>
              <li>Cloud database storage and time-series forecasting (Prophet / LSTM).</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-slate-900">Python & Streamlit Repository Explorer</h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Complete GitHub-ready Python/Scikit-learn/Streamlit codebase generated in SmartPark-AI/
            </p>
          </div>
          <div className="text-xs font-mono-num bg-slate-100 text-slate-800 px-3 py-1.5 rounded-lg">
            pip install -r requirements.txt &amp;&amp; streamlit run app.py
          </div>
        </div>

        {sourceFiles.length > 0 && (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-1.5">
              {sourceFiles.map((f) => (
                <button
                  key={f.path}
                  type="button"
                  onClick={() => setActiveFile(f.path)}
                  className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    activeFile === f.path
                      ? 'bg-[#0B192C] text-white'
                      : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  {f.name}
                </button>
              ))}
            </div>

            {currentFile && (
              <div className="border border-slate-200 rounded-xl overflow-hidden">
                <div className="bg-slate-50 px-4 py-2.5 border-b border-slate-200 flex items-center justify-between gap-3">
                  <span className="text-xs font-mono-num font-semibold text-slate-700">{currentFile.path}</span>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={handleCopy}
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded-md hover:bg-slate-50 cursor-pointer"
                    >
                      {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                      {copied ? 'Copied' : 'Copy'}
                    </button>
                    <button
                      type="button"
                      onClick={handleDownloadFile}
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-white bg-[#0B192C] rounded-md hover:bg-[#152C4E] cursor-pointer"
                    >
                      <Download className="w-3.5 h-3.5" />
                      Download
                    </button>
                  </div>
                </div>
                <pre className="p-4 text-xs font-mono-num bg-[#0B192C] text-slate-100 overflow-x-auto max-h-96 leading-relaxed">
                  <code>{currentFile.content}</code>
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
