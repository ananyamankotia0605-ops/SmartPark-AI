import type {
  AnalyticsSummary,
  AvailabilityStatus,
  DayName,
  DayStat,
  FeatureImportanceItem,
  ForecastStep,
  HourStat,
  LocationSnapshot,
  ModelMetrics,
  ParkingRecord,
  PredictionResult,
  ValidationSummary,
} from '../types/parking';

export const VALID_DAYS: DayName[] = [
  'Monday',
  'Tuesday',
  'Wednesday',
  'Thursday',
  'Friday',
  'Saturday',
  'Sunday',
];

const BASE_NUMERIC_COUNT = 4; // Hour, Previous_Occupied, Total_Capacity, Occupancy_Percentage
const DAY_OFFSET = BASE_NUMERIC_COUNT; // indices 4..10 are Monday..Sunday

interface EncodedSample {
  features: number[];
  label: 0 | 1; // 0 = HIGH, 1 = LOW
}

interface TreeNode {
  isLeaf: boolean;
  probLow: number;
  predictedLabel: 0 | 1;
  featureIndex?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
}

export interface TrainedModelsBundle {
  forest: TreeNode[];
  singleTree: TreeNode;
  locations: string[];
  primaryMetrics: ModelMetrics;
  comparisonMetrics: ModelMetrics;
  featureImportances: FeatureImportanceItem[];
  trainRecords: number;
  testRecords: number;
  totalRecords: number;
  thresholdPct: number;
  trainedAt: string;
  cohortByLocDayHour: Record<string, number>;
  cohortByLocHour: Record<string, number>;
  cohortByHour: Record<number, number>;
  overallMeanRatio: number;
}

function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

export function parseCsvText(csvText: string): Record<string, string>[] {
  const lines = csvText
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .split('\n')
    .filter((line) => line.trim().length > 0);

  if (lines.length < 2) {
    return [];
  }

  const splitLine = (line: string): string[] => {
    const result: string[] = [];
    let current = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (inQuotes && line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (ch === ',' && !inQuotes) {
        result.push(current.trim());
        current = '';
      } else {
        current += ch;
      }
    }
    result.push(current.trim());
    return result;
  };

  const headers = splitLine(lines[0]).map((h) => h.trim());
  const rows: Record<string, string>[] = [];

  for (let i = 1; i < lines.length; i++) {
    const values = splitLine(lines[i]);
    if (values.every((v) => v === '')) continue;
    const row: Record<string, string> = {};
    headers.forEach((h, idx) => {
      row[h] = values[idx] ?? '';
    });
    rows.push(row);
  }
  return rows;
}

function parseHourFromTime(timeVal: string): number | null {
  const text = (timeVal || '').trim();
  if (!text) return null;
  const match24 = text.match(/^(\d{1,2}):(\d{2})(?::\d{2})?$/);
  if (match24) {
    const hr = Number(match24[1]);
    const min = Number(match24[2]);
    if (hr >= 0 && hr <= 23 && min >= 0 && min <= 59) return hr;
  }
  const match12 = text.match(/^(\d{1,2}):(\d{2})\s*(AM|PM)$/i);
  if (match12) {
    let hr = Number(match12[1]);
    const min = Number(match12[2]);
    const meridiem = match12[3].toUpperCase();
    if (hr >= 1 && hr <= 12 && min >= 0 && min <= 59) {
      if (meridiem === 'PM' && hr !== 12) hr += 12;
      if (meridiem === 'AM' && hr === 12) hr = 0;
      return hr;
    }
  }
  const num = Number(text);
  if (Number.isFinite(num) && num >= 0 && num <= 23) return Math.floor(num);
  return null;
}

function normalizeDay(raw: string): DayName | null {
  const clean = (raw || '').trim().toLowerCase();
  const found = VALID_DAYS.find((d) => d.toLowerCase() === clean);
  return found ?? null;
}

export function validateAndCleanDataset(
  rawRows: Record<string, string>[],
  availabilityThresholdPct = 30
): { records: ParkingRecord[]; summary: ValidationSummary } {
  if (!rawRows || rawRows.length === 0) {
    throw new Error('The dataset is empty. Please upload a CSV file with historical parking records.');
  }

  const keys = Object.keys(rawRows[0]).map((k) => k.trim());
  const required = ['Day', 'Total_Capacity', 'Previous_Occupied'];
  const missing = required.filter((col) => !keys.includes(col));
  if (missing.length > 0) {
    throw new Error(
      `Missing required column(s): ${missing.join(', ')}. Expected Day, Total_Capacity, Previous_Occupied, and Hour or Time.`
    );
  }
  if (!keys.includes('Hour') && !keys.includes('Time')) {
    throw new Error("Dataset must contain either an 'Hour' column (0-23) or a 'Time' column (HH:MM).");
  }

  const records: ParkingRecord[] = [];
  let droppedRows = 0;
  let invalidCapacityCount = 0;
  let invalidOccCount = 0;
  let invalidDayCount = 0;
  let imputedAvailCount = 0;
  const hasAvailCol = keys.includes('Available_Spaces');
  const hasLocationCol = keys.includes('Location');
  const todayStr = new Date().toISOString().slice(0, 10);

  for (const row of rawRows) {
    const day = normalizeDay(row.Day ?? '');
    if (!day) {
      droppedRows++;
      invalidDayCount++;
      continue;
    }

    let hour: number | null = null;
    if (row.Hour !== undefined && row.Hour !== '') {
      const parsedHr = Number(row.Hour);
      if (Number.isInteger(parsedHr) && parsedHr >= 0 && parsedHr <= 23) {
        hour = parsedHr;
      }
    }
    if (hour === null && row.Time) {
      hour = parseHourFromTime(row.Time);
    }

    const totalCapacity = Number(row.Total_Capacity);
    const prevOccupied = Number(row.Previous_Occupied);

    if (
      hour === null ||
      !Number.isFinite(totalCapacity) ||
      !Number.isFinite(prevOccupied)
    ) {
      droppedRows++;
      continue;
    }

    if (totalCapacity <= 0) {
      droppedRows++;
      invalidCapacityCount++;
      continue;
    }

    if (prevOccupied < 0 || prevOccupied > totalCapacity) {
      droppedRows++;
      invalidOccCount++;
      continue;
    }

    const capInt = Math.round(totalCapacity);
    const prevInt = Math.round(prevOccupied);
    const occPct = Math.round((prevInt / capInt) * 10000) / 100;

    let availInt: number;
    const rawAvail = Number(row.Available_Spaces);
    if (!hasAvailCol || row.Available_Spaces === '' || !Number.isFinite(rawAvail) || rawAvail < 0 || rawAvail > capInt) {
      availInt = capInt - prevInt;
      if (hasAvailCol) imputedAvailCount++;
    } else {
      availInt = Math.round(rawAvail);
    }

    const availPct = (availInt / capInt) * 100;
    const status: AvailabilityStatus = availPct > availabilityThresholdPct ? 'HIGH' : 'LOW';
    const timeStr = row.Time && row.Time.trim() ? row.Time.trim() : `${String(hour).padStart(2, '0')}:00`;
    const dateStr = row.Date && row.Date.trim() ? row.Date.trim() : todayStr;
    const locationStr =
      hasLocationCol && row.Location && row.Location.trim()
        ? row.Location.trim()
        : 'Main Gate Lot';

    records.push({
      Date: dateStr,
      Time: timeStr,
      Location: locationStr,
      Day: day,
      Hour: hour,
      Total_Capacity: capInt,
      Previous_Occupied: prevInt,
      Occupancy_Percentage: occPct,
      Available_Spaces: availInt,
      Availability_Status: status,
    });
  }

  if (records.length === 0) {
    throw new Error('No valid parking records remained after cleaning invalid or missing rows.');
  }

  const issues: string[] = [];
  if (!hasLocationCol) {
    issues.push("Column 'Location' was not provided; assigned default 'Main Gate Lot'.");
  }
  if (invalidCapacityCount > 0) {
    issues.push(`Removed ${invalidCapacityCount} row(s) with non-positive Total_Capacity.`);
  }
  if (invalidOccCount > 0) {
    issues.push(`Removed ${invalidOccCount} row(s) where Previous_Occupied was < 0 or exceeded Total_Capacity.`);
  }
  if (invalidDayCount > 0) {
    issues.push(`Removed ${invalidDayCount} row(s) with invalid Day values.`);
  }
  if (!hasAvailCol) {
    issues.push("Column 'Available_Spaces' not provided; derived from Total_Capacity - Previous_Occupied.");
  } else if (imputedAvailCount > 0) {
    issues.push(`Imputed ${imputedAvailCount} missing or out-of-range Available_Spaces values.`);
  }

  const highCount = records.filter((r) => r.Availability_Status === 'HIGH').length;
  const lowCount = records.length - highCount;
  const locations = Array.from(new Set(records.map((r) => r.Location))).sort();

  return {
    records,
    summary: {
      initialRows: rawRows.length,
      validRows: records.length,
      droppedRows,
      thresholdPct: availabilityThresholdPct,
      highCount,
      lowCount,
      locations,
      issues,
    },
  };
}

export function validatePredictionInput(
  day: string,
  entryTime: string,
  totalCapacity: number,
  previousOccupied: number,
  location = 'Main Gate Lot',
  validLocations?: string[]
): { valid: boolean; error: string; hour: number; occupancyPct: number } {
  const cleanLoc = (location || '').trim();
  if (!cleanLoc) {
    return { valid: false, error: 'Please select a parking location.', hour: 0, occupancyPct: 0 };
  }
  if (validLocations && validLocations.length > 0 && !validLocations.includes(cleanLoc)) {
    return { valid: false, error: `Unknown parking location '${cleanLoc}'.`, hour: 0, occupancyPct: 0 };
  }
  if (!normalizeDay(day)) {
    return { valid: false, error: 'Please select a valid day from Monday to Sunday.', hour: 0, occupancyPct: 0 };
  }
  const hour = parseHourFromTime(entryTime);
  if (hour === null) {
    return { valid: false, error: 'Entry time must be in valid HH:MM format.', hour: 0, occupancyPct: 0 };
  }
  if (!Number.isFinite(totalCapacity) || totalCapacity <= 0 || !Number.isInteger(totalCapacity)) {
    return { valid: false, error: 'Total parking capacity must be a positive whole number.', hour, occupancyPct: 0 };
  }
  if (!Number.isFinite(previousOccupied) || previousOccupied < 0 || !Number.isInteger(previousOccupied)) {
    return { valid: false, error: 'Previous occupied spaces must be a non-negative whole number.', hour, occupancyPct: 0 };
  }
  if (previousOccupied > totalCapacity) {
    return {
      valid: false,
      error: `Previous occupied spaces (${previousOccupied}) cannot exceed total parking capacity (${totalCapacity}).`,
      hour,
      occupancyPct: 0,
    };
  }
  return {
    valid: true,
    error: '',
    hour,
    occupancyPct: Math.round((previousOccupied / totalCapacity) * 10000) / 100,
  };
}

function encodeRecord(
  location: string,
  day: DayName,
  hour: number,
  previousOccupied: number,
  totalCapacity: number,
  occupancyPct: number,
  locations: string[]
): number[] {
  const vec: number[] = [
    hour,
    previousOccupied,
    totalCapacity,
    occupancyPct,
    day === 'Monday' ? 1 : 0,
    day === 'Tuesday' ? 1 : 0,
    day === 'Wednesday' ? 1 : 0,
    day === 'Thursday' ? 1 : 0,
    day === 'Friday' ? 1 : 0,
    day === 'Saturday' ? 1 : 0,
    day === 'Sunday' ? 1 : 0,
  ];
  for (const loc of locations) {
    vec.push(location === loc ? 1 : 0);
  }
  return vec;
}

function giniFromCounts(lowCount: number, total: number): number {
  if (total === 0) return 0;
  const pLow = lowCount / total;
  const pHigh = 1 - pLow;
  return 1 - (pLow * pLow + pHigh * pHigh);
}

function buildCartTree(
  samples: EncodedSample[],
  numFeatures: number,
  maxDepth: number,
  minSamplesSplit: number,
  maxFeatures: number,
  rng: () => number,
  importancesAcc: number[],
  depth = 0
): TreeNode {
  const total = samples.length;
  let lowCount = 0;
  for (let i = 0; i < total; i++) {
    lowCount += samples[i].label;
  }
  const probLow = total > 0 ? lowCount / total : 0.5;
  const predictedLabel: 0 | 1 = probLow >= 0.5 ? 1 : 0;

  if (depth >= maxDepth || total < minSamplesSplit || lowCount === 0 || lowCount === total) {
    return { isLeaf: true, probLow, predictedLabel };
  }

  const candidateIndices: number[] = Array.from({ length: numFeatures }, (_, i) => i);
  if (maxFeatures < numFeatures) {
    for (let i = candidateIndices.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = candidateIndices[i];
      candidateIndices[i] = candidateIndices[j];
      candidateIndices[j] = tmp;
    }
  }
  const selectedFeatures = candidateIndices.slice(0, maxFeatures);

  const parentGini = giniFromCounts(lowCount, total);
  let bestGain = 0;
  let bestFeature = -1;
  let bestThreshold = 0;

  for (const fIdx of selectedFeatures) {
    const pairs = samples.map((s) => ({ val: s.features[fIdx], label: s.label }));
    pairs.sort((a, b) => a.val - b.val);

    let leftLow = 0;
    let leftTotal = 0;

    for (let i = 0; i < pairs.length - 1; i++) {
      leftTotal++;
      leftLow += pairs[i].label;

      if (pairs[i].val === pairs[i + 1].val) continue;

      const rightTotal = total - leftTotal;
      if (leftTotal < 2 || rightTotal < 2) continue;

      const rightLow = lowCount - leftLow;
      const weightedGini =
        (leftTotal / total) * giniFromCounts(leftLow, leftTotal) +
        (rightTotal / total) * giniFromCounts(rightLow, rightTotal);
      const gain = (parentGini - weightedGini) * total;

      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = fIdx;
        bestThreshold = (pairs[i].val + pairs[i + 1].val) / 2;
      }
    }
  }

  if (bestFeature === -1 || bestGain <= 1e-9) {
    return { isLeaf: true, probLow, predictedLabel };
  }

  importancesAcc[bestFeature] += bestGain;

  const leftSamples: EncodedSample[] = [];
  const rightSamples: EncodedSample[] = [];
  for (const s of samples) {
    if (s.features[bestFeature] <= bestThreshold) {
      leftSamples.push(s);
    } else {
      rightSamples.push(s);
    }
  }

  return {
    isLeaf: false,
    probLow,
    predictedLabel,
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left: buildCartTree(
      leftSamples,
      numFeatures,
      maxDepth,
      minSamplesSplit,
      maxFeatures,
      rng,
      importancesAcc,
      depth + 1
    ),
    right: buildCartTree(
      rightSamples,
      numFeatures,
      maxDepth,
      minSamplesSplit,
      maxFeatures,
      rng,
      importancesAcc,
      depth + 1
    ),
  };
}

function predictTreeProbLow(node: TreeNode, features: number[]): number {
  let curr = node;
  while (!curr.isLeaf && curr.featureIndex !== undefined && curr.threshold !== undefined) {
    curr = features[curr.featureIndex] <= curr.threshold ? curr.left! : curr.right!;
  }
  return curr.probLow;
}

function evaluatePredictions(
  yTrue: (0 | 1)[],
  yPred: (0 | 1)[],
  algorithm: string
): ModelMetrics {
  let hh = 0;
  let hl = 0;
  let lh = 0;
  let ll = 0;

  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === 0 && yPred[i] === 0) hh++;
    else if (yTrue[i] === 0 && yPred[i] === 1) hl++;
    else if (yTrue[i] === 1 && yPred[i] === 0) lh++;
    else ll++;
  }

  const total = yTrue.length || 1;
  const accuracy = (hh + ll) / total;

  const highSupport = hh + hl;
  const lowSupport = lh + ll;

  const highPrec = hh + lh > 0 ? hh / (hh + lh) : 0;
  const highRec = highSupport > 0 ? hh / highSupport : 0;
  const highF1 = highPrec + highRec > 0 ? (2 * highPrec * highRec) / (highPrec + highRec) : 0;

  const lowPrec = hl + ll > 0 ? ll / (hl + ll) : 0;
  const lowRec = lowSupport > 0 ? ll / lowSupport : 0;
  const lowF1 = lowPrec + lowRec > 0 ? (2 * lowPrec * lowRec) / (lowPrec + lowRec) : 0;

  const weightedPrec = (highPrec * highSupport + lowPrec * lowSupport) / total;
  const weightedRec = (highRec * highSupport + lowRec * lowSupport) / total;
  const weightedF1 = (highF1 * highSupport + lowF1 * lowSupport) / total;

  return {
    algorithm,
    accuracy,
    precision: weightedPrec,
    recall: weightedRec,
    f1Score: weightedF1,
    confusionMatrix: [
      [hh, hl],
      [lh, ll],
    ],
    labels: ['HIGH', 'LOW'],
    perClass: {
      HIGH: { precision: highPrec, recall: highRec, f1Score: highF1, support: highSupport },
      LOW: { precision: lowPrec, recall: lowRec, f1Score: lowF1, support: lowSupport },
    },
  };
}

export function trainAndEvaluateEngine(
  records: ParkingRecord[],
  thresholdPct = 30
): TrainedModelsBundle {
  if (records.length < 20) {
    throw new Error('At least 20 valid records are required to train and evaluate the classification models.');
  }

  const locations = Array.from(new Set(records.map((r) => r.Location))).sort();
  const numFeatures = BASE_NUMERIC_COUNT + VALID_DAYS.length + locations.length;

  const highIndices: number[] = [];
  const lowIndices: number[] = [];
  const encoded: EncodedSample[] = records.map((r, idx) => {
    const label: 0 | 1 = r.Availability_Status === 'LOW' ? 1 : 0;
    if (label === 0) highIndices.push(idx);
    else lowIndices.push(idx);
    return {
      features: encodeRecord(
        r.Location,
        r.Day,
        r.Hour,
        r.Previous_Occupied,
        r.Total_Capacity,
        r.Occupancy_Percentage,
        locations
      ),
      label,
    };
  });

  if (highIndices.length === 0 || lowIndices.length === 0) {
    const only = highIndices.length > 0 ? 'HIGH' : 'LOW';
    throw new Error(
      `Dataset contains only '${only}' availability records at the ${thresholdPct}% threshold. Both HIGH and LOW samples are required.`
    );
  }

  const rng = createRng(42);
  const shuffle = (arr: number[]) => {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
  };
  shuffle(highIndices);
  shuffle(lowIndices);

  const highTestCount = Math.max(1, Math.round(highIndices.length * 0.2));
  const lowTestCount = Math.max(1, Math.round(lowIndices.length * 0.2));

  const testIndices = [
    ...highIndices.slice(0, highTestCount),
    ...lowIndices.slice(0, lowTestCount),
  ];
  const trainIndices = [
    ...highIndices.slice(highTestCount),
    ...lowIndices.slice(lowTestCount),
  ];

  const trainSamples = trainIndices.map((idx) => encoded[idx]);
  const testSamples = testIndices.map((idx) => encoded[idx]);

  const rfImportancesAcc = new Array<number>(numFeatures).fill(0);
  const forest: TreeNode[] = [];
  const nEstimators = 35;
  const maxFeatures = Math.max(3, Math.round(Math.sqrt(numFeatures)));

  for (let t = 0; t < nEstimators; t++) {
    const bootstrap: EncodedSample[] = [];
    for (let i = 0; i < trainSamples.length; i++) {
      bootstrap.push(trainSamples[Math.floor(rng() * trainSamples.length)]);
    }
    forest.push(buildCartTree(bootstrap, numFeatures, 10, 4, maxFeatures, rng, rfImportancesAcc));
  }

  const dtImportancesAcc = new Array<number>(numFeatures).fill(0);
  const singleTree = buildCartTree(
    trainSamples,
    numFeatures,
    8,
    6,
    numFeatures,
    rng,
    dtImportancesAcc
  );

  const yTrue = testSamples.map((s) => s.label);
  const rfPreds = testSamples.map((s) => {
    const avgProbLow =
      forest.reduce((sum, tree) => sum + predictTreeProbLow(tree, s.features), 0) / forest.length;
    return (avgProbLow >= 0.5 ? 1 : 0) as 0 | 1;
  });
  const dtPreds = testSamples.map((s) => {
    const probLow = predictTreeProbLow(singleTree, s.features);
    return (probLow >= 0.5 ? 1 : 0) as 0 | 1;
  });

  const primaryMetrics = evaluatePredictions(yTrue, rfPreds, 'Random Forest Classifier');
  const comparisonMetrics = evaluatePredictions(yTrue, dtPreds, 'Decision Tree Classifier');

  const dayImportanceSum = rfImportancesAcc.slice(DAY_OFFSET, DAY_OFFSET + 7).reduce((a, b) => a + b, 0);
  const locImportanceSum = rfImportancesAcc.slice(DAY_OFFSET + 7).reduce((a, b) => a + b, 0);

  const rawSix: { feature: string; label: string; raw: number }[] = [
    { feature: 'Hour', label: 'Hour of Day', raw: rfImportancesAcc[0] },
    { feature: 'Previous_Occupied', label: 'Previous Occupied', raw: rfImportancesAcc[1] },
    { feature: 'Total_Capacity', label: 'Total Capacity', raw: rfImportancesAcc[2] },
    { feature: 'Occupancy_Percentage', label: 'Occupancy %', raw: rfImportancesAcc[3] },
    { feature: 'Day', label: 'Day of Week', raw: dayImportanceSum },
    { feature: 'Location', label: 'Parking Location', raw: locImportanceSum },
  ];
  const totalImp = rawSix.reduce((s, item) => s + item.raw, 0) || 1;
  const featureImportances: FeatureImportanceItem[] = rawSix
    .map((item) => ({
      feature: item.feature,
      label: item.label,
      importancePct: Math.round((item.raw / totalImp) * 1000) / 10,
    }))
    .sort((a, b) => b.importancePct - a.importancePct);

  const ldhSum: Record<string, { sum: number; count: number }> = {};
  const lhSum: Record<string, { sum: number; count: number }> = {};
  const hSum: Record<number, { sum: number; count: number }> = {};
  let totalRatio = 0;

  for (const r of records) {
    const occRatio = (r.Total_Capacity - r.Available_Spaces) / r.Total_Capacity;
    totalRatio += occRatio;

    const ldhKey = `${r.Location}|${r.Day}|${r.Hour}`;
    if (!ldhSum[ldhKey]) ldhSum[ldhKey] = { sum: 0, count: 0 };
    ldhSum[ldhKey].sum += occRatio;
    ldhSum[ldhKey].count += 1;

    const lhKey = `${r.Location}|${r.Hour}`;
    if (!lhSum[lhKey]) lhSum[lhKey] = { sum: 0, count: 0 };
    lhSum[lhKey].sum += occRatio;
    lhSum[lhKey].count += 1;

    if (!hSum[r.Hour]) hSum[r.Hour] = { sum: 0, count: 0 };
    hSum[r.Hour].sum += occRatio;
    hSum[r.Hour].count += 1;
  }

  const cohortByLocDayHour: Record<string, number> = {};
  Object.entries(ldhSum).forEach(([k, v]) => {
    cohortByLocDayHour[k] = v.sum / v.count;
  });
  const cohortByLocHour: Record<string, number> = {};
  Object.entries(lhSum).forEach(([k, v]) => {
    cohortByLocHour[k] = v.sum / v.count;
  });
  const cohortByHour: Record<number, number> = {};
  Object.entries(hSum).forEach(([k, v]) => {
    cohortByHour[Number(k)] = v.sum / v.count;
  });

  return {
    forest,
    singleTree,
    locations,
    primaryMetrics,
    comparisonMetrics,
    featureImportances,
    trainRecords: trainSamples.length,
    testRecords: testSamples.length,
    totalRecords: records.length,
    thresholdPct,
    trainedAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    cohortByLocDayHour,
    cohortByLocHour,
    cohortByHour,
    overallMeanRatio: totalRatio / records.length,
  };
}

export function predictParkingAvailability(
  bundle: TrainedModelsBundle,
  day: DayName,
  hour: number,
  previousOccupied: number,
  totalCapacity: number,
  location = 'Main Gate Lot'
): PredictionResult {
  const prevOccPct = Math.round((previousOccupied / totalCapacity) * 10000) / 100;
  const features = encodeRecord(
    location,
    day,
    hour,
    previousOccupied,
    totalCapacity,
    prevOccPct,
    bundle.locations
  );

  const probLow =
    bundle.forest.reduce((sum, tree) => sum + predictTreeProbLow(tree, features), 0) /
    bundle.forest.length;
  const probHigh = 1 - probLow;

  const predictedStatus: AvailabilityStatus = probHigh > probLow ? 'HIGH' : 'LOW';
  const confidencePct = Math.round(Math.max(probHigh, probLow) * 1000) / 10;

  const cohortRatio =
    bundle.cohortByLocDayHour[`${location}|${day}|${hour}`] ??
    bundle.cohortByLocHour[`${location}|${hour}`] ??
    bundle.cohortByHour[hour] ??
    bundle.overallMeanRatio;

  const prevRatio = previousOccupied / totalCapacity;
  const blendedRatio = Math.max(0.02, Math.min(0.99, 0.55 * cohortRatio + 0.45 * prevRatio));
  let estimatedOccupied = Math.round(blendedRatio * totalCapacity);

  const maxOccForHigh = Math.max(0, Math.floor(totalCapacity * (1 - bundle.thresholdPct / 100)) - 1);
  const minOccForLow = Math.min(totalCapacity, Math.ceil(totalCapacity * (1 - bundle.thresholdPct / 100)));

  if (predictedStatus === 'HIGH' && estimatedOccupied > maxOccForHigh) {
    estimatedOccupied = maxOccForHigh;
  } else if (predictedStatus === 'LOW' && estimatedOccupied < minOccForLow) {
    estimatedOccupied = minOccForLow;
  }

  const estimatedAvailable = Math.max(0, totalCapacity - estimatedOccupied);
  const estimatedOccupancyPct = Math.round((estimatedOccupied / totalCapacity) * 1000) / 10;
  const estimatedAvailabilityPct = Math.round((estimatedAvailable / totalCapacity) * 1000) / 10;

  const recommendation =
    predictedStatus === 'HIGH'
      ? `Good news! Parking availability at ${location} is expected to be high. You can proceed towards the parking area.`
      : `${location} may be crowded. Consider arriving earlier, comparing another zone in Parking Finder, or checking again later.`;

  const topFeat = bundle.featureImportances[0];
  const explanations: string[] = [
    `Primary learned signal: ${topFeat.label} carries ${topFeat.importancePct.toFixed(1)}% importance in the trained Random Forest.`,
    `Prior utilization at ${location}: ${prevOccPct.toFixed(1)}% (${previousOccupied} of ${totalCapacity} occupied, leaving ${totalCapacity - previousOccupied} open).`,
    `Historical ${location} occupancy on ${day} around ${String(hour).padStart(2, '0')}:00 averages ${(cohortRatio * 100).toFixed(1)}%.`,
  ];

  return {
    location,
    predictedStatus,
    confidencePct,
    probHighPct: Math.round(probHigh * 1000) / 10,
    probLowPct: Math.round(probLow * 1000) / 10,
    estimatedOccupied,
    estimatedAvailable,
    estimatedOccupancyPct,
    estimatedAvailabilityPct,
    previousOccupancyPct: prevOccPct,
    recommendation,
    explanations,
    thresholdPct: bundle.thresholdPct,
  };
}

/**
 * Leak-free multi-hour future forecast:
 * Step 0 receives the user's known initialOccupied count.
 * Steps 1..N-1 chain ONLY the previous step's ML-estimated occupancy —
 * never reading future actual occupancy from the dataset.
 */
export function forecastParkingHours(
  bundle: TrainedModelsBundle,
  location: string,
  day: DayName,
  startHour: number,
  initialOccupied: number,
  totalCapacity: number,
  horizonHours = 4
): ForecastStep[] {
  const steps: ForecastStep[] = [];
  let runningOccupied = initialOccupied;
  const startDayIdx = Math.max(0, VALID_DAYS.indexOf(day));

  for (let offset = 0; offset < horizonHours; offset++) {
    const rawHour = startHour + offset;
    const targetHour = rawHour % 24;
    const targetDay = VALID_DAYS[(startDayIdx + Math.floor(rawHour / 24)) % 7];

    const pred = predictParkingAvailability(
      bundle,
      targetDay,
      targetHour,
      runningOccupied,
      totalCapacity,
      location
    );

    steps.push({
      ...pred,
      stepOffset: offset,
      day: targetDay,
      hour: targetHour,
      timeLabel: `${String(targetHour).padStart(2, '0')}:00`,
      inputPrevOccupied: runningOccupied,
    });

    runningOccupied = pred.estimatedOccupied;
  }

  return steps;
}

export function getLocationSnapshots(records: ParkingRecord[]): LocationSnapshot[] {
  const byLoc: Record<string, ParkingRecord[]> = {};
  for (const r of records) {
    if (!byLoc[r.Location]) byLoc[r.Location] = [];
    byLoc[r.Location].push(r);
  }

  return Object.keys(byLoc)
    .sort()
    .map((loc) => {
      const list = [...byLoc[loc]].sort((a, b) => {
        const cmpDate = a.Date.localeCompare(b.Date);
        if (cmpDate !== 0) return cmpDate;
        if (a.Hour !== b.Hour) return a.Hour - b.Hour;
        return a.Time.localeCompare(b.Time);
      });
      const latest = list[list.length - 1];
      const cap = latest.Total_Capacity;
      const avail = latest.Available_Spaces;
      const occ = Math.max(0, cap - avail);
      return {
        location: loc,
        totalCapacity: cap,
        latestOccupied: occ,
        latestAvailable: avail,
        latestAvailabilityPct: cap > 0 ? Math.round((avail / cap) * 1000) / 10 : 0,
        latestOccupancyPct: cap > 0 ? Math.round((occ / cap) * 1000) / 10 : 0,
        latestStatus: latest.Availability_Status,
        latestDate: latest.Date,
        latestTime: latest.Time,
        latestDay: latest.Day,
        latestHour: latest.Hour,
        recordCount: list.length,
      };
    });
}

export function filterRecordsByLocation(
  records: ParkingRecord[],
  location: string
): ParkingRecord[] {
  if (!location || location === 'All Campus Zones') return records;
  return records.filter((r) => r.Location === location);
}

export function computeAnalyticsSummary(records: ParkingRecord[]): AnalyticsSummary {
  const totalRecords = records.length;
  if (totalRecords === 0) {
    return {
      totalRecords: 0,
      totalCapacity: 0,
      latestOccupied: 0,
      latestAvailable: 0,
      latestAvailabilityPct: 0,
      avgCapacity: 0,
      avgOccupied: 0,
      avgAvailable: 0,
      avgOccupancyPct: 0,
      avgAvailabilityPct: 0,
      highCount: 0,
      lowCount: 0,
      highPct: 0,
      lowPct: 0,
      busiestHours: [],
      leastBusyHours: [],
      hourlySeries: [],
      daySeries: [],
      weeklySeries: [],
      occupancyHistogram: [],
      bestTimeRecommendation: 'No historical records available.',
    };
  }

  const snapshots = getLocationSnapshots(records);
  const totalCapacity = snapshots.reduce((s, x) => s + x.totalCapacity, 0);
  const latestOccupied = snapshots.reduce((s, x) => s + x.latestOccupied, 0);
  const latestAvailable = snapshots.reduce((s, x) => s + x.latestAvailable, 0);
  const latestAvailabilityPct =
    totalCapacity > 0 ? Math.round((latestAvailable / totalCapacity) * 1000) / 10 : 0;

  let capSum = 0;
  let availSum = 0;
  let occPctSum = 0;
  let highCount = 0;

  const hourBuckets: Record<number, { occSum: number; count: number }> = {};
  const dayBuckets: Record<DayName, { occSum: number; count: number; high: number; low: number }> = {
    Monday: { occSum: 0, count: 0, high: 0, low: 0 },
    Tuesday: { occSum: 0, count: 0, high: 0, low: 0 },
    Wednesday: { occSum: 0, count: 0, high: 0, low: 0 },
    Thursday: { occSum: 0, count: 0, high: 0, low: 0 },
    Friday: { occSum: 0, count: 0, high: 0, low: 0 },
    Saturday: { occSum: 0, count: 0, high: 0, low: 0 },
    Sunday: { occSum: 0, count: 0, high: 0, low: 0 },
  };
  const weekBuckets: Record<string, { occSum: number; count: number }> = {};

  const histBins = Array.from({ length: 10 }, (_, idx) => ({
    bucket: `${idx * 10}–${idx * 10 + 10}%`,
    min: idx * 10,
    max: idx * 10 + 10,
    high: 0,
    low: 0,
  }));

  for (const r of records) {
    capSum += r.Total_Capacity;
    availSum += r.Available_Spaces;
    const actualOccPct = ((r.Total_Capacity - r.Available_Spaces) / r.Total_Capacity) * 100;
    occPctSum += actualOccPct;

    if (r.Availability_Status === 'HIGH') highCount++;

    if (!hourBuckets[r.Hour]) hourBuckets[r.Hour] = { occSum: 0, count: 0 };
    hourBuckets[r.Hour].occSum += actualOccPct;
    hourBuckets[r.Hour].count += 1;

    dayBuckets[r.Day].occSum += actualOccPct;
    dayBuckets[r.Day].count += 1;
    if (r.Availability_Status === 'HIGH') dayBuckets[r.Day].high += 1;
    else dayBuckets[r.Day].low += 1;

    const parsed = new Date(`${r.Date}T00:00:00Z`);
    if (!Number.isNaN(parsed.getTime())) {
      const dayOfWeek = (parsed.getUTCDay() + 6) % 7;
      const monday = new Date(parsed.getTime() - dayOfWeek * 86400000);
      const weekKey = monday.toISOString().slice(0, 10);
      if (!weekBuckets[weekKey]) weekBuckets[weekKey] = { occSum: 0, count: 0 };
      weekBuckets[weekKey].occSum += actualOccPct;
      weekBuckets[weekKey].count += 1;
    }

    const binIdx = Math.min(9, Math.max(0, Math.floor(r.Occupancy_Percentage / 10)));
    if (r.Availability_Status === 'HIGH') histBins[binIdx].high++;
    else histBins[binIdx].low++;
  }

  const lowCount = totalRecords - highCount;
  const avgCapacity = Math.round(capSum / totalRecords);
  const avgAvailable = Math.round(availSum / totalRecords);
  const avgOccupied = Math.max(0, avgCapacity - avgAvailable);
  const avgOccupancyPct = Math.round((occPctSum / totalRecords) * 10) / 10;
  const avgAvailabilityPct = Math.round((100 - avgOccupancyPct) * 10) / 10;

  const hourlySeries: HourStat[] = Object.entries(hourBuckets)
    .map(([hr, val]) => ({
      hour: Number(hr),
      avgOccPct: Math.round((val.occSum / val.count) * 10) / 10,
      records: val.count,
    }))
    .sort((a, b) => a.hour - b.hour);

  const sortedByOcc = [...hourlySeries].sort((a, b) => b.avgOccPct - a.avgOccPct);
  const busiestHours = sortedByOcc.slice(0, 3);
  const leastBusyHours = [...hourlySeries].sort((a, b) => a.avgOccPct - b.avgOccPct).slice(0, 3);

  const daySeries: DayStat[] = VALID_DAYS.map((day) => {
    const b = dayBuckets[day];
    return {
      day,
      avgOccPct: b.count > 0 ? Math.round((b.occSum / b.count) * 10) / 10 : 0,
      highCount: b.high,
      lowCount: b.low,
    };
  });

  const weeklySeries = Object.entries(weekBuckets)
    .map(([weekStart, val]) => ({
      weekStart,
      avgOccPct: Math.round((val.occSum / val.count) * 10) / 10,
      records: val.count,
    }))
    .sort((a, b) => a.weekStart.localeCompare(b.weekStart));

  const bestTimeRecommendation =
    leastBusyHours.length > 0
      ? `Recommended arrival windows: ${leastBusyHours
          .map((h) => `${String(h.hour).padStart(2, '0')}:00`)
          .join(', ')} (lowest average occupancy at ${String(leastBusyHours[0].hour).padStart(
          2,
          '0'
        )}:00 with ${leastBusyHours[0].avgOccPct.toFixed(1)}% filled).`
      : 'Insufficient hourly diversity for arrival recommendations.';

  return {
    totalRecords,
    totalCapacity,
    latestOccupied,
    latestAvailable,
    latestAvailabilityPct,
    avgCapacity,
    avgOccupied,
    avgAvailable,
    avgOccupancyPct,
    avgAvailabilityPct,
    highCount,
    lowCount,
    highPct: Math.round((highCount / totalRecords) * 1000) / 10,
    lowPct: Math.round((lowCount / totalRecords) * 1000) / 10,
    busiestHours,
    leastBusyHours,
    hourlySeries,
    daySeries,
    weeklySeries,
    occupancyHistogram: histBins,
    bestTimeRecommendation,
  };
}

export const CSV_UPLOAD_TEMPLATE = [
  'Date,Time,Location,Day,Hour,Total_Capacity,Previous_Occupied,Occupancy_Percentage,Available_Spaces,Availability_Status',
  '2026-10-12,09:00,Main Gate Lot,Monday,9,300,240,80.0,45,LOW',
  '2026-10-12,17:30,Engineering Block Lot,Monday,17,250,95,38.0,158,HIGH',
  '2026-10-13,11:00,Central Library Lot,Tuesday,11,200,170,85.0,22,LOW',
  '2026-10-17,10:00,Hostel & Sports Lot,Saturday,10,150,60,40.0,88,HIGH',
].join('\n') + '\n';
