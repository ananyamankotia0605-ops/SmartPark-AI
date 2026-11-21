export type AvailabilityStatus = 'HIGH' | 'LOW';

export type DayName =
  | 'Monday'
  | 'Tuesday'
  | 'Wednesday'
  | 'Thursday'
  | 'Friday'
  | 'Saturday'
  | 'Sunday';

export interface ParkingRecord {
  Date: string;
  Time: string;
  Location: string;
  Day: DayName;
  Hour: number;
  Total_Capacity: number;
  Previous_Occupied: number;
  Occupancy_Percentage: number;
  Available_Spaces: number;
  Availability_Status: AvailabilityStatus;
}

export interface LocationSnapshot {
  location: string;
  totalCapacity: number;
  latestOccupied: number;
  latestAvailable: number;
  latestAvailabilityPct: number;
  latestOccupancyPct: number;
  latestStatus: AvailabilityStatus;
  latestDate: string;
  latestTime: string;
  latestDay: DayName;
  latestHour: number;
  recordCount: number;
}

export interface ValidationSummary {
  initialRows: number;
  validRows: number;
  droppedRows: number;
  thresholdPct: number;
  highCount: number;
  lowCount: number;
  locations: string[];
  issues: string[];
}

export interface ClassMetrics {
  precision: number;
  recall: number;
  f1Score: number;
  support: number;
}

export interface ModelMetrics {
  algorithm: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: [[number, number], [number, number]];
  labels: ['HIGH', 'LOW'];
  perClass: {
    HIGH: ClassMetrics;
    LOW: ClassMetrics;
  };
}

export interface FeatureImportanceItem {
  feature: string;
  label: string;
  importancePct: number;
}

export interface PredictionResult {
  location: string;
  predictedStatus: AvailabilityStatus;
  confidencePct: number;
  probHighPct: number;
  probLowPct: number;
  estimatedOccupied: number;
  estimatedAvailable: number;
  estimatedOccupancyPct: number;
  estimatedAvailabilityPct: number;
  previousOccupancyPct: number;
  recommendation: string;
  explanations: string[];
  thresholdPct: number;
}

export interface ForecastStep extends PredictionResult {
  stepOffset: number;
  day: DayName;
  hour: number;
  timeLabel: string;
  inputPrevOccupied: number;
}

export interface PredictionHistoryRow {
  Timestamp: string;
  Location: string;
  Day: string;
  Entry_Time: string;
  Hour: number;
  Total_Capacity: number;
  Previous_Occupied: number;
  Previous_Occupancy_Pct: number;
  Predicted_Status: AvailabilityStatus;
  Confidence_Pct: number;
  Estimated_Occupied: number;
  Estimated_Available: number;
  Threshold_Pct: number;
}

export interface HourStat {
  hour: number;
  avgOccPct: number;
  records: number;
}

export interface DayStat {
  day: DayName;
  avgOccPct: number;
  highCount: number;
  lowCount: number;
}

export interface AnalyticsSummary {
  totalRecords: number;
  totalCapacity: number;
  latestOccupied: number;
  latestAvailable: number;
  latestAvailabilityPct: number;
  avgCapacity: number;
  avgOccupied: number;
  avgAvailable: number;
  avgOccupancyPct: number;
  avgAvailabilityPct: number;
  highCount: number;
  lowCount: number;
  highPct: number;
  lowPct: number;
  busiestHours: HourStat[];
  leastBusyHours: HourStat[];
  hourlySeries: HourStat[];
  daySeries: DayStat[];
  weeklySeries: { weekStart: string; avgOccPct: number; records: number }[];
  occupancyHistogram: { bucket: string; min: number; max: number; high: number; low: number }[];
  bestTimeRecommendation: string;
}

export interface SourceFileItem {
  path: string;
  name: string;
  language: string;
  content: string;
}
