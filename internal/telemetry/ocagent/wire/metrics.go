// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wire

type ExportMetricsServiceRequest struct {
	Node     *Node     `json:"node,omitempty"`
	Metrics  []*Metric `json:"metrics,omitempty"`
	Resource *Resource `json:"resource,omitempty"`
}

type Metric struct {
	MetricDescriptor *MetricDescriptor `json:"metric_descriptor,omitempty"`
	Timeseries       []*TimeSeries     `json:"timeseries,omitempty"`
	Resource         *Resource         `json:"resource,omitempty"`
}

type MetricDescriptor struct {
	Name        string                `json:"name,omitempty"`
	Description string                `json:"description,omitempty"`
	Unit        string                `json:"unit,omitempty"`
	Type        MetricDescriptor_Type `json:"type,omitempty"`
	LabelKeys   []*LabelKey           `json:"label_keys,omitempty"`
}

type MetricDescriptor_Type int32

const (
	MetricDescriptor_UNSPECIFIED             MetricDescriptor_Type = 0
	MetricDescriptor_GAUGE_INT64             MetricDescriptor_Type = 1
	MetricDescriptor_GAUGE_DOUBLE            MetricDescriptor_Type = 2
	MetricDescriptor_GAUGE_DISTRIBUTION      MetricDescriptor_Type = 3
	MetricDescriptor_CUMULATIVE_INT64        MetricDescriptor_Type = 4
	MetricDescriptor_CUMULATIVE_DOUBLE       MetricDescriptor_Type = 5
	MetricDescriptor_CUMULATIVE_DISTRIBUTION MetricDescriptor_Type = 6
	MetricDescriptor_SUMMARY                 MetricDescriptor_Type = 7
)

type LabelKey struct {
	Key         string `json:"key,omitempty"`
	Description string `json:"description,omitempty"`
}

type TimeSeries struct {
	StartTimestamp *Timestamp    `json:"start_timestamp,omitempty"`
	LabelValues    []*LabelValue `json:"label_values,omitempty"`
	Points         []*Point      `json:"points,omitempty"`
}

type LabelValue struct {
	Value    string `json:"value,omitempty"`
	HasValue bool   `json:"has_value,omitempty"`
}

type Point struct {
	Timestamp *Timestamp `json:"timestamp,omitempty"`
	Value     PointValue `json:"value,omitempty"`
}

type PointInt64Value struct {
	Int64Value int64 `json:"int64Value,omitempty"`
}

type PointDoubleValue struct {
	DoubleValue float64 `json:"doubleValue,omitempty"`
}

type PointDistributionValue struct {
	DistributionValue *DistributionValue `json:"distributionValue,omitempty"`
}

type PointSummaryValue struct {
	SummaryValue *SummaryValue `json:"summaryValue,omitempty"`
}

type PointValue interface {
	tagPointValue()
}

func (PointInt64Value) tagPointValue()        {}
func (PointDoubleValue) tagPointValue()       {}
func (PointDistributionValue) tagPointValue() {}
func (PointSummaryValue) tagPointValue()      {}

type DistributionValue struct {
	Count                 int64         `json:"count,omitempty"`
	Sum                   float64       `json:"sum,omitempty"`
	SumOfSquaredDeviation float64       `json:"sum_of_squared_deviation,omitempty"`
	BucketOptions         BucketOptions `json:"bucket_options,omitempty"`
	Buckets               []*Bucket     `json:"buckets,omitempty"`
}

type BucketOptionsExplicit struct {
	Bounds []float64 `json:"bounds,omitempty"`
}

type BucketOptions interface {
	tagBucketOptions()
}

func (BucketOptionsExplicit) tagBucketOptions() {}

type Bucket struct {
	Count    int64     `json:"count,omitempty"`
	Exemplar *Exemplar `json:"exemplar,omitempty"`
}

type Exemplar struct {
	Value       float64           `json:"value,omitempty"`
	Timestamp   *Timestamp        `json:"timestamp,omitempty"`
	Attachments map[string]string `json:"attachments,omitempty"`
}

type SummaryValue struct {
	Count    *Int64Value  `json:"count,omitempty"`
	Sum      *DoubleValue `json:"sum,omitempty"`
	Snapshot *Snapshot    `json:"snapshot,omitempty"`
}

type Snapshot struct {
	Count            *Int64Value                  `json:"count,omitempty"`
	Sum              *DoubleValue                 `json:"sum,omitempty"`
	PercentileValues []*SnapshotValueAtPercentile `json:"percentile_values,omitempty"`
}

type SnapshotValueAtPercentile struct {
	Percentile float64 `json:"percentile,omitempty"`
	Value      float64 `json:"value,omitempty"`
}
