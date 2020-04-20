// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ocagent

import (
	"time"

	"golang.org/x/tools/internal/event/export/metric"
	"golang.org/x/tools/internal/event/export/ocagent/wire"
	"golang.org/x/tools/internal/event/label"
)

// dataToMetricDescriptor return a *wire.MetricDescriptor based on data.
func dataToMetricDescriptor(data metric.Data) *wire.MetricDescriptor {
	if data == nil {
		return nil
	}
	descriptor := &wire.MetricDescriptor{
		Name:        data.Handle(),
		Description: getDescription(data),
		// TODO: Unit?
		Type:      dataToMetricDescriptorType(data),
		LabelKeys: getLabelKeys(data),
	}

	return descriptor
}

// getDescription returns the description of data.
func getDescription(data metric.Data) string {
	switch d := data.(type) {
	case *metric.Int64Data:
		return d.Info.Description

	case *metric.Float64Data:
		return d.Info.Description

	case *metric.HistogramInt64Data:
		return d.Info.Description

	case *metric.HistogramFloat64Data:
		return d.Info.Description
	}

	return ""
}

// getLabelKeys returns a slice of *wire.LabelKeys based on the keys
// in data.
func getLabelKeys(data metric.Data) []*wire.LabelKey {
	switch d := data.(type) {
	case *metric.Int64Data:
		return infoKeysToLabelKeys(d.Info.Keys)

	case *metric.Float64Data:
		return infoKeysToLabelKeys(d.Info.Keys)

	case *metric.HistogramInt64Data:
		return infoKeysToLabelKeys(d.Info.Keys)

	case *metric.HistogramFloat64Data:
		return infoKeysToLabelKeys(d.Info.Keys)
	}

	return nil
}

// dataToMetricDescriptorType returns a wire.MetricDescriptor_Type based on the
// underlying type of data.
func dataToMetricDescriptorType(data metric.Data) wire.MetricDescriptor_Type {
	switch d := data.(type) {
	case *metric.Int64Data:
		if d.IsGauge {
			return wire.MetricDescriptor_GAUGE_INT64
		}
		return wire.MetricDescriptor_CUMULATIVE_INT64

	case *metric.Float64Data:
		if d.IsGauge {
			return wire.MetricDescriptor_GAUGE_DOUBLE
		}
		return wire.MetricDescriptor_CUMULATIVE_DOUBLE

	case *metric.HistogramInt64Data:
		return wire.MetricDescriptor_CUMULATIVE_DISTRIBUTION

	case *metric.HistogramFloat64Data:
		return wire.MetricDescriptor_CUMULATIVE_DISTRIBUTION
	}

	return wire.MetricDescriptor_UNSPECIFIED
}

// dataToTimeseries returns a slice of *wire.TimeSeries based on the
// points in data.
func dataToTimeseries(data metric.Data, start time.Time) []*wire.TimeSeries {
	if data == nil {
		return nil
	}

	numRows := numRows(data)
	startTimestamp := convertTimestamp(start)
	timeseries := make([]*wire.TimeSeries, 0, numRows)

	for i := 0; i < numRows; i++ {
		timeseries = append(timeseries, &wire.TimeSeries{
			StartTimestamp: &startTimestamp,
			// TODO: labels?
			Points: dataToPoints(data, i),
		})
	}

	return timeseries
}

// numRows returns the number of rows in data.
func numRows(data metric.Data) int {
	switch d := data.(type) {
	case *metric.Int64Data:
		return len(d.Rows)
	case *metric.Float64Data:
		return len(d.Rows)
	case *metric.HistogramInt64Data:
		return len(d.Rows)
	case *metric.HistogramFloat64Data:
		return len(d.Rows)
	}

	return 0
}

// dataToPoints returns an array of *wire.Points based on the point(s)
// in data at index i.
func dataToPoints(data metric.Data, i int) []*wire.Point {
	switch d := data.(type) {
	case *metric.Int64Data:
		timestamp := convertTimestamp(d.EndTime)
		return []*wire.Point{
			{
				Value: wire.PointInt64Value{
					Int64Value: d.Rows[i],
				},
				Timestamp: &timestamp,
			},
		}
	case *metric.Float64Data:
		timestamp := convertTimestamp(d.EndTime)
		return []*wire.Point{
			{
				Value: wire.PointDoubleValue{
					DoubleValue: d.Rows[i],
				},
				Timestamp: &timestamp,
			},
		}
	case *metric.HistogramInt64Data:
		row := d.Rows[i]
		bucketBounds := make([]float64, len(d.Info.Buckets))
		for i, val := range d.Info.Buckets {
			bucketBounds[i] = float64(val)
		}
		return distributionToPoints(row.Values, row.Count, float64(row.Sum), bucketBounds, d.EndTime)
	case *metric.HistogramFloat64Data:
		row := d.Rows[i]
		return distributionToPoints(row.Values, row.Count, row.Sum, d.Info.Buckets, d.EndTime)
	}

	return nil
}

// distributionToPoints returns an array of *wire.Points containing a
// wire.PointDistributionValue representing a distribution with the
// supplied counts, count, and sum.
func distributionToPoints(counts []int64, count int64, sum float64, bucketBounds []float64, end time.Time) []*wire.Point {
	buckets := make([]*wire.Bucket, len(counts))
	for i := 0; i < len(counts); i++ {
		buckets[i] = &wire.Bucket{
			Count: counts[i],
		}
	}
	timestamp := convertTimestamp(end)
	return []*wire.Point{
		{
			Value: wire.PointDistributionValue{
				DistributionValue: &wire.DistributionValue{
					Count: count,
					Sum:   sum,
					// TODO: SumOfSquaredDeviation?
					Buckets: buckets,
					BucketOptions: &wire.BucketOptionsExplicit{
						Bounds: bucketBounds,
					},
				},
			},
			Timestamp: &timestamp,
		},
	}
}

// infoKeysToLabelKeys returns an array of *wire.LabelKeys containing the
// string values of the elements of labelKeys.
func infoKeysToLabelKeys(infoKeys []label.Key) []*wire.LabelKey {
	labelKeys := make([]*wire.LabelKey, 0, len(infoKeys))
	for _, key := range infoKeys {
		labelKeys = append(labelKeys, &wire.LabelKey{
			Key: key.Name(),
		})
	}

	return labelKeys
}
