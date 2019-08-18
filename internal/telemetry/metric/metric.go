// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package metric aggregates stats into metrics that can be exported.
package metric

import (
	"context"
	"sort"

	"golang.org/x/tools/internal/telemetry"
	"golang.org/x/tools/internal/telemetry/export"
	"golang.org/x/tools/internal/telemetry/stats"
	"golang.org/x/tools/internal/telemetry/tag"
)

// Scalar represents the construction information for a scalar metric.
type Scalar struct {
	// Name is the unique name of this metric.
	Name string
	// Description can be used by observers to describe the metric to users.
	Description string
	// Keys is the set of tags that collectively describe rows of the metric.
	Keys []interface{}
}

// HistogramInt64 represents the construction information for an int64 histogram metric.
type HistogramInt64 struct {
	// Name is the unique name of this metric.
	Name string
	// Description can be used by observers to describe the metric to users.
	Description string
	// Keys is the set of tags that collectively describe rows of the metric.
	Keys []interface{}
	// Buckets holds the inclusive upper bound of each bucket in the histogram.
	Buckets []int64
}

// HistogramFloat64 represents the construction information for an float64 histogram metric.
type HistogramFloat64 struct {
	// Name is the unique name of this metric.
	Name string
	// Description can be used by observers to describe the metric to users.
	Description string
	// Keys is the set of tags that collectively describe rows of the metric.
	Keys []interface{}
	// Buckets holds the inclusive upper bound of each bucket in the histogram.
	Buckets []float64
}

// CountInt64 creates a new metric based on the Scalar information that counts
// the number of times the supplied int64 measure is set.
// Metrics of this type will use Int64Data.
func (info Scalar) CountInt64(measure *stats.Int64Measure) string {
	data := &Int64Data{Info: &info}
	measure.Subscribe(data.countInt64)
	return info.Name
}

// SumInt64 creates a new metric based on the Scalar information that sums all
// the values recorded on the int64 measure.
// Metrics of this type will use Int64Data.
func (info Scalar) SumInt64(measure *stats.Int64Measure) string {
	data := &Int64Data{Info: &info}
	measure.Subscribe(data.sum)
	_ = data
	return info.Name
}

// LatestInt64 creates a new metric based on the Scalar information that tracks
// the most recent value recorded on the int64 measure.
// Metrics of this type will use Int64Data.
func (info Scalar) LatestInt64(measure *stats.Int64Measure) string {
	data := &Int64Data{Info: &info, IsGauge: true}
	measure.Subscribe(data.latest)
	return info.Name
}

// CountFloat64 creates a new metric based on the Scalar information that counts
// the number of times the supplied float64 measure is set.
// Metrics of this type will use Int64Data.
func (info Scalar) CountFloat64(measure *stats.Float64Measure) string {
	data := &Int64Data{Info: &info}
	measure.Subscribe(data.countFloat64)
	return info.Name
}

// SumFloat64 creates a new metric based on the Scalar information that sums all
// the values recorded on the float64 measure.
// Metrics of this type will use Float64Data.
func (info Scalar) SumFloat64(measure *stats.Float64Measure) string {
	data := &Float64Data{Info: &info}
	measure.Subscribe(data.sum)
	return info.Name
}

// LatestFloat64 creates a new metric based on the Scalar information that tracks
// the most recent value recorded on the float64 measure.
// Metrics of this type will use Float64Data.
func (info Scalar) LatestFloat64(measure *stats.Float64Measure) string {
	data := &Float64Data{Info: &info, IsGauge: true}
	measure.Subscribe(data.latest)
	return info.Name
}

// Record creates a new metric based on the HistogramInt64 information that
// tracks the bucketized counts of values recorded on the int64 measure.
// Metrics of this type will use HistogramInt64Data.
func (info HistogramInt64) Record(measure *stats.Int64Measure) string {
	data := &HistogramInt64Data{Info: &info}
	measure.Subscribe(data.record)
	return info.Name
}

// Record creates a new metric based on the HistogramFloat64 information that
// tracks the bucketized counts of values recorded on the float64 measure.
// Metrics of this type will use HistogramFloat64Data.
func (info HistogramFloat64) Record(measure *stats.Float64Measure) string {
	data := &HistogramFloat64Data{Info: &info}
	measure.Subscribe(data.record)
	return info.Name
}

// Int64Data is a concrete implementation of Data for int64 scalar metrics.
type Int64Data struct {
	// Info holds the original consruction information.
	Info *Scalar
	// IsGauge is true for metrics that track values, rather than increasing over time.
	IsGauge bool
	// Rows holds the per group values for the metric.
	Rows []int64

	groups []telemetry.TagList
}

// Float64Data is a concrete implementation of Data for float64 scalar metrics.
type Float64Data struct {
	// Info holds the original consruction information.
	Info *Scalar
	// IsGauge is true for metrics that track values, rather than increasing over time.
	IsGauge bool
	// Rows holds the per group values for the metric.
	Rows []float64

	groups []telemetry.TagList
}

// HistogramInt64Data is a concrete implementation of Data for int64 histogram metrics.
type HistogramInt64Data struct {
	// Info holds the original consruction information.
	Info *HistogramInt64
	// Rows holds the per group values for the metric.
	Rows []*HistogramInt64Row

	groups []telemetry.TagList
}

// HistogramInt64Row holds the values for a single row of a HistogramInt64Data.
type HistogramInt64Row struct {
	// Values is the counts per bucket.
	Values []int64
	// Count is the total count.
	Count int64
	// Sum is the sum of all the values recorded.
	Sum int64
	// Min is the smallest recorded value.
	Min int64
	// Max is the largest recorded value.
	Max int64
}

// HistogramFloat64Data is a concrete implementation of Data for float64 histogram metrics.
type HistogramFloat64Data struct {
	// Info holds the original consruction information.
	Info *HistogramFloat64
	// Rows holds the per group values for the metric.
	Rows []*HistogramFloat64Row

	groups []telemetry.TagList
}

// HistogramFloat64Row holds the values for a single row of a HistogramFloat64Data.
type HistogramFloat64Row struct {
	// Values is the counts per bucket.
	Values []int64
	// Count is the total count.
	Count int64
	// Sum is the sum of all the values recorded.
	Sum float64
	// Min is the smallest recorded value.
	Min float64
	// Max is the largest recorded value.
	Max float64
}

func getGroup(ctx context.Context, g *[]telemetry.TagList, keys []interface{}) (int, bool) {
	group := tag.Get(ctx, keys...)
	old := *g
	index := sort.Search(len(old), func(i int) bool {
		return !old[i].Less(group)
	})
	if index < len(old) && group.Equal(old[index]) {
		// not a new group
		return index, false
	}
	*g = make([]telemetry.TagList, len(old)+1)
	copy(*g, old[:index])
	copy((*g)[index+1:], old[index:])
	(*g)[index] = group
	return index, true
}

func (data *Int64Data) Handle() string              { return data.Info.Name }
func (data *Int64Data) Groups() []telemetry.TagList { return data.groups }

func (data *Int64Data) modify(ctx context.Context, f func(v int64) int64) {
	index, insert := getGroup(ctx, &data.groups, data.Info.Keys)
	old := data.Rows
	if insert {
		data.Rows = make([]int64, len(old)+1)
		copy(data.Rows, old[:index])
		copy(data.Rows[index+1:], old[index:])
	} else {
		data.Rows = make([]int64, len(old))
		copy(data.Rows, old)
	}
	data.Rows[index] = f(data.Rows[index])
	frozen := *data
	export.Metric(ctx, &frozen)
}

func (data *Int64Data) countInt64(ctx context.Context, measure *stats.Int64Measure, value int64) {
	data.modify(ctx, func(v int64) int64 { return v + 1 })
}

func (data *Int64Data) countFloat64(ctx context.Context, measure *stats.Float64Measure, value float64) {
	data.modify(ctx, func(v int64) int64 { return v + 1 })
}

func (data *Int64Data) sum(ctx context.Context, measure *stats.Int64Measure, value int64) {
	data.modify(ctx, func(v int64) int64 { return v + value })
}

func (data *Int64Data) latest(ctx context.Context, measure *stats.Int64Measure, value int64) {
	data.modify(ctx, func(v int64) int64 { return value })
}

func (data *Float64Data) Handle() string              { return data.Info.Name }
func (data *Float64Data) Groups() []telemetry.TagList { return data.groups }

func (data *Float64Data) modify(ctx context.Context, f func(v float64) float64) {
	index, insert := getGroup(ctx, &data.groups, data.Info.Keys)
	old := data.Rows
	if insert {
		data.Rows = make([]float64, len(old)+1)
		copy(data.Rows, old[:index])
		copy(data.Rows[index+1:], old[index:])
	} else {
		data.Rows = make([]float64, len(old))
		copy(data.Rows, old)
	}
	data.Rows[index] = f(data.Rows[index])
	frozen := *data
	export.Metric(ctx, &frozen)
}

func (data *Float64Data) sum(ctx context.Context, measure *stats.Float64Measure, value float64) {
	data.modify(ctx, func(v float64) float64 { return v + value })
}

func (data *Float64Data) latest(ctx context.Context, measure *stats.Float64Measure, value float64) {
	data.modify(ctx, func(v float64) float64 { return value })
}

func (data *HistogramInt64Data) Handle() string              { return data.Info.Name }
func (data *HistogramInt64Data) Groups() []telemetry.TagList { return data.groups }

func (data *HistogramInt64Data) modify(ctx context.Context, f func(v *HistogramInt64Row)) {
	index, insert := getGroup(ctx, &data.groups, data.Info.Keys)
	old := data.Rows
	var v HistogramInt64Row
	if insert {
		data.Rows = make([]*HistogramInt64Row, len(old)+1)
		copy(data.Rows, old[:index])
		copy(data.Rows[index+1:], old[index:])
	} else {
		data.Rows = make([]*HistogramInt64Row, len(old))
		copy(data.Rows, old)
		v = *data.Rows[index]
	}
	oldValues := v.Values
	v.Values = make([]int64, len(data.Info.Buckets))
	copy(v.Values, oldValues)
	f(&v)
	data.Rows[index] = &v
	frozen := *data
	export.Metric(ctx, &frozen)
}

func (data *HistogramInt64Data) record(ctx context.Context, measure *stats.Int64Measure, value int64) {
	data.modify(ctx, func(v *HistogramInt64Row) {
		v.Sum += value
		if v.Min > value || v.Count == 0 {
			v.Min = value
		}
		if v.Max < value || v.Count == 0 {
			v.Max = value
		}
		v.Count++
		for i, b := range data.Info.Buckets {
			if value <= b {
				v.Values[i]++
			}
		}
	})
}

func (data *HistogramFloat64Data) Handle() string              { return data.Info.Name }
func (data *HistogramFloat64Data) Groups() []telemetry.TagList { return data.groups }

func (data *HistogramFloat64Data) modify(ctx context.Context, f func(v *HistogramFloat64Row)) {
	index, insert := getGroup(ctx, &data.groups, data.Info.Keys)
	old := data.Rows
	var v HistogramFloat64Row
	if insert {
		data.Rows = make([]*HistogramFloat64Row, len(old)+1)
		copy(data.Rows, old[:index])
		copy(data.Rows[index+1:], old[index:])
	} else {
		data.Rows = make([]*HistogramFloat64Row, len(old))
		copy(data.Rows, old)
		v = *data.Rows[index]
	}
	oldValues := v.Values
	v.Values = make([]int64, len(data.Info.Buckets))
	copy(v.Values, oldValues)
	f(&v)
	data.Rows[index] = &v
	frozen := *data
	export.Metric(ctx, &frozen)
}

func (data *HistogramFloat64Data) record(ctx context.Context, measure *stats.Float64Measure, value float64) {
	data.modify(ctx, func(v *HistogramFloat64Row) {
		v.Sum += value
		if v.Min > value || v.Count == 0 {
			v.Min = value
		}
		if v.Max < value || v.Count == 0 {
			v.Max = value
		}
		v.Count++
		for i, b := range data.Info.Buckets {
			if value <= b {
				v.Values[i]++
			}
		}
	})
}
