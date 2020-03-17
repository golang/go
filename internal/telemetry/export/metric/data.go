// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package metric

import (
	"context"
	"sort"
	"time"

	"golang.org/x/tools/internal/telemetry/event"
)

// Data represents a single point in the time series of a metric.
// This provides the common interface to all metrics no matter their data
// format.
// To get the actual values for the metric you must type assert to a concrete
// metric type.
type Data interface {
	// Handle returns the metric handle this data is for.
	//TODO: rethink the concept of metric handles
	Handle() string
	// Groups reports the rows that currently exist for this metric.
	Groups() []event.TagSet
}

// Int64Data is a concrete implementation of Data for int64 scalar metrics.
type Int64Data struct {
	// Info holds the original construction information.
	Info *Scalar
	// IsGauge is true for metrics that track values, rather than increasing over time.
	IsGauge bool
	// Rows holds the per group values for the metric.
	Rows []int64
	// End is the last time this metric was updated.
	EndTime time.Time

	groups []event.TagSet
}

// Float64Data is a concrete implementation of Data for float64 scalar metrics.
type Float64Data struct {
	// Info holds the original construction information.
	Info *Scalar
	// IsGauge is true for metrics that track values, rather than increasing over time.
	IsGauge bool
	// Rows holds the per group values for the metric.
	Rows []float64
	// End is the last time this metric was updated.
	EndTime time.Time

	groups []event.TagSet
}

// HistogramInt64Data is a concrete implementation of Data for int64 histogram metrics.
type HistogramInt64Data struct {
	// Info holds the original construction information.
	Info *HistogramInt64
	// Rows holds the per group values for the metric.
	Rows []*HistogramInt64Row
	// End is the last time this metric was updated.
	EndTime time.Time

	groups []event.TagSet
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
	// Info holds the original construction information.
	Info *HistogramFloat64
	// Rows holds the per group values for the metric.
	Rows []*HistogramFloat64Row
	// End is the last time this metric was updated.
	EndTime time.Time

	groups []event.TagSet
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

func getGroup(ctx context.Context, g *[]event.TagSet, keys []event.Key) (int, bool) {
	group := event.Query(ctx, keys...)
	old := *g
	index := sort.Search(len(old), func(i int) bool {
		return !old[i].Less(group)
	})
	if index < len(old) && group.Equal(old[index]) {
		// not a new group
		return index, false
	}
	*g = make([]event.TagSet, len(old)+1)
	copy(*g, old[:index])
	copy((*g)[index+1:], old[index:])
	(*g)[index] = group
	return index, true
}

func (data *Int64Data) Handle() string         { return data.Info.Name }
func (data *Int64Data) Groups() []event.TagSet { return data.groups }

func (data *Int64Data) modify(ctx context.Context, ev event.Event, tag event.Tag, f func(v int64) int64) Data {
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
	data.EndTime = ev.At
	frozen := *data
	return &frozen
}

func (data *Int64Data) countInt64(ctx context.Context, ev event.Event, tag event.Tag) Data {
	return data.modify(ctx, ev, tag, func(v int64) int64 {
		return v + 1
	})
}

func (data *Int64Data) countFloat64(ctx context.Context, ev event.Event, tag event.Tag) Data {
	return data.modify(ctx, ev, tag, func(v int64) int64 {
		return v + 1
	})
}

func (data *Int64Data) sum(ctx context.Context, ev event.Event, tag event.Tag) Data {
	return data.modify(ctx, ev, tag, func(v int64) int64 {
		return v + tag.Value().(int64)
	})
}

func (data *Int64Data) latest(ctx context.Context, ev event.Event, tag event.Tag) Data {
	return data.modify(ctx, ev, tag, func(v int64) int64 {
		return tag.Value().(int64)
	})
}

func (data *Float64Data) Handle() string         { return data.Info.Name }
func (data *Float64Data) Groups() []event.TagSet { return data.groups }

func (data *Float64Data) modify(ctx context.Context, ev event.Event, tag event.Tag, f func(v float64) float64) Data {
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
	data.EndTime = ev.At
	frozen := *data
	return &frozen
}

func (data *Float64Data) sum(ctx context.Context, ev event.Event, tag event.Tag) Data {
	return data.modify(ctx, ev, tag, func(v float64) float64 {
		return v + tag.Value().(float64)
	})
}

func (data *Float64Data) latest(ctx context.Context, ev event.Event, tag event.Tag) Data {
	return data.modify(ctx, ev, tag, func(v float64) float64 {
		return tag.Value().(float64)
	})
}

func (data *HistogramInt64Data) Handle() string         { return data.Info.Name }
func (data *HistogramInt64Data) Groups() []event.TagSet { return data.groups }

func (data *HistogramInt64Data) modify(ctx context.Context, ev event.Event, tag event.Tag, f func(v *HistogramInt64Row)) Data {
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
	data.EndTime = ev.At
	frozen := *data
	return &frozen
}

func (data *HistogramInt64Data) record(ctx context.Context, ev event.Event, tag event.Tag) Data {
	return data.modify(ctx, ev, tag, func(v *HistogramInt64Row) {
		value := tag.Value().(int64)
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

func (data *HistogramFloat64Data) Handle() string         { return data.Info.Name }
func (data *HistogramFloat64Data) Groups() []event.TagSet { return data.groups }

func (data *HistogramFloat64Data) modify(ctx context.Context, ev event.Event, tag event.Tag, f func(v *HistogramFloat64Row)) Data {
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
	data.EndTime = ev.At
	frozen := *data
	return &frozen
}

func (data *HistogramFloat64Data) record(ctx context.Context, ev event.Event, tag event.Tag) Data {
	return data.modify(ctx, ev, tag, func(v *HistogramFloat64Row) {
		value := tag.Value().(float64)
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
