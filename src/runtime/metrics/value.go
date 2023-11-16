// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package metrics

import (
	"math"
	"unsafe"
)

// ValueKind is a tag for a metric [Value] which indicates its type.
type ValueKind int

const (
	// KindBad indicates that the Value has no type and should not be used.
	KindBad ValueKind = iota

	// KindUint64 indicates that the type of the Value is a uint64.
	KindUint64

	// KindFloat64 indicates that the type of the Value is a float64.
	KindFloat64

	// KindFloat64Histogram indicates that the type of the Value is a *Float64Histogram.
	KindFloat64Histogram
)

// Value represents a metric value returned by the runtime.
type Value struct {
	kind    ValueKind
	scalar  uint64         // contains scalar values for scalar Kinds.
	pointer unsafe.Pointer // contains non-scalar values.
}

// Kind returns the tag representing the kind of value this is.
func (v Value) Kind() ValueKind {
	return v.kind
}

// Uint64 returns the internal uint64 value for the metric.
//
// If v.Kind() != KindUint64, this method panics.
func (v Value) Uint64() uint64 {
	if v.kind != KindUint64 {
		panic("called Uint64 on non-uint64 metric value")
	}
	return v.scalar
}

// Float64 returns the internal float64 value for the metric.
//
// If v.Kind() != KindFloat64, this method panics.
func (v Value) Float64() float64 {
	if v.kind != KindFloat64 {
		panic("called Float64 on non-float64 metric value")
	}
	return math.Float64frombits(v.scalar)
}

// Float64Histogram returns the internal *Float64Histogram value for the metric.
//
// If v.Kind() != KindFloat64Histogram, this method panics.
func (v Value) Float64Histogram() *Float64Histogram {
	if v.kind != KindFloat64Histogram {
		panic("called Float64Histogram on non-Float64Histogram metric value")
	}
	return (*Float64Histogram)(v.pointer)
}
