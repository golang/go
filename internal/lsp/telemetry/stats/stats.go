// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package stats provides support for recording telemetry statistics.
package stats

import "context"

type Measure interface {
	Name() string
	Description() string
	Unit() string
}

type Float64Measure interface {
	Measure
	M(v float64) Measurement
}

type Int64Measure interface {
	Measure
	M(v int64) Measurement
}

type Measurement interface {
	Measure() Measure
	Value() float64
}

type nullMeasure struct{}
type nullFloat64Measure struct{ nullMeasure }
type nullInt64Measure struct{ nullMeasure }

func (nullMeasure) Name() string        { return "" }
func (nullMeasure) Description() string { return "" }
func (nullMeasure) Unit() string        { return "" }

func (nullFloat64Measure) M(v float64) Measurement { return nil }
func (nullInt64Measure) M(v int64) Measurement     { return nil }

func NullFloat64Measure() Float64Measure { return nullFloat64Measure{} }
func NullInt64Measure() Int64Measure     { return nullInt64Measure{} }

var (
	Record = func(ctx context.Context, ms ...Measurement) {}
)
