// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package stats provides support for recording telemetry statistics.
// It acts as a coordination point between things that want to record stats,
// and things that want to aggregate and report stats.
package stats

import (
	"context"
)

// Int64Measure is used to record integer values.
type Int64Measure struct {
	name        string
	description string
	unit        Unit
	subscribers []Int64Subscriber
}

// Int64Measure is used to record floating point values.
type Float64Measure struct {
	name        string
	description string
	unit        Unit
	subscribers []Float64Subscriber
}

// Int64Subscriber is the type for functions that want to listen to
// integer statistic events.
type Int64Subscriber func(context.Context, *Int64Measure, int64)

// Float64Subscriber is the type for functions that want to listen to
// floating point statistic events.
type Float64Subscriber func(context.Context, *Float64Measure, float64)

// Unit is used to specify the units for a given measure.
// This is can used for display purposes.
type Unit int

const (
	// UnitDimensionless indicates that a measure has no specified units.
	UnitDimensionless = Unit(iota)
	// UnitBytes indicates that that a measure is recording number of bytes.
	UnitBytes
	// UnitMilliseconds indicates that a measure is recording a duration in milliseconds.
	UnitMilliseconds
)

// Int64 creates a new Int64Measure and prepares it for use.
func Int64(name string, description string, unit Unit) *Int64Measure {
	return &Int64Measure{
		name:        name,
		description: description,
		unit:        unit,
	}
}

// Float64 creates a new Float64Measure and prepares it for use.
func Float64(name string, description string, unit Unit) *Float64Measure {
	return &Float64Measure{
		name:        name,
		description: description,
		unit:        unit,
	}
}

// Name returns the name this measure was given on construction.
func (m *Int64Measure) Name() string { return m.name }

// Description returns the description this measure was given on construction.
func (m *Int64Measure) Description() string { return m.description }

// Unit returns the units this measure was given on construction.
func (m *Int64Measure) Unit() Unit { return m.unit }

// Subscribe adds a new subscriber to this measure.
func (m *Int64Measure) Subscribe(s Int64Subscriber) { m.subscribers = append(m.subscribers, s) }

// Record delivers a new value to the subscribers of this measure.
func (m *Int64Measure) Record(ctx context.Context, value int64) {
	for _, s := range m.subscribers {
		s(ctx, m, value)
	}
}

// Name returns the name this measure was given on construction.
func (m *Float64Measure) Name() string { return m.name }

// Description returns the description this measure was given on construction.
func (m *Float64Measure) Description() string { return m.description }

// Unit returns the units this measure was given on construction.
func (m *Float64Measure) Unit() Unit { return m.unit }

// Subscribe adds a new subscriber to this measure.
func (m *Float64Measure) Subscribe(s Float64Subscriber) { m.subscribers = append(m.subscribers, s) }

// Record delivers a new value to the subscribers of this measure.
func (m *Float64Measure) Record(ctx context.Context, value float64) {
	for _, s := range m.subscribers {
		s(ctx, m, value)
	}
}
