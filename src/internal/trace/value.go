// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import "fmt"

// Value is a dynamically-typed value obtained from a trace.
type Value struct {
	kind   ValueKind
	scalar uint64
}

// ValueKind is the type of a dynamically-typed value from a trace.
type ValueKind uint8

const (
	ValueBad ValueKind = iota
	ValueUint64
)

// Kind returns the ValueKind of the value.
//
// It represents the underlying structure of the value.
//
// New ValueKinds may be added in the future. Users of this type must be robust
// to that possibility.
func (v Value) Kind() ValueKind {
	return v.kind
}

// Uint64 returns the uint64 value for a MetricSampleUint64.
//
// Panics if this metric sample's Kind is not MetricSampleUint64.
func (v Value) Uint64() uint64 {
	if v.kind != ValueUint64 {
		panic("Uint64 called on Value of a different Kind")
	}
	return v.scalar
}

// valueAsString produces a debug string value.
//
// This isn't just Value.String because we may want to use that to store
// string values in the future.
func valueAsString(v Value) string {
	switch v.Kind() {
	case ValueUint64:
		return fmt.Sprintf("Uint64(%d)", v.scalar)
	}
	return "Bad"
}
