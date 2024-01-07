// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Benchmark for accessing Value values.

package slog

import (
	"testing"
	"time"
)

// The "As" form is the slowest.
// The switch-panic and visitor times are almost the same.
// BenchmarkDispatch/switch-checked-8         	 8669427	       137.7 ns/op
// BenchmarkDispatch/As-8                     	 8212087	       145.3 ns/op
// BenchmarkDispatch/Visit-8                  	 8926146	       135.3 ns/op
func BenchmarkDispatch(b *testing.B) {
	vs := []Value{
		Int64Value(32768),
		Uint64Value(0xfacecafe),
		StringValue("anything"),
		BoolValue(true),
		Float64Value(1.2345),
		DurationValue(time.Second),
		AnyValue(b),
	}
	var (
		ii int64
		s  string
		bb bool
		u  uint64
		d  time.Duration
		f  float64
		a  any
	)
	b.Run("switch-checked", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, v := range vs {
				switch v.Kind() {
				case KindString:
					s = v.String()
				case KindInt64:
					ii = v.Int64()
				case KindUint64:
					u = v.Uint64()
				case KindFloat64:
					f = v.Float64()
				case KindBool:
					bb = v.Bool()
				case KindDuration:
					d = v.Duration()
				case KindAny:
					a = v.Any()
				default:
					panic("bad kind")
				}
			}
		}
		_ = ii
		_ = s
		_ = bb
		_ = u
		_ = d
		_ = f
		_ = a

	})
	b.Run("As", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, kv := range vs {
				if v, ok := kv.AsString(); ok {
					s = v
				} else if v, ok := kv.AsInt64(); ok {
					ii = v
				} else if v, ok := kv.AsUint64(); ok {
					u = v
				} else if v, ok := kv.AsFloat64(); ok {
					f = v
				} else if v, ok := kv.AsBool(); ok {
					bb = v
				} else if v, ok := kv.AsDuration(); ok {
					d = v
				} else if v, ok := kv.AsAny(); ok {
					a = v
				} else {
					panic("bad kind")
				}
			}
		}
		_ = ii
		_ = s
		_ = bb
		_ = u
		_ = d
		_ = f
		_ = a
	})

	b.Run("Visit", func(b *testing.B) {
		v := &setVisitor{}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for _, kv := range vs {
				kv.Visit(v)
			}
		}
	})
}

type setVisitor struct {
	i int64
	s string
	b bool
	u uint64
	d time.Duration
	f float64
	a any
}

func (v *setVisitor) String(s string)          { v.s = s }
func (v *setVisitor) Int64(i int64)            { v.i = i }
func (v *setVisitor) Uint64(x uint64)          { v.u = x }
func (v *setVisitor) Float64(x float64)        { v.f = x }
func (v *setVisitor) Bool(x bool)              { v.b = x }
func (v *setVisitor) Duration(x time.Duration) { v.d = x }
func (v *setVisitor) Any(x any)                { v.a = x }

// When dispatching on all types, the "As" functions are slightly slower
// than switching on the kind and then calling a function that checks
// the kind again. See BenchmarkDispatch above.

func (a Value) AsString() (string, bool) {
	if a.Kind() == KindString {
		return a.str(), true
	}
	return "", false
}

func (a Value) AsInt64() (int64, bool) {
	if a.Kind() == KindInt64 {
		return int64(a.num), true
	}
	return 0, false
}

func (a Value) AsUint64() (uint64, bool) {
	if a.Kind() == KindUint64 {
		return a.num, true
	}
	return 0, false
}

func (a Value) AsFloat64() (float64, bool) {
	if a.Kind() == KindFloat64 {
		return a.float(), true
	}
	return 0, false
}

func (a Value) AsBool() (bool, bool) {
	if a.Kind() == KindBool {
		return a.bool(), true
	}
	return false, false
}

func (a Value) AsDuration() (time.Duration, bool) {
	if a.Kind() == KindDuration {
		return a.duration(), true
	}
	return 0, false
}

func (a Value) AsAny() (any, bool) {
	if a.Kind() == KindAny {
		return a.any, true
	}
	return nil, false
}

// Problem: adding a type means adding a method, which is a breaking change.
// Using an unexported method to force embedding will make programs compile,
// But they will panic at runtime when we call the new method.
type Visitor interface {
	String(string)
	Int64(int64)
	Uint64(uint64)
	Float64(float64)
	Bool(bool)
	Duration(time.Duration)
	Any(any)
}

func (a Value) Visit(v Visitor) {
	switch a.Kind() {
	case KindString:
		v.String(a.str())
	case KindInt64:
		v.Int64(int64(a.num))
	case KindUint64:
		v.Uint64(a.num)
	case KindBool:
		v.Bool(a.bool())
	case KindFloat64:
		v.Float64(a.float())
	case KindDuration:
		v.Duration(a.duration())
	case KindAny:
		v.Any(a.any)
	default:
		panic("bad kind")
	}
}
