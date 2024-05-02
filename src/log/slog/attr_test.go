// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog

import (
	"internal/testenv"
	"testing"
	"time"
)

func TestAttrNoAlloc(t *testing.T) {
	testenv.SkipIfOptimizationOff(t)
	// Assign values just to make sure the compiler doesn't optimize away the statements.
	var (
		i int64
		u uint64
		f float64
		b bool
		s string
		x any
		p = &i
		d time.Duration
	)
	a := int(testing.AllocsPerRun(5, func() {
		i = Int64("key", 1).Value.Int64()
		u = Uint64("key", 1).Value.Uint64()
		f = Float64("key", 1).Value.Float64()
		b = Bool("key", true).Value.Bool()
		s = String("key", "foo").Value.String()
		d = Duration("key", d).Value.Duration()
		x = Any("key", p).Value.Any()
	}))
	if a != 0 {
		t.Errorf("got %d allocs, want zero", a)
	}
	_ = u
	_ = f
	_ = b
	_ = s
	_ = x
}

func BenchmarkAttrString(b *testing.B) {
	var (
		is string
		u  string
		f  string
		bn string
		s  string
		x  string
		ds string
		p  = &is
		d  time.Duration
	)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		is = Int64("key", 1).String()
		u = Uint64("key", 1).String()
		f = Float64("key", 1).String()
		bn = Bool("key", true).String()
		s = String("key", "foo").String()
		ds = Duration("key", d).String()
		x = Any("key", p).String()
	}
	_ = u
	_ = f
	_ = bn
	_ = s
	_ = x
	_ = ds
	_ = p
}
