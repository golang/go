// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"reflect"
	"sort"
	"testing"

	"golang.org/x/tools/benchmark/parse"
)

func TestDelta(t *testing.T) {
	cases := []struct {
		before  float64
		after   float64
		mag     float64
		f       float64
		changed bool
		pct     string
		mult    string
	}{
		{before: 1, after: 1, mag: 1, f: 1, changed: false, pct: "+0.00%", mult: "1.00x"},
		{before: 1, after: 2, mag: 0.5, f: 2, changed: true, pct: "+100.00%", mult: "2.00x"},
		{before: 2, after: 1, mag: 0.5, f: 0.5, changed: true, pct: "-50.00%", mult: "0.50x"},
		{before: 0, after: 0, mag: 1, f: 1, changed: false, pct: "+0.00%", mult: "1.00x"},
		{before: 1, after: 0, mag: math.Inf(1), f: 0, changed: true, pct: "-100.00%", mult: "0.00x"},
		{before: 0, after: 1, mag: math.Inf(1), f: math.Inf(1), changed: true, pct: "+Inf%", mult: "+Infx"},
	}
	for _, tt := range cases {
		d := Delta{tt.before, tt.after}
		if want, have := tt.mag, d.mag(); want != have {
			t.Errorf("%s.mag(): want %f have %f", d, want, have)
		}
		if want, have := tt.f, d.Float64(); want != have {
			t.Errorf("%s.Float64(): want %f have %f", d, want, have)
		}
		if want, have := tt.changed, d.Changed(); want != have {
			t.Errorf("%s.Changed(): want %t have %t", d, want, have)
		}
		if want, have := tt.pct, d.Percent(); want != have {
			t.Errorf("%s.Percent(): want %q have %q", d, want, have)
		}
		if want, have := tt.mult, d.Multiple(); want != have {
			t.Errorf("%s.Multiple(): want %q have %q", d, want, have)
		}
	}
}

func TestCorrelate(t *testing.T) {
	// Benches that are going to be successfully correlated get N thus:
	//   0x<counter><num benches><b = before | a = after>
	// Read this: "<counter> of <num benches>, from <before|after>".
	before := parse.Set{
		"BenchmarkOneEach":   []*parse.Benchmark{{Name: "BenchmarkOneEach", N: 0x11b}},
		"BenchmarkOneToNone": []*parse.Benchmark{{Name: "BenchmarkOneToNone"}},
		"BenchmarkOneToTwo":  []*parse.Benchmark{{Name: "BenchmarkOneToTwo"}},
		"BenchmarkTwoToOne": []*parse.Benchmark{
			{Name: "BenchmarkTwoToOne"},
			{Name: "BenchmarkTwoToOne"},
		},
		"BenchmarkTwoEach": []*parse.Benchmark{
			{Name: "BenchmarkTwoEach", N: 0x12b},
			{Name: "BenchmarkTwoEach", N: 0x22b},
		},
	}

	after := parse.Set{
		"BenchmarkOneEach":   []*parse.Benchmark{{Name: "BenchmarkOneEach", N: 0x11a}},
		"BenchmarkNoneToOne": []*parse.Benchmark{{Name: "BenchmarkNoneToOne"}},
		"BenchmarkTwoToOne":  []*parse.Benchmark{{Name: "BenchmarkTwoToOne"}},
		"BenchmarkOneToTwo": []*parse.Benchmark{
			{Name: "BenchmarkOneToTwo"},
			{Name: "BenchmarkOneToTwo"},
		},
		"BenchmarkTwoEach": []*parse.Benchmark{
			{Name: "BenchmarkTwoEach", N: 0x12a},
			{Name: "BenchmarkTwoEach", N: 0x22a},
		},
	}

	pairs, errs := Correlate(before, after)

	// Fail to match: BenchmarkOneToNone, BenchmarkOneToTwo, BenchmarkTwoToOne.
	// Correlate does not notice BenchmarkNoneToOne.
	if len(errs) != 3 {
		t.Errorf("Correlated expected 4 errors, got %d: %v", len(errs), errs)
	}

	// Want three correlated pairs: one BenchmarkOneEach, two BenchmarkTwoEach.
	if len(pairs) != 3 {
		t.Fatalf("Correlated expected 3 pairs, got %v", pairs)
	}

	for _, pair := range pairs {
		if pair.Before.N&0xF != 0xb {
			t.Errorf("unexpected Before in pair %s", pair)
		}
		if pair.After.N&0xF != 0xa {
			t.Errorf("unexpected After in pair %s", pair)
		}
		if pair.Before.N>>4 != pair.After.N>>4 {
			t.Errorf("mismatched pair %s", pair)
		}
	}
}

func TestBenchCmpSorting(t *testing.T) {
	c := []BenchCmp{
		{&parse.Benchmark{Name: "BenchmarkMuchFaster", NsPerOp: 10, Ord: 3}, &parse.Benchmark{Name: "BenchmarkMuchFaster", NsPerOp: 1}},
		{&parse.Benchmark{Name: "BenchmarkSameB", NsPerOp: 5, Ord: 1}, &parse.Benchmark{Name: "BenchmarkSameB", NsPerOp: 5}},
		{&parse.Benchmark{Name: "BenchmarkSameA", NsPerOp: 5, Ord: 2}, &parse.Benchmark{Name: "BenchmarkSameA", NsPerOp: 5}},
		{&parse.Benchmark{Name: "BenchmarkSlower", NsPerOp: 10, Ord: 0}, &parse.Benchmark{Name: "BenchmarkSlower", NsPerOp: 11}},
	}

	// Test just one magnitude-based sort order; they are symmetric.
	sort.Sort(ByDeltaNsPerOp(c))
	want := []string{"BenchmarkMuchFaster", "BenchmarkSlower", "BenchmarkSameA", "BenchmarkSameB"}
	have := []string{c[0].Name(), c[1].Name(), c[2].Name(), c[3].Name()}
	if !reflect.DeepEqual(want, have) {
		t.Errorf("ByDeltaNsOp incorrect sorting: want %v have %v", want, have)
	}

	sort.Sort(ByParseOrder(c))
	want = []string{"BenchmarkSlower", "BenchmarkSameB", "BenchmarkSameA", "BenchmarkMuchFaster"}
	have = []string{c[0].Name(), c[1].Name(), c[2].Name(), c[3].Name()}
	if !reflect.DeepEqual(want, have) {
		t.Errorf("ByParseOrder incorrect sorting: want %v have %v", want, have)
	}
}
