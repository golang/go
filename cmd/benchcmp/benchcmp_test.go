// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"
	"testing"

	"golang.org/x/tools/benchmark/parse"
)

func TestSelectBest(t *testing.T) {
	have := parse.Set{
		"Benchmark1": []*parse.Benchmark{
			{
				Name: "Benchmark1",
				N:    10, NsPerOp: 100, Measured: parse.NsPerOp,
				Ord: 0,
			},
			{
				Name: "Benchmark1",
				N:    10, NsPerOp: 50, Measured: parse.NsPerOp,
				Ord: 3,
			},
		},
		"Benchmark2": []*parse.Benchmark{
			{
				Name: "Benchmark2",
				N:    10, NsPerOp: 60, Measured: parse.NsPerOp,
				Ord: 1,
			},
			{
				Name: "Benchmark2",
				N:    10, NsPerOp: 500, Measured: parse.NsPerOp,
				Ord: 2,
			},
		},
	}

	want := parse.Set{
		"Benchmark1": []*parse.Benchmark{
			{
				Name: "Benchmark1",
				N:    10, NsPerOp: 50, Measured: parse.NsPerOp,
				Ord: 0,
			},
		},
		"Benchmark2": []*parse.Benchmark{
			{
				Name: "Benchmark2",
				N:    10, NsPerOp: 60, Measured: parse.NsPerOp,
				Ord: 1,
			},
		},
	}

	selectBest(have)
	if !reflect.DeepEqual(want, have) {
		t.Errorf("filtered bench set incorrectly, want %v have %v", want, have)
	}
}

func TestFormatNs(t *testing.T) {
	tests := []struct {
		input    float64
		expected string
	}{
		{input: 0, expected: "0.00"},
		{input: 0.2, expected: "0.20"},
		{input: 2, expected: "2.00"},
		{input: 2.2, expected: "2.20"},
		{input: 4, expected: "4.00"},
		{input: 16, expected: "16.0"},
		{input: 16.08, expected: "16.1"},
		{input: 128, expected: "128"},
		{input: 256.2, expected: "256"},
	}

	for _, tt := range tests {
		actual := formatNs(tt.input)
		if actual != tt.expected {
			t.Fatalf("%f. got %q, want %q", tt.input, actual, tt.expected)
		}
	}
}
