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
