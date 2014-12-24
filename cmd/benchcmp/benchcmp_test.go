package main

import (
	"reflect"
	"testing"

	"golang.org/x/tools/benchmark/parse"
)

func TestSelectBest(t *testing.T) {
	have := parse.BenchSet{
		"Benchmark1": []*parse.Bench{
			{
				Name: "Benchmark1",
				N:    10, NsOp: 100, Measured: parse.NsOp,
				Ord: 0,
			},
			{
				Name: "Benchmark1",
				N:    10, NsOp: 50, Measured: parse.NsOp,
				Ord: 3,
			},
		},
		"Benchmark2": []*parse.Bench{
			{
				Name: "Benchmark2",
				N:    10, NsOp: 60, Measured: parse.NsOp,
				Ord: 1,
			},
			{
				Name: "Benchmark2",
				N:    10, NsOp: 500, Measured: parse.NsOp,
				Ord: 2,
			},
		},
	}

	want := parse.BenchSet{
		"Benchmark1": []*parse.Bench{
			{
				Name: "Benchmark1",
				N:    10, NsOp: 50, Measured: parse.NsOp,
				Ord: 0,
			},
		},
		"Benchmark2": []*parse.Bench{
			{
				Name: "Benchmark2",
				N:    10, NsOp: 60, Measured: parse.NsOp,
				Ord: 1,
			},
		},
	}

	selectBest(have)
	if !reflect.DeepEqual(want, have) {
		t.Errorf("filtered bench set incorrectly, want %v have %v", want, have)
	}
}
