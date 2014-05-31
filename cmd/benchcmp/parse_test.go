// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"
	"strings"
	"testing"
)

func TestParseLine(t *testing.T) {
	cases := []struct {
		line string
		want *Bench
		err  bool // expect an error
	}{
		{
			line: "BenchmarkEncrypt	100000000	        19.6 ns/op",
			want: &Bench{
				Name: "BenchmarkEncrypt",
				N:    100000000, NsOp: 19.6,
				Measured: NsOp,
			},
		},
		{
			line: "BenchmarkEncrypt	100000000	        19.6 ns/op	 817.77 MB/s",
			want: &Bench{
				Name: "BenchmarkEncrypt",
				N:    100000000, NsOp: 19.6, MbS: 817.77,
				Measured: NsOp | MbS,
			},
		},
		{
			line: "BenchmarkEncrypt	100000000	        19.6 ns/op	 817.77",
			want: &Bench{
				Name: "BenchmarkEncrypt",
				N:    100000000, NsOp: 19.6,
				Measured: NsOp,
			},
		},
		{
			line: "BenchmarkEncrypt	100000000	        19.6 ns/op	 817.77 MB/s	       5 allocs/op",
			want: &Bench{
				Name: "BenchmarkEncrypt",
				N:    100000000, NsOp: 19.6, MbS: 817.77, AllocsOp: 5,
				Measured: NsOp | MbS | AllocsOp,
			},
		},
		{
			line: "BenchmarkEncrypt	100000000	        19.6 ns/op	 817.77 MB/s	       3 B/op	       5 allocs/op",
			want: &Bench{
				Name: "BenchmarkEncrypt",
				N:    100000000, NsOp: 19.6, MbS: 817.77, BOp: 3, AllocsOp: 5,
				Measured: NsOp | MbS | BOp | AllocsOp,
			},
		},
		// error handling cases
		{
			line: "BenchPress	100	        19.6 ns/op", // non-benchmark
			err: true,
		},
		{
			line: "BenchmarkEncrypt	lots	        19.6 ns/op", // non-int iterations
			err: true,
		},
		{
			line: "BenchmarkBridge	100000000	        19.6 smoots", // unknown unit
			want: &Bench{
				Name: "BenchmarkBridge",
				N:    100000000,
			},
		},
		{
			line: "PASS",
			err:  true,
		},
	}

	for _, tt := range cases {
		have, err := ParseLine(tt.line)
		if tt.err && err == nil {
			t.Errorf("parsing line %q should have failed", tt.line)
			continue
		}
		if !reflect.DeepEqual(have, tt.want) {
			t.Errorf("parsed line %q incorrectly, want %v have %v", tt.line, tt.want, have)
		}
	}
}

func TestParseBenchSet(t *testing.T) {
	// Test two things:
	// 1. The noise that can accompany testing.B output gets ignored.
	// 2. Benchmarks with the same name have their order preserved.
	in := `
		?   	crypto	[no test files]
		PASS
				pem_decrypt_test.go:17: test 4. %!s(x509.PEMCipher=5)
			... [output truncated]

		BenchmarkEncrypt	100000000	        19.6 ns/op
		BenchmarkEncrypt	 5000000	       517 ns/op
		=== RUN TestChunk
		--- PASS: TestChunk (0.00 seconds)
		--- SKIP: TestLinuxSendfile (0.00 seconds)
			fs_test.go:716: skipping; linux-only test
		BenchmarkReadRequestApachebench	 1000000	      2960 ns/op	  27.70 MB/s	     839 B/op	       9 allocs/op
		BenchmarkClientServerParallel64	   50000	     59192 ns/op	    7028 B/op	      60 allocs/op
		ok  	net/http	95.783s
	`

	want := BenchSet{
		"BenchmarkReadRequestApachebench": []*Bench{
			{
				Name: "BenchmarkReadRequestApachebench",
				N:    1000000, NsOp: 2960, MbS: 27.70, BOp: 839, AllocsOp: 9,
				Measured: NsOp | MbS | BOp | AllocsOp,
				ord:      2,
			},
		},
		"BenchmarkClientServerParallel64": []*Bench{
			{
				Name: "BenchmarkClientServerParallel64",
				N:    50000, NsOp: 59192, BOp: 7028, AllocsOp: 60,
				Measured: NsOp | BOp | AllocsOp,
				ord:      3,
			},
		},
		"BenchmarkEncrypt": []*Bench{
			{
				Name: "BenchmarkEncrypt",
				N:    100000000, NsOp: 19.6,
				Measured: NsOp,
				ord:      0,
			},
			{
				Name: "BenchmarkEncrypt",
				N:    5000000, NsOp: 517,
				Measured: NsOp,
				ord:      1,
			},
		},
	}

	have, err := ParseBenchSet(strings.NewReader(in))
	if err != nil {
		t.Fatalf("unexpected err during ParseBenchSet: %v", err)
	}
	if !reflect.DeepEqual(want, have) {
		t.Errorf("parsed bench set incorrectly, want %v have %v", want, have)
	}
}

func TestParseBenchSetBest(t *testing.T) {
	// Test that -best mode takes best ns/op.
	*best = true
	defer func() {
		*best = false
	}()

	in := `
		Benchmark1 10 100 ns/op
		Benchmark2 10 60 ns/op
		Benchmark2 10 500 ns/op
		Benchmark1 10 50 ns/op
	`

	want := BenchSet{
		"Benchmark1": []*Bench{
			{
				Name: "Benchmark1",
				N:    10, NsOp: 50, Measured: NsOp,
				ord: 0,
			},
		},
		"Benchmark2": []*Bench{
			{
				Name: "Benchmark2",
				N:    10, NsOp: 60, Measured: NsOp,
				ord: 1,
			},
		},
	}

	have, err := ParseBenchSet(strings.NewReader(in))
	if err != nil {
		t.Fatalf("unexpected err during ParseBenchSet: %v", err)
	}
	if !reflect.DeepEqual(want, have) {
		t.Errorf("parsed bench set incorrectly, want %v have %v", want, have)
	}
}
