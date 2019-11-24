// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parse

import (
	"reflect"
	"strings"
	"testing"
)

func TestParseLine(t *testing.T) {
	cases := []struct {
		line string
		want *Benchmark
		err  bool // expect an error
	}{
		{
			line: "BenchmarkEncrypt	100000000	        19.6 ns/op",
			want: &Benchmark{
				Name: "BenchmarkEncrypt",
				N:    100000000, NsPerOp: 19.6,
				Measured: NsPerOp,
			},
		},
		{
			line: "BenchmarkEncrypt	100000000	        19.6 ns/op	 817.77 MB/s",
			want: &Benchmark{
				Name: "BenchmarkEncrypt",
				N:    100000000, NsPerOp: 19.6, MBPerS: 817.77,
				Measured: NsPerOp | MBPerS,
			},
		},
		{
			line: "BenchmarkEncrypt	100000000	        19.6 ns/op	 817.77",
			want: &Benchmark{
				Name: "BenchmarkEncrypt",
				N:    100000000, NsPerOp: 19.6,
				Measured: NsPerOp,
			},
		},
		{
			line: "BenchmarkEncrypt	100000000	        19.6 ns/op	 817.77 MB/s	       5 allocs/op",
			want: &Benchmark{
				Name: "BenchmarkEncrypt",
				N:    100000000, NsPerOp: 19.6, MBPerS: 817.77, AllocsPerOp: 5,
				Measured: NsPerOp | MBPerS | AllocsPerOp,
			},
		},
		{
			line: "BenchmarkEncrypt	100000000	        19.6 ns/op	 817.77 MB/s	       3 B/op	       5 allocs/op",
			want: &Benchmark{
				Name: "BenchmarkEncrypt",
				N:    100000000, NsPerOp: 19.6, MBPerS: 817.77, AllocedBytesPerOp: 3, AllocsPerOp: 5,
				Measured: NsPerOp | MBPerS | AllocedBytesPerOp | AllocsPerOp,
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
			want: &Benchmark{
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

func TestParseSet(t *testing.T) {
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

	want := Set{
		"BenchmarkReadRequestApachebench": []*Benchmark{
			{
				Name: "BenchmarkReadRequestApachebench",
				N:    1000000, NsPerOp: 2960, MBPerS: 27.70, AllocedBytesPerOp: 839, AllocsPerOp: 9,
				Measured: NsPerOp | MBPerS | AllocedBytesPerOp | AllocsPerOp,
				Ord:      2,
			},
		},
		"BenchmarkClientServerParallel64": []*Benchmark{
			{
				Name: "BenchmarkClientServerParallel64",
				N:    50000, NsPerOp: 59192, AllocedBytesPerOp: 7028, AllocsPerOp: 60,
				Measured: NsPerOp | AllocedBytesPerOp | AllocsPerOp,
				Ord:      3,
			},
		},
		"BenchmarkEncrypt": []*Benchmark{
			{
				Name: "BenchmarkEncrypt",
				N:    100000000, NsPerOp: 19.6,
				Measured: NsPerOp,
				Ord:      0,
			},
			{
				Name: "BenchmarkEncrypt",
				N:    5000000, NsPerOp: 517,
				Measured: NsPerOp,
				Ord:      1,
			},
		},
	}

	have, err := ParseSet(strings.NewReader(in))
	if err != nil {
		t.Fatalf("unexpected err during ParseSet: %v", err)
	}
	if !reflect.DeepEqual(want, have) {
		t.Errorf("parsed bench set incorrectly, want %v have %v", want, have)
	}
}

func TestString(t *testing.T) {
	tests := []struct {
		name   string
		input  *Benchmark
		wanted string
	}{
		{
			name: "nsTest",
			input: &Benchmark{
				Name: "BenchmarkTest",
				N:    100000000, NsPerOp: 19.6,
				Measured: NsPerOp,
			},
			wanted: "BenchmarkTest 100000000 19.60 ns/op",
		},
		{
			name: "mbTest",
			input: &Benchmark{
				Name: "BenchmarkTest",
				N:    100000000, MBPerS: 19.6,
				Measured: MBPerS,
			},
			wanted: "BenchmarkTest 100000000 19.60 MB/s",
		},
		{
			name: "allocatedBytesTest",
			input: &Benchmark{
				Name: "BenchmarkTest",
				N:    100000000, AllocedBytesPerOp: 5,
				Measured: AllocedBytesPerOp,
			},
			wanted: "BenchmarkTest 100000000 5 B/op",
		},
		{
			name: "allocsTest",
			input: &Benchmark{
				Name: "BenchmarkTest",
				N:    100000000, AllocsPerOp: 5,
				Measured: AllocsPerOp,
			},
			wanted: "BenchmarkTest 100000000 5 allocs/op",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.input.String()
			if result != tt.wanted {
				t.Errorf("String() is called, want %q, have %q", tt.wanted, result)
			}
		})
	}
}
