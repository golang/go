// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"os"
	"runtime"
	"testing"

	fast "./fast"
	slow "./slow"
)

var buf = make([]byte, 1048576)

func BenchmarkFastNonASCII(b *testing.B) {
	for i := 0; i < b.N; i++ {
		fast.NonASCII(buf, 0)
	}
}

func BenchmarkSlowNonASCII(b *testing.B) {
	for i := 0; i < b.N; i++ {
		slow.NonASCII(buf, 0)
	}
}

func main() {
	testing.Init()
	os.Args = []string{os.Args[0], "-test.benchtime=100ms"}
	flag.Parse()

	rslow := testing.Benchmark(BenchmarkSlowNonASCII)
	rfast := testing.Benchmark(BenchmarkFastNonASCII)
	tslow := rslow.NsPerOp()
	tfast := rfast.NsPerOp()

	// Optimization should be good for at least 2x, but be forgiving.
	// On the ARM simulator we see closer to 1.5x.
	speedup := float64(tslow) / float64(tfast)
	want := 1.8
	if runtime.GOARCH == "arm" {
		want = 1.3
	}
	if speedup < want {
		// TODO(rsc): doesn't work on linux-amd64 or darwin-amd64 builders, nor on
		// a Lenovo x200 (linux-amd64) laptop.
		// println("fast:", tfast, "slow:", tslow, "speedup:", speedup, "want:", want)
		// println("not fast enough")
		// os.Exit(1)
	}
}
