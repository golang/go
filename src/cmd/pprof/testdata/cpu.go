// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"os"
	"runtime/pprof"
	"time"
)

var output = flag.String("output", "", "pprof profile output file")

func main() {
	flag.Parse()
	if *output == "" {
		fmt.Fprintf(os.Stderr, "usage: %s -output file.pprof\n", os.Args[0])
		os.Exit(2)
	}

	f, err := os.Create(*output)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	defer f.Close()

	if err := pprof.StartCPUProfile(f); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	defer pprof.StopCPUProfile()

	// Spin for long enough to collect some samples.
	start := time.Now()
	for time.Since(start) < time.Second {
	}
}
