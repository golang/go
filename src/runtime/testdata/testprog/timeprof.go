// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"runtime/pprof"
	"time"
)

func init() {
	register("TimeProf", TimeProf)
}

func TimeProf() {
	f, err := os.CreateTemp("", "timeprof")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	if err := pprof.StartCPUProfile(f); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	t0 := time.Now()
	// We should get a profiling signal 100 times a second,
	// so running for 1/10 second should be sufficient.
	for time.Since(t0) < time.Second/10 {
	}

	pprof.StopCPUProfile()

	name := f.Name()
	if err := f.Close(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	fmt.Println(name)
}
