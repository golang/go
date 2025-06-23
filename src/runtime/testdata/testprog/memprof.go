// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
)

func init() {
	register("MemProf", MemProf)
}

var memProfBuf bytes.Buffer
var memProfStr string

func MemProf() {
	// Force heap sampling for determinism.
	runtime.MemProfileRate = 1

	for i := 0; i < 10; i++ {
		fmt.Fprintf(&memProfBuf, "%*d\n", i, i)
	}
	memProfStr = memProfBuf.String()

	runtime.GC()

	f, err := os.CreateTemp("", "memprof")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	if err := pprof.WriteHeapProfile(f); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	name := f.Name()
	if err := f.Close(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	fmt.Println(name)
}
