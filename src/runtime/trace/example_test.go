// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace_test

import (
	"fmt"
	"log"
	"os"
	"runtime/trace"
)

// Example demonstrates the use of the trace package to trace
// the execution of a Go program. The trace output will be
// written to the file trace.out
func Example() {
	f, err := os.Create("trace.out")
	if err != nil {
		log.Fatalf("failed to create trace output file: %v", err)
	}
	defer func() {
		if err := f.Close(); err != nil {
			log.Fatalf("failed to close trace file: %v", err)
		}
	}()

	if err := trace.Start(f); err != nil {
		log.Fatalf("failed to start trace: %v", err)
	}
	defer trace.Stop()

	// your program here
	RunMyProgram()
}

func RunMyProgram() {
	fmt.Printf("this function will be traced")
}
