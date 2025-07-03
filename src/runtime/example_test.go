// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"os"
	"runtime"
	"strings"
)

func ExampleFrames() {
	c := func() {
		// Ask runtime.Callers for up to 10 PCs, including runtime.Callers itself.
		pc := make([]uintptr, 10)
		n := runtime.Callers(0, pc)
		if n == 0 {
			// No PCs available. This can happen if the first argument to
			// runtime.Callers is large.
			//
			// Return now to avoid processing the zero Frame that would
			// otherwise be returned by frames.Next below.
			return
		}

		pc = pc[:n] // pass only valid pcs to runtime.CallersFrames
		frames := runtime.CallersFrames(pc)

		// Loop to get frames.
		// A fixed number of PCs can expand to an indefinite number of Frames.
		for {
			frame, more := frames.Next()

			// Canonicalize function name and skip callers of this function
			// for predictable example output.
			// You probably don't need this in your own code.
			function := strings.ReplaceAll(frame.Function, "main.main", "runtime_test.ExampleFrames")
			fmt.Printf("- more:%v | %s\n", more, function)
			if function == "runtime_test.ExampleFrames" {
				break
			}

			// Check whether there are more frames to process after this one.
			if !more {
				break
			}
		}
	}

	b := func() { c() }
	a := func() { b() }

	a()
	// Output:
	// - more:true | runtime.Callers
	// - more:true | runtime_test.ExampleFrames.func1
	// - more:true | runtime_test.ExampleFrames.func2
	// - more:true | runtime_test.ExampleFrames.func3
	// - more:true | runtime_test.ExampleFrames
}

func ExampleAddCleanup() {
	tempFile, err := os.CreateTemp(os.TempDir(), "file.*")
	if err != nil {
		fmt.Println("failed to create temp file:", err)
		return
	}

	ch := make(chan struct{})

	// Attach a cleanup function to the file object.
	runtime.AddCleanup(&tempFile, func(fileName string) {
		if err := os.Remove(fileName); err == nil {
			fmt.Println("temp file has been removed")
		}
		ch <- struct{}{}
	}, tempFile.Name())

	if err := tempFile.Close(); err != nil {
		fmt.Println("failed to close temp file:", err)
		return
	}

	// Run the garbage collector to reclaim unreachable objects
	// and enqueue their cleanup functions.
	runtime.GC()

	// Wait until cleanup function is done.
	<-ch

	// Output:
	// temp file has been removed
}
