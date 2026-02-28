// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js

// TODO(@musiol, @odeke-em): re-unify this entire file back into
// example.go when js/wasm gets an os.Pipe implementation
// and no longer needs this separation.

package testing

import (
	"fmt"
	"io"
	"os"
	"strings"
	"time"
)

func runExample(eg InternalExample) (ok bool) {
	if *chatty {
		fmt.Printf("=== RUN   %s\n", eg.Name)
	}

	// Capture stdout.
	stdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	os.Stdout = w
	outC := make(chan string)
	go func() {
		var buf strings.Builder
		_, err := io.Copy(&buf, r)
		r.Close()
		if err != nil {
			fmt.Fprintf(os.Stderr, "testing: copying pipe: %v\n", err)
			os.Exit(1)
		}
		outC <- buf.String()
	}()

	finished := false
	start := time.Now()

	// Clean up in a deferred call so we can recover if the example panics.
	defer func() {
		timeSpent := time.Since(start)

		// Close pipe, restore stdout, get output.
		w.Close()
		os.Stdout = stdout
		out := <-outC

		err := recover()
		ok = eg.processRunResult(out, timeSpent, finished, err)
	}()

	// Run example.
	eg.F()
	finished = true
	return
}
