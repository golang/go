// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build js

package testing

import (
	"fmt"
	"io"
	"os"
	"strings"
	"time"
)

// TODO(@musiol, @odeke-em): unify this code back into
// example.go when js/wasm gets an os.Pipe implementation.
func runExample(eg InternalExample) (ok bool) {
	if *chatty {
		fmt.Printf("=== RUN   %s\n", eg.Name)
	}

	// Capture stdout to temporary file. We're not using
	// os.Pipe because it is not supported on js/wasm.
	stdout := os.Stdout
	f := createTempFile(eg.Name)
	os.Stdout = f
	start := time.Now()

	// Clean up in a deferred call so we can recover if the example panics.
	defer func() {
		timeSpent := time.Since(start)

		// Restore stdout, get output and remove temporary file.
		os.Stdout = stdout
		var buf strings.Builder
		_, seekErr := f.Seek(0, os.SEEK_SET)
		_, readErr := io.Copy(&buf, f)
		out := buf.String()
		f.Close()
		os.Remove(f.Name())
		if seekErr != nil {
			fmt.Fprintf(os.Stderr, "testing: seek temp file: %v\n", seekErr)
			os.Exit(1)
		}
		if readErr != nil {
			fmt.Fprintf(os.Stderr, "testing: read temp file: %v\n", readErr)
			os.Exit(1)
		}

		err := recover()
		ok = eg.processRunResult(out, timeSpent, err)
	}()

	// Run example.
	eg.F()
	return
}

func createTempFile(exampleName string) *os.File {
	for i := 0; ; i++ {
		name := fmt.Sprintf("%s/go-example-stdout-%s-%d.txt", os.TempDir(), exampleName, i)
		f, err := os.OpenFile(name, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			if os.IsExist(err) {
				continue
			}
			fmt.Fprintf(os.Stderr, "testing: open temp file: %v\n", err)
			os.Exit(1)
		}
		return f
	}
}
