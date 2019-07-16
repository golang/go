// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that the Go environment variables are present and accessible through
// package os and package runtime.

package main

import (
	"os"
	"runtime"
)

func main() {
	ga := os.Getenv("GOARCH")
	if ga != runtime.GOARCH {
		print("$GOARCH=", ga, "!= runtime.GOARCH=", runtime.GOARCH, "\n")
		os.Exit(1)
	}
	xxx := os.Getenv("DOES_NOT_EXIST")
	if xxx != "" {
		print("$DOES_NOT_EXIST=", xxx, "\n")
		os.Exit(1)
	}
}
