// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The gostacks command processes stdin looking for things that look like
// stack traces and simplifying them to make the log more readable.
// It collates stack traces that have the same path as well as simplifying the
// individual lines of the trace.
// The processed log is printed to stdout.
package main

import (
	"fmt"
	"os"

	"golang.org/x/tools/internal/stack"
)

func main() {
	if err := stack.Process(os.Stdout, os.Stdin); err != nil {
		fmt.Fprintln(os.Stderr, err)
	}
}
