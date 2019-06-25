// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin linux

package pprof

import (
	"fmt"
	"io"
	"syscall"
)

// Adds MaxRSS to platforms that are supported.
func addMaxRSS(w io.Writer) {
	var rusage syscall.Rusage
	syscall.Getrusage(0, &rusage)
	fmt.Fprintf(w, "# MaxRSS = %d\n", rusage.Maxrss)
}
