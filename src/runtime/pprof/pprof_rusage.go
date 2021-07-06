// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || linux
// +build darwin linux

package pprof

import (
	"fmt"
	"io"
	"runtime"
	"syscall"
)

// Adds MaxRSS to platforms that are supported.
func addMaxRSS(w io.Writer) {
	var rssToBytes uintptr
	switch runtime.GOOS {
	case "linux", "android":
		rssToBytes = 1024
	case "darwin", "ios":
		rssToBytes = 1
	default:
		panic("unsupported OS")
	}

	var rusage syscall.Rusage
	syscall.Getrusage(0, &rusage)
	fmt.Fprintf(w, "# MaxRSS = %d\n", uintptr(rusage.Maxrss)*rssToBytes)
}
