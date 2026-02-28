// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package main

// Test a shared library created by -buildmode=c-shared that does not
// export anything.

import (
	"fmt"
	"os"
	"syscall"
)

// To test this we want to communicate between the main program and
// the shared library without using any exported symbols.  The init
// function creates a pipe and Dups the read end to a known number
// that the C code can also use.

const (
	fd = 30
)

func init() {
	var p [2]int
	if e := syscall.Pipe(p[0:]); e != nil {
		fmt.Fprintf(os.Stderr, "pipe: %v\n", e)
		os.Exit(2)
	}

	if e := dup2(p[0], fd); e != nil {
		fmt.Fprintf(os.Stderr, "dup2: %v\n", e)
		os.Exit(2)
	}

	const str = "PASS"
	if n, e := syscall.Write(p[1], []byte(str)); e != nil || n != len(str) {
		fmt.Fprintf(os.Stderr, "write: %d %v\n", n, e)
		os.Exit(2)
	}

	if e := syscall.Close(p[1]); e != nil {
		fmt.Fprintf(os.Stderr, "close: %v\n", e)
		os.Exit(2)
	}
}

func main() {
}
