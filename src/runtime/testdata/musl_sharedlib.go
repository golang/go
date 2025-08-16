// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && cgo

package main

import "C"
import (
	"fmt"
	"os"
)

// This is compiled as a c-shared library to test standards compliance.
// On non-glibc systems, DT_INIT_ARRAY functions are not passed argc/argv/envp
// per ELF specification, so we need to handle null argv gracefully.

//export TestMuslInit
func TestMuslInit() {
	// If we got here without SIGSEGV, the fix is working
	fmt.Fprintf(os.Stderr, "MUSL_INIT_SUCCESS\n")
}

//export GetArgCount  
func GetArgCount() int {
	// Return the number of command line arguments
	// This tests if argc is accessible
	return len(os.Args)
}

//export GetArg
func GetArg(index int) string {
	// Return a specific command line argument
	// This tests if argv is accessible
	if index < 0 || index >= len(os.Args) {
		return ""
	}
	return os.Args[index]
}

func main() {
	// This is needed for c-shared but won't be called
}
