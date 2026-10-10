//go:build sharedtool

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "C"

import (
	"cmd/internal/sharedentry"
	"cmd/go/internal/base"
	"unsafe"
)

//export dll_go_main
func dll_go_main(argc C.int, argv **C.char) C.int {
	if argc < 0 {
		return 2
	}
	cargs := unsafe.Slice(argv, int(argc))
	args := make([]string, len(cargs))
	for i, arg := range cargs {
		args[i] = C.GoString(arg)
	}
	base.ResetExitStatus()
	return C.int(sharedentry.Invoke(args, main))
}
