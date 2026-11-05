// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && (((amd64 || arm64) && (darwin || linux || netbsd || openbsd)) || ((386 || riscv64 || loong64) && (linux || netbsd || openbsd)) || (windows && amd64))

package cgo

import (
	"internal/abi"
	"runtime"
	"testing"
	"unsafe"
)

func trampolineCallers()

func TestTrampolineCallers(t *testing.T) {
	var pcs [16]uintptr
	var pin runtime.Pinner
	pin.Pin(&pcs[0])
	defer pin.Unpin()
	n := asmcgocall(abi.FuncPCABI0(trampolineCallers), uintptr(unsafe.Pointer(&pcs[0])), uintptr(len(pcs)), abi.FuncPCABIInternal(runtime.Callers), 0, 0, 0)
	caller := "runtime.asmcgocall"
	if runtime.GOOS == "windows" && runtime.GOARCH == "amd64" {
		// The Windows landing pad calls the trampoline instead of tail-calling it.
		caller = "runtime.asmcgocall_landingpad"
	}
	var names []string
	for _, pc := range pcs[:n] {
		name := runtime.FuncForPC(pc - 1).Name()
		if len(names) != 0 && names[len(names)-1] == "runtime/cgo.cgoCallTrampoline" && name == caller {
			return
		}
		names = append(names, name)
	}
	t.Fatalf("missing trampoline to %s transition in %v", caller, names)
}
