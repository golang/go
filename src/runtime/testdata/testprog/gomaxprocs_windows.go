// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"runtime"
	"syscall"
	"unsafe"
)

func init() {
	register("WindowsUpdateGOMAXPROCS", WindowsUpdateGOMAXPROCS)
	register("WindowsDontUpdateGOMAXPROCS", WindowsDontUpdateGOMAXPROCS)
}

// Set CPU affinity mask to only two CPUs.
//
// Skips the test if CPUs 0 and 1 are not available.
func setAffinity2() {
	kernel32 := syscall.MustLoadDLL("kernel32.dll")
	_GetProcessAffinityMask := kernel32.MustFindProc("GetProcessAffinityMask")
	_SetProcessAffinityMask := kernel32.MustFindProc("SetProcessAffinityMask")

	h, err := syscall.GetCurrentProcess()
	if err != nil {
		panic(err)
	}

	var mask, sysmask uintptr
	ret, _, err := _GetProcessAffinityMask.Call(uintptr(h), uintptr(unsafe.Pointer(&mask)), uintptr(unsafe.Pointer(&sysmask)))
	if ret == 0 {
		panic(err)
	}

	// We're going to restrict to CPUs 0 and 1. Make sure those are already available.
	if mask & 0b11 != 0b11 {
		println("SKIP: CPUs 0 and 1 not available")
		os.Exit(0)
	}

	mask = 0b11
	ret, _, err = _SetProcessAffinityMask.Call(uintptr(h), mask)
	if ret == 0 {
		panic(err)
	}
}

func WindowsUpdateGOMAXPROCS() {
	ncpu := runtime.NumCPU()
	setAffinity2()
	waitForMaxProcsChange(ncpu, 2)
	println("OK")
}

func WindowsDontUpdateGOMAXPROCS() {
	procs := runtime.GOMAXPROCS(0)
	setAffinity2()
	mustNotChangeMaxProcs(procs)
	println("OK")
}
