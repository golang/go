// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"syscall"
	"unsafe"
)

var (
	modkernel32       = syscall.NewLazyDLL("kernel32.dll")
	procGetSystemInfo = modkernel32.NewProc("GetSystemInfo")
)

// see https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/ns-sysinfoapi-system_info
type systeminfo struct {
	wProcessorArchitecture      uint16
	wReserved                   uint16
	dwPageSize                  uint32
	lpMinimumApplicationAddress uintptr
	lpMaximumApplicationAddress uintptr
	dwActiveProcessorMask       uintptr
	dwNumberOfProcessors        uint32
	dwProcessorType             uint32
	dwAllocationGranularity     uint32
	wProcessorLevel             uint16
	wProcessorRevision          uint16
}

// See https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/ns-sysinfoapi-system_info
const (
	PROCESSOR_ARCHITECTURE_AMD64 = 9
	PROCESSOR_ARCHITECTURE_INTEL = 0
	PROCESSOR_ARCHITECTURE_ARM   = 5
	PROCESSOR_ARCHITECTURE_ARM64 = 12
	PROCESSOR_ARCHITECTURE_IA64  = 6
)

var sysinfo systeminfo

func sysinit() {
	syscall.Syscall(procGetSystemInfo.Addr(), 1, uintptr(unsafe.Pointer(&sysinfo)), 0, 0)
	switch sysinfo.wProcessorArchitecture {
	case PROCESSOR_ARCHITECTURE_AMD64:
		gohostarch = "amd64"
	case PROCESSOR_ARCHITECTURE_INTEL:
		gohostarch = "386"
	case PROCESSOR_ARCHITECTURE_ARM:
		gohostarch = "arm"
	case PROCESSOR_ARCHITECTURE_ARM64:
		gohostarch = "arm64"
	default:
		fatalf("unknown processor architecture")
	}
}
