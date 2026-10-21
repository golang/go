// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package runtime

import "unsafe"

//go:cgo_import_dynamic runtime._GlobalMemoryStatusEx GlobalMemoryStatusEx%1 "kernel32.dll"

var _GlobalMemoryStatusEx stdFunction

// logosMemoryStatusEx is the Windows MEMORYSTATUSEX structure.
type logosMemoryStatusEx struct {
	length               uint32
	memoryLoad           uint32
	totalPhys            uint64
	availPhys            uint64
	totalPageFile        uint64
	availPageFile        uint64
	totalVirtual         uint64
	availVirtual         uint64
	availExtendedVirtual uint64
}

// logosPhysicalMemory returns ullTotalPhys of GlobalMemoryStatusEx in bytes,
// or 0 when the call fails.
func logosPhysicalMemory() uint64 {
	var ms logosMemoryStatusEx
	ms.length = uint32(unsafe.Sizeof(ms))
	if stdcall(_GlobalMemoryStatusEx, uintptr(unsafe.Pointer(&ms))) == 0 {
		return 0
	}
	return ms.totalPhys
}
