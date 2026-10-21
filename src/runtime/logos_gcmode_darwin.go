// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin

package runtime

import "unsafe"

var logosHWMemsize = []byte("hw.memsize\x00")

// logosPhysicalMemory returns hw.memsize in bytes, or 0 when sysctl fails.
func logosPhysicalMemory() uint64 {
	var mem uint64
	n := unsafe.Sizeof(mem)
	if sysctlbyname(&logosHWMemsize[0], (*byte)(unsafe.Pointer(&mem)), &n, nil, 0) != 0 {
		return 0
	}
	return mem
}
