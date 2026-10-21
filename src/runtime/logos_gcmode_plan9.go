// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build plan9

package runtime

import "unsafe"

var logosSwapPath = []byte("/dev/swap\x00")

// logosPhysicalMemory returns the first number of /dev/swap, the bytes of
// physical memory ("N memory"), or 0 when the file cannot be read.
func logosPhysicalMemory() uint64 {
	var buf [32]byte
	fd := open(&logosSwapPath[0], 0, 0)
	if fd < 0 {
		return 0
	}
	n := pread(fd, noescape(unsafe.Pointer(&buf[0])), int32(len(buf)), 0)
	closefd(fd)
	if n <= 0 {
		return 0
	}
	value, _ := logosParseUint(buf[:n])
	return value
}
