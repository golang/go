// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package runtime

import "unsafe"

const logosORdonly = 0

var logosMemInfoPath = []byte("/proc/meminfo\x00")

// logosPhysicalMemory returns MemTotal of /proc/meminfo in bytes, or 0 when
// the file cannot be read.
func logosPhysicalMemory() uint64 {
	var buf [256]byte
	fd := open(&logosMemInfoPath[0], logosORdonly, 0)
	if fd < 0 {
		return 0
	}
	n := read(fd, noescape(unsafe.Pointer(&buf[0])), int32(len(buf)))
	closefd(fd)
	if n <= 0 {
		return 0
	}
	return logosParseMemTotal(buf[:n])
}

// logosParseMemTotal returns the MemTotal value of the head of
// /proc/meminfo in bytes, or 0 when the head has another shape.
func logosParseMemTotal(head []byte) uint64 {
	const key = "MemTotal:"
	if len(head) < len(key) || string(head[:len(key)]) != key {
		return 0
	}
	kib, _ := logosParseUint(head[len(key):])
	return kib << 10
}
