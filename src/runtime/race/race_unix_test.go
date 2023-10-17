// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build race && (darwin || freebsd || linux)

package race_test

import (
	"sync/atomic"
	"syscall"
	"testing"
	"unsafe"
)

// Test that race detector does not crash when accessing non-Go allocated memory (issue 9136).
func TestNonGoMemory(t *testing.T) {
	data, err := syscall.Mmap(-1, 0, 4096, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("failed to mmap memory: %v", err)
	}
	defer syscall.Munmap(data)
	p := (*uint32)(unsafe.Pointer(&data[0]))
	atomic.AddUint32(p, 1)
	(*p)++
	if *p != 2 {
		t.Fatalf("data[0] = %v, expect 2", *p)
	}
}
