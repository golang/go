// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package unix_test

import (
	"testing"

	"golang.org/x/sys/unix"
)

func TestMmap(t *testing.T) {
	b, err := unix.Mmap(-1, 0, unix.Getpagesize(), unix.PROT_NONE, unix.MAP_ANON|unix.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("Mmap: %v", err)
	}
	if err := unix.Mprotect(b, unix.PROT_READ|unix.PROT_WRITE); err != nil {
		t.Fatalf("Mprotect: %v", err)
	}

	b[0] = 42

	if err := unix.Msync(b, unix.MS_SYNC); err != nil {
		t.Fatalf("Msync: %v", err)
	}
	if err := unix.Madvise(b, unix.MADV_DONTNEED); err != nil {
		t.Fatalf("Madvise: %v", err)
	}
	if err := unix.Munmap(b); err != nil {
		t.Fatalf("Munmap: %v", err)
	}
}
