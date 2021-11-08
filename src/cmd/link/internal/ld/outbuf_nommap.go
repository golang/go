// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !aix && !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd && !windows
// +build !aix,!darwin,!dragonfly,!freebsd,!linux,!netbsd,!openbsd,!windows

package ld

// Mmap allocates an in-heap output buffer with the given size. It copies
// any old data (if any) to the new buffer.
func (out *OutBuf) Mmap(filesize uint64) error {
	// We need space to put all the symbols before we apply relocations.
	oldheap := out.heap
	if filesize < uint64(len(oldheap)) {
		panic("mmap size too small")
	}
	out.heap = make([]byte, filesize)
	copy(out.heap, oldheap)
	return nil
}

func (out *OutBuf) munmap() { panic("unreachable") }
