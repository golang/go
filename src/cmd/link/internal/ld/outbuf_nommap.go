// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !darwin,!dragonfly,!freebsd,!linux,!openbsd,!windows

package ld

func (out *OutBuf) Mmap(filesize uint64) error {
	// We need space to put all the symbols before we apply relocations.
	out.heap = make([]byte, filesize)
	return nil
}

func (out *OutBuf) munmap() { panic("unreachable") }
