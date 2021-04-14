// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin

package unix

import "unsafe"

// ReadDirent reads directory entries from fd and writes them into buf.
func ReadDirent(fd int, buf []byte) (n int, err error) {
	// Final argument is (basep *uintptr) and the syscall doesn't take nil.
	// 64 bits should be enough. (32 bits isn't even on 386). Since the
	// actual system call is getdirentries64, 64 is a good guess.
	// TODO(rsc): Can we use a single global basep for all calls?
	var base = (*uintptr)(unsafe.Pointer(new(uint64)))
	return Getdirentries(fd, buf, base)
}
