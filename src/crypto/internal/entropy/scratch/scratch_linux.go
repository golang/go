// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scratch

import "syscall"

func mmap(n int) ([]byte, error) {
	// MAP_POPULATE faults in all the pages at once, which is much faster than
	// faulting them in one at a time as they are first accessed.
	return syscall.Mmap(-1, 0, n, syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_ANON|syscall.MAP_PRIVATE|syscall.MAP_POPULATE)
}

func munmap(b []byte) {
	syscall.Munmap(b)
}
