// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux,amd64

#include <errno.h>
#include <stdint.h>
#include <sys/mman.h>

void *
x_cgo_mmap(void *addr, uintptr_t length, int32_t prot, int32_t flags, int32_t fd, uint32_t offset) {
	void *p;

	p = mmap(addr, length, prot, flags, fd, offset);
	if (p == MAP_FAILED) {
		/* This is what the Go code expects on failure.  */
		p = (void *) (uintptr_t) errno;
	}
	return p;
}
