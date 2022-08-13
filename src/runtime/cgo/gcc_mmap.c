// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux,amd64 linux,arm64 linux,ppc64le

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>

#include "libcgo.h"

uintptr_t
x_cgo_mmap(void *addr, uintptr_t length, int32_t prot, int32_t flags, int32_t fd, uint32_t offset) {
	void *p;

	_cgo_tsan_acquire();
	p = mmap(addr, length, prot, flags, fd, offset);
	_cgo_tsan_release();
	if (p == MAP_FAILED) {
		/* This is what the Go code expects on failure.  */
		return (uintptr_t)errno;
	}
	return (uintptr_t)p;
}

void
x_cgo_munmap(void *addr, uintptr_t length) {
	int r;

	_cgo_tsan_acquire();
	r = munmap(addr, length);
	_cgo_tsan_release();
	if (r < 0) {
		/* The Go runtime is not prepared for munmap to fail.  */
		abort();
	}
}
