// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build asan && linux && (arm64 || amd64)

package asan

/*
#cgo CFLAGS: -fsanitize=address
#cgo LDFLAGS: -fsanitize=address

#include <stdbool.h>
#include <stdint.h>
#include <sanitizer/asan_interface.h>

void __asan_read_go(void *addr, uintptr_t sz, void *sp, void *pc) {
	if (__asan_region_is_poisoned(addr, sz)) {
		__asan_report_error(pc, 0, sp, addr, false, sz);
	}
}

void __asan_write_go(void *addr, uintptr_t sz, void *sp, void *pc) {
	if (__asan_region_is_poisoned(addr, sz)) {
		__asan_report_error(pc, 0, sp, addr, true, sz);
	}
}

void __asan_unpoison_go(void *addr, uintptr_t sz) {
	__asan_unpoison_memory_region(addr, sz);
}

void __asan_poison_go(void *addr, uintptr_t sz) {
	__asan_poison_memory_region(addr, sz);
}

*/
import "C"
