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

extern void __asan_report_load1(void*);
extern void __asan_report_load2(void*);
extern void __asan_report_load4(void*);
extern void __asan_report_load8(void*);
extern void __asan_report_load_n(void*, uintptr_t);
extern void __asan_report_store1(void*);
extern void __asan_report_store2(void*);
extern void __asan_report_store4(void*);
extern void __asan_report_store8(void*);
extern void __asan_report_store_n(void*, uintptr_t);

void __asan_read_go(void *addr, uintptr_t sz) {
	if (__asan_region_is_poisoned(addr, sz)) {
		switch (sz) {
		case 1: __asan_report_load1(addr); break;
		case 2: __asan_report_load2(addr); break;
		case 4: __asan_report_load4(addr); break;
		case 8: __asan_report_load8(addr); break;
		default: __asan_report_load_n(addr, sz); break;
		}
	}
}

void __asan_write_go(void *addr, uintptr_t sz) {
	if (__asan_region_is_poisoned(addr, sz)) {
		switch (sz) {
		case 1: __asan_report_store1(addr); break;
		case 2: __asan_report_store2(addr); break;
		case 4: __asan_report_store4(addr); break;
		case 8: __asan_report_store8(addr); break;
		default: __asan_report_store_n(addr, sz); break;
		}
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
