// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build asan && linux && (arm64 || amd64 || loong64 || riscv64 || ppc64le)

package asan

/*
#cgo CFLAGS: -fsanitize=address
#cgo LDFLAGS: -fsanitize=address

#include <stdbool.h>
#include <stdint.h>
#include <sanitizer/asan_interface.h>
#include <sanitizer/lsan_interface.h>

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

void __lsan_register_root_region_go(void *addr, uintptr_t sz) {
	__lsan_register_root_region(addr, sz);
}

void __lsan_do_leak_check_go(void) {
	__lsan_do_leak_check();
}

// Keep in sync with the definition in compiler-rt
// https://github.com/llvm/llvm-project/blob/main/compiler-rt/lib/asan/asan_interface_internal.h#L41
// This structure is used to describe the source location of
// a place where global was defined.
struct _asan_global_source_location {
	const char *filename;
	int line_no;
	int column_no;
};

// Keep in sync with the definition in compiler-rt
// https://github.com/llvm/llvm-project/blob/main/compiler-rt/lib/asan/asan_interface_internal.h#L48
// So far, the current implementation is only compatible with the ASan library from version v7 to v9.
// https://github.com/llvm/llvm-project/blob/main/compiler-rt/lib/asan/asan_init_version.h
// This structure describes an instrumented global variable.
//
// TODO: If a later version of the ASan library changes __asan_global or __asan_global_source_location
// structure, we need to make the same changes.
struct _asan_global {
	uintptr_t beg;
	uintptr_t size;
	uintptr_t size_with_redzone;
	const char *name;
	const char *module_name;
	uintptr_t has_dynamic_init;
	struct _asan_global_source_location *location;
	uintptr_t odr_indicator;
};


extern void __asan_register_globals(void*, long int);

// Register global variables.
// The 'globals' is an array of structures describing 'n' globals.
void __asan_register_globals_go(void *addr, uintptr_t n) {
	struct _asan_global *globals = (struct _asan_global *)(addr);
	__asan_register_globals(globals, n);
}
*/
import "C"
