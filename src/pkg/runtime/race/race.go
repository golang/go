// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race,linux,amd64 race,darwin,amd64

// Package race provides low-level facilities for data race detection.
package race

/*
void __tsan_init(void);
void __tsan_fini(void);
void __tsan_map_shadow(void *addr, void *size);
void __tsan_go_start(int pgoid, int chgoid, void *pc);
void __tsan_go_end(int goid);
void __tsan_read(int goid, void *addr, void *pc);
void __tsan_write(int goid, void *addr, void *pc);
void __tsan_func_enter(int goid, void *pc);
void __tsan_func_exit(int goid);
void __tsan_malloc(int goid, void *p, long sz, void *pc);
void __tsan_free(void *p);
void __tsan_acquire(int goid, void *addr);
void __tsan_release(int goid, void *addr);
void __tsan_release_merge(int goid, void *addr);
void __tsan_finalizer_goroutine(int tid);
*/
import "C"

import (
	"runtime"
	"unsafe"
)

func Initialize() {
	C.__tsan_init()
}

func Finalize() {
	C.__tsan_fini()
}

func MapShadow(addr, size uintptr) {
	C.__tsan_map_shadow(unsafe.Pointer(addr), unsafe.Pointer(size))
}

func FinalizerGoroutine(goid int32) {
	C.__tsan_finalizer_goroutine(C.int(goid))
}

func Read(goid int32, addr, pc uintptr) {
	C.__tsan_read(C.int(goid), unsafe.Pointer(addr), unsafe.Pointer(pc))
}

func Write(goid int32, addr, pc uintptr) {
	C.__tsan_write(C.int(goid), unsafe.Pointer(addr), unsafe.Pointer(pc))
}

func FuncEnter(goid int32, pc uintptr) {
	C.__tsan_func_enter(C.int(goid), unsafe.Pointer(pc))
}

func FuncExit(goid int32) {
	C.__tsan_func_exit(C.int(goid))
}

func Malloc(goid int32, p, sz, pc uintptr) {
	C.__tsan_malloc(C.int(goid), unsafe.Pointer(p), C.long(sz), unsafe.Pointer(pc))
}

func Free(p uintptr) {
	C.__tsan_free(unsafe.Pointer(p))
}

func GoStart(pgoid, chgoid int32, pc uintptr) {
	C.__tsan_go_start(C.int(pgoid), C.int(chgoid), unsafe.Pointer(pc))
}

func GoEnd(goid int32) {
	C.__tsan_go_end(C.int(goid))
}

func Acquire(goid int32, addr uintptr) {
	C.__tsan_acquire(C.int(goid), unsafe.Pointer(addr))
}

func Release(goid int32, addr uintptr) {
	C.__tsan_release(C.int(goid), unsafe.Pointer(addr))
}

func ReleaseMerge(goid int32, addr uintptr) {
	C.__tsan_release_merge(C.int(goid), unsafe.Pointer(addr))
}

//export __tsan_symbolize
func __tsan_symbolize(pc uintptr, fun, file **C.char, line, off *C.int) C.int {
	f := runtime.FuncForPC(pc)
	if f == nil {
		*fun = C.CString("??")
		*file = C.CString("-")
		*line = 0
		*off = C.int(pc)
		return 1
	}
	fi, l := f.FileLine(pc)
	*fun = C.CString(f.Name())
	*file = C.CString(fi)
	*line = C.int(l)
	*off = C.int(pc - f.Entry())
	return 1
}
