// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race,linux,amd64 race,darwin,amd64 race,windows,amd64

package race

/*
void __tsan_init(void **racectx);
void __tsan_fini(void);
void __tsan_map_shadow(void *addr, void *size);
void __tsan_go_start(void *racectx, void **chracectx, void *pc);
void __tsan_go_end(void *racectx);
void __tsan_read(void *racectx, void *addr, void *pc);
void __tsan_write(void *racectx, void *addr, void *pc);
void __tsan_read_range(void *racectx, void *addr, long sz, long step, void *pc);
void __tsan_write_range(void *racectx, void *addr, long sz, long step, void *pc);
void __tsan_func_enter(void *racectx, void *pc);
void __tsan_func_exit(void *racectx);
void __tsan_malloc(void *racectx, void *p, long sz, void *pc);
void __tsan_free(void *p);
void __tsan_acquire(void *racectx, void *addr);
void __tsan_release(void *racectx, void *addr);
void __tsan_release_merge(void *racectx, void *addr);
void __tsan_finalizer_goroutine(void *racectx);
*/
import "C"

import (
	"runtime"
	"unsafe"
)

func Initialize(racectx *uintptr) {
	C.__tsan_init((*unsafe.Pointer)(unsafe.Pointer(racectx)))
}

func Finalize() {
	C.__tsan_fini()
}

func MapShadow(addr, size uintptr) {
	C.__tsan_map_shadow(unsafe.Pointer(addr), unsafe.Pointer(size))
}

func FinalizerGoroutine(racectx uintptr) {
	C.__tsan_finalizer_goroutine(unsafe.Pointer(racectx))
}

func Read(racectx uintptr, addr, pc uintptr) {
	C.__tsan_read(unsafe.Pointer(racectx), unsafe.Pointer(addr), unsafe.Pointer(pc))
}

func Write(racectx uintptr, addr, pc uintptr) {
	C.__tsan_write(unsafe.Pointer(racectx), unsafe.Pointer(addr), unsafe.Pointer(pc))
}

func ReadRange(racectx uintptr, addr, sz, step, pc uintptr) {
	C.__tsan_read_range(unsafe.Pointer(racectx), unsafe.Pointer(addr),
		C.long(sz), C.long(step), unsafe.Pointer(pc))
}

func WriteRange(racectx uintptr, addr, sz, step, pc uintptr) {
	C.__tsan_write_range(unsafe.Pointer(racectx), unsafe.Pointer(addr),
		C.long(sz), C.long(step), unsafe.Pointer(pc))
}

func FuncEnter(racectx uintptr, pc uintptr) {
	C.__tsan_func_enter(unsafe.Pointer(racectx), unsafe.Pointer(pc))
}

func FuncExit(racectx uintptr) {
	C.__tsan_func_exit(unsafe.Pointer(racectx))
}

func Malloc(racectx uintptr, p, sz, pc uintptr) {
	C.__tsan_malloc(unsafe.Pointer(racectx), unsafe.Pointer(p), C.long(sz), unsafe.Pointer(pc))
}

func Free(p uintptr) {
	C.__tsan_free(unsafe.Pointer(p))
}

func GoStart(racectx uintptr, chracectx *uintptr, pc uintptr) {
	C.__tsan_go_start(unsafe.Pointer(racectx), (*unsafe.Pointer)(unsafe.Pointer(chracectx)), unsafe.Pointer(pc))
}

func GoEnd(racectx uintptr) {
	C.__tsan_go_end(unsafe.Pointer(racectx))
}

func Acquire(racectx uintptr, addr uintptr) {
	C.__tsan_acquire(unsafe.Pointer(racectx), unsafe.Pointer(addr))
}

func Release(racectx uintptr, addr uintptr) {
	C.__tsan_release(unsafe.Pointer(racectx), unsafe.Pointer(addr))
}

func ReleaseMerge(racectx uintptr, addr uintptr) {
	C.__tsan_release_merge(unsafe.Pointer(racectx), unsafe.Pointer(addr))
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
