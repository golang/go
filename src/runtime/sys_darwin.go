// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// The *_trampoline functions convert from the Go calling convention to the C calling convention
// and then call the underlying libc function.  They are defined in sys_darwin_$ARCH.s.

//go:nowritebarrier
func pthread_attr_init(attr *pthreadattr) (errno int32) {
	systemstack(func() {
		errno = pthread_attr_init_trampoline(attr)
	})
	return
}

//go:noescape
func pthread_attr_init_trampoline(attr *pthreadattr) int32

//go:nowritebarrier
func pthread_attr_setstacksize(attr *pthreadattr, size uintptr) (errno int32) {
	systemstack(func() {
		errno = pthread_attr_setstacksize_trampoline(attr, size)
	})
	return
}

//go:noescape
func pthread_attr_setstacksize_trampoline(attr *pthreadattr, size uintptr) int32

//go:nowritebarrier
func pthread_attr_setdetachstate(attr *pthreadattr, state int) (errno int32) {
	systemstack(func() {
		errno = pthread_attr_setdetachstate_trampoline(attr, state)
	})
	return
}

//go:noescape
func pthread_attr_setdetachstate_trampoline(attr *pthreadattr, state int) int32

//go:nowritebarrier
func pthread_create(attr *pthreadattr, start uintptr, arg unsafe.Pointer) (t pthread, errno int32) {
	systemstack(func() {
		errno = pthread_create_trampoline(&t, attr, start, arg)
	})
	return
}

//go:noescape
func pthread_create_trampoline(t *pthread, attr *pthreadattr, start uintptr, arg unsafe.Pointer) int32

//go:nowritebarrier
func pthread_kill(thread pthread, sig int) (errno int32) {
	systemstack(func() {
		errno = pthread_kill_trampoline(thread, sig)
	})
	return
}

//go:noescape
func pthread_kill_trampoline(thread pthread, sig int) int32

//go:nowritebarrier
func pthread_self() (t pthread) {
	systemstack(func() {
		t = pthread_self_trampoline()
	})
	return
}

//go:noescape
func pthread_self_trampoline() pthread

// Tell the linker that the libc_* functions are to be found
// in a system library, with the libc_ prefix missing.

//go:cgo_import_dynamic libc_pthread_attr_init pthread_attr_init "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_attr_setstacksize pthread_attr_setstacksize "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_attr_setdetachstate pthread_attr_setdetachstate "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_create pthread_create "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_exit exit "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_kill pthread_kill "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_self pthread_self "/usr/lib/libSystem.B.dylib"

//go:cgo_import_dynamic libc_mmap mmap "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_munmap munmap "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_madvise madvise "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_error __error "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_usleep usleep "/usr/lib/libSystem.B.dylib"

// Magic incantation to get libSystem actually dynamically linked.
// TODO: Why does the code require this?  See cmd/compile/internal/ld/go.go:210
//go:cgo_import_dynamic _ _ "/usr/lib/libSystem.B.dylib"
