// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build openbsd && !mips64

package runtime

import (
	"internal/abi"
	"unsafe"
)

// The *_trampoline functions convert from the Go calling convention to the C calling convention
// and then call the underlying libc function. These are defined in sys_openbsd_$ARCH.s.

//go:nosplit
//go:cgo_unsafe_args
func pthread_attr_init(attr *pthreadattr) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_attr_init_trampoline)), unsafe.Pointer(&attr))
	KeepAlive(attr)
	return ret
}
func pthread_attr_init_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_attr_destroy(attr *pthreadattr) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_attr_destroy_trampoline)), unsafe.Pointer(&attr))
	KeepAlive(attr)
	return ret
}
func pthread_attr_destroy_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_attr_getstacksize(attr *pthreadattr, size *uintptr) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_attr_getstacksize_trampoline)), unsafe.Pointer(&attr))
	KeepAlive(attr)
	KeepAlive(size)
	return ret
}
func pthread_attr_getstacksize_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_attr_setdetachstate(attr *pthreadattr, state int) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_attr_setdetachstate_trampoline)), unsafe.Pointer(&attr))
	KeepAlive(attr)
	return ret
}
func pthread_attr_setdetachstate_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_create(attr *pthreadattr, start uintptr, arg unsafe.Pointer) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_create_trampoline)), unsafe.Pointer(&attr))
	KeepAlive(attr)
	KeepAlive(arg) // Just for consistency. Arg of course needs to be kept alive for the start function.
	return ret
}
func pthread_create_trampoline()

// Tell the linker that the libc_* functions are to be found
// in a system library, with the libc_ prefix missing.

//go:cgo_import_dynamic libc_pthread_attr_init pthread_attr_init "libpthread.so"
//go:cgo_import_dynamic libc_pthread_attr_destroy pthread_attr_destroy "libpthread.so"
//go:cgo_import_dynamic libc_pthread_attr_getstacksize pthread_attr_getstacksize "libpthread.so"
//go:cgo_import_dynamic libc_pthread_attr_setdetachstate pthread_attr_setdetachstate "libpthread.so"
//go:cgo_import_dynamic libc_pthread_create pthread_create "libpthread.so"
//go:cgo_import_dynamic libc_pthread_sigmask pthread_sigmask "libpthread.so"

//go:cgo_import_dynamic _ _ "libpthread.so"
//go:cgo_import_dynamic _ _ "libc.so"
