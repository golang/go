// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

// libc function wrappers. Must run on system stack.

//go:nosplit
//go:cgo_unsafe_args
func g0_pthread_key_create(k *pthreadkey, destructor uintptr) int32 {
	return asmcgocall(unsafe.Pointer(funcPC(pthread_key_create_trampoline)), unsafe.Pointer(&k))
}
func pthread_key_create_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func g0_pthread_setspecific(k pthreadkey, value uintptr) int32 {
	return asmcgocall(unsafe.Pointer(funcPC(pthread_setspecific_trampoline)), unsafe.Pointer(&k))
}
func pthread_setspecific_trampoline()

//go:cgo_import_dynamic libc_pthread_key_create pthread_key_create "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_setspecific pthread_setspecific "/usr/lib/libSystem.B.dylib"

// tlsinit allocates a thread-local storage slot for g.
//
// It finds the first available slot using pthread_key_create and uses
// it as the offset value for runtime.tlsg.
//
// This runs at startup on g0 stack, but before g is set, so it must
// not split stack (transitively). g is expected to be nil, so things
// (e.g. asmcgocall) will skip saving or reading g.
//
//go:nosplit
func tlsinit(tlsg *uintptr, tlsbase *[_PTHREAD_KEYS_MAX]uintptr) {
	var k pthreadkey
	err := g0_pthread_key_create(&k, 0)
	if err != 0 {
		abort()
	}

	const magic = 0xc476c475c47957
	err = g0_pthread_setspecific(k, magic)
	if err != 0 {
		abort()
	}

	for i, x := range tlsbase {
		if x == magic {
			*tlsg = uintptr(i * sys.PtrSize)
			g0_pthread_setspecific(k, 0)
			return
		}
	}
	abort()
}
