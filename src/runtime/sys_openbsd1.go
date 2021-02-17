// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build openbsd,amd64 openbsd,arm64

package runtime

import "unsafe"

//go:nosplit
//go:cgo_unsafe_args
func thrsleep(ident uintptr, clock_id int32, tsp *timespec, lock uintptr, abort *uint32) int32 {
	return libcCall(unsafe.Pointer(funcPC(thrsleep_trampoline)), unsafe.Pointer(&ident))
}
func thrsleep_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func thrwakeup(ident uintptr, n int32) int32 {
	return libcCall(unsafe.Pointer(funcPC(thrwakeup_trampoline)), unsafe.Pointer(&ident))
}
func thrwakeup_trampoline()

func osyield() {
	libcCall(unsafe.Pointer(funcPC(sched_yield_trampoline)), unsafe.Pointer(nil))
}
func sched_yield_trampoline()

//go:cgo_import_dynamic libc_thrsleep __thrsleep "libc.so"
//go:cgo_import_dynamic libc_thrwakeup __thrwakeup "libc.so"
//go:cgo_import_dynamic libc_sched_yield sched_yield "libc.so"

//go:cgo_import_dynamic _ _ "libc.so"
