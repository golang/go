// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"unsafe"
)

func lwp_mcontext_init(mc *mcontextt, stk unsafe.Pointer, mp *m, gp *g, fn uintptr) {
	// Machine dependent mcontext initialisation for LWP.
	mc.__gregs[_REG_R15] = uint32(abi.FuncPCABI0(lwp_tramp))
	mc.__gregs[_REG_R13] = uint32(uintptr(stk))
	mc.__gregs[_REG_R0] = uint32(uintptr(unsafe.Pointer(mp)))
	mc.__gregs[_REG_R1] = uint32(uintptr(unsafe.Pointer(gp)))
	mc.__gregs[_REG_R2] = uint32(fn)
}

func checkgoarm() {
	// TODO(minux): FP checks like in os_linux_arm.go.

	// osinit not called yet, so ncpu not set: must use getncpu directly.
	if getncpu() > 1 && goarm < 7 {
		print("runtime: this system has multiple CPUs and must use\n")
		print("atomic synchronization instructions. Recompile using GOARM=7.\n")
		exit(1)
	}
}

//go:nosplit
func cputicks() int64 {
	// Currently cputicks() is used in blocking profiler and to seed runtime·fastrand().
	// runtime·nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	return nanotime()
}
