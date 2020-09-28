// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/cpu"
	"unsafe"
)

func lwp_mcontext_init(mc *mcontextt, stk unsafe.Pointer, mp *m, gp *g, fn uintptr) {
	// Machine dependent mcontext initialisation for LWP.
	mc.__gregs[_REG_ELR] = uint64(funcPC(lwp_tramp))
	mc.__gregs[_REG_X31] = uint64(uintptr(stk))
	mc.__gregs[_REG_X0] = uint64(uintptr(unsafe.Pointer(mp)))
	mc.__gregs[_REG_X1] = uint64(uintptr(unsafe.Pointer(mp.g0)))
	mc.__gregs[_REG_X2] = uint64(fn)
}

//go:nosplit
func cputicks() int64 {
	// Currently cputicks() is used in blocking profiler and to seed runtime·fastrand().
	// runtime·nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	return nanotime()
}

func archauxv(auxv []uintptr) {
	// NetBSD does not supply AT_HWCAP, however we still need to initialise cpu.HWCaps.
	// For now specify the bare minimum until we add some form of capabilities
	// detection. See issue https://golang.org/issue/30824#issuecomment-494901591
	cpu.HWCap = 1<<1 | 1<<0 // ASIMD, FP
}
