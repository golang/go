// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func lwp_mcontext_init(mc *mcontextt, stk unsafe.Pointer, mp *m, gp *g, fn uintptr) {
	// Machine dependent mcontext initialisation for LWP.
	mc.__gregs[_REG_R15] = uint32(funcPC(lwp_tramp))
	mc.__gregs[_REG_R13] = uint32(uintptr(stk))
	mc.__gregs[_REG_R0] = uint32(uintptr(unsafe.Pointer(mp)))
	mc.__gregs[_REG_R1] = uint32(uintptr(unsafe.Pointer(gp)))
	mc.__gregs[_REG_R2] = uint32(fn)
}

func checkgoarm() {
	// TODO(minux)
}

//go:nosplit
func cputicks() int64 {
	// Currently cputicks() is used in blocking profiler and to seed runtime·fastrand1().
	// runtime·nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	// TODO: need more entropy to better seed fastrand1.
	return nanotime()
}
