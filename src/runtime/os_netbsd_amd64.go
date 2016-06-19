// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func lwp_mcontext_init(mc *mcontextt, stk unsafe.Pointer, mp *m, gp *g, fn uintptr) {
	// Machine dependent mcontext initialisation for LWP.
	mc.__gregs[_REG_RIP] = uint64(funcPC(lwp_tramp))
	mc.__gregs[_REG_RSP] = uint64(uintptr(stk))
	mc.__gregs[_REG_R8] = uint64(uintptr(unsafe.Pointer(mp)))
	mc.__gregs[_REG_R9] = uint64(uintptr(unsafe.Pointer(gp)))
	mc.__gregs[_REG_R12] = uint64(fn)
}
