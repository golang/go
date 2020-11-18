// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func lwp_mcontext_init(mc *mcontextt, stk unsafe.Pointer, mp *m, gp *g, fn uintptr) {
	// Machine dependent mcontext initialisation for LWP.
	mc.__gregs[_REG_EIP] = uint32(funcPC(lwp_tramp))
	mc.__gregs[_REG_UESP] = uint32(uintptr(stk))
	mc.__gregs[_REG_EBX] = uint32(uintptr(unsafe.Pointer(mp)))
	mc.__gregs[_REG_EDX] = uint32(uintptr(unsafe.Pointer(gp)))
	mc.__gregs[_REG_ESI] = uint32(fn)
}
