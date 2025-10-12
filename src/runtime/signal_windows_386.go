// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "internal/runtime/syscall/windows"

func prepareContextForSigResume(c *windows.Context) {
	c.Edx = c.Esp
	c.Ecx = c.Eip
}

func dumpregs(r *windows.Context) {
	print("eax     ", hex(r.Eax), "\n")
	print("ebx     ", hex(r.Ebx), "\n")
	print("ecx     ", hex(r.Ecx), "\n")
	print("edx     ", hex(r.Edx), "\n")
	print("edi     ", hex(r.Edi), "\n")
	print("esi     ", hex(r.Esi), "\n")
	print("ebp     ", hex(r.Ebp), "\n")
	print("esp     ", hex(r.Esp), "\n")
	print("eip     ", hex(r.Eip), "\n")
	print("eflags  ", hex(r.EFlags), "\n")
	print("cs      ", hex(r.SegCs), "\n")
	print("fs      ", hex(r.SegFs), "\n")
	print("gs      ", hex(r.SegGs), "\n")
}
