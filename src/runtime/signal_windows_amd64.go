// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "internal/runtime/syscall/windows"

func prepareContextForSigResume(c *windows.Context) {
	c.R8 = c.Rsp
	c.R9 = c.Rip
}

func dumpregs(r *windows.Context) {
	print("rax     ", hex(r.Rax), "\n")
	print("rbx     ", hex(r.Rbx), "\n")
	print("rcx     ", hex(r.Rcx), "\n")
	print("rdx     ", hex(r.Rdx), "\n")
	print("rdi     ", hex(r.Rdi), "\n")
	print("rsi     ", hex(r.Rsi), "\n")
	print("rbp     ", hex(r.Rbp), "\n")
	print("rsp     ", hex(r.Rsp), "\n")
	print("r8      ", hex(r.R8), "\n")
	print("r9      ", hex(r.R9), "\n")
	print("r10     ", hex(r.R10), "\n")
	print("r11     ", hex(r.R11), "\n")
	print("r12     ", hex(r.R12), "\n")
	print("r13     ", hex(r.R13), "\n")
	print("r14     ", hex(r.R14), "\n")
	print("r15     ", hex(r.R15), "\n")
	print("rip     ", hex(r.Rip), "\n")
	print("rflags  ", hex(r.EFlags), "\n")
	print("cs      ", hex(r.SegCs), "\n")
	print("fs      ", hex(r.SegFs), "\n")
	print("gs      ", hex(r.SegGs), "\n")
}
