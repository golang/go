// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "internal/runtime/syscall/windows"

func prepareContextForSigResume(c *windows.Context) {
	c.X[0] = c.XSp
	c.X[1] = c.Pc
}

func dumpregs(r *windows.Context) {
	print("r0   ", hex(r.X[0]), "\n")
	print("r1   ", hex(r.X[1]), "\n")
	print("r2   ", hex(r.X[2]), "\n")
	print("r3   ", hex(r.X[3]), "\n")
	print("r4   ", hex(r.X[4]), "\n")
	print("r5   ", hex(r.X[5]), "\n")
	print("r6   ", hex(r.X[6]), "\n")
	print("r7   ", hex(r.X[7]), "\n")
	print("r8   ", hex(r.X[8]), "\n")
	print("r9   ", hex(r.X[9]), "\n")
	print("r10  ", hex(r.X[10]), "\n")
	print("r11  ", hex(r.X[11]), "\n")
	print("r12  ", hex(r.X[12]), "\n")
	print("r13  ", hex(r.X[13]), "\n")
	print("r14  ", hex(r.X[14]), "\n")
	print("r15  ", hex(r.X[15]), "\n")
	print("r16  ", hex(r.X[16]), "\n")
	print("r17  ", hex(r.X[17]), "\n")
	print("r18  ", hex(r.X[18]), "\n")
	print("r19  ", hex(r.X[19]), "\n")
	print("r20  ", hex(r.X[20]), "\n")
	print("r21  ", hex(r.X[21]), "\n")
	print("r22  ", hex(r.X[22]), "\n")
	print("r23  ", hex(r.X[23]), "\n")
	print("r24  ", hex(r.X[24]), "\n")
	print("r25  ", hex(r.X[25]), "\n")
	print("r26  ", hex(r.X[26]), "\n")
	print("r27  ", hex(r.X[27]), "\n")
	print("r28  ", hex(r.X[28]), "\n")
	print("r29  ", hex(r.X[29]), "\n")
	print("lr   ", hex(r.X[30]), "\n")
	print("sp   ", hex(r.XSp), "\n")
	print("pc   ", hex(r.Pc), "\n")
	print("cpsr ", hex(r.Cpsr), "\n")
}
