// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const _PAGESHIFT = 16
const _PAGESIZE = 1 << _PAGESHIFT

type ureg struct {
	/* AArch64 registers */
	r0  uint64 /* general registers */
	r1  uint64 /* ... */
	r2  uint64 /* ... */
	r3  uint64 /* ... */
	r4  uint64 /* ... */
	r5  uint64 /* ... */
	r6  uint64 /* ... */
	r7  uint64 /* ... */
	r8  uint64 /* ... */
	r9  uint64 /* ... */
	r10 uint64 /* ... */
	r11 uint64 /* ... */
	r12 uint64 /* ... */
	r13 uint64 /* ... */
	r14 uint64 /* ... */
	r15 uint64 /* ... */
	r16 uint64 /* ... */
	r17 uint64 /* ... */
	r18 uint64 /* ... */
	r19 uint64 /* ... */
	r20 uint64 /* ... */
	r21 uint64 /* ... */
	r22 uint64 /* ... */
	r23 uint64 /* ... */
	r24 uint64 /* ... */
	r25 uint64 /* ... */
	r26 uint64 /* ... */
	r27 uint64 /* ... */
	r28 uint64 /* ... */
	r29 uint64 /* ... */
	r30 uint64 /* link (lr) */
	sp  uint64
	pc  uint64 /* interrupted addr */
	psr uint64
	typ uint64 /* of exception */
}

type sigctxt struct {
	u *ureg
}

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) pc() uintptr { return uintptr(c.u.pc) }

func (c *sigctxt) sp() uintptr { return uintptr(c.u.sp) }
func (c *sigctxt) lr() uintptr { return uintptr(c.u.r30) }

func (c *sigctxt) setpc(x uintptr)  { c.u.pc = uint64(x) }
func (c *sigctxt) setsp(x uintptr)  { c.u.sp = uint64(x) }
func (c *sigctxt) setlr(x uintptr)  { c.u.r30 = uint64(x) }
func (c *sigctxt) savelr(x uintptr) { c.u.r0 = uint64(x) }

func dumpregs(u *ureg) {
	print("r0    ", hex(u.r0), "\n")
	print("r1    ", hex(u.r1), "\n")
	print("r2    ", hex(u.r2), "\n")
	print("r3    ", hex(u.r3), "\n")
	print("r4    ", hex(u.r4), "\n")
	print("r5    ", hex(u.r5), "\n")
	print("r6    ", hex(u.r6), "\n")
	print("r7    ", hex(u.r7), "\n")
	print("r8    ", hex(u.r8), "\n")
	print("r9    ", hex(u.r9), "\n")
	print("r10   ", hex(u.r10), "\n")
	print("r11   ", hex(u.r11), "\n")
	print("r12   ", hex(u.r12), "\n")
	print("r13   ", hex(u.r13), "\n")
	print("r14   ", hex(u.r14), "\n")
	print("r15   ", hex(u.r15), "\n")
	print("r16   ", hex(u.r16), "\n")
	print("r17   ", hex(u.r17), "\n")
	print("r18   ", hex(u.r18), "\n")
	print("r19   ", hex(u.r19), "\n")
	print("r20   ", hex(u.r20), "\n")
	print("r21   ", hex(u.r21), "\n")
	print("r22   ", hex(u.r22), "\n")
	print("r23   ", hex(u.r23), "\n")
	print("r24   ", hex(u.r24), "\n")
	print("r25   ", hex(u.r25), "\n")
	print("r26   ", hex(u.r26), "\n")
	print("r27   ", hex(u.r27), "\n")
	print("r28   ", hex(u.r28), "\n")
	print("r29   ", hex(u.r29), "\n")
	print("r30   ", hex(u.r30), "\n")
	print("sp    ", hex(u.sp), "\n")
	print("pc    ", hex(u.pc), "\n")
	print("psr   ", hex(u.psr), "\n")
	print("type  ", hex(u.typ), "\n")
}

func sigpanictramp()
