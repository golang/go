// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const _PAGESIZE = 0x1000

type ureg struct {
	r0   uint32 /* general registers */
	r1   uint32 /* ... */
	r2   uint32 /* ... */
	r3   uint32 /* ... */
	r4   uint32 /* ... */
	r5   uint32 /* ... */
	r6   uint32 /* ... */
	r7   uint32 /* ... */
	r8   uint32 /* ... */
	r9   uint32 /* ... */
	r10  uint32 /* ... */
	r11  uint32 /* ... */
	r12  uint32 /* ... */
	sp   uint32
	link uint32 /* ... */
	trap uint32 /* trap type */
	psr  uint32
	pc   uint32 /* interrupted addr */
}

type sigctxt struct {
	u *ureg
}

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) pc() uintptr { return uintptr(c.u.pc) }

func (c *sigctxt) sp() uintptr { return uintptr(c.u.sp) }
func (c *sigctxt) lr() uintptr { return uintptr(c.u.link) }

func (c *sigctxt) setpc(x uintptr)  { c.u.pc = uint32(x) }
func (c *sigctxt) setsp(x uintptr)  { c.u.sp = uint32(x) }
func (c *sigctxt) setlr(x uintptr)  { c.u.link = uint32(x) }
func (c *sigctxt) savelr(x uintptr) { c.u.r0 = uint32(x) }

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
	print("sp    ", hex(u.sp), "\n")
	print("link  ", hex(u.link), "\n")
	print("pc    ", hex(u.pc), "\n")
	print("psr   ", hex(u.psr), "\n")
}

func sigpanictramp()
