// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package amd64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
)

// no floating point in note handlers on Plan 9
var isPlan9 = obj.GOOS == "plan9"

func defframe(ptxt *obj.Prog) {
	// fill in argument size, stack size
	ptxt.To.Type = obj.TYPE_TEXTSIZE

	ptxt.To.Val = int32(gc.Rnd(gc.Curfn.Type.ArgWidth(), int64(gc.Widthptr)))
	frame := uint32(gc.Rnd(gc.Stksize+gc.Maxarg, int64(gc.Widthreg)))
	ptxt.To.Offset = int64(frame)

	// insert code to zero ambiguously live variables
	// so that the garbage collector only sees initialized values
	// when it looks for pointers.
	p := ptxt

	hi := int64(0)
	lo := hi
	ax := uint32(0)
	x0 := uint32(0)

	// iterate through declarations - they are sorted in decreasing xoffset order.
	for _, n := range gc.Curfn.Func.Dcl {
		if !n.Name.Needzero {
			continue
		}
		if n.Class != gc.PAUTO {
			gc.Fatalf("needzero class %d", n.Class)
		}
		if n.Type.Width%int64(gc.Widthptr) != 0 || n.Xoffset%int64(gc.Widthptr) != 0 || n.Type.Width == 0 {
			gc.Fatalf("var %L has size %d offset %d", n, int(n.Type.Width), int(n.Xoffset))
		}

		if lo != hi && n.Xoffset+n.Type.Width >= lo-int64(2*gc.Widthreg) {
			// merge with range we already have
			lo = n.Xoffset

			continue
		}

		// zero old range
		p = zerorange(p, int64(frame), lo, hi, &ax, &x0)

		// set new range
		hi = n.Xoffset + n.Type.Width

		lo = n.Xoffset
	}

	// zero final range
	zerorange(p, int64(frame), lo, hi, &ax, &x0)
}

// DUFFZERO consists of repeated blocks of 4 MOVUPSs + ADD,
// See runtime/mkduff.go.
const (
	dzBlocks    = 16 // number of MOV/ADD blocks
	dzBlockLen  = 4  // number of clears per block
	dzBlockSize = 19 // size of instructions in a single block
	dzMovSize   = 4  // size of single MOV instruction w/ offset
	dzAddSize   = 4  // size of single ADD instruction
	dzClearStep = 16 // number of bytes cleared by each MOV instruction

	dzClearLen = dzClearStep * dzBlockLen // bytes cleared by one block
	dzSize     = dzBlocks * dzBlockSize
)

// dzOff returns the offset for a jump into DUFFZERO.
// b is the number of bytes to zero.
func dzOff(b int64) int64 {
	off := int64(dzSize)
	off -= b / dzClearLen * dzBlockSize
	tailLen := b % dzClearLen
	if tailLen >= dzClearStep {
		off -= dzAddSize + dzMovSize*(tailLen/dzClearStep)
	}
	return off
}

// duffzeroDI returns the pre-adjustment to DI for a call to DUFFZERO.
// b is the number of bytes to zero.
func dzDI(b int64) int64 {
	tailLen := b % dzClearLen
	if tailLen < dzClearStep {
		return 0
	}
	tailSteps := tailLen / dzClearStep
	return -dzClearStep * (dzBlockLen - tailSteps)
}

func zerorange(p *obj.Prog, frame int64, lo int64, hi int64, ax *uint32, x0 *uint32) *obj.Prog {
	cnt := hi - lo
	if cnt == 0 {
		return p
	}

	if cnt%int64(gc.Widthreg) != 0 {
		// should only happen with nacl
		if cnt%int64(gc.Widthptr) != 0 {
			gc.Fatalf("zerorange count not a multiple of widthptr %d", cnt)
		}
		if *ax == 0 {
			p = gc.Appendpp(p, x86.AMOVQ, obj.TYPE_CONST, 0, 0, obj.TYPE_REG, x86.REG_AX, 0)
			*ax = 1
		}
		p = gc.Appendpp(p, x86.AMOVL, obj.TYPE_REG, x86.REG_AX, 0, obj.TYPE_MEM, x86.REG_SP, frame+lo)
		lo += int64(gc.Widthptr)
		cnt -= int64(gc.Widthptr)
	}

	if cnt == 8 {
		if *ax == 0 {
			p = gc.Appendpp(p, x86.AMOVQ, obj.TYPE_CONST, 0, 0, obj.TYPE_REG, x86.REG_AX, 0)
			*ax = 1
		}
		p = gc.Appendpp(p, x86.AMOVQ, obj.TYPE_REG, x86.REG_AX, 0, obj.TYPE_MEM, x86.REG_SP, frame+lo)
	} else if !isPlan9 && cnt <= int64(8*gc.Widthreg) {
		if *x0 == 0 {
			p = gc.Appendpp(p, x86.AXORPS, obj.TYPE_REG, x86.REG_X0, 0, obj.TYPE_REG, x86.REG_X0, 0)
			*x0 = 1
		}

		for i := int64(0); i < cnt/16; i++ {
			p = gc.Appendpp(p, x86.AMOVUPS, obj.TYPE_REG, x86.REG_X0, 0, obj.TYPE_MEM, x86.REG_SP, frame+lo+i*16)
		}

		if cnt%16 != 0 {
			p = gc.Appendpp(p, x86.AMOVUPS, obj.TYPE_REG, x86.REG_X0, 0, obj.TYPE_MEM, x86.REG_SP, frame+lo+cnt-int64(16))
		}
	} else if !gc.Nacl && !isPlan9 && (cnt <= int64(128*gc.Widthreg)) {
		if *x0 == 0 {
			p = gc.Appendpp(p, x86.AXORPS, obj.TYPE_REG, x86.REG_X0, 0, obj.TYPE_REG, x86.REG_X0, 0)
			*x0 = 1
		}
		p = gc.Appendpp(p, leaptr, obj.TYPE_MEM, x86.REG_SP, frame+lo+dzDI(cnt), obj.TYPE_REG, x86.REG_DI, 0)
		p = gc.Appendpp(p, obj.ADUFFZERO, obj.TYPE_NONE, 0, 0, obj.TYPE_ADDR, 0, dzOff(cnt))
		p.To.Sym = gc.Linksym(gc.Pkglookup("duffzero", gc.Runtimepkg))

		if cnt%16 != 0 {
			p = gc.Appendpp(p, x86.AMOVUPS, obj.TYPE_REG, x86.REG_X0, 0, obj.TYPE_MEM, x86.REG_DI, -int64(8))
		}
	} else {
		if *ax == 0 {
			p = gc.Appendpp(p, x86.AMOVQ, obj.TYPE_CONST, 0, 0, obj.TYPE_REG, x86.REG_AX, 0)
			*ax = 1
		}

		p = gc.Appendpp(p, x86.AMOVQ, obj.TYPE_CONST, 0, cnt/int64(gc.Widthreg), obj.TYPE_REG, x86.REG_CX, 0)
		p = gc.Appendpp(p, leaptr, obj.TYPE_MEM, x86.REG_SP, frame+lo, obj.TYPE_REG, x86.REG_DI, 0)
		p = gc.Appendpp(p, x86.AREP, obj.TYPE_NONE, 0, 0, obj.TYPE_NONE, 0, 0)
		p = gc.Appendpp(p, x86.ASTOSQ, obj.TYPE_NONE, 0, 0, obj.TYPE_NONE, 0, 0)
	}

	return p
}

func ginsnop() {
	// This is actually not the x86 NOP anymore,
	// but at the point where it gets used, AX is dead
	// so it's okay if we lose the high bits.
	p := gc.Prog(x86.AXCHGL)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = x86.REG_AX
	p.To.Type = obj.TYPE_REG
	p.To.Reg = x86.REG_AX
}
