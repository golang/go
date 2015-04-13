// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/arm"
)

/*
 * generate array index into res.
 * n might be any size; res is 32-bit.
 * returns Prog* to patch to panic call.
 */
func cgenindex(n *gc.Node, res *gc.Node, bounded bool) *obj.Prog {
	if !gc.Is64(n.Type) {
		gc.Cgen(n, res)
		return nil
	}

	var tmp gc.Node
	gc.Tempname(&tmp, gc.Types[gc.TINT64])
	gc.Cgen(n, &tmp)
	var lo gc.Node
	var hi gc.Node
	split64(&tmp, &lo, &hi)
	gmove(&lo, res)
	if bounded {
		splitclean()
		return nil
	}

	var n1 gc.Node
	gc.Regalloc(&n1, gc.Types[gc.TINT32], nil)
	var n2 gc.Node
	gc.Regalloc(&n2, gc.Types[gc.TINT32], nil)
	var zero gc.Node
	gc.Nodconst(&zero, gc.Types[gc.TINT32], 0)
	gmove(&hi, &n1)
	gmove(&zero, &n2)
	gins(arm.ACMP, &n1, &n2)
	gc.Regfree(&n2)
	gc.Regfree(&n1)
	splitclean()
	return gc.Gbranch(arm.ABNE, nil, -1)
}

func igenindex(n *gc.Node, res *gc.Node, bounded bool) *obj.Prog {
	gc.Tempname(res, n.Type)
	return cgenindex(n, res, bounded)
}

func gencmp0(n *gc.Node, t *gc.Type, o int, likely int, to *obj.Prog) {
	var n1 gc.Node

	gc.Regalloc(&n1, t, nil)
	gc.Cgen(n, &n1)
	a := optoas(gc.OCMP, t)
	if a != arm.ACMP {
		var n2 gc.Node
		gc.Nodconst(&n2, t, 0)
		var n3 gc.Node
		gc.Regalloc(&n3, t, nil)
		gmove(&n2, &n3)
		gins(a, &n1, &n3)
		gc.Regfree(&n3)
	} else {
		gins(arm.ATST, &n1, nil)
	}
	a = optoas(o, t)
	gc.Patch(gc.Gbranch(a, t, likely), to)
	gc.Regfree(&n1)
}

func stackcopy(n, res *gc.Node, osrc, odst, w int64) {
	// determine alignment.
	// want to avoid unaligned access, so have to use
	// smaller operations for less aligned types.
	// for example moving [4]byte must use 4 MOVB not 1 MOVW.
	align := int(n.Type.Align)

	var op int
	switch align {
	default:
		gc.Fatal("sgen: invalid alignment %d for %v", align, gc.Tconv(n.Type, 0))

	case 1:
		op = arm.AMOVB

	case 2:
		op = arm.AMOVH

	case 4:
		op = arm.AMOVW
	}

	if w%int64(align) != 0 {
		gc.Fatal("sgen: unaligned size %d (align=%d) for %v", w, align, gc.Tconv(n.Type, 0))
	}
	c := int32(w / int64(align))

	if osrc%int64(align) != 0 || odst%int64(align) != 0 {
		gc.Fatal("sgen: unaligned offset src %d or dst %d (align %d)", osrc, odst, align)
	}

	// if we are copying forward on the stack and
	// the src and dst overlap, then reverse direction
	dir := align
	if osrc < odst && int64(odst) < int64(osrc)+w {
		dir = -dir
	}

	if op == arm.AMOVW && !gc.Nacl && dir > 0 && c >= 4 && c <= 128 {
		var r0 gc.Node
		r0.Op = gc.OREGISTER
		r0.Reg = arm.REG_R0
		var r1 gc.Node
		r1.Op = gc.OREGISTER
		r1.Reg = arm.REG_R0 + 1
		var r2 gc.Node
		r2.Op = gc.OREGISTER
		r2.Reg = arm.REG_R0 + 2

		var src gc.Node
		gc.Regalloc(&src, gc.Types[gc.Tptr], &r1)
		var dst gc.Node
		gc.Regalloc(&dst, gc.Types[gc.Tptr], &r2)
		if n.Ullman >= res.Ullman {
			// eval n first
			gc.Agen(n, &src)

			if res.Op == gc.ONAME {
				gc.Gvardef(res)
			}
			gc.Agen(res, &dst)
		} else {
			// eval res first
			if res.Op == gc.ONAME {
				gc.Gvardef(res)
			}
			gc.Agen(res, &dst)
			gc.Agen(n, &src)
		}

		var tmp gc.Node
		gc.Regalloc(&tmp, gc.Types[gc.Tptr], &r0)
		f := gc.Sysfunc("duffcopy")
		p := gins(obj.ADUFFCOPY, nil, f)
		gc.Afunclit(&p.To, f)

		// 8 and 128 = magic constants: see ../../runtime/asm_arm.s
		p.To.Offset = 8 * (128 - int64(c))

		gc.Regfree(&tmp)
		gc.Regfree(&src)
		gc.Regfree(&dst)
		return
	}

	var dst gc.Node
	var src gc.Node
	if n.Ullman >= res.Ullman {
		gc.Agenr(n, &dst, res) // temporarily use dst
		gc.Regalloc(&src, gc.Types[gc.Tptr], nil)
		gins(arm.AMOVW, &dst, &src)
		if res.Op == gc.ONAME {
			gc.Gvardef(res)
		}
		gc.Agen(res, &dst)
	} else {
		if res.Op == gc.ONAME {
			gc.Gvardef(res)
		}
		gc.Agenr(res, &dst, res)
		gc.Agenr(n, &src, nil)
	}

	var tmp gc.Node
	gc.Regalloc(&tmp, gc.Types[gc.TUINT32], nil)

	// set up end marker
	var nend gc.Node

	if c >= 4 {
		gc.Regalloc(&nend, gc.Types[gc.TUINT32], nil)

		p := gins(arm.AMOVW, &src, &nend)
		p.From.Type = obj.TYPE_ADDR
		if dir < 0 {
			p.From.Offset = int64(dir)
		} else {
			p.From.Offset = w
		}
	}

	// move src and dest to the end of block if necessary
	if dir < 0 {
		p := gins(arm.AMOVW, &src, &src)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = w + int64(dir)

		p = gins(arm.AMOVW, &dst, &dst)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = w + int64(dir)
	}

	// move
	if c >= 4 {
		p := gins(op, &src, &tmp)
		p.From.Type = obj.TYPE_MEM
		p.From.Offset = int64(dir)
		p.Scond |= arm.C_PBIT
		ploop := p

		p = gins(op, &tmp, &dst)
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = int64(dir)
		p.Scond |= arm.C_PBIT

		p = gins(arm.ACMP, &src, nil)
		raddr(&nend, p)

		gc.Patch(gc.Gbranch(arm.ABNE, nil, 0), ploop)
		gc.Regfree(&nend)
	} else {
		var p *obj.Prog
		for {
			tmp14 := c
			c--
			if tmp14 <= 0 {
				break
			}
			p = gins(op, &src, &tmp)
			p.From.Type = obj.TYPE_MEM
			p.From.Offset = int64(dir)
			p.Scond |= arm.C_PBIT

			p = gins(op, &tmp, &dst)
			p.To.Type = obj.TYPE_MEM
			p.To.Offset = int64(dir)
			p.Scond |= arm.C_PBIT
		}
	}

	gc.Regfree(&dst)
	gc.Regfree(&src)
	gc.Regfree(&tmp)
}
