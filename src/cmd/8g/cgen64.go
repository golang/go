// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/obj"
	"cmd/internal/obj/i386"
)
import "cmd/internal/gc"

/*
 * attempt to generate 64-bit
 *	res = n
 * return 1 on success, 0 if op not handled.
 */
func cgen64(n *gc.Node, res *gc.Node) {
	if res.Op != gc.OINDREG && res.Op != gc.ONAME {
		gc.Dump("n", n)
		gc.Dump("res", res)
		gc.Fatal("cgen64 %v of %v", gc.Oconv(int(n.Op), 0), gc.Oconv(int(res.Op), 0))
	}

	switch n.Op {
	default:
		gc.Fatal("cgen64 %v", gc.Oconv(int(n.Op), 0))

	case gc.OMINUS:
		cgen(n.Left, res)
		var hi1 gc.Node
		var lo1 gc.Node
		split64(res, &lo1, &hi1)
		gins(i386.ANEGL, nil, &lo1)
		gins(i386.AADCL, ncon(0), &hi1)
		gins(i386.ANEGL, nil, &hi1)
		splitclean()
		return

	case gc.OCOM:
		cgen(n.Left, res)
		var lo1 gc.Node
		var hi1 gc.Node
		split64(res, &lo1, &hi1)
		gins(i386.ANOTL, nil, &lo1)
		gins(i386.ANOTL, nil, &hi1)
		splitclean()
		return

		// binary operators.
	// common setup below.
	case gc.OADD,
		gc.OSUB,
		gc.OMUL,
		gc.OLROT,
		gc.OLSH,
		gc.ORSH,
		gc.OAND,
		gc.OOR,
		gc.OXOR:
		break
	}

	l := n.Left
	r := n.Right
	if l.Addable == 0 {
		var t1 gc.Node
		gc.Tempname(&t1, l.Type)
		cgen(l, &t1)
		l = &t1
	}

	if r != nil && r.Addable == 0 {
		var t2 gc.Node
		gc.Tempname(&t2, r.Type)
		cgen(r, &t2)
		r = &t2
	}

	var ax gc.Node
	gc.Nodreg(&ax, gc.Types[gc.TINT32], i386.REG_AX)
	var cx gc.Node
	gc.Nodreg(&cx, gc.Types[gc.TINT32], i386.REG_CX)
	var dx gc.Node
	gc.Nodreg(&dx, gc.Types[gc.TINT32], i386.REG_DX)

	// Setup for binary operation.
	var hi1 gc.Node
	var lo1 gc.Node
	split64(l, &lo1, &hi1)

	var lo2 gc.Node
	var hi2 gc.Node
	if gc.Is64(r.Type) {
		split64(r, &lo2, &hi2)
	}

	// Do op.  Leave result in DX:AX.
	switch n.Op {
	// TODO: Constants
	case gc.OADD:
		gins(i386.AMOVL, &lo1, &ax)

		gins(i386.AMOVL, &hi1, &dx)
		gins(i386.AADDL, &lo2, &ax)
		gins(i386.AADCL, &hi2, &dx)

		// TODO: Constants.
	case gc.OSUB:
		gins(i386.AMOVL, &lo1, &ax)

		gins(i386.AMOVL, &hi1, &dx)
		gins(i386.ASUBL, &lo2, &ax)
		gins(i386.ASBBL, &hi2, &dx)

		// let's call the next two EX and FX.
	case gc.OMUL:
		var ex gc.Node
		regalloc(&ex, gc.Types[gc.TPTR32], nil)

		var fx gc.Node
		regalloc(&fx, gc.Types[gc.TPTR32], nil)

		// load args into DX:AX and EX:CX.
		gins(i386.AMOVL, &lo1, &ax)

		gins(i386.AMOVL, &hi1, &dx)
		gins(i386.AMOVL, &lo2, &cx)
		gins(i386.AMOVL, &hi2, &ex)

		// if DX and EX are zero, use 32 x 32 -> 64 unsigned multiply.
		gins(i386.AMOVL, &dx, &fx)

		gins(i386.AORL, &ex, &fx)
		p1 := gc.Gbranch(i386.AJNE, nil, 0)
		gins(i386.AMULL, &cx, nil) // implicit &ax
		p2 := gc.Gbranch(obj.AJMP, nil, 0)
		gc.Patch(p1, gc.Pc)

		// full 64x64 -> 64, from 32x32 -> 64.
		gins(i386.AIMULL, &cx, &dx)

		gins(i386.AMOVL, &ax, &fx)
		gins(i386.AIMULL, &ex, &fx)
		gins(i386.AADDL, &dx, &fx)
		gins(i386.AMOVL, &cx, &dx)
		gins(i386.AMULL, &dx, nil) // implicit &ax
		gins(i386.AADDL, &fx, &dx)
		gc.Patch(p2, gc.Pc)

		regfree(&ex)
		regfree(&fx)

		// We only rotate by a constant c in [0,64).
	// if c >= 32:
	//	lo, hi = hi, lo
	//	c -= 32
	// if c == 0:
	//	no-op
	// else:
	//	t = hi
	//	shld hi:lo, c
	//	shld lo:t, c
	case gc.OLROT:
		v := uint64(gc.Mpgetfix(r.Val.U.Xval))

		if v >= 32 {
			// reverse during load to do the first 32 bits of rotate
			v -= 32

			gins(i386.AMOVL, &lo1, &dx)
			gins(i386.AMOVL, &hi1, &ax)
		} else {
			gins(i386.AMOVL, &lo1, &ax)
			gins(i386.AMOVL, &hi1, &dx)
		}

		if v == 0 {
		} else // done
		{
			gins(i386.AMOVL, &dx, &cx)
			p1 := gins(i386.ASHLL, ncon(uint32(v)), &dx)
			p1.From.Index = i386.REG_AX // double-width shift
			p1.From.Scale = 0
			p1 = gins(i386.ASHLL, ncon(uint32(v)), &ax)
			p1.From.Index = i386.REG_CX // double-width shift
			p1.From.Scale = 0
		}

	case gc.OLSH:
		if r.Op == gc.OLITERAL {
			v := uint64(gc.Mpgetfix(r.Val.U.Xval))
			if v >= 64 {
				if gc.Is64(r.Type) {
					splitclean()
				}
				splitclean()
				split64(res, &lo2, &hi2)
				gins(i386.AMOVL, ncon(0), &lo2)
				gins(i386.AMOVL, ncon(0), &hi2)
				splitclean()
				goto out
			}

			if v >= 32 {
				if gc.Is64(r.Type) {
					splitclean()
				}
				split64(res, &lo2, &hi2)
				gmove(&lo1, &hi2)
				if v > 32 {
					gins(i386.ASHLL, ncon(uint32(v-32)), &hi2)
				}

				gins(i386.AMOVL, ncon(0), &lo2)
				splitclean()
				splitclean()
				goto out
			}

			// general shift
			gins(i386.AMOVL, &lo1, &ax)

			gins(i386.AMOVL, &hi1, &dx)
			p1 := gins(i386.ASHLL, ncon(uint32(v)), &dx)
			p1.From.Index = i386.REG_AX // double-width shift
			p1.From.Scale = 0
			gins(i386.ASHLL, ncon(uint32(v)), &ax)
			break
		}

		// load value into DX:AX.
		gins(i386.AMOVL, &lo1, &ax)

		gins(i386.AMOVL, &hi1, &dx)

		// load shift value into register.
		// if high bits are set, zero value.
		p1 := (*obj.Prog)(nil)

		if gc.Is64(r.Type) {
			gins(i386.ACMPL, &hi2, ncon(0))
			p1 = gc.Gbranch(i386.AJNE, nil, +1)
			gins(i386.AMOVL, &lo2, &cx)
		} else {
			cx.Type = gc.Types[gc.TUINT32]
			gmove(r, &cx)
		}

		// if shift count is >=64, zero value
		gins(i386.ACMPL, &cx, ncon(64))

		p2 := gc.Gbranch(optoas(gc.OLT, gc.Types[gc.TUINT32]), nil, +1)
		if p1 != nil {
			gc.Patch(p1, gc.Pc)
		}
		gins(i386.AXORL, &dx, &dx)
		gins(i386.AXORL, &ax, &ax)
		gc.Patch(p2, gc.Pc)

		// if shift count is >= 32, zero low.
		gins(i386.ACMPL, &cx, ncon(32))

		p1 = gc.Gbranch(optoas(gc.OLT, gc.Types[gc.TUINT32]), nil, +1)
		gins(i386.AMOVL, &ax, &dx)
		gins(i386.ASHLL, &cx, &dx) // SHLL only uses bottom 5 bits of count
		gins(i386.AXORL, &ax, &ax)
		p2 = gc.Gbranch(obj.AJMP, nil, 0)
		gc.Patch(p1, gc.Pc)

		// general shift
		p1 = gins(i386.ASHLL, &cx, &dx)

		p1.From.Index = i386.REG_AX // double-width shift
		p1.From.Scale = 0
		gins(i386.ASHLL, &cx, &ax)
		gc.Patch(p2, gc.Pc)

	case gc.ORSH:
		if r.Op == gc.OLITERAL {
			v := uint64(gc.Mpgetfix(r.Val.U.Xval))
			if v >= 64 {
				if gc.Is64(r.Type) {
					splitclean()
				}
				splitclean()
				split64(res, &lo2, &hi2)
				if hi1.Type.Etype == gc.TINT32 {
					gmove(&hi1, &lo2)
					gins(i386.ASARL, ncon(31), &lo2)
					gmove(&hi1, &hi2)
					gins(i386.ASARL, ncon(31), &hi2)
				} else {
					gins(i386.AMOVL, ncon(0), &lo2)
					gins(i386.AMOVL, ncon(0), &hi2)
				}

				splitclean()
				goto out
			}

			if v >= 32 {
				if gc.Is64(r.Type) {
					splitclean()
				}
				split64(res, &lo2, &hi2)
				gmove(&hi1, &lo2)
				if v > 32 {
					gins(optoas(gc.ORSH, hi1.Type), ncon(uint32(v-32)), &lo2)
				}
				if hi1.Type.Etype == gc.TINT32 {
					gmove(&hi1, &hi2)
					gins(i386.ASARL, ncon(31), &hi2)
				} else {
					gins(i386.AMOVL, ncon(0), &hi2)
				}
				splitclean()
				splitclean()
				goto out
			}

			// general shift
			gins(i386.AMOVL, &lo1, &ax)

			gins(i386.AMOVL, &hi1, &dx)
			p1 := gins(i386.ASHRL, ncon(uint32(v)), &ax)
			p1.From.Index = i386.REG_DX // double-width shift
			p1.From.Scale = 0
			gins(optoas(gc.ORSH, hi1.Type), ncon(uint32(v)), &dx)
			break
		}

		// load value into DX:AX.
		gins(i386.AMOVL, &lo1, &ax)

		gins(i386.AMOVL, &hi1, &dx)

		// load shift value into register.
		// if high bits are set, zero value.
		p1 := (*obj.Prog)(nil)

		if gc.Is64(r.Type) {
			gins(i386.ACMPL, &hi2, ncon(0))
			p1 = gc.Gbranch(i386.AJNE, nil, +1)
			gins(i386.AMOVL, &lo2, &cx)
		} else {
			cx.Type = gc.Types[gc.TUINT32]
			gmove(r, &cx)
		}

		// if shift count is >=64, zero or sign-extend value
		gins(i386.ACMPL, &cx, ncon(64))

		p2 := gc.Gbranch(optoas(gc.OLT, gc.Types[gc.TUINT32]), nil, +1)
		if p1 != nil {
			gc.Patch(p1, gc.Pc)
		}
		if hi1.Type.Etype == gc.TINT32 {
			gins(i386.ASARL, ncon(31), &dx)
			gins(i386.AMOVL, &dx, &ax)
		} else {
			gins(i386.AXORL, &dx, &dx)
			gins(i386.AXORL, &ax, &ax)
		}

		gc.Patch(p2, gc.Pc)

		// if shift count is >= 32, sign-extend hi.
		gins(i386.ACMPL, &cx, ncon(32))

		p1 = gc.Gbranch(optoas(gc.OLT, gc.Types[gc.TUINT32]), nil, +1)
		gins(i386.AMOVL, &dx, &ax)
		if hi1.Type.Etype == gc.TINT32 {
			gins(i386.ASARL, &cx, &ax) // SARL only uses bottom 5 bits of count
			gins(i386.ASARL, ncon(31), &dx)
		} else {
			gins(i386.ASHRL, &cx, &ax)
			gins(i386.AXORL, &dx, &dx)
		}

		p2 = gc.Gbranch(obj.AJMP, nil, 0)
		gc.Patch(p1, gc.Pc)

		// general shift
		p1 = gins(i386.ASHRL, &cx, &ax)

		p1.From.Index = i386.REG_DX // double-width shift
		p1.From.Scale = 0
		gins(optoas(gc.ORSH, hi1.Type), &cx, &dx)
		gc.Patch(p2, gc.Pc)

		// make constant the right side (it usually is anyway).
	case gc.OXOR,
		gc.OAND,
		gc.OOR:
		if lo1.Op == gc.OLITERAL {
			nswap(&lo1, &lo2)
			nswap(&hi1, &hi2)
		}

		if lo2.Op == gc.OLITERAL {
			// special cases for constants.
			lv := uint32(gc.Mpgetfix(lo2.Val.U.Xval))

			hv := uint32(gc.Mpgetfix(hi2.Val.U.Xval))
			splitclean() // right side
			split64(res, &lo2, &hi2)
			switch n.Op {
			case gc.OXOR:
				gmove(&lo1, &lo2)
				gmove(&hi1, &hi2)
				switch lv {
				case 0:
					break

				case 0xffffffff:
					gins(i386.ANOTL, nil, &lo2)

				default:
					gins(i386.AXORL, ncon(lv), &lo2)
				}

				switch hv {
				case 0:
					break

				case 0xffffffff:
					gins(i386.ANOTL, nil, &hi2)

				default:
					gins(i386.AXORL, ncon(hv), &hi2)
				}

			case gc.OAND:
				switch lv {
				case 0:
					gins(i386.AMOVL, ncon(0), &lo2)

				default:
					gmove(&lo1, &lo2)
					if lv != 0xffffffff {
						gins(i386.AANDL, ncon(lv), &lo2)
					}
				}

				switch hv {
				case 0:
					gins(i386.AMOVL, ncon(0), &hi2)

				default:
					gmove(&hi1, &hi2)
					if hv != 0xffffffff {
						gins(i386.AANDL, ncon(hv), &hi2)
					}
				}

			case gc.OOR:
				switch lv {
				case 0:
					gmove(&lo1, &lo2)

				case 0xffffffff:
					gins(i386.AMOVL, ncon(0xffffffff), &lo2)

				default:
					gmove(&lo1, &lo2)
					gins(i386.AORL, ncon(lv), &lo2)
				}

				switch hv {
				case 0:
					gmove(&hi1, &hi2)

				case 0xffffffff:
					gins(i386.AMOVL, ncon(0xffffffff), &hi2)

				default:
					gmove(&hi1, &hi2)
					gins(i386.AORL, ncon(hv), &hi2)
				}
			}

			splitclean()
			splitclean()
			goto out
		}

		gins(i386.AMOVL, &lo1, &ax)
		gins(i386.AMOVL, &hi1, &dx)
		gins(optoas(int(n.Op), lo1.Type), &lo2, &ax)
		gins(optoas(int(n.Op), lo1.Type), &hi2, &dx)
	}

	if gc.Is64(r.Type) {
		splitclean()
	}
	splitclean()

	split64(res, &lo1, &hi1)
	gins(i386.AMOVL, &ax, &lo1)
	gins(i386.AMOVL, &dx, &hi1)
	splitclean()

out:
}

/*
 * generate comparison of nl, nr, both 64-bit.
 * nl is memory; nr is constant or memory.
 */
func cmp64(nl *gc.Node, nr *gc.Node, op int, likely int, to *obj.Prog) {
	var lo1 gc.Node
	var hi1 gc.Node
	var lo2 gc.Node
	var hi2 gc.Node
	var rr gc.Node

	split64(nl, &lo1, &hi1)
	split64(nr, &lo2, &hi2)

	// compare most significant word;
	// if they differ, we're done.
	t := hi1.Type

	if nl.Op == gc.OLITERAL || nr.Op == gc.OLITERAL {
		gins(i386.ACMPL, &hi1, &hi2)
	} else {
		regalloc(&rr, gc.Types[gc.TINT32], nil)
		gins(i386.AMOVL, &hi1, &rr)
		gins(i386.ACMPL, &rr, &hi2)
		regfree(&rr)
	}

	br := (*obj.Prog)(nil)
	switch op {
	default:
		gc.Fatal("cmp64 %v %v", gc.Oconv(int(op), 0), gc.Tconv(t, 0))

		// cmp hi
	// jne L
	// cmp lo
	// jeq to
	// L:
	case gc.OEQ:
		br = gc.Gbranch(i386.AJNE, nil, -likely)

		// cmp hi
	// jne to
	// cmp lo
	// jne to
	case gc.ONE:
		gc.Patch(gc.Gbranch(i386.AJNE, nil, likely), to)

		// cmp hi
	// jgt to
	// jlt L
	// cmp lo
	// jge to (or jgt to)
	// L:
	case gc.OGE,
		gc.OGT:
		gc.Patch(gc.Gbranch(optoas(gc.OGT, t), nil, likely), to)

		br = gc.Gbranch(optoas(gc.OLT, t), nil, -likely)

		// cmp hi
	// jlt to
	// jgt L
	// cmp lo
	// jle to (or jlt to)
	// L:
	case gc.OLE,
		gc.OLT:
		gc.Patch(gc.Gbranch(optoas(gc.OLT, t), nil, likely), to)

		br = gc.Gbranch(optoas(gc.OGT, t), nil, -likely)
	}

	// compare least significant word
	t = lo1.Type

	if nl.Op == gc.OLITERAL || nr.Op == gc.OLITERAL {
		gins(i386.ACMPL, &lo1, &lo2)
	} else {
		regalloc(&rr, gc.Types[gc.TINT32], nil)
		gins(i386.AMOVL, &lo1, &rr)
		gins(i386.ACMPL, &rr, &lo2)
		regfree(&rr)
	}

	// jump again
	gc.Patch(gc.Gbranch(optoas(op, t), nil, likely), to)

	// point first branch down here if appropriate
	if br != nil {
		gc.Patch(br, gc.Pc)
	}

	splitclean()
	splitclean()
}
