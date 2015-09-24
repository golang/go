// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/arm"
)

/*
 * attempt to generate 64-bit
 *	res = n
 * return 1 on success, 0 if op not handled.
 */
func cgen64(n *gc.Node, res *gc.Node) {
	if res.Op != gc.OINDREG && res.Op != gc.ONAME {
		gc.Dump("n", n)
		gc.Dump("res", res)
		gc.Fatalf("cgen64 %v of %v", gc.Oconv(int(n.Op), 0), gc.Oconv(int(res.Op), 0))
	}

	l := n.Left
	var t1 gc.Node
	if !l.Addable {
		gc.Tempname(&t1, l.Type)
		gc.Cgen(l, &t1)
		l = &t1
	}

	var hi1 gc.Node
	var lo1 gc.Node
	split64(l, &lo1, &hi1)
	switch n.Op {
	default:
		gc.Fatalf("cgen64 %v", gc.Oconv(int(n.Op), 0))

	case gc.OMINUS:
		var lo2 gc.Node
		var hi2 gc.Node
		split64(res, &lo2, &hi2)

		gc.Regalloc(&t1, lo1.Type, nil)
		var al gc.Node
		gc.Regalloc(&al, lo1.Type, nil)
		var ah gc.Node
		gc.Regalloc(&ah, hi1.Type, nil)

		gins(arm.AMOVW, &lo1, &al)
		gins(arm.AMOVW, &hi1, &ah)

		gmove(ncon(0), &t1)
		p1 := gins(arm.ASUB, &al, &t1)
		p1.Scond |= arm.C_SBIT
		gins(arm.AMOVW, &t1, &lo2)

		gmove(ncon(0), &t1)
		gins(arm.ASBC, &ah, &t1)
		gins(arm.AMOVW, &t1, &hi2)

		gc.Regfree(&t1)
		gc.Regfree(&al)
		gc.Regfree(&ah)
		splitclean()
		splitclean()
		return

	case gc.OCOM:
		gc.Regalloc(&t1, lo1.Type, nil)
		gmove(ncon(^uint32(0)), &t1)

		var lo2 gc.Node
		var hi2 gc.Node
		split64(res, &lo2, &hi2)
		var n1 gc.Node
		gc.Regalloc(&n1, lo1.Type, nil)

		gins(arm.AMOVW, &lo1, &n1)
		gins(arm.AEOR, &t1, &n1)
		gins(arm.AMOVW, &n1, &lo2)

		gins(arm.AMOVW, &hi1, &n1)
		gins(arm.AEOR, &t1, &n1)
		gins(arm.AMOVW, &n1, &hi2)

		gc.Regfree(&t1)
		gc.Regfree(&n1)
		splitclean()
		splitclean()
		return

		// binary operators.
	// common setup below.
	case gc.OADD,
		gc.OSUB,
		gc.OMUL,
		gc.OLSH,
		gc.ORSH,
		gc.OAND,
		gc.OOR,
		gc.OXOR,
		gc.OLROT:
		break
	}

	// setup for binary operators
	r := n.Right

	if r != nil && !r.Addable {
		var t2 gc.Node
		gc.Tempname(&t2, r.Type)
		gc.Cgen(r, &t2)
		r = &t2
	}

	var hi2 gc.Node
	var lo2 gc.Node
	if gc.Is64(r.Type) {
		split64(r, &lo2, &hi2)
	}

	var al gc.Node
	gc.Regalloc(&al, lo1.Type, nil)
	var ah gc.Node
	gc.Regalloc(&ah, hi1.Type, nil)

	// Do op.  Leave result in ah:al.
	switch n.Op {
	default:
		gc.Fatalf("cgen64: not implemented: %v\n", n)

		// TODO: Constants
	case gc.OADD:
		var bl gc.Node
		gc.Regalloc(&bl, gc.Types[gc.TPTR32], nil)

		var bh gc.Node
		gc.Regalloc(&bh, gc.Types[gc.TPTR32], nil)
		gins(arm.AMOVW, &hi1, &ah)
		gins(arm.AMOVW, &lo1, &al)
		gins(arm.AMOVW, &hi2, &bh)
		gins(arm.AMOVW, &lo2, &bl)
		p1 := gins(arm.AADD, &bl, &al)
		p1.Scond |= arm.C_SBIT
		gins(arm.AADC, &bh, &ah)
		gc.Regfree(&bl)
		gc.Regfree(&bh)

		// TODO: Constants.
	case gc.OSUB:
		var bl gc.Node
		gc.Regalloc(&bl, gc.Types[gc.TPTR32], nil)

		var bh gc.Node
		gc.Regalloc(&bh, gc.Types[gc.TPTR32], nil)
		gins(arm.AMOVW, &lo1, &al)
		gins(arm.AMOVW, &hi1, &ah)
		gins(arm.AMOVW, &lo2, &bl)
		gins(arm.AMOVW, &hi2, &bh)
		p1 := gins(arm.ASUB, &bl, &al)
		p1.Scond |= arm.C_SBIT
		gins(arm.ASBC, &bh, &ah)
		gc.Regfree(&bl)
		gc.Regfree(&bh)

		// TODO(kaib): this can be done with 4 regs and does not need 6
	case gc.OMUL:
		var bl gc.Node
		gc.Regalloc(&bl, gc.Types[gc.TPTR32], nil)

		var bh gc.Node
		gc.Regalloc(&bh, gc.Types[gc.TPTR32], nil)
		var cl gc.Node
		gc.Regalloc(&cl, gc.Types[gc.TPTR32], nil)
		var ch gc.Node
		gc.Regalloc(&ch, gc.Types[gc.TPTR32], nil)

		// load args into bh:bl and bh:bl.
		gins(arm.AMOVW, &hi1, &bh)

		gins(arm.AMOVW, &lo1, &bl)
		gins(arm.AMOVW, &hi2, &ch)
		gins(arm.AMOVW, &lo2, &cl)

		// bl * cl -> ah al
		p1 := gins(arm.AMULLU, nil, nil)

		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = bl.Reg
		p1.Reg = cl.Reg
		p1.To.Type = obj.TYPE_REGREG
		p1.To.Reg = ah.Reg
		p1.To.Offset = int64(al.Reg)

		//print("%v\n", p1);

		// bl * ch + ah -> ah
		p1 = gins(arm.AMULA, nil, nil)

		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = bl.Reg
		p1.Reg = ch.Reg
		p1.To.Type = obj.TYPE_REGREG2
		p1.To.Reg = ah.Reg
		p1.To.Offset = int64(ah.Reg)

		//print("%v\n", p1);

		// bh * cl + ah -> ah
		p1 = gins(arm.AMULA, nil, nil)

		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = bh.Reg
		p1.Reg = cl.Reg
		p1.To.Type = obj.TYPE_REGREG2
		p1.To.Reg = ah.Reg
		p1.To.Offset = int64(ah.Reg)

		//print("%v\n", p1);

		gc.Regfree(&bh)

		gc.Regfree(&bl)
		gc.Regfree(&ch)
		gc.Regfree(&cl)

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
		v := uint64(r.Int())

		var bl gc.Node
		gc.Regalloc(&bl, lo1.Type, nil)
		var bh gc.Node
		gc.Regalloc(&bh, hi1.Type, nil)
		if v >= 32 {
			// reverse during load to do the first 32 bits of rotate
			v -= 32

			gins(arm.AMOVW, &hi1, &bl)
			gins(arm.AMOVW, &lo1, &bh)
		} else {
			gins(arm.AMOVW, &hi1, &bh)
			gins(arm.AMOVW, &lo1, &bl)
		}

		if v == 0 {
			gins(arm.AMOVW, &bh, &ah)
			gins(arm.AMOVW, &bl, &al)
		} else {
			// rotate by 1 <= v <= 31
			//	MOVW	bl<<v, al
			//	MOVW	bh<<v, ah
			//	OR		bl>>(32-v), ah
			//	OR		bh>>(32-v), al
			gshift(arm.AMOVW, &bl, arm.SHIFT_LL, int32(v), &al)

			gshift(arm.AMOVW, &bh, arm.SHIFT_LL, int32(v), &ah)
			gshift(arm.AORR, &bl, arm.SHIFT_LR, int32(32-v), &ah)
			gshift(arm.AORR, &bh, arm.SHIFT_LR, int32(32-v), &al)
		}

		gc.Regfree(&bl)
		gc.Regfree(&bh)

	case gc.OLSH:
		var bl gc.Node
		gc.Regalloc(&bl, lo1.Type, nil)
		var bh gc.Node
		gc.Regalloc(&bh, hi1.Type, nil)
		gins(arm.AMOVW, &hi1, &bh)
		gins(arm.AMOVW, &lo1, &bl)

		var p6 *obj.Prog
		var s gc.Node
		var n1 gc.Node
		var creg gc.Node
		var p1 *obj.Prog
		var p2 *obj.Prog
		var p3 *obj.Prog
		var p4 *obj.Prog
		var p5 *obj.Prog
		if r.Op == gc.OLITERAL {
			v := uint64(r.Int())
			if v >= 64 {
				// TODO(kaib): replace with gins(AMOVW, nodintconst(0), &al)
				// here and below (verify it optimizes to EOR)
				gins(arm.AEOR, &al, &al)

				gins(arm.AEOR, &ah, &ah)
			} else if v > 32 {
				gins(arm.AEOR, &al, &al)

				//	MOVW	bl<<(v-32), ah
				gshift(arm.AMOVW, &bl, arm.SHIFT_LL, int32(v-32), &ah)
			} else if v == 32 {
				gins(arm.AEOR, &al, &al)
				gins(arm.AMOVW, &bl, &ah)
			} else if v > 0 {
				//	MOVW	bl<<v, al
				gshift(arm.AMOVW, &bl, arm.SHIFT_LL, int32(v), &al)

				//	MOVW	bh<<v, ah
				gshift(arm.AMOVW, &bh, arm.SHIFT_LL, int32(v), &ah)

				//	OR		bl>>(32-v), ah
				gshift(arm.AORR, &bl, arm.SHIFT_LR, int32(32-v), &ah)
			} else {
				gins(arm.AMOVW, &bl, &al)
				gins(arm.AMOVW, &bh, &ah)
			}

			goto olsh_break
		}

		gc.Regalloc(&s, gc.Types[gc.TUINT32], nil)
		gc.Regalloc(&creg, gc.Types[gc.TUINT32], nil)
		if gc.Is64(r.Type) {
			// shift is >= 1<<32
			var cl gc.Node
			var ch gc.Node
			split64(r, &cl, &ch)

			gmove(&ch, &s)
			gins(arm.ATST, &s, nil)
			p6 = gc.Gbranch(arm.ABNE, nil, 0)
			gmove(&cl, &s)
			splitclean()
		} else {
			gmove(r, &s)
			p6 = nil
		}

		gins(arm.ATST, &s, nil)

		// shift == 0
		p1 = gins(arm.AMOVW, &bl, &al)

		p1.Scond = arm.C_SCOND_EQ
		p1 = gins(arm.AMOVW, &bh, &ah)
		p1.Scond = arm.C_SCOND_EQ
		p2 = gc.Gbranch(arm.ABEQ, nil, 0)

		// shift is < 32
		gc.Nodconst(&n1, gc.Types[gc.TUINT32], 32)

		gmove(&n1, &creg)
		gins(arm.ACMP, &s, &creg)

		//	MOVW.LO		bl<<s, al
		p1 = gregshift(arm.AMOVW, &bl, arm.SHIFT_LL, &s, &al)

		p1.Scond = arm.C_SCOND_LO

		//	MOVW.LO		bh<<s, ah
		p1 = gregshift(arm.AMOVW, &bh, arm.SHIFT_LL, &s, &ah)

		p1.Scond = arm.C_SCOND_LO

		//	SUB.LO		s, creg
		p1 = gins(arm.ASUB, &s, &creg)

		p1.Scond = arm.C_SCOND_LO

		//	OR.LO		bl>>creg, ah
		p1 = gregshift(arm.AORR, &bl, arm.SHIFT_LR, &creg, &ah)

		p1.Scond = arm.C_SCOND_LO

		//	BLO	end
		p3 = gc.Gbranch(arm.ABLO, nil, 0)

		// shift == 32
		p1 = gins(arm.AEOR, &al, &al)

		p1.Scond = arm.C_SCOND_EQ
		p1 = gins(arm.AMOVW, &bl, &ah)
		p1.Scond = arm.C_SCOND_EQ
		p4 = gc.Gbranch(arm.ABEQ, nil, 0)

		// shift is < 64
		gc.Nodconst(&n1, gc.Types[gc.TUINT32], 64)

		gmove(&n1, &creg)
		gins(arm.ACMP, &s, &creg)

		//	EOR.LO	al, al
		p1 = gins(arm.AEOR, &al, &al)

		p1.Scond = arm.C_SCOND_LO

		//	MOVW.LO		creg>>1, creg
		p1 = gshift(arm.AMOVW, &creg, arm.SHIFT_LR, 1, &creg)

		p1.Scond = arm.C_SCOND_LO

		//	SUB.LO		creg, s
		p1 = gins(arm.ASUB, &creg, &s)

		p1.Scond = arm.C_SCOND_LO

		//	MOVW	bl<<s, ah
		p1 = gregshift(arm.AMOVW, &bl, arm.SHIFT_LL, &s, &ah)

		p1.Scond = arm.C_SCOND_LO

		p5 = gc.Gbranch(arm.ABLO, nil, 0)

		// shift >= 64
		if p6 != nil {
			gc.Patch(p6, gc.Pc)
		}
		gins(arm.AEOR, &al, &al)
		gins(arm.AEOR, &ah, &ah)

		gc.Patch(p2, gc.Pc)
		gc.Patch(p3, gc.Pc)
		gc.Patch(p4, gc.Pc)
		gc.Patch(p5, gc.Pc)
		gc.Regfree(&s)
		gc.Regfree(&creg)

	olsh_break:
		gc.Regfree(&bl)
		gc.Regfree(&bh)

	case gc.ORSH:
		var bl gc.Node
		gc.Regalloc(&bl, lo1.Type, nil)
		var bh gc.Node
		gc.Regalloc(&bh, hi1.Type, nil)
		gins(arm.AMOVW, &hi1, &bh)
		gins(arm.AMOVW, &lo1, &bl)

		var p4 *obj.Prog
		var p5 *obj.Prog
		var n1 gc.Node
		var p6 *obj.Prog
		var s gc.Node
		var p1 *obj.Prog
		var p2 *obj.Prog
		var creg gc.Node
		var p3 *obj.Prog
		if r.Op == gc.OLITERAL {
			v := uint64(r.Int())
			if v >= 64 {
				if bh.Type.Etype == gc.TINT32 {
					//	MOVW	bh->31, al
					gshift(arm.AMOVW, &bh, arm.SHIFT_AR, 31, &al)

					//	MOVW	bh->31, ah
					gshift(arm.AMOVW, &bh, arm.SHIFT_AR, 31, &ah)
				} else {
					gins(arm.AEOR, &al, &al)
					gins(arm.AEOR, &ah, &ah)
				}
			} else if v > 32 {
				if bh.Type.Etype == gc.TINT32 {
					//	MOVW	bh->(v-32), al
					gshift(arm.AMOVW, &bh, arm.SHIFT_AR, int32(v-32), &al)

					//	MOVW	bh->31, ah
					gshift(arm.AMOVW, &bh, arm.SHIFT_AR, 31, &ah)
				} else {
					//	MOVW	bh>>(v-32), al
					gshift(arm.AMOVW, &bh, arm.SHIFT_LR, int32(v-32), &al)

					gins(arm.AEOR, &ah, &ah)
				}
			} else if v == 32 {
				gins(arm.AMOVW, &bh, &al)
				if bh.Type.Etype == gc.TINT32 {
					//	MOVW	bh->31, ah
					gshift(arm.AMOVW, &bh, arm.SHIFT_AR, 31, &ah)
				} else {
					gins(arm.AEOR, &ah, &ah)
				}
			} else if v > 0 {
				//	MOVW	bl>>v, al
				gshift(arm.AMOVW, &bl, arm.SHIFT_LR, int32(v), &al)

				//	OR		bh<<(32-v), al
				gshift(arm.AORR, &bh, arm.SHIFT_LL, int32(32-v), &al)

				if bh.Type.Etype == gc.TINT32 {
					//	MOVW	bh->v, ah
					gshift(arm.AMOVW, &bh, arm.SHIFT_AR, int32(v), &ah)
				} else {
					//	MOVW	bh>>v, ah
					gshift(arm.AMOVW, &bh, arm.SHIFT_LR, int32(v), &ah)
				}
			} else {
				gins(arm.AMOVW, &bl, &al)
				gins(arm.AMOVW, &bh, &ah)
			}

			goto orsh_break
		}

		gc.Regalloc(&s, gc.Types[gc.TUINT32], nil)
		gc.Regalloc(&creg, gc.Types[gc.TUINT32], nil)
		if gc.Is64(r.Type) {
			// shift is >= 1<<32
			var ch gc.Node
			var cl gc.Node
			split64(r, &cl, &ch)

			gmove(&ch, &s)
			gins(arm.ATST, &s, nil)
			var p1 *obj.Prog
			if bh.Type.Etype == gc.TINT32 {
				p1 = gshift(arm.AMOVW, &bh, arm.SHIFT_AR, 31, &ah)
			} else {
				p1 = gins(arm.AEOR, &ah, &ah)
			}
			p1.Scond = arm.C_SCOND_NE
			p6 = gc.Gbranch(arm.ABNE, nil, 0)
			gmove(&cl, &s)
			splitclean()
		} else {
			gmove(r, &s)
			p6 = nil
		}

		gins(arm.ATST, &s, nil)

		// shift == 0
		p1 = gins(arm.AMOVW, &bl, &al)

		p1.Scond = arm.C_SCOND_EQ
		p1 = gins(arm.AMOVW, &bh, &ah)
		p1.Scond = arm.C_SCOND_EQ
		p2 = gc.Gbranch(arm.ABEQ, nil, 0)

		// check if shift is < 32
		gc.Nodconst(&n1, gc.Types[gc.TUINT32], 32)

		gmove(&n1, &creg)
		gins(arm.ACMP, &s, &creg)

		//	MOVW.LO		bl>>s, al
		p1 = gregshift(arm.AMOVW, &bl, arm.SHIFT_LR, &s, &al)

		p1.Scond = arm.C_SCOND_LO

		//	SUB.LO		s,creg
		p1 = gins(arm.ASUB, &s, &creg)

		p1.Scond = arm.C_SCOND_LO

		//	OR.LO		bh<<(32-s), al
		p1 = gregshift(arm.AORR, &bh, arm.SHIFT_LL, &creg, &al)

		p1.Scond = arm.C_SCOND_LO

		if bh.Type.Etype == gc.TINT32 {
			//	MOVW	bh->s, ah
			p1 = gregshift(arm.AMOVW, &bh, arm.SHIFT_AR, &s, &ah)
		} else {
			//	MOVW	bh>>s, ah
			p1 = gregshift(arm.AMOVW, &bh, arm.SHIFT_LR, &s, &ah)
		}

		p1.Scond = arm.C_SCOND_LO

		//	BLO	end
		p3 = gc.Gbranch(arm.ABLO, nil, 0)

		// shift == 32
		p1 = gins(arm.AMOVW, &bh, &al)

		p1.Scond = arm.C_SCOND_EQ
		if bh.Type.Etype == gc.TINT32 {
			gshift(arm.AMOVW, &bh, arm.SHIFT_AR, 31, &ah)
		} else {
			gins(arm.AEOR, &ah, &ah)
		}
		p4 = gc.Gbranch(arm.ABEQ, nil, 0)

		// check if shift is < 64
		gc.Nodconst(&n1, gc.Types[gc.TUINT32], 64)

		gmove(&n1, &creg)
		gins(arm.ACMP, &s, &creg)

		//	MOVW.LO		creg>>1, creg
		p1 = gshift(arm.AMOVW, &creg, arm.SHIFT_LR, 1, &creg)

		p1.Scond = arm.C_SCOND_LO

		//	SUB.LO		creg, s
		p1 = gins(arm.ASUB, &creg, &s)

		p1.Scond = arm.C_SCOND_LO

		if bh.Type.Etype == gc.TINT32 {
			//	MOVW	bh->(s-32), al
			p1 := gregshift(arm.AMOVW, &bh, arm.SHIFT_AR, &s, &al)

			p1.Scond = arm.C_SCOND_LO
		} else {
			//	MOVW	bh>>(v-32), al
			p1 := gregshift(arm.AMOVW, &bh, arm.SHIFT_LR, &s, &al)

			p1.Scond = arm.C_SCOND_LO
		}

		//	BLO	end
		p5 = gc.Gbranch(arm.ABLO, nil, 0)

		// s >= 64
		if p6 != nil {
			gc.Patch(p6, gc.Pc)
		}
		if bh.Type.Etype == gc.TINT32 {
			//	MOVW	bh->31, al
			gshift(arm.AMOVW, &bh, arm.SHIFT_AR, 31, &al)
		} else {
			gins(arm.AEOR, &al, &al)
		}

		gc.Patch(p2, gc.Pc)
		gc.Patch(p3, gc.Pc)
		gc.Patch(p4, gc.Pc)
		gc.Patch(p5, gc.Pc)
		gc.Regfree(&s)
		gc.Regfree(&creg)

	orsh_break:
		gc.Regfree(&bl)
		gc.Regfree(&bh)

		// TODO(kaib): literal optimizations
	// make constant the right side (it usually is anyway).
	//		if(lo1.op == OLITERAL) {
	//			nswap(&lo1, &lo2);
	//			nswap(&hi1, &hi2);
	//		}
	//		if(lo2.op == OLITERAL) {
	//			// special cases for constants.
	//			lv = mpgetfix(lo2.val.u.xval);
	//			hv = mpgetfix(hi2.val.u.xval);
	//			splitclean();	// right side
	//			split64(res, &lo2, &hi2);
	//			switch(n->op) {
	//			case OXOR:
	//				gmove(&lo1, &lo2);
	//				gmove(&hi1, &hi2);
	//				switch(lv) {
	//				case 0:
	//					break;
	//				case 0xffffffffu:
	//					gins(ANOTL, N, &lo2);
	//					break;
	//				default:
	//					gins(AXORL, ncon(lv), &lo2);
	//					break;
	//				}
	//				switch(hv) {
	//				case 0:
	//					break;
	//				case 0xffffffffu:
	//					gins(ANOTL, N, &hi2);
	//					break;
	//				default:
	//					gins(AXORL, ncon(hv), &hi2);
	//					break;
	//				}
	//				break;

	//			case OAND:
	//				switch(lv) {
	//				case 0:
	//					gins(AMOVL, ncon(0), &lo2);
	//					break;
	//				default:
	//					gmove(&lo1, &lo2);
	//					if(lv != 0xffffffffu)
	//						gins(AANDL, ncon(lv), &lo2);
	//					break;
	//				}
	//				switch(hv) {
	//				case 0:
	//					gins(AMOVL, ncon(0), &hi2);
	//					break;
	//				default:
	//					gmove(&hi1, &hi2);
	//					if(hv != 0xffffffffu)
	//						gins(AANDL, ncon(hv), &hi2);
	//					break;
	//				}
	//				break;

	//			case OOR:
	//				switch(lv) {
	//				case 0:
	//					gmove(&lo1, &lo2);
	//					break;
	//				case 0xffffffffu:
	//					gins(AMOVL, ncon(0xffffffffu), &lo2);
	//					break;
	//				default:
	//					gmove(&lo1, &lo2);
	//					gins(AORL, ncon(lv), &lo2);
	//					break;
	//				}
	//				switch(hv) {
	//				case 0:
	//					gmove(&hi1, &hi2);
	//					break;
	//				case 0xffffffffu:
	//					gins(AMOVL, ncon(0xffffffffu), &hi2);
	//					break;
	//				default:
	//					gmove(&hi1, &hi2);
	//					gins(AORL, ncon(hv), &hi2);
	//					break;
	//				}
	//				break;
	//			}
	//			splitclean();
	//			splitclean();
	//			goto out;
	//		}
	case gc.OXOR,
		gc.OAND,
		gc.OOR:
		var n1 gc.Node
		gc.Regalloc(&n1, lo1.Type, nil)

		gins(arm.AMOVW, &lo1, &al)
		gins(arm.AMOVW, &hi1, &ah)
		gins(arm.AMOVW, &lo2, &n1)
		gins(optoas(n.Op, lo1.Type), &n1, &al)
		gins(arm.AMOVW, &hi2, &n1)
		gins(optoas(n.Op, lo1.Type), &n1, &ah)
		gc.Regfree(&n1)
	}

	if gc.Is64(r.Type) {
		splitclean()
	}
	splitclean()

	split64(res, &lo1, &hi1)
	gins(arm.AMOVW, &al, &lo1)
	gins(arm.AMOVW, &ah, &hi1)
	splitclean()

	//out:
	gc.Regfree(&al)

	gc.Regfree(&ah)
}

/*
 * generate comparison of nl, nr, both 64-bit.
 * nl is memory; nr is constant or memory.
 */
func cmp64(nl *gc.Node, nr *gc.Node, op gc.Op, likely int, to *obj.Prog) {
	var lo1 gc.Node
	var hi1 gc.Node
	var lo2 gc.Node
	var hi2 gc.Node
	var r1 gc.Node
	var r2 gc.Node

	split64(nl, &lo1, &hi1)
	split64(nr, &lo2, &hi2)

	// compare most significant word;
	// if they differ, we're done.
	t := hi1.Type

	gc.Regalloc(&r1, gc.Types[gc.TINT32], nil)
	gc.Regalloc(&r2, gc.Types[gc.TINT32], nil)
	gins(arm.AMOVW, &hi1, &r1)
	gins(arm.AMOVW, &hi2, &r2)
	gins(arm.ACMP, &r1, &r2)
	gc.Regfree(&r1)
	gc.Regfree(&r2)

	var br *obj.Prog
	switch op {
	default:
		gc.Fatalf("cmp64 %v %v", gc.Oconv(int(op), 0), t)

		// cmp hi
	// bne L
	// cmp lo
	// beq to
	// L:
	case gc.OEQ:
		br = gc.Gbranch(arm.ABNE, nil, -likely)

		// cmp hi
	// bne to
	// cmp lo
	// bne to
	case gc.ONE:
		gc.Patch(gc.Gbranch(arm.ABNE, nil, likely), to)

		// cmp hi
	// bgt to
	// blt L
	// cmp lo
	// bge to (or bgt to)
	// L:
	case gc.OGE,
		gc.OGT:
		gc.Patch(gc.Gbranch(optoas(gc.OGT, t), nil, likely), to)

		br = gc.Gbranch(optoas(gc.OLT, t), nil, -likely)

		// cmp hi
	// blt to
	// bgt L
	// cmp lo
	// ble to (or jlt to)
	// L:
	case gc.OLE,
		gc.OLT:
		gc.Patch(gc.Gbranch(optoas(gc.OLT, t), nil, likely), to)

		br = gc.Gbranch(optoas(gc.OGT, t), nil, -likely)
	}

	// compare least significant word
	t = lo1.Type

	gc.Regalloc(&r1, gc.Types[gc.TINT32], nil)
	gc.Regalloc(&r2, gc.Types[gc.TINT32], nil)
	gins(arm.AMOVW, &lo1, &r1)
	gins(arm.AMOVW, &lo2, &r2)
	gins(arm.ACMP, &r1, &r2)
	gc.Regfree(&r1)
	gc.Regfree(&r2)

	// jump again
	gc.Patch(gc.Gbranch(optoas(op, t), nil, likely), to)

	// point first branch down here if appropriate
	if br != nil {
		gc.Patch(br, gc.Pc)
	}

	splitclean()
	splitclean()
}
