// Inferno utils/5c/peep.c
// http://code.google.com/p/inferno-os/source/browse/utils/5c/peep.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package arm

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/arm"
	"fmt"
)

var gactive uint32

// UNUSED
func peep(firstp *obj.Prog) {
	g := (*gc.Graph)(gc.Flowstart(firstp, nil))
	if g == nil {
		return
	}
	gactive = 0

	var r *gc.Flow
	var p *obj.Prog
	var t int
loop1:
	if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
		gc.Dumpit("loop1", g.Start, 0)
	}

	t = 0
	for r = g.Start; r != nil; r = r.Link {
		p = r.Prog
		switch p.As {
		/*
		 * elide shift into TYPE_SHIFT operand of subsequent instruction
		 */
		//			if(shiftprop(r)) {
		//				excise(r);
		//				t++;
		//				break;
		//			}
		case arm.ASLL,
			arm.ASRL,
			arm.ASRA:
			break

		case arm.AMOVB,
			arm.AMOVH,
			arm.AMOVW,
			arm.AMOVF,
			arm.AMOVD:
			if regtyp(&p.From) {
				if p.From.Type == p.To.Type && isfloatreg(&p.From) == isfloatreg(&p.To) {
					if p.Scond == arm.C_SCOND_NONE {
						if copyprop(g, r) {
							excise(r)
							t++
							break
						}

						if subprop(r) && copyprop(g, r) {
							excise(r)
							t++
							break
						}
					}
				}
			}

		case arm.AMOVHS,
			arm.AMOVHU,
			arm.AMOVBS,
			arm.AMOVBU:
			if p.From.Type == obj.TYPE_REG {
				if shortprop(r) {
					t++
				}
			}
		}
	}

	/*
		if(p->scond == C_SCOND_NONE)
		if(regtyp(&p->to))
		if(isdconst(&p->from)) {
			constprop(&p->from, &p->to, r->s1);
		}
		break;
	*/
	if t != 0 {
		goto loop1
	}

	for r := (*gc.Flow)(g.Start); r != nil; r = r.Link {
		p = r.Prog
		switch p.As {
		/*
		 * EOR -1,x,y => MVN x,y
		 */
		case arm.AEOR:
			if isdconst(&p.From) && p.From.Offset == -1 {
				p.As = arm.AMVN
				p.From.Type = obj.TYPE_REG
				if p.Reg != 0 {
					p.From.Reg = p.Reg
				} else {
					p.From.Reg = p.To.Reg
				}
				p.Reg = 0
			}
		}
	}

	for r := (*gc.Flow)(g.Start); r != nil; r = r.Link {
		p = r.Prog
		switch p.As {
		case arm.AMOVW,
			arm.AMOVB,
			arm.AMOVBS,
			arm.AMOVBU:
			if p.From.Type == obj.TYPE_MEM && p.From.Offset == 0 {
				xtramodes(g, r, &p.From)
			} else if p.To.Type == obj.TYPE_MEM && p.To.Offset == 0 {
				xtramodes(g, r, &p.To)
			} else {
				continue
			}
		}
	}

	//		case ACMP:
	//			/*
	//			 * elide CMP $0,x if calculation of x can set condition codes
	//			 */
	//			if(isdconst(&p->from) || p->from.offset != 0)
	//				continue;
	//			r2 = r->s1;
	//			if(r2 == nil)
	//				continue;
	//			t = r2->prog->as;
	//			switch(t) {
	//			default:
	//				continue;
	//			case ABEQ:
	//			case ABNE:
	//			case ABMI:
	//			case ABPL:
	//				break;
	//			case ABGE:
	//				t = ABPL;
	//				break;
	//			case ABLT:
	//				t = ABMI;
	//				break;
	//			case ABHI:
	//				t = ABNE;
	//				break;
	//			case ABLS:
	//				t = ABEQ;
	//				break;
	//			}
	//			r1 = r;
	//			do
	//				r1 = uniqp(r1);
	//			while (r1 != nil && r1->prog->as == ANOP);
	//			if(r1 == nil)
	//				continue;
	//			p1 = r1->prog;
	//			if(p1->to.type != TYPE_REG)
	//				continue;
	//			if(p1->to.reg != p->reg)
	//			if(!(p1->as == AMOVW && p1->from.type == TYPE_REG && p1->from.reg == p->reg))
	//				continue;
	//
	//			switch(p1->as) {
	//			default:
	//				continue;
	//			case AMOVW:
	//				if(p1->from.type != TYPE_REG)
	//					continue;
	//			case AAND:
	//			case AEOR:
	//			case AORR:
	//			case ABIC:
	//			case AMVN:
	//			case ASUB:
	//			case ARSB:
	//			case AADD:
	//			case AADC:
	//			case ASBC:
	//			case ARSC:
	//				break;
	//			}
	//			p1->scond |= C_SBIT;
	//			r2->prog->as = t;
	//			excise(r);
	//			continue;

	//	predicate(g);

	gc.Flowend(g)
}

func regtyp(a *obj.Addr) bool {
	return a.Type == obj.TYPE_REG && (arm.REG_R0 <= a.Reg && a.Reg <= arm.REG_R15 || arm.REG_F0 <= a.Reg && a.Reg <= arm.REG_F15)
}

/*
 * the idea is to substitute
 * one register for another
 * from one MOV to another
 *	MOV	a, R0
 *	ADD	b, R0	/ no use of R1
 *	MOV	R0, R1
 * would be converted to
 *	MOV	a, R1
 *	ADD	b, R1
 *	MOV	R1, R0
 * hopefully, then the former or latter MOV
 * will be eliminated by copy propagation.
 */
func subprop(r0 *gc.Flow) bool {
	p := (*obj.Prog)(r0.Prog)
	v1 := (*obj.Addr)(&p.From)
	if !regtyp(v1) {
		return false
	}
	v2 := (*obj.Addr)(&p.To)
	if !regtyp(v2) {
		return false
	}
	for r := gc.Uniqp(r0); r != nil; r = gc.Uniqp(r) {
		if gc.Uniqs(r) == nil {
			break
		}
		p = r.Prog
		if p.As == obj.AVARDEF || p.As == obj.AVARKILL {
			continue
		}
		if p.Info.Flags&gc.Call != 0 {
			return false
		}

		// TODO(rsc): Whatever invalidated the info should have done this call.
		proginfo(p)

		if (p.Info.Flags&gc.CanRegRead != 0) && p.To.Type == obj.TYPE_REG {
			p.Info.Flags |= gc.RegRead
			p.Info.Flags &^= (gc.CanRegRead | gc.RightRead)
			p.Reg = p.To.Reg
		}

		switch p.As {
		case arm.AMULLU,
			arm.AMULA,
			arm.AMVN:
			return false
		}

		if p.Info.Flags&(gc.RightRead|gc.RightWrite) == gc.RightWrite {
			if p.To.Type == v1.Type {
				if p.To.Reg == v1.Reg {
					if p.Scond == arm.C_SCOND_NONE {
						copysub(&p.To, v1, v2, 1)
						if gc.Debug['P'] != 0 {
							fmt.Printf("gotit: %v->%v\n%v", gc.Ctxt.Dconv(v1), gc.Ctxt.Dconv(v2), r.Prog)
							if p.From.Type == v2.Type {
								fmt.Printf(" excise")
							}
							fmt.Printf("\n")
						}

						for r = gc.Uniqs(r); r != r0; r = gc.Uniqs(r) {
							p = r.Prog
							copysub(&p.From, v1, v2, 1)
							copysub1(p, v1, v2, 1)
							copysub(&p.To, v1, v2, 1)
							if gc.Debug['P'] != 0 {
								fmt.Printf("%v\n", r.Prog)
							}
						}

						t := int(int(v1.Reg))
						v1.Reg = v2.Reg
						v2.Reg = int16(t)
						if gc.Debug['P'] != 0 {
							fmt.Printf("%v last\n", r.Prog)
						}
						return true
					}
				}
			}
		}

		if copyau(&p.From, v2) || copyau1(p, v2) || copyau(&p.To, v2) {
			break
		}
		if copysub(&p.From, v1, v2, 0) != 0 || copysub1(p, v1, v2, 0) != 0 || copysub(&p.To, v1, v2, 0) != 0 {
			break
		}
	}

	return false
}

/*
 * The idea is to remove redundant copies.
 *	v1->v2	F=0
 *	(use v2	s/v2/v1/)*
 *	set v1	F=1
 *	use v2	return fail
 *	-----------------
 *	v1->v2	F=0
 *	(use v2	s/v2/v1/)*
 *	set v1	F=1
 *	set v2	return success
 */
func copyprop(g *gc.Graph, r0 *gc.Flow) bool {
	p := (*obj.Prog)(r0.Prog)
	v1 := (*obj.Addr)(&p.From)
	v2 := (*obj.Addr)(&p.To)
	if copyas(v1, v2) {
		return true
	}
	gactive++
	return copy1(v1, v2, r0.S1, 0)
}

func copy1(v1 *obj.Addr, v2 *obj.Addr, r *gc.Flow, f int) bool {
	if uint32(r.Active) == gactive {
		if gc.Debug['P'] != 0 {
			fmt.Printf("act set; return 1\n")
		}
		return true
	}

	r.Active = int32(gactive)
	if gc.Debug['P'] != 0 {
		fmt.Printf("copy %v->%v f=%d\n", gc.Ctxt.Dconv(v1), gc.Ctxt.Dconv(v2), f)
	}
	var t int
	var p *obj.Prog
	for ; r != nil; r = r.S1 {
		p = r.Prog
		if gc.Debug['P'] != 0 {
			fmt.Printf("%v", p)
		}
		if f == 0 && gc.Uniqp(r) == nil {
			f = 1
			if gc.Debug['P'] != 0 {
				fmt.Printf("; merge; f=%d", f)
			}
		}

		t = copyu(p, v2, nil)
		switch t {
		case 2: /* rar, can't split */
			if gc.Debug['P'] != 0 {
				fmt.Printf("; %vrar; return 0\n", gc.Ctxt.Dconv(v2))
			}
			return false

		case 3: /* set */
			if gc.Debug['P'] != 0 {
				fmt.Printf("; %vset; return 1\n", gc.Ctxt.Dconv(v2))
			}
			return true

		case 1, /* used, substitute */
			4: /* use and set */
			if f != 0 {
				if gc.Debug['P'] == 0 {
					return false
				}
				if t == 4 {
					fmt.Printf("; %vused+set and f=%d; return 0\n", gc.Ctxt.Dconv(v2), f)
				} else {
					fmt.Printf("; %vused and f=%d; return 0\n", gc.Ctxt.Dconv(v2), f)
				}
				return false
			}

			if copyu(p, v2, v1) != 0 {
				if gc.Debug['P'] != 0 {
					fmt.Printf("; sub fail; return 0\n")
				}
				return false
			}

			if gc.Debug['P'] != 0 {
				fmt.Printf("; sub%v/%v", gc.Ctxt.Dconv(v2), gc.Ctxt.Dconv(v1))
			}
			if t == 4 {
				if gc.Debug['P'] != 0 {
					fmt.Printf("; %vused+set; return 1\n", gc.Ctxt.Dconv(v2))
				}
				return true
			}
		}

		if f == 0 {
			t = copyu(p, v1, nil)
			if f == 0 && (t == 2 || t == 3 || t == 4) {
				f = 1
				if gc.Debug['P'] != 0 {
					fmt.Printf("; %vset and !f; f=%d", gc.Ctxt.Dconv(v1), f)
				}
			}
		}

		if gc.Debug['P'] != 0 {
			fmt.Printf("\n")
		}
		if r.S2 != nil {
			if !copy1(v1, v2, r.S2, f) {
				return false
			}
		}
	}

	return true
}

// UNUSED
/*
 * The idea is to remove redundant constants.
 *	$c1->v1
 *	($c1->v2 s/$c1/v1)*
 *	set v1  return
 * The v1->v2 should be eliminated by copy propagation.
 */
func constprop(c1 *obj.Addr, v1 *obj.Addr, r *gc.Flow) {
	if gc.Debug['P'] != 0 {
		fmt.Printf("constprop %v->%v\n", gc.Ctxt.Dconv(c1), gc.Ctxt.Dconv(v1))
	}
	var p *obj.Prog
	for ; r != nil; r = r.S1 {
		p = r.Prog
		if gc.Debug['P'] != 0 {
			fmt.Printf("%v", p)
		}
		if gc.Uniqp(r) == nil {
			if gc.Debug['P'] != 0 {
				fmt.Printf("; merge; return\n")
			}
			return
		}

		if p.As == arm.AMOVW && copyas(&p.From, c1) {
			if gc.Debug['P'] != 0 {
				fmt.Printf("; sub%v/%v", gc.Ctxt.Dconv(&p.From), gc.Ctxt.Dconv(v1))
			}
			p.From = *v1
		} else if copyu(p, v1, nil) > 1 {
			if gc.Debug['P'] != 0 {
				fmt.Printf("; %vset; return\n", gc.Ctxt.Dconv(v1))
			}
			return
		}

		if gc.Debug['P'] != 0 {
			fmt.Printf("\n")
		}
		if r.S2 != nil {
			constprop(c1, v1, r.S2)
		}
	}
}

/*
 * shortprop eliminates redundant zero/sign extensions.
 *
 *   MOVBS x, R
 *   <no use R>
 *   MOVBS R, R'
 *
 * changed to
 *
 *   MOVBS x, R
 *   ...
 *   MOVB  R, R' (compiled to mov)
 *
 * MOVBS above can be a MOVBS, MOVBU, MOVHS or MOVHU.
 */
func shortprop(r *gc.Flow) bool {
	p := (*obj.Prog)(r.Prog)
	r1 := (*gc.Flow)(findpre(r, &p.From))
	if r1 == nil {
		return false
	}

	p1 := (*obj.Prog)(r1.Prog)
	if p1.As == p.As {
		// Two consecutive extensions.
		goto gotit
	}

	if p1.As == arm.AMOVW && isdconst(&p1.From) && p1.From.Offset >= 0 && p1.From.Offset < 128 {
		// Loaded an immediate.
		goto gotit
	}

	return false

gotit:
	if gc.Debug['P'] != 0 {
		fmt.Printf("shortprop\n%v\n%v", p1, p)
	}
	switch p.As {
	case arm.AMOVBS,
		arm.AMOVBU:
		p.As = arm.AMOVB

	case arm.AMOVHS,
		arm.AMOVHU:
		p.As = arm.AMOVH
	}

	if gc.Debug['P'] != 0 {
		fmt.Printf(" => %v\n", obj.Aconv(int(p.As)))
	}
	return true
}

// UNUSED
/*
 * ASLL x,y,w
 * .. (not use w, not set x y w)
 * AXXX w,a,b (a != w)
 * .. (not use w)
 * (set w)
 * ----------- changed to
 * ..
 * AXXX (x<<y),a,b
 * ..
 */
func shiftprop(r *gc.Flow) bool {
	p := (*obj.Prog)(r.Prog)
	if p.To.Type != obj.TYPE_REG {
		if gc.Debug['P'] != 0 {
			fmt.Printf("\tBOTCH: result not reg; FAILURE\n")
		}
		return false
	}

	n := int(int(p.To.Reg))
	a := obj.Addr(obj.Addr{})
	if p.Reg != 0 && p.Reg != p.To.Reg {
		a.Type = obj.TYPE_REG
		a.Reg = p.Reg
	}

	if gc.Debug['P'] != 0 {
		fmt.Printf("shiftprop\n%v", p)
	}
	r1 := (*gc.Flow)(r)
	var p1 *obj.Prog
	for {
		/* find first use of shift result; abort if shift operands or result are changed */
		r1 = gc.Uniqs(r1)

		if r1 == nil {
			if gc.Debug['P'] != 0 {
				fmt.Printf("\tbranch; FAILURE\n")
			}
			return false
		}

		if gc.Uniqp(r1) == nil {
			if gc.Debug['P'] != 0 {
				fmt.Printf("\tmerge; FAILURE\n")
			}
			return false
		}

		p1 = r1.Prog
		if gc.Debug['P'] != 0 {
			fmt.Printf("\n%v", p1)
		}
		switch copyu(p1, &p.To, nil) {
		case 0: /* not used or set */
			if (p.From.Type == obj.TYPE_REG && copyu(p1, &p.From, nil) > 1) || (a.Type == obj.TYPE_REG && copyu(p1, &a, nil) > 1) {
				if gc.Debug['P'] != 0 {
					fmt.Printf("\targs modified; FAILURE\n")
				}
				return false
			}

			continue
		case 3: /* set, not used */
			{
				if gc.Debug['P'] != 0 {
					fmt.Printf("\tBOTCH: noref; FAILURE\n")
				}
				return false
			}
		}

		break
	}

	/* check whether substitution can be done */
	switch p1.As {
	default:
		if gc.Debug['P'] != 0 {
			fmt.Printf("\tnon-dpi; FAILURE\n")
		}
		return false

	case arm.AAND,
		arm.AEOR,
		arm.AADD,
		arm.AADC,
		arm.AORR,
		arm.ASUB,
		arm.ASBC,
		arm.ARSB,
		arm.ARSC:
		if int(p1.Reg) == n || (p1.Reg == 0 && p1.To.Type == obj.TYPE_REG && int(p1.To.Reg) == n) {
			if p1.From.Type != obj.TYPE_REG {
				if gc.Debug['P'] != 0 {
					fmt.Printf("\tcan't swap; FAILURE\n")
				}
				return false
			}

			p1.Reg = p1.From.Reg
			p1.From.Reg = int16(n)
			switch p1.As {
			case arm.ASUB:
				p1.As = arm.ARSB

			case arm.ARSB:
				p1.As = arm.ASUB

			case arm.ASBC:
				p1.As = arm.ARSC

			case arm.ARSC:
				p1.As = arm.ASBC
			}

			if gc.Debug['P'] != 0 {
				fmt.Printf("\t=>%v", p1)
			}
		}
		fallthrough

	case arm.ABIC,
		arm.ATST,
		arm.ACMP,
		arm.ACMN:
		if int(p1.Reg) == n {
			if gc.Debug['P'] != 0 {
				fmt.Printf("\tcan't swap; FAILURE\n")
			}
			return false
		}

		if p1.Reg == 0 && int(p1.To.Reg) == n {
			if gc.Debug['P'] != 0 {
				fmt.Printf("\tshift result used twice; FAILURE\n")
			}
			return false
		}

		//	case AMVN:
		if p1.From.Type == obj.TYPE_SHIFT {
			if gc.Debug['P'] != 0 {
				fmt.Printf("\tshift result used in shift; FAILURE\n")
			}
			return false
		}

		if p1.From.Type != obj.TYPE_REG || int(p1.From.Reg) != n {
			if gc.Debug['P'] != 0 {
				fmt.Printf("\tBOTCH: where is it used?; FAILURE\n")
			}
			return false
		}
	}

	/* check whether shift result is used subsequently */
	p2 := (*obj.Prog)(p1)

	if int(p1.To.Reg) != n {
		var p1 *obj.Prog
		for {
			r1 = gc.Uniqs(r1)
			if r1 == nil {
				if gc.Debug['P'] != 0 {
					fmt.Printf("\tinconclusive; FAILURE\n")
				}
				return false
			}

			p1 = r1.Prog
			if gc.Debug['P'] != 0 {
				fmt.Printf("\n%v", p1)
			}
			switch copyu(p1, &p.To, nil) {
			case 0: /* not used or set */
				continue

			case 3: /* set, not used */
				break

			default: /* used */
				if gc.Debug['P'] != 0 {
					fmt.Printf("\treused; FAILURE\n")
				}
				return false
			}

			break
		}
	}

	/* make the substitution */
	p2.From.Reg = 0

	o := int(int(p.Reg))
	if o == 0 {
		o = int(p.To.Reg)
	}
	o &= 15

	switch p.From.Type {
	case obj.TYPE_CONST:
		o |= int((p.From.Offset & 0x1f) << 7)

	case obj.TYPE_REG:
		o |= 1<<4 | (int(p.From.Reg)&15)<<8
	}

	switch p.As {
	case arm.ASLL:
		o |= 0 << 5

	case arm.ASRL:
		o |= 1 << 5

	case arm.ASRA:
		o |= 2 << 5
	}

	p2.From = obj.Addr{}
	p2.From.Type = obj.TYPE_SHIFT
	p2.From.Offset = int64(o)
	if gc.Debug['P'] != 0 {
		fmt.Printf("\t=>%v\tSUCCEED\n", p2)
	}
	return true
}

/*
 * findpre returns the last instruction mentioning v
 * before r. It must be a set, and there must be
 * a unique path from that instruction to r.
 */
func findpre(r *gc.Flow, v *obj.Addr) *gc.Flow {
	var r1 *gc.Flow

	for r1 = gc.Uniqp(r); r1 != nil; r, r1 = r1, gc.Uniqp(r1) {
		if gc.Uniqs(r1) != r {
			return nil
		}
		switch copyu(r1.Prog, v, nil) {
		case 1, /* used */
			2: /* read-alter-rewrite */
			return nil

		case 3, /* set */
			4: /* set and used */
			return r1
		}
	}

	return nil
}

/*
 * findinc finds ADD instructions with a constant
 * argument which falls within the immed_12 range.
 */
func findinc(r *gc.Flow, r2 *gc.Flow, v *obj.Addr) *gc.Flow {
	var r1 *gc.Flow
	var p *obj.Prog

	for r1 = gc.Uniqs(r); r1 != nil && r1 != r2; r, r1 = r1, gc.Uniqs(r1) {
		if gc.Uniqp(r1) != r {
			return nil
		}
		switch copyu(r1.Prog, v, nil) {
		case 0: /* not touched */
			continue

		case 4: /* set and used */
			p = r1.Prog

			if p.As == arm.AADD {
				if isdconst(&p.From) {
					if p.From.Offset > -4096 && p.From.Offset < 4096 {
						return r1
					}
				}
			}
			fallthrough

		default:
			return nil
		}
	}

	return nil
}

func nochange(r *gc.Flow, r2 *gc.Flow, p *obj.Prog) bool {
	if r == r2 {
		return true
	}
	n := int(0)
	var a [3]obj.Addr
	if p.Reg != 0 && p.Reg != p.To.Reg {
		a[n].Type = obj.TYPE_REG
		a[n].Reg = p.Reg
		n++
	}

	switch p.From.Type {
	case obj.TYPE_SHIFT:
		a[n].Type = obj.TYPE_REG
		a[n].Reg = int16(arm.REG_R0 + (p.From.Offset & 0xf))
		n++
		fallthrough

	case obj.TYPE_REG:
		a[n].Type = obj.TYPE_REG
		a[n].Reg = p.From.Reg
		n++
	}

	if n == 0 {
		return true
	}
	var i int
	for ; r != nil && r != r2; r = gc.Uniqs(r) {
		p = r.Prog
		for i = 0; i < n; i++ {
			if copyu(p, &a[i], nil) > 1 {
				return false
			}
		}
	}

	return true
}

func findu1(r *gc.Flow, v *obj.Addr) bool {
	for ; r != nil; r = r.S1 {
		if r.Active != 0 {
			return false
		}
		r.Active = 1
		switch copyu(r.Prog, v, nil) {
		case 1, /* used */
			2, /* read-alter-rewrite */
			4: /* set and used */
			return true

		case 3: /* set */
			return false
		}

		if r.S2 != nil {
			if findu1(r.S2, v) {
				return true
			}
		}
	}

	return false
}

func finduse(g *gc.Graph, r *gc.Flow, v *obj.Addr) bool {
	for r1 := (*gc.Flow)(g.Start); r1 != nil; r1 = r1.Link {
		r1.Active = 0
	}
	return findu1(r, v)
}

/*
 * xtramodes enables the ARM post increment and
 * shift offset addressing modes to transform
 *   MOVW   0(R3),R1
 *   ADD    $4,R3,R3
 * into
 *   MOVW.P 4(R3),R1
 * and
 *   ADD    R0,R1
 *   MOVBU  0(R1),R0
 * into
 *   MOVBU  R0<<0(R1),R0
 */
func xtramodes(g *gc.Graph, r *gc.Flow, a *obj.Addr) bool {
	p := (*obj.Prog)(r.Prog)
	v := obj.Addr(*a)
	v.Type = obj.TYPE_REG
	r1 := (*gc.Flow)(findpre(r, &v))
	if r1 != nil {
		p1 := r1.Prog
		if p1.To.Type == obj.TYPE_REG && p1.To.Reg == v.Reg {
			switch p1.As {
			case arm.AADD:
				if p1.Scond&arm.C_SBIT != 0 {
					// avoid altering ADD.S/ADC sequences.
					break
				}

				if p1.From.Type == obj.TYPE_REG || (p1.From.Type == obj.TYPE_SHIFT && p1.From.Offset&(1<<4) == 0 && ((p.As != arm.AMOVB && p.As != arm.AMOVBS) || (a == &p.From && p1.From.Offset&^0xf == 0))) || ((p1.From.Type == obj.TYPE_ADDR || p1.From.Type == obj.TYPE_CONST) && p1.From.Offset > -4096 && p1.From.Offset < 4096) {
					if nochange(gc.Uniqs(r1), r, p1) {
						if a != &p.From || v.Reg != p.To.Reg {
							if finduse(g, r.S1, &v) {
								if p1.Reg == 0 || p1.Reg == v.Reg {
									/* pre-indexing */
									p.Scond |= arm.C_WBIT
								} else {
									return false
								}
							}
						}

						switch p1.From.Type {
						/* register offset */
						case obj.TYPE_REG:
							if gc.Nacl {
								return false
							}
							*a = obj.Addr{}
							a.Type = obj.TYPE_SHIFT
							a.Offset = int64(p1.From.Reg) & 15

							/* scaled register offset */
						case obj.TYPE_SHIFT:
							if gc.Nacl {
								return false
							}
							*a = obj.Addr{}
							a.Type = obj.TYPE_SHIFT
							fallthrough

							/* immediate offset */
						case obj.TYPE_CONST,
							obj.TYPE_ADDR:
							a.Offset = p1.From.Offset
						}

						if p1.Reg != 0 {
							a.Reg = p1.Reg
						}
						excise(r1)
						return true
					}
				}

			case arm.AMOVW:
				if p1.From.Type == obj.TYPE_REG {
					r2 := (*gc.Flow)(findinc(r1, r, &p1.From))
					if r2 != nil {
						var r3 *gc.Flow
						for r3 = gc.Uniqs(r2); r3.Prog.As == obj.ANOP; r3 = gc.Uniqs(r3) {
						}
						if r3 == r {
							/* post-indexing */
							p1 := r2.Prog

							a.Reg = p1.To.Reg
							a.Offset = p1.From.Offset
							p.Scond |= arm.C_PBIT
							if !finduse(g, r, &r1.Prog.To) {
								excise(r1)
							}
							excise(r2)
							return true
						}
					}
				}
			}
		}
	}

	if a != &p.From || a.Reg != p.To.Reg {
		r1 := (*gc.Flow)(findinc(r, nil, &v))
		if r1 != nil {
			/* post-indexing */
			p1 := r1.Prog

			a.Offset = p1.From.Offset
			p.Scond |= arm.C_PBIT
			excise(r1)
			return true
		}
	}

	return false
}

/*
 * return
 * 1 if v only used (and substitute),
 * 2 if read-alter-rewrite
 * 3 if set
 * 4 if set and used
 * 0 otherwise (not touched)
 */
func copyu(p *obj.Prog, v *obj.Addr, s *obj.Addr) int {
	switch p.As {
	default:
		fmt.Printf("copyu: can't find %v\n", obj.Aconv(int(p.As)))
		return 2

	case arm.AMOVM:
		if v.Type != obj.TYPE_REG {
			return 0
		}
		if p.From.Type == obj.TYPE_CONST { /* read reglist, read/rar */
			if s != nil {
				if p.From.Offset&(1<<uint(v.Reg)) != 0 {
					return 1
				}
				if copysub(&p.To, v, s, 1) != 0 {
					return 1
				}
				return 0
			}

			if copyau(&p.To, v) {
				if p.Scond&arm.C_WBIT != 0 {
					return 2
				}
				return 1
			}

			if p.From.Offset&(1<<uint(v.Reg)) != 0 {
				return 1 /* read/rar, write reglist */
			}
		} else {
			if s != nil {
				if p.To.Offset&(1<<uint(v.Reg)) != 0 {
					return 1
				}
				if copysub(&p.From, v, s, 1) != 0 {
					return 1
				}
				return 0
			}

			if copyau(&p.From, v) {
				if p.Scond&arm.C_WBIT != 0 {
					return 2
				}
				if p.To.Offset&(1<<uint(v.Reg)) != 0 {
					return 4
				}
				return 1
			}

			if p.To.Offset&(1<<uint(v.Reg)) != 0 {
				return 3
			}
		}

		return 0

	case obj.ANOP, /* read,, write */
		arm.ASQRTD,
		arm.AMOVW,
		arm.AMOVF,
		arm.AMOVD,
		arm.AMOVH,
		arm.AMOVHS,
		arm.AMOVHU,
		arm.AMOVB,
		arm.AMOVBS,
		arm.AMOVBU,
		arm.AMOVFW,
		arm.AMOVWF,
		arm.AMOVDW,
		arm.AMOVWD,
		arm.AMOVFD,
		arm.AMOVDF:
		if p.Scond&(arm.C_WBIT|arm.C_PBIT) != 0 {
			if v.Type == obj.TYPE_REG {
				if p.From.Type == obj.TYPE_MEM || p.From.Type == obj.TYPE_SHIFT {
					if p.From.Reg == v.Reg {
						return 2
					}
				} else {
					if p.To.Reg == v.Reg {
						return 2
					}
				}
			}
		}

		if s != nil {
			if copysub(&p.From, v, s, 1) != 0 {
				return 1
			}
			if !copyas(&p.To, v) {
				if copysub(&p.To, v, s, 1) != 0 {
					return 1
				}
			}
			return 0
		}

		if copyas(&p.To, v) {
			if p.Scond != arm.C_SCOND_NONE {
				return 2
			}
			if copyau(&p.From, v) {
				return 4
			}
			return 3
		}

		if copyau(&p.From, v) {
			return 1
		}
		if copyau(&p.To, v) {
			return 1
		}
		return 0

	case arm.AMULLU, /* read, read, write, write */
		arm.AMULL,
		arm.AMULA,
		arm.AMVN:
		return 2

	case arm.AADD, /* read, read, write */
		arm.AADC,
		arm.ASUB,
		arm.ASBC,
		arm.ARSB,
		arm.ASLL,
		arm.ASRL,
		arm.ASRA,
		arm.AORR,
		arm.AAND,
		arm.AEOR,
		arm.AMUL,
		arm.AMULU,
		arm.ADIV,
		arm.ADIVU,
		arm.AMOD,
		arm.AMODU,
		arm.AADDF,
		arm.AADDD,
		arm.ASUBF,
		arm.ASUBD,
		arm.AMULF,
		arm.AMULD,
		arm.ADIVF,
		arm.ADIVD,
		obj.ACHECKNIL,
		/* read */
		arm.ACMPF, /* read, read, */
		arm.ACMPD,
		arm.ACMP,
		arm.ACMN,
		arm.ATST:
		/* read,, */
		if s != nil {
			if copysub(&p.From, v, s, 1) != 0 {
				return 1
			}
			if copysub1(p, v, s, 1) != 0 {
				return 1
			}
			if !copyas(&p.To, v) {
				if copysub(&p.To, v, s, 1) != 0 {
					return 1
				}
			}
			return 0
		}

		if copyas(&p.To, v) {
			if p.Scond != arm.C_SCOND_NONE {
				return 2
			}
			if p.Reg == 0 {
				p.Reg = p.To.Reg
			}
			if copyau(&p.From, v) {
				return 4
			}
			if copyau1(p, v) {
				return 4
			}
			return 3
		}

		if copyau(&p.From, v) {
			return 1
		}
		if copyau1(p, v) {
			return 1
		}
		if copyau(&p.To, v) {
			return 1
		}
		return 0

	case arm.ABEQ, /* read, read */
		arm.ABNE,
		arm.ABCS,
		arm.ABHS,
		arm.ABCC,
		arm.ABLO,
		arm.ABMI,
		arm.ABPL,
		arm.ABVS,
		arm.ABVC,
		arm.ABHI,
		arm.ABLS,
		arm.ABGE,
		arm.ABLT,
		arm.ABGT,
		arm.ABLE:
		if s != nil {
			if copysub(&p.From, v, s, 1) != 0 {
				return 1
			}
			return copysub1(p, v, s, 1)
		}

		if copyau(&p.From, v) {
			return 1
		}
		if copyau1(p, v) {
			return 1
		}
		return 0

	case arm.AB: /* funny */
		if s != nil {
			if copysub(&p.To, v, s, 1) != 0 {
				return 1
			}
			return 0
		}

		if copyau(&p.To, v) {
			return 1
		}
		return 0

	case obj.ARET: /* funny */
		if s != nil {
			return 1
		}
		return 3

	case arm.ABL: /* funny */
		if v.Type == obj.TYPE_REG {
			// TODO(rsc): REG_R0 and REG_F0 used to be
			// (when register numbers started at 0) exregoffset and exfregoffset,
			// which are unset entirely.
			// It's strange that this handles R0 and F0 differently from the other
			// registers. Possible failure to optimize?
			if arm.REG_R0 < v.Reg && v.Reg <= arm.REGEXT {
				return 2
			}
			if v.Reg == arm.REGARG {
				return 2
			}
			if arm.REG_F0 < v.Reg && v.Reg <= arm.FREGEXT {
				return 2
			}
		}

		if p.From.Type == obj.TYPE_REG && v.Type == obj.TYPE_REG && p.From.Reg == v.Reg {
			return 2
		}

		if s != nil {
			if copysub(&p.To, v, s, 1) != 0 {
				return 1
			}
			return 0
		}

		if copyau(&p.To, v) {
			return 4
		}
		return 3

		// R0 is zero, used by DUFFZERO, cannot be substituted.
	// R1 is ptr to memory, used and set, cannot be substituted.
	case obj.ADUFFZERO:
		if v.Type == obj.TYPE_REG {
			if v.Reg == arm.REG_R0 {
				return 1
			}
			if v.Reg == arm.REG_R0+1 {
				return 2
			}
		}

		return 0

		// R0 is scratch, set by DUFFCOPY, cannot be substituted.
	// R1, R2 areptr to src, dst, used and set, cannot be substituted.
	case obj.ADUFFCOPY:
		if v.Type == obj.TYPE_REG {
			if v.Reg == arm.REG_R0 {
				return 3
			}
			if v.Reg == arm.REG_R0+1 || v.Reg == arm.REG_R0+2 {
				return 2
			}
		}

		return 0

	case obj.ATEXT: /* funny */
		if v.Type == obj.TYPE_REG {
			if v.Reg == arm.REGARG {
				return 3
			}
		}
		return 0

	case obj.APCDATA,
		obj.AFUNCDATA,
		obj.AVARDEF,
		obj.AVARKILL,
		obj.AVARLIVE,
		obj.AUSEFIELD:
		return 0
	}
}

/*
 * direct reference,
 * could be set/use depending on
 * semantics
 */
func copyas(a *obj.Addr, v *obj.Addr) bool {
	if regtyp(v) {
		if a.Type == v.Type {
			if a.Reg == v.Reg {
				return true
			}
		}
	} else if v.Type == obj.TYPE_CONST { /* for constprop */
		if a.Type == v.Type {
			if a.Name == v.Name {
				if a.Sym == v.Sym {
					if a.Reg == v.Reg {
						if a.Offset == v.Offset {
							return true
						}
					}
				}
			}
		}
	}

	return false
}

func sameaddr(a *obj.Addr, v *obj.Addr) bool {
	if a.Type != v.Type {
		return false
	}
	if regtyp(v) && a.Reg == v.Reg {
		return true
	}

	// TODO(rsc): Change v->type to v->name and enable.
	//if(v->type == NAME_AUTO || v->type == NAME_PARAM) {
	//	if(v->offset == a->offset)
	//		return 1;
	//}
	return false
}

/*
 * either direct or indirect
 */
func copyau(a *obj.Addr, v *obj.Addr) bool {
	if copyas(a, v) {
		return true
	}
	if v.Type == obj.TYPE_REG {
		if a.Type == obj.TYPE_ADDR && a.Reg != 0 {
			if a.Reg == v.Reg {
				return true
			}
		} else if a.Type == obj.TYPE_MEM {
			if a.Reg == v.Reg {
				return true
			}
		} else if a.Type == obj.TYPE_REGREG || a.Type == obj.TYPE_REGREG2 {
			if a.Reg == v.Reg {
				return true
			}
			if a.Offset == int64(v.Reg) {
				return true
			}
		} else if a.Type == obj.TYPE_SHIFT {
			if a.Offset&0xf == int64(v.Reg-arm.REG_R0) {
				return true
			}
			if (a.Offset&(1<<4) != 0) && (a.Offset>>8)&0xf == int64(v.Reg-arm.REG_R0) {
				return true
			}
		}
	}

	return false
}

/*
 * compare v to the center
 * register in p (p->reg)
 */
func copyau1(p *obj.Prog, v *obj.Addr) bool {
	if v.Type == obj.TYPE_REG && v.Reg == 0 {
		return false
	}
	return p.Reg == v.Reg
}

/*
 * substitute s for v in a
 * return failure to substitute
 */
func copysub(a *obj.Addr, v *obj.Addr, s *obj.Addr, f int) int {
	if f != 0 {
		if copyau(a, v) {
			if a.Type == obj.TYPE_SHIFT {
				if a.Offset&0xf == int64(v.Reg-arm.REG_R0) {
					a.Offset = a.Offset&^0xf | int64(s.Reg)&0xf
				}
				if (a.Offset&(1<<4) != 0) && (a.Offset>>8)&0xf == int64(v.Reg-arm.REG_R0) {
					a.Offset = a.Offset&^(0xf<<8) | (int64(s.Reg)&0xf)<<8
				}
			} else if a.Type == obj.TYPE_REGREG || a.Type == obj.TYPE_REGREG2 {
				if a.Offset == int64(v.Reg) {
					a.Offset = int64(s.Reg)
				}
				if a.Reg == v.Reg {
					a.Reg = s.Reg
				}
			} else {
				a.Reg = s.Reg
			}
		}
	}

	return 0
}

func copysub1(p1 *obj.Prog, v *obj.Addr, s *obj.Addr, f int) int {
	if f != 0 {
		if copyau1(p1, v) {
			p1.Reg = s.Reg
		}
	}
	return 0
}

var predinfo = []struct {
	opcode    int
	notopcode int
	scond     int
	notscond  int
}{
	{arm.ABEQ, arm.ABNE, 0x0, 0x1},
	{arm.ABNE, arm.ABEQ, 0x1, 0x0},
	{arm.ABCS, arm.ABCC, 0x2, 0x3},
	{arm.ABHS, arm.ABLO, 0x2, 0x3},
	{arm.ABCC, arm.ABCS, 0x3, 0x2},
	{arm.ABLO, arm.ABHS, 0x3, 0x2},
	{arm.ABMI, arm.ABPL, 0x4, 0x5},
	{arm.ABPL, arm.ABMI, 0x5, 0x4},
	{arm.ABVS, arm.ABVC, 0x6, 0x7},
	{arm.ABVC, arm.ABVS, 0x7, 0x6},
	{arm.ABHI, arm.ABLS, 0x8, 0x9},
	{arm.ABLS, arm.ABHI, 0x9, 0x8},
	{arm.ABGE, arm.ABLT, 0xA, 0xB},
	{arm.ABLT, arm.ABGE, 0xB, 0xA},
	{arm.ABGT, arm.ABLE, 0xC, 0xD},
	{arm.ABLE, arm.ABGT, 0xD, 0xC},
}

type Joininfo struct {
	start *gc.Flow
	last  *gc.Flow
	end   *gc.Flow
	len   int
}

const (
	Join = iota
	Split
	End
	Branch
	Setcond
	Toolong
)

const (
	Falsecond = iota
	Truecond
	Delbranch
	Keepbranch
)

func isbranch(p *obj.Prog) bool {
	return (arm.ABEQ <= p.As) && (p.As <= arm.ABLE)
}

func predicable(p *obj.Prog) bool {
	switch p.As {
	case obj.ANOP,
		obj.AXXX,
		obj.ADATA,
		obj.AGLOBL,
		obj.ATEXT,
		arm.AWORD:
		return false
	}

	if isbranch(p) {
		return false
	}
	return true
}

/*
 * Depends on an analysis of the encodings performed by 5l.
 * These seem to be all of the opcodes that lead to the "S" bit
 * being set in the instruction encodings.
 *
 * C_SBIT may also have been set explicitly in p->scond.
 */
func modifiescpsr(p *obj.Prog) bool {
	switch p.As {
	case arm.AMULLU,
		arm.AMULA,
		arm.AMULU,
		arm.ADIVU,
		arm.ATEQ,
		arm.ACMN,
		arm.ATST,
		arm.ACMP,
		arm.AMUL,
		arm.ADIV,
		arm.AMOD,
		arm.AMODU,
		arm.ABL:
		return true
	}

	if p.Scond&arm.C_SBIT != 0 {
		return true
	}
	return false
}

/*
 * Find the maximal chain of instructions starting with r which could
 * be executed conditionally
 */
func joinsplit(r *gc.Flow, j *Joininfo) int {
	j.start = r
	j.last = r
	j.len = 0
	for {
		if r.P2 != nil && (r.P1 != nil || r.P2.P2link != nil) {
			j.end = r
			return Join
		}

		if r.S1 != nil && r.S2 != nil {
			j.end = r
			return Split
		}

		j.last = r
		if r.Prog.As != obj.ANOP {
			j.len++
		}
		if r.S1 == nil && r.S2 == nil {
			j.end = r.Link
			return End
		}

		if r.S2 != nil {
			j.end = r.S2
			return Branch
		}

		if modifiescpsr(r.Prog) {
			j.end = r.S1
			return Setcond
		}

		r = r.S1
		if j.len >= 4 {
			break
		}
	}

	j.end = r
	return Toolong
}

func successor(r *gc.Flow) *gc.Flow {
	if r.S1 != nil {
		return r.S1
	} else {
		return r.S2
	}
}

func applypred(rstart *gc.Flow, j *Joininfo, cond int, branch int) {
	if j.len == 0 {
		return
	}
	var pred int
	if cond == Truecond {
		pred = predinfo[rstart.Prog.As-arm.ABEQ].scond
	} else {
		pred = predinfo[rstart.Prog.As-arm.ABEQ].notscond
	}

	for r := (*gc.Flow)(j.start); ; r = successor(r) {
		if r.Prog.As == arm.AB {
			if r != j.last || branch == Delbranch {
				excise(r)
			} else {
				if cond == Truecond {
					r.Prog.As = int16(predinfo[rstart.Prog.As-arm.ABEQ].opcode)
				} else {
					r.Prog.As = int16(predinfo[rstart.Prog.As-arm.ABEQ].notopcode)
				}
			}
		} else if predicable(r.Prog) {
			r.Prog.Scond = uint8(int(r.Prog.Scond&^arm.C_SCOND) | pred)
		}
		if r.S1 != r.Link {
			r.S1 = r.Link
			r.Link.P1 = r
		}

		if r == j.last {
			break
		}
	}
}

func predicate(g *gc.Graph) {
	var t1 int
	var t2 int
	var j1 Joininfo
	var j2 Joininfo

	for r := (*gc.Flow)(g.Start); r != nil; r = r.Link {
		if isbranch(r.Prog) {
			t1 = joinsplit(r.S1, &j1)
			t2 = joinsplit(r.S2, &j2)
			if j1.last.Link != j2.start {
				continue
			}
			if j1.end == j2.end {
				if (t1 == Branch && (t2 == Join || t2 == Setcond)) || (t2 == Join && (t1 == Join || t1 == Setcond)) {
					applypred(r, &j1, Falsecond, Delbranch)
					applypred(r, &j2, Truecond, Delbranch)
					excise(r)
					continue
				}
			}

			if t1 == End || t1 == Branch {
				applypred(r, &j1, Falsecond, Keepbranch)
				excise(r)
				continue
			}
		}
	}
}

func isdconst(a *obj.Addr) bool {
	return a.Type == obj.TYPE_CONST
}

func isfloatreg(a *obj.Addr) bool {
	return arm.REG_F0 <= a.Reg && a.Reg <= arm.REG_F15
}

func stackaddr(a *obj.Addr) bool {
	return regtyp(a) && a.Reg == arm.REGSP
}

func smallindir(a *obj.Addr, reg *obj.Addr) bool {
	return reg.Type == obj.TYPE_REG && a.Type == obj.TYPE_MEM && a.Reg == reg.Reg && 0 <= a.Offset && a.Offset < 4096
}

func excise(r *gc.Flow) {
	p := (*obj.Prog)(r.Prog)
	obj.Nopout(p)
}
