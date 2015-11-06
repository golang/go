// Derived from Inferno utils/6c/peep.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/peep.c
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

package amd64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
	"fmt"
)

var gactive uint32

const (
	exregoffset = x86.REG_R15
)

// do we need the carry bit
func needc(p *obj.Prog) bool {
	for p != nil {
		flags := progcarryflags(p)
		if flags&gc.UseCarry != 0 {
			return true
		}
		if flags&(gc.SetCarry|gc.KillCarry) != 0 {
			return false
		}
		p = p.Link
	}

	return false
}

func rnops(r *gc.Flow) *gc.Flow {
	if r != nil {
		var p *obj.Prog
		var r1 *gc.Flow
		for {
			p = r.Prog
			if p.As != obj.ANOP || p.From.Type != obj.TYPE_NONE || p.To.Type != obj.TYPE_NONE {
				break
			}
			r1 = gc.Uniqs(r)
			if r1 == nil {
				break
			}
			r = r1
		}
	}

	return r
}

func peep(firstp *obj.Prog) {
	g := (*gc.Graph)(gc.Flowstart(firstp, nil))
	if g == nil {
		return
	}
	gactive = 0

	// byte, word arithmetic elimination.
	elimshortmov(g)

	// constant propagation
	// find MOV $con,R followed by
	// another MOV $con,R without
	// setting R in the interim
	var p *obj.Prog
	for r := (*gc.Flow)(g.Start); r != nil; r = r.Link {
		p = r.Prog
		switch p.As {
		case x86.ALEAL,
			x86.ALEAQ:
			if regtyp(&p.To) {
				if p.From.Sym != nil {
					if p.From.Index == x86.REG_NONE {
						conprop(r)
					}
				}
			}

		case x86.AMOVB,
			x86.AMOVW,
			x86.AMOVL,
			x86.AMOVQ,
			x86.AMOVSS,
			x86.AMOVSD:
			if regtyp(&p.To) {
				if p.From.Type == obj.TYPE_CONST || p.From.Type == obj.TYPE_FCONST {
					conprop(r)
				}
			}
		}
	}

	var r *gc.Flow
	var r1 *gc.Flow
	var p1 *obj.Prog
	var t int
loop1:
	if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
		gc.Dumpit("loop1", g.Start, 0)
	}

	t = 0
	for r = g.Start; r != nil; r = r.Link {
		p = r.Prog
		switch p.As {
		case x86.AMOVL,
			x86.AMOVQ,
			x86.AMOVSS,
			x86.AMOVSD:
			if regtyp(&p.To) {
				if regtyp(&p.From) {
					if copyprop(g, r) {
						excise(r)
						t++
					} else if subprop(r) && copyprop(g, r) {
						excise(r)
						t++
					}
				}
			}

		case x86.AMOVBLZX,
			x86.AMOVWLZX,
			x86.AMOVBLSX,
			x86.AMOVWLSX:
			if regtyp(&p.To) {
				r1 = rnops(gc.Uniqs(r))
				if r1 != nil {
					p1 = r1.Prog
					if p.As == p1.As && p.To.Type == p1.From.Type && p.To.Reg == p1.From.Reg {
						p1.As = x86.AMOVL
						t++
					}
				}
			}

		case x86.AMOVBQSX,
			x86.AMOVBQZX,
			x86.AMOVWQSX,
			x86.AMOVWQZX,
			x86.AMOVLQSX,
			x86.AMOVLQZX,
			x86.AMOVQL:
			if regtyp(&p.To) {
				r1 = rnops(gc.Uniqs(r))
				if r1 != nil {
					p1 = r1.Prog
					if p.As == p1.As && p.To.Type == p1.From.Type && p.To.Reg == p1.From.Reg {
						p1.As = x86.AMOVQ
						t++
					}
				}
			}

		case x86.AADDL,
			x86.AADDQ,
			x86.AADDW:
			if p.From.Type != obj.TYPE_CONST || needc(p.Link) {
				break
			}
			if p.From.Offset == -1 {
				if p.As == x86.AADDQ {
					p.As = x86.ADECQ
				} else if p.As == x86.AADDL {
					p.As = x86.ADECL
				} else {
					p.As = x86.ADECW
				}
				p.From = obj.Addr{}
				break
			}

			if p.From.Offset == 1 {
				if p.As == x86.AADDQ {
					p.As = x86.AINCQ
				} else if p.As == x86.AADDL {
					p.As = x86.AINCL
				} else {
					p.As = x86.AINCW
				}
				p.From = obj.Addr{}
				break
			}

		case x86.ASUBL,
			x86.ASUBQ,
			x86.ASUBW:
			if p.From.Type != obj.TYPE_CONST || needc(p.Link) {
				break
			}
			if p.From.Offset == -1 {
				if p.As == x86.ASUBQ {
					p.As = x86.AINCQ
				} else if p.As == x86.ASUBL {
					p.As = x86.AINCL
				} else {
					p.As = x86.AINCW
				}
				p.From = obj.Addr{}
				break
			}

			if p.From.Offset == 1 {
				if p.As == x86.ASUBQ {
					p.As = x86.ADECQ
				} else if p.As == x86.ASUBL {
					p.As = x86.ADECL
				} else {
					p.As = x86.ADECW
				}
				p.From = obj.Addr{}
				break
			}
		}
	}

	if t != 0 {
		goto loop1
	}

	// MOVLQZX removal.
	// The MOVLQZX exists to avoid being confused for a
	// MOVL that is just copying 32-bit data around during
	// copyprop.  Now that copyprop is done, remov MOVLQZX R1, R2
	// if it is dominated by an earlier ADDL/MOVL/etc into R1 that
	// will have already cleared the high bits.
	//
	// MOVSD removal.
	// We never use packed registers, so a MOVSD between registers
	// can be replaced by MOVAPD, which moves the pair of float64s
	// instead of just the lower one.  We only use the lower one, but
	// the processor can do better if we do moves using both.
	for r := (*gc.Flow)(g.Start); r != nil; r = r.Link {
		p = r.Prog
		if p.As == x86.AMOVLQZX {
			if regtyp(&p.From) {
				if p.From.Type == p.To.Type && p.From.Reg == p.To.Reg {
					if prevl(r, int(p.From.Reg)) {
						excise(r)
					}
				}
			}
		}

		if p.As == x86.AMOVSD {
			if regtyp(&p.From) {
				if regtyp(&p.To) {
					p.As = x86.AMOVAPD
				}
			}
		}
	}

	// load pipelining
	// push any load from memory as early as possible
	// to give it time to complete before use.
	for r := (*gc.Flow)(g.Start); r != nil; r = r.Link {
		p = r.Prog
		switch p.As {
		case x86.AMOVB,
			x86.AMOVW,
			x86.AMOVL,
			x86.AMOVQ,
			x86.AMOVLQZX:
			if regtyp(&p.To) && !regconsttyp(&p.From) {
				pushback(r)
			}
		}
	}

	gc.Flowend(g)
}

func pushback(r0 *gc.Flow) {
	var r *gc.Flow
	var p *obj.Prog

	var b *gc.Flow
	p0 := (*obj.Prog)(r0.Prog)
	for r = gc.Uniqp(r0); r != nil && gc.Uniqs(r) != nil; r = gc.Uniqp(r) {
		p = r.Prog
		if p.As != obj.ANOP {
			if !regconsttyp(&p.From) || !regtyp(&p.To) {
				break
			}
			if copyu(p, &p0.To, nil) != 0 || copyu(p0, &p.To, nil) != 0 {
				break
			}
		}

		if p.As == obj.ACALL {
			break
		}
		b = r
	}

	if b == nil {
		if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
			fmt.Printf("no pushback: %v\n", r0.Prog)
			if r != nil {
				fmt.Printf("\t%v [%v]\n", r.Prog, gc.Uniqs(r) != nil)
			}
		}

		return
	}

	if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
		fmt.Printf("pushback\n")
		for r := (*gc.Flow)(b); ; r = r.Link {
			fmt.Printf("\t%v\n", r.Prog)
			if r == r0 {
				break
			}
		}
	}

	t := obj.Prog(*r0.Prog)
	for r = gc.Uniqp(r0); ; r = gc.Uniqp(r) {
		p0 = r.Link.Prog
		p = r.Prog
		p0.As = p.As
		p0.Lineno = p.Lineno
		p0.From = p.From
		p0.To = p.To

		if r == b {
			break
		}
	}

	p0 = r.Prog
	p0.As = t.As
	p0.Lineno = t.Lineno
	p0.From = t.From
	p0.To = t.To

	if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
		fmt.Printf("\tafter\n")
		for r := (*gc.Flow)(b); ; r = r.Link {
			fmt.Printf("\t%v\n", r.Prog)
			if r == r0 {
				break
			}
		}
	}
}

func excise(r *gc.Flow) {
	p := (*obj.Prog)(r.Prog)
	if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
		fmt.Printf("%v ===delete===\n", p)
	}

	obj.Nopout(p)

	gc.Ostats.Ndelmov++
}

func regtyp(a *obj.Addr) bool {
	return a.Type == obj.TYPE_REG && (x86.REG_AX <= a.Reg && a.Reg <= x86.REG_R15 || x86.REG_X0 <= a.Reg && a.Reg <= x86.REG_X15)
}

// movb elimination.
// movb is simulated by the linker
// when a register other than ax, bx, cx, dx
// is used, so rewrite to other instructions
// when possible.  a movb into a register
// can smash the entire 32-bit register without
// causing any trouble.
//
// TODO: Using the Q forms here instead of the L forms
// seems unnecessary, and it makes the instructions longer.
func elimshortmov(g *gc.Graph) {
	var p *obj.Prog

	for r := (*gc.Flow)(g.Start); r != nil; r = r.Link {
		p = r.Prog
		if regtyp(&p.To) {
			switch p.As {
			case x86.AINCB,
				x86.AINCW:
				p.As = x86.AINCQ

			case x86.ADECB,
				x86.ADECW:
				p.As = x86.ADECQ

			case x86.ANEGB,
				x86.ANEGW:
				p.As = x86.ANEGQ

			case x86.ANOTB,
				x86.ANOTW:
				p.As = x86.ANOTQ
			}

			if regtyp(&p.From) || p.From.Type == obj.TYPE_CONST {
				// move or artihmetic into partial register.
				// from another register or constant can be movl.
				// we don't switch to 64-bit arithmetic if it can
				// change how the carry bit is set (and the carry bit is needed).
				switch p.As {
				case x86.AMOVB,
					x86.AMOVW:
					p.As = x86.AMOVQ

				case x86.AADDB,
					x86.AADDW:
					if !needc(p.Link) {
						p.As = x86.AADDQ
					}

				case x86.ASUBB,
					x86.ASUBW:
					if !needc(p.Link) {
						p.As = x86.ASUBQ
					}

				case x86.AMULB,
					x86.AMULW:
					p.As = x86.AMULQ

				case x86.AIMULB,
					x86.AIMULW:
					p.As = x86.AIMULQ

				case x86.AANDB,
					x86.AANDW:
					p.As = x86.AANDQ

				case x86.AORB,
					x86.AORW:
					p.As = x86.AORQ

				case x86.AXORB,
					x86.AXORW:
					p.As = x86.AXORQ

				case x86.ASHLB,
					x86.ASHLW:
					p.As = x86.ASHLQ
				}
			} else if p.From.Type != obj.TYPE_REG {
				// explicit zero extension, but don't
				// do that if source is a byte register
				// (only AH can occur and it's forbidden).
				switch p.As {
				case x86.AMOVB:
					p.As = x86.AMOVBQZX

				case x86.AMOVW:
					p.As = x86.AMOVWQZX
				}
			}
		}
	}
}

// is 'a' a register or constant?
func regconsttyp(a *obj.Addr) bool {
	if regtyp(a) {
		return true
	}
	switch a.Type {
	case obj.TYPE_CONST,
		obj.TYPE_FCONST,
		obj.TYPE_SCONST,
		obj.TYPE_ADDR: // TODO(rsc): Not all TYPE_ADDRs are constants.
		return true
	}

	return false
}

// is reg guaranteed to be truncated by a previous L instruction?
func prevl(r0 *gc.Flow, reg int) bool {
	for r := (*gc.Flow)(gc.Uniqp(r0)); r != nil; r = gc.Uniqp(r) {
		p := r.Prog
		if p.To.Type == obj.TYPE_REG && int(p.To.Reg) == reg {
			flags := progflags(p)
			if flags&gc.RightWrite != 0 {
				if flags&gc.SizeL != 0 {
					return true
				}
				return false
			}
		}
	}

	return false
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
	if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
		fmt.Printf("subprop %v\n", r0.Prog)
	}
	p := (*obj.Prog)(r0.Prog)
	v1 := (*obj.Addr)(&p.From)
	if !regtyp(v1) {
		if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
			fmt.Printf("\tnot regtype %v; return 0\n", gc.Ctxt.Dconv(v1))
		}
		return false
	}

	v2 := (*obj.Addr)(&p.To)
	if !regtyp(v2) {
		if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
			fmt.Printf("\tnot regtype %v; return 0\n", gc.Ctxt.Dconv(v2))
		}
		return false
	}

	for r := gc.Uniqp(r0); r != nil; r = gc.Uniqp(r) {
		if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
			fmt.Printf("\t? %v\n", r.Prog)
		}
		if gc.Uniqs(r) == nil {
			if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
				fmt.Printf("\tno unique successor\n")
			}
			break
		}

		p = r.Prog
		if p.As == obj.AVARDEF || p.As == obj.AVARKILL {
			continue
		}
		if p.Info.Flags&gc.Call != 0 {
			if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
				fmt.Printf("\tfound %v; return 0\n", p)
			}
			return false
		}

		if p.Info.Reguse|p.Info.Regset != 0 {
			if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
				fmt.Printf("\tfound %v; return 0\n", p)
			}
			return false
		}

		if (p.Info.Flags&gc.Move != 0) && (p.Info.Flags&(gc.SizeL|gc.SizeQ|gc.SizeF|gc.SizeD) != 0) && p.To.Type == v1.Type && p.To.Reg == v1.Reg {
			copysub(&p.To, v1, v2, 1)
			if gc.Debug['P'] != 0 {
				fmt.Printf("gotit: %v->%v\n%v", gc.Ctxt.Dconv(v1), gc.Ctxt.Dconv(v2), r.Prog)
				if p.From.Type == v2.Type && p.From.Reg == v2.Reg {
					fmt.Printf(" excise")
				}
				fmt.Printf("\n")
			}

			for r = gc.Uniqs(r); r != r0; r = gc.Uniqs(r) {
				p = r.Prog
				copysub(&p.From, v1, v2, 1)
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

		if copyau(&p.From, v2) || copyau(&p.To, v2) {
			if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
				fmt.Printf("\tcopyau %v failed\n", gc.Ctxt.Dconv(v2))
			}
			break
		}

		if copysub(&p.From, v1, v2, 0) != 0 || copysub(&p.To, v1, v2, 0) != 0 {
			if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
				fmt.Printf("\tcopysub failed\n")
			}
			break
		}
	}

	if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
		fmt.Printf("\tran off end; return 0\n")
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
	if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
		fmt.Printf("copyprop %v\n", r0.Prog)
	}
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
				fmt.Printf("; %v rar; return 0\n", gc.Ctxt.Dconv(v2))
			}
			return false

		case 3: /* set */
			if gc.Debug['P'] != 0 {
				fmt.Printf("; %v set; return 1\n", gc.Ctxt.Dconv(v2))
			}
			return true

		case 1, /* used, substitute */
			4: /* use and set */
			if f != 0 {
				if gc.Debug['P'] == 0 {
					return false
				}
				if t == 4 {
					fmt.Printf("; %v used+set and f=%d; return 0\n", gc.Ctxt.Dconv(v2), f)
				} else {
					fmt.Printf("; %v used and f=%d; return 0\n", gc.Ctxt.Dconv(v2), f)
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
				fmt.Printf("; sub %v/%v", gc.Ctxt.Dconv(v2), gc.Ctxt.Dconv(v1))
			}
			if t == 4 {
				if gc.Debug['P'] != 0 {
					fmt.Printf("; %v used+set; return 1\n", gc.Ctxt.Dconv(v2))
				}
				return true
			}
		}

		if f == 0 {
			t = copyu(p, v1, nil)
			if f == 0 && (t == 2 || t == 3 || t == 4) {
				f = 1
				if gc.Debug['P'] != 0 {
					fmt.Printf("; %v set and !f; f=%d", gc.Ctxt.Dconv(v1), f)
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
	case obj.AJMP:
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

	case obj.ARET:
		if s != nil {
			return 1
		}
		return 3

	case obj.ACALL:
		if x86.REGEXT != 0 /*TypeKind(100016)*/ && v.Type == obj.TYPE_REG && v.Reg <= x86.REGEXT && v.Reg > exregoffset {
			return 2
		}
		if x86.REGARG >= 0 && v.Type == obj.TYPE_REG && v.Reg == x86.REGARG {
			return 2
		}
		if v.Type == p.From.Type && v.Reg == p.From.Reg {
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

	case obj.ATEXT:
		if x86.REGARG >= 0 && v.Type == obj.TYPE_REG && v.Reg == x86.REGARG {
			return 3
		}
		return 0
	}

	if p.As == obj.AVARDEF || p.As == obj.AVARKILL {
		return 0
	}

	if (p.Info.Reguse|p.Info.Regset)&RtoB(int(v.Reg)) != 0 {
		return 2
	}

	if (p.Info.Reguse|p.Info.Regset)&FtoB(int(v.Reg)) != 0 {
		return 2
	}

	if p.Info.Flags&gc.LeftAddr != 0 {
		if copyas(&p.From, v) {
			return 2
		}
	}

	if p.Info.Flags&(gc.RightRead|gc.RightWrite) == gc.RightRead|gc.RightWrite {
		if copyas(&p.To, v) {
			return 2
		}
	}

	if p.Info.Flags&gc.RightWrite != 0 {
		if copyas(&p.To, v) {
			if s != nil {
				return copysub(&p.From, v, s, 1)
			}
			if copyau(&p.From, v) {
				return 4
			}
			return 3
		}
	}

	if p.Info.Flags&(gc.LeftAddr|gc.LeftRead|gc.LeftWrite|gc.RightAddr|gc.RightRead|gc.RightWrite) != 0 {
		if s != nil {
			if copysub(&p.From, v, s, 1) != 0 {
				return 1
			}
			return copysub(&p.To, v, s, 1)
		}

		if copyau(&p.From, v) {
			return 1
		}
		if copyau(&p.To, v) {
			return 1
		}
	}

	return 0
}

/*
 * direct reference,
 * could be set/use depending on
 * semantics
 */
func copyas(a *obj.Addr, v *obj.Addr) bool {
	if x86.REG_AL <= a.Reg && a.Reg <= x86.REG_R15B {
		gc.Fatalf("use of byte register")
	}
	if x86.REG_AL <= v.Reg && v.Reg <= x86.REG_R15B {
		gc.Fatalf("use of byte register")
	}

	if a.Type != v.Type || a.Name != v.Name || a.Reg != v.Reg {
		return false
	}
	if regtyp(v) {
		return true
	}
	if v.Type == obj.TYPE_MEM && (v.Name == obj.NAME_AUTO || v.Name == obj.NAME_PARAM) {
		if v.Offset == a.Offset {
			return true
		}
	}
	return false
}

func sameaddr(a *obj.Addr, v *obj.Addr) bool {
	if a.Type != v.Type || a.Name != v.Name || a.Reg != v.Reg {
		return false
	}
	if regtyp(v) {
		return true
	}
	if v.Type == obj.TYPE_MEM && (v.Name == obj.NAME_AUTO || v.Name == obj.NAME_PARAM) {
		if v.Offset == a.Offset {
			return true
		}
	}
	return false
}

/*
 * either direct or indirect
 */
func copyau(a *obj.Addr, v *obj.Addr) bool {
	if copyas(a, v) {
		if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
			fmt.Printf("\tcopyau: copyas returned 1\n")
		}
		return true
	}

	if regtyp(v) {
		if a.Type == obj.TYPE_MEM && a.Reg == v.Reg {
			if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
				fmt.Printf("\tcopyau: found indir use - return 1\n")
			}
			return true
		}

		if a.Index == v.Reg {
			if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
				fmt.Printf("\tcopyau: found index use - return 1\n")
			}
			return true
		}
	}

	return false
}

/*
 * substitute s for v in a
 * return failure to substitute
 */
func copysub(a *obj.Addr, v *obj.Addr, s *obj.Addr, f int) int {
	if copyas(a, v) {
		reg := int(int(s.Reg))
		if reg >= x86.REG_AX && reg <= x86.REG_R15 || reg >= x86.REG_X0 && reg <= x86.REG_X0+15 {
			if f != 0 {
				a.Reg = int16(reg)
			}
		}

		return 0
	}

	if regtyp(v) {
		reg := int(int(v.Reg))
		if a.Type == obj.TYPE_MEM && int(a.Reg) == reg {
			if (s.Reg == x86.REG_BP || s.Reg == x86.REG_R13) && a.Index != x86.REG_NONE {
				return 1 /* can't use BP-base with index */
			}
			if f != 0 {
				a.Reg = s.Reg
			}
		}

		//			return 0;
		if int(a.Index) == reg {
			if f != 0 {
				a.Index = s.Reg
			}
			return 0
		}

		return 0
	}

	return 0
}

func conprop(r0 *gc.Flow) {
	var p *obj.Prog
	var t int

	p0 := (*obj.Prog)(r0.Prog)
	v0 := (*obj.Addr)(&p0.To)
	r := (*gc.Flow)(r0)

loop:
	r = gc.Uniqs(r)
	if r == nil || r == r0 {
		return
	}
	if gc.Uniqp(r) == nil {
		return
	}

	p = r.Prog
	t = copyu(p, v0, nil)
	switch t {
	case 0, // miss
		1: // use
		goto loop

	case 2, // rar
		4: // use and set
		break

	case 3: // set
		if p.As == p0.As {
			if p.From.Type == p0.From.Type {
				if p.From.Reg == p0.From.Reg {
					if p.From.Node == p0.From.Node {
						if p.From.Offset == p0.From.Offset {
							if p.From.Scale == p0.From.Scale {
								if p.From.Type == obj.TYPE_FCONST && p.From.Val.(float64) == p0.From.Val.(float64) {
									if p.From.Index == p0.From.Index {
										excise(r)
										goto loop
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

func smallindir(a *obj.Addr, reg *obj.Addr) bool {
	return regtyp(reg) && a.Type == obj.TYPE_MEM && a.Reg == reg.Reg && a.Index == x86.REG_NONE && 0 <= a.Offset && a.Offset < 4096
}

func stackaddr(a *obj.Addr) bool {
	return a.Type == obj.TYPE_REG && a.Reg == x86.REG_SP
}
