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

package s390x

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/s390x"
	"fmt"
)

type usage int

const (
	_None          usage = iota // no usage found
	_Read                       // only read from
	_ReadWriteSame              // both read from and written to in a single operand
	_Write                      // only written to
	_ReadWriteDiff              // both read from and written to in different operands
)

var gactive uint32

func peep(firstp *obj.Prog) {
	g := gc.Flowstart(firstp, nil)
	if g == nil {
		return
	}
	gactive = 0

	run := func(name string, pass func(r *gc.Flow) int) int {
		n := pass(g.Start)
		if gc.Debug['P'] != 0 {
			fmt.Println(name, ":", n)
		}
		if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
			gc.Dumpit(name, g.Start, 0)
		}
		return n
	}

	for {
		n := 0
		n += run("constant propagation", constantPropagation)
		n += run("copy propagation", copyPropagation)
		n += run("cast propagation", castPropagation)
		n += run("remove load-hit-stores", removeLoadHitStores)
		n += run("dead code elimination", deadCodeElimination)
		if n == 0 {
			break
		}
	}
	run("fuse op moves", fuseOpMoves)
	run("fuse clears", fuseClear)
	run("load pipelining", loadPipelining)
	run("fuse compare branch", fuseCompareBranch)
	run("simplify ops", simplifyOps)
	run("dead code elimination", deadCodeElimination)

	// TODO(mundaym): load/store multiple aren't currently handled by copyu
	// so this pass must be last.
	run("fuse multiple", fuseMultiple)

	gc.Flowend(g)
}

func pushback(r0 *gc.Flow) {
	var r *gc.Flow

	var b *gc.Flow
	p0 := r0.Prog
	for r = gc.Uniqp(r0); r != nil && gc.Uniqs(r) != nil; r = gc.Uniqp(r) {
		p := r.Prog
		if p.As != obj.ANOP {
			if !(isReg(&p.From) || isConst(&p.From)) || !isReg(&p.To) {
				break
			}
			if copyu(p, &p0.To, nil) != _None || copyu(p0, &p.To, nil) != _None {
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
		for r := b; ; r = r.Link {
			fmt.Printf("\t%v\n", r.Prog)
			if r == r0 {
				break
			}
		}
	}

	t := *r0.Prog
	for r = gc.Uniqp(r0); ; r = gc.Uniqp(r) {
		p0 = r.Link.Prog
		p := r.Prog
		p0.As = p.As
		p0.Lineno = p.Lineno
		p0.From = p.From
		p0.To = p.To
		p0.From3 = p.From3
		p0.Reg = p.Reg
		p0.RegTo2 = p.RegTo2
		if r == b {
			break
		}
	}

	p0 = r.Prog
	p0.As = t.As
	p0.Lineno = t.Lineno
	p0.From = t.From
	p0.To = t.To
	p0.From3 = t.From3
	p0.Reg = t.Reg
	p0.RegTo2 = t.RegTo2

	if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
		fmt.Printf("\tafter\n")
		for r := b; ; r = r.Link {
			fmt.Printf("\t%v\n", r.Prog)
			if r == r0 {
				break
			}
		}
	}
}

// excise replaces the given instruction with a NOP and clears
// its operands.
func excise(r *gc.Flow) {
	p := r.Prog
	if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
		fmt.Printf("%v ===delete===\n", p)
	}
	obj.Nopout(p)
	gc.Ostats.Ndelmov++
}

// isZero returns true if a is either the constant 0 or the register
// REGZERO.
func isZero(a *obj.Addr) bool {
	if a.Type == obj.TYPE_CONST && a.Offset == 0 {
		return true
	}
	if a.Type == obj.TYPE_REG && a.Reg == s390x.REGZERO {
		return true
	}
	return false
}

// isReg returns true if a is a general purpose or floating point
// register (GPR or FPR).
//
// TODO(mundaym): currently this excludes REGZER0, but not other
// special registers.
func isReg(a *obj.Addr) bool {
	return a.Type == obj.TYPE_REG &&
		s390x.REG_R0 <= a.Reg &&
		a.Reg <= s390x.REG_F15 &&
		a.Reg != s390x.REGZERO
}

// isGPR returns true if a is a general purpose register (GPR).
// REGZERO is treated as a GPR.
func isGPR(a *obj.Addr) bool {
	return a.Type == obj.TYPE_REG &&
		s390x.REG_R0 <= a.Reg &&
		a.Reg <= s390x.REG_R15
}

// isFPR returns true if a is a floating point register (FPR).
func isFPR(a *obj.Addr) bool {
	return a.Type == obj.TYPE_REG &&
		s390x.REG_F0 <= a.Reg &&
		a.Reg <= s390x.REG_F15
}

// isConst returns true if a refers to a constant (integer or
// floating point, not string currently).
func isConst(a *obj.Addr) bool {
	return a.Type == obj.TYPE_CONST || a.Type == obj.TYPE_FCONST
}

// isBDMem returns true if a refers to a memory location addressable by a
// base register (B) and a displacement (D), such as:
// 	x+8(R1)
// and
//	0(R10)
// It returns false if the address contains an index register (X) such as:
// 	16(R1)(R2*1)
// or if a relocation is required.
func isBDMem(a *obj.Addr) bool {
	return a.Type == obj.TYPE_MEM &&
		a.Index == 0 &&
		(a.Name == obj.NAME_NONE || a.Name == obj.NAME_AUTO || a.Name == obj.NAME_PARAM)
}

// the idea is to substitute
// one register for another
// from one MOV to another
//	MOV	a, R1
//	ADD	b, R1	/ no use of R2
//	MOV	R1, R2
// would be converted to
//	MOV	a, R2
//	ADD	b, R2
//	MOV	R2, R1
// hopefully, then the former or latter MOV
// will be eliminated by copy propagation.
//
// r0 (the argument, not the register) is the MOV at the end of the
// above sequences. subprop returns true if it modified any instructions.
func subprop(r0 *gc.Flow) bool {
	p := r0.Prog
	v1 := &p.From
	if !isReg(v1) {
		return false
	}
	v2 := &p.To
	if !isReg(v2) {
		return false
	}
	cast := false
	switch p.As {
	case s390x.AMOVW, s390x.AMOVWZ,
		s390x.AMOVH, s390x.AMOVHZ,
		s390x.AMOVB, s390x.AMOVBZ:
		cast = true
	}
	for r := gc.Uniqp(r0); r != nil; r = gc.Uniqp(r) {
		if gc.Uniqs(r) == nil {
			break
		}
		p = r.Prog
		switch copyu(p, v1, nil) {
		case _Write, _ReadWriteDiff:
			if p.As == obj.ACALL {
				return false
			}
			if (!cast || p.As == r0.Prog.As) && p.To.Type == v1.Type && p.To.Reg == v1.Reg {
				copysub(&p.To, v1, v2)
				for r = gc.Uniqs(r); r != r0; r = gc.Uniqs(r) {
					p = r.Prog
					copysub(&p.From, v1, v2)
					copysub1(p, v1, v2)
					copysub(&p.To, v1, v2)
				}
				v1.Reg, v2.Reg = v2.Reg, v1.Reg
				return true
			}
			if cast {
				return false
			}
		case _ReadWriteSame:
			if cast {
				return false
			}
		}
		if copyu(p, v2, nil) != _None {
			return false
		}
	}
	return false
}

// The idea is to remove redundant copies.
//     v1->v2  F=0
//     (use v2 s/v2/v1/)*
//     set v1  F=1
//     use v2  return fail (v1->v2 move must remain)
//     -----------------
//     v1->v2  F=0
//     (use v2 s/v2/v1/)*
//     set v1  F=1
//     set v2  return success (caller can remove v1->v2 move)
func copyprop(r *gc.Flow) bool {
	p := r.Prog

	canSub := false
	switch p.As {
	case s390x.AFMOVS, s390x.AFMOVD, s390x.AMOVD:
		canSub = true
	default:
		for rr := gc.Uniqp(r); rr != nil; rr = gc.Uniqp(rr) {
			if gc.Uniqs(rr) == nil {
				break
			}
			switch copyu(rr.Prog, &p.From, nil) {
			case _Read, _None:
				continue
			}
			// write
			if rr.Prog.As == p.As {
				canSub = true
			}
			break
		}
	}
	if !canSub {
		return false
	}
	if copyas(&p.From, &p.To) {
		return true
	}

	gactive++
	return copy1(&p.From, &p.To, r.S1, 0)
}

// copy1 replaces uses of v2 with v1 starting at r and returns true if
// all uses were rewritten.
func copy1(v1 *obj.Addr, v2 *obj.Addr, r *gc.Flow, f int) bool {
	if uint32(r.Active) == gactive {
		return true
	}
	r.Active = int32(gactive)
	for ; r != nil; r = r.S1 {
		p := r.Prog
		if f == 0 && gc.Uniqp(r) == nil {
			// Multiple predecessors; conservatively
			// assume v1 was set on other path
			f = 1
		}
		t := copyu(p, v2, nil)
		switch t {
		case _ReadWriteSame:
			return false
		case _Write:
			return true
		case _Read, _ReadWriteDiff:
			if f != 0 {
				return false
			}
			if copyu(p, v2, v1) != 0 {
				return false
			}
			if t == _ReadWriteDiff {
				return true
			}
		}
		if f == 0 {
			switch copyu(p, v1, nil) {
			case _ReadWriteSame, _ReadWriteDiff, _Write:
				f = 1
			}
		}
		if r.S2 != nil {
			if !copy1(v1, v2, r.S2, f) {
				return false
			}
		}
	}
	return true
}

// If s==nil, copyu returns the set/use of v in p; otherwise, it
// modifies p to replace reads of v with reads of s and returns 0 for
// success or non-zero for failure.
//
// If s==nil, copy returns one of the following values:
// 	_Read           if v only used
//	_ReadWriteSame  if v is set and used in one address (read-alter-rewrite;
// 	                can't substitute)
//	_Write          if v is only set
//	_ReadWriteDiff  if v is set in one address and used in another (so addresses
// 	                can be rewritten independently)
//	_None           otherwise (not touched)
func copyu(p *obj.Prog, v *obj.Addr, s *obj.Addr) usage {
	if p.From3Type() != obj.TYPE_NONE && p.From3Type() != obj.TYPE_CONST {
		// Currently we never generate a From3 with anything other than a constant in it.
		fmt.Printf("copyu: From3 (%v) not implemented\n", gc.Ctxt.Dconv(p.From3))
	}

	switch p.As {
	default:
		fmt.Printf("copyu: can't find %v\n", obj.Aconv(p.As))
		return _ReadWriteSame

	case // read p.From, write p.To
		s390x.AMOVH,
		s390x.AMOVHZ,
		s390x.AMOVB,
		s390x.AMOVBZ,
		s390x.AMOVW,
		s390x.AMOVWZ,
		s390x.AMOVD,
		s390x.ANEG,
		s390x.AADDME,
		s390x.AADDZE,
		s390x.ASUBME,
		s390x.ASUBZE,
		s390x.AFMOVS,
		s390x.AFMOVD,
		s390x.ALEDBR,
		s390x.AFNEG,
		s390x.ALDEBR,
		s390x.ACLFEBR,
		s390x.ACLGEBR,
		s390x.ACLFDBR,
		s390x.ACLGDBR,
		s390x.ACFEBRA,
		s390x.ACGEBRA,
		s390x.ACFDBRA,
		s390x.ACGDBRA,
		s390x.ACELFBR,
		s390x.ACELGBR,
		s390x.ACDLFBR,
		s390x.ACDLGBR,
		s390x.ACEFBRA,
		s390x.ACEGBRA,
		s390x.ACDFBRA,
		s390x.ACDGBRA,
		s390x.AFSQRT:

		if s != nil {
			copysub(&p.From, v, s)

			// Update only indirect uses of v in p.To
			if !copyas(&p.To, v) {
				copysub(&p.To, v, s)
			}
			return _None
		}

		if copyas(&p.To, v) {
			// Fix up implicit from
			if p.From.Type == obj.TYPE_NONE {
				p.From = p.To
			}
			if copyau(&p.From, v) {
				return _ReadWriteDiff
			}
			return _Write
		}

		if copyau(&p.From, v) {
			return _Read
		}
		if copyau(&p.To, v) {
			// p.To only indirectly uses v
			return _Read
		}

		return _None

	// read p.From, read p.Reg, write p.To
	case s390x.AADD,
		s390x.AADDC,
		s390x.AADDE,
		s390x.ASUB,
		s390x.ASLW,
		s390x.ASRW,
		s390x.ASRAW,
		s390x.ASLD,
		s390x.ASRD,
		s390x.ASRAD,
		s390x.ARLL,
		s390x.ARLLG,
		s390x.AOR,
		s390x.AORN,
		s390x.AAND,
		s390x.AANDN,
		s390x.ANAND,
		s390x.ANOR,
		s390x.AXOR,
		s390x.AMULLW,
		s390x.AMULLD,
		s390x.AMULHD,
		s390x.AMULHDU,
		s390x.ADIVW,
		s390x.ADIVD,
		s390x.ADIVWU,
		s390x.ADIVDU,
		s390x.AFADDS,
		s390x.AFADD,
		s390x.AFSUBS,
		s390x.AFSUB,
		s390x.AFMULS,
		s390x.AFMUL,
		s390x.AFDIVS,
		s390x.AFDIV:
		if s != nil {
			copysub(&p.From, v, s)
			copysub1(p, v, s)

			// Update only indirect uses of v in p.To
			if !copyas(&p.To, v) {
				copysub(&p.To, v, s)
			}
		}

		if copyas(&p.To, v) {
			if p.Reg == 0 {
				p.Reg = p.To.Reg
			}
			if copyau(&p.From, v) || copyau1(p, v) {
				return _ReadWriteDiff
			}
			return _Write
		}

		if copyau(&p.From, v) {
			return _Read
		}
		if copyau1(p, v) {
			return _Read
		}
		if copyau(&p.To, v) {
			return _Read
		}
		return _None

	case s390x.ABEQ,
		s390x.ABGT,
		s390x.ABGE,
		s390x.ABLT,
		s390x.ABLE,
		s390x.ABNE,
		s390x.ABVC,
		s390x.ABVS:
		return _None

	case obj.ACHECKNIL, // read p.From
		s390x.ACMP, // read p.From, read p.To
		s390x.ACMPU,
		s390x.ACMPW,
		s390x.ACMPWU,
		s390x.AFCMPO,
		s390x.AFCMPU,
		s390x.ACEBR,
		s390x.AMVC,
		s390x.ACLC,
		s390x.AXC,
		s390x.AOC,
		s390x.ANC:
		if s != nil {
			copysub(&p.From, v, s)
			copysub(&p.To, v, s)
			return _None
		}

		if copyau(&p.From, v) {
			return _Read
		}
		if copyau(&p.To, v) {
			return _Read
		}
		return _None

	case s390x.ACMPBNE, s390x.ACMPBEQ,
		s390x.ACMPBLT, s390x.ACMPBLE,
		s390x.ACMPBGT, s390x.ACMPBGE,
		s390x.ACMPUBNE, s390x.ACMPUBEQ,
		s390x.ACMPUBLT, s390x.ACMPUBLE,
		s390x.ACMPUBGT, s390x.ACMPUBGE:
		if s != nil {
			copysub(&p.From, v, s)
			copysub1(p, v, s)
			return _None
		}
		if copyau(&p.From, v) {
			return _Read
		}
		if copyau1(p, v) {
			return _Read
		}
		return _None

	case s390x.ACLEAR:
		if s != nil {
			copysub(&p.To, v, s)
			return _None
		}
		if copyau(&p.To, v) {
			return _Read
		}
		return _None

	// go never generates a branch to a GPR
	// read p.To
	case s390x.ABR:
		if s != nil {
			copysub(&p.To, v, s)
			return _None
		}

		if copyau(&p.To, v) {
			return _Read
		}
		return _None

	case obj.ARET, obj.AUNDEF:
		if s != nil {
			return _None
		}

		// All registers die at this point, so claim
		// everything is set (and not used).
		return _Write

	case s390x.ABL:
		if v.Type == obj.TYPE_REG {
			if s390x.REGARG != -1 && v.Reg == s390x.REGARG {
				return _ReadWriteSame
			}
			if p.From.Type == obj.TYPE_REG && p.From.Reg == v.Reg {
				return _ReadWriteSame
			}
			if v.Reg == s390x.REGZERO {
				// Deliberately inserted nops set R0.
				return _ReadWriteSame
			}
			if v.Reg == s390x.REGCTXT {
				// Context register for closures.
				// TODO(mundaym): not sure if we need to exclude this.
				return _ReadWriteSame
			}
		}
		if s != nil {
			copysub(&p.To, v, s)
			return _None
		}
		if copyau(&p.To, v) {
			return _ReadWriteDiff
		}
		return _Write

	case obj.ATEXT:
		if v.Type == obj.TYPE_REG {
			if v.Reg == s390x.REGARG {
				return _Write
			}
		}
		return _None

	case obj.APCDATA,
		obj.AFUNCDATA,
		obj.AVARDEF,
		obj.AVARKILL,
		obj.AVARLIVE,
		obj.AUSEFIELD,
		obj.ANOP:
		return _None
	}
}

// copyas returns 1 if a and v address the same register.
//
// If a is the from operand, this means this operation reads the
// register in v.  If a is the to operand, this means this operation
// writes the register in v.
func copyas(a *obj.Addr, v *obj.Addr) bool {
	if isReg(v) {
		if a.Type == v.Type {
			if a.Reg == v.Reg {
				return true
			}
		}
	}
	return false
}

// copyau returns 1 if a either directly or indirectly addresses the
// same register as v.
//
// If a is the from operand, this means this operation reads the
// register in v.  If a is the to operand, this means the operation
// either reads or writes the register in v (if !copyas(a, v), then
// the operation reads the register in v).
func copyau(a *obj.Addr, v *obj.Addr) bool {
	if copyas(a, v) {
		return true
	}
	if v.Type == obj.TYPE_REG {
		if a.Type == obj.TYPE_MEM || (a.Type == obj.TYPE_ADDR && a.Reg != 0) {
			if v.Reg == a.Reg {
				return true
			}
		}
	}
	return false
}

// copyau1 returns 1 if p.Reg references the same register as v and v
// is a direct reference.
func copyau1(p *obj.Prog, v *obj.Addr) bool {
	if isReg(v) && v.Reg != 0 {
		if p.Reg == v.Reg {
			return true
		}
	}
	return false
}

// copysub replaces v.Reg with s.Reg if a.Reg and v.Reg are direct
// references to the same register.
func copysub(a, v, s *obj.Addr) {
	if copyau(a, v) {
		a.Reg = s.Reg
	}
}

// copysub1 replaces p.Reg with s.Reg if p.Reg and v.Reg are direct
// references to the same register.
func copysub1(p *obj.Prog, v, s *obj.Addr) {
	if copyau1(p, v) {
		p.Reg = s.Reg
	}
}

func sameaddr(a *obj.Addr, v *obj.Addr) bool {
	if a.Type != v.Type {
		return false
	}
	if isReg(v) && a.Reg == v.Reg {
		return true
	}
	if v.Type == obj.NAME_AUTO || v.Type == obj.NAME_PARAM {
		// TODO(mundaym): is the offset enough here? Node?
		if v.Offset == a.Offset {
			return true
		}
	}
	return false
}

func smallindir(a *obj.Addr, reg *obj.Addr) bool {
	return reg.Type == obj.TYPE_REG &&
		a.Type == obj.TYPE_MEM &&
		a.Reg == reg.Reg &&
		0 <= a.Offset && a.Offset < 4096
}

func stackaddr(a *obj.Addr) bool {
	// TODO(mundaym): the name implies this should check
	// for TYPE_ADDR with a base register REGSP.
	return a.Type == obj.TYPE_REG && a.Reg == s390x.REGSP
}

// isMove returns true if p is a move. Moves may imply
// sign/zero extension.
func isMove(p *obj.Prog) bool {
	switch p.As {
	case s390x.AMOVD,
		s390x.AMOVW, s390x.AMOVWZ,
		s390x.AMOVH, s390x.AMOVHZ,
		s390x.AMOVB, s390x.AMOVBZ,
		s390x.AFMOVD, s390x.AFMOVS:
		return true
	}
	return false
}

// isLoad returns true if p is a move from memory to a register.
func isLoad(p *obj.Prog) bool {
	if !isMove(p) {
		return false
	}
	if !(isGPR(&p.To) || isFPR(&p.To)) {
		return false
	}
	if p.From.Type != obj.TYPE_MEM {
		return false
	}
	return true
}

// isStore returns true if p is a move from a register to memory.
func isStore(p *obj.Prog) bool {
	if !isMove(p) {
		return false
	}
	if !(isGPR(&p.From) || isFPR(&p.From) || isConst(&p.From)) {
		return false
	}
	if p.To.Type != obj.TYPE_MEM {
		return false
	}
	return true
}

// sameStackMem returns true if a and b are both memory operands
// and address the same location which must reside on the stack.
func sameStackMem(a, b *obj.Addr) bool {
	if a.Type != obj.TYPE_MEM ||
		b.Type != obj.TYPE_MEM ||
		a.Name != b.Name ||
		a.Sym != b.Sym ||
		a.Node != b.Node ||
		a.Reg != b.Reg ||
		a.Index != b.Index ||
		a.Offset != b.Offset {
		return false
	}
	switch a.Name {
	case obj.NAME_NONE:
		return a.Reg == s390x.REGSP
	case obj.NAME_PARAM, obj.NAME_AUTO:
		// params and autos are always on the stack
		return true
	}
	return false
}

// removeLoadHitStores trys to remove loads that take place
// immediately after a store to the same location. Returns
// true if load-hit-stores were removed.
//
// For example:
// 	MOVD	R1, 0(R15)
// 	MOVD	0(R15), R2
// Would become:
// 	MOVD	R1, 0(R15)
// 	MOVD	R1, R2
func removeLoadHitStores(r *gc.Flow) int {
	n := 0
	for ; r != nil; r = r.Link {
		p := r.Prog
		if !isStore(p) {
			continue
		}
		for rr := gc.Uniqs(r); rr != nil; rr = gc.Uniqs(rr) {
			pp := rr.Prog
			if gc.Uniqp(rr) == nil {
				break
			}
			if pp.As == obj.ANOP {
				continue
			}
			if isLoad(pp) && sameStackMem(&p.To, &pp.From) {
				if size(p.As) >= size(pp.As) && isGPR(&p.From) == isGPR(&pp.To) {
					pp.From = p.From
				}
			}
			if !isMove(pp) || isStore(pp) {
				break
			}
			if copyau(&p.From, &pp.To) {
				break
			}
		}
	}
	return n
}

// size returns the width of the given move.
func size(as obj.As) int {
	switch as {
	case s390x.AMOVD, s390x.AFMOVD:
		return 8
	case s390x.AMOVW, s390x.AMOVWZ, s390x.AFMOVS:
		return 4
	case s390x.AMOVH, s390x.AMOVHZ:
		return 2
	case s390x.AMOVB, s390x.AMOVBZ:
		return 1
	}
	return -1
}

// castPropagation tries to eliminate unecessary casts.
//
// For example:
// 	MOVHZ	R1, R2     // uint16
//	MOVB	R2, 0(R15) // int8
// Can be simplified to:
//	MOVB	R1, 0(R15)
func castPropagation(r *gc.Flow) int {
	n := 0
	for ; r != nil; r = r.Link {
		p := r.Prog
		if !isMove(p) || !isGPR(&p.To) {
			continue
		}

		// r is a move with a destination register
		var move *gc.Flow
		for rr := gc.Uniqs(r); rr != nil; rr = gc.Uniqs(rr) {
			if gc.Uniqp(rr) == nil {
				// branch target: leave alone
				break
			}
			pp := rr.Prog
			if isMove(pp) && copyas(&pp.From, &p.To) {
				if pp.To.Type == obj.TYPE_MEM {
					if p.From.Type == obj.TYPE_MEM ||
						p.From.Type == obj.TYPE_ADDR {
						break
					}
					if p.From.Type == obj.TYPE_CONST &&
						int64(int16(p.From.Offset)) != p.From.Offset {
						break
					}
				}
				move = rr
				break
			}
			if pp.As == obj.ANOP {
				continue
			}
			break
		}
		if move == nil {
			continue
		}

		// we have a move that reads from our destination reg, check if any future
		// instructions also read from the reg
		mp := move.Prog
		if !copyas(&mp.From, &mp.To) {
			safe := false
			for rr := gc.Uniqs(move); rr != nil; rr = gc.Uniqs(rr) {
				if gc.Uniqp(rr) == nil {
					break
				}
				switch copyu(rr.Prog, &p.To, nil) {
				case _None:
					continue
				case _Write:
					safe = true
				}
				break
			}
			if !safe {
				continue
			}
		}

		// at this point we have something like:
		// MOV* const/mem/reg, reg
		// MOV* reg, reg/mem
		// now check if this is a cast that cannot be forward propagated
		execute := false
		if p.As == mp.As || isZero(&p.From) || size(p.As) == size(mp.As) {
			execute = true
		} else if isGPR(&p.From) && size(p.As) >= size(mp.As) {
			execute = true
		}

		if execute {
			mp.From = p.From
			excise(r)
			n++
		}
	}
	return n
}

// fuseClear merges memory clear operations.
//
// Looks for this pattern (sequence of clears):
// 	MOVD	R0, n(R15)
// 	MOVD	R0, n+8(R15)
// 	MOVD	R0, n+16(R15)
// Replaces with:
//	CLEAR	$24, n(R15)
func fuseClear(r *gc.Flow) int {
	n := 0
	var align int64
	var clear *obj.Prog
	for ; r != nil; r = r.Link {
		// If there is a branch into the instruction stream then
		// we can't fuse into previous instructions.
		if gc.Uniqp(r) == nil {
			clear = nil
		}

		p := r.Prog
		if p.As == obj.ANOP {
			continue
		}
		if p.As == s390x.AXC {
			if p.From.Reg == p.To.Reg && p.From.Offset == p.To.Offset {
				// TODO(mundaym): merge clears?
				p.As = s390x.ACLEAR
				p.From.Offset = p.From3.Offset
				p.From3 = nil
				p.From.Type = obj.TYPE_CONST
				p.From.Reg = 0
				clear = p
			} else {
				clear = nil
			}
			continue
		}

		// Is our source a constant zero?
		if !isZero(&p.From) {
			clear = nil
			continue
		}

		// Are we moving to memory?
		if p.To.Type != obj.TYPE_MEM ||
			p.To.Index != 0 ||
			p.To.Offset >= 4096 ||
			!(p.To.Name == obj.NAME_NONE || p.To.Name == obj.NAME_AUTO || p.To.Name == obj.NAME_PARAM) {
			clear = nil
			continue
		}

		size := int64(0)
		switch p.As {
		default:
			clear = nil
			continue
		case s390x.AMOVB, s390x.AMOVBZ:
			size = 1
		case s390x.AMOVH, s390x.AMOVHZ:
			size = 2
		case s390x.AMOVW, s390x.AMOVWZ:
			size = 4
		case s390x.AMOVD:
			size = 8
		}

		// doubleword aligned clears should be kept doubleword
		// aligned
		if (size == 8 && align != 8) || (size != 8 && align == 8) {
			clear = nil
		}

		if clear != nil &&
			clear.To.Reg == p.To.Reg &&
			clear.To.Name == p.To.Name &&
			clear.To.Node == p.To.Node &&
			clear.To.Sym == p.To.Sym {

			min := clear.To.Offset
			max := clear.To.Offset + clear.From.Offset

			// previous clear is already clearing this region
			if min <= p.To.Offset && max >= p.To.Offset+size {
				excise(r)
				n++
				continue
			}

			// merge forwards
			if max == p.To.Offset {
				clear.From.Offset += size
				excise(r)
				n++
				continue
			}

			// merge backwards
			if min-size == p.To.Offset {
				clear.From.Offset += size
				clear.To.Offset -= size
				excise(r)
				n++
				continue
			}
		}

		// transform into clear
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = size
		p.From.Reg = 0
		p.As = s390x.ACLEAR
		clear = p
		align = size
	}
	return n
}

// fuseMultiple merges memory loads and stores into load multiple and
// store multiple operations.
//
// Looks for this pattern (sequence of loads or stores):
// 	MOVD	R1, 0(R15)
//	MOVD	R2, 8(R15)
//	MOVD	R3, 16(R15)
// Replaces with:
//	STMG	R1, R3, 0(R15)
func fuseMultiple(r *gc.Flow) int {
	n := 0
	var fused *obj.Prog
	for ; r != nil; r = r.Link {
		// If there is a branch into the instruction stream then
		// we can't fuse into previous instructions.
		if gc.Uniqp(r) == nil {
			fused = nil
		}

		p := r.Prog

		isStore := isGPR(&p.From) && isBDMem(&p.To)
		isLoad := isGPR(&p.To) && isBDMem(&p.From)

		// are we a candidate?
		size := int64(0)
		switch p.As {
		default:
			fused = nil
			continue
		case obj.ANOP:
			// skip over nops
			continue
		case s390x.AMOVW, s390x.AMOVWZ:
			size = 4
			// TODO(mundaym): 32-bit load multiple is currently not supported
			// as it requires sign/zero extension.
			if !isStore {
				fused = nil
				continue
			}
		case s390x.AMOVD:
			size = 8
			if !isLoad && !isStore {
				fused = nil
				continue
			}
		}

		// If we merge two loads/stores with different source/destination Nodes
		// then we will lose a reference the second Node which means that the
		// compiler might mark the Node as unused and free its slot on the stack.
		// TODO(mundaym): allow this by adding a dummy reference to the Node.
		if fused == nil ||
			fused.From.Node != p.From.Node ||
			fused.From.Type != p.From.Type ||
			fused.To.Node != p.To.Node ||
			fused.To.Type != p.To.Type {
			fused = p
			continue
		}

		// check two addresses
		ca := func(a, b *obj.Addr, offset int64) bool {
			return a.Reg == b.Reg && a.Offset+offset == b.Offset &&
				a.Sym == b.Sym && a.Name == b.Name
		}

		switch fused.As {
		default:
			fused = p
		case s390x.AMOVW, s390x.AMOVWZ:
			if size == 4 && fused.From.Reg+1 == p.From.Reg && ca(&fused.To, &p.To, 4) {
				fused.As = s390x.ASTMY
				fused.Reg = p.From.Reg
				excise(r)
				n++
			} else {
				fused = p
			}
		case s390x.AMOVD:
			if size == 8 && fused.From.Reg+1 == p.From.Reg && ca(&fused.To, &p.To, 8) {
				fused.As = s390x.ASTMG
				fused.Reg = p.From.Reg
				excise(r)
				n++
			} else if size == 8 && fused.To.Reg+1 == p.To.Reg && ca(&fused.From, &p.From, 8) {
				fused.As = s390x.ALMG
				fused.Reg = fused.To.Reg
				fused.To.Reg = p.To.Reg
				excise(r)
				n++
			} else {
				fused = p
			}
		case s390x.ASTMG, s390x.ASTMY:
			if (fused.As == s390x.ASTMY && size != 4) ||
				(fused.As == s390x.ASTMG && size != 8) {
				fused = p
				continue
			}
			offset := size * int64(fused.Reg-fused.From.Reg+1)
			if fused.Reg+1 == p.From.Reg && ca(&fused.To, &p.To, offset) {
				fused.Reg = p.From.Reg
				excise(r)
				n++
			} else {
				fused = p
			}
		case s390x.ALMG:
			offset := 8 * int64(fused.To.Reg-fused.Reg+1)
			if size == 8 && fused.To.Reg+1 == p.To.Reg && ca(&fused.From, &p.From, offset) {
				fused.To.Reg = p.To.Reg
				excise(r)
				n++
			} else {
				fused = p
			}
		}
	}
	return n
}

// simplifyOps looks for side-effect free ops that can be removed or
// replaced with moves.
//
// For example:
// 	XOR $0, R1 => NOP
//	ADD $0, R1, R2 => MOVD R1, R2
func simplifyOps(r *gc.Flow) int {
	n := 0
	for ; r != nil; r = r.Link {
		p := r.Prog

		// if the target is R0 then this is a required NOP
		if isGPR(&p.To) && p.To.Reg == s390x.REGZERO {
			continue
		}

		switch p.As {
		case s390x.AADD, s390x.ASUB,
			s390x.AOR, s390x.AXOR,
			s390x.ASLW, s390x.ASRW, s390x.ASRAW,
			s390x.ASLD, s390x.ASRD, s390x.ASRAD,
			s390x.ARLL, s390x.ARLLG:
			if isZero(&p.From) && isGPR(&p.To) {
				if p.Reg == 0 || p.Reg == p.To.Reg {
					excise(r)
					n++
				} else {
					p.As = s390x.AMOVD
					p.From.Type = obj.TYPE_REG
					p.From.Reg = p.Reg
					p.Reg = 0
				}
			}
		case s390x.AMULLW, s390x.AAND:
			if isZero(&p.From) && isGPR(&p.To) {
				p.As = s390x.AMOVD
				p.From.Type = obj.TYPE_REG
				p.From.Reg = s390x.REGZERO
				p.Reg = 0
			}
		}
	}
	return n
}

// fuseOpMoves looks for moves following 2-operand operations and trys to merge them into
// a 3-operand operation.
//
// For example:
//	ADD R1, R2
//	MOVD R2, R3
// might become
//	ADD R1, R2, R3
func fuseOpMoves(r *gc.Flow) int {
	n := 0
	for ; r != nil; r = r.Link {
		p := r.Prog
		switch p.As {
		case s390x.AADD:
		case s390x.ASUB:
			if isConst(&p.From) && int64(int16(p.From.Offset)) != p.From.Offset {
				continue
			}
		case s390x.ASLW,
			s390x.ASRW,
			s390x.ASRAW,
			s390x.ASLD,
			s390x.ASRD,
			s390x.ASRAD,
			s390x.ARLL,
			s390x.ARLLG:
			// ok - p.From will be a reg or a constant
		case s390x.AOR,
			s390x.AORN,
			s390x.AAND,
			s390x.AANDN,
			s390x.ANAND,
			s390x.ANOR,
			s390x.AXOR,
			s390x.AMULLW,
			s390x.AMULLD:
			if isConst(&p.From) {
				// these instructions can either use 3 register form
				// or have an immediate but not both
				continue
			}
		default:
			continue
		}

		if p.Reg != 0 && p.Reg != p.To.Reg {
			continue
		}

		var move *gc.Flow
		rr := gc.Uniqs(r)
		for {
			if rr == nil || gc.Uniqp(rr) == nil || rr == r {
				break
			}
			pp := rr.Prog
			switch copyu(pp, &p.To, nil) {
			case _None:
				rr = gc.Uniqs(rr)
				continue
			case _Read:
				if move == nil && pp.As == s390x.AMOVD && isGPR(&pp.From) && isGPR(&pp.To) {
					move = rr
					rr = gc.Uniqs(rr)
					continue
				}
			case _Write:
				if move == nil {
					// dead code
					excise(r)
					n++
				} else {
					for prev := gc.Uniqp(move); prev != r; prev = gc.Uniqp(prev) {
						if copyu(prev.Prog, &move.Prog.To, nil) != 0 {
							move = nil
							break
						}
					}
					if move == nil {
						break
					}
					p.Reg, p.To.Reg = p.To.Reg, move.Prog.To.Reg
					excise(move)
					n++

					// clean up
					if p.From.Reg == p.To.Reg && isCommutative(p.As) {
						p.From.Reg, p.Reg = p.Reg, 0
					}
					if p.To.Reg == p.Reg {
						p.Reg = 0
					}
					// we could try again if p has become a 2-operand op
					// but in testing nothing extra was extracted
				}
			}
			break
		}
	}
	return n
}

// isCommutative returns true if the order of input operands
// does not affect the result. For example:
//	x + y == y + x so ADD is commutative
//	x ^ y == y ^ x so XOR is commutative
func isCommutative(as obj.As) bool {
	switch as {
	case s390x.AADD,
		s390x.AOR,
		s390x.AAND,
		s390x.AXOR,
		s390x.AMULLW,
		s390x.AMULLD:
		return true
	}
	return false
}

// applyCast applies the cast implied by the given move
// instruction to v and returns the result.
func applyCast(cast obj.As, v int64) int64 {
	switch cast {
	case s390x.AMOVWZ:
		return int64(uint32(v))
	case s390x.AMOVHZ:
		return int64(uint16(v))
	case s390x.AMOVBZ:
		return int64(uint8(v))
	case s390x.AMOVW:
		return int64(int32(v))
	case s390x.AMOVH:
		return int64(int16(v))
	case s390x.AMOVB:
		return int64(int8(v))
	}
	return v
}

// constantPropagation removes redundant constant copies.
func constantPropagation(r *gc.Flow) int {
	n := 0
	// find MOV $con,R followed by
	// another MOV $con,R without
	// setting R in the interim
	for ; r != nil; r = r.Link {
		p := r.Prog
		if isMove(p) {
			if !isReg(&p.To) {
				continue
			}
			if !isConst(&p.From) {
				continue
			}
		} else {
			continue
		}

		rr := r
		for {
			rr = gc.Uniqs(rr)
			if rr == nil || rr == r {
				break
			}
			if gc.Uniqp(rr) == nil {
				break
			}

			pp := rr.Prog
			t := copyu(pp, &p.To, nil)
			switch t {
			case _None:
				continue
			case _Read:
				if !isGPR(&pp.From) || !isMove(pp) {
					continue
				}
				if p.From.Type == obj.TYPE_CONST {
					v := applyCast(p.As, p.From.Offset)
					if isGPR(&pp.To) {
						if int64(int32(v)) == v || ((v>>32)<<32) == v {
							pp.From.Reg = 0
							pp.From.Offset = v
							pp.From.Type = obj.TYPE_CONST
							n++
						}
					} else if int64(int16(v)) == v {
						pp.From.Reg = 0
						pp.From.Offset = v
						pp.From.Type = obj.TYPE_CONST
						n++
					}
				}
				continue
			case _Write:
				if p.As != pp.As || p.From.Type != pp.From.Type {
					break
				}
				if p.From.Type == obj.TYPE_CONST && p.From.Offset == pp.From.Offset {
					excise(rr)
					n++
					continue
				} else if p.From.Type == obj.TYPE_FCONST {
					if p.From.Val.(float64) == pp.From.Val.(float64) {
						excise(rr)
						n++
						continue
					}
				}
			}
			break
		}
	}
	return n
}

// copyPropagation tries to eliminate register-to-register moves.
func copyPropagation(r *gc.Flow) int {
	n := 0
	for ; r != nil; r = r.Link {
		p := r.Prog
		if isMove(p) && isReg(&p.To) {
			// Convert uses to $0 to uses of R0 and
			// propagate R0
			if isGPR(&p.To) && isZero(&p.From) {
				p.From.Type = obj.TYPE_REG
				p.From.Reg = s390x.REGZERO
			}

			// Try to eliminate reg->reg moves
			if isGPR(&p.From) || isFPR(&p.From) {
				if copyprop(r) || (subprop(r) && copyprop(r)) {
					excise(r)
					n++
				}
			}
		}
	}
	return n
}

// loadPipelining pushes any load from memory as early as possible.
func loadPipelining(r *gc.Flow) int {
	for ; r != nil; r = r.Link {
		p := r.Prog
		if isLoad(p) {
			pushback(r)
		}
	}
	return 0
}

// fuseCompareBranch finds comparisons followed by a branch and converts
// them into a compare-and-branch instruction (which avoid setting the
// condition code).
func fuseCompareBranch(r *gc.Flow) int {
	n := 0
	for ; r != nil; r = r.Link {
		p := r.Prog
		r1 := gc.Uniqs(r)
		if r1 == nil {
			continue
		}
		p1 := r1.Prog

		var ins obj.As
		switch p.As {
		case s390x.ACMP:
			switch p1.As {
			case s390x.ABCL, s390x.ABC:
				continue
			case s390x.ABEQ:
				ins = s390x.ACMPBEQ
			case s390x.ABGE:
				ins = s390x.ACMPBGE
			case s390x.ABGT:
				ins = s390x.ACMPBGT
			case s390x.ABLE:
				ins = s390x.ACMPBLE
			case s390x.ABLT:
				ins = s390x.ACMPBLT
			case s390x.ABNE:
				ins = s390x.ACMPBNE
			default:
				continue
			}

		case s390x.ACMPU:
			switch p1.As {
			case s390x.ABCL, s390x.ABC:
				continue
			case s390x.ABEQ:
				ins = s390x.ACMPUBEQ
			case s390x.ABGE:
				ins = s390x.ACMPUBGE
			case s390x.ABGT:
				ins = s390x.ACMPUBGT
			case s390x.ABLE:
				ins = s390x.ACMPUBLE
			case s390x.ABLT:
				ins = s390x.ACMPUBLT
			case s390x.ABNE:
				ins = s390x.ACMPUBNE
			default:
				continue
			}

		case s390x.ACMPW, s390x.ACMPWU:
			continue

		default:
			continue
		}

		if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
			fmt.Printf("cnb %v; %v  ", p, p1)
		}

		if p1.To.Sym != nil {
			continue
		}

		if p.To.Type == obj.TYPE_REG {
			p1.As = ins
			p1.From = p.From
			p1.Reg = p.To.Reg
			p1.From3 = nil
		} else if p.To.Type == obj.TYPE_CONST {
			switch p.As {
			case s390x.ACMP, s390x.ACMPW:
				if (p.To.Offset < -(1 << 7)) || (p.To.Offset >= ((1 << 7) - 1)) {
					continue
				}
			case s390x.ACMPU, s390x.ACMPWU:
				if p.To.Offset >= (1 << 8) {
					continue
				}
			default:
			}
			p1.As = ins
			p1.From = p.From
			p1.Reg = 0
			p1.From3 = new(obj.Addr)
			*(p1.From3) = p.To
		} else {
			continue
		}

		if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
			fmt.Printf("%v\n", p1)
		}
		excise(r)
		n++
	}
	return n
}

// deadCodeElimination removes writes to registers which are written
// to again before they are next read.
func deadCodeElimination(r *gc.Flow) int {
	n := 0
	for ; r != nil; r = r.Link {
		p := r.Prog
		// Currently there are no instructions which write to multiple
		// registers in copyu. This check will need to change if there
		// ever are.
		if !(isGPR(&p.To) || isFPR(&p.To)) || copyu(p, &p.To, nil) != _Write {
			continue
		}
		for rr := gc.Uniqs(r); rr != nil; rr = gc.Uniqs(rr) {
			t := copyu(rr.Prog, &p.To, nil)
			if t == _None {
				continue
			}
			if t == _Write {
				excise(r)
				n++
			}
			break
		}
	}
	return n
}
