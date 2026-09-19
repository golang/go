// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package s390x

import (
	"math"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/s390x"
	"internal/abi"
)

// ssaMarkMoves marks any MOVXconst ops that need to avoid clobbering flags.
func ssaMarkMoves(s *ssagen.State, b *ssa.Block) {
	flive := b.FlagsLiveAtEnd
	for _, c := range b.ControlValues() {
		flive = c.Type.IsFlags() || flive
	}
	for i := len(b.Values) - 1; i >= 0; i-- {
		v := b.Values[i]
		if flive && v.Op == ssaop.OpS390XMOVDconst {
			// The "mark" is any non-nil Aux value.
			v.Aux = ssa.AuxMark
		}
		if v.Type.IsFlags() {
			flive = false
		}
		for _, a := range v.Args {
			if a.Type.IsFlags() {
				flive = true
			}
		}
	}
}

// loadByType returns the load instruction of the given type.
func loadByType(t *types.Type) obj.As {
	if t.IsFloat() {
		switch t.Size() {
		case 4:
			return s390x.AFMOVS
		case 8:
			return s390x.AFMOVD
		}
	} else {
		switch t.Size() {
		case 1:
			if t.IsSigned() {
				return s390x.AMOVB
			} else {
				return s390x.AMOVBZ
			}
		case 2:
			if t.IsSigned() {
				return s390x.AMOVH
			} else {
				return s390x.AMOVHZ
			}
		case 4:
			if t.IsSigned() {
				return s390x.AMOVW
			} else {
				return s390x.AMOVWZ
			}
		case 8:
			return s390x.AMOVD
		}
	}
	panic("bad load type")
}

// storeByType returns the store instruction of the given type.
func storeByType(t *types.Type) obj.As {
	width := t.Size()
	if t.IsFloat() {
		switch width {
		case 4:
			return s390x.AFMOVS
		case 8:
			return s390x.AFMOVD
		}
	} else {
		switch width {
		case 1:
			return s390x.AMOVB
		case 2:
			return s390x.AMOVH
		case 4:
			return s390x.AMOVW
		case 8:
			return s390x.AMOVD
		}
	}
	panic("bad store type")
}

// moveByType returns the reg->reg move instruction of the given type.
func moveByType(t *types.Type) obj.As {
	if t.IsFloat() {
		return s390x.AFMOVD
	} else {
		switch t.Size() {
		case 1:
			if t.IsSigned() {
				return s390x.AMOVB
			} else {
				return s390x.AMOVBZ
			}
		case 2:
			if t.IsSigned() {
				return s390x.AMOVH
			} else {
				return s390x.AMOVHZ
			}
		case 4:
			if t.IsSigned() {
				return s390x.AMOVW
			} else {
				return s390x.AMOVWZ
			}
		case 8:
			return s390x.AMOVD
		}
	}
	panic("bad move type")
}

// opregreg emits instructions for
//
//	dest := dest(To) op src(From)
//
// and also returns the created obj.Prog so it
// may be further adjusted (offset, scale, etc).
func opregreg(s *ssagen.State, op obj.As, dest, src int16) *obj.Prog {
	p := s.Prog(op)
	p.From.Type = obj.TYPE_REG
	p.To.Type = obj.TYPE_REG
	p.To.Reg = dest
	p.From.Reg = src
	return p
}

// opregregimm emits instructions for
//
//	dest := src(From) op off
//
// and also returns the created obj.Prog so it
// may be further adjusted (offset, scale, etc).
func opregregimm(s *ssagen.State, op obj.As, dest, src int16, off int64) *obj.Prog {
	p := s.Prog(op)
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = off
	p.Reg = src
	p.To.Reg = dest
	p.To.Type = obj.TYPE_REG
	return p
}

func ssaGenValue(s *ssagen.State, v *ssa.Value) {
	switch v.Op {
	case ssaop.OpS390XSLD, ssaop.OpS390XSLW,
		ssaop.OpS390XSRD, ssaop.OpS390XSRW,
		ssaop.OpS390XSRAD, ssaop.OpS390XSRAW,
		ssaop.OpS390XRLLG, ssaop.OpS390XRLL:
		r := v.Reg()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		if r2 == s390x.REG_R0 {
			v.Fatalf("cannot use R0 as shift value %s", v.LongString())
		}
		p := opregreg(s, v.Op.Asm(), r, r2)
		if r != r1 {
			p.Reg = r1
		}
	case ssaop.OpS390XRXSBG:
		r2 := v.Args[1].Reg()
		i := v.Aux.(s390x.RotateParams)
		p := s.Prog(v.Op.Asm())
		p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: int64(i.Start)}
		p.AddRestSourceArgs([]obj.Addr{
			{Type: obj.TYPE_CONST, Offset: int64(i.End)},
			{Type: obj.TYPE_CONST, Offset: int64(i.Amount)},
			{Type: obj.TYPE_REG, Reg: r2},
		})
		p.To = obj.Addr{Type: obj.TYPE_REG, Reg: v.Reg()}
	case ssaop.OpS390XRISBGZ:
		r1 := v.Reg()
		r2 := v.Args[0].Reg()
		i := v.Aux.(s390x.RotateParams)
		p := s.Prog(v.Op.Asm())
		p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: int64(i.Start)}
		p.AddRestSourceArgs([]obj.Addr{
			{Type: obj.TYPE_CONST, Offset: int64(i.End)},
			{Type: obj.TYPE_CONST, Offset: int64(i.Amount)},
			{Type: obj.TYPE_REG, Reg: r2},
		})
		p.To = obj.Addr{Type: obj.TYPE_REG, Reg: r1}
	case ssaop.OpS390XADD, ssaop.OpS390XADDW,
		ssaop.OpS390XSUB, ssaop.OpS390XSUBW,
		ssaop.OpS390XAND, ssaop.OpS390XANDW,
		ssaop.OpS390XOR, ssaop.OpS390XORW,
		ssaop.OpS390XXOR, ssaop.OpS390XXORW:
		r := v.Reg()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		p := opregreg(s, v.Op.Asm(), r, r2)
		if r != r1 {
			p.Reg = r1
		}
	case ssaop.OpS390XADDC:
		r1 := v.Reg0()
		r2 := v.Args[0].Reg()
		r3 := v.Args[1].Reg()
		if r1 == r2 {
			r2, r3 = r3, r2
		}
		p := opregreg(s, v.Op.Asm(), r1, r2)
		if r3 != r1 {
			p.Reg = r3
		}
	case ssaop.OpS390XSUBC:
		r1 := v.Reg0()
		r2 := v.Args[0].Reg()
		r3 := v.Args[1].Reg()
		p := opregreg(s, v.Op.Asm(), r1, r3)
		if r1 != r2 {
			p.Reg = r2
		}
	case ssaop.OpS390XADDE, ssaop.OpS390XSUBE:
		r2 := v.Args[1].Reg()
		opregreg(s, v.Op.Asm(), v.Reg0(), r2)
	case ssaop.OpS390XADDCconst:
		r1 := v.Reg0()
		r3 := v.Args[0].Reg()
		i2 := int64(int16(v.AuxInt))
		opregregimm(s, v.Op.Asm(), r1, r3, i2)
	// 2-address opcode arithmetic
	case ssaop.OpS390XMULLD, ssaop.OpS390XMULLW,
		ssaop.OpS390XMULHD, ssaop.OpS390XMULHDU,
		ssaop.OpS390XFMULS, ssaop.OpS390XFMUL, ssaop.OpS390XFDIVS, ssaop.OpS390XFDIV:
		opregreg(s, v.Op.Asm(), v.Reg(), v.Args[1].Reg())
	case ssaop.OpS390XFSUBS, ssaop.OpS390XFSUB,
		ssaop.OpS390XFADDS, ssaop.OpS390XFADD:
		opregreg(s, v.Op.Asm(), v.Reg0(), v.Args[1].Reg())
	case ssaop.OpS390XMLGR:
		// MLGR Rx R3 -> R2:R3
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		if r1 != s390x.REG_R3 {
			v.Fatalf("We require the multiplcand to be stored in R3 for MLGR %s", v.LongString())
		}
		p := s.Prog(s390x.AMLGR)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r0
		p.To.Reg = s390x.REG_R2
		p.To.Type = obj.TYPE_REG
	case ssaop.OpS390XFMADD, ssaop.OpS390XFMADDS,
		ssaop.OpS390XFMSUB, ssaop.OpS390XFMSUBS:
		r1 := v.Args[1].Reg()
		r2 := v.Args[2].Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r1
		p.Reg = r2
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpS390XFIDBR:
		switch v.AuxInt {
		case 0, 1, 3, 4, 5, 6, 7:
			opregregimm(s, v.Op.Asm(), v.Reg(), v.Args[0].Reg(), v.AuxInt)
		default:
			v.Fatalf("invalid FIDBR mask: %v", v.AuxInt)
		}
	case ssaop.OpS390XCPSDR:
		p := opregreg(s, v.Op.Asm(), v.Reg(), v.Args[1].Reg())
		p.Reg = v.Args[0].Reg()
	case ssaop.OpS390XWFMAXDB, ssaop.OpS390XWFMAXSB,
		ssaop.OpS390XWFMINDB, ssaop.OpS390XWFMINSB:
		p := opregregimm(s, v.Op.Asm(), v.Reg(), v.Args[0].Reg(), 1 /* Java Math.Max() */)
		p.AddRestSource(obj.Addr{Type: obj.TYPE_REG, Reg: v.Args[1].Reg()})
	case ssaop.OpS390XDIVD, ssaop.OpS390XDIVW,
		ssaop.OpS390XDIVDU, ssaop.OpS390XDIVWU,
		ssaop.OpS390XMODD, ssaop.OpS390XMODW,
		ssaop.OpS390XMODDU, ssaop.OpS390XMODWU:

		// TODO(mundaym): use the temp registers every time like x86 does with AX?
		dividend := v.Args[0].Reg()
		divisor := v.Args[1].Reg()

		// CPU faults upon signed overflow, which occurs when most
		// negative int is divided by -1.
		var j *obj.Prog
		if v.Op == ssaop.OpS390XDIVD || v.Op == ssaop.OpS390XDIVW ||
			v.Op == ssaop.OpS390XMODD || v.Op == ssaop.OpS390XMODW {

			var c *obj.Prog
			c = s.Prog(s390x.ACMP)
			j = s.Prog(s390x.ABEQ)

			c.From.Type = obj.TYPE_REG
			c.From.Reg = divisor
			c.To.Type = obj.TYPE_CONST
			c.To.Offset = -1

			j.To.Type = obj.TYPE_BRANCH

		}

		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = divisor
		p.Reg = 0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = dividend

		// signed division, rest of the check for -1 case
		if j != nil {
			j2 := s.Prog(s390x.ABR)
			j2.To.Type = obj.TYPE_BRANCH

			var n *obj.Prog
			if v.Op == ssaop.OpS390XDIVD || v.Op == ssaop.OpS390XDIVW {
				// n * -1 = -n
				n = s.Prog(s390x.ANEG)
				n.To.Type = obj.TYPE_REG
				n.To.Reg = dividend
			} else {
				// n % -1 == 0
				n = s.Prog(s390x.AXOR)
				n.From.Type = obj.TYPE_REG
				n.From.Reg = dividend
				n.To.Type = obj.TYPE_REG
				n.To.Reg = dividend
			}

			j.To.SetTarget(n)
			j2.To.SetTarget(s.Pc())
		}
	case ssaop.OpS390XADDconst, ssaop.OpS390XADDWconst:
		opregregimm(s, v.Op.Asm(), v.Reg(), v.Args[0].Reg(), v.AuxInt)
	case ssaop.OpS390XMULLDconst, ssaop.OpS390XMULLWconst,
		ssaop.OpS390XSUBconst, ssaop.OpS390XSUBWconst,
		ssaop.OpS390XANDconst, ssaop.OpS390XANDWconst,
		ssaop.OpS390XORconst, ssaop.OpS390XORWconst,
		ssaop.OpS390XXORconst, ssaop.OpS390XXORWconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpS390XSLDconst, ssaop.OpS390XSLWconst,
		ssaop.OpS390XSRDconst, ssaop.OpS390XSRWconst,
		ssaop.OpS390XSRADconst, ssaop.OpS390XSRAWconst,
		ssaop.OpS390XRLLconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		r := v.Reg()
		r1 := v.Args[0].Reg()
		if r != r1 {
			p.Reg = r1
		}
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssaop.OpS390XMOVDaddridx:
		r := v.Args[0].Reg()
		i := v.Args[1].Reg()
		p := s.Prog(s390x.AMOVD)
		p.From.Scale = 1
		if i == s390x.REGSP {
			r, i = i, r
		}
		p.From.Type = obj.TYPE_ADDR
		p.From.Reg = r
		p.From.Index = i
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpS390XMOVDaddr:
		p := s.Prog(s390x.AMOVD)
		p.From.Type = obj.TYPE_ADDR
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpS390XCMP, ssaop.OpS390XCMPW, ssaop.OpS390XCMPU, ssaop.OpS390XCMPWU:
		opregreg(s, v.Op.Asm(), v.Args[1].Reg(), v.Args[0].Reg())
	case ssaop.OpS390XFCMPS, ssaop.OpS390XFCMP:
		opregreg(s, v.Op.Asm(), v.Args[1].Reg(), v.Args[0].Reg())
	case ssaop.OpS390XCMPconst, ssaop.OpS390XCMPWconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = v.AuxInt
	case ssaop.OpS390XCMPUconst, ssaop.OpS390XCMPWUconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = int64(uint32(v.AuxInt))
	case ssaop.OpS390XMOVDconst:
		x := v.Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = x
	case ssaop.OpS390XFMOVSconst, ssaop.OpS390XFMOVDconst:
		x := v.Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = x
	case ssaop.OpS390XADDWload, ssaop.OpS390XADDload,
		ssaop.OpS390XMULLWload, ssaop.OpS390XMULLDload,
		ssaop.OpS390XSUBWload, ssaop.OpS390XSUBload,
		ssaop.OpS390XANDWload, ssaop.OpS390XANDload,
		ssaop.OpS390XORWload, ssaop.OpS390XORload,
		ssaop.OpS390XXORWload, ssaop.OpS390XXORload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[1].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpS390XMOVDload,
		ssaop.OpS390XMOVWZload, ssaop.OpS390XMOVHZload, ssaop.OpS390XMOVBZload,
		ssaop.OpS390XMOVDBRload, ssaop.OpS390XMOVWBRload, ssaop.OpS390XMOVHBRload,
		ssaop.OpS390XMOVBload, ssaop.OpS390XMOVHload, ssaop.OpS390XMOVWload,
		ssaop.OpS390XFMOVSload, ssaop.OpS390XFMOVDload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpS390XMOVBZloadidx, ssaop.OpS390XMOVHZloadidx, ssaop.OpS390XMOVWZloadidx,
		ssaop.OpS390XMOVBloadidx, ssaop.OpS390XMOVHloadidx, ssaop.OpS390XMOVWloadidx, ssaop.OpS390XMOVDloadidx,
		ssaop.OpS390XMOVHBRloadidx, ssaop.OpS390XMOVWBRloadidx, ssaop.OpS390XMOVDBRloadidx,
		ssaop.OpS390XFMOVSloadidx, ssaop.OpS390XFMOVDloadidx:
		r := v.Args[0].Reg()
		i := v.Args[1].Reg()
		if i == s390x.REGSP {
			r, i = i, r
		}
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = r
		p.From.Scale = 1
		p.From.Index = i
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpS390XMOVBstore, ssaop.OpS390XMOVHstore, ssaop.OpS390XMOVWstore, ssaop.OpS390XMOVDstore,
		ssaop.OpS390XMOVHBRstore, ssaop.OpS390XMOVWBRstore, ssaop.OpS390XMOVDBRstore,
		ssaop.OpS390XFMOVSstore, ssaop.OpS390XFMOVDstore:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssaop.OpS390XMOVBstoreidx, ssaop.OpS390XMOVHstoreidx, ssaop.OpS390XMOVWstoreidx, ssaop.OpS390XMOVDstoreidx,
		ssaop.OpS390XMOVHBRstoreidx, ssaop.OpS390XMOVWBRstoreidx, ssaop.OpS390XMOVDBRstoreidx,
		ssaop.OpS390XFMOVSstoreidx, ssaop.OpS390XFMOVDstoreidx:
		r := v.Args[0].Reg()
		i := v.Args[1].Reg()
		if i == s390x.REGSP {
			r, i = i, r
		}
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = r
		p.To.Scale = 1
		p.To.Index = i
		ssagen.AddAux(&p.To, v)
	case ssaop.OpS390XMOVDstoreconst, ssaop.OpS390XMOVWstoreconst, ssaop.OpS390XMOVHstoreconst, ssaop.OpS390XMOVBstoreconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		sc := v.AuxValAndOff()
		p.From.Offset = sc.Val64()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux2(&p.To, v, sc.Off64())
	case ssaop.OpS390XMOVBreg, ssaop.OpS390XMOVHreg, ssaop.OpS390XMOVWreg,
		ssaop.OpS390XMOVBZreg, ssaop.OpS390XMOVHZreg, ssaop.OpS390XMOVWZreg,
		ssaop.OpS390XLDGR, ssaop.OpS390XLGDR,
		ssaop.OpS390XCEFBRA, ssaop.OpS390XCDFBRA, ssaop.OpS390XCEGBRA, ssaop.OpS390XCDGBRA,
		ssaop.OpS390XCFEBRA, ssaop.OpS390XCFDBRA, ssaop.OpS390XCGEBRA, ssaop.OpS390XCGDBRA,
		ssaop.OpS390XCELFBR, ssaop.OpS390XCDLFBR, ssaop.OpS390XCELGBR, ssaop.OpS390XCDLGBR,
		ssaop.OpS390XCLFEBR, ssaop.OpS390XCLFDBR, ssaop.OpS390XCLGEBR, ssaop.OpS390XCLGDBR,
		ssaop.OpS390XLDEBR, ssaop.OpS390XLEDBR,
		ssaop.OpS390XFNEG, ssaop.OpS390XFNEGS,
		ssaop.OpS390XLPDFR, ssaop.OpS390XLNDFR:
		opregreg(s, v.Op.Asm(), v.Reg(), v.Args[0].Reg())
	case ssaop.OpS390XCLEAR:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		sc := v.AuxValAndOff()
		p.From.Offset = sc.Val64()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux2(&p.To, v, sc.Off64())
	case ssaop.OpCopy:
		if v.Type.IsMemory() {
			return
		}
		x := v.Args[0].Reg()
		y := v.Reg()
		if x != y {
			opregreg(s, moveByType(v.Type), y, x)
		}
	case ssaop.OpLoadReg:
		if v.Type.IsFlags() {
			v.Fatalf("load flags not implemented: %v", v.LongString())
			return
		}
		p := s.Prog(loadByType(v.Type))
		ssagen.AddrAuto(&p.From, v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpStoreReg:
		if v.Type.IsFlags() {
			v.Fatalf("store flags not implemented: %v", v.LongString())
			return
		}
		p := s.Prog(storeByType(v.Type))
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddrAuto(&p.To, v)
	case ssaop.OpArgIntReg, ssaop.OpArgFloatReg:
		// The assembler needs to wrap the entry safepoint/stack growth code with spill/unspill
		// The loop only runs once.
		for _, a := range v.Block.Func.RegArgs {
			// Pass the spill/unspill information along to the assembler, offset by size of
			// the saved LR slot.
			addr := ssagen.SpillSlotAddr(a, s390x.REGSP, base.Ctxt.Arch.FixedFrameSize)
			s.FuncInfo().AddSpill(
				obj.RegSpill{Reg: a.Reg, Addr: addr, Unspill: loadByType(a.Type), Spill: storeByType(a.Type)})
		}
		v.Block.Func.RegArgs = nil

		ssagen.CheckArgReg(v)
	case ssaop.OpS390XLoweredGetClosurePtr:
		// Closure pointer is R12 (already)
		ssagen.CheckLoweredGetClosurePtr(v)
	case ssaop.OpS390XLoweredRound32F, ssaop.OpS390XLoweredRound64F:
		// input is already rounded
	case ssaop.OpS390XLoweredGetG:
		r := v.Reg()
		p := s.Prog(s390x.AMOVD)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = s390x.REGG
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssaop.OpS390XLoweredGetCallerSP:
		// caller's SP is FixedFrameSize below the address of the first arg
		p := s.Prog(s390x.AMOVD)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = -base.Ctxt.Arch.FixedFrameSize
		p.From.Name = obj.NAME_PARAM
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpS390XLoweredGetCallerPC:
		p := s.Prog(obj.AGETCALLERPC)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpS390XCALLstatic, ssaop.OpS390XCALLclosure, ssaop.OpS390XCALLinter:
		s.Call(v)
	case ssaop.OpS390XCALLtail, ssaop.OpS390XCALLtailinter:
		s.TailCall(v)
	case ssaop.OpS390XLoweredWB:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		// AuxInt encodes how many buffer entries we need.
		p.To.Sym = ir.Syms.GCWriteBarrier[v.AuxInt-1]

	case ssaop.OpS390XLoweredPanicBoundsRR, ssaop.OpS390XLoweredPanicBoundsRC, ssaop.OpS390XLoweredPanicBoundsCR, ssaop.OpS390XLoweredPanicBoundsCC:
		// Compute the constant we put in the PCData entry for this call.
		code, signed := ssa.BoundsKind(v.AuxInt).Code()
		xIsReg := false
		yIsReg := false
		xVal := 0
		yVal := 0
		switch v.Op {
		case ssaop.OpS390XLoweredPanicBoundsRR:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - s390x.REG_R0)
			yIsReg = true
			yVal = int(v.Args[1].Reg() - s390x.REG_R0)
		case ssaop.OpS390XLoweredPanicBoundsRC:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - s390x.REG_R0)
			c := v.Aux.(ssa.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				yIsReg = true
				if yVal == xVal {
					yVal = 1
				}
				p := s.Prog(s390x.AMOVD)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = s390x.REG_R0 + int16(yVal)
			}
		case ssaop.OpS390XLoweredPanicBoundsCR:
			yIsReg = true
			yVal = int(v.Args[0].Reg() - s390x.REG_R0)
			c := v.Aux.(ssa.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				xVal = int(c)
			} else {
				// Move constant to a register
				if xVal == yVal {
					xVal = 1
				}
				p := s.Prog(s390x.AMOVD)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = s390x.REG_R0 + int16(xVal)
			}
		case ssaop.OpS390XLoweredPanicBoundsCC:
			c := v.Aux.(ssa.PanicBoundsCC).Cx
			if c >= 0 && c <= abi.BoundsMaxConst {
				xVal = int(c)
			} else {
				// Move constant to a register
				xIsReg = true
				p := s.Prog(s390x.AMOVD)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = s390x.REG_R0 + int16(xVal)
			}
			c = v.Aux.(ssa.PanicBoundsCC).Cy
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				yIsReg = true
				yVal = 1
				p := s.Prog(s390x.AMOVD)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = s390x.REG_R0 + int16(yVal)
			}
		}
		c := abi.BoundsEncode(code, signed, xIsReg, yIsReg, xVal, yVal)

		p := s.Prog(obj.APCDATA)
		p.From.SetConst(abi.PCDATA_PanicBounds)
		p.To.SetConst(int64(c))
		p = s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ir.Syms.PanicBounds

	case ssaop.OpS390XFLOGR, ssaop.OpS390XPOPCNT,
		ssaop.OpS390XNEG, ssaop.OpS390XNEGW,
		ssaop.OpS390XMOVWBR, ssaop.OpS390XMOVDBR:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpS390XNOT, ssaop.OpS390XNOTW:
		v.Fatalf("NOT/NOTW generated %s", v.LongString())
	case ssaop.OpS390XSumBytes2, ssaop.OpS390XSumBytes4, ssaop.OpS390XSumBytes8:
		v.Fatalf("SumBytes generated %s", v.LongString())
	case ssaop.OpS390XLOCGR:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(v.Aux.(s390x.CCMask))
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpS390XFSQRTS, ssaop.OpS390XFSQRT:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpS390XLTDBR, ssaop.OpS390XLTEBR:
		opregreg(s, v.Op.Asm(), v.Args[0].Reg(), v.Args[0].Reg())
	case ssaop.OpS390XInvertFlags:
		v.Fatalf("InvertFlags should never make it to codegen %v", v.LongString())
	case ssaop.OpS390XFlagEQ, ssaop.OpS390XFlagLT, ssaop.OpS390XFlagGT, ssaop.OpS390XFlagOV:
		v.Fatalf("Flag* ops should never make it to codegen %v", v.LongString())
	case ssaop.OpS390XAddTupleFirst32, ssaop.OpS390XAddTupleFirst64:
		v.Fatalf("AddTupleFirst* should never make it to codegen %v", v.LongString())
	case ssaop.OpS390XLoweredNilCheck:
		// Issue a load which will fault if the input is nil.
		p := s.Prog(s390x.AMOVBZ)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = s390x.REGTMP
		if logopt.Enabled() {
			logopt.LogOpt(v.Pos, "nilcheck", "genssa", v.Block.Func.Name)
		}
		if base.Debug.Nil != 0 && v.Pos.Line() > 1 { // v.Pos.Line()==1 in generated wrappers
			base.WarnfAt(v.Pos, "generated nil check")
		}
	case ssaop.OpS390XMVC:
		vo := v.AuxValAndOff()
		p := s.Prog(s390x.AMVC)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = vo.Val64()
		p.AddRestSource(obj.Addr{
			Type:   obj.TYPE_MEM,
			Reg:    v.Args[1].Reg(),
			Offset: vo.Off64(),
		})
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.To.Offset = vo.Off64()
	case ssaop.OpS390XSTMG2, ssaop.OpS390XSTMG3, ssaop.OpS390XSTMG4,
		ssaop.OpS390XSTM2, ssaop.OpS390XSTM3, ssaop.OpS390XSTM4:
		for i := 2; i < len(v.Args)-1; i++ {
			if v.Args[i].Reg() != v.Args[i-1].Reg()+1 {
				v.Fatalf("invalid store multiple %s", v.LongString())
			}
		}
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.Reg = v.Args[len(v.Args)-2].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssaop.OpS390XLoweredMove:
		// Inputs must be valid pointers to memory,
		// so adjust arg0 and arg1 as part of the expansion.
		// arg2 should be src+size,
		//
		// mvc: MVC  $256, 0(R2), 0(R1)
		//      MOVD $256(R1), R1
		//      MOVD $256(R2), R2
		//      CMP  R2, Rarg2
		//      BNE  mvc
		//      MVC  $rem, 0(R2), 0(R1) // if rem > 0
		// arg2 is the last address to move in the loop + 256
		mvc := s.Prog(s390x.AMVC)
		mvc.From.Type = obj.TYPE_CONST
		mvc.From.Offset = 256
		mvc.AddRestSource(obj.Addr{Type: obj.TYPE_MEM, Reg: v.Args[1].Reg()})
		mvc.To.Type = obj.TYPE_MEM
		mvc.To.Reg = v.Args[0].Reg()

		for i := 0; i < 2; i++ {
			movd := s.Prog(s390x.AMOVD)
			movd.From.Type = obj.TYPE_ADDR
			movd.From.Reg = v.Args[i].Reg()
			movd.From.Offset = 256
			movd.To.Type = obj.TYPE_REG
			movd.To.Reg = v.Args[i].Reg()
		}

		cmpu := s.Prog(s390x.ACMPU)
		cmpu.From.Reg = v.Args[1].Reg()
		cmpu.From.Type = obj.TYPE_REG
		cmpu.To.Reg = v.Args[2].Reg()
		cmpu.To.Type = obj.TYPE_REG

		bne := s.Prog(s390x.ABLT)
		bne.To.Type = obj.TYPE_BRANCH
		bne.To.SetTarget(mvc)

		if v.AuxInt > 0 {
			mvc := s.Prog(s390x.AMVC)
			mvc.From.Type = obj.TYPE_CONST
			mvc.From.Offset = v.AuxInt
			mvc.AddRestSource(obj.Addr{Type: obj.TYPE_MEM, Reg: v.Args[1].Reg()})
			mvc.To.Type = obj.TYPE_MEM
			mvc.To.Reg = v.Args[0].Reg()
		}
	case ssaop.OpS390XLoweredZero:
		// Input must be valid pointers to memory,
		// so adjust arg0 as part of the expansion.
		// arg1 should be src+size,
		//
		// clear: CLEAR $256, 0(R1)
		//        MOVD  $256(R1), R1
		//        CMP   R1, Rarg1
		//        BNE   clear
		//        CLEAR $rem, 0(R1) // if rem > 0
		// arg1 is the last address to zero in the loop + 256
		clear := s.Prog(s390x.ACLEAR)
		clear.From.Type = obj.TYPE_CONST
		clear.From.Offset = 256
		clear.To.Type = obj.TYPE_MEM
		clear.To.Reg = v.Args[0].Reg()

		movd := s.Prog(s390x.AMOVD)
		movd.From.Type = obj.TYPE_ADDR
		movd.From.Reg = v.Args[0].Reg()
		movd.From.Offset = 256
		movd.To.Type = obj.TYPE_REG
		movd.To.Reg = v.Args[0].Reg()

		cmpu := s.Prog(s390x.ACMPU)
		cmpu.From.Reg = v.Args[0].Reg()
		cmpu.From.Type = obj.TYPE_REG
		cmpu.To.Reg = v.Args[1].Reg()
		cmpu.To.Type = obj.TYPE_REG

		bne := s.Prog(s390x.ABLT)
		bne.To.Type = obj.TYPE_BRANCH
		bne.To.SetTarget(clear)

		if v.AuxInt > 0 {
			clear := s.Prog(s390x.ACLEAR)
			clear.From.Type = obj.TYPE_CONST
			clear.From.Offset = v.AuxInt
			clear.To.Type = obj.TYPE_MEM
			clear.To.Reg = v.Args[0].Reg()
		}
	case ssaop.OpS390XMOVBZatomicload, ssaop.OpS390XMOVWZatomicload, ssaop.OpS390XMOVDatomicload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
	case ssaop.OpS390XMOVBatomicstore, ssaop.OpS390XMOVWatomicstore, ssaop.OpS390XMOVDatomicstore:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssaop.OpS390XLAN, ssaop.OpS390XLAO:
		// LA(N|O) Ry, TMP, 0(Rx)
		op := s.Prog(v.Op.Asm())
		op.From.Type = obj.TYPE_REG
		op.From.Reg = v.Args[1].Reg()
		op.Reg = s390x.REGTMP
		op.To.Type = obj.TYPE_MEM
		op.To.Reg = v.Args[0].Reg()
	case ssaop.OpS390XLANfloor, ssaop.OpS390XLAOfloor:
		r := v.Args[0].Reg() // clobbered, assumed R1 in comments

		// Round ptr down to nearest multiple of 4.
		// ANDW $~3, R1
		ptr := s.Prog(s390x.AANDW)
		ptr.From.Type = obj.TYPE_CONST
		ptr.From.Offset = 0xfffffffc
		ptr.To.Type = obj.TYPE_REG
		ptr.To.Reg = r

		// Redirect output of LA(N|O) into R1 since it is clobbered anyway.
		// LA(N|O) Rx, R1, 0(R1)
		op := s.Prog(v.Op.Asm())
		op.From.Type = obj.TYPE_REG
		op.From.Reg = v.Args[1].Reg()
		op.Reg = r
		op.To.Type = obj.TYPE_MEM
		op.To.Reg = r
	case ssaop.OpS390XLAA, ssaop.OpS390XLAAG:
		p := s.Prog(v.Op.Asm())
		p.Reg = v.Reg0()
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssaop.OpS390XLoweredAtomicCas32, ssaop.OpS390XLoweredAtomicCas64:
		// Convert the flags output of CS{,G} into a bool.
		//    CS{,G} arg1, arg2, arg0
		//    MOVD   $0, ret
		//    BNE    2(PC)
		//    MOVD   $1, ret
		//    NOP (so the BNE has somewhere to land)

		// CS{,G} arg1, arg2, arg0
		cs := s.Prog(v.Op.Asm())
		cs.From.Type = obj.TYPE_REG
		cs.From.Reg = v.Args[1].Reg() // old
		cs.Reg = v.Args[2].Reg()      // new
		cs.To.Type = obj.TYPE_MEM
		cs.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&cs.To, v)

		// MOVD $0, ret
		movd := s.Prog(s390x.AMOVD)
		movd.From.Type = obj.TYPE_CONST
		movd.From.Offset = 0
		movd.To.Type = obj.TYPE_REG
		movd.To.Reg = v.Reg0()

		// BNE 2(PC)
		bne := s.Prog(s390x.ABNE)
		bne.To.Type = obj.TYPE_BRANCH

		// MOVD $1, ret
		movd = s.Prog(s390x.AMOVD)
		movd.From.Type = obj.TYPE_CONST
		movd.From.Offset = 1
		movd.To.Type = obj.TYPE_REG
		movd.To.Reg = v.Reg0()

		// NOP (so the BNE has somewhere to land)
		nop := s.Prog(obj.ANOP)
		bne.To.SetTarget(nop)
	case ssaop.OpS390XLoweredAtomicExchange32, ssaop.OpS390XLoweredAtomicExchange64:
		// Loop until the CS{,G} succeeds.
		//     MOV{WZ,D} arg0, ret
		// cs: CS{,G}    ret, arg1, arg0
		//     BNE       cs

		// MOV{WZ,D} arg0, ret
		load := s.Prog(loadByType(v.Type.FieldType(0)))
		load.From.Type = obj.TYPE_MEM
		load.From.Reg = v.Args[0].Reg()
		load.To.Type = obj.TYPE_REG
		load.To.Reg = v.Reg0()
		ssagen.AddAux(&load.From, v)

		// CS{,G} ret, arg1, arg0
		cs := s.Prog(v.Op.Asm())
		cs.From.Type = obj.TYPE_REG
		cs.From.Reg = v.Reg0()   // old
		cs.Reg = v.Args[1].Reg() // new
		cs.To.Type = obj.TYPE_MEM
		cs.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&cs.To, v)

		// BNE cs
		bne := s.Prog(s390x.ABNE)
		bne.To.Type = obj.TYPE_BRANCH
		bne.To.SetTarget(cs)
	case ssaop.OpS390XSYNC:
		s.Prog(s390x.ASYNC)
	case ssaop.OpClobber, ssaop.OpClobberReg:
		// TODO: implement for clobberdead experiment. Nop is ok for now.
	default:
		v.Fatalf("genValue not implemented: %s", v.LongString())
	}
}

func blockAsm(b *ssa.Block) obj.As {
	switch b.Kind {
	case block.BlockS390XBRC:
		return s390x.ABRC
	case block.BlockS390XCRJ:
		return s390x.ACRJ
	case block.BlockS390XCGRJ:
		return s390x.ACGRJ
	case block.BlockS390XCLRJ:
		return s390x.ACLRJ
	case block.BlockS390XCLGRJ:
		return s390x.ACLGRJ
	case block.BlockS390XCIJ:
		return s390x.ACIJ
	case block.BlockS390XCGIJ:
		return s390x.ACGIJ
	case block.BlockS390XCLIJ:
		return s390x.ACLIJ
	case block.BlockS390XCLGIJ:
		return s390x.ACLGIJ
	}
	b.Fatalf("blockAsm not implemented: %s", b.LongString())
	panic("unreachable")
}

func ssaGenBlock(s *ssagen.State, b, next *ssa.Block) {
	// Handle generic blocks first.
	switch b.Kind {
	case block.BlockPlain, block.BlockDefer:
		if b.Succs[0].Block() != next {
			p := s.Prog(s390x.ABR)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, ssagen.Branch{P: p, B: b.Succs[0].Block()})
		}
		return
	case block.BlockExit, block.BlockRetJmp:
		return
	case block.BlockRet:
		s.Prog(obj.ARET)
		return
	}

	// Handle s390x-specific blocks. These blocks all have a
	// condition code mask in the Aux value and 2 successors.
	succs := [...]*ssa.Block{b.Succs[0].Block(), b.Succs[1].Block()}
	mask := b.Aux.(s390x.CCMask)

	// TODO: take into account Likely property for forward/backward
	// branches. We currently can't do this because we don't know
	// whether a block has already been emitted. In general forward
	// branches are assumed 'not taken' and backward branches are
	// assumed 'taken'.
	if next == succs[0] {
		succs[0], succs[1] = succs[1], succs[0]
		mask = mask.Inverse()
	}

	p := s.Br(blockAsm(b), succs[0])
	switch b.Kind {
	case block.BlockS390XBRC:
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(mask)
	case block.BlockS390XCGRJ, block.BlockS390XCRJ,
		block.BlockS390XCLGRJ, block.BlockS390XCLRJ:
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(mask & s390x.NotUnordered) // unordered is not possible
		p.Reg = b.Controls[0].Reg()
		p.AddRestSourceReg(b.Controls[1].Reg())
	case block.BlockS390XCGIJ, block.BlockS390XCIJ:
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(mask & s390x.NotUnordered) // unordered is not possible
		p.Reg = b.Controls[0].Reg()
		p.AddRestSourceConst(int64(int8(b.AuxInt)))
	case block.BlockS390XCLGIJ, block.BlockS390XCLIJ:
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(mask & s390x.NotUnordered) // unordered is not possible
		p.Reg = b.Controls[0].Reg()
		p.AddRestSourceConst(int64(uint8(b.AuxInt)))
	default:
		b.Fatalf("branch not implemented: %s", b.LongString())
	}
	if next != succs[1] {
		s.Br(s390x.ABR, succs[1])
	}
}

func loadRegResult(s *ssagen.State, f *ssa.Func, t *types.Type, reg int16, n *ir.Name, off int64) *obj.Prog {
	p := s.Prog(loadByType(t))
	p.From.Type = obj.TYPE_MEM
	p.From.Name = obj.NAME_AUTO
	p.From.Sym = n.Linksym()
	p.From.Offset = n.FrameOffset() + off
	p.To.Type = obj.TYPE_REG
	p.To.Reg = reg
	return p
}

func spillArgReg(pp *objw.Progs, p *obj.Prog, f *ssa.Func, t *types.Type, reg int16, n *ir.Name, off int64) *obj.Prog {
	p = pp.Append(p, storeByType(t), obj.TYPE_REG, reg, 0, obj.TYPE_MEM, 0, n.FrameOffset()+off)
	p.To.Name = obj.NAME_PARAM
	p.To.Sym = n.Linksym()
	p.Pos = p.Pos.WithNotStmt()
	return p
}
