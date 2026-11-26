// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is the SVE decoder's runtime: the part of it that is written by hand.
//
// An SVE instruction is decoded by walking sveForms, a table of encodings, and
// turning each operand of the first form that matches into an Arg with the
// operand's descriptor. Everything that is specific to one instruction is data
// in the tables declared in sve_gen.go; this file has one case per descriptor
// knob and one case per operand kind, so it does not grow with the number of
// instructions.
//
// The tables are the contract between this runtime and whatever produces them,
// and their declarations say what each one holds.
//
// A word is declined, not guessed at, when any operand of a form fails to
// resolve: a field outside its range, a value with no row in its table, or a
// register number the file does not have. Declining lets a later, less
// specific form claim the word, and otherwise leaves the word undecodable.

package arm64asm

import (
	"fmt"
	"strconv"
	"strings"
)

// ---------------------------------------------------------------------------
// The generated data's shapes.
// ---------------------------------------------------------------------------

// sveField is one contiguous piece of an encoded value.
type sveField struct{ lo, width int }

// sveRow is one value-table row. Don't-care bits in the ARM table become mask
// zeros, so a row matches when the gathered field value & mask == value.
type sveRow struct {
	mask, value uint32
	sym         uint16
}

// role says what an element contributes to its operand.
type role uint8

const (
	roleSkip role = iota
	roleReg
	roleArng
	roleQual
	roleIndex
	roleImm
	roleOff
	roleMod
	roleAmount
	roleVL
	rolePrfop
)

// regfile is the register file a roleReg descriptor names.
type regfile uint8

const (
	rfNone regfile = iota
	rfZ
	rfP
	rfPN
	rfR32
	rfR64
	rfRauto
	rfV
)

// sveDesc is the descriptor of one part of an operand. Its knobs are the facts
// that ARM's operand descriptions state about a field: which bits make it up
// (fields, low part first), whether it is signed, its range, a stride and bias
// applied to the raw value, a modulus, the register file it names, whether
// register 31 is the stack pointer, an optional "LSL #8" alternative, and a
// table that maps field values to symbols. The remaining fields cover the
// readings that the description gives in prose: an 8-bit floating-point
// immediate, a value relative to the element size, a logical immediate, an
// explicit SIMD&FP register width, and a value that is a constant minus the
// field.
type sveDesc struct {
	role     role
	fields   []sveField // low-to-high significance
	signed   bool
	hasRange bool
	min, max int64
	stride   int64
	bias     int64
	modulo   int64
	regfile  regfile
	sp       bool // "or stack pointer": field 31 spells SP
	altBit   int  // the "LSL #8" alternative's bit, or 0 when unused
	altShift uint
	fixed    int16 // a pinned symbol index, or -1
	table    []sveRow
	fp8      bool    // the field is an 8-bit floating-point immediate
	tsz      tszMode // how the element size enters the value
	bitmask  bool    // the field is a DecodeBitMasks logical immediate
	vwidth   int64   // an explicitly stated SIMD&FP register width, or 0
	invert   int64   // value = invert - field, when ARM's ASL inverts
}

// tszMode names the three families whose value is stated relative to the element
// size. ARM spells each one out in prose ("the number of bits per element", "one
// less than the number of elements in 128 bits") and names the field composite
// that carries it, so the only thing the decoder has to supply is the size.
type tszMode uint8

const (
	tszNone tszMode = iota
	tszIndex
	tszShiftR
	tszShiftL
)

// sveOpBase is the Op of the first SVE mnemonic. It sits above every Op that
// tables.go defines, so Op.String reaches the SVE names only for an Op that the
// stock table does not know.
const sveOpBase Op = 512

// sveReject, if non-nil, reports that the architecture declares w an invalid
// instance of a form even though the form's mask, value and descriptors accept
// it. It is called for the forms whose reject field is not zero, with that
// field, which names the constraint. Decode then declines w, without trying the
// later forms. It is a hook so that a set of such constraints can be added
// without changing the runtime.
var sveReject func(class int, w uint32) bool

// operand kinds. One rendering case each; nothing per instruction.
type kind uint8

const (
	kArng kind = iota
	kReg
	kPredQual
	kImm
	kArngIdx
	kRegIdx
	kList
	kListRange
	kSpecial
	kMemOff
	kMemMulVL
	kMemVecBase
	kMemExt
	kMemVecIndex
)

type sveArg struct {
	kind  kind
	descs []uint16
}

type sveForm struct {
	mask, value uint32
	op          uint16
	args        uint16
	gpr         int16  // 32 or 64 when the form pins its <R> width, else 0
	reject      uint16 // the constraint on the word that sveReject checks, or 0
}

// ---------------------------------------------------------------------------
// Knob interpretation: gather the bits, then apply the arithmetic.
// ---------------------------------------------------------------------------

func (d *sveDesc) totalWidth() int {
	n := 0
	for _, f := range d.fields {
		n += f.width
	}
	return n
}

// gather assembles the descriptor's pieces into one value, low piece first.
func (d *sveDesc) gather(w uint32) uint32 {
	var n uint32
	sh := uint(0)
	for _, f := range d.fields {
		n |= ((w >> uint(f.lo)) & (1<<uint(f.width) - 1)) << sh
		sh += uint(f.width)
	}
	return n
}

// value applies the descriptor's arithmetic in the encoder's inverse order:
// sign, then stride and bias, then modulo, then the LSL #8 alternative, then the
// range check. Out of range is a decode failure, not a clamped answer.
func (d *sveDesc) value(w uint32) (int64, bool) {
	n := int64(d.gather(w))
	if d.signed {
		if tw := d.totalWidth(); n&(1<<uint(tw-1)) != 0 {
			n -= 1 << uint(tw)
		}
	}
	stride := d.stride
	if stride == 0 {
		stride = 1
	}
	if d.invert != 0 {
		// One family of shifts is specified as "shift = esize - UInt(imm3)", where
		// the operand description states only the range.
		v := d.invert - n
		if d.hasRange && (v < d.min || v > d.max) {
			return 0, false
		}
		return v, true
	}
	v := n*stride + d.bias
	if d.modulo > 0 {
		v = ((v % d.modulo) + d.modulo) % d.modulo
	}
	// The range applies to the field's own value, and the optional shift scales it
	// afterwards. Skipping the check when altBit is set -- which is the obvious way
	// to keep the shifted value from failing it -- lets a whole descriptor through
	// unvalidated.
	if d.hasRange && (v < d.min || v > d.max) {
		return 0, false
	}
	if d.altBit > 0 && w&(1<<uint(d.altBit)) != 0 {
		v <<= d.altShift
	}
	return v, true
}

// esizeValue applies the three element-size-relative readings.
//
// The element size comes from the form's own arrangement, which a value table
// already decodes, so the readings need no data of their own. Decoding is easier
// here than encoding, where an operand's encoder is handed that operand's value
// and cannot see the element size; a decoder holds the whole word.
func (d *sveDesc) esizeValue(w uint32, esize int64) (int64, bool) {
	if esize == 0 {
		return 0, false
	}
	n := int64(d.gather(w))
	switch d.tsz {
	case tszIndex:
		// The composite's low part is the size-selecting field itself, so the
		// index is what remains above it.
		sh := uint(0)
		for e := esize / 8; e > 1; e >>= 1 {
			sh++
		}
		return n >> (sh + 1), true
	case tszShiftR:
		v := 2*esize - n
		if v < 1 || v > esize {
			return 0, false
		}
		return v, true
	case tszShiftL:
		v := n - esize
		if v < 0 || v >= esize {
			return 0, false
		}
		return v, true
	}
	return 0, false
}

// decodeBitMasks is ASL's DecodeBitMasks for the imm13 logical immediate of AND,
// ORR, EOR and DUPM. It returns one lane of the mask, or false for a reserved
// encoding.
func decodeBitMasks(imm13 uint32) (uint64, bool) {
	n := imm13 >> 12 & 1
	immr := imm13 >> 6 & 0x3f
	imms := imm13 & 0x3f
	// len = HighestSetBit(N:NOT(imms)): 6 when N is set, else the position of the
	// highest zero in imms. len == 0 is the reserved encoding.
	length := -1
	if n == 1 {
		length = 6
	} else {
		for i := 5; i >= 0; i-- {
			if imms>>uint(i)&1 == 0 {
				length = i
				break
			}
		}
	}
	if length < 1 {
		return 0, false
	}
	esize := uint(1) << uint(length)
	lv := uint32(esize) - 1
	if imms&lv == lv {
		return 0, false
	}
	sb := imms & lv
	r := uint(immr & lv)
	// The element is sb+1 set bits rotated right by r within esize bits. The
	// result is one lane of the operand, which is what the Go assembler takes:
	// $7 on .H, not 0x0007000700070007. A lane is at least a byte, so a 2- or
	// 4-bit element is repeated to fill it: 0x05800f80 is
	// and z0.b, z0.b, #0xaa, from the element 0b10.
	mask := ^uint64(0)
	if esize < 64 {
		mask = 1<<esize - 1
	}
	welem := uint64(1)<<uint(sb+1) - 1
	rot := welem >> r
	if r != 0 {
		rot |= welem << (esize - r)
	}
	rot &= mask
	for ; esize < 8; esize *= 2 {
		rot |= rot << esize
	}
	return rot, true
}

// symbol resolves a value-table-bearing descriptor to the spelling ARM's own
// table names. A field value with no row is a decode failure: tables omit the
// RESERVED rows, so a missing row means the encoding is not architecturally
// valid and declining is the right answer.
func (d *sveDesc) symbol(w uint32) (string, bool) {
	if d.fixed >= 0 {
		return sveSyms[d.fixed], true
	}
	f := d.gather(w)
	for _, r := range d.table {
		if f&r.mask == r.value {
			return sveSyms[r.sym], true
		}
	}
	return "", false
}

// reg builds the register the descriptor names.
func (d *sveDesc) reg(w uint32, esize, gprWidth int64) (Reg, bool) {
	v, ok := d.value(w)
	if !ok || v < 0 {
		return 0, false
	}
	switch d.regfile {
	case rfRauto:
		if v > 31 {
			return 0, false
		}
		if gprWidth == 64 || (gprWidth == 0 && esize == 64) {
			return X0 + Reg(v), true
		}
		return W0 + Reg(v), true
	case rfZ:
		if v > 31 {
			return 0, false
		}
		return Z0 + Reg(v), true
	case rfP:
		if v > 15 {
			return 0, false
		}
		return P0 + Reg(v), true
	case rfPN:
		if v > 15 {
			return 0, false
		}
		return PN0 + Reg(v), true
	case rfV:
		if v > 31 {
			return 0, false
		}
		// ARM names a SIMD&FP scalar by its width -- b12, h12, s12, d12, q12 --
		// while the Go assembler writes V12 for all of them. Building the
		// width-specific register makes GNU syntax right, and goReg folds the
		// whole B0..Q31 range back to V for Go syntax.
		width := esize
		if d.vwidth != 0 {
			width = d.vwidth
		}
		switch width {
		case 8:
			return B0 + Reg(v), true
		case 16:
			return H0 + Reg(v), true
		case 32:
			return S0 + Reg(v), true
		case 64:
			return D0 + Reg(v), true
		case 128:
			return Q0 + Reg(v), true
		}
		return V0 + Reg(v), true
	case rfR32:
		if v > 31 {
			return 0, false
		}
		return W0 + Reg(v), true
	case rfR64:
		if v > 31 {
			return 0, false
		}
		return X0 + Reg(v), true
	}
	return 0, false
}

// ---------------------------------------------------------------------------
// Symbol interpretation. Each of these is one small closed domain, read out of
// the ARM value tables rather than authored here.
// ---------------------------------------------------------------------------

// sveArrangement returns the arrangement that s spells: an element size, a
// predication qualifier, or one of the four NEON arrangements that SVE forms
// name.
func sveArrangement(s string) (Arrangement, bool) {
	switch s {
	case "B":
		return ArrangementB, true
	case "H":
		return ArrangementH, true
	case "S":
		return ArrangementS, true
	case "D":
		return ArrangementD, true
	case "Q":
		return ArrangementQ, true
	case "M":
		return ArrangementM, true
	case "Z":
		return ArrangementZ, true
	case "16B":
		return Arrangement16B, true
	case "8H":
		return Arrangement8H, true
	case "4S":
		return Arrangement4S, true
	case "2D":
		return Arrangement2D, true
	}
	return 0, false
}

func sveExtShift(s string) (ExtShift, bool) {
	switch s {
	case "LSL":
		return lsl, true
	case "UXTW":
		return uxtw, true
	case "SXTW":
		return sxtw, true
	case "[absent]":
		return ExtShift(0), true
	}
	return 0, false
}

// ---------------------------------------------------------------------------
// Operand assembly. One case per kind.
//
// Every case reads the same three things off its descriptors -- the registers,
// the one arrangement, and the one numeric or symbolic extra -- so the shared
// work is done once in sveParts and each case only says how those combine into
// an Arg.
// ---------------------------------------------------------------------------

// sveParts is the resolved content of one operand.
type sveParts struct {
	regs     []Reg
	arrs     []Arrangement // one per register that pinned or named an arrangement
	arng     Arrangement
	qual     Arrangement
	num      int64
	hasN     bool
	sym      string
	mod      ExtShift
	amt      uint8
	prf      uint8
	hasP     bool
	fp       Imm_fp
	hasFP    bool
	unsigned bool
	sp       bool // some register in this operand may be the stack pointer
}

// sveResolve reads an operand's descriptors. It fails rather than guessing:
// every descriptor must resolve, because a descriptor that cannot be read means
// the word is not a valid instance of this form.
func sveResolve(a *sveArg, w uint32, esize, gprWidth int64) (sveParts, bool) {
	var p sveParts
	for _, di := range a.descs {
		d := &sveDescs[di]
		switch d.role {
		case roleReg:
			r, ok := d.reg(w, esize, gprWidth)
			if !ok {
				return p, false
			}
			p.regs = append(p.regs, r)
			p.sp = p.sp || d.sp
		case roleArng:
			s, ok := d.symbol(w)
			if !ok {
				return p, false
			}
			arr, ok := sveArrangement(s)
			if !ok {
				return p, false
			}
			// A list pins one arrangement per member; they always agree, so the
			// last write is the operand's arrangement and the per-register slice
			// is only needed to keep the counts aligned.
			p.arng = arr
			p.arrs = append(p.arrs, arr)
		case roleQual:
			s, ok := d.symbol(w)
			if !ok {
				return p, false
			}
			q, ok := sveArrangement(s)
			if !ok {
				return p, false
			}
			p.qual = q
		case roleMod:
			s, ok := d.symbol(w)
			if !ok {
				return p, false
			}
			m, ok := sveExtShift(s)
			if !ok {
				return p, false
			}
			p.mod = m
		case roleAmount:
			s, ok := d.symbol(w)
			if !ok {
				return p, false
			}
			if s == "[absent]" {
				break
			}
			var n int
			if _, err := fmt.Sscanf(s, "#%d", &n); err != nil {
				return p, false
			}
			p.amt = uint8(n)
		case roleVL:
			s, ok := d.symbol(w)
			if !ok {
				return p, false
			}
			p.sym = s
		case rolePrfop:
			p.prf, p.hasP = uint8(d.gather(w)), true
			if s, ok := d.symbol(w); ok && s == "#uimm4" {
				// ARM's table gives three quarters of the field named operations and
				// spells the rest as a plain 4-bit immediate. Widening those into
				// Imm_prfop prints #0x17 where every other disassembler prints #15.
				p.hasP = false
				p.num, p.hasN = int64(d.gather(w)), true
			}
		case roleIndex, roleImm, roleOff:
			if d.tsz != tszNone {
				v, ok := d.esizeValue(w, esize)
				if !ok {
					return p, false
				}
				p.num, p.hasN = v, true
				break
			}
			if d.bitmask {
				v, ok := decodeBitMasks(d.gather(w))
				if !ok {
					return p, false
				}
				// Unsigned: the element pattern 0xffc0000000001fff is not
				// -18014398509473793, and the Go assembler writes these positive.
				p.num, p.hasN, p.unsigned = int64(v), true, true
				break
			}
			if d.fp8 {
				// Imm_fp represents this exactly. The exponent is not a plain biased
				// field: ARM's VFPExpandImm builds it as NOT(imm8[6]) : imm8[5:4],
				// and reading it as imm8[6:4]-3 gets six of the eight values wrong.
				// This is the same expression decode.go uses for FMOV.
				f := d.gather(w)
				exp := (1 - (f >> 6 & 1)) << 2
				exp += f >> 4 & 3
				p.fp = Imm_fp{uint8(f >> 7 & 1), int8(exp) - 3, uint8(f & 15)}
				p.hasFP = true
				break
			}
			if len(d.table) > 0 || d.fixed >= 0 {
				s, ok := d.symbol(w)
				if !ok {
					return p, false
				}
				p.sym = s
				break
			}
			v, ok := d.value(w)
			if !ok {
				return p, false
			}
			p.num, p.hasN = v, true
		default:
			// A role this runtime does not know, or roleSkip, which no table
			// row has. Resolving the operand without it would accept a word
			// that the table did not describe.
			return p, false
		}
	}
	return p, true
}

// sveBuild turns one resolved operand into an Arg.
func sveBuild(a *sveArg, w uint32, esize, gprWidth int64) (Arg, bool) {
	p, ok := sveResolve(a, w, esize, gprWidth)
	if !ok {
		return nil, false
	}
	switch a.kind {
	case kArng:
		if len(p.regs) != 1 || p.arng == 0 {
			return nil, false
		}
		return RegisterWithArrangement{asVector(p.regs[0]), p.arng, 0}, true

	case kReg:
		if len(p.regs) != 1 {
			return nil, false
		}
		// RegSP means "a GPR where 31 is the stack pointer" and Reg means "a GPR
		// where 31 is the zero register". The prose says which, so this is the one
		// place the ZR-versus-SP spelling is decided, and it is decided from data
		// rather than per instruction.
		if p.sp {
			return RegSP(p.regs[0]), true
		}
		return p.regs[0], true

	case kPredQual:
		if len(p.regs) != 1 || p.qual == 0 {
			return nil, false
		}
		return RegisterWithArrangement{p.regs[0], p.qual, 0}, true

	case kImm:
		if p.hasFP {
			return p.fp, true
		}
		if p.sym != "" {
			return sveImmSym(p.sym)
		}
		if !p.hasN {
			return nil, false
		}
		if p.unsigned {
			// A logical-immediate bit pattern. Decimal=false makes the GNU spelling
			// hex, which is what GNU and LLVM print, and tells the Go printer
			// not to read the top bit as a sign.
			return Imm64{Imm: uint64(p.num), Decimal: false}, true
		}
		if p.num < 0 {
			return ImmSigned(p.num), true
		}
		return Imm64{Imm: uint64(p.num), Decimal: true}, true

	case kArngIdx:
		if len(p.regs) != 1 || p.arng == 0 || !p.hasN {
			return nil, false
		}
		return RegisterWithArrangementAndIndex{asVector(p.regs[0]), p.arng, uint8(p.num), 0}, true

	case kRegIdx:
		if len(p.regs) != 1 || !p.hasN {
			return nil, false
		}
		return RegisterWithArrangementAndIndex{p.regs[0], 0, uint8(p.num), 0}, true

	case kList, kListRange:
		if len(p.regs) == 0 || p.arng == 0 {
			return nil, false
		}
		// ARM describes a multi-vector group by naming its first and last member
		// only, so the count is their distance and not the number of descriptors.
		// Taking len(p.regs) instead silently turns every four-register group into
		// a two-register one.
		size := fileSize(p.regs[0])
		cnt := uint8((uint16(p.regs[len(p.regs)-1])-uint16(p.regs[0]))%size + 1)
		if a.kind == kListRange {
			cnt |= sveRangeBit
		}
		return RegisterWithArrangement{asVector(p.regs[0]), p.arng, cnt}, true

	case kSpecial:
		if p.hasP {
			// SVE's prfop field is four bits and Imm_prfop's encoding is five: the
			// base PRFM type field is two bits (PLD/PLI/PST) where SVE's is one
			// (PLD/PST), so the type bit is widened by one; without that every
			// PST spelling comes out as PLI.
			return Imm_prfop(p.prf>>3&1<<4 | p.prf&7), true
		}
		if strings.HasPrefix(p.sym, "VLx") {
			if p.sym == "VLx2" {
				return VLx2, true
			}
			return VLx4, true
		}
		if p.hasN {
			return Imm64{Imm: uint64(p.num), Decimal: true}, true
		}
		return nil, false

	case kMemOff:
		if len(p.regs) != 1 {
			return nil, false
		}
		return MemImmediate{RegSP(p.regs[0]), AddrOffset, int32(p.num)}, true

	case kMemMulVL:
		if len(p.regs) != 1 {
			return nil, false
		}
		return MemScaledVL{RegSP(p.regs[0]), int32(p.num)}, true

	case kMemVecBase:
		if len(p.regs) != 1 || p.arng == 0 {
			return nil, false
		}
		return MemVectorBase{p.regs[0], p.arng, int32(p.num)}, true

	case kMemExt:
		if len(p.regs) != 2 {
			return nil, false
		}
		mod := p.mod
		if mod == ExtShift(0) {
			// MemExtend.String prints "[x25,x4,]" for a zero ExtShift, because it
			// tests for lsl rather than for absence. lsl with amount 0 is the same
			// addressing mode and prints "[x25, x4]".
			mod = lsl
		}
		return MemExtend{RegSP(p.regs[0]), p.regs[1], mod, p.amt, false}, true

	case kMemVecIndex:
		if len(p.regs) != 2 {
			return nil, false
		}
		ba, ia := Arrangement(0), Arrangement(0)
		// The base's arrangement is pinned by the base element and the index's by
		// the index element, in descriptor order.
		if len(p.arrs) == 2 {
			ba, ia = p.arrs[0], p.arrs[1]
		} else if len(p.arrs) == 1 {
			// Only the vector half carries one.
			if isGPR(p.regs[0]) {
				ia = p.arrs[0]
			} else {
				ba = p.arrs[0]
			}
		}
		return MemVectorIndex{p.regs[0], ba, p.regs[1], ia, p.mod, p.amt}, true
	}
	return nil, false
}

func isGPR(r Reg) bool { return r <= XZR }

// asVector folds a width-specific SIMD&FP name back to the V file.
//
// ARM names a *scalar* SIMD&FP register by its width -- "clasta d12" -- but names
// the same file "v5.4s" when the operand carries an arrangement, which the SVE
// quadword reductions do (ADDQV, FMAXQV and friends write "v5.4s"). Keeping the
// width there produces "s5.4s", which is nonsense and which both objdump and
// llvm-mc disagree with; the width is already in the arrangement.
func asVector(r Reg) Reg {
	if B0 <= r && r <= Q31 {
		return V0 + Reg(uint16(r)&31)
	}
	return r
}

// sveImmSym spells the immediates the ARM tables name: the SVE floating-point
// set {0.0, 0.5, 1.0, 2.0} and the FCMLA/FCADD rotations {0, 90, 180, 270}.
func sveImmSym(s string) (Arg, bool) {
	t := strings.TrimPrefix(s, "#")
	if !strings.Contains(t, ".") {
		n, err := strconv.ParseInt(t, 10, 64)
		if err != nil {
			return nil, false
		}
		return Imm64{Imm: uint64(n), Decimal: true}, true
	}
	return sveFPImm(t)
}

// sveFPImm is the one place a symbol becomes a number. The whole SVE set is
// {0.0, 0.5, 1.0, 2.0}, and Imm_fp's {sign, exp, pre} triple represents each
// exactly.
func sveFPImm(t string) (Arg, bool) {
	switch t {
	case "0.0":
		return sveZeroFP{}, true
	case "0.5":
		return Imm_fp{0, -1, 0}, true
	case "1.0":
		return Imm_fp{0, 0, 0}, true
	case "2.0":
		return Imm_fp{0, 1, 0}, true
	}
	return nil, false
}

// sveZeroFP is the literal #0.0 operand. Imm_fp's representation cannot express
// zero -- its value is always pre/16 * 2^exp, which is never 0 -- and the
// encoding carries no bits for it, so the operand is a constant.
type sveZeroFP struct{}

func (sveZeroFP) isArg()         {}
func (sveZeroFP) String() string { return "#0.0" }

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

// sveDecode decodes w as an SVE instruction. Forms are ordered
// most-specific-mask-first, so the first match whose operands all resolve is the
// right one; a form whose operands do not resolve is not a match, because the
// descriptors' range and table checks are part of the encoding's definition.
func sveDecode(w uint32) (Inst, bool) {
	// No form matches a word whose bits 31:25 are outside the gate, so skip the
	// scan for it. Without this check every word that the stock tables reject
	// is compared with every form.
	if g := w >> 25; g < 64 && sveGateLo>>g&1 == 0 || g >= 64 && sveGateHi>>(g-64)&1 == 0 {
		return Inst{}, false
	}
	return sveScan(w)
}

// sveScan tries the forms in order, most specific mask first.
func sveScan(w uint32) (Inst, bool) {
	for i := range sveForms {
		f := &sveForms[i]
		if w&f.mask != f.value {
			continue
		}
		// A word that the decode pseudocode of its encoding declares invalid is
		// UNDEFINED, so no later form may decode it.
		if f.reject != 0 && sveReject != nil && sveReject(int(f.reject), w) {
			return Inst{}, false
		}
		esize := sveESize(f, w)
		var args Args
		ok := true
		for j, ai := range sveArgLists[f.args] {
			a, good := sveBuild(&sveArgs[ai], w, esize, int64(f.gpr))
			if !good {
				ok = false
				break
			}
			args[j] = a
		}
		if !ok {
			continue
		}
		return Inst{Op: sveOpBase + Op(f.op), Args: args, Enc: w}, true
	}
	return Inst{}, false
}

// sveESize returns the form's element size in bits, read off the arrangements its
// operands decode to.
//
// It takes the *smallest*. A widening shift names two arrangements from the same
// tsz field -- SSHLLB is "<Zd>.<T>, <Zn>.<Tb>, #<const>" with T one step wider
// than Tb -- and ARM's "number of bits per element" means the source, which is
// the narrower. Taking the first instead takes the destination, and the shift
// then falls outside its range and the form declines.
func sveESize(f *sveForm, w uint32) int64 {
	var best int64
	for _, ai := range sveArgLists[f.args] {
		for _, di := range sveArgs[ai].descs {
			d := &sveDescs[di]
			if d.role != roleArng {
				continue
			}
			s, ok := d.symbol(w)
			if !ok {
				continue
			}
			if n := arngESize(s); n != 0 && (best == 0 || n < best) {
				best = n
			}
		}
	}
	return best
}

// arngESize is the element size an arrangement names. ARM writes the SVE ones as
// a bare letter and the NEON ones with a lane count in front.
func arngESize(s string) int64 {
	for len(s) > 0 && s[0] >= '0' && s[0] <= '9' {
		s = s[1:]
	}
	switch s {
	case "B":
		return 8
	case "H":
		return 16
	case "S":
		return 32
	case "D":
		return 64
	case "Q":
		return 128
	}
	return 0
}

// sveIsOp reports whether op is one of the generated SVE opcodes.
func sveIsOp(op Op) bool {
	return op >= sveOpBase && int(op-sveOpBase) < len(sveOpARM)
}

// sveOpString is the added case in Op.String. It returns the ARM manual
// mnemonic, which is arm64asm's convention and what GNUSyntax needs.
func sveOpString(op Op) string {
	if sveIsOp(op) {
		return sveOpARM[op-sveOpBase]
	}
	return ""
}
