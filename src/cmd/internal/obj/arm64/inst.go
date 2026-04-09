// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"cmd/internal/obj"
	"fmt"
	"iter"
	"math"
	"math/bits"
)

// instEncoder represents an instruction encoder.
type instEncoder struct {
	goOp      obj.As    // Go opcode mnemonic
	fixedBits uint32    // Known bits
	args      []operand // Operands, in Go order
}

type varBits struct {
	// The low and high bit index in the binary encoding, exclusive on hi
	lo, hi  int
	encoded bool // If true then its value is already encoded
	bits    uint32
}

// component is the component of an binary encoding.
// e.g. for operand <Zda>.<T>, <T>'s encoding function might be described as:
//
//	For the "Byte and halfword" variant: is the size specifier,
//	sz	<T>
//	0	B
//	1	H
//	bit range mappings:
//	sz: [22:23)
//
// Then sz is the component of the binary encoding.
type component uint16

type elemEncoder struct {
	fn func(uint32) (uint32, bool)
	// comp is the component of the binary encoding.
	comp component
}

// operand is the operand type of an instruction.
type operand struct {
	class AClass // Operand class, register, constant, memory operation etc.
	// The elements that this operand includes, this only includes the encoding-related parts
	// They are represented as a list of pointers to the encoding functions.
	// The first returned value is the encoded binary, the second is the ok signal.
	// The encoding functions return the ok signal for deduplication purposes:
	// For example:
	//	SDOT  <Zda>.<T>, <Zn>.<Tb>, <Zm>.<Tb>
	//	SDOT  <Zda>.H, <Zn>.B, <Zm>.B
	//	SDOT  <Zda>.S, <Zn>.H, <Zm>.H
	//
	// <T> and <Tb> are specified in the encoding text, that there is a constraint "T = 4*Tb".
	// We don't know this fact by looking at the encoding format solely, without this information
	// the first encoding domain entails the other 2. And at instruction matching phase we simply
	// cannot deduplicate them. So we defer this deduplication to the encoding phase.
	// We need the ok signal with [elemEncoder.comp] field to deduplicate them.
	elemEncoders []elemEncoder
}

// opsInProg returns an iterator over the operands ([Addr]) of p
func opsInProg(p *obj.Prog) iter.Seq[*obj.Addr] {
	return func(yield func(*obj.Addr) bool) {
		// Go order: From, Reg, RestArgs..., To
		// For SVE, Reg is unused as it's so common that registers have arrangements.
		if p.From.Type != obj.TYPE_NONE {
			if !yield(&p.From) {
				return
			}
		}
		for j := range p.RestArgs {
			if !yield(&p.RestArgs[j].Addr) {
				return
			}
		}
		if p.To.Type != obj.TYPE_NONE {
			if !yield(&p.To) {
				return
			}
		}
	}
}

// aclass returns the AClass of an Addr.
func aclass(a *obj.Addr) AClass {
	if a.Type == obj.TYPE_REG {
		if a.Reg >= REG_Z0 && a.Reg <= REG_Z31 {
			return AC_ZREG
		}
		if a.Reg >= REG_P0 && a.Reg <= REG_P15 {
			return AC_PREG
		}
		if a.Reg >= REG_ARNG && a.Reg < REG_ELEM {
			return AC_ARNG
		}
		if a.Reg >= REG_ZARNG && a.Reg < REG_ZARNGELEM {
			return AC_ARNG
		}
		if a.Reg >= REG_ZARNGELEM && a.Reg < REG_PZELEM {
			return AC_ARNGIDX
		}
		if a.Reg >= REG_PZELEM && a.Reg < REG_PARNGZM {
			if a.Reg&(1<<5) == 0 {
				return AC_ZREGIDX
			} else {
				return AC_PREGIDX
			}
		}
		if a.Reg >= REG_PARNGZM && a.Reg < REG_PARNGZM_END {
			switch (a.Reg >> 5) & 15 {
			case PRED_M, PRED_Z:
				return AC_PREGZM
			default:
				return AC_ARNG
			}
		}
		if a.Reg >= REG_V0 && a.Reg <= REG_V31 {
			return AC_VREG
		}
		if a.Reg >= REG_R0 && a.Reg <= REG_R31 || a.Reg == REG_RSP {
			return AC_SPZGREG
		}
	}
	if a.Type == obj.TYPE_CONST || a.Type == obj.TYPE_FCONST {
		return AC_IMM
	}
	if a.Type == obj.TYPE_REGLIST {
		switch (a.Offset >> 12) & 0xf {
		case 0x7:
			return AC_REGLIST1
		case 0xa:
			return AC_REGLIST2
		case 0x6:
			return AC_REGLIST3
		case 0x2:
			return AC_REGLIST4
		}
	}
	if a.Type == obj.TYPE_MEM {
		return AC_MEMEXT
	}
	if a.Type == obj.TYPE_SPECIAL {
		return AC_SPECIAL
	}
	panic("unknown AClass")
}

// addrComponent returns the binary (component) of the stored element in a at index, for operand
// of type aclass.
//
// For example, for operand of type AC_ARNG, it has 2 permissible components (identified by index)
//  0. register: <reg>
//  1. arrangement: <T>
//
// They are stored in a.Reg as:
//
//	reg | (arrangement << 5)
//
// More details are in the comments in the switch cases of this function.
func addrComponent(a *obj.Addr, acl AClass, index int) uint32 {
	switch acl {
	//	AClass: AC_ARNG, AC_PREG, AC_PREGZ, AC_PREGM, AC_ZREG
	//	GNU mnemonic: <reg>.<T> Or <reg>/<T> (T is M or Z)
	//	Go mnemonic:
	//		reg.<T>
	//	Encoding:
	//		Type = TYPE_REG
	// 		Reg = reg | (arrangement or predication << 5)
	case AC_ARNG, AC_PREG, AC_PREGZM, AC_ZREG:
		switch index {
		case 0:
			return uint32(a.Reg & 31)
		case 1:
			return uint32((a.Reg >> 5) & 15)
		default:
			panic(fmt.Errorf("unknown elm index at %d in AClass %d", index, acl))
		}
	//	AClass: AC_ARNGIDX, AC_PREGIDX, AC_ZREGIDX
	//	GNU mnemonic: <reg>.<T>[<index>]
	//	Go mnemonic:
	//		reg.T[index]
	//	Encoding:
	//		Type = TYPE_REG
	// 		Reg = reg | (arrangement << 5)
	//		Index = index
	case AC_ARNGIDX, AC_PREGIDX, AC_ZREGIDX:
		switch index {
		case 0:
			return uint32(a.Reg & 31)
		case 1:
			// Arrangement
			return uint32((a.Reg >> 5) & 15)
		case 2:
			// Index
			return uint32(a.Index)
		default:
			panic(fmt.Errorf("unknown elm index at %d in AClass %d", index, acl))
		}
	//	AClass: AC_SPZGREG, AC_VREG
	//	GNU mnemonic: <width><reg>
	//	Go mnemonic:
	//		reg (the width is already represented in the opcode)
	//	Encoding:
	//		Type = TYPE_REG
	// 		Reg = reg
	case AC_SPZGREG, AC_VREG:
		switch index {
		case 0:
			// These are all width checks, they should map to no-op checks altogether.
			return 0
		case 1:
			return uint32(a.Reg)
		default:
			panic(fmt.Errorf("unknown elm index at %d in AClass %d", index, acl))
		}
	//	AClass: AC_IMM
	//	GNU mnemonic: <imm>, <shift>
	//	Go mnemonic:
	//		$imm<<shift
	//	Encoding:
	//		Type = TYPE_CONST or TYPE_FCONST
	//		Offset = imm (shift already applied)
	case AC_IMM:
		switch index {
		case 0:
			if a.Type == obj.TYPE_FCONST {
				switch v := a.Val.(type) {
				case float64:
					return math.Float32bits(float32(v))
				default:
					panic(fmt.Errorf("unknown float immediate value %v", a.Val))
				}
			}
			return uint32(a.Offset)
		default:
			panic(fmt.Errorf("unknown elm index at %d in AClass %d", index, acl))
		}
	//	AClass: AC_REGLIST1, AC_REGLIST2, AC_REGLIST3, AC_REGLIST4
	//	GNU mnemonic: {reg1.T, reg2.T, ...}
	//	Go mnemonic:
	//		[reg1.T, reg2.T, ...]
	//	Encoding:
	//		Type = TYPE_REGLIST
	// 		Offset = register prefix | register count | arrangement (opcode) | first register
	case AC_REGLIST1, AC_REGLIST2, AC_REGLIST3, AC_REGLIST4:
		firstReg := int(a.Offset & 31)
		prefix := a.Offset >> 32 & 0b11
		sum := 32
		if prefix == 2 {
			sum = 16
		}
		switch acl {
		case AC_REGLIST1:
			if index > 2 {
				panic(fmt.Errorf("unknown elm index at %d in AClass %d", index, acl))
			}
		case AC_REGLIST2:
			if index > 4 {
				panic(fmt.Errorf("unknown elm index at %d in AClass %d", index, acl))
			}
		case AC_REGLIST3:
			if index > 6 {
				panic(fmt.Errorf("unknown elm index at %d in AClass %d", index, acl))
			}
		case AC_REGLIST4:
			if index > 8 {
				panic(fmt.Errorf("unknown elm index at %d in AClass %d", index, acl))
			}
		}
		switch index % 2 {
		case 0:
			// register
			return uint32((firstReg + index/2) % sum)
		case 1:
			// arrangement
			curQ := a.Offset >> 30 & 0b11
			curSize := a.Offset >> 10 & 0b11
			switch curQ {
			case 0:
				switch curSize {
				case 0:
					return ARNG_8B
				case 1:
					return ARNG_4H
				case 2:
					return ARNG_2S
				case 3:
					return ARNG_1D
				default:
					panic(fmt.Errorf("unknown size value at %d in AClass %d", index, acl))
				}
			case 1:
				switch curSize {
				case 0:
					return ARNG_16B
				case 1:
					return ARNG_8H
				case 2:
					return ARNG_4S
				case 3:
					return ARNG_2D
				default:
					panic(fmt.Errorf("unknown size value at %d in AClass %d", index, acl))
				}
			case 2:
				switch curSize {
				case 1:
					return ARNG_B
				case 2:
					return ARNG_H
				case 3:
					return ARNG_S
				default:
					panic(fmt.Errorf("unknown size value at %d in AClass %d", index, acl))
				}
			case 3:
				switch curSize {
				case 1:
					return ARNG_D
				case 2:
					return ARNG_Q
				default:
					panic(fmt.Errorf("unknown size value at %d in AClass %d", index, acl))
				}
			default:
				panic(fmt.Errorf("unknown Q value at %d in AClass %d", index, acl))
			}
		default:
			panic(fmt.Errorf("unknown elm index at %d in AClass %d", index, acl))
		}
	//	AClass: AC_SPECIAL
	//	GNU mnemonic: <special>
	//	Go mnemonic:
	//		special
	//	Encoding:
	//		Type = TYPE_SPECIAL
	//		Offset = SpecialOperand enum value
	case AC_SPECIAL:
		switch index {
		case 0:
			return uint32(a.Offset)
		default:
			panic(fmt.Errorf("unknown elm index at %d in AClass %d", index, acl))
		}
	//	AClass: AC_MEMEXT
	//	GNU mnemonic: [<reg1>.<T1>, <reg2>.<T2>, <mod> <amount>]
	//	Go mnemonic:
	//		(reg2.T2.mod<<amount)(reg1.T1)
	//	Encoding:
	//		Type = TYPE_MEM
	//		Reg = Index register (with arrangement if applicable)
	//		Index = Base register (with arrangement if applicable)
	//		Scale = Packed mod and amount
	case AC_MEMEXT:
		switch index {
		case 0:
			return uint32(a.Index)
		case 1:
			return uint32((a.Index >> 5) & 15)
		case 2:
			return uint32(a.Reg)
		case 3:
			return uint32((a.Reg >> 5) & 15)
		case 4:
			// mod is either 1 (UXTW), 2 (SXTW), or 4 (LSL)
			mod := uint32((a.Scale >> 9) & 0x7)
			amount := uint32((a.Scale >> 12) & 0x7)
			if mod == 0 && amount > 0 {
				// LSL is implied when no extension is specified but amount > 0
				mod |= 1 << 2
			}
			return mod
		case 5:
			return uint32((a.Scale >> 12) & 0x7)
		default:
			panic(fmt.Errorf("unknown elm index at %d in AClass %d", index, acl))
		}
	}
	// TODO: handle more AClasses.
	panic(fmt.Errorf("unknown AClass %d", acl))
}

var codeI1Tsz uint32 = 0xffffffff
var codeImm2Tsz uint32 = 0xfffffffe
var codeShift161919212223 uint32 = 0xfffffffd
var codeShift161919212224 uint32 = 0xfffffffc
var codeShift588102224 uint32 = 0xfffffffb
var codeLogicalImmArrEncoding uint32 = 0xfffffffa
var codeNoOp uint32 = 0xfffffff9

// encodeI1Tsz is the implementation of the following encoding logic:
// Is the immediate index, in the range 0 to one less than the number of elements in 128 bits, encoded in "i1:tsz".
// bit range mappings:
// i1: [20:21)
// tsz: [16:20)
// Note:
//
//	arr is the arrangement.
//	This encoding is aligned to the high bit of the box, according to the spec.
func encodeI1Tsz(v, arr uint32) (uint32, bool) {
	switch arr {
	case ARNG_B:
		if v > 15 {
			return 0, false
		}
		return v << 17, true
	case ARNG_H:
		if v > 7 {
			return 0, false
		}
		return v << 18, true
	case ARNG_S:
		if v > 3 {
			return 0, false
		}
		return v << 19, true
	case ARNG_D:
		if v > 1 {
			return 0, false
		}
		return v << 20, true
	case ARNG_Q:
		if v > 0 {
			return 0, false
		}
		return 0, true
	default:
		return 0, false
	}
}

// encodeImm2Tsz is the implementation of the following encoding logic:
// Is the immediate index, in the range 0 to one less than the number of elements in 512 bits, encoded in "imm2:tsz".
// bit range mappings:
// imm2: [22:24)
// tsz: [16:21)
// Note:
//
//	arr is the arrangement.
//	This encoding is aligned to the high bit of the box, according to the spec.
func encodeImm2Tsz(v, arr uint32) (uint32, bool) {
	switch arr {
	case ARNG_B:
		if v > 63 {
			return 0, false
		}
		v <<= 1
		return (v&31)<<16 | (v>>5)<<22, true
	case ARNG_H:
		if v > 31 {
			return 0, false
		}
		v <<= 2
		return (v&31)<<16 | (v>>5)<<22, true
	case ARNG_S:
		if v > 15 {
			return 0, false
		}
		v <<= 3
		return (v&31)<<16 | (v>>5)<<22, true
	case ARNG_D:
		if v > 7 {
			return 0, false
		}
		v <<= 4
		return (v&31)<<16 | (v>>5)<<22, true
	case ARNG_Q:
		if v > 3 {
			return 0, false
		}
		v <<= 5
		return (v&31)<<16 | (v>>5)<<22, true
	default:
		return 0, false
	}
}

type arrAlignType int

const (
	arrAlignBHSD arrAlignType = iota
	arrAlignHSD
	arrAlignBHS
)

// encodeShiftTriple encodes an shift immediate value in "tszh:tszl:imm3".
// tszh, tszl, imm3 are in ranges, sorted by bit position.
// These shifts are also bounded by arrangement element size.
func encodeShiftTriple(v uint32, r [6]int, prevAddr *obj.Addr, op obj.As) (uint32, bool) {
	// The previous op must be a scalable vector, and we need its arrangement.
	acl := aclass(prevAddr)
	if acl != AC_ARNG {
		return 0, false
	}
	arr := addrComponent(prevAddr, acl, 1) // Get arrangement
	elemBits := uint32(0)
	switch arr {
	case ARNG_B:
		elemBits = 8
	case ARNG_H:
		elemBits = 16
	case ARNG_S:
		elemBits = 32
	case ARNG_D:
		elemBits = 64
	default:
		return 0, false
	}
	if v >= elemBits {
		return 0, false
	}
	var C uint32
	// Unfortunately these information are in the decoding ASL.
	// For these instructions, the esize (see comment in the switch below)
	// is derived from the destination arrangement, however how this function is called is deriving
	// the esize from one of the source.
	// We need to address this discrepancy.
	effectiveEsize := elemBits
	switch op {
	case AZRSHRNB, AZRSHRNT, AZSHRNB, AZSHRNT, AZSQRSHRNB, AZSQRSHRNT, AZSQRSHRUNB, AZSQRSHRUNT,
		AZSQSHRNB, AZSQSHRNT, AZSQSHRUNB, AZSQSHRUNT, AZUQRSHRNB, AZUQRSHRNT, AZUQSHRNB, AZUQSHRNT:
		effectiveEsize = elemBits / 2
	}
	switch op {
	case AZASR, AZLSR, AZURSHR, AZASRD,
		AZRSHRNB, AZRSHRNT, AZSHRNB, AZSHRNT, AZSQRSHRNB, AZSQRSHRNT, AZSQRSHRUNB, AZSQRSHRUNT,
		AZSQSHRNB, AZSQSHRNT, AZSQSHRUNB, AZSQSHRUNT, AZSRSHR, AZUQRSHRNB, AZUQRSHRNT, AZUQSHRNB, AZUQSHRNT,
		AZURSRA, AZUSRA, AZXAR, AZSRI, AZSRSRA, AZSSRA:
		// ASL: let shift : integer = (2 * esize) - UInt(tsize::imm3);
		if v == 0 {
			return 0, false
		}
		C = (2 * effectiveEsize) - v
	default:
		// ASL: let shift : integer = UInt(tsize::imm3) - esize;
		C = effectiveEsize + v
	}
	var chunks [3]uint32
	for i := 0; i < 6; i += 2 {
		chunks[i/2] = C & ((1 << (r[i+1] - r[i])) - 1)
		C >>= (r[i+1] - r[i])
	}
	return uint32((chunks[0] << r[0]) |
		(chunks[1] << r[2]) |
		(chunks[2] << r[4])), true
}

// encodeLogicalImmEncoding is the implementation of the following encoding logic:
// Is the size specifier,
// imm13	<T>
// 0xxxxxx0xxxxx	S
// 0xxxxxx10xxxx	H
// 0xxxxxx110xxx	B
// 0xxxxxx1110xx	B
// 0xxxxxx11110x	B
// 0xxxxxx11111x	RESERVED
// 1xxxxxxxxxxxx	D
// At the meantime:
// Is a 64, 32, 16 or 8-bit bitmask consisting of replicated 2, 4, 8, 16, 32 or 64 bit fields,
// each field containing a rotated run of non-zero bits, encoded in the "imm13" field.
//
// bit range mappings:
// imm13: [5:18)
//
// ARM created a "clever" recipe that can generate useful repeating 8-64 bit bitmasks.
// Instead of storing the literal binary number, the processor reads a 13-bit recipe
// using three fields (bits from high to low):
// N (1 bit), immr (6 bits), and imms (6 bits).
//
// How the recipe works:
// Every logical immediate represents a repeating pattern (like repeating tiles). The processor
// uses the three fields to figure out the size of the tile, how many 1s are in the tile, and
// how far to rotate it.
// The N bit combined with the upper bits of imms determines the width of the repeating block.
// Depending on these bits, the fundamental block can be 2, 4, 8, 16, 32, or 64 bits wide.
// The lower bits of imms dictate exactly how many contiguous 1s exist inside that block.
// The immr value tells the processor how many bits to rotate that block to the right.
// Finally, the resulting block is duplicated to fill a standard 64-bit lane.
func encodeLogicalImmArrEncoding(v uint32, adjacentAddr *obj.Addr) (uint32, bool) {
	acl := aclass(adjacentAddr)
	if acl != AC_ARNG {
		return 0, false
	}
	arr := addrComponent(adjacentAddr, acl, 1)

	// Replicate the given immediate to fill a full 64-bit lane.
	// This ensures our pattern-shrinking logic naturally respects the vector lane bounds.
	var val uint64
	switch arr {
	case ARNG_B: // 8-bit lane
		v8 := uint64(v & 0xFF)
		val = v8 * 0x0101010101010101
	case ARNG_H: // 16-bit lane
		v16 := uint64(v & 0xFFFF)
		val = v16 * 0x0001000100010001
	case ARNG_S: // 32-bit lane
		v32 := uint64(v)
		val = v32 | (v32 << 32)
	case ARNG_D: // 64-bit lane
		val = uint64(v) // Top 32 bits are implicitly 0
	default:
		return 0, false
	}

	// Reject all zeros or all ones (handled by MOV/EOR, invalid for AND/ORR immediates)
	if val == 0 || val == ^uint64(0) {
		return 0, false
	}

	// Find the absolute smallest repeating pattern size (64 down to 2)
	size := uint64(64)
	for size > 2 {
		half := size / 2
		mask := (uint64(1) << half) - 1
		lower := val & mask
		upper := (val >> half) & mask

		// If the top half matches the bottom half, shrink our window
		if lower == upper {
			size = half
			val = lower
		} else {
			break
		}
	}

	// Count the contiguous ones in this minimal pattern
	mask := (uint64(1) << size) - 1
	val &= mask
	ones := bits.OnesCount64(val)

	// Find the right-rotation (rot) needed to align the 1s at the bottom
	expected := (uint64(1) << ones) - 1
	rot := -1
	for r := 0; r < int(size); r++ {
		// Right rotate 'val' by 'r' bits within a 'size'-bit window
		rotated := ((val >> r) | (val << (int(size) - r))) & mask
		if rotated == expected {
			rot = r
			break
		}
	}

	if rot == -1 {
		return 0, false
	}

	// immr is the amount the hardware must right-rotate the base pattern.
	// Since 'rot' is how much we right-rotated the target to find the base,
	// the hardware needs the inverse rotation.
	immr := uint32((int(size) - rot) % int(size))

	// If we couldn't find a rotation that forms a perfect contiguous block of 1s, it's invalid.
	if rot == -1 {
		return 0, false
	}

	// Encode N, immr, and imms
	n := uint32(0)
	if size == 64 {
		n = 1
	}

	// The imms prefix is mathematically generated by (~(size*2 - 1) & 0x3F).
	// We then OR it with the number of ones (minus 1).
	imms := (uint32(^(size*2 - 1)) & 0x3F) | uint32(ones-1)

	// Construct the final 13-bit field: N (1) | immr (6) | imms (6)
	imm13 := (n << 12) | (immr << 6) | imms

	// Shift by 5 to place imm13 into instruction bits [5:17]
	return imm13 << 5, true
}

// tryEncode tries to encode p with i, it returns the encoded binary and ok signal.
func (i *instEncoder) tryEncode(p *obj.Prog) (uint32, bool) {
	bin := i.fixedBits
	// Some elements are encoded in the same component, they need to be equal.
	// For example { <Zn1>.<Tb>-<Zn2>.<Tb> }.
	// The 2 instances of <Tb> must encode to the same value.
	encoded := map[component]uint32{}
	var addrs []*obj.Addr
	for addr := range opsInProg(p) {
		addrs = append(addrs, addr)
	}
	if len(addrs) != len(i.args) {
		return 0, false
	}
	for opIdx, addr := range addrs {
		if opIdx >= len(i.args) {
			return 0, false
		}
		op := i.args[opIdx]
		acl := aclass(addr)
		if acl != op.class {
			return 0, false
		}
		for i, enc := range op.elemEncoders {
			val := addrComponent(addr, acl, i)
			if (p.As == AZFCPY || p.As == AZFDUP) && acl == AC_IMM {
				// These instructions expects ARM's 8-bit float encoding.
				// Reinterpret the uint32 bits back as a float32, then convert to float64 for chipfloat7
				fval := float64(math.Float32frombits(val))
				encode := (&ctxt7{}).chipfloat7(fval)
				if encode == -1 {
					// Handle error or return false to indicate mismatch
					return 0, false
				}
				val = uint32(encode)
			}
			if b, ok := enc.fn(val); ok || b != 0 {
				specialB := uint32(b)
				if !ok {
					specialB = b
					switch b {
					case codeI1Tsz:
						b, ok = encodeI1Tsz(val, addrComponent(addr, acl, i-1))
					case codeImm2Tsz:
						b, ok = encodeImm2Tsz(val, addrComponent(addr, acl, i-1))
					case codeShift161919212223:
						b, ok = encodeShiftTriple(val, [6]int{16, 19, 19, 21, 22, 23}, addrs[opIdx+1], p.As)
					case codeShift161919212224:
						b, ok = encodeShiftTriple(val, [6]int{16, 19, 19, 21, 22, 24}, addrs[opIdx+1], p.As)
					case codeShift588102224:
						b, ok = encodeShiftTriple(val, [6]int{5, 8, 8, 10, 22, 24}, addrs[opIdx+1], p.As)
					case codeLogicalImmArrEncoding:
						b, ok = encodeLogicalImmArrEncoding(val, addrs[opIdx+1])
					case codeNoOp:
						b, ok = 0, true
					default:
						panic(fmt.Errorf("unknown encoding function code %d", b))
					}
				}
				if !ok {
					return 0, false
				}
				bin |= b
				if _, ok := encoded[enc.comp]; ok && b != encoded[enc.comp] {
					if specialB == codeNoOp {
						// NoOp encodings don't need checks.
						continue
					}
					return 0, false
				}
				if enc.comp != enc_NIL && specialB != codeNoOp {
					// NoOp encodings don't need bookkeeping.
					encoded[enc.comp] = b
				}
			} else {
				return 0, false
			}
		}
	}
	return bin, true
}
