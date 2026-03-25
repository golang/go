// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"cmd/internal/obj"
	"fmt"
	"iter"
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
		if a.Reg >= REG_ZARNG && a.Reg < REG_PARNGZM {
			return AC_ARNG
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
	}
	// TODO: handle more AClasses.
	panic(fmt.Errorf("unknown AClass %d", acl))
}

// tryEncode tries to encode p with i, it returns the encoded binary and ok signal.
func (i *instEncoder) tryEncode(p *obj.Prog) (uint32, bool) {
	bin := i.fixedBits
	// Some elements are encoded in the same component, they need to be equal.
	// For example { <Zn1>.<Tb>-<Zn2>.<Tb> }.
	// The 2 instances of <Tb> must encode to the same value.
	encoded := map[component]uint32{}
	opIdx := 0
	for addr := range opsInProg(p) {
		if opIdx >= len(i.args) {
			return 0, false
		}
		op := i.args[opIdx]
		opIdx++
		acl := aclass(addr)
		if acl != op.class {
			return 0, false
		}
		for i, enc := range op.elemEncoders {
			val := addrComponent(addr, acl, i)
			if b, ok := enc.fn(val); ok {
				bin |= b
				if _, ok := encoded[enc.comp]; ok && b != encoded[enc.comp] {
					return 0, false
				}
				if enc.comp != enc_NIL {
					encoded[enc.comp] = b
				}
			} else {
				return 0, false
			}
		}
	}
	if opIdx != len(i.args) {
		return 0, false
	}
	return bin, true
}
