// Copyright 2024 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package s390xasm

import (
	"encoding/binary"
	"fmt"
)

// instFormat is a decoding rule for one specific instruction form.
// An instruction ins matches the rule if ins&Mask == Value.
// DontCare bits are mainly used for finding the same instruction
// name differing with the number of argument fields.
// The Args are stored in the same order as the instruction manual.
type instFormat struct {
	Op       Op
	Mask     uint64
	Value    uint64
	DontCare uint64
	Args     [8]*argField
}

// argField indicate how to decode an argument to an instruction.
// First parse the value from the BitFields, shift it left by Shift
// bits to get the actual numerical value.
type argField struct {
	Type  ArgType
	flags uint16
	BitField
}

// Parse parses the Arg out from the given binary instruction i.
func (a argField) Parse(i uint64) Arg {
	switch a.Type {
	default:
		return nil
	case TypeUnknown:
		return nil
	case TypeReg:
		return R0 + Reg(a.BitField.Parse(i))
	case TypeFPReg:
		return F0 + Reg(a.BitField.Parse(i))
	case TypeCReg:
		return C0 + Reg(a.BitField.Parse(i))
	case TypeACReg:
		return A0 + Reg(a.BitField.Parse(i))
	case TypeBaseReg:
		return B0 + Base(a.BitField.Parse(i))
	case TypeIndexReg:
		return X0 + Index(a.BitField.Parse(i))
	case TypeDispUnsigned:
		return Disp12(a.BitField.Parse(i))
	case TypeDispSigned20:
		return Disp20(a.BitField.ParseSigned(i))
	case TypeVecReg:
		m := i >> 24 // Handling RXB field(bits 36 to 39)
		if ((m>>3)&0x1 == 1) && (a.BitField.Offs == 8) {
			return V0 + VReg(a.BitField.Parse(i)) + VReg(16)
		} else if ((m>>2)&0x1 == 1) && (a.BitField.Offs == 12) {
			return V0 + VReg(a.BitField.Parse(i)) + VReg(16)
		} else if ((m>>1)&0x1 == 1) && (a.BitField.Offs == 16) {
			return V0 + VReg(a.BitField.Parse(i)) + VReg(16)
		} else if ((m)&0x1 == 1) && (a.BitField.Offs == 32) {
			return V0 + VReg(a.BitField.Parse(i)) + VReg(16)
		} else {
			return V0 + VReg(a.BitField.Parse(i))
		}
	case TypeImmSigned8:
		return Sign8(a.BitField.ParseSigned(i))
	case TypeImmSigned16:
		return Sign16(a.BitField.ParseSigned(i))
	case TypeImmSigned32:
		return Sign32(a.BitField.ParseSigned(i))
	case TypeImmUnsigned:
		return Imm(a.BitField.Parse(i))
	case TypeRegImSigned12:
		return RegIm12(a.BitField.ParseSigned(i))
	case TypeRegImSigned16:
		return RegIm16(a.BitField.ParseSigned(i))
	case TypeRegImSigned24:
		return RegIm24(a.BitField.ParseSigned(i))
	case TypeRegImSigned32:
		return RegIm32(a.BitField.ParseSigned(i))
	case TypeMask:
		return Mask(a.BitField.Parse(i))
	case TypeLen:
		return Len(a.BitField.Parse(i))
	}
}

type ArgType int8

const (
	TypeUnknown       ArgType = iota
	TypeReg                   // integer register
	TypeFPReg                 // floating point register
	TypeACReg                 // access register
	TypeCReg                  // control register
	TypeVecReg                // vector register
	TypeImmUnsigned           // unsigned immediate/flag/mask, this is the catch-all type
	TypeImmSigned8            // Signed 8-bit Immdediate
	TypeImmSigned16           // Signed 16-bit Immdediate
	TypeImmSigned32           // Signed 32-bit Immdediate
	TypeBaseReg               // Base Register for accessing memory
	TypeIndexReg              // Index Register
	TypeDispUnsigned          // Displacement 12-bit unsigned for memory address
	TypeDispSigned20          // Displacement 20-bit signed for memory address
	TypeRegImSigned12         // RegisterImmediate 12-bit signed data
	TypeRegImSigned16         // RegisterImmediate 16-bit signed data
	TypeRegImSigned24         // RegisterImmediate 24-bit signed data
	TypeRegImSigned32         // RegisterImmediate 32-bit signed data
	TypeMask                  // 4-bit Mask
	TypeLen                   // Length of Memory Operand
	TypeLast
)

func (t ArgType) String() string {
	switch t {
	default:
		return fmt.Sprintf("ArgType(%d)", int(t))
	case TypeUnknown:
		return "Unknown"
	case TypeReg:
		return "Reg"
	case TypeFPReg:
		return "FPReg"
	case TypeACReg:
		return "ACReg"
	case TypeCReg:
		return "CReg"
	case TypeDispUnsigned:
		return "DispUnsigned"
	case TypeDispSigned20:
		return "DispSigned20"
	case TypeBaseReg:
		return "BaseReg"
	case TypeIndexReg:
		return "IndexReg"
	case TypeVecReg:
		return "VecReg"
	case TypeImmSigned8:
		return "ImmSigned8"
	case TypeImmSigned16:
		return "ImmSigned16"
	case TypeImmSigned32:
		return "ImmSigned32"
	case TypeImmUnsigned:
		return "ImmUnsigned"
	case TypeRegImSigned12:
		return "RegImSigned12"
	case TypeRegImSigned16:
		return "RegImSigned16"
	case TypeRegImSigned24:
		return "RegImSigned24"
	case TypeRegImSigned32:
		return "RegImSigned32"
	case TypeMask:
		return "Mask"
	case TypeLen:
		return "Len"
	}
}

func (t ArgType) GoString() string {
	s := t.String()
	if t > 0 && t < TypeLast {
		return "Type" + s
	}
	return s
}

var (
	// Errors
	errShort   = fmt.Errorf("truncated instruction")
	errUnknown = fmt.Errorf("unknown instruction")
)

var decoderCover []bool

// Decode decodes the leading bytes in src as a single instruction using
// byte order ord.
func Decode(src []byte) (inst Inst, err error) {
	if len(src) < 2 {
		return inst, errShort
	}
	if decoderCover == nil {
		decoderCover = make([]bool, len(instFormats))
	}
	bit_check := binary.BigEndian.Uint16(src[:2])
	bit_check = bit_check >> 14
	l := int(0)
	if (bit_check & 0x03) == 0 {
		l = 2
	} else if bit_check&0x03 == 3 {
		l = 6
	} else if (bit_check&0x01 == 1) || (bit_check&0x02 == 2) {
		l = 4
	}
	inst.Len = l
	ui_extn := uint64(0)
	switch l {
	case 2:
		ui_extn = uint64(binary.BigEndian.Uint16(src[:inst.Len]))
		inst.Enc = ui_extn
		ui_extn = ui_extn << 48
	case 4:
		ui_extn = uint64(binary.BigEndian.Uint32(src[:inst.Len]))
		inst.Enc = ui_extn
		ui_extn = ui_extn << 32
	case 6:
		u1 := binary.BigEndian.Uint32(src[:(inst.Len - 2)])
		u2 := binary.BigEndian.Uint16(src[(inst.Len - 2):inst.Len])
		ui_extn = uint64(u1)<<16 | uint64(u2)
		ui_extn = ui_extn << 16
		inst.Enc = ui_extn
	default:
		return inst, errShort
	}
	for _, iform := range instFormats {
		if ui_extn&iform.Mask != iform.Value {
			continue
		}
		if (iform.DontCare & ^(ui_extn)) != iform.DontCare {
			continue
		}
		for j, argfield := range iform.Args {
			if argfield == nil {
				break
			}
			inst.Args[j] = argfield.Parse(ui_extn)
		}
		inst.Op = iform.Op
		break
	}
	if inst.Op == 0 && inst.Enc != 0 {
		return inst, errUnknown
	}
	return inst, nil
}
