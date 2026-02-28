// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package riscv64asm

import (
	"encoding/binary"
	"errors"
)

type argTypeList [6]argType

// An instFormat describes the format of an instruction encoding.
type instFormat struct {
	mask  uint32
	value uint32
	op    Op
	// args describe how to decode the instruction arguments.
	// args is stored as a fixed-size array.
	// if there are fewer than len(args) arguments, args[i] == 0 marks
	// the end of the argument list.
	args argTypeList
}

var (
	errShort   = errors.New("truncated instruction")
	errUnknown = errors.New("unknown instruction")
)

var decoderCover []bool

func init() {
	decoderCover = make([]bool, len(instFormats))
}

// Decode decodes the 4 bytes in src as a single instruction.
func Decode(src []byte) (Inst, error) {
	length := len(src)
	if length < 2 {
		return Inst{}, errShort
	}

	var x uint32
	// Non-RVC instructions always starts with 0x11
	// So check whether src[0] & 3 == 3
	if src[0]&3 == 3 {
		if length < 4 {
			return Inst{}, errShort
		}
		length = 4
		x = binary.LittleEndian.Uint32(src)
	} else {
		length = 2
		x = uint32(binary.LittleEndian.Uint16(src))
	}

Search:
	for i, f := range instFormats {
		if (x & f.mask) != f.value {
			continue
		}

		// Decode args.
		var args Args
		k := 0
		for _, aop := range f.args {
			if aop == 0 {
				break
			}
			arg := decodeArg(aop, x, i)
			if arg == nil {
				if aop == arg_vm {
					continue
				}
				if f.op != C_NOP {
					// Cannot decode argument.
					continue Search
				}
			}
			args[k] = arg
			k++
		}

		if length == 2 {
			args = convertCompressedIns(&f, args)
		}

		decoderCover[i] = true
		inst := Inst{
			Op:   f.op,
			Args: args,
			Enc:  x,
			Len:  length,
		}
		return inst, nil
	}
	return Inst{}, errUnknown
}

// decodeArg decodes the arg described by aop from the instruction bits x.
// It returns nil if x cannot be decoded according to aop.
func decodeArg(aop argType, x uint32, index int) Arg {
	switch aop {
	case arg_rd:
		return X0 + Reg((x>>7)&((1<<5)-1))

	case arg_rs1:
		return X0 + Reg((x>>15)&((1<<5)-1))

	case arg_rs2:
		return X0 + Reg((x>>20)&((1<<5)-1))

	case arg_rs3:
		return X0 + Reg((x>>27)&((1<<5)-1))

	case arg_fd:
		return F0 + Reg((x>>7)&((1<<5)-1))

	case arg_fs1:
		return F0 + Reg((x>>15)&((1<<5)-1))

	case arg_fs2:
		return F0 + Reg((x>>20)&((1<<5)-1))

	case arg_fs3:
		return F0 + Reg((x>>27)&((1<<5)-1))

	case arg_vd:
		return V0 + Reg((x>>7)&((1<<5)-1))

	case arg_vm:
		if x&(1<<25) == 0 {
			return V0
		} else {
			return nil
		}

	case arg_vs1:
		return V0 + Reg((x>>15)&((1<<5)-1))

	case arg_vs2:
		return V0 + Reg((x>>20)&((1<<5)-1))

	case arg_vs3:
		return V0 + Reg((x>>7)&((1<<5)-1))

	case arg_rs1_ptr:
		return RegPtr{X0 + Reg((x>>15)&((1<<5)-1))}

	case arg_rs1_mem:
		imm := x >> 20
		// Sign-extend
		if imm>>uint32(12-1) == 1 {
			imm |= 0xfffff << 12
		}
		return RegOffset{X0 + Reg((x>>15)&((1<<5)-1)), Simm{int32(imm), true, 12}}

	case arg_rs1_store:
		imm := (x<<20)>>27 | (x>>25)<<5
		// Sign-extend
		if imm>>uint32(12-1) == 1 {
			imm |= 0xfffff << 12
		}
		return RegOffset{X0 + Reg((x>>15)&((1<<5)-1)), Simm{int32(imm), true, 12}}

	case arg_pred:
		imm := x << 4 >> 28
		return MemOrder(uint8(imm))

	case arg_succ:
		imm := x << 8 >> 28
		return MemOrder(uint8(imm))

	case arg_csr:
		imm := x >> 20
		return CSR(imm)

	case arg_zimm:
		imm := x << 12 >> 27
		return Uimm{imm, true}

	case arg_shamt5:
		imm := x << 7 >> 27
		return Uimm{imm, false}

	case arg_shamt6:
		imm := x << 6 >> 26
		return Uimm{imm, false}

	case arg_imm12:
		imm := x >> 20
		// Sign-extend
		if imm>>uint32(12-1) == 1 {
			imm |= 0xfffff << 12
		}
		return Simm{int32(imm), true, 12}

	case arg_imm20:
		imm := x >> 12
		return Uimm{imm, false}

	case arg_jimm20:
		imm := (x>>31)<<20 | (x<<1)>>22<<1 | (x<<11)>>31<<11 | (x<<12)>>24<<12
		// Sign-extend
		if imm>>uint32(21-1) == 1 {
			imm |= 0x7ff << 21
		}
		return Simm{int32(imm), true, 21}

	case arg_simm12:
		imm := (x<<20)>>27 | (x>>25)<<5
		// Sign-extend
		if imm>>uint32(12-1) == 1 {
			imm |= 0xfffff << 12
		}
		return Simm{int32(imm), true, 12}

	case arg_bimm12:
		imm := (x<<20)>>28<<1 | (x<<1)>>26<<5 | (x<<24)>>31<<11 | (x>>31)<<12
		// Sign-extend
		if imm>>uint32(13-1) == 1 {
			imm |= 0x7ffff << 13
		}
		return Simm{int32(imm), true, 13}

	case arg_simm5:
		imm := x << 12 >> 27
		// Sign-extend
		if imm>>uint32(5-1) == 1 {
			imm |= 0x7ffffff << 5
		}
		return Simm{int32(imm), true, 5}

	case arg_zimm5:
		imm := x << 12 >> 27
		return Uimm{imm, true}

	case arg_vtype_zimm10:
		imm := x << 2 >> 22
		return VType(imm)

	case arg_vtype_zimm11:
		imm := x << 1 >> 21
		return VType(imm)

	case arg_rd_p, arg_rs2_p:
		return X8 + Reg((x>>2)&((1<<3)-1))

	case arg_fd_p, arg_fs2_p:
		return F8 + Reg((x>>2)&((1<<3)-1))

	case arg_rs1_p, arg_rd_rs1_p:
		return X8 + Reg((x>>7)&((1<<3)-1))

	case arg_rd_n0, arg_rs1_n0, arg_rd_rs1_n0, arg_c_rs1_n0:
		if X0+Reg((x>>7)&((1<<5)-1)) == X0 {
			return nil
		}
		return X0 + Reg((x>>7)&((1<<5)-1))

	case arg_c_rs2_n0:
		if X0+Reg((x>>2)&((1<<5)-1)) == X0 {
			return nil
		}
		return X0 + Reg((x>>2)&((1<<5)-1))

	case arg_c_fs2:
		return F0 + Reg((x>>2)&((1<<5)-1))

	case arg_c_rs2:
		return X0 + Reg((x>>2)&((1<<5)-1))

	case arg_rd_n2:
		if X0+Reg((x>>7)&((1<<5)-1)) == X0 || X0+Reg((x>>7)&((1<<5)-1)) == X2 {
			return nil
		}
		return X0 + Reg((x>>7)&((1<<5)-1))

	case arg_c_imm6:
		imm := (x<<25)>>27 | (x<<19)>>31<<5
		// Sign-extend
		if imm>>uint32(6-1) == 1 {
			imm |= 0x3ffffff << 6
		}
		return Simm{int32(imm), true, 6}

	case arg_c_nzimm6:
		imm := (x<<25)>>27 | (x<<19)>>31<<5
		// Sign-extend
		if imm>>uint32(6-1) == 1 {
			imm |= 0x3ffffff << 6
		}
		if int32(imm) == 0 {
			return nil
		}
		return Simm{int32(imm), true, 6}

	case arg_c_nzuimm6:
		imm := (x<<25)>>27 | (x<<19)>>31<<5
		if int32(imm) == 0 {
			return nil
		}
		return Uimm{imm, false}

	case arg_c_uimm7:
		imm := (x<<26)>>31<<6 | (x<<25)>>31<<2 | (x<<19)>>29<<3
		return Uimm{imm, false}

	case arg_c_uimm8:
		imm := (x<<25)>>30<<6 | (x<<19)>>29<<3
		return Uimm{imm, false}

	case arg_c_uimm8sp_s:
		imm := (x<<23)>>30<<6 | (x<<19)>>28<<2
		return Uimm{imm, false}

	case arg_c_uimm8sp:
		imm := (x<<25)>>29<<2 | (x<<19)>>31<<5 | (x<<28)>>30<<6
		return Uimm{imm, false}

	case arg_c_uimm9sp_s:
		imm := (x<<22)>>29<<6 | (x<<19)>>29<<3
		return Uimm{imm, false}

	case arg_c_uimm9sp:
		imm := (x<<25)>>30<<3 | (x<<19)>>31<<5 | (x<<27)>>29<<6
		return Uimm{imm, false}

	case arg_c_bimm9:
		imm := (x<<29)>>31<<5 | (x<<27)>>30<<1 | (x<<25)>>30<<6 | (x<<19)>>31<<8 | (x<<20)>>30<<3
		// Sign-extend
		if imm>>uint32(9-1) == 1 {
			imm |= 0x7fffff << 9
		}
		return Simm{int32(imm), true, 9}

	case arg_c_nzimm10:
		imm := (x<<29)>>31<<5 | (x<<27)>>30<<7 | (x<<26)>>31<<6 | (x<<25)>>31<<4 | (x<<19)>>31<<9
		// Sign-extend
		if imm>>uint32(10-1) == 1 {
			imm |= 0x3fffff << 10
		}
		if int32(imm) == 0 {
			return nil
		}
		return Simm{int32(imm), true, 10}

	case arg_c_nzuimm10:
		imm := (x<<26)>>31<<3 | (x<<25)>>31<<2 | (x<<21)>>28<<6 | (x<<19)>>30<<4
		if int32(imm) == 0 {
			return nil
		}
		return Uimm{imm, false}

	case arg_c_imm12:
		imm := (x<<29)>>31<<5 | (x<<26)>>28<<1 | (x<<25)>>31<<7 | (x<<24)>>31<<6 | (x<<23)>>31<<10 | (x<<21)>>30<<8 | (x<<20)>>31<<4 | (x<<19)>>31<<11
		// Sign-extend
		if imm>>uint32(12-1) == 1 {
			imm |= 0xfffff << 12
		}
		return Simm{int32(imm), true, 12}

	case arg_c_nzimm18:
		imm := (x<<25)>>27<<12 | (x<<19)>>31<<17
		// Sign-extend
		if imm>>uint32(18-1) == 1 {
			imm |= 0x3fff << 18
		}
		if int32(imm) == 0 {
			return nil
		}
		return Simm{int32(imm), true, 18}

	default:
		return nil
	}
}

// convertCompressedIns rewrites the RVC Instruction to regular Instructions
func convertCompressedIns(f *instFormat, args Args) Args {
	var newargs Args
	switch f.op {
	case C_ADDI4SPN:
		f.op = ADDI
		newargs[0] = args[0]
		newargs[1] = Reg(X2)
		newargs[2] = Simm{int32(args[1].(Uimm).Imm), true, 12}

	case C_LW:
		f.op = LW
		newargs[0] = args[0]
		newargs[1] = RegOffset{args[1].(Reg), Simm{int32(args[2].(Uimm).Imm), true, 12}}

	case C_SW:
		f.op = SW
		newargs[0] = args[1]
		newargs[1] = RegOffset{args[0].(Reg), Simm{int32(args[2].(Uimm).Imm), true, 12}}

	case C_NOP:
		f.op = ADDI
		newargs[0] = X0
		newargs[1] = X0
		newargs[2] = Simm{0, true, 12}

	case C_ADDI:
		f.op = ADDI
		newargs[0] = args[0]
		newargs[1] = args[0]
		newargs[2] = Simm{args[1].(Simm).Imm, true, 12}

	case C_LI:
		f.op = ADDI
		newargs[0] = args[0]
		newargs[1] = Reg(X0)
		newargs[2] = Simm{args[1].(Simm).Imm, true, 12}

	case C_ADDI16SP:
		f.op = ADDI
		newargs[0] = Reg(X2)
		newargs[1] = Reg(X2)
		newargs[2] = Simm{args[0].(Simm).Imm, true, 12}

	case C_LUI:
		f.op = LUI
		newargs[0] = args[0]
		newargs[1] = Uimm{uint32(args[1].(Simm).Imm >> 12), false}

	case C_ANDI:
		f.op = ANDI
		newargs[0] = args[0]
		newargs[1] = args[0]
		newargs[2] = Simm{args[1].(Simm).Imm, true, 12}

	case C_SUB:
		f.op = SUB
		newargs[0] = args[0]
		newargs[1] = args[0]
		newargs[2] = args[1]

	case C_XOR:
		f.op = XOR
		newargs[0] = args[0]
		newargs[1] = args[0]
		newargs[2] = args[1]

	case C_OR:
		f.op = OR
		newargs[0] = args[0]
		newargs[1] = args[0]
		newargs[2] = args[1]

	case C_AND:
		f.op = AND
		newargs[0] = args[0]
		newargs[1] = args[0]
		newargs[2] = args[1]

	case C_J:
		f.op = JAL
		newargs[0] = Reg(X0)
		newargs[1] = Simm{args[0].(Simm).Imm, true, 21}

	case C_BEQZ:
		f.op = BEQ
		newargs[0] = args[0]
		newargs[1] = Reg(X0)
		newargs[2] = Simm{args[1].(Simm).Imm, true, 13}

	case C_BNEZ:
		f.op = BNE
		newargs[0] = args[0]
		newargs[1] = Reg(X0)
		newargs[2] = Simm{args[1].(Simm).Imm, true, 13}

	case C_LWSP:
		f.op = LW
		newargs[0] = args[0]
		newargs[1] = RegOffset{Reg(X2), Simm{int32(args[1].(Uimm).Imm), true, 12}}

	case C_JR:
		f.op = JALR
		newargs[0] = Reg(X0)
		newargs[1] = RegOffset{args[0].(Reg), Simm{0, true, 12}}

	case C_MV:
		f.op = ADD
		newargs[0] = args[0]
		newargs[1] = Reg(X0)
		newargs[2] = args[1]

	case C_EBREAK:
		f.op = EBREAK

	case C_JALR:
		f.op = JALR
		newargs[0] = Reg(X1)
		newargs[1] = RegOffset{args[0].(Reg), Simm{0, true, 12}}

	case C_ADD:
		f.op = ADD
		newargs[0] = args[0]
		newargs[1] = args[0]
		newargs[2] = args[1]

	case C_SWSP:
		f.op = SW
		newargs[0] = args[0]
		newargs[1] = RegOffset{Reg(X2), Simm{int32(args[1].(Uimm).Imm), true, 12}}

	// riscv64 compressed instructions
	case C_LD:
		f.op = LD
		newargs[0] = args[0]
		newargs[1] = RegOffset{args[1].(Reg), Simm{int32(args[2].(Uimm).Imm), true, 12}}

	case C_SD:
		f.op = SD
		newargs[0] = args[1]
		newargs[1] = RegOffset{args[0].(Reg), Simm{int32(args[2].(Uimm).Imm), true, 12}}

	case C_ADDIW:
		f.op = ADDIW
		newargs[0] = args[0]
		newargs[1] = args[0]
		newargs[2] = Simm{args[1].(Simm).Imm, true, 12}

	case C_SRLI:
		f.op = SRLI
		newargs[0] = args[0]
		newargs[1] = args[0]
		newargs[2] = args[1]

	case C_SRAI:
		f.op = SRAI
		newargs[0] = args[0]
		newargs[1] = args[0]
		newargs[2] = args[1]

	case C_SUBW:
		f.op = SUBW
		newargs[0] = args[0]
		newargs[1] = args[0]
		newargs[2] = args[1]

	case C_ADDW:
		f.op = ADDW
		newargs[0] = args[0]
		newargs[1] = args[0]
		newargs[2] = args[1]

	case C_SLLI:
		f.op = SLLI
		newargs[0] = args[0]
		newargs[1] = args[0]
		newargs[2] = args[1]

	case C_LDSP:
		f.op = LD
		newargs[0] = args[0]
		newargs[1] = RegOffset{Reg(X2), Simm{int32(args[1].(Uimm).Imm), true, 12}}

	case C_SDSP:
		f.op = SD
		newargs[0] = args[0]
		newargs[1] = RegOffset{Reg(X2), Simm{int32(args[1].(Uimm).Imm), true, 12}}

	// riscv double precision floating point compressed instructions
	case C_FLD:
		f.op = FLD
		newargs[0] = args[0]
		newargs[1] = RegOffset{args[1].(Reg), Simm{int32(args[2].(Uimm).Imm), true, 12}}

	case C_FSD:
		f.op = FSD
		newargs[0] = args[1]
		newargs[1] = RegOffset{args[0].(Reg), Simm{int32(args[2].(Uimm).Imm), true, 12}}

	case C_FLDSP:
		f.op = FLD
		newargs[0] = args[0]
		newargs[1] = RegOffset{Reg(X2), Simm{int32(args[1].(Uimm).Imm), true, 12}}

	case C_FSDSP:
		f.op = FSD
		newargs[0] = args[0]
		newargs[1] = RegOffset{Reg(X2), Simm{int32(args[1].(Uimm).Imm), true, 12}}

	case C_UNIMP:
		f.op = CSRRW
		newargs[0] = Reg(X0)
		newargs[1] = CSR(CYCLE)
		newargs[2] = Reg(X0)
	}
	return newargs
}
