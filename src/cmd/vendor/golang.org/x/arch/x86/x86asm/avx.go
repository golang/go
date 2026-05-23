// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86asm

import (
	"encoding/binary"
	"errors"
)

// This file contains the handling of AVX instructions, based on
// tables (avx_tables.go) generated from the XED data.

//go:generate go run _gen/genavx.go -o avx_tables.go

// decodeAVX decodes AVX/AVX2/AVX-512 instructions.
// It is called from decode1 when a VEX or EVEX prefix is detected.
func decodeAVX(src []byte, pos int, vex Prefix, vexIndex int, inst Inst, mode int) (Inst, error) {
	var vexP, vexL, vexW uint8
	var mapSelect uint8
	var vvvv uint8
	var vexR, vexX, vexB uint8 // Inverted from VEX/EVEX
	var evex bool
	var evexR_prime, evexV_prime uint8 // Inverted
	var evex_aaa, evex_z uint8
	var evex_b uint8

	vexR = 1
	vexX = 1
	vexB = 1 // Default to 1 (inactive inverted)
	evexR_prime = 1
	evexV_prime = 1

	if vex == 0xC5 { // 2-byte VEX
		b1 := uint8(inst.Prefix[vexIndex+1])
		vexR = (b1 >> 7) & 1
		vvvv = (b1 >> 3) & 0xF
		vexL = (b1 >> 2) & 1
		vexP = b1 & 3
		mapSelect = 1 // 0F
	} else if vex == 0xC4 { // 3-byte VEX
		b1 := uint8(inst.Prefix[vexIndex+1])
		b2 := uint8(inst.Prefix[vexIndex+2])
		vexR = (b1 >> 7) & 1
		vexX = (b1 >> 6) & 1
		vexB = (b1 >> 5) & 1
		mapSelect = b1 & 0x1F

		vexW = (b2 >> 7) & 1
		vvvv = (b2 >> 3) & 0xF
		vexL = (b2 >> 2) & 1
		vexP = b2 & 3
	} else if vex == 0x62 { // EVEX
		evex = true
		b1 := uint8(inst.Prefix[vexIndex+1])
		b2 := uint8(inst.Prefix[vexIndex+2])
		b3 := uint8(inst.Prefix[vexIndex+3])

		vexR = (b1 >> 7) & 1
		vexX = (b1 >> 6) & 1
		vexB = (b1 >> 5) & 1
		evexR_prime = (b1 >> 4) & 1
		mapSelect = b1 & 3

		vexW = (b2 >> 7) & 1
		vvvv = (b2 >> 3) & 0xF
		vexP = b2 & 3

		evex_z = (b3 >> 7) & 1
		vexL = (b3 >> 5) & 3
		evex_b = (b3 >> 4) & 1
		evexV_prime = (b3 >> 3) & 1
		evex_aaa = b3 & 7
	}

	_ = evex_z // TODO: use zeroing mask if needed for output

	opbyte := src[pos]
	pos++

	var candidates []*avxOptab
	switch mapSelect {
	case 1:
		candidates = avxMap0F[opbyte]
	case 2:
		candidates = avxMap0F38[opbyte]
	case 3:
		candidates = avxMap0F3A[opbyte]
	}

	if len(candidates) == 0 {
		return inst, errors.New("unknown AVX Opcode")
	}

	var modrm uint8
	var haveModRM bool
	if pos < len(src) {
		modrm = src[pos]
		haveModRM = true
	}

	var match *avxOptab

	for i := range candidates {
		c := candidates[i]
		if evex != c.evex {
			continue
		}
		c_vexP := c.vexP
		p_match := false
		switch c_vexP {
		case 0:
			p_match = vexP == 0
		case 1:
			p_match = vexP == 1
		case 2:
			p_match = vexP == 3
		case 3:
			p_match = vexP == 2
		}
		if !p_match {
			continue
		}

		match_vexL := vexL
		if evex && evex_b != 0 && haveModRM && (modrm>>6) == 3 {
			hasZmm := false
			for j := range candidates {
				if candidates[j].evex == c.evex && candidates[j].vexP == c.vexP && candidates[j].vexW == c.vexW && candidates[j].vexL == 2 {
					hasZmm = true
					break
				}
			}
			if hasZmm {
				match_vexL = 2
			} else {
				match_vexL = 0
			}
		}
		if c.vexL != match_vexL {
			continue
		}
		if c.vexW != vexW {
			continue
		}

		if haveModRM {
			mod := modrm >> 6
			reg := (modrm >> 3) & 7
			if c.opdigit != -1 && reg != uint8(c.opdigit) {
				continue
			}
			if c.ismem == 1 && mod == 3 {
				continue
			}
			if c.ismem == 0 && mod != 3 {
				continue
			}
		}
		match = c
		break
	}

	if match == nil {
		return Inst{Len: 1}, ErrUnrecognized
	}

	inst.Op = match.op

	var mod, reg, rm uint8
	var sib uint8
	var haveSIB bool
	var mem Mem
	var addrMode = mode

	if haveModRM {
		mod = modrm >> 6
		reg = (modrm >> 3) & 7
		rm = modrm & 7
		pos++

		if mod != 3 && rm == 4 {
			if pos >= len(src) {
				return inst, errors.New("truncated")
			}
			sib = src[pos]
			haveSIB = true
			pos++
		}

		var disp int64
		if mod == 0 && (rm == 5 || (haveSIB && (sib&7) == 5)) || mod == 2 {
			if pos+4 > len(src) {
				return inst, errors.New("truncated")
			}
			disp = int64(int32(binary.LittleEndian.Uint32(src[pos:])))
			pos += 4
		} else if mod == 1 {
			if pos >= len(src) {
				return inst, errors.New("truncated")
			}
			disp = int64(int8(src[pos]))
			pos++
			if evex && match.dispScale > 0 {
				scale := match.dispScale
				if evex_b != 0 && match.bcstScale > 0 {
					scale = match.bcstScale
				}
				disp *= int64(scale)
			}
		}
		mem.Disp = disp

		if haveSIB {
			scale := sib >> 6
			index := (sib >> 3) & 7
			base := sib & 7

			if vexX == 0 {
				index |= 8
			}
			if vexB == 0 {
				base |= 8
			}

			mem.Scale = 1 << uint(scale)
			if index != 4 {
				mem.Index = baseRegForBits(addrMode) + Reg(index)
			}
			if base&7 != 5 || mod != 0 {
				mem.Base = baseRegForBits(addrMode) + Reg(base)
			}
		} else {
			if vexB == 0 {
				rm |= 8
			}

			if mod != 3 {
				if !(mod == 0 && rm&7 == 5) {
					mem.Base = baseRegForBits(addrMode) + Reg(rm)
				}
			}
		}
	}

	// Decode Args
	for i, argType := range match.args {
		if argType == argNone {
			continue
		}
		var arg Arg

		switch argType {
		case argImm8:
			if pos >= len(src) {
				return inst, errors.New("truncated")
			}
			arg = Imm(src[pos])
			pos++
		case argImm8u:
			if pos >= len(src) {
				return inst, errors.New("truncated")
			}
			arg = Imm(src[pos])
			pos++
		case argXmm_SE, argYmm_SE:
			if pos >= len(src) {
				return inst, errors.New("truncated")
			}
			idx := (src[pos] >> 4) & 0xF
			if argType == argXmm_SE {
				arg = X0 + Reg(idx)
			} else {
				arg = Y0 + Reg(idx)
			}
			pos++
		case argGPR_R, argGPR32_R, argGPR64_R:
			idx := reg
			if vexR == 0 {
				idx |= 8
			}
			base := baseRegForBits(mode)
			if argType == argGPR32_R {
				base = EAX
			} else if argType == argGPR64_R {
				base = RAX
			}
			arg = base + Reg(idx)
		case argGPR_N, argGPR32_N, argGPR64_N:
			idx := ^vvvv & 15 // 1s complement
			base := baseRegForBits(mode)
			if argType == argGPR32_N {
				base = EAX
			} else if argType == argGPR64_N {
				base = RAX
			} else if vexW == 1 {
				base = RAX
			}
			arg = base + Reg(idx)
		case argGPR_B, argGPR32_B, argGPR64_B:
			idx := rm
			if vexB == 0 {
				idx |= 8
			}
			base := baseRegForBits(mode)
			if argType == argGPR32_B {
				base = EAX
			} else if argType == argGPR64_B {
				base = RAX
			}
			arg = base + Reg(idx)
		// VEX/EVEX encoding uses inverted bits for register specifiers (0 means bit is set).
		case argXmm_R, argXmmEvex_R:
			idx := reg
			if vexR == 0 {
				idx |= 8
			}
			if evex && evexR_prime == 0 {
				idx |= 16
			}
			arg = X0 + Reg(idx)
		case argXmm_B, argXmmEvex_B:
			idx := rm
			if vexB == 0 {
				idx |= 8
			}
			if evex && vexX == 0 {
				idx |= 16
			}
			arg = X0 + Reg(idx)
		case argXmm_N, argXmmEvex_N:
			idx := 15 - vvvv
			if evex && evexV_prime == 0 {
				idx |= 16
			}
			arg = X0 + Reg(idx)
		case argYmm_R, argYmmEvex_R:
			idx := reg
			if vexR == 0 {
				idx |= 8
			}
			if evex && evexR_prime == 0 {
				idx |= 16
			}
			arg = Y0 + Reg(idx)
		case argYmm_B, argYmmEvex_B:
			idx := rm
			if vexB == 0 {
				idx |= 8
			}
			if evex && vexX == 0 {
				idx |= 16
			}
			arg = Y0 + Reg(idx)
		case argYmm_N, argYmmEvex_N:
			idx := 15 - vvvv
			if evex && evexV_prime == 0 {
				idx |= 16
			}
			arg = Y0 + Reg(idx)
		case argZmm_R:
			vl := vexL
			if evex && evex_b != 0 && match.ismem == 0 {
				vl = 2 // RC / SAE implies 512-bit vector length
			}
			idx := reg
			if vexR == 0 {
				idx |= 8
			}
			if evexR_prime == 0 {
				idx |= 16
			}
			if vl == 0 {
				arg = X0 + Reg(idx)
			} else if vl == 1 {
				arg = Y0 + Reg(idx)
			} else {
				arg = Z0 + Reg(idx)
			}
		case argZmm_B:
			vl := vexL
			if evex && evex_b != 0 && match.ismem == 0 {
				vl = 2
			}
			if match.ismem != 0 {
				arg = mem
			} else {
				idx := rm
				if vexB == 0 {
					idx |= 8
				}
				if vexX == 0 {
					idx |= 16
				}
				if vl == 0 {
					arg = X0 + Reg(idx)
				} else if vl == 1 {
					arg = Y0 + Reg(idx)
				} else {
					arg = Z0 + Reg(idx)
				}
			}
		case argZmm_N:
			vl := vexL
			if evex && evex_b != 0 && match.ismem == 0 {
				vl = 2
			}
			idx := 15 - vvvv
			if evexV_prime == 0 {
				idx |= 16
			}
			if vl == 0 {
				arg = X0 + Reg(idx)
			} else if vl == 1 {
				arg = Y0 + Reg(idx)
			} else {
				arg = Z0 + Reg(idx)
			}
		case argK_R:
			arg = K0 + Reg(reg&7)
		case argK_B:
			arg = K0 + Reg(rm&7)
		case argK_N:
			arg = K0 + Reg((15-vvvv)&7)
		case argKmask:
			if evex_aaa != 0 {
				arg = K0 + Reg(evex_aaa)
			}
		case argKnot0:
			if evex_aaa == 0 {
				return inst, errors.New("k0 mask not allowed")
			}
			arg = K0 + Reg(evex_aaa)
		case argM:
			arg = mem
		}

		if arg != nil {
			inst.Args[i] = arg
		}
	}

	n := 0
	for i := range len(inst.Args) {
		if inst.Args[i] != nil {
			if n != i {
				inst.Args[n] = inst.Args[i]
				inst.Args[i] = nil
			}
			n++
		}
	}
	inst.MemBytes = int(match.memBytes)
	if inst.MemBytes == 0 && match.ismem != 0 && match.dispScale != 0 {
		inst.MemBytes = int(match.dispScale)
	}
	if evex {
		inst.Zeroing = evex_z != 0
		if evex_b != 0 {
			if match.bcstScale > 0 {
				inst.Broadcast = true
				inst.MemBytes = int(match.bcstScale)
			} else if match.ismem == 0 {
				inst.SAE = true
				inst.Rounding = int8(vexL)
			}
		}
	}
	inst.Len = pos

	if match.vsib && haveSIB {
		fixVSIB(&inst, vexL, evex, evexV_prime, vexX, sib)
	}

	return inst, nil
}

// fixVSIB calculates the correct vector register size based on data and index element sizes.
func fixVSIB(inst *Inst, vexL uint8, evex bool, evexV_prime uint8, vexX uint8, sib uint8) {
	var indexElemBits, dataElemBits int
	switch inst.Op {
	case VPGATHERDD, VGATHERDPS, VPSCATTERDD, VSCATTERDPS:
		indexElemBits = 32
		dataElemBits = 32
	case VPGATHERDQ, VGATHERDPD, VPSCATTERDQ, VSCATTERDPD:
		indexElemBits = 32
		dataElemBits = 64
	case VPGATHERQD, VGATHERQPS, VPSCATTERQD, VSCATTERQPS:
		indexElemBits = 64
		dataElemBits = 32
	case VPGATHERQQ, VGATHERQPD, VPSCATTERQQ, VSCATTERQPD:
		indexElemBits = 64
		dataElemBits = 64
	case VGATHERPF0DPS, VGATHERPF1DPS, VSCATTERPF0DPS, VSCATTERPF1DPS:
		indexElemBits = 32
		dataElemBits = 32
	case VGATHERPF0DPD, VGATHERPF1DPD, VSCATTERPF0DPD, VSCATTERPF1DPD:
		indexElemBits = 32
		dataElemBits = 64
	case VGATHERPF0QPS, VGATHERPF1QPS, VSCATTERPF0QPS, VSCATTERPF1QPS:
		indexElemBits = 64
		dataElemBits = 32
	case VGATHERPF0QPD, VGATHERPF1QPD, VSCATTERPF0QPD, VSCATTERPF1QPD:
		indexElemBits = 64
		dataElemBits = 64
	default:
		return
	}

	maxBits := 128 << vexL

	var destBits, indexVectorBits int
	if indexElemBits > dataElemBits {
		indexVectorBits = maxBits
		numElements := indexVectorBits / indexElemBits
		destBits = numElements * dataElemBits
	} else if dataElemBits > indexElemBits {
		destBits = maxBits
		numElements := destBits / dataElemBits
		indexVectorBits = numElements * indexElemBits
	} else {
		indexVectorBits = maxBits
		destBits = maxBits
	}

	// Override MemBytes to match objdump's output expectation (memory accessed is based on dest size)
	inst.MemBytes = destBits / 8

	if indexVectorBits < 128 {
		indexVectorBits = 128
	}

	var baseReg Reg
	switch indexVectorBits {
	case 128:
		baseReg = X0
	case 256:
		baseReg = Y0
	case 512:
		baseReg = Z0
	default:
		baseReg = X0
	}

	for i, arg := range inst.Args {
		if mem, ok := arg.(Mem); ok {
			idx := (sib >> 3) & 7
			if vexX == 0 {
				idx |= 8
			}
			if evex && evexV_prime == 0 {
				idx |= 16
			}
			mem.Index = baseReg + Reg(idx)
			inst.Args[i] = mem
			break
		}
	}
}

// argType defines how to decode an argument.
// It corresponds to the arg type notation in XED.
type argType uint8

const (
	argNone argType = iota
	argImm8
	argImm8u
	argImm16
	argImm32
	argImm64

	// GPRs
	argGPR_R   // ModRM.reg (default mode size)
	argGPR_B   // ModRM.rm (default mode size)
	argGPR_N   // VEX.vvvv (default mode size)
	argGPR32_R // ModRM.reg (32-bit forced)
	argGPR32_B // ModRM.rm (32-bit forced)
	argGPR32_N // VEX.vvvv (32-bit forced)
	argGPR64_R // ModRM.reg (64-bit forced)
	argGPR64_B // ModRM.rm (64-bit forced)
	argGPR64_N // VEX.vvvv (64-bit forced)

	// XMM
	argXmm_R
	argXmm_B
	argXmm_N
	argXmmEvex_R
	argXmmEvex_B
	argXmmEvex_N
	argXmm_SE // is4 immediate

	// YMM
	argYmm_R
	argYmm_B
	argYmm_N
	argYmmEvex_R
	argYmmEvex_B
	argYmmEvex_N
	argYmm_SE

	// ZMM
	argZmm_R
	argZmm_B
	argZmm_N

	// Mask
	argK_R
	argK_B
	argK_N

	argM     // Memory operand (ModRM.rm)
	argKnot0 // Mask register k1-k7
	argKmask // Mask register k0-k7
)

// hasRC returns true if the instruction supports static rounding control in AVX-512.
func hasRC(op Op) bool {
	switch op {
	case VADDPD, VADDPS, VADDSD, VADDSS,
		VSUBPD, VSUBPS, VSUBSD, VSUBSS,
		VMULPD, VMULPS, VMULSD, VMULSS,
		VDIVPD, VDIVPS, VDIVSD, VDIVSS,
		VSQRTPD, VSQRTPS, VSQRTSD, VSQRTSS,
		VSCALEFPD, VSCALEFPS, VSCALEFSD, VSCALEFSS,
		VFMADD132PD, VFMADD132PS, VFMADD132SD, VFMADD132SS,
		VFMADD213PD, VFMADD213PS, VFMADD213SD, VFMADD213SS,
		VFMADD231PD, VFMADD231PS, VFMADD231SD, VFMADD231SS,
		VFMSUB132PD, VFMSUB132PS, VFMSUB132SD, VFMSUB132SS,
		VFMSUB213PD, VFMSUB213PS, VFMSUB213SD, VFMSUB213SS,
		VFMSUB231PD, VFMSUB231PS, VFMSUB231SD, VFMSUB231SS,
		VFNMADD132PD, VFNMADD132PS, VFNMADD132SD, VFNMADD132SS,
		VFNMADD213PD, VFNMADD213PS, VFNMADD213SD, VFNMADD213SS,
		VFNMADD231PD, VFNMADD231PS, VFNMADD231SD, VFNMADD231SS,
		VFNMSUB132PD, VFNMSUB132PS, VFNMSUB132SD, VFNMSUB132SS,
		VFNMSUB213PD, VFNMSUB213PS, VFNMSUB213SD, VFNMSUB213SS,
		VFNMSUB231PD, VFNMSUB231PS, VFNMSUB231SD, VFNMSUB231SS,
		VFMADDSUB132PD, VFMADDSUB132PS, VFMADDSUB213PD, VFMADDSUB213PS,
		VFMADDSUB231PD, VFMADDSUB231PS, VFMSUBADD132PD, VFMSUBADD132PS,
		VFMSUBADD213PD, VFMSUBADD213PS, VFMSUBADD231PD, VFMSUBADD231PS,
		VCVTPS2DQ, VCVTPD2DQ, VCVTPS2UDQ, VCVTPD2UDQ,
		VCVTPS2QQ, VCVTPD2QQ, VCVTPS2UQQ, VCVTPD2UQQ,
		VCVTUDQ2PS, VCVTUDQ2PD, VCVTQQ2PS, VCVTQQ2PD,
		VCVTUQQ2PS, VCVTUQQ2PD, VCVTDQ2PS, VCVTDQ2PD,
		VCVTPS2PD, VCVTPD2PS, VCVTSS2SD, VCVTSD2SS,
		VCVTUSI2SS, VCVTUSI2SD, VCVTSI2SS, VCVTSI2SD,
		VCVTSS2USI, VCVTSD2USI, VCVTSS2SI, VCVTSD2SI,
		VPERMT2PD, VPERMT2PS, VPERMI2PD, VPERMI2PS:
		return true
	}
	return false
}
