// Code generated from gen/RISCV64.rules; DO NOT EDIT.
// generated with: cd gen; go run *.go

package ssa

import "math"
import "cmd/compile/internal/types"

func rewriteValueRISCV64(v *Value) bool {
	switch v.Op {
	case OpAbs:
		v.Op = OpRISCV64FABSD
		return true
	case OpAdd16:
		v.Op = OpRISCV64ADD
		return true
	case OpAdd32:
		v.Op = OpRISCV64ADD
		return true
	case OpAdd32F:
		v.Op = OpRISCV64FADDS
		return true
	case OpAdd64:
		v.Op = OpRISCV64ADD
		return true
	case OpAdd64F:
		v.Op = OpRISCV64FADDD
		return true
	case OpAdd8:
		v.Op = OpRISCV64ADD
		return true
	case OpAddPtr:
		v.Op = OpRISCV64ADD
		return true
	case OpAddr:
		return rewriteValueRISCV64_OpAddr(v)
	case OpAnd16:
		v.Op = OpRISCV64AND
		return true
	case OpAnd32:
		v.Op = OpRISCV64AND
		return true
	case OpAnd64:
		v.Op = OpRISCV64AND
		return true
	case OpAnd8:
		v.Op = OpRISCV64AND
		return true
	case OpAndB:
		v.Op = OpRISCV64AND
		return true
	case OpAtomicAdd32:
		v.Op = OpRISCV64LoweredAtomicAdd32
		return true
	case OpAtomicAdd64:
		v.Op = OpRISCV64LoweredAtomicAdd64
		return true
	case OpAtomicAnd32:
		v.Op = OpRISCV64LoweredAtomicAnd32
		return true
	case OpAtomicAnd8:
		return rewriteValueRISCV64_OpAtomicAnd8(v)
	case OpAtomicCompareAndSwap32:
		v.Op = OpRISCV64LoweredAtomicCas32
		return true
	case OpAtomicCompareAndSwap64:
		v.Op = OpRISCV64LoweredAtomicCas64
		return true
	case OpAtomicExchange32:
		v.Op = OpRISCV64LoweredAtomicExchange32
		return true
	case OpAtomicExchange64:
		v.Op = OpRISCV64LoweredAtomicExchange64
		return true
	case OpAtomicLoad32:
		v.Op = OpRISCV64LoweredAtomicLoad32
		return true
	case OpAtomicLoad64:
		v.Op = OpRISCV64LoweredAtomicLoad64
		return true
	case OpAtomicLoad8:
		v.Op = OpRISCV64LoweredAtomicLoad8
		return true
	case OpAtomicLoadPtr:
		v.Op = OpRISCV64LoweredAtomicLoad64
		return true
	case OpAtomicOr32:
		v.Op = OpRISCV64LoweredAtomicOr32
		return true
	case OpAtomicOr8:
		return rewriteValueRISCV64_OpAtomicOr8(v)
	case OpAtomicStore32:
		v.Op = OpRISCV64LoweredAtomicStore32
		return true
	case OpAtomicStore64:
		v.Op = OpRISCV64LoweredAtomicStore64
		return true
	case OpAtomicStore8:
		v.Op = OpRISCV64LoweredAtomicStore8
		return true
	case OpAtomicStorePtrNoWB:
		v.Op = OpRISCV64LoweredAtomicStore64
		return true
	case OpAvg64u:
		return rewriteValueRISCV64_OpAvg64u(v)
	case OpClosureCall:
		v.Op = OpRISCV64CALLclosure
		return true
	case OpCom16:
		v.Op = OpRISCV64NOT
		return true
	case OpCom32:
		v.Op = OpRISCV64NOT
		return true
	case OpCom64:
		v.Op = OpRISCV64NOT
		return true
	case OpCom8:
		v.Op = OpRISCV64NOT
		return true
	case OpConst16:
		return rewriteValueRISCV64_OpConst16(v)
	case OpConst32:
		return rewriteValueRISCV64_OpConst32(v)
	case OpConst32F:
		return rewriteValueRISCV64_OpConst32F(v)
	case OpConst64:
		return rewriteValueRISCV64_OpConst64(v)
	case OpConst64F:
		return rewriteValueRISCV64_OpConst64F(v)
	case OpConst8:
		return rewriteValueRISCV64_OpConst8(v)
	case OpConstBool:
		return rewriteValueRISCV64_OpConstBool(v)
	case OpConstNil:
		return rewriteValueRISCV64_OpConstNil(v)
	case OpConvert:
		v.Op = OpRISCV64MOVconvert
		return true
	case OpCopysign:
		v.Op = OpRISCV64FSGNJD
		return true
	case OpCvt32Fto32:
		v.Op = OpRISCV64FCVTWS
		return true
	case OpCvt32Fto64:
		v.Op = OpRISCV64FCVTLS
		return true
	case OpCvt32Fto64F:
		v.Op = OpRISCV64FCVTDS
		return true
	case OpCvt32to32F:
		v.Op = OpRISCV64FCVTSW
		return true
	case OpCvt32to64F:
		v.Op = OpRISCV64FCVTDW
		return true
	case OpCvt64Fto32:
		v.Op = OpRISCV64FCVTWD
		return true
	case OpCvt64Fto32F:
		v.Op = OpRISCV64FCVTSD
		return true
	case OpCvt64Fto64:
		v.Op = OpRISCV64FCVTLD
		return true
	case OpCvt64to32F:
		v.Op = OpRISCV64FCVTSL
		return true
	case OpCvt64to64F:
		v.Op = OpRISCV64FCVTDL
		return true
	case OpCvtBoolToUint8:
		v.Op = OpCopy
		return true
	case OpDiv16:
		return rewriteValueRISCV64_OpDiv16(v)
	case OpDiv16u:
		return rewriteValueRISCV64_OpDiv16u(v)
	case OpDiv32:
		return rewriteValueRISCV64_OpDiv32(v)
	case OpDiv32F:
		v.Op = OpRISCV64FDIVS
		return true
	case OpDiv32u:
		v.Op = OpRISCV64DIVUW
		return true
	case OpDiv64:
		return rewriteValueRISCV64_OpDiv64(v)
	case OpDiv64F:
		v.Op = OpRISCV64FDIVD
		return true
	case OpDiv64u:
		v.Op = OpRISCV64DIVU
		return true
	case OpDiv8:
		return rewriteValueRISCV64_OpDiv8(v)
	case OpDiv8u:
		return rewriteValueRISCV64_OpDiv8u(v)
	case OpEq16:
		return rewriteValueRISCV64_OpEq16(v)
	case OpEq32:
		return rewriteValueRISCV64_OpEq32(v)
	case OpEq32F:
		v.Op = OpRISCV64FEQS
		return true
	case OpEq64:
		return rewriteValueRISCV64_OpEq64(v)
	case OpEq64F:
		v.Op = OpRISCV64FEQD
		return true
	case OpEq8:
		return rewriteValueRISCV64_OpEq8(v)
	case OpEqB:
		return rewriteValueRISCV64_OpEqB(v)
	case OpEqPtr:
		return rewriteValueRISCV64_OpEqPtr(v)
	case OpFMA:
		v.Op = OpRISCV64FMADDD
		return true
	case OpGetCallerPC:
		v.Op = OpRISCV64LoweredGetCallerPC
		return true
	case OpGetCallerSP:
		v.Op = OpRISCV64LoweredGetCallerSP
		return true
	case OpGetClosurePtr:
		v.Op = OpRISCV64LoweredGetClosurePtr
		return true
	case OpHmul32:
		return rewriteValueRISCV64_OpHmul32(v)
	case OpHmul32u:
		return rewriteValueRISCV64_OpHmul32u(v)
	case OpHmul64:
		v.Op = OpRISCV64MULH
		return true
	case OpHmul64u:
		v.Op = OpRISCV64MULHU
		return true
	case OpInterCall:
		v.Op = OpRISCV64CALLinter
		return true
	case OpIsInBounds:
		v.Op = OpLess64U
		return true
	case OpIsNonNil:
		v.Op = OpRISCV64SNEZ
		return true
	case OpIsSliceInBounds:
		v.Op = OpLeq64U
		return true
	case OpLeq16:
		return rewriteValueRISCV64_OpLeq16(v)
	case OpLeq16U:
		return rewriteValueRISCV64_OpLeq16U(v)
	case OpLeq32:
		return rewriteValueRISCV64_OpLeq32(v)
	case OpLeq32F:
		v.Op = OpRISCV64FLES
		return true
	case OpLeq32U:
		return rewriteValueRISCV64_OpLeq32U(v)
	case OpLeq64:
		return rewriteValueRISCV64_OpLeq64(v)
	case OpLeq64F:
		v.Op = OpRISCV64FLED
		return true
	case OpLeq64U:
		return rewriteValueRISCV64_OpLeq64U(v)
	case OpLeq8:
		return rewriteValueRISCV64_OpLeq8(v)
	case OpLeq8U:
		return rewriteValueRISCV64_OpLeq8U(v)
	case OpLess16:
		return rewriteValueRISCV64_OpLess16(v)
	case OpLess16U:
		return rewriteValueRISCV64_OpLess16U(v)
	case OpLess32:
		return rewriteValueRISCV64_OpLess32(v)
	case OpLess32F:
		v.Op = OpRISCV64FLTS
		return true
	case OpLess32U:
		return rewriteValueRISCV64_OpLess32U(v)
	case OpLess64:
		v.Op = OpRISCV64SLT
		return true
	case OpLess64F:
		v.Op = OpRISCV64FLTD
		return true
	case OpLess64U:
		v.Op = OpRISCV64SLTU
		return true
	case OpLess8:
		return rewriteValueRISCV64_OpLess8(v)
	case OpLess8U:
		return rewriteValueRISCV64_OpLess8U(v)
	case OpLoad:
		return rewriteValueRISCV64_OpLoad(v)
	case OpLocalAddr:
		return rewriteValueRISCV64_OpLocalAddr(v)
	case OpLsh16x16:
		return rewriteValueRISCV64_OpLsh16x16(v)
	case OpLsh16x32:
		return rewriteValueRISCV64_OpLsh16x32(v)
	case OpLsh16x64:
		return rewriteValueRISCV64_OpLsh16x64(v)
	case OpLsh16x8:
		return rewriteValueRISCV64_OpLsh16x8(v)
	case OpLsh32x16:
		return rewriteValueRISCV64_OpLsh32x16(v)
	case OpLsh32x32:
		return rewriteValueRISCV64_OpLsh32x32(v)
	case OpLsh32x64:
		return rewriteValueRISCV64_OpLsh32x64(v)
	case OpLsh32x8:
		return rewriteValueRISCV64_OpLsh32x8(v)
	case OpLsh64x16:
		return rewriteValueRISCV64_OpLsh64x16(v)
	case OpLsh64x32:
		return rewriteValueRISCV64_OpLsh64x32(v)
	case OpLsh64x64:
		return rewriteValueRISCV64_OpLsh64x64(v)
	case OpLsh64x8:
		return rewriteValueRISCV64_OpLsh64x8(v)
	case OpLsh8x16:
		return rewriteValueRISCV64_OpLsh8x16(v)
	case OpLsh8x32:
		return rewriteValueRISCV64_OpLsh8x32(v)
	case OpLsh8x64:
		return rewriteValueRISCV64_OpLsh8x64(v)
	case OpLsh8x8:
		return rewriteValueRISCV64_OpLsh8x8(v)
	case OpMod16:
		return rewriteValueRISCV64_OpMod16(v)
	case OpMod16u:
		return rewriteValueRISCV64_OpMod16u(v)
	case OpMod32:
		return rewriteValueRISCV64_OpMod32(v)
	case OpMod32u:
		v.Op = OpRISCV64REMUW
		return true
	case OpMod64:
		return rewriteValueRISCV64_OpMod64(v)
	case OpMod64u:
		v.Op = OpRISCV64REMU
		return true
	case OpMod8:
		return rewriteValueRISCV64_OpMod8(v)
	case OpMod8u:
		return rewriteValueRISCV64_OpMod8u(v)
	case OpMove:
		return rewriteValueRISCV64_OpMove(v)
	case OpMul16:
		return rewriteValueRISCV64_OpMul16(v)
	case OpMul32:
		v.Op = OpRISCV64MULW
		return true
	case OpMul32F:
		v.Op = OpRISCV64FMULS
		return true
	case OpMul64:
		v.Op = OpRISCV64MUL
		return true
	case OpMul64F:
		v.Op = OpRISCV64FMULD
		return true
	case OpMul64uhilo:
		v.Op = OpRISCV64LoweredMuluhilo
		return true
	case OpMul64uover:
		v.Op = OpRISCV64LoweredMuluover
		return true
	case OpMul8:
		return rewriteValueRISCV64_OpMul8(v)
	case OpNeg16:
		v.Op = OpRISCV64NEG
		return true
	case OpNeg32:
		v.Op = OpRISCV64NEG
		return true
	case OpNeg32F:
		v.Op = OpRISCV64FNEGS
		return true
	case OpNeg64:
		v.Op = OpRISCV64NEG
		return true
	case OpNeg64F:
		v.Op = OpRISCV64FNEGD
		return true
	case OpNeg8:
		v.Op = OpRISCV64NEG
		return true
	case OpNeq16:
		return rewriteValueRISCV64_OpNeq16(v)
	case OpNeq32:
		return rewriteValueRISCV64_OpNeq32(v)
	case OpNeq32F:
		v.Op = OpRISCV64FNES
		return true
	case OpNeq64:
		return rewriteValueRISCV64_OpNeq64(v)
	case OpNeq64F:
		v.Op = OpRISCV64FNED
		return true
	case OpNeq8:
		return rewriteValueRISCV64_OpNeq8(v)
	case OpNeqB:
		v.Op = OpRISCV64XOR
		return true
	case OpNeqPtr:
		return rewriteValueRISCV64_OpNeqPtr(v)
	case OpNilCheck:
		v.Op = OpRISCV64LoweredNilCheck
		return true
	case OpNot:
		v.Op = OpRISCV64SEQZ
		return true
	case OpOffPtr:
		return rewriteValueRISCV64_OpOffPtr(v)
	case OpOr16:
		v.Op = OpRISCV64OR
		return true
	case OpOr32:
		v.Op = OpRISCV64OR
		return true
	case OpOr64:
		v.Op = OpRISCV64OR
		return true
	case OpOr8:
		v.Op = OpRISCV64OR
		return true
	case OpOrB:
		v.Op = OpRISCV64OR
		return true
	case OpPanicBounds:
		return rewriteValueRISCV64_OpPanicBounds(v)
	case OpRISCV64ADD:
		return rewriteValueRISCV64_OpRISCV64ADD(v)
	case OpRISCV64ADDI:
		return rewriteValueRISCV64_OpRISCV64ADDI(v)
	case OpRISCV64AND:
		return rewriteValueRISCV64_OpRISCV64AND(v)
	case OpRISCV64ANDI:
		return rewriteValueRISCV64_OpRISCV64ANDI(v)
	case OpRISCV64FMADDD:
		return rewriteValueRISCV64_OpRISCV64FMADDD(v)
	case OpRISCV64FMSUBD:
		return rewriteValueRISCV64_OpRISCV64FMSUBD(v)
	case OpRISCV64FNMADDD:
		return rewriteValueRISCV64_OpRISCV64FNMADDD(v)
	case OpRISCV64FNMSUBD:
		return rewriteValueRISCV64_OpRISCV64FNMSUBD(v)
	case OpRISCV64MOVBUload:
		return rewriteValueRISCV64_OpRISCV64MOVBUload(v)
	case OpRISCV64MOVBUreg:
		return rewriteValueRISCV64_OpRISCV64MOVBUreg(v)
	case OpRISCV64MOVBload:
		return rewriteValueRISCV64_OpRISCV64MOVBload(v)
	case OpRISCV64MOVBreg:
		return rewriteValueRISCV64_OpRISCV64MOVBreg(v)
	case OpRISCV64MOVBstore:
		return rewriteValueRISCV64_OpRISCV64MOVBstore(v)
	case OpRISCV64MOVBstorezero:
		return rewriteValueRISCV64_OpRISCV64MOVBstorezero(v)
	case OpRISCV64MOVDload:
		return rewriteValueRISCV64_OpRISCV64MOVDload(v)
	case OpRISCV64MOVDnop:
		return rewriteValueRISCV64_OpRISCV64MOVDnop(v)
	case OpRISCV64MOVDreg:
		return rewriteValueRISCV64_OpRISCV64MOVDreg(v)
	case OpRISCV64MOVDstore:
		return rewriteValueRISCV64_OpRISCV64MOVDstore(v)
	case OpRISCV64MOVDstorezero:
		return rewriteValueRISCV64_OpRISCV64MOVDstorezero(v)
	case OpRISCV64MOVHUload:
		return rewriteValueRISCV64_OpRISCV64MOVHUload(v)
	case OpRISCV64MOVHUreg:
		return rewriteValueRISCV64_OpRISCV64MOVHUreg(v)
	case OpRISCV64MOVHload:
		return rewriteValueRISCV64_OpRISCV64MOVHload(v)
	case OpRISCV64MOVHreg:
		return rewriteValueRISCV64_OpRISCV64MOVHreg(v)
	case OpRISCV64MOVHstore:
		return rewriteValueRISCV64_OpRISCV64MOVHstore(v)
	case OpRISCV64MOVHstorezero:
		return rewriteValueRISCV64_OpRISCV64MOVHstorezero(v)
	case OpRISCV64MOVWUload:
		return rewriteValueRISCV64_OpRISCV64MOVWUload(v)
	case OpRISCV64MOVWUreg:
		return rewriteValueRISCV64_OpRISCV64MOVWUreg(v)
	case OpRISCV64MOVWload:
		return rewriteValueRISCV64_OpRISCV64MOVWload(v)
	case OpRISCV64MOVWreg:
		return rewriteValueRISCV64_OpRISCV64MOVWreg(v)
	case OpRISCV64MOVWstore:
		return rewriteValueRISCV64_OpRISCV64MOVWstore(v)
	case OpRISCV64MOVWstorezero:
		return rewriteValueRISCV64_OpRISCV64MOVWstorezero(v)
	case OpRISCV64NEG:
		return rewriteValueRISCV64_OpRISCV64NEG(v)
	case OpRISCV64NEGW:
		return rewriteValueRISCV64_OpRISCV64NEGW(v)
	case OpRISCV64OR:
		return rewriteValueRISCV64_OpRISCV64OR(v)
	case OpRISCV64ORI:
		return rewriteValueRISCV64_OpRISCV64ORI(v)
	case OpRISCV64SEQZ:
		return rewriteValueRISCV64_OpRISCV64SEQZ(v)
	case OpRISCV64SLL:
		return rewriteValueRISCV64_OpRISCV64SLL(v)
	case OpRISCV64SLLI:
		return rewriteValueRISCV64_OpRISCV64SLLI(v)
	case OpRISCV64SLT:
		return rewriteValueRISCV64_OpRISCV64SLT(v)
	case OpRISCV64SLTI:
		return rewriteValueRISCV64_OpRISCV64SLTI(v)
	case OpRISCV64SLTIU:
		return rewriteValueRISCV64_OpRISCV64SLTIU(v)
	case OpRISCV64SLTU:
		return rewriteValueRISCV64_OpRISCV64SLTU(v)
	case OpRISCV64SNEZ:
		return rewriteValueRISCV64_OpRISCV64SNEZ(v)
	case OpRISCV64SRA:
		return rewriteValueRISCV64_OpRISCV64SRA(v)
	case OpRISCV64SRAI:
		return rewriteValueRISCV64_OpRISCV64SRAI(v)
	case OpRISCV64SRL:
		return rewriteValueRISCV64_OpRISCV64SRL(v)
	case OpRISCV64SRLI:
		return rewriteValueRISCV64_OpRISCV64SRLI(v)
	case OpRISCV64SUB:
		return rewriteValueRISCV64_OpRISCV64SUB(v)
	case OpRISCV64SUBW:
		return rewriteValueRISCV64_OpRISCV64SUBW(v)
	case OpRISCV64XOR:
		return rewriteValueRISCV64_OpRISCV64XOR(v)
	case OpRotateLeft16:
		return rewriteValueRISCV64_OpRotateLeft16(v)
	case OpRotateLeft32:
		return rewriteValueRISCV64_OpRotateLeft32(v)
	case OpRotateLeft64:
		return rewriteValueRISCV64_OpRotateLeft64(v)
	case OpRotateLeft8:
		return rewriteValueRISCV64_OpRotateLeft8(v)
	case OpRound32F:
		v.Op = OpCopy
		return true
	case OpRound64F:
		v.Op = OpCopy
		return true
	case OpRsh16Ux16:
		return rewriteValueRISCV64_OpRsh16Ux16(v)
	case OpRsh16Ux32:
		return rewriteValueRISCV64_OpRsh16Ux32(v)
	case OpRsh16Ux64:
		return rewriteValueRISCV64_OpRsh16Ux64(v)
	case OpRsh16Ux8:
		return rewriteValueRISCV64_OpRsh16Ux8(v)
	case OpRsh16x16:
		return rewriteValueRISCV64_OpRsh16x16(v)
	case OpRsh16x32:
		return rewriteValueRISCV64_OpRsh16x32(v)
	case OpRsh16x64:
		return rewriteValueRISCV64_OpRsh16x64(v)
	case OpRsh16x8:
		return rewriteValueRISCV64_OpRsh16x8(v)
	case OpRsh32Ux16:
		return rewriteValueRISCV64_OpRsh32Ux16(v)
	case OpRsh32Ux32:
		return rewriteValueRISCV64_OpRsh32Ux32(v)
	case OpRsh32Ux64:
		return rewriteValueRISCV64_OpRsh32Ux64(v)
	case OpRsh32Ux8:
		return rewriteValueRISCV64_OpRsh32Ux8(v)
	case OpRsh32x16:
		return rewriteValueRISCV64_OpRsh32x16(v)
	case OpRsh32x32:
		return rewriteValueRISCV64_OpRsh32x32(v)
	case OpRsh32x64:
		return rewriteValueRISCV64_OpRsh32x64(v)
	case OpRsh32x8:
		return rewriteValueRISCV64_OpRsh32x8(v)
	case OpRsh64Ux16:
		return rewriteValueRISCV64_OpRsh64Ux16(v)
	case OpRsh64Ux32:
		return rewriteValueRISCV64_OpRsh64Ux32(v)
	case OpRsh64Ux64:
		return rewriteValueRISCV64_OpRsh64Ux64(v)
	case OpRsh64Ux8:
		return rewriteValueRISCV64_OpRsh64Ux8(v)
	case OpRsh64x16:
		return rewriteValueRISCV64_OpRsh64x16(v)
	case OpRsh64x32:
		return rewriteValueRISCV64_OpRsh64x32(v)
	case OpRsh64x64:
		return rewriteValueRISCV64_OpRsh64x64(v)
	case OpRsh64x8:
		return rewriteValueRISCV64_OpRsh64x8(v)
	case OpRsh8Ux16:
		return rewriteValueRISCV64_OpRsh8Ux16(v)
	case OpRsh8Ux32:
		return rewriteValueRISCV64_OpRsh8Ux32(v)
	case OpRsh8Ux64:
		return rewriteValueRISCV64_OpRsh8Ux64(v)
	case OpRsh8Ux8:
		return rewriteValueRISCV64_OpRsh8Ux8(v)
	case OpRsh8x16:
		return rewriteValueRISCV64_OpRsh8x16(v)
	case OpRsh8x32:
		return rewriteValueRISCV64_OpRsh8x32(v)
	case OpRsh8x64:
		return rewriteValueRISCV64_OpRsh8x64(v)
	case OpRsh8x8:
		return rewriteValueRISCV64_OpRsh8x8(v)
	case OpSelect0:
		return rewriteValueRISCV64_OpSelect0(v)
	case OpSelect1:
		return rewriteValueRISCV64_OpSelect1(v)
	case OpSignExt16to32:
		v.Op = OpRISCV64MOVHreg
		return true
	case OpSignExt16to64:
		v.Op = OpRISCV64MOVHreg
		return true
	case OpSignExt32to64:
		v.Op = OpRISCV64MOVWreg
		return true
	case OpSignExt8to16:
		v.Op = OpRISCV64MOVBreg
		return true
	case OpSignExt8to32:
		v.Op = OpRISCV64MOVBreg
		return true
	case OpSignExt8to64:
		v.Op = OpRISCV64MOVBreg
		return true
	case OpSlicemask:
		return rewriteValueRISCV64_OpSlicemask(v)
	case OpSqrt:
		v.Op = OpRISCV64FSQRTD
		return true
	case OpSqrt32:
		v.Op = OpRISCV64FSQRTS
		return true
	case OpStaticCall:
		v.Op = OpRISCV64CALLstatic
		return true
	case OpStore:
		return rewriteValueRISCV64_OpStore(v)
	case OpSub16:
		v.Op = OpRISCV64SUB
		return true
	case OpSub32:
		v.Op = OpRISCV64SUB
		return true
	case OpSub32F:
		v.Op = OpRISCV64FSUBS
		return true
	case OpSub64:
		v.Op = OpRISCV64SUB
		return true
	case OpSub64F:
		v.Op = OpRISCV64FSUBD
		return true
	case OpSub8:
		v.Op = OpRISCV64SUB
		return true
	case OpSubPtr:
		v.Op = OpRISCV64SUB
		return true
	case OpTailCall:
		v.Op = OpRISCV64CALLtail
		return true
	case OpTrunc16to8:
		v.Op = OpCopy
		return true
	case OpTrunc32to16:
		v.Op = OpCopy
		return true
	case OpTrunc32to8:
		v.Op = OpCopy
		return true
	case OpTrunc64to16:
		v.Op = OpCopy
		return true
	case OpTrunc64to32:
		v.Op = OpCopy
		return true
	case OpTrunc64to8:
		v.Op = OpCopy
		return true
	case OpWB:
		v.Op = OpRISCV64LoweredWB
		return true
	case OpXor16:
		v.Op = OpRISCV64XOR
		return true
	case OpXor32:
		v.Op = OpRISCV64XOR
		return true
	case OpXor64:
		v.Op = OpRISCV64XOR
		return true
	case OpXor8:
		v.Op = OpRISCV64XOR
		return true
	case OpZero:
		return rewriteValueRISCV64_OpZero(v)
	case OpZeroExt16to32:
		v.Op = OpRISCV64MOVHUreg
		return true
	case OpZeroExt16to64:
		v.Op = OpRISCV64MOVHUreg
		return true
	case OpZeroExt32to64:
		v.Op = OpRISCV64MOVWUreg
		return true
	case OpZeroExt8to16:
		v.Op = OpRISCV64MOVBUreg
		return true
	case OpZeroExt8to32:
		v.Op = OpRISCV64MOVBUreg
		return true
	case OpZeroExt8to64:
		v.Op = OpRISCV64MOVBUreg
		return true
	}
	return false
}
func rewriteValueRISCV64_OpAddr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Addr {sym} base)
	// result: (MOVaddr {sym} [0] base)
	for {
		sym := auxToSym(v.Aux)
		base := v_0
		v.reset(OpRISCV64MOVaddr)
		v.AuxInt = int32ToAuxInt(0)
		v.Aux = symToAux(sym)
		v.AddArg(base)
		return true
	}
}
func rewriteValueRISCV64_OpAtomicAnd8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicAnd8 ptr val mem)
	// result: (LoweredAtomicAnd32 (ANDI <typ.Uintptr> [^3] ptr) (NOT <typ.UInt32> (SLL <typ.UInt32> (XORI <typ.UInt32> [0xff] (ZeroExt8to32 val)) (SLLI <typ.UInt64> [3] (ANDI <typ.UInt64> [3] ptr)))) mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpRISCV64LoweredAtomicAnd32)
		v0 := b.NewValue0(v.Pos, OpRISCV64ANDI, typ.Uintptr)
		v0.AuxInt = int64ToAuxInt(^3)
		v0.AddArg(ptr)
		v1 := b.NewValue0(v.Pos, OpRISCV64NOT, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpRISCV64XORI, typ.UInt32)
		v3.AuxInt = int64ToAuxInt(0xff)
		v4 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v4.AddArg(val)
		v3.AddArg(v4)
		v5 := b.NewValue0(v.Pos, OpRISCV64SLLI, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(3)
		v6 := b.NewValue0(v.Pos, OpRISCV64ANDI, typ.UInt64)
		v6.AuxInt = int64ToAuxInt(3)
		v6.AddArg(ptr)
		v5.AddArg(v6)
		v2.AddArg2(v3, v5)
		v1.AddArg(v2)
		v.AddArg3(v0, v1, mem)
		return true
	}
}
func rewriteValueRISCV64_OpAtomicOr8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicOr8 ptr val mem)
	// result: (LoweredAtomicOr32 (ANDI <typ.Uintptr> [^3] ptr) (SLL <typ.UInt32> (ZeroExt8to32 val) (SLLI <typ.UInt64> [3] (ANDI <typ.UInt64> [3] ptr))) mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpRISCV64LoweredAtomicOr32)
		v0 := b.NewValue0(v.Pos, OpRISCV64ANDI, typ.Uintptr)
		v0.AuxInt = int64ToAuxInt(^3)
		v0.AddArg(ptr)
		v1 := b.NewValue0(v.Pos, OpRISCV64SLL, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v2.AddArg(val)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLLI, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(3)
		v4 := b.NewValue0(v.Pos, OpRISCV64ANDI, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(3)
		v4.AddArg(ptr)
		v3.AddArg(v4)
		v1.AddArg2(v2, v3)
		v.AddArg3(v0, v1, mem)
		return true
	}
}
func rewriteValueRISCV64_OpAvg64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Avg64u <t> x y)
	// result: (ADD (ADD <t> (SRLI <t> [1] x) (SRLI <t> [1] y)) (ANDI <t> [1] (AND <t> x y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpRISCV64ADD)
		v0 := b.NewValue0(v.Pos, OpRISCV64ADD, t)
		v1 := b.NewValue0(v.Pos, OpRISCV64SRLI, t)
		v1.AuxInt = int64ToAuxInt(1)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpRISCV64SRLI, t)
		v2.AuxInt = int64ToAuxInt(1)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpRISCV64ANDI, t)
		v3.AuxInt = int64ToAuxInt(1)
		v4 := b.NewValue0(v.Pos, OpRISCV64AND, t)
		v4.AddArg2(x, y)
		v3.AddArg(v4)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueRISCV64_OpConst16(v *Value) bool {
	// match: (Const16 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := auxIntToInt16(v.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValueRISCV64_OpConst32(v *Value) bool {
	// match: (Const32 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := auxIntToInt32(v.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValueRISCV64_OpConst32F(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Const32F [val])
	// result: (FMVSX (MOVDconst [int64(math.Float32bits(val))]))
	for {
		val := auxIntToFloat32(v.AuxInt)
		v.reset(OpRISCV64FMVSX)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(int64(math.Float32bits(val)))
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpConst64(v *Value) bool {
	// match: (Const64 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := auxIntToInt64(v.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValueRISCV64_OpConst64F(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Const64F [val])
	// result: (FMVDX (MOVDconst [int64(math.Float64bits(val))]))
	for {
		val := auxIntToFloat64(v.AuxInt)
		v.reset(OpRISCV64FMVDX)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(int64(math.Float64bits(val)))
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpConst8(v *Value) bool {
	// match: (Const8 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := auxIntToInt8(v.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValueRISCV64_OpConstBool(v *Value) bool {
	// match: (ConstBool [val])
	// result: (MOVDconst [int64(b2i(val))])
	for {
		val := auxIntToBool(v.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(b2i(val)))
		return true
	}
}
func rewriteValueRISCV64_OpConstNil(v *Value) bool {
	// match: (ConstNil)
	// result: (MOVDconst [0])
	for {
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
}
func rewriteValueRISCV64_OpDiv16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 x y [false])
	// result: (DIVW (SignExt16to32 x) (SignExt16to32 y))
	for {
		if auxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.reset(OpRISCV64DIVW)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpDiv16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16u x y)
	// result: (DIVUW (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64DIVUW)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueRISCV64_OpDiv32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div32 x y [false])
	// result: (DIVW x y)
	for {
		if auxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.reset(OpRISCV64DIVW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpDiv64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div64 x y [false])
	// result: (DIV x y)
	for {
		if auxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.reset(OpRISCV64DIV)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpDiv8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// result: (DIVW (SignExt8to32 x) (SignExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64DIVW)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueRISCV64_OpDiv8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// result: (DIVUW (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64DIVUW)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueRISCV64_OpEq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq16 x y)
	// result: (SEQZ (SUB <x.Type> (ZeroExt16to64 x) (ZeroExt16to64 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpEq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq32 x y)
	// result: (SEQZ (SUB <x.Type> (ZeroExt32to64 x) (ZeroExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpEq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64 x y)
	// result: (SEQZ (SUB <x.Type> x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpEq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq8 x y)
	// result: (SEQZ (SUB <x.Type> (ZeroExt8to64 x) (ZeroExt8to64 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpEqB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqB x y)
	// result: (SEQZ (XOR <typ.Bool> x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64XOR, typ.Bool)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpEqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqPtr x y)
	// result: (SEQZ (SUB <typ.Uintptr> x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, typ.Uintptr)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpHmul32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul32 x y)
	// result: (SRAI [32] (MUL (SignExt32to64 x) (SignExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SRAI)
		v.AuxInt = int64ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, OpRISCV64MUL, typ.Int64)
		v1 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpHmul32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul32u x y)
	// result: (SRLI [32] (MUL (ZeroExt32to64 x) (ZeroExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SRLI)
		v.AuxInt = int64ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, OpRISCV64MUL, typ.Int64)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16 x y)
	// result: (Not (Less16 y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess16, typ.Bool)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16U x y)
	// result: (Not (Less16U y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess16U, typ.Bool)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32 x y)
	// result: (Not (Less32 y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess32, typ.Bool)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32U x y)
	// result: (Not (Less32U y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess32U, typ.Bool)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq64 x y)
	// result: (Not (Less64 y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess64, typ.Bool)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq64U x y)
	// result: (Not (Less64U y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess64U, typ.Bool)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8 x y)
	// result: (Not (Less8 y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess8, typ.Bool)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8U x y)
	// result: (Not (Less8U y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess8U, typ.Bool)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLess16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16 x y)
	// result: (SLT (SignExt16to64 x) (SignExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SLT)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueRISCV64_OpLess16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16U x y)
	// result: (SLTU (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SLTU)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueRISCV64_OpLess32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32 x y)
	// result: (SLT (SignExt32to64 x) (SignExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SLT)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueRISCV64_OpLess32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32U x y)
	// result: (SLTU (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SLTU)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueRISCV64_OpLess8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8 x y)
	// result: (SLT (SignExt8to64 x) (SignExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SLT)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueRISCV64_OpLess8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8U x y)
	// result: (SLTU (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SLTU)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueRISCV64_OpLoad(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Load <t> ptr mem)
	// cond: t.IsBoolean()
	// result: (MOVBUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.IsBoolean()) {
			break
		}
		v.reset(OpRISCV64MOVBUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ( is8BitInt(t) && isSigned(t))
	// result: (MOVBload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is8BitInt(t) && isSigned(t)) {
			break
		}
		v.reset(OpRISCV64MOVBload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ( is8BitInt(t) && !isSigned(t))
	// result: (MOVBUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is8BitInt(t) && !isSigned(t)) {
			break
		}
		v.reset(OpRISCV64MOVBUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is16BitInt(t) && isSigned(t))
	// result: (MOVHload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is16BitInt(t) && isSigned(t)) {
			break
		}
		v.reset(OpRISCV64MOVHload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is16BitInt(t) && !isSigned(t))
	// result: (MOVHUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is16BitInt(t) && !isSigned(t)) {
			break
		}
		v.reset(OpRISCV64MOVHUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is32BitInt(t) && isSigned(t))
	// result: (MOVWload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is32BitInt(t) && isSigned(t)) {
			break
		}
		v.reset(OpRISCV64MOVWload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is32BitInt(t) && !isSigned(t))
	// result: (MOVWUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is32BitInt(t) && !isSigned(t)) {
			break
		}
		v.reset(OpRISCV64MOVWUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (MOVDload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpRISCV64MOVDload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is32BitFloat(t)
	// result: (FMOVWload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is32BitFloat(t)) {
			break
		}
		v.reset(OpRISCV64FMOVWload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is64BitFloat(t)
	// result: (FMOVDload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is64BitFloat(t)) {
			break
		}
		v.reset(OpRISCV64FMOVDload)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpLocalAddr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (LocalAddr {sym} base _)
	// result: (MOVaddr {sym} base)
	for {
		sym := auxToSym(v.Aux)
		base := v_0
		v.reset(OpRISCV64MOVaddr)
		v.Aux = symToAux(sym)
		v.AddArg(base)
		return true
	}
}
func rewriteValueRISCV64_OpLsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg16, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh16x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpLsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg16, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh16x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpLsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh16x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg16 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg16, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh16x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpLsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg16, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh16x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpLsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg32, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh32x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpLsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg32, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh32x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpLsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh32x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg32 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg32, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh32x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpLsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg32, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh32x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpLsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg64, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh64x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpLsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg64, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh64x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpLsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh64x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg64 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg64, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh64x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpLsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg64, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh64x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpLsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg8, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh8x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpLsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg8, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh8x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpLsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh8x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg8 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg8, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh8x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpLsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg8, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh8x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpMod16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16 x y [false])
	// result: (REMW (SignExt16to32 x) (SignExt16to32 y))
	for {
		if auxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.reset(OpRISCV64REMW)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpMod16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16u x y)
	// result: (REMUW (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64REMUW)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueRISCV64_OpMod32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mod32 x y [false])
	// result: (REMW x y)
	for {
		if auxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.reset(OpRISCV64REMW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpMod64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mod64 x y [false])
	// result: (REM x y)
	for {
		if auxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.reset(OpRISCV64REM)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpMod8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8 x y)
	// result: (REMW (SignExt8to32 x) (SignExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64REMW)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueRISCV64_OpMod8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8u x y)
	// result: (REMUW (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64REMUW)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueRISCV64_OpMove(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Move [0] _ _ mem)
	// result: mem
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.copyOf(mem)
		return true
	}
	// match: (Move [1] dst src mem)
	// result: (MOVBstore dst (MOVBload src mem) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpRISCV64MOVBstore)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVBload, typ.Int8)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [2] {t} dst src mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore dst (MOVHload src mem) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 2 {
			break
		}
		t := auxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.reset(OpRISCV64MOVHstore)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVHload, typ.Int16)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [2] dst src mem)
	// result: (MOVBstore [1] dst (MOVBload [1] src mem) (MOVBstore dst (MOVBload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 2 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpRISCV64MOVBstore)
		v.AuxInt = int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVBload, typ.Int8)
		v0.AuxInt = int32ToAuxInt(1)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVBstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVBload, typ.Int8)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [4] {t} dst src mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore dst (MOVWload src mem) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 4 {
			break
		}
		t := auxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.reset(OpRISCV64MOVWstore)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVWload, typ.Int32)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [4] {t} dst src mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [2] dst (MOVHload [2] src mem) (MOVHstore dst (MOVHload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 4 {
			break
		}
		t := auxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.reset(OpRISCV64MOVHstore)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVHload, typ.Int16)
		v0.AuxInt = int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVHstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVHload, typ.Int16)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [4] dst src mem)
	// result: (MOVBstore [3] dst (MOVBload [3] src mem) (MOVBstore [2] dst (MOVBload [2] src mem) (MOVBstore [1] dst (MOVBload [1] src mem) (MOVBstore dst (MOVBload src mem) mem))))
	for {
		if auxIntToInt64(v.AuxInt) != 4 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpRISCV64MOVBstore)
		v.AuxInt = int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVBload, typ.Int8)
		v0.AuxInt = int32ToAuxInt(3)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVBstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(2)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVBload, typ.Int8)
		v2.AuxInt = int32ToAuxInt(2)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVBstore, types.TypeMem)
		v3.AuxInt = int32ToAuxInt(1)
		v4 := b.NewValue0(v.Pos, OpRISCV64MOVBload, typ.Int8)
		v4.AuxInt = int32ToAuxInt(1)
		v4.AddArg2(src, mem)
		v5 := b.NewValue0(v.Pos, OpRISCV64MOVBstore, types.TypeMem)
		v6 := b.NewValue0(v.Pos, OpRISCV64MOVBload, typ.Int8)
		v6.AddArg2(src, mem)
		v5.AddArg3(dst, v6, mem)
		v3.AddArg3(dst, v4, v5)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [8] {t} dst src mem)
	// cond: t.Alignment()%8 == 0
	// result: (MOVDstore dst (MOVDload src mem) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		t := auxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%8 == 0) {
			break
		}
		v.reset(OpRISCV64MOVDstore)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDload, typ.Int64)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [8] {t} dst src mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore [4] dst (MOVWload [4] src mem) (MOVWstore dst (MOVWload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		t := auxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.reset(OpRISCV64MOVWstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVWload, typ.Int32)
		v0.AuxInt = int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVWload, typ.Int32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [8] {t} dst src mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [6] dst (MOVHload [6] src mem) (MOVHstore [4] dst (MOVHload [4] src mem) (MOVHstore [2] dst (MOVHload [2] src mem) (MOVHstore dst (MOVHload src mem) mem))))
	for {
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		t := auxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.reset(OpRISCV64MOVHstore)
		v.AuxInt = int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVHload, typ.Int16)
		v0.AuxInt = int32ToAuxInt(6)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVHstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(4)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVHload, typ.Int16)
		v2.AuxInt = int32ToAuxInt(4)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVHstore, types.TypeMem)
		v3.AuxInt = int32ToAuxInt(2)
		v4 := b.NewValue0(v.Pos, OpRISCV64MOVHload, typ.Int16)
		v4.AuxInt = int32ToAuxInt(2)
		v4.AddArg2(src, mem)
		v5 := b.NewValue0(v.Pos, OpRISCV64MOVHstore, types.TypeMem)
		v6 := b.NewValue0(v.Pos, OpRISCV64MOVHload, typ.Int16)
		v6.AddArg2(src, mem)
		v5.AddArg3(dst, v6, mem)
		v3.AddArg3(dst, v4, v5)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [3] dst src mem)
	// result: (MOVBstore [2] dst (MOVBload [2] src mem) (MOVBstore [1] dst (MOVBload [1] src mem) (MOVBstore dst (MOVBload src mem) mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 3 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpRISCV64MOVBstore)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVBload, typ.Int8)
		v0.AuxInt = int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVBstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(1)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVBload, typ.Int8)
		v2.AuxInt = int32ToAuxInt(1)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVBstore, types.TypeMem)
		v4 := b.NewValue0(v.Pos, OpRISCV64MOVBload, typ.Int8)
		v4.AddArg2(src, mem)
		v3.AddArg3(dst, v4, mem)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [6] {t} dst src mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [4] dst (MOVHload [4] src mem) (MOVHstore [2] dst (MOVHload [2] src mem) (MOVHstore dst (MOVHload src mem) mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 6 {
			break
		}
		t := auxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.reset(OpRISCV64MOVHstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVHload, typ.Int16)
		v0.AuxInt = int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVHstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(2)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVHload, typ.Int16)
		v2.AuxInt = int32ToAuxInt(2)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVHstore, types.TypeMem)
		v4 := b.NewValue0(v.Pos, OpRISCV64MOVHload, typ.Int16)
		v4.AddArg2(src, mem)
		v3.AddArg3(dst, v4, mem)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [12] {t} dst src mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore [8] dst (MOVWload [8] src mem) (MOVWstore [4] dst (MOVWload [4] src mem) (MOVWstore dst (MOVWload src mem) mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 12 {
			break
		}
		t := auxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.reset(OpRISCV64MOVWstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVWload, typ.Int32)
		v0.AuxInt = int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVWstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(4)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVWload, typ.Int32)
		v2.AuxInt = int32ToAuxInt(4)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVWstore, types.TypeMem)
		v4 := b.NewValue0(v.Pos, OpRISCV64MOVWload, typ.Int32)
		v4.AddArg2(src, mem)
		v3.AddArg3(dst, v4, mem)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [16] {t} dst src mem)
	// cond: t.Alignment()%8 == 0
	// result: (MOVDstore [8] dst (MOVDload [8] src mem) (MOVDstore dst (MOVDload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 16 {
			break
		}
		t := auxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%8 == 0) {
			break
		}
		v.reset(OpRISCV64MOVDstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDload, typ.Int64)
		v0.AuxInt = int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVDstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVDload, typ.Int64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [24] {t} dst src mem)
	// cond: t.Alignment()%8 == 0
	// result: (MOVDstore [16] dst (MOVDload [16] src mem) (MOVDstore [8] dst (MOVDload [8] src mem) (MOVDstore dst (MOVDload src mem) mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 24 {
			break
		}
		t := auxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%8 == 0) {
			break
		}
		v.reset(OpRISCV64MOVDstore)
		v.AuxInt = int32ToAuxInt(16)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDload, typ.Int64)
		v0.AuxInt = int32ToAuxInt(16)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVDstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(8)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVDload, typ.Int64)
		v2.AuxInt = int32ToAuxInt(8)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVDstore, types.TypeMem)
		v4 := b.NewValue0(v.Pos, OpRISCV64MOVDload, typ.Int64)
		v4.AddArg2(src, mem)
		v3.AddArg3(dst, v4, mem)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [32] {t} dst src mem)
	// cond: t.Alignment()%8 == 0
	// result: (MOVDstore [24] dst (MOVDload [24] src mem) (MOVDstore [16] dst (MOVDload [16] src mem) (MOVDstore [8] dst (MOVDload [8] src mem) (MOVDstore dst (MOVDload src mem) mem))))
	for {
		if auxIntToInt64(v.AuxInt) != 32 {
			break
		}
		t := auxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%8 == 0) {
			break
		}
		v.reset(OpRISCV64MOVDstore)
		v.AuxInt = int32ToAuxInt(24)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDload, typ.Int64)
		v0.AuxInt = int32ToAuxInt(24)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVDstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(16)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVDload, typ.Int64)
		v2.AuxInt = int32ToAuxInt(16)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVDstore, types.TypeMem)
		v3.AuxInt = int32ToAuxInt(8)
		v4 := b.NewValue0(v.Pos, OpRISCV64MOVDload, typ.Int64)
		v4.AuxInt = int32ToAuxInt(8)
		v4.AddArg2(src, mem)
		v5 := b.NewValue0(v.Pos, OpRISCV64MOVDstore, types.TypeMem)
		v6 := b.NewValue0(v.Pos, OpRISCV64MOVDload, typ.Int64)
		v6.AddArg2(src, mem)
		v5.AddArg3(dst, v6, mem)
		v3.AddArg3(dst, v4, v5)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] {t} dst src mem)
	// cond: s%8 == 0 && s <= 8*128 && t.Alignment()%8 == 0 && !config.noDuffDevice && logLargeCopy(v, s)
	// result: (DUFFCOPY [16 * (128 - s/8)] dst src mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		t := auxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s%8 == 0 && s <= 8*128 && t.Alignment()%8 == 0 && !config.noDuffDevice && logLargeCopy(v, s)) {
			break
		}
		v.reset(OpRISCV64DUFFCOPY)
		v.AuxInt = int64ToAuxInt(16 * (128 - s/8))
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (Move [s] {t} dst src mem)
	// cond: (s <= 16 || logLargeCopy(v, s))
	// result: (LoweredMove [t.Alignment()] dst src (ADDI <src.Type> [s-moveSize(t.Alignment(), config)] src) mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		t := auxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s <= 16 || logLargeCopy(v, s)) {
			break
		}
		v.reset(OpRISCV64LoweredMove)
		v.AuxInt = int64ToAuxInt(t.Alignment())
		v0 := b.NewValue0(v.Pos, OpRISCV64ADDI, src.Type)
		v0.AuxInt = int64ToAuxInt(s - moveSize(t.Alignment(), config))
		v0.AddArg(src)
		v.AddArg4(dst, src, v0, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpMul16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mul16 x y)
	// result: (MULW (SignExt16to32 x) (SignExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64MULW)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueRISCV64_OpMul8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mul8 x y)
	// result: (MULW (SignExt8to32 x) (SignExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64MULW)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueRISCV64_OpNeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq16 x y)
	// result: (SNEZ (SUB <x.Type> (ZeroExt16to64 x) (ZeroExt16to64 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpNeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq32 x y)
	// result: (SNEZ (SUB <x.Type> (ZeroExt32to64 x) (ZeroExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpNeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64 x y)
	// result: (SNEZ (SUB <x.Type> x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpNeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq8 x y)
	// result: (SNEZ (SUB <x.Type> (ZeroExt8to64 x) (ZeroExt8to64 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpNeqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NeqPtr x y)
	// result: (SNEZ (SUB <typ.Uintptr> x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, typ.Uintptr)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpOffPtr(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (OffPtr [off] ptr:(SP))
	// cond: is32Bit(off)
	// result: (MOVaddr [int32(off)] ptr)
	for {
		off := auxIntToInt64(v.AuxInt)
		ptr := v_0
		if ptr.Op != OpSP || !(is32Bit(off)) {
			break
		}
		v.reset(OpRISCV64MOVaddr)
		v.AuxInt = int32ToAuxInt(int32(off))
		v.AddArg(ptr)
		return true
	}
	// match: (OffPtr [off] ptr)
	// cond: is32Bit(off)
	// result: (ADDI [off] ptr)
	for {
		off := auxIntToInt64(v.AuxInt)
		ptr := v_0
		if !(is32Bit(off)) {
			break
		}
		v.reset(OpRISCV64ADDI)
		v.AuxInt = int64ToAuxInt(off)
		v.AddArg(ptr)
		return true
	}
	// match: (OffPtr [off] ptr)
	// result: (ADD (MOVDconst [off]) ptr)
	for {
		off := auxIntToInt64(v.AuxInt)
		ptr := v_0
		v.reset(OpRISCV64ADD)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(off)
		v.AddArg2(v0, ptr)
		return true
	}
}
func rewriteValueRISCV64_OpPanicBounds(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (PanicBounds [kind] x y mem)
	// cond: boundsABI(kind) == 0
	// result: (LoweredPanicBoundsA [kind] x y mem)
	for {
		kind := auxIntToInt64(v.AuxInt)
		x := v_0
		y := v_1
		mem := v_2
		if !(boundsABI(kind) == 0) {
			break
		}
		v.reset(OpRISCV64LoweredPanicBoundsA)
		v.AuxInt = int64ToAuxInt(kind)
		v.AddArg3(x, y, mem)
		return true
	}
	// match: (PanicBounds [kind] x y mem)
	// cond: boundsABI(kind) == 1
	// result: (LoweredPanicBoundsB [kind] x y mem)
	for {
		kind := auxIntToInt64(v.AuxInt)
		x := v_0
		y := v_1
		mem := v_2
		if !(boundsABI(kind) == 1) {
			break
		}
		v.reset(OpRISCV64LoweredPanicBoundsB)
		v.AuxInt = int64ToAuxInt(kind)
		v.AddArg3(x, y, mem)
		return true
	}
	// match: (PanicBounds [kind] x y mem)
	// cond: boundsABI(kind) == 2
	// result: (LoweredPanicBoundsC [kind] x y mem)
	for {
		kind := auxIntToInt64(v.AuxInt)
		x := v_0
		y := v_1
		mem := v_2
		if !(boundsABI(kind) == 2) {
			break
		}
		v.reset(OpRISCV64LoweredPanicBoundsC)
		v.AuxInt = int64ToAuxInt(kind)
		v.AddArg3(x, y, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64ADD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADD (MOVDconst [val]) x)
	// cond: is32Bit(val)
	// result: (ADDI [val] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64MOVDconst {
				continue
			}
			val := auxIntToInt64(v_0.AuxInt)
			x := v_1
			if !(is32Bit(val)) {
				continue
			}
			v.reset(OpRISCV64ADDI)
			v.AuxInt = int64ToAuxInt(val)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64ADDI(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ADDI [c] (MOVaddr [d] {s} x))
	// cond: is32Bit(c+int64(d))
	// result: (MOVaddr [int32(c)+d] {s} x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		s := auxToSym(v_0.Aux)
		x := v_0.Args[0]
		if !(is32Bit(c + int64(d))) {
			break
		}
		v.reset(OpRISCV64MOVaddr)
		v.AuxInt = int32ToAuxInt(int32(c) + d)
		v.Aux = symToAux(s)
		v.AddArg(x)
		return true
	}
	// match: (ADDI [0] x)
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (ADDI [x] (MOVDconst [y]))
	// cond: is32Bit(x + y)
	// result: (MOVDconst [x + y])
	for {
		x := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		y := auxIntToInt64(v_0.AuxInt)
		if !(is32Bit(x + y)) {
			break
		}
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(x + y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64AND(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AND (MOVDconst [val]) x)
	// cond: is32Bit(val)
	// result: (ANDI [val] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64MOVDconst {
				continue
			}
			val := auxIntToInt64(v_0.AuxInt)
			x := v_1
			if !(is32Bit(val)) {
				continue
			}
			v.reset(OpRISCV64ANDI)
			v.AuxInt = int64ToAuxInt(val)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64ANDI(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ANDI [0] x)
	// result: (MOVDconst [0])
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (ANDI [-1] x)
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != -1 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (ANDI [x] (MOVDconst [y]))
	// result: (MOVDconst [x & y])
	for {
		x := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		y := auxIntToInt64(v_0.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(x & y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64FMADDD(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMADDD neg:(FNEGD x) y z)
	// cond: neg.Uses == 1
	// result: (FNMADDD x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			neg := v_0
			if neg.Op != OpRISCV64FNEGD {
				continue
			}
			x := neg.Args[0]
			y := v_1
			z := v_2
			if !(neg.Uses == 1) {
				continue
			}
			v.reset(OpRISCV64FNMADDD)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (FMADDD x y neg:(FNEGD z))
	// cond: neg.Uses == 1
	// result: (FMSUBD x y z)
	for {
		x := v_0
		y := v_1
		neg := v_2
		if neg.Op != OpRISCV64FNEGD {
			break
		}
		z := neg.Args[0]
		if !(neg.Uses == 1) {
			break
		}
		v.reset(OpRISCV64FMSUBD)
		v.AddArg3(x, y, z)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64FMSUBD(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMSUBD neg:(FNEGD x) y z)
	// cond: neg.Uses == 1
	// result: (FNMSUBD x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			neg := v_0
			if neg.Op != OpRISCV64FNEGD {
				continue
			}
			x := neg.Args[0]
			y := v_1
			z := v_2
			if !(neg.Uses == 1) {
				continue
			}
			v.reset(OpRISCV64FNMSUBD)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (FMSUBD x y neg:(FNEGD z))
	// cond: neg.Uses == 1
	// result: (FMADDD x y z)
	for {
		x := v_0
		y := v_1
		neg := v_2
		if neg.Op != OpRISCV64FNEGD {
			break
		}
		z := neg.Args[0]
		if !(neg.Uses == 1) {
			break
		}
		v.reset(OpRISCV64FMADDD)
		v.AddArg3(x, y, z)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64FNMADDD(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FNMADDD neg:(FNEGD x) y z)
	// cond: neg.Uses == 1
	// result: (FMADDD x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			neg := v_0
			if neg.Op != OpRISCV64FNEGD {
				continue
			}
			x := neg.Args[0]
			y := v_1
			z := v_2
			if !(neg.Uses == 1) {
				continue
			}
			v.reset(OpRISCV64FMADDD)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (FNMADDD x y neg:(FNEGD z))
	// cond: neg.Uses == 1
	// result: (FNMSUBD x y z)
	for {
		x := v_0
		y := v_1
		neg := v_2
		if neg.Op != OpRISCV64FNEGD {
			break
		}
		z := neg.Args[0]
		if !(neg.Uses == 1) {
			break
		}
		v.reset(OpRISCV64FNMSUBD)
		v.AddArg3(x, y, z)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64FNMSUBD(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FNMSUBD neg:(FNEGD x) y z)
	// cond: neg.Uses == 1
	// result: (FMSUBD x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			neg := v_0
			if neg.Op != OpRISCV64FNEGD {
				continue
			}
			x := neg.Args[0]
			y := v_1
			z := v_2
			if !(neg.Uses == 1) {
				continue
			}
			v.reset(OpRISCV64FMSUBD)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (FNMSUBD x y neg:(FNEGD z))
	// cond: neg.Uses == 1
	// result: (FNMADDD x y z)
	for {
		x := v_0
		y := v_1
		neg := v_2
		if neg.Op != OpRISCV64FNEGD {
			break
		}
		z := neg.Args[0]
		if !(neg.Uses == 1) {
			break
		}
		v.reset(OpRISCV64FNMADDD)
		v.AddArg3(x, y, z)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVBUload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBUload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVBUload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVBUload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVBUload [off1] {sym} (ADDI [off2] base) mem)
	// cond: is32Bit(int64(off1)+off2)
	// result: (MOVBUload [off1+int32(off2)] {sym} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + off2)) {
			break
		}
		v.reset(OpRISCV64MOVBUload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVBUreg(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVBUreg x:(FLES _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpRISCV64FLES {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(FLTS _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpRISCV64FLTS {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(FEQS _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpRISCV64FEQS {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(FNES _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpRISCV64FNES {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(FLED _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpRISCV64FLED {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(FLTD _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpRISCV64FLTD {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(FEQD _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpRISCV64FEQD {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(FNED _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpRISCV64FNED {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(SEQZ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpRISCV64SEQZ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(SNEZ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpRISCV64SNEZ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(SLT _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpRISCV64SLT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(SLTU _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpRISCV64SLTU {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(ANDI [c] y))
	// cond: c >= 0 && int64(uint8(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != OpRISCV64ANDI {
			break
		}
		c := auxIntToInt64(x.AuxInt)
		if !(c >= 0 && int64(uint8(c)) == c) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg (MOVDconst [c]))
	// result: (MOVDconst [int64(uint8(c))])
	for {
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(uint8(c)))
		return true
	}
	// match: (MOVBUreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVBUload {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVBUreg {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg <t> x:(MOVBload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBUload <t> [off] {sym} ptr mem)
	for {
		t := v.Type
		x := v_0
		if x.Op != OpRISCV64MOVBload {
			break
		}
		off := auxIntToInt32(x.AuxInt)
		sym := auxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, OpRISCV64MOVBUload, t)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVBload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVBload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVBload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVBload [off1] {sym} (ADDI [off2] base) mem)
	// cond: is32Bit(int64(off1)+off2)
	// result: (MOVBload [off1+int32(off2)] {sym} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + off2)) {
			break
		}
		v.reset(OpRISCV64MOVBload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVBreg(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVBreg x:(ANDI [c] y))
	// cond: c >= 0 && int64(int8(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != OpRISCV64ANDI {
			break
		}
		c := auxIntToInt64(x.AuxInt)
		if !(c >= 0 && int64(int8(c)) == c) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBreg (MOVDconst [c]))
	// result: (MOVDconst [int64(int8(c))])
	for {
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(int8(c)))
		return true
	}
	// match: (MOVBreg x:(MOVBload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVBload {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVBreg {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg <t> x:(MOVBUload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBload <t> [off] {sym} ptr mem)
	for {
		t := v.Type
		x := v_0
		if x.Op != OpRISCV64MOVBUload {
			break
		}
		off := auxIntToInt32(x.AuxInt)
		sym := auxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, OpRISCV64MOVBload, t)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVBstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBstore [off1] {sym1} (MOVaddr [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVBstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVBstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVBstore [off1] {sym} (ADDI [off2] base) val mem)
	// cond: is32Bit(int64(off1)+off2)
	// result: (MOVBstore [off1+int32(off2)] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + off2)) {
			break
		}
		v.reset(OpRISCV64MOVBstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVDconst [0]) mem)
	// result: (MOVBstorezero [off] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpRISCV64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.reset(OpRISCV64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpRISCV64MOVBreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpRISCV64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVHreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpRISCV64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpRISCV64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVWreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpRISCV64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpRISCV64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBUreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpRISCV64MOVBUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpRISCV64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVHUreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpRISCV64MOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpRISCV64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVWUreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpRISCV64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpRISCV64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVBstorezero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBstorezero [off1] {sym1} (MOVaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2))
	// result: (MOVBstorezero [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2))) {
			break
		}
		v.reset(OpRISCV64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstorezero [off1] {sym} (ADDI [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2)
	// result: (MOVBstorezero [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + off2)) {
			break
		}
		v.reset(OpRISCV64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVDload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVDload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVDload [off1] {sym} (ADDI [off2] base) mem)
	// cond: is32Bit(int64(off1)+off2)
	// result: (MOVDload [off1+int32(off2)] {sym} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + off2)) {
			break
		}
		v.reset(OpRISCV64MOVDload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVDnop(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVDnop (MOVDconst [c]))
	// result: (MOVDconst [c])
	for {
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVDreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVDreg x)
	// cond: x.Uses == 1
	// result: (MOVDnop x)
	for {
		x := v_0
		if !(x.Uses == 1) {
			break
		}
		v.reset(OpRISCV64MOVDnop)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVDstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDstore [off1] {sym1} (MOVaddr [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVDstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVDstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym} (ADDI [off2] base) val mem)
	// cond: is32Bit(int64(off1)+off2)
	// result: (MOVDstore [off1+int32(off2)] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + off2)) {
			break
		}
		v.reset(OpRISCV64MOVDstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVDstore [off] {sym} ptr (MOVDconst [0]) mem)
	// result: (MOVDstorezero [off] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpRISCV64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.reset(OpRISCV64MOVDstorezero)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVDstorezero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDstorezero [off1] {sym1} (MOVaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2))
	// result: (MOVDstorezero [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2))) {
			break
		}
		v.reset(OpRISCV64MOVDstorezero)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDstorezero [off1] {sym} (ADDI [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2)
	// result: (MOVDstorezero [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + off2)) {
			break
		}
		v.reset(OpRISCV64MOVDstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVHUload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHUload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVHUload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVHUload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVHUload [off1] {sym} (ADDI [off2] base) mem)
	// cond: is32Bit(int64(off1)+off2)
	// result: (MOVHUload [off1+int32(off2)] {sym} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + off2)) {
			break
		}
		v.reset(OpRISCV64MOVHUload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVHUreg(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVHUreg x:(ANDI [c] y))
	// cond: c >= 0 && int64(uint16(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != OpRISCV64ANDI {
			break
		}
		c := auxIntToInt64(x.AuxInt)
		if !(c >= 0 && int64(uint16(c)) == c) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVHUreg (MOVDconst [c]))
	// result: (MOVDconst [int64(uint16(c))])
	for {
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(uint16(c)))
		return true
	}
	// match: (MOVHUreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVBUload {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVHUload {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVBUreg {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVHUreg {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg <t> x:(MOVHload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVHUload <t> [off] {sym} ptr mem)
	for {
		t := v.Type
		x := v_0
		if x.Op != OpRISCV64MOVHload {
			break
		}
		off := auxIntToInt32(x.AuxInt)
		sym := auxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, OpRISCV64MOVHUload, t)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVHload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVHload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVHload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVHload [off1] {sym} (ADDI [off2] base) mem)
	// cond: is32Bit(int64(off1)+off2)
	// result: (MOVHload [off1+int32(off2)] {sym} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + off2)) {
			break
		}
		v.reset(OpRISCV64MOVHload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVHreg(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVHreg x:(ANDI [c] y))
	// cond: c >= 0 && int64(int16(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != OpRISCV64ANDI {
			break
		}
		c := auxIntToInt64(x.AuxInt)
		if !(c >= 0 && int64(int16(c)) == c) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVHreg (MOVDconst [c]))
	// result: (MOVDconst [int64(int16(c))])
	for {
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(int16(c)))
		return true
	}
	// match: (MOVHreg x:(MOVBload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVBload {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVBUload {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVHload {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVBreg {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVBUreg {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVHreg {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg <t> x:(MOVHUload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVHload <t> [off] {sym} ptr mem)
	for {
		t := v.Type
		x := v_0
		if x.Op != OpRISCV64MOVHUload {
			break
		}
		off := auxIntToInt32(x.AuxInt)
		sym := auxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, OpRISCV64MOVHload, t)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVHstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstore [off1] {sym1} (MOVaddr [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVHstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVHstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVHstore [off1] {sym} (ADDI [off2] base) val mem)
	// cond: is32Bit(int64(off1)+off2)
	// result: (MOVHstore [off1+int32(off2)] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + off2)) {
			break
		}
		v.reset(OpRISCV64MOVHstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVDconst [0]) mem)
	// result: (MOVHstorezero [off] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpRISCV64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.reset(OpRISCV64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVHreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpRISCV64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpRISCV64MOVHstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVWreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpRISCV64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpRISCV64MOVHstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVHUreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpRISCV64MOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpRISCV64MOVHstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVWUreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpRISCV64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpRISCV64MOVHstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVHstorezero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstorezero [off1] {sym1} (MOVaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2))
	// result: (MOVHstorezero [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2))) {
			break
		}
		v.reset(OpRISCV64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHstorezero [off1] {sym} (ADDI [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2)
	// result: (MOVHstorezero [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + off2)) {
			break
		}
		v.reset(OpRISCV64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVWUload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWUload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVWUload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVWUload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVWUload [off1] {sym} (ADDI [off2] base) mem)
	// cond: is32Bit(int64(off1)+off2)
	// result: (MOVWUload [off1+int32(off2)] {sym} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + off2)) {
			break
		}
		v.reset(OpRISCV64MOVWUload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVWUreg(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVWUreg x:(ANDI [c] y))
	// cond: c >= 0 && int64(uint32(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != OpRISCV64ANDI {
			break
		}
		c := auxIntToInt64(x.AuxInt)
		if !(c >= 0 && int64(uint32(c)) == c) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVWUreg (MOVDconst [c]))
	// result: (MOVDconst [int64(uint32(c))])
	for {
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(uint32(c)))
		return true
	}
	// match: (MOVWUreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVBUload {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVHUload {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVWUload {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVBUreg {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVHUreg {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVWUreg {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg <t> x:(MOVWload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVWUload <t> [off] {sym} ptr mem)
	for {
		t := v.Type
		x := v_0
		if x.Op != OpRISCV64MOVWload {
			break
		}
		off := auxIntToInt32(x.AuxInt)
		sym := auxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, OpRISCV64MOVWUload, t)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVWload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVWload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVWload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVWload [off1] {sym} (ADDI [off2] base) mem)
	// cond: is32Bit(int64(off1)+off2)
	// result: (MOVWload [off1+int32(off2)] {sym} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + off2)) {
			break
		}
		v.reset(OpRISCV64MOVWload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVWreg(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVWreg x:(ANDI [c] y))
	// cond: c >= 0 && int64(int32(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != OpRISCV64ANDI {
			break
		}
		c := auxIntToInt64(x.AuxInt)
		if !(c >= 0 && int64(int32(c)) == c) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVWreg (MOVDconst [c]))
	// result: (MOVDconst [int64(int32(c))])
	for {
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(int32(c)))
		return true
	}
	// match: (MOVWreg x:(MOVBload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVBload {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVBUload {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVHload {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVHUload {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVWload {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVBreg {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVBUreg {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVHreg {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpRISCV64MOVWreg {
			break
		}
		v.reset(OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg <t> x:(MOVWUload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVWload <t> [off] {sym} ptr mem)
	for {
		t := v.Type
		x := v_0
		if x.Op != OpRISCV64MOVWUload {
			break
		}
		off := auxIntToInt32(x.AuxInt)
		sym := auxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, OpRISCV64MOVWload, t)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVWstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstore [off1] {sym1} (MOVaddr [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVWstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVWstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym} (ADDI [off2] base) val mem)
	// cond: is32Bit(int64(off1)+off2)
	// result: (MOVWstore [off1+int32(off2)] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + off2)) {
			break
		}
		v.reset(OpRISCV64MOVWstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVDconst [0]) mem)
	// result: (MOVWstorezero [off] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpRISCV64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.reset(OpRISCV64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWreg x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpRISCV64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpRISCV64MOVWstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWUreg x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpRISCV64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpRISCV64MOVWstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVWstorezero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstorezero [off1] {sym1} (MOVaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2))
	// result: (MOVWstorezero [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2))) {
			break
		}
		v.reset(OpRISCV64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstorezero [off1] {sym} (ADDI [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2)
	// result: (MOVWstorezero [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + off2)) {
			break
		}
		v.reset(OpRISCV64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64NEG(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NEG (MOVDconst [x]))
	// result: (MOVDconst [-x])
	for {
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(-x)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64NEGW(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NEGW (MOVDconst [x]))
	// result: (MOVDconst [int64(int32(-x))])
	for {
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(int32(-x)))
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64OR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (OR (MOVDconst [val]) x)
	// cond: is32Bit(val)
	// result: (ORI [val] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64MOVDconst {
				continue
			}
			val := auxIntToInt64(v_0.AuxInt)
			x := v_1
			if !(is32Bit(val)) {
				continue
			}
			v.reset(OpRISCV64ORI)
			v.AuxInt = int64ToAuxInt(val)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64ORI(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ORI [0] x)
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (ORI [-1] x)
	// result: (MOVDconst [-1])
	for {
		if auxIntToInt64(v.AuxInt) != -1 {
			break
		}
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	// match: (ORI [x] (MOVDconst [y]))
	// result: (MOVDconst [x | y])
	for {
		x := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		y := auxIntToInt64(v_0.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(x | y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64SEQZ(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SEQZ (NEG x))
	// result: (SEQZ x)
	for {
		if v_0.Op != OpRISCV64NEG {
			break
		}
		x := v_0.Args[0]
		v.reset(OpRISCV64SEQZ)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64SLL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLL x (MOVDconst [val]))
	// result: (SLLI [int64(val&63)] x)
	for {
		x := v_0
		if v_1.Op != OpRISCV64MOVDconst {
			break
		}
		val := auxIntToInt64(v_1.AuxInt)
		v.reset(OpRISCV64SLLI)
		v.AuxInt = int64ToAuxInt(int64(val & 63))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64SLLI(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SLLI [x] (MOVDconst [y]))
	// cond: is32Bit(y << uint32(x))
	// result: (MOVDconst [y << uint32(x)])
	for {
		x := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		y := auxIntToInt64(v_0.AuxInt)
		if !(is32Bit(y << uint32(x))) {
			break
		}
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(y << uint32(x))
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64SLT(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLT x x)
	// result: (MOVDconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64SLTI(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SLTI [x] (MOVDconst [y]))
	// result: (MOVDconst [b2i(int64(y) < int64(x))])
	for {
		x := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		y := auxIntToInt64(v_0.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(b2i(int64(y) < int64(x)))
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64SLTIU(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SLTIU [x] (MOVDconst [y]))
	// result: (MOVDconst [b2i(uint64(y) < uint64(x))])
	for {
		x := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		y := auxIntToInt64(v_0.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(b2i(uint64(y) < uint64(x)))
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64SLTU(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLTU x x)
	// result: (MOVDconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64SNEZ(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SNEZ (NEG x))
	// result: (SNEZ x)
	for {
		if v_0.Op != OpRISCV64NEG {
			break
		}
		x := v_0.Args[0]
		v.reset(OpRISCV64SNEZ)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64SRA(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRA x (MOVDconst [val]))
	// result: (SRAI [int64(val&63)] x)
	for {
		x := v_0
		if v_1.Op != OpRISCV64MOVDconst {
			break
		}
		val := auxIntToInt64(v_1.AuxInt)
		v.reset(OpRISCV64SRAI)
		v.AuxInt = int64ToAuxInt(int64(val & 63))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64SRAI(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SRAI [x] (MOVDconst [y]))
	// result: (MOVDconst [int64(y) >> uint32(x)])
	for {
		x := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		y := auxIntToInt64(v_0.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(y) >> uint32(x))
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64SRL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRL x (MOVDconst [val]))
	// result: (SRLI [int64(val&63)] x)
	for {
		x := v_0
		if v_1.Op != OpRISCV64MOVDconst {
			break
		}
		val := auxIntToInt64(v_1.AuxInt)
		v.reset(OpRISCV64SRLI)
		v.AuxInt = int64ToAuxInt(int64(val & 63))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64SRLI(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SRLI [x] (MOVDconst [y]))
	// result: (MOVDconst [int64(uint64(y) >> uint32(x))])
	for {
		x := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		y := auxIntToInt64(v_0.AuxInt)
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(y) >> uint32(x)))
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64SUB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUB x (MOVDconst [val]))
	// cond: is32Bit(-val)
	// result: (ADDI [-val] x)
	for {
		x := v_0
		if v_1.Op != OpRISCV64MOVDconst {
			break
		}
		val := auxIntToInt64(v_1.AuxInt)
		if !(is32Bit(-val)) {
			break
		}
		v.reset(OpRISCV64ADDI)
		v.AuxInt = int64ToAuxInt(-val)
		v.AddArg(x)
		return true
	}
	// match: (SUB <t> (MOVDconst [val]) y)
	// cond: is32Bit(-val)
	// result: (NEG (ADDI <t> [-val] y))
	for {
		t := v.Type
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		val := auxIntToInt64(v_0.AuxInt)
		y := v_1
		if !(is32Bit(-val)) {
			break
		}
		v.reset(OpRISCV64NEG)
		v0 := b.NewValue0(v.Pos, OpRISCV64ADDI, t)
		v0.AuxInt = int64ToAuxInt(-val)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SUB x (MOVDconst [0]))
	// result: x
	for {
		x := v_0
		if v_1.Op != OpRISCV64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (SUB (MOVDconst [0]) x)
	// result: (NEG x)
	for {
		if v_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_1
		v.reset(OpRISCV64NEG)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64SUBW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBW x (MOVDconst [0]))
	// result: (ADDIW [0] x)
	for {
		x := v_0
		if v_1.Op != OpRISCV64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.reset(OpRISCV64ADDIW)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg(x)
		return true
	}
	// match: (SUBW (MOVDconst [0]) x)
	// result: (NEGW x)
	for {
		if v_0.Op != OpRISCV64MOVDconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_1
		v.reset(OpRISCV64NEGW)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64XOR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XOR (MOVDconst [val]) x)
	// cond: is32Bit(val)
	// result: (XORI [val] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpRISCV64MOVDconst {
				continue
			}
			val := auxIntToInt64(v_0.AuxInt)
			x := v_1
			if !(is32Bit(val)) {
				continue
			}
			v.reset(OpRISCV64XORI)
			v.AuxInt = int64ToAuxInt(val)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValueRISCV64_OpRotateLeft16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft16 <t> x (MOVDconst [c]))
	// result: (Or16 (Lsh16x64 <t> x (MOVDconst [c&15])) (Rsh16Ux64 <t> x (MOVDconst [-c&15])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpRISCV64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpOr16)
		v0 := b.NewValue0(v.Pos, OpLsh16x64, t)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v1.AuxInt = int64ToAuxInt(c & 15)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpRsh16Ux64, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(-c & 15)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRotateLeft32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft32 <t> x (MOVDconst [c]))
	// result: (Or32 (Lsh32x64 <t> x (MOVDconst [c&31])) (Rsh32Ux64 <t> x (MOVDconst [-c&31])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpRISCV64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpOr32)
		v0 := b.NewValue0(v.Pos, OpLsh32x64, t)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v1.AuxInt = int64ToAuxInt(c & 31)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpRsh32Ux64, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(-c & 31)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRotateLeft64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft64 <t> x (MOVDconst [c]))
	// result: (Or64 (Lsh64x64 <t> x (MOVDconst [c&63])) (Rsh64Ux64 <t> x (MOVDconst [-c&63])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpRISCV64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpOr64)
		v0 := b.NewValue0(v.Pos, OpLsh64x64, t)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v1.AuxInt = int64ToAuxInt(c & 63)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpRsh64Ux64, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(-c & 63)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRotateLeft8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft8 <t> x (MOVDconst [c]))
	// result: (Or8 (Lsh8x64 <t> x (MOVDconst [c&7])) (Rsh8Ux64 <t> x (MOVDconst [-c&7])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpRISCV64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpOr8)
		v0 := b.NewValue0(v.Pos, OpLsh8x64, t)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v1.AuxInt = int64ToAuxInt(c & 7)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpRsh8Ux64, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(-c & 7)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh16Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt16to64 x) y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpNeg16, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh16Ux16 x y)
	// cond: shiftIsBounded(v)
	// result: (SRL (ZeroExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh16Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt16to64 x) y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpNeg16, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh16Ux32 x y)
	// cond: shiftIsBounded(v)
	// result: (SRL (ZeroExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh16Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt16to64 x) y) (Neg16 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpNeg16, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh16Ux64 x y)
	// cond: shiftIsBounded(v)
	// result: (SRL (ZeroExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh16Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt16to64 x) y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpNeg16, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh16Ux8 x y)
	// cond: shiftIsBounded(v)
	// result: (SRL (ZeroExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA <t> (SignExt16to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt16to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh16x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SRA (SignExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA <t> (SignExt16to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt32to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh16x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SRA (SignExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA <t> (SignExt16to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh16x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SRA (SignExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA <t> (SignExt16to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt8to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh16x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SRA (SignExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh32Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt32to64 x) y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpNeg32, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh32Ux16 x y)
	// cond: shiftIsBounded(v)
	// result: (SRL (ZeroExt32to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh32Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt32to64 x) y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpNeg32, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh32Ux32 x y)
	// cond: shiftIsBounded(v)
	// result: (SRL (ZeroExt32to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh32Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt32to64 x) y) (Neg32 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpNeg32, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh32Ux64 x y)
	// cond: shiftIsBounded(v)
	// result: (SRL (ZeroExt32to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh32Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt32to64 x) y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpNeg32, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh32Ux8 x y)
	// cond: shiftIsBounded(v)
	// result: (SRL (ZeroExt32to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA <t> (SignExt32to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt16to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh32x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SRA (SignExt32to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA <t> (SignExt32to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt32to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh32x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SRA (SignExt32to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA <t> (SignExt32to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh32x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SRA (SignExt32to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA <t> (SignExt32to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt8to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh32x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SRA (SignExt32to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh64Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SRL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg64, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh64Ux16 x y)
	// cond: shiftIsBounded(v)
	// result: (SRL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh64Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SRL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg64, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh64Ux32 x y)
	// cond: shiftIsBounded(v)
	// result: (SRL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh64Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64Ux64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SRL <t> x y) (Neg64 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg64, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh64Ux64 x y)
	// cond: shiftIsBounded(v)
	// result: (SRL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh64Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SRL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpNeg64, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh64Ux8 x y)
	// cond: shiftIsBounded(v)
	// result: (SRL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt16to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v1 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v1.AuxInt = int64ToAuxInt(-1)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SRA x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt32to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v1 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v1.AuxInt = int64ToAuxInt(-1)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SRA x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v1 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v1.AuxInt = int64ToAuxInt(-1)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v2.AuxInt = int64ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SRA x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt8to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v1 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v1.AuxInt = int64ToAuxInt(-1)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SRA x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh8Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt8to64 x) y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpNeg8, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh8Ux16 x y)
	// cond: shiftIsBounded(v)
	// result: (SRL (ZeroExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh8Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt8to64 x) y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpNeg8, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh8Ux32 x y)
	// cond: shiftIsBounded(v)
	// result: (SRL (ZeroExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh8Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt8to64 x) y) (Neg8 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpNeg8, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh8Ux64 x y)
	// cond: shiftIsBounded(v)
	// result: (SRL (ZeroExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh8Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt8to64 x) y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpNeg8, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh8Ux8 x y)
	// cond: shiftIsBounded(v)
	// result: (SRL (ZeroExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA <t> (SignExt8to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt16to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh8x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SRA (SignExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA <t> (SignExt8to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt32to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh8x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SRA (SignExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA <t> (SignExt8to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh8x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SRA (SignExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA <t> (SignExt8to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt8to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh8x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SRA (SignExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpSelect0(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select0 (Add64carry x y c))
	// result: (ADD (ADD <typ.UInt64> x y) c)
	for {
		if v_0.Op != OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpRISCV64ADD)
		v0 := b.NewValue0(v.Pos, OpRISCV64ADD, typ.UInt64)
		v0.AddArg2(x, y)
		v.AddArg2(v0, c)
		return true
	}
	// match: (Select0 (Sub64borrow x y c))
	// result: (SUB (SUB <typ.UInt64> x y) c)
	for {
		if v_0.Op != OpSub64borrow {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpRISCV64SUB)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, typ.UInt64)
		v0.AddArg2(x, y)
		v.AddArg2(v0, c)
		return true
	}
	// match: (Select0 m:(LoweredMuluhilo x y))
	// cond: m.Uses == 1
	// result: (MULHU x y)
	for {
		m := v_0
		if m.Op != OpRISCV64LoweredMuluhilo {
			break
		}
		y := m.Args[1]
		x := m.Args[0]
		if !(m.Uses == 1) {
			break
		}
		v.reset(OpRISCV64MULHU)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpSelect1(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select1 (Add64carry x y c))
	// result: (OR (SLTU <typ.UInt64> s:(ADD <typ.UInt64> x y) x) (SLTU <typ.UInt64> (ADD <typ.UInt64> s c) s))
	for {
		if v_0.Op != OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpRISCV64OR)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLTU, typ.UInt64)
		s := b.NewValue0(v.Pos, OpRISCV64ADD, typ.UInt64)
		s.AddArg2(x, y)
		v0.AddArg2(s, x)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTU, typ.UInt64)
		v3 := b.NewValue0(v.Pos, OpRISCV64ADD, typ.UInt64)
		v3.AddArg2(s, c)
		v2.AddArg2(v3, s)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Select1 (Sub64borrow x y c))
	// result: (OR (SLTU <typ.UInt64> x s:(SUB <typ.UInt64> x y)) (SLTU <typ.UInt64> s (SUB <typ.UInt64> s c)))
	for {
		if v_0.Op != OpSub64borrow {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpRISCV64OR)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLTU, typ.UInt64)
		s := b.NewValue0(v.Pos, OpRISCV64SUB, typ.UInt64)
		s.AddArg2(x, y)
		v0.AddArg2(x, s)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTU, typ.UInt64)
		v3 := b.NewValue0(v.Pos, OpRISCV64SUB, typ.UInt64)
		v3.AddArg2(s, c)
		v2.AddArg2(s, v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Select1 m:(LoweredMuluhilo x y))
	// cond: m.Uses == 1
	// result: (MUL x y)
	for {
		m := v_0
		if m.Op != OpRISCV64LoweredMuluhilo {
			break
		}
		y := m.Args[1]
		x := m.Args[0]
		if !(m.Uses == 1) {
			break
		}
		v.reset(OpRISCV64MUL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpSlicemask(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Slicemask <t> x)
	// result: (SRAI [63] (NEG <t> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpRISCV64SRAI)
		v.AuxInt = int64ToAuxInt(63)
		v0 := b.NewValue0(v.Pos, OpRISCV64NEG, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpStore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 1
	// result: (MOVBstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 1) {
			break
		}
		v.reset(OpRISCV64MOVBstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 2
	// result: (MOVHstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 2) {
			break
		}
		v.reset(OpRISCV64MOVHstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && !is32BitFloat(val.Type)
	// result: (MOVWstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && !is32BitFloat(val.Type)) {
			break
		}
		v.reset(OpRISCV64MOVWstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && !is64BitFloat(val.Type)
	// result: (MOVDstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && !is64BitFloat(val.Type)) {
			break
		}
		v.reset(OpRISCV64MOVDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && is32BitFloat(val.Type)
	// result: (FMOVWstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && is32BitFloat(val.Type)) {
			break
		}
		v.reset(OpRISCV64FMOVWstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && is64BitFloat(val.Type)
	// result: (FMOVDstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && is64BitFloat(val.Type)) {
			break
		}
		v.reset(OpRISCV64FMOVDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpZero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Zero [0] _ mem)
	// result: mem
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		mem := v_1
		v.copyOf(mem)
		return true
	}
	// match: (Zero [1] ptr mem)
	// result: (MOVBstore ptr (MOVDconst [0]) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 1 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpRISCV64MOVBstore)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [2] {t} ptr mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore ptr (MOVDconst [0]) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 2 {
			break
		}
		t := auxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.reset(OpRISCV64MOVHstore)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [2] ptr mem)
	// result: (MOVBstore [1] ptr (MOVDconst [0]) (MOVBstore ptr (MOVDconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 2 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpRISCV64MOVBstore)
		v.AuxInt = int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVBstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [4] {t} ptr mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore ptr (MOVDconst [0]) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 4 {
			break
		}
		t := auxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.reset(OpRISCV64MOVWstore)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [4] {t} ptr mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [2] ptr (MOVDconst [0]) (MOVHstore ptr (MOVDconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 4 {
			break
		}
		t := auxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.reset(OpRISCV64MOVHstore)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVHstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [4] ptr mem)
	// result: (MOVBstore [3] ptr (MOVDconst [0]) (MOVBstore [2] ptr (MOVDconst [0]) (MOVBstore [1] ptr (MOVDconst [0]) (MOVBstore ptr (MOVDconst [0]) mem))))
	for {
		if auxIntToInt64(v.AuxInt) != 4 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpRISCV64MOVBstore)
		v.AuxInt = int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVBstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(2)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVBstore, types.TypeMem)
		v2.AuxInt = int32ToAuxInt(1)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVBstore, types.TypeMem)
		v3.AddArg3(ptr, v0, mem)
		v2.AddArg3(ptr, v0, v3)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [8] {t} ptr mem)
	// cond: t.Alignment()%8 == 0
	// result: (MOVDstore ptr (MOVDconst [0]) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		t := auxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%8 == 0) {
			break
		}
		v.reset(OpRISCV64MOVDstore)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [8] {t} ptr mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore [4] ptr (MOVDconst [0]) (MOVWstore ptr (MOVDconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		t := auxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.reset(OpRISCV64MOVWstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVWstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [8] {t} ptr mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [6] ptr (MOVDconst [0]) (MOVHstore [4] ptr (MOVDconst [0]) (MOVHstore [2] ptr (MOVDconst [0]) (MOVHstore ptr (MOVDconst [0]) mem))))
	for {
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		t := auxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.reset(OpRISCV64MOVHstore)
		v.AuxInt = int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVHstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(4)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVHstore, types.TypeMem)
		v2.AuxInt = int32ToAuxInt(2)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVHstore, types.TypeMem)
		v3.AddArg3(ptr, v0, mem)
		v2.AddArg3(ptr, v0, v3)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [3] ptr mem)
	// result: (MOVBstore [2] ptr (MOVDconst [0]) (MOVBstore [1] ptr (MOVDconst [0]) (MOVBstore ptr (MOVDconst [0]) mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 3 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpRISCV64MOVBstore)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVBstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(1)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVBstore, types.TypeMem)
		v2.AddArg3(ptr, v0, mem)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [6] {t} ptr mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [4] ptr (MOVDconst [0]) (MOVHstore [2] ptr (MOVDconst [0]) (MOVHstore ptr (MOVDconst [0]) mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 6 {
			break
		}
		t := auxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.reset(OpRISCV64MOVHstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVHstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(2)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVHstore, types.TypeMem)
		v2.AddArg3(ptr, v0, mem)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [12] {t} ptr mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore [8] ptr (MOVDconst [0]) (MOVWstore [4] ptr (MOVDconst [0]) (MOVWstore ptr (MOVDconst [0]) mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 12 {
			break
		}
		t := auxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.reset(OpRISCV64MOVWstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVWstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(4)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVWstore, types.TypeMem)
		v2.AddArg3(ptr, v0, mem)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [16] {t} ptr mem)
	// cond: t.Alignment()%8 == 0
	// result: (MOVDstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 16 {
			break
		}
		t := auxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%8 == 0) {
			break
		}
		v.reset(OpRISCV64MOVDstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVDstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [24] {t} ptr mem)
	// cond: t.Alignment()%8 == 0
	// result: (MOVDstore [16] ptr (MOVDconst [0]) (MOVDstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 24 {
			break
		}
		t := auxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%8 == 0) {
			break
		}
		v.reset(OpRISCV64MOVDstore)
		v.AuxInt = int32ToAuxInt(16)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVDstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(8)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVDstore, types.TypeMem)
		v2.AddArg3(ptr, v0, mem)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [32] {t} ptr mem)
	// cond: t.Alignment()%8 == 0
	// result: (MOVDstore [24] ptr (MOVDconst [0]) (MOVDstore [16] ptr (MOVDconst [0]) (MOVDstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))))
	for {
		if auxIntToInt64(v.AuxInt) != 32 {
			break
		}
		t := auxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%8 == 0) {
			break
		}
		v.reset(OpRISCV64MOVDstore)
		v.AuxInt = int32ToAuxInt(24)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVDstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(16)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVDstore, types.TypeMem)
		v2.AuxInt = int32ToAuxInt(8)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVDstore, types.TypeMem)
		v3.AddArg3(ptr, v0, mem)
		v2.AddArg3(ptr, v0, v3)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [s] {t} ptr mem)
	// cond: s%8 == 0 && s <= 8*128 && t.Alignment()%8 == 0 && !config.noDuffDevice
	// result: (DUFFZERO [8 * (128 - s/8)] ptr mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		t := auxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(s%8 == 0 && s <= 8*128 && t.Alignment()%8 == 0 && !config.noDuffDevice) {
			break
		}
		v.reset(OpRISCV64DUFFZERO)
		v.AuxInt = int64ToAuxInt(8 * (128 - s/8))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Zero [s] {t} ptr mem)
	// result: (LoweredZero [t.Alignment()] ptr (ADD <ptr.Type> ptr (MOVDconst [s-moveSize(t.Alignment(), config)])) mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		t := auxToType(v.Aux)
		ptr := v_0
		mem := v_1
		v.reset(OpRISCV64LoweredZero)
		v.AuxInt = int64ToAuxInt(t.Alignment())
		v0 := b.NewValue0(v.Pos, OpRISCV64ADD, ptr.Type)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v1.AuxInt = int64ToAuxInt(s - moveSize(t.Alignment(), config))
		v0.AddArg2(ptr, v1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
}
func rewriteBlockRISCV64(b *Block) bool {
	typ := &b.Func.Config.Types
	switch b.Kind {
	case BlockRISCV64BEQ:
		// match: (BEQ (MOVDconst [0]) cond yes no)
		// result: (BEQZ cond yes no)
		for b.Controls[0].Op == OpRISCV64MOVDconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			cond := b.Controls[1]
			b.resetWithControl(BlockRISCV64BEQZ, cond)
			return true
		}
		// match: (BEQ cond (MOVDconst [0]) yes no)
		// result: (BEQZ cond yes no)
		for b.Controls[1].Op == OpRISCV64MOVDconst {
			cond := b.Controls[0]
			v_1 := b.Controls[1]
			if auxIntToInt64(v_1.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockRISCV64BEQZ, cond)
			return true
		}
	case BlockRISCV64BEQZ:
		// match: (BEQZ (SEQZ x) yes no)
		// result: (BNEZ x yes no)
		for b.Controls[0].Op == OpRISCV64SEQZ {
			v_0 := b.Controls[0]
			x := v_0.Args[0]
			b.resetWithControl(BlockRISCV64BNEZ, x)
			return true
		}
		// match: (BEQZ (SNEZ x) yes no)
		// result: (BEQZ x yes no)
		for b.Controls[0].Op == OpRISCV64SNEZ {
			v_0 := b.Controls[0]
			x := v_0.Args[0]
			b.resetWithControl(BlockRISCV64BEQZ, x)
			return true
		}
		// match: (BEQZ (NEG x) yes no)
		// result: (BEQZ x yes no)
		for b.Controls[0].Op == OpRISCV64NEG {
			v_0 := b.Controls[0]
			x := v_0.Args[0]
			b.resetWithControl(BlockRISCV64BEQZ, x)
			return true
		}
		// match: (BEQZ (FNES <t> x y) yes no)
		// result: (BNEZ (FEQS <t> x y) yes no)
		for b.Controls[0].Op == OpRISCV64FNES {
			v_0 := b.Controls[0]
			t := v_0.Type
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				v0 := b.NewValue0(v_0.Pos, OpRISCV64FEQS, t)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockRISCV64BNEZ, v0)
				return true
			}
		}
		// match: (BEQZ (FNED <t> x y) yes no)
		// result: (BNEZ (FEQD <t> x y) yes no)
		for b.Controls[0].Op == OpRISCV64FNED {
			v_0 := b.Controls[0]
			t := v_0.Type
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				v0 := b.NewValue0(v_0.Pos, OpRISCV64FEQD, t)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockRISCV64BNEZ, v0)
				return true
			}
		}
		// match: (BEQZ (SUB x y) yes no)
		// result: (BEQ x y yes no)
		for b.Controls[0].Op == OpRISCV64SUB {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.resetWithControl2(BlockRISCV64BEQ, x, y)
			return true
		}
		// match: (BEQZ (SLT x y) yes no)
		// result: (BGE x y yes no)
		for b.Controls[0].Op == OpRISCV64SLT {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.resetWithControl2(BlockRISCV64BGE, x, y)
			return true
		}
		// match: (BEQZ (SLTU x y) yes no)
		// result: (BGEU x y yes no)
		for b.Controls[0].Op == OpRISCV64SLTU {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.resetWithControl2(BlockRISCV64BGEU, x, y)
			return true
		}
	case BlockRISCV64BGE:
		// match: (BGE (MOVDconst [0]) cond yes no)
		// result: (BLEZ cond yes no)
		for b.Controls[0].Op == OpRISCV64MOVDconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			cond := b.Controls[1]
			b.resetWithControl(BlockRISCV64BLEZ, cond)
			return true
		}
		// match: (BGE cond (MOVDconst [0]) yes no)
		// result: (BGEZ cond yes no)
		for b.Controls[1].Op == OpRISCV64MOVDconst {
			cond := b.Controls[0]
			v_1 := b.Controls[1]
			if auxIntToInt64(v_1.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockRISCV64BGEZ, cond)
			return true
		}
	case BlockRISCV64BLT:
		// match: (BLT (MOVDconst [0]) cond yes no)
		// result: (BGTZ cond yes no)
		for b.Controls[0].Op == OpRISCV64MOVDconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			cond := b.Controls[1]
			b.resetWithControl(BlockRISCV64BGTZ, cond)
			return true
		}
		// match: (BLT cond (MOVDconst [0]) yes no)
		// result: (BLTZ cond yes no)
		for b.Controls[1].Op == OpRISCV64MOVDconst {
			cond := b.Controls[0]
			v_1 := b.Controls[1]
			if auxIntToInt64(v_1.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockRISCV64BLTZ, cond)
			return true
		}
	case BlockRISCV64BNE:
		// match: (BNE (MOVDconst [0]) cond yes no)
		// result: (BNEZ cond yes no)
		for b.Controls[0].Op == OpRISCV64MOVDconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			cond := b.Controls[1]
			b.resetWithControl(BlockRISCV64BNEZ, cond)
			return true
		}
		// match: (BNE cond (MOVDconst [0]) yes no)
		// result: (BNEZ cond yes no)
		for b.Controls[1].Op == OpRISCV64MOVDconst {
			cond := b.Controls[0]
			v_1 := b.Controls[1]
			if auxIntToInt64(v_1.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockRISCV64BNEZ, cond)
			return true
		}
	case BlockRISCV64BNEZ:
		// match: (BNEZ (SEQZ x) yes no)
		// result: (BEQZ x yes no)
		for b.Controls[0].Op == OpRISCV64SEQZ {
			v_0 := b.Controls[0]
			x := v_0.Args[0]
			b.resetWithControl(BlockRISCV64BEQZ, x)
			return true
		}
		// match: (BNEZ (SNEZ x) yes no)
		// result: (BNEZ x yes no)
		for b.Controls[0].Op == OpRISCV64SNEZ {
			v_0 := b.Controls[0]
			x := v_0.Args[0]
			b.resetWithControl(BlockRISCV64BNEZ, x)
			return true
		}
		// match: (BNEZ (NEG x) yes no)
		// result: (BNEZ x yes no)
		for b.Controls[0].Op == OpRISCV64NEG {
			v_0 := b.Controls[0]
			x := v_0.Args[0]
			b.resetWithControl(BlockRISCV64BNEZ, x)
			return true
		}
		// match: (BNEZ (FNES <t> x y) yes no)
		// result: (BEQZ (FEQS <t> x y) yes no)
		for b.Controls[0].Op == OpRISCV64FNES {
			v_0 := b.Controls[0]
			t := v_0.Type
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				v0 := b.NewValue0(v_0.Pos, OpRISCV64FEQS, t)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockRISCV64BEQZ, v0)
				return true
			}
		}
		// match: (BNEZ (FNED <t> x y) yes no)
		// result: (BEQZ (FEQD <t> x y) yes no)
		for b.Controls[0].Op == OpRISCV64FNED {
			v_0 := b.Controls[0]
			t := v_0.Type
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				v0 := b.NewValue0(v_0.Pos, OpRISCV64FEQD, t)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockRISCV64BEQZ, v0)
				return true
			}
		}
		// match: (BNEZ (SUB x y) yes no)
		// result: (BNE x y yes no)
		for b.Controls[0].Op == OpRISCV64SUB {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.resetWithControl2(BlockRISCV64BNE, x, y)
			return true
		}
		// match: (BNEZ (SLT x y) yes no)
		// result: (BLT x y yes no)
		for b.Controls[0].Op == OpRISCV64SLT {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.resetWithControl2(BlockRISCV64BLT, x, y)
			return true
		}
		// match: (BNEZ (SLTU x y) yes no)
		// result: (BLTU x y yes no)
		for b.Controls[0].Op == OpRISCV64SLTU {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.resetWithControl2(BlockRISCV64BLTU, x, y)
			return true
		}
	case BlockIf:
		// match: (If cond yes no)
		// result: (BNEZ (MOVBUreg <typ.UInt64> cond) yes no)
		for {
			cond := b.Controls[0]
			v0 := b.NewValue0(cond.Pos, OpRISCV64MOVBUreg, typ.UInt64)
			v0.AddArg(cond)
			b.resetWithControl(BlockRISCV64BNEZ, v0)
			return true
		}
	}
	return false
}
