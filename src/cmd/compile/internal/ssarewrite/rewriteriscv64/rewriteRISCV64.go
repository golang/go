// Code generated from _gen/RISCV64.rules using 'go generate'; DO NOT EDIT.

package rewriteriscv64

import "internal/buildcfg"
import "math"
import "math/bits"
import "cmd/compile/internal/types"
import "cmd/compile/internal/ssa/block"
import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa"

func RewriteValue(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.OpAbs:
		v.Op = ssaop.OpRISCV64FABSD
		return true
	case ssaop.OpAdd16:
		v.Op = ssaop.OpRISCV64ADD
		return true
	case ssaop.OpAdd32:
		v.Op = ssaop.OpRISCV64ADD
		return true
	case ssaop.OpAdd32F:
		v.Op = ssaop.OpRISCV64FADDS
		return true
	case ssaop.OpAdd64:
		v.Op = ssaop.OpRISCV64ADD
		return true
	case ssaop.OpAdd64F:
		v.Op = ssaop.OpRISCV64FADDD
		return true
	case ssaop.OpAdd8:
		v.Op = ssaop.OpRISCV64ADD
		return true
	case ssaop.OpAddPtr:
		v.Op = ssaop.OpRISCV64ADD
		return true
	case ssaop.OpAddr:
		return rewriteValue_OpAddr(v)
	case ssaop.OpAnd16:
		v.Op = ssaop.OpRISCV64AND
		return true
	case ssaop.OpAnd32:
		v.Op = ssaop.OpRISCV64AND
		return true
	case ssaop.OpAnd64:
		v.Op = ssaop.OpRISCV64AND
		return true
	case ssaop.OpAnd8:
		v.Op = ssaop.OpRISCV64AND
		return true
	case ssaop.OpAndB:
		v.Op = ssaop.OpRISCV64AND
		return true
	case ssaop.OpAtomicAdd32:
		v.Op = ssaop.OpRISCV64LoweredAtomicAdd32
		return true
	case ssaop.OpAtomicAdd64:
		v.Op = ssaop.OpRISCV64LoweredAtomicAdd64
		return true
	case ssaop.OpAtomicAnd32:
		v.Op = ssaop.OpRISCV64LoweredAtomicAnd32
		return true
	case ssaop.OpAtomicAnd8:
		return rewriteValue_OpAtomicAnd8(v)
	case ssaop.OpAtomicCompareAndSwap32:
		return rewriteValue_OpAtomicCompareAndSwap32(v)
	case ssaop.OpAtomicCompareAndSwap64:
		v.Op = ssaop.OpRISCV64LoweredAtomicCas64
		return true
	case ssaop.OpAtomicExchange32:
		v.Op = ssaop.OpRISCV64LoweredAtomicExchange32
		return true
	case ssaop.OpAtomicExchange64:
		v.Op = ssaop.OpRISCV64LoweredAtomicExchange64
		return true
	case ssaop.OpAtomicLoad32:
		v.Op = ssaop.OpRISCV64LoweredAtomicLoad32
		return true
	case ssaop.OpAtomicLoad64:
		v.Op = ssaop.OpRISCV64LoweredAtomicLoad64
		return true
	case ssaop.OpAtomicLoad8:
		v.Op = ssaop.OpRISCV64LoweredAtomicLoad8
		return true
	case ssaop.OpAtomicLoadPtr:
		v.Op = ssaop.OpRISCV64LoweredAtomicLoad64
		return true
	case ssaop.OpAtomicOr32:
		v.Op = ssaop.OpRISCV64LoweredAtomicOr32
		return true
	case ssaop.OpAtomicOr8:
		return rewriteValue_OpAtomicOr8(v)
	case ssaop.OpAtomicStore32:
		v.Op = ssaop.OpRISCV64LoweredAtomicStore32
		return true
	case ssaop.OpAtomicStore64:
		v.Op = ssaop.OpRISCV64LoweredAtomicStore64
		return true
	case ssaop.OpAtomicStore8:
		v.Op = ssaop.OpRISCV64LoweredAtomicStore8
		return true
	case ssaop.OpAtomicStorePtrNoWB:
		v.Op = ssaop.OpRISCV64LoweredAtomicStore64
		return true
	case ssaop.OpAvg64u:
		return rewriteValue_OpAvg64u(v)
	case ssaop.OpBitLen16:
		return rewriteValue_OpBitLen16(v)
	case ssaop.OpBitLen32:
		return rewriteValue_OpBitLen32(v)
	case ssaop.OpBitLen64:
		return rewriteValue_OpBitLen64(v)
	case ssaop.OpBitLen8:
		return rewriteValue_OpBitLen8(v)
	case ssaop.OpBswap16:
		return rewriteValue_OpBswap16(v)
	case ssaop.OpBswap32:
		return rewriteValue_OpBswap32(v)
	case ssaop.OpBswap64:
		v.Op = ssaop.OpRISCV64REV8
		return true
	case ssaop.OpClosureCall:
		v.Op = ssaop.OpRISCV64CALLclosure
		return true
	case ssaop.OpCom16:
		v.Op = ssaop.OpRISCV64NOT
		return true
	case ssaop.OpCom32:
		v.Op = ssaop.OpRISCV64NOT
		return true
	case ssaop.OpCom64:
		v.Op = ssaop.OpRISCV64NOT
		return true
	case ssaop.OpCom8:
		v.Op = ssaop.OpRISCV64NOT
		return true
	case ssaop.OpCondSelect:
		return rewriteValue_OpCondSelect(v)
	case ssaop.OpConst16:
		return rewriteValue_OpConst16(v)
	case ssaop.OpConst32:
		return rewriteValue_OpConst32(v)
	case ssaop.OpConst32F:
		v.Op = ssaop.OpRISCV64FMOVFconst
		return true
	case ssaop.OpConst64:
		return rewriteValue_OpConst64(v)
	case ssaop.OpConst64F:
		v.Op = ssaop.OpRISCV64FMOVDconst
		return true
	case ssaop.OpConst8:
		return rewriteValue_OpConst8(v)
	case ssaop.OpConstBool:
		return rewriteValue_OpConstBool(v)
	case ssaop.OpConstNil:
		return rewriteValue_OpConstNil(v)
	case ssaop.OpCopysign:
		v.Op = ssaop.OpRISCV64FSGNJD
		return true
	case ssaop.OpCtz16:
		return rewriteValue_OpCtz16(v)
	case ssaop.OpCtz16NonZero:
		v.Op = ssaop.OpCtz64
		return true
	case ssaop.OpCtz32:
		v.Op = ssaop.OpRISCV64CTZW
		return true
	case ssaop.OpCtz32NonZero:
		v.Op = ssaop.OpCtz64
		return true
	case ssaop.OpCtz64:
		v.Op = ssaop.OpRISCV64CTZ
		return true
	case ssaop.OpCtz64NonZero:
		v.Op = ssaop.OpCtz64
		return true
	case ssaop.OpCtz8:
		return rewriteValue_OpCtz8(v)
	case ssaop.OpCtz8NonZero:
		v.Op = ssaop.OpCtz64
		return true
	case ssaop.OpCvt32Fto32:
		v.Op = ssaop.OpRISCV64FCVTWS
		return true
	case ssaop.OpCvt32Fto64:
		v.Op = ssaop.OpRISCV64FCVTLS
		return true
	case ssaop.OpCvt32Fto64F:
		v.Op = ssaop.OpRISCV64FCVTDS
		return true
	case ssaop.OpCvt32to32F:
		v.Op = ssaop.OpRISCV64FCVTSW
		return true
	case ssaop.OpCvt32to64F:
		v.Op = ssaop.OpRISCV64FCVTDW
		return true
	case ssaop.OpCvt64Fto32:
		v.Op = ssaop.OpRISCV64FCVTWD
		return true
	case ssaop.OpCvt64Fto32F:
		v.Op = ssaop.OpRISCV64FCVTSD
		return true
	case ssaop.OpCvt64Fto64:
		v.Op = ssaop.OpRISCV64FCVTLD
		return true
	case ssaop.OpCvt64to32F:
		v.Op = ssaop.OpRISCV64FCVTSL
		return true
	case ssaop.OpCvt64to64F:
		v.Op = ssaop.OpRISCV64FCVTDL
		return true
	case ssaop.OpCvtBoolToUint8:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpDiv16:
		return rewriteValue_OpDiv16(v)
	case ssaop.OpDiv16u:
		return rewriteValue_OpDiv16u(v)
	case ssaop.OpDiv32:
		return rewriteValue_OpDiv32(v)
	case ssaop.OpDiv32F:
		v.Op = ssaop.OpRISCV64FDIVS
		return true
	case ssaop.OpDiv32u:
		v.Op = ssaop.OpRISCV64DIVUW
		return true
	case ssaop.OpDiv64:
		return rewriteValue_OpDiv64(v)
	case ssaop.OpDiv64F:
		v.Op = ssaop.OpRISCV64FDIVD
		return true
	case ssaop.OpDiv64u:
		v.Op = ssaop.OpRISCV64DIVU
		return true
	case ssaop.OpDiv8:
		return rewriteValue_OpDiv8(v)
	case ssaop.OpDiv8u:
		return rewriteValue_OpDiv8u(v)
	case ssaop.OpEq16:
		return rewriteValue_OpEq16(v)
	case ssaop.OpEq32:
		return rewriteValue_OpEq32(v)
	case ssaop.OpEq32F:
		v.Op = ssaop.OpRISCV64FEQS
		return true
	case ssaop.OpEq64:
		return rewriteValue_OpEq64(v)
	case ssaop.OpEq64F:
		v.Op = ssaop.OpRISCV64FEQD
		return true
	case ssaop.OpEq8:
		return rewriteValue_OpEq8(v)
	case ssaop.OpEqB:
		return rewriteValue_OpEqB(v)
	case ssaop.OpEqPtr:
		return rewriteValue_OpEqPtr(v)
	case ssaop.OpFMA:
		v.Op = ssaop.OpRISCV64FMADDD
		return true
	case ssaop.OpGetCallerPC:
		v.Op = ssaop.OpRISCV64LoweredGetCallerPC
		return true
	case ssaop.OpGetCallerSP:
		v.Op = ssaop.OpRISCV64LoweredGetCallerSP
		return true
	case ssaop.OpGetClosurePtr:
		v.Op = ssaop.OpRISCV64LoweredGetClosurePtr
		return true
	case ssaop.OpHmul32:
		return rewriteValue_OpHmul32(v)
	case ssaop.OpHmul32u:
		return rewriteValue_OpHmul32u(v)
	case ssaop.OpHmul64:
		v.Op = ssaop.OpRISCV64MULH
		return true
	case ssaop.OpHmul64u:
		v.Op = ssaop.OpRISCV64MULHU
		return true
	case ssaop.OpInterCall:
		v.Op = ssaop.OpRISCV64CALLinter
		return true
	case ssaop.OpIsInBounds:
		v.Op = ssaop.OpLess64U
		return true
	case ssaop.OpIsNonNil:
		v.Op = ssaop.OpRISCV64SNEZ
		return true
	case ssaop.OpIsSliceInBounds:
		v.Op = ssaop.OpLeq64U
		return true
	case ssaop.OpLeq16:
		return rewriteValue_OpLeq16(v)
	case ssaop.OpLeq16U:
		return rewriteValue_OpLeq16U(v)
	case ssaop.OpLeq32:
		return rewriteValue_OpLeq32(v)
	case ssaop.OpLeq32F:
		v.Op = ssaop.OpRISCV64FLES
		return true
	case ssaop.OpLeq32U:
		return rewriteValue_OpLeq32U(v)
	case ssaop.OpLeq64:
		return rewriteValue_OpLeq64(v)
	case ssaop.OpLeq64F:
		v.Op = ssaop.OpRISCV64FLED
		return true
	case ssaop.OpLeq64U:
		return rewriteValue_OpLeq64U(v)
	case ssaop.OpLeq8:
		return rewriteValue_OpLeq8(v)
	case ssaop.OpLeq8U:
		return rewriteValue_OpLeq8U(v)
	case ssaop.OpLess16:
		return rewriteValue_OpLess16(v)
	case ssaop.OpLess16U:
		return rewriteValue_OpLess16U(v)
	case ssaop.OpLess32:
		return rewriteValue_OpLess32(v)
	case ssaop.OpLess32F:
		v.Op = ssaop.OpRISCV64FLTS
		return true
	case ssaop.OpLess32U:
		return rewriteValue_OpLess32U(v)
	case ssaop.OpLess64:
		v.Op = ssaop.OpRISCV64SLT
		return true
	case ssaop.OpLess64F:
		v.Op = ssaop.OpRISCV64FLTD
		return true
	case ssaop.OpLess64U:
		v.Op = ssaop.OpRISCV64SLTU
		return true
	case ssaop.OpLess8:
		return rewriteValue_OpLess8(v)
	case ssaop.OpLess8U:
		return rewriteValue_OpLess8U(v)
	case ssaop.OpLoad:
		return rewriteValue_OpLoad(v)
	case ssaop.OpLocalAddr:
		return rewriteValue_OpLocalAddr(v)
	case ssaop.OpLsh16x16:
		return rewriteValue_OpLsh16x16(v)
	case ssaop.OpLsh16x32:
		return rewriteValue_OpLsh16x32(v)
	case ssaop.OpLsh16x64:
		return rewriteValue_OpLsh16x64(v)
	case ssaop.OpLsh16x8:
		return rewriteValue_OpLsh16x8(v)
	case ssaop.OpLsh32x16:
		return rewriteValue_OpLsh32x16(v)
	case ssaop.OpLsh32x32:
		return rewriteValue_OpLsh32x32(v)
	case ssaop.OpLsh32x64:
		return rewriteValue_OpLsh32x64(v)
	case ssaop.OpLsh32x8:
		return rewriteValue_OpLsh32x8(v)
	case ssaop.OpLsh64x16:
		return rewriteValue_OpLsh64x16(v)
	case ssaop.OpLsh64x32:
		return rewriteValue_OpLsh64x32(v)
	case ssaop.OpLsh64x64:
		return rewriteValue_OpLsh64x64(v)
	case ssaop.OpLsh64x8:
		return rewriteValue_OpLsh64x8(v)
	case ssaop.OpLsh8x16:
		return rewriteValue_OpLsh8x16(v)
	case ssaop.OpLsh8x32:
		return rewriteValue_OpLsh8x32(v)
	case ssaop.OpLsh8x64:
		return rewriteValue_OpLsh8x64(v)
	case ssaop.OpLsh8x8:
		return rewriteValue_OpLsh8x8(v)
	case ssaop.OpMax32F:
		v.Op = ssaop.OpRISCV64LoweredFMAXS
		return true
	case ssaop.OpMax64:
		return rewriteValue_OpMax64(v)
	case ssaop.OpMax64F:
		v.Op = ssaop.OpRISCV64LoweredFMAXD
		return true
	case ssaop.OpMax64u:
		return rewriteValue_OpMax64u(v)
	case ssaop.OpMin32F:
		v.Op = ssaop.OpRISCV64LoweredFMINS
		return true
	case ssaop.OpMin64:
		return rewriteValue_OpMin64(v)
	case ssaop.OpMin64F:
		v.Op = ssaop.OpRISCV64LoweredFMIND
		return true
	case ssaop.OpMin64u:
		return rewriteValue_OpMin64u(v)
	case ssaop.OpMod16:
		return rewriteValue_OpMod16(v)
	case ssaop.OpMod16u:
		return rewriteValue_OpMod16u(v)
	case ssaop.OpMod32:
		return rewriteValue_OpMod32(v)
	case ssaop.OpMod32u:
		v.Op = ssaop.OpRISCV64REMUW
		return true
	case ssaop.OpMod64:
		return rewriteValue_OpMod64(v)
	case ssaop.OpMod64u:
		v.Op = ssaop.OpRISCV64REMU
		return true
	case ssaop.OpMod8:
		return rewriteValue_OpMod8(v)
	case ssaop.OpMod8u:
		return rewriteValue_OpMod8u(v)
	case ssaop.OpMove:
		return rewriteValue_OpMove(v)
	case ssaop.OpMul16:
		v.Op = ssaop.OpRISCV64MULW
		return true
	case ssaop.OpMul32:
		v.Op = ssaop.OpRISCV64MULW
		return true
	case ssaop.OpMul32F:
		v.Op = ssaop.OpRISCV64FMULS
		return true
	case ssaop.OpMul64:
		v.Op = ssaop.OpRISCV64MUL
		return true
	case ssaop.OpMul64F:
		v.Op = ssaop.OpRISCV64FMULD
		return true
	case ssaop.OpMul64uhilo:
		v.Op = ssaop.OpRISCV64LoweredMuluhilo
		return true
	case ssaop.OpMul64uover:
		v.Op = ssaop.OpRISCV64LoweredMuluover
		return true
	case ssaop.OpMul8:
		v.Op = ssaop.OpRISCV64MULW
		return true
	case ssaop.OpNeg16:
		v.Op = ssaop.OpRISCV64NEG
		return true
	case ssaop.OpNeg32:
		v.Op = ssaop.OpRISCV64NEG
		return true
	case ssaop.OpNeg32F:
		v.Op = ssaop.OpRISCV64FNEGS
		return true
	case ssaop.OpNeg64:
		v.Op = ssaop.OpRISCV64NEG
		return true
	case ssaop.OpNeg64F:
		v.Op = ssaop.OpRISCV64FNEGD
		return true
	case ssaop.OpNeg8:
		v.Op = ssaop.OpRISCV64NEG
		return true
	case ssaop.OpNeq16:
		return rewriteValue_OpNeq16(v)
	case ssaop.OpNeq32:
		return rewriteValue_OpNeq32(v)
	case ssaop.OpNeq32F:
		v.Op = ssaop.OpRISCV64FNES
		return true
	case ssaop.OpNeq64:
		return rewriteValue_OpNeq64(v)
	case ssaop.OpNeq64F:
		v.Op = ssaop.OpRISCV64FNED
		return true
	case ssaop.OpNeq8:
		return rewriteValue_OpNeq8(v)
	case ssaop.OpNeqB:
		return rewriteValue_OpNeqB(v)
	case ssaop.OpNeqPtr:
		return rewriteValue_OpNeqPtr(v)
	case ssaop.OpNilCheck:
		v.Op = ssaop.OpRISCV64LoweredNilCheck
		return true
	case ssaop.OpNot:
		v.Op = ssaop.OpRISCV64SEQZ
		return true
	case ssaop.OpOffPtr:
		return rewriteValue_OpOffPtr(v)
	case ssaop.OpOr16:
		v.Op = ssaop.OpRISCV64OR
		return true
	case ssaop.OpOr32:
		v.Op = ssaop.OpRISCV64OR
		return true
	case ssaop.OpOr64:
		v.Op = ssaop.OpRISCV64OR
		return true
	case ssaop.OpOr8:
		v.Op = ssaop.OpRISCV64OR
		return true
	case ssaop.OpOrB:
		v.Op = ssaop.OpRISCV64OR
		return true
	case ssaop.OpPanicBounds:
		v.Op = ssaop.OpRISCV64LoweredPanicBoundsRR
		return true
	case ssaop.OpPopCount16:
		return rewriteValue_OpPopCount16(v)
	case ssaop.OpPopCount32:
		v.Op = ssaop.OpRISCV64CPOPW
		return true
	case ssaop.OpPopCount64:
		v.Op = ssaop.OpRISCV64CPOP
		return true
	case ssaop.OpPopCount8:
		return rewriteValue_OpPopCount8(v)
	case ssaop.OpPubBarrier:
		v.Op = ssaop.OpRISCV64LoweredPubBarrier
		return true
	case ssaop.OpRISCV64ADD:
		return rewriteValue_OpRISCV64ADD(v)
	case ssaop.OpRISCV64ADDI:
		return rewriteValue_OpRISCV64ADDI(v)
	case ssaop.OpRISCV64AND:
		return rewriteValue_OpRISCV64AND(v)
	case ssaop.OpRISCV64ANDI:
		return rewriteValue_OpRISCV64ANDI(v)
	case ssaop.OpRISCV64CZEROEQZ:
		return rewriteValue_OpRISCV64CZEROEQZ(v)
	case ssaop.OpRISCV64CZERONEZ:
		return rewriteValue_OpRISCV64CZERONEZ(v)
	case ssaop.OpRISCV64FADDD:
		return rewriteValue_OpRISCV64FADDD(v)
	case ssaop.OpRISCV64FADDS:
		return rewriteValue_OpRISCV64FADDS(v)
	case ssaop.OpRISCV64FCVTSD:
		return rewriteValue_OpRISCV64FCVTSD(v)
	case ssaop.OpRISCV64FEQD:
		return rewriteValue_OpRISCV64FEQD(v)
	case ssaop.OpRISCV64FLED:
		return rewriteValue_OpRISCV64FLED(v)
	case ssaop.OpRISCV64FLTD:
		return rewriteValue_OpRISCV64FLTD(v)
	case ssaop.OpRISCV64FMADDD:
		return rewriteValue_OpRISCV64FMADDD(v)
	case ssaop.OpRISCV64FMADDS:
		return rewriteValue_OpRISCV64FMADDS(v)
	case ssaop.OpRISCV64FMOVDload:
		return rewriteValue_OpRISCV64FMOVDload(v)
	case ssaop.OpRISCV64FMOVDstore:
		return rewriteValue_OpRISCV64FMOVDstore(v)
	case ssaop.OpRISCV64FMOVWload:
		return rewriteValue_OpRISCV64FMOVWload(v)
	case ssaop.OpRISCV64FMOVWstore:
		return rewriteValue_OpRISCV64FMOVWstore(v)
	case ssaop.OpRISCV64FMSUBD:
		return rewriteValue_OpRISCV64FMSUBD(v)
	case ssaop.OpRISCV64FMSUBS:
		return rewriteValue_OpRISCV64FMSUBS(v)
	case ssaop.OpRISCV64FNED:
		return rewriteValue_OpRISCV64FNED(v)
	case ssaop.OpRISCV64FNMADDD:
		return rewriteValue_OpRISCV64FNMADDD(v)
	case ssaop.OpRISCV64FNMADDS:
		return rewriteValue_OpRISCV64FNMADDS(v)
	case ssaop.OpRISCV64FNMSUBD:
		return rewriteValue_OpRISCV64FNMSUBD(v)
	case ssaop.OpRISCV64FNMSUBS:
		return rewriteValue_OpRISCV64FNMSUBS(v)
	case ssaop.OpRISCV64FSUBD:
		return rewriteValue_OpRISCV64FSUBD(v)
	case ssaop.OpRISCV64FSUBS:
		return rewriteValue_OpRISCV64FSUBS(v)
	case ssaop.OpRISCV64LoweredPanicBoundsCR:
		return rewriteValue_OpRISCV64LoweredPanicBoundsCR(v)
	case ssaop.OpRISCV64LoweredPanicBoundsRC:
		return rewriteValue_OpRISCV64LoweredPanicBoundsRC(v)
	case ssaop.OpRISCV64LoweredPanicBoundsRR:
		return rewriteValue_OpRISCV64LoweredPanicBoundsRR(v)
	case ssaop.OpRISCV64MOVBUload:
		return rewriteValue_OpRISCV64MOVBUload(v)
	case ssaop.OpRISCV64MOVBUreg:
		return rewriteValue_OpRISCV64MOVBUreg(v)
	case ssaop.OpRISCV64MOVBload:
		return rewriteValue_OpRISCV64MOVBload(v)
	case ssaop.OpRISCV64MOVBreg:
		return rewriteValue_OpRISCV64MOVBreg(v)
	case ssaop.OpRISCV64MOVBstore:
		return rewriteValue_OpRISCV64MOVBstore(v)
	case ssaop.OpRISCV64MOVBstorezero:
		return rewriteValue_OpRISCV64MOVBstorezero(v)
	case ssaop.OpRISCV64MOVDload:
		return rewriteValue_OpRISCV64MOVDload(v)
	case ssaop.OpRISCV64MOVDnop:
		return rewriteValue_OpRISCV64MOVDnop(v)
	case ssaop.OpRISCV64MOVDreg:
		return rewriteValue_OpRISCV64MOVDreg(v)
	case ssaop.OpRISCV64MOVDstore:
		return rewriteValue_OpRISCV64MOVDstore(v)
	case ssaop.OpRISCV64MOVDstorezero:
		return rewriteValue_OpRISCV64MOVDstorezero(v)
	case ssaop.OpRISCV64MOVHUload:
		return rewriteValue_OpRISCV64MOVHUload(v)
	case ssaop.OpRISCV64MOVHUreg:
		return rewriteValue_OpRISCV64MOVHUreg(v)
	case ssaop.OpRISCV64MOVHload:
		return rewriteValue_OpRISCV64MOVHload(v)
	case ssaop.OpRISCV64MOVHreg:
		return rewriteValue_OpRISCV64MOVHreg(v)
	case ssaop.OpRISCV64MOVHstore:
		return rewriteValue_OpRISCV64MOVHstore(v)
	case ssaop.OpRISCV64MOVHstorezero:
		return rewriteValue_OpRISCV64MOVHstorezero(v)
	case ssaop.OpRISCV64MOVWUload:
		return rewriteValue_OpRISCV64MOVWUload(v)
	case ssaop.OpRISCV64MOVWUreg:
		return rewriteValue_OpRISCV64MOVWUreg(v)
	case ssaop.OpRISCV64MOVWload:
		return rewriteValue_OpRISCV64MOVWload(v)
	case ssaop.OpRISCV64MOVWreg:
		return rewriteValue_OpRISCV64MOVWreg(v)
	case ssaop.OpRISCV64MOVWstore:
		return rewriteValue_OpRISCV64MOVWstore(v)
	case ssaop.OpRISCV64MOVWstorezero:
		return rewriteValue_OpRISCV64MOVWstorezero(v)
	case ssaop.OpRISCV64MUL:
		return rewriteValue_OpRISCV64MUL(v)
	case ssaop.OpRISCV64MULW:
		return rewriteValue_OpRISCV64MULW(v)
	case ssaop.OpRISCV64NEG:
		return rewriteValue_OpRISCV64NEG(v)
	case ssaop.OpRISCV64NEGW:
		return rewriteValue_OpRISCV64NEGW(v)
	case ssaop.OpRISCV64OR:
		return rewriteValue_OpRISCV64OR(v)
	case ssaop.OpRISCV64ORI:
		return rewriteValue_OpRISCV64ORI(v)
	case ssaop.OpRISCV64ORN:
		return rewriteValue_OpRISCV64ORN(v)
	case ssaop.OpRISCV64ROL:
		return rewriteValue_OpRISCV64ROL(v)
	case ssaop.OpRISCV64ROLW:
		return rewriteValue_OpRISCV64ROLW(v)
	case ssaop.OpRISCV64ROR:
		return rewriteValue_OpRISCV64ROR(v)
	case ssaop.OpRISCV64RORW:
		return rewriteValue_OpRISCV64RORW(v)
	case ssaop.OpRISCV64SEQZ:
		return rewriteValue_OpRISCV64SEQZ(v)
	case ssaop.OpRISCV64SLL:
		return rewriteValue_OpRISCV64SLL(v)
	case ssaop.OpRISCV64SLLI:
		return rewriteValue_OpRISCV64SLLI(v)
	case ssaop.OpRISCV64SLLW:
		return rewriteValue_OpRISCV64SLLW(v)
	case ssaop.OpRISCV64SLT:
		return rewriteValue_OpRISCV64SLT(v)
	case ssaop.OpRISCV64SLTI:
		return rewriteValue_OpRISCV64SLTI(v)
	case ssaop.OpRISCV64SLTIU:
		return rewriteValue_OpRISCV64SLTIU(v)
	case ssaop.OpRISCV64SLTU:
		return rewriteValue_OpRISCV64SLTU(v)
	case ssaop.OpRISCV64SNEZ:
		return rewriteValue_OpRISCV64SNEZ(v)
	case ssaop.OpRISCV64SRA:
		return rewriteValue_OpRISCV64SRA(v)
	case ssaop.OpRISCV64SRAI:
		return rewriteValue_OpRISCV64SRAI(v)
	case ssaop.OpRISCV64SRAW:
		return rewriteValue_OpRISCV64SRAW(v)
	case ssaop.OpRISCV64SRL:
		return rewriteValue_OpRISCV64SRL(v)
	case ssaop.OpRISCV64SRLI:
		return rewriteValue_OpRISCV64SRLI(v)
	case ssaop.OpRISCV64SRLW:
		return rewriteValue_OpRISCV64SRLW(v)
	case ssaop.OpRISCV64SUB:
		return rewriteValue_OpRISCV64SUB(v)
	case ssaop.OpRISCV64SUBW:
		return rewriteValue_OpRISCV64SUBW(v)
	case ssaop.OpRISCV64XOR:
		return rewriteValue_OpRISCV64XOR(v)
	case ssaop.OpRotateLeft16:
		return rewriteValue_OpRotateLeft16(v)
	case ssaop.OpRotateLeft32:
		v.Op = ssaop.OpRISCV64ROLW
		return true
	case ssaop.OpRotateLeft64:
		v.Op = ssaop.OpRISCV64ROL
		return true
	case ssaop.OpRotateLeft8:
		return rewriteValue_OpRotateLeft8(v)
	case ssaop.OpRound32F:
		v.Op = ssaop.OpRISCV64LoweredRound32F
		return true
	case ssaop.OpRound64F:
		v.Op = ssaop.OpRISCV64LoweredRound64F
		return true
	case ssaop.OpRsh16Ux16:
		return rewriteValue_OpRsh16Ux16(v)
	case ssaop.OpRsh16Ux32:
		return rewriteValue_OpRsh16Ux32(v)
	case ssaop.OpRsh16Ux64:
		return rewriteValue_OpRsh16Ux64(v)
	case ssaop.OpRsh16Ux8:
		return rewriteValue_OpRsh16Ux8(v)
	case ssaop.OpRsh16x16:
		return rewriteValue_OpRsh16x16(v)
	case ssaop.OpRsh16x32:
		return rewriteValue_OpRsh16x32(v)
	case ssaop.OpRsh16x64:
		return rewriteValue_OpRsh16x64(v)
	case ssaop.OpRsh16x8:
		return rewriteValue_OpRsh16x8(v)
	case ssaop.OpRsh32Ux16:
		return rewriteValue_OpRsh32Ux16(v)
	case ssaop.OpRsh32Ux32:
		return rewriteValue_OpRsh32Ux32(v)
	case ssaop.OpRsh32Ux64:
		return rewriteValue_OpRsh32Ux64(v)
	case ssaop.OpRsh32Ux8:
		return rewriteValue_OpRsh32Ux8(v)
	case ssaop.OpRsh32x16:
		return rewriteValue_OpRsh32x16(v)
	case ssaop.OpRsh32x32:
		return rewriteValue_OpRsh32x32(v)
	case ssaop.OpRsh32x64:
		return rewriteValue_OpRsh32x64(v)
	case ssaop.OpRsh32x8:
		return rewriteValue_OpRsh32x8(v)
	case ssaop.OpRsh64Ux16:
		return rewriteValue_OpRsh64Ux16(v)
	case ssaop.OpRsh64Ux32:
		return rewriteValue_OpRsh64Ux32(v)
	case ssaop.OpRsh64Ux64:
		return rewriteValue_OpRsh64Ux64(v)
	case ssaop.OpRsh64Ux8:
		return rewriteValue_OpRsh64Ux8(v)
	case ssaop.OpRsh64x16:
		return rewriteValue_OpRsh64x16(v)
	case ssaop.OpRsh64x32:
		return rewriteValue_OpRsh64x32(v)
	case ssaop.OpRsh64x64:
		return rewriteValue_OpRsh64x64(v)
	case ssaop.OpRsh64x8:
		return rewriteValue_OpRsh64x8(v)
	case ssaop.OpRsh8Ux16:
		return rewriteValue_OpRsh8Ux16(v)
	case ssaop.OpRsh8Ux32:
		return rewriteValue_OpRsh8Ux32(v)
	case ssaop.OpRsh8Ux64:
		return rewriteValue_OpRsh8Ux64(v)
	case ssaop.OpRsh8Ux8:
		return rewriteValue_OpRsh8Ux8(v)
	case ssaop.OpRsh8x16:
		return rewriteValue_OpRsh8x16(v)
	case ssaop.OpRsh8x32:
		return rewriteValue_OpRsh8x32(v)
	case ssaop.OpRsh8x64:
		return rewriteValue_OpRsh8x64(v)
	case ssaop.OpRsh8x8:
		return rewriteValue_OpRsh8x8(v)
	case ssaop.OpSelect0:
		return rewriteValue_OpSelect0(v)
	case ssaop.OpSelect1:
		return rewriteValue_OpSelect1(v)
	case ssaop.OpSignExt16to32:
		v.Op = ssaop.OpRISCV64MOVHreg
		return true
	case ssaop.OpSignExt16to64:
		v.Op = ssaop.OpRISCV64MOVHreg
		return true
	case ssaop.OpSignExt32to64:
		v.Op = ssaop.OpRISCV64MOVWreg
		return true
	case ssaop.OpSignExt8to16:
		v.Op = ssaop.OpRISCV64MOVBreg
		return true
	case ssaop.OpSignExt8to32:
		v.Op = ssaop.OpRISCV64MOVBreg
		return true
	case ssaop.OpSignExt8to64:
		v.Op = ssaop.OpRISCV64MOVBreg
		return true
	case ssaop.OpSlicemask:
		return rewriteValue_OpSlicemask(v)
	case ssaop.OpSqrt:
		v.Op = ssaop.OpRISCV64FSQRTD
		return true
	case ssaop.OpSqrt32:
		v.Op = ssaop.OpRISCV64FSQRTS
		return true
	case ssaop.OpStaticCall:
		v.Op = ssaop.OpRISCV64CALLstatic
		return true
	case ssaop.OpStore:
		return rewriteValue_OpStore(v)
	case ssaop.OpSub16:
		v.Op = ssaop.OpRISCV64SUB
		return true
	case ssaop.OpSub32:
		v.Op = ssaop.OpRISCV64SUB
		return true
	case ssaop.OpSub32F:
		v.Op = ssaop.OpRISCV64FSUBS
		return true
	case ssaop.OpSub64:
		v.Op = ssaop.OpRISCV64SUB
		return true
	case ssaop.OpSub64F:
		v.Op = ssaop.OpRISCV64FSUBD
		return true
	case ssaop.OpSub8:
		v.Op = ssaop.OpRISCV64SUB
		return true
	case ssaop.OpSubPtr:
		v.Op = ssaop.OpRISCV64SUB
		return true
	case ssaop.OpTailCall:
		v.Op = ssaop.OpRISCV64CALLtail
		return true
	case ssaop.OpTailCallInter:
		v.Op = ssaop.OpRISCV64CALLtailinter
		return true
	case ssaop.OpTrunc16to8:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpTrunc32to16:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpTrunc32to8:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpTrunc64to16:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpTrunc64to32:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpTrunc64to8:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpWB:
		v.Op = ssaop.OpRISCV64LoweredWB
		return true
	case ssaop.OpXor16:
		v.Op = ssaop.OpRISCV64XOR
		return true
	case ssaop.OpXor32:
		v.Op = ssaop.OpRISCV64XOR
		return true
	case ssaop.OpXor64:
		v.Op = ssaop.OpRISCV64XOR
		return true
	case ssaop.OpXor8:
		v.Op = ssaop.OpRISCV64XOR
		return true
	case ssaop.OpZero:
		return rewriteValue_OpZero(v)
	case ssaop.OpZeroExt16to32:
		v.Op = ssaop.OpRISCV64MOVHUreg
		return true
	case ssaop.OpZeroExt16to64:
		v.Op = ssaop.OpRISCV64MOVHUreg
		return true
	case ssaop.OpZeroExt32to64:
		v.Op = ssaop.OpRISCV64MOVWUreg
		return true
	case ssaop.OpZeroExt8to16:
		v.Op = ssaop.OpRISCV64MOVBUreg
		return true
	case ssaop.OpZeroExt8to32:
		v.Op = ssaop.OpRISCV64MOVBUreg
		return true
	case ssaop.OpZeroExt8to64:
		v.Op = ssaop.OpRISCV64MOVBUreg
		return true
	}
	return false
}
func rewriteValue_OpAddr(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Addr {sym} base)
	// result: (MOVaddr {sym} [0] base)
	for {
		sym := ssa.AuxToSym(v.Aux)
		base := v_0
		v.Reset(ssaop.OpRISCV64MOVaddr)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg(base)
		return true
	}
}
func rewriteValue_OpAtomicAnd8(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpRISCV64LoweredAtomicAnd32)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, typ.Uintptr)
		v0.AuxInt = ssa.Int64ToAuxInt(^3)
		v0.AddArg(ptr)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64NOT, typ.UInt32)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64XORI, typ.UInt32)
		v3.AuxInt = ssa.Int64ToAuxInt(0xff)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v4.AddArg(val)
		v3.AddArg(v4)
		v5 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.UInt64)
		v5.AuxInt = ssa.Int64ToAuxInt(3)
		v6 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, typ.UInt64)
		v6.AuxInt = ssa.Int64ToAuxInt(3)
		v6.AddArg(ptr)
		v5.AddArg(v6)
		v2.AddArg2(v3, v5)
		v1.AddArg(v2)
		v.AddArg3(v0, v1, mem)
		return true
	}
}
func rewriteValue_OpAtomicCompareAndSwap32(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicCompareAndSwap32 ptr old new mem)
	// result: (LoweredAtomicCas32 ptr (SignExt32to64 old) new mem)
	for {
		ptr := v_0
		old := v_1
		new := v_2
		mem := v_3
		v.Reset(ssaop.OpRISCV64LoweredAtomicCas32)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(old)
		v.AddArg4(ptr, v0, new, mem)
		return true
	}
}
func rewriteValue_OpAtomicOr8(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpRISCV64LoweredAtomicOr32)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, typ.Uintptr)
		v0.AuxInt = ssa.Int64ToAuxInt(^3)
		v0.AddArg(ptr)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, typ.UInt32)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(val)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(3)
		v4 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(3)
		v4.AddArg(ptr)
		v3.AddArg(v4)
		v1.AddArg2(v2, v3)
		v.AddArg3(v0, v1, mem)
		return true
	}
}
func rewriteValue_OpAvg64u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Avg64u <t> x y)
	// result: (ADD (ADD <t> (SRLI <t> [1] x) (SRLI <t> [1] y)) (ANDI <t> [1] (AND <t> x y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64ADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRLI, t)
		v1.AuxInt = ssa.Int64ToAuxInt(1)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRLI, t)
		v2.AuxInt = ssa.Int64ToAuxInt(1)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, t)
		v3.AuxInt = ssa.Int64ToAuxInt(1)
		v4 := b.NewValue0(v.Pos, ssaop.OpRISCV64AND, t)
		v4.AddArg2(x, y)
		v3.AddArg(v4)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValue_OpBitLen16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen16 x)
	// result: (BitLen64 (ZeroExt16to64 x))
	for {
		x := v_0
		v.Reset(ssaop.OpBitLen64)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpBitLen32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen32 <t> x)
	// result: (SUB (MOVDconst [32]) (CLZW <t> x))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpRISCV64SUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(32)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64CLZW, t)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpBitLen64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen64 <t> x)
	// result: (SUB (MOVDconst [64]) (CLZ <t> x))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpRISCV64SUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(64)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64CLZ, t)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpBitLen8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen8 x)
	// result: (BitLen64 (ZeroExt8to64 x))
	for {
		x := v_0
		v.Reset(ssaop.OpBitLen64)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpBswap16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Bswap16 <t> x)
	// result: (SRLI [48] (REV8 <t> x))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpRISCV64SRLI)
		v.AuxInt = ssa.Int64ToAuxInt(48)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64REV8, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpBswap32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Bswap32 <t> x)
	// result: (SRLI [32] (REV8 <t> x))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpRISCV64SRLI)
		v.AuxInt = ssa.Int64ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64REV8, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpCondSelect(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CondSelect <t> x y cond)
	// result: (OR (CZEROEQZ <t> x (MOVBUreg <typ.UInt64> cond)) (CZERONEZ <t> y (MOVBUreg <typ.UInt64> cond)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		cond := v_2
		v.Reset(ssaop.OpRISCV64OR)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64CZEROEQZ, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBUreg, typ.UInt64)
		v1.AddArg(cond)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64CZERONEZ, t)
		v2.AddArg2(y, v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValue_OpConst16(v *ssa.Value) bool {
	// match: (Const16 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := ssa.AuxIntToInt16(v.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValue_OpConst32(v *ssa.Value) bool {
	// match: (Const32 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := ssa.AuxIntToInt32(v.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValue_OpConst64(v *ssa.Value) bool {
	// match: (Const64 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := ssa.AuxIntToInt64(v.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValue_OpConst8(v *ssa.Value) bool {
	// match: (Const8 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := ssa.AuxIntToInt8(v.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValue_OpConstBool(v *ssa.Value) bool {
	// match: (ConstBool [val])
	// result: (MOVDconst [int64(ssa.B2i(val))])
	for {
		val := ssa.AuxIntToBool(v.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(ssa.B2i(val)))
		return true
	}
}
func rewriteValue_OpConstNil(v *ssa.Value) bool {
	// match: (ConstNil)
	// result: (MOVDconst [0])
	for {
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
}
func rewriteValue_OpCtz16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz16 x)
	// result: (CTZW (ORI <typ.UInt32> [1<<16] x))
	for {
		x := v_0
		v.Reset(ssaop.OpRISCV64CTZW)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ORI, typ.UInt32)
		v0.AuxInt = ssa.Int64ToAuxInt(1 << 16)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpCtz8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz8 x)
	// result: (CTZW (ORI <typ.UInt32> [1<<8] x))
	for {
		x := v_0
		v.Reset(ssaop.OpRISCV64CTZW)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ORI, typ.UInt32)
		v0.AuxInt = ssa.Int64ToAuxInt(1 << 8)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpDiv16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 x y [false])
	// result: (DIVW (SignExt16to32 x) (SignExt16to32 y))
	for {
		if ssa.AuxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64DIVW)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValue_OpDiv16u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16u x y)
	// result: (DIVUW (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64DIVUW)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpDiv32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div32 x y [false])
	// result: (DIVW x y)
	for {
		if ssa.AuxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64DIVW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpDiv64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div64 x y [false])
	// result: (DIV x y)
	for {
		if ssa.AuxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64DIV)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpDiv8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// result: (DIVW (SignExt8to32 x) (SignExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64DIVW)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpDiv8u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// result: (DIVUW (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64DIVUW)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpEq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq16 x y)
	// result: (SEQZ (SUB <x.Type> (ZeroExt16to64 x) (ZeroExt16to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SUB, x.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpEq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq32 x y)
	// cond: x.Type.IsSigned()
	// result: (SEQZ (SUB <x.Type> (SignExt32to64 x) (SignExt32to64 y)))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			y := v_1
			if !(x.Type.IsSigned()) {
				continue
			}
			v.Reset(ssaop.OpRISCV64SEQZ)
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SUB, x.Type)
			v1 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
			v1.AddArg(x)
			v2 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
			v2.AddArg(y)
			v0.AddArg2(v1, v2)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (Eq32 x y)
	// cond: !x.Type.IsSigned()
	// result: (SEQZ (SUB <x.Type> (ZeroExt32to64 x) (ZeroExt32to64 y)))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			y := v_1
			if !(!x.Type.IsSigned()) {
				continue
			}
			v.Reset(ssaop.OpRISCV64SEQZ)
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SUB, x.Type)
			v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
			v1.AddArg(x)
			v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
			v2.AddArg(y)
			v0.AddArg2(v1, v2)
			v.AddArg(v0)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpEq64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64 x y)
	// result: (SEQZ (SUB <x.Type> x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SUB, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpEq8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq8 x y)
	// result: (SEQZ (SUB <x.Type> (ZeroExt8to64 x) (ZeroExt8to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SUB, x.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpEqB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqB x y)
	// result: (SEQZ (SUB <typ.Bool> x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SUB, typ.Bool)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpEqPtr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqPtr x y)
	// result: (SEQZ (SUB <typ.Uintptr> x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SUB, typ.Uintptr)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpHmul32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul32 x y)
	// result: (SRAI [32] (MUL (SignExt32to64 x) (SignExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64SRAI)
		v.AuxInt = ssa.Int64ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MUL, typ.Int64)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpHmul32u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul32u x y)
	// result: (SRLI [32] (MUL (ZeroExt32to64 x) (ZeroExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64SRLI)
		v.AuxInt = ssa.Int64ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MUL, typ.Int64)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16 x y)
	// result: (Not (Less16 y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpNot)
		v0 := b.NewValue0(v.Pos, ssaop.OpLess16, typ.Bool)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq16U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16U x y)
	// result: (Not (Less16U y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpNot)
		v0 := b.NewValue0(v.Pos, ssaop.OpLess16U, typ.Bool)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32 x y)
	// result: (Not (Less32 y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpNot)
		v0 := b.NewValue0(v.Pos, ssaop.OpLess32, typ.Bool)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq32U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32U x y)
	// result: (Not (Less32U y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpNot)
		v0 := b.NewValue0(v.Pos, ssaop.OpLess32U, typ.Bool)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq64 x y)
	// result: (Not (Less64 y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpNot)
		v0 := b.NewValue0(v.Pos, ssaop.OpLess64, typ.Bool)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq64U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq64U x y)
	// result: (Not (Less64U y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpNot)
		v0 := b.NewValue0(v.Pos, ssaop.OpLess64U, typ.Bool)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8 x y)
	// result: (Not (Less8 y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpNot)
		v0 := b.NewValue0(v.Pos, ssaop.OpLess8, typ.Bool)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq8U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8U x y)
	// result: (Not (Less8U y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpNot)
		v0 := b.NewValue0(v.Pos, ssaop.OpLess8U, typ.Bool)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLess16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16 x y)
	// result: (SLT (SignExt16to64 x) (SignExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64SLT)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLess16U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16U x y)
	// result: (SLTU (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64SLTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLess32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32 x y)
	// result: (SLT (SignExt32to64 x) (SignExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64SLT)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLess32U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32U x y)
	// result: (SLTU (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64SLTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLess8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8 x y)
	// result: (SLT (SignExt8to64 x) (SignExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64SLT)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLess8U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8U x y)
	// result: (SLTU (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64SLTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLoad(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpRISCV64MOVBUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ( ssa.Is8BitInt(t) && t.IsSigned())
	// result: (MOVBload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is8BitInt(t) && t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVBload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ( ssa.Is8BitInt(t) && !t.IsSigned())
	// result: (MOVBUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is8BitInt(t) && !t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVBUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (ssa.Is16BitInt(t) && t.IsSigned())
	// result: (MOVHload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is16BitInt(t) && t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (ssa.Is16BitInt(t) && !t.IsSigned())
	// result: (MOVHUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is16BitInt(t) && !t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (ssa.Is32BitInt(t) && t.IsSigned())
	// result: (MOVWload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is32BitInt(t) && t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVWload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (ssa.Is32BitInt(t) && !t.IsSigned())
	// result: (MOVWUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is32BitInt(t) && !t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVWUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (ssa.Is64BitInt(t) ||ssa.IsPtr(t))
	// result: (MOVDload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is64BitInt(t) || ssa.IsPtr(t)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ssa.Is32BitFloat(t)
	// result: (FMOVWload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is32BitFloat(t)) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMOVWload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ssa.Is64BitFloat(t)
	// result: (FMOVDload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is64BitFloat(t)) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMOVDload)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpLocalAddr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LocalAddr <t> {sym} base mem)
	// cond: t.Elem().HasPointers()
	// result: (MOVaddr {sym} (SPanchored base mem))
	for {
		t := v.Type
		sym := ssa.AuxToSym(v.Aux)
		base := v_0
		mem := v_1
		if !(t.Elem().HasPointers()) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVaddr)
		v.Aux = ssa.SymToAux(sym)
		v0 := b.NewValue0(v.Pos, ssaop.OpSPanchored, typ.Uintptr)
		v0.AddArg2(base, mem)
		v.AddArg(v0)
		return true
	}
	// match: (LocalAddr <t> {sym} base _)
	// cond: !t.Elem().HasPointers()
	// result: (MOVaddr {sym} base)
	for {
		t := v.Type
		sym := ssa.AuxToSym(v.Aux)
		base := v_0
		if !(!t.Elem().HasPointers()) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVaddr)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg(base)
		return true
	}
	return false
}
func rewriteValue_OpLsh16x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg16, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh16x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLsh16x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg16, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh16x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLsh16x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh16x64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg16 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg16, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh16x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLsh16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg16, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh16x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLsh32x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg32, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh32x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLsh32x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg32, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh32x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLsh32x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh32x64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg32 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg32, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh32x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLsh32x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg32, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh32x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLsh64x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg64, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh64x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLsh64x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg64, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh64x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLsh64x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh64x64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg64 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg64, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh64x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLsh64x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg64, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh64x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLsh8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg8, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh8x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLsh8x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg8, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh8x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLsh8x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh8x64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg8 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg8, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh8x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLsh8x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SLL <t> x y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg8, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh8x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpMax64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Max64 x y)
	// cond: buildcfg.GORISCV64 >= 22
	// result: (MAX x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GORISCV64 >= 22) {
			break
		}
		v.Reset(ssaop.OpRISCV64MAX)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpMax64u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Max64u x y)
	// cond: buildcfg.GORISCV64 >= 22
	// result: (MAXU x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GORISCV64 >= 22) {
			break
		}
		v.Reset(ssaop.OpRISCV64MAXU)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpMin64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Min64 x y)
	// cond: buildcfg.GORISCV64 >= 22
	// result: (MIN x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GORISCV64 >= 22) {
			break
		}
		v.Reset(ssaop.OpRISCV64MIN)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpMin64u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Min64u x y)
	// cond: buildcfg.GORISCV64 >= 22
	// result: (MINU x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GORISCV64 >= 22) {
			break
		}
		v.Reset(ssaop.OpRISCV64MINU)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpMod16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16 x y [false])
	// result: (REMW (SignExt16to32 x) (SignExt16to32 y))
	for {
		if ssa.AuxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64REMW)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValue_OpMod16u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16u x y)
	// result: (REMUW (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64REMUW)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpMod32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mod32 x y [false])
	// result: (REMW x y)
	for {
		if ssa.AuxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64REMW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpMod64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mod64 x y [false])
	// result: (REM x y)
	for {
		if ssa.AuxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64REM)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpMod8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8 x y)
	// result: (REMW (SignExt8to32 x) (SignExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64REMW)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpMod8u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8u x y)
	// result: (REMUW (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64REMUW)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpMove(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Move [0] _ _ mem)
	// result: mem
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.CopyOf(mem)
		return true
	}
	// match: (Move [1] dst src mem)
	// result: (MOVBstore dst (MOVBload src mem) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVBstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBload, typ.Int8)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [2] {t} dst src mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore dst (MOVHload src mem) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		t := ssa.AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHload, typ.Int16)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [2] dst src mem)
	// result: (MOVBstore [1] dst (MOVBload [1] src mem) (MOVBstore dst (MOVBload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBload, typ.Int8)
		v0.AuxInt = ssa.Int32ToAuxInt(1)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBload, typ.Int8)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [4] {t} dst src mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore dst (MOVWload src mem) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		t := ssa.AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVWstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVWload, typ.Int32)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [4] {t} dst src mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [2] dst (MOVHload [2] src mem) (MOVHstore dst (MOVHload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		t := ssa.AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHload, typ.Int16)
		v0.AuxInt = ssa.Int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHload, typ.Int16)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [4] dst src mem)
	// result: (MOVBstore [3] dst (MOVBload [3] src mem) (MOVBstore [2] dst (MOVBload [2] src mem) (MOVBstore [1] dst (MOVBload [1] src mem) (MOVBstore dst (MOVBload src mem) mem))))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBload, typ.Int8)
		v0.AuxInt = ssa.Int32ToAuxInt(3)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(2)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBload, typ.Int8)
		v2.AuxInt = ssa.Int32ToAuxInt(2)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBstore, types.TypeMem)
		v3.AuxInt = ssa.Int32ToAuxInt(1)
		v4 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBload, typ.Int8)
		v4.AuxInt = ssa.Int32ToAuxInt(1)
		v4.AddArg2(src, mem)
		v5 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBstore, types.TypeMem)
		v6 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBload, typ.Int8)
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
		if ssa.AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		t := ssa.AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%8 == 0) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDload, typ.Int64)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [8] {t} dst src mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore [4] dst (MOVWload [4] src mem) (MOVWstore dst (MOVWload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		t := ssa.AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVWload, typ.Int32)
		v0.AuxInt = ssa.Int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVWload, typ.Int32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [8] {t} dst src mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [6] dst (MOVHload [6] src mem) (MOVHstore [4] dst (MOVHload [4] src mem) (MOVHstore [2] dst (MOVHload [2] src mem) (MOVHstore dst (MOVHload src mem) mem))))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		t := ssa.AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHload, typ.Int16)
		v0.AuxInt = ssa.Int32ToAuxInt(6)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(4)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHload, typ.Int16)
		v2.AuxInt = ssa.Int32ToAuxInt(4)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHstore, types.TypeMem)
		v3.AuxInt = ssa.Int32ToAuxInt(2)
		v4 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHload, typ.Int16)
		v4.AuxInt = ssa.Int32ToAuxInt(2)
		v4.AddArg2(src, mem)
		v5 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHstore, types.TypeMem)
		v6 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHload, typ.Int16)
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
		if ssa.AuxIntToInt64(v.AuxInt) != 3 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBload, typ.Int8)
		v0.AuxInt = ssa.Int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBload, typ.Int8)
		v2.AuxInt = ssa.Int32ToAuxInt(1)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBstore, types.TypeMem)
		v4 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBload, typ.Int8)
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
		if ssa.AuxIntToInt64(v.AuxInt) != 6 {
			break
		}
		t := ssa.AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHload, typ.Int16)
		v0.AuxInt = ssa.Int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(2)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHload, typ.Int16)
		v2.AuxInt = ssa.Int32ToAuxInt(2)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHstore, types.TypeMem)
		v4 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHload, typ.Int16)
		v4.AddArg2(src, mem)
		v3.AddArg3(dst, v4, mem)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] {t} dst src mem)
	// cond: s > 0 && s <= 3*8*ssa.MoveSize(t.Alignment(), config) && ssa.LogLargeCopyValue(v, s)
	// result: (LoweredMove [ssa.MakeValAndOff(int32(s),int32(t.Alignment()))] dst src mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		t := ssa.AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 0 && s <= 3*8*ssa.MoveSize(t.Alignment(), config) && ssa.LogLargeCopyValue(v, s)) {
			break
		}
		v.Reset(ssaop.OpRISCV64LoweredMove)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(int32(s), int32(t.Alignment())))
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (Move [s] {t} dst src mem)
	// cond: s > 3*8*ssa.MoveSize(t.Alignment(), config) && ssa.LogLargeCopyValue(v, s)
	// result: (LoweredMoveLoop [ssa.MakeValAndOff(int32(s),int32(t.Alignment()))] dst src mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		t := ssa.AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 3*8*ssa.MoveSize(t.Alignment(), config) && ssa.LogLargeCopyValue(v, s)) {
			break
		}
		v.Reset(ssaop.OpRISCV64LoweredMoveLoop)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(int32(s), int32(t.Alignment())))
		v.AddArg3(dst, src, mem)
		return true
	}
	return false
}
func rewriteValue_OpNeq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq16 x y)
	// result: (Not (Eq16 x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpNot)
		v0 := b.NewValue0(v.Pos, ssaop.OpEq16, typ.Bool)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpNeq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq32 x y)
	// result: (Not (Eq32 x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpNot)
		v0 := b.NewValue0(v.Pos, ssaop.OpEq32, typ.Bool)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpNeq64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq64 x y)
	// result: (Not (Eq64 x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpNot)
		v0 := b.NewValue0(v.Pos, ssaop.OpEq64, typ.Bool)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpNeq8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq8 x y)
	// result: (Not (Eq8 x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpNot)
		v0 := b.NewValue0(v.Pos, ssaop.OpEq8, typ.Bool)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpNeqB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NeqB x y)
	// result: (SNEZ (SUB <typ.Bool> x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SUB, typ.Bool)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpNeqPtr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NeqPtr x y)
	// result: (Not (EqPtr x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpNot)
		v0 := b.NewValue0(v.Pos, ssaop.OpEqPtr, typ.Bool)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpOffPtr(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (OffPtr [off] ptr:(SP))
	// cond: ssa.Is32Bit(off)
	// result: (MOVaddr [int32(off)] ptr)
	for {
		off := ssa.AuxIntToInt64(v.AuxInt)
		ptr := v_0
		if ptr.Op != ssaop.OpSP || !(ssa.Is32Bit(off)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVaddr)
		v.AuxInt = ssa.Int32ToAuxInt(int32(off))
		v.AddArg(ptr)
		return true
	}
	// match: (OffPtr [off] ptr)
	// cond: ssa.Is32Bit(off)
	// result: (ADDI [off] ptr)
	for {
		off := ssa.AuxIntToInt64(v.AuxInt)
		ptr := v_0
		if !(ssa.Is32Bit(off)) {
			break
		}
		v.Reset(ssaop.OpRISCV64ADDI)
		v.AuxInt = ssa.Int64ToAuxInt(off)
		v.AddArg(ptr)
		return true
	}
	// match: (OffPtr [off] ptr)
	// result: (ADD (MOVDconst [off]) ptr)
	for {
		off := ssa.AuxIntToInt64(v.AuxInt)
		ptr := v_0
		v.Reset(ssaop.OpRISCV64ADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(off)
		v.AddArg2(v0, ptr)
		return true
	}
}
func rewriteValue_OpPopCount16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount16 x)
	// result: (CPOP (ZeroExt16to64 x))
	for {
		x := v_0
		v.Reset(ssaop.OpRISCV64CPOP)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpPopCount8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount8 x)
	// result: (CPOP (ZeroExt8to64 x))
	for {
		x := v_0
		v.Reset(ssaop.OpRISCV64CPOP)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpRISCV64ADD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADD (MOVDconst <t> [val]) x)
	// cond: ssa.Is32Bit(val) && !t.IsPtr()
	// result: (ADDI [val] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64MOVDconst {
				continue
			}
			t := v_0.Type
			val := ssa.AuxIntToInt64(v_0.AuxInt)
			x := v_1
			if !(ssa.Is32Bit(val) && !t.IsPtr()) {
				continue
			}
			v.Reset(ssaop.OpRISCV64ADDI)
			v.AuxInt = ssa.Int64ToAuxInt(val)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ADD x (NEG y))
	// result: (SUB x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpRISCV64NEG {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpRISCV64SUB)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADD (SLLI [1] x) y)
	// cond: buildcfg.GORISCV64 >= 22
	// result: (SH1ADD x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64SLLI || ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			if !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.Reset(ssaop.OpRISCV64SH1ADD)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADD (SLLI [2] x) y)
	// cond: buildcfg.GORISCV64 >= 22
	// result: (SH2ADD x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64SLLI || ssa.AuxIntToInt64(v_0.AuxInt) != 2 {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			if !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.Reset(ssaop.OpRISCV64SH2ADD)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADD (SLLI [3] x) y)
	// cond: buildcfg.GORISCV64 >= 22
	// result: (SH3ADD x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64SLLI || ssa.AuxIntToInt64(v_0.AuxInt) != 3 {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			if !(buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.Reset(ssaop.OpRISCV64SH3ADD)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpRISCV64ADDI(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ADDI [c] (MOVaddr [d] {s} x))
	// cond: ssa.Is32Bit(c+int64(d))
	// result: (MOVaddr [int32(c)+d] {s} x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		s := ssa.AuxToSym(v_0.Aux)
		x := v_0.Args[0]
		if !(ssa.Is32Bit(c + int64(d))) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVaddr)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c) + d)
		v.Aux = ssa.SymToAux(s)
		v.AddArg(x)
		return true
	}
	// match: (ADDI [0] x)
	// result: x
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (ADDI [x] (MOVDconst [y]))
	// cond: ssa.Is32Bit(x + y)
	// result: (MOVDconst [x + y])
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		y := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(ssa.Is32Bit(x + y)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(x + y)
		return true
	}
	// match: (ADDI [x] (ADDI [y] z))
	// cond: ssa.Is32Bit(x + y)
	// result: (ADDI [x + y] z)
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		y := ssa.AuxIntToInt64(v_0.AuxInt)
		z := v_0.Args[0]
		if !(ssa.Is32Bit(x + y)) {
			break
		}
		v.Reset(ssaop.OpRISCV64ADDI)
		v.AuxInt = ssa.Int64ToAuxInt(x + y)
		v.AddArg(z)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64AND(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AND (MOVDconst [val]) x)
	// cond: ssa.Is32Bit(val)
	// result: (ANDI [val] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64MOVDconst {
				continue
			}
			val := ssa.AuxIntToInt64(v_0.AuxInt)
			x := v_1
			if !(ssa.Is32Bit(val)) {
				continue
			}
			v.Reset(ssaop.OpRISCV64ANDI)
			v.AuxInt = ssa.Int64ToAuxInt(val)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (AND x x)
	// result: x
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64ANDI(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ANDI [0] x)
	// result: (MOVDconst [0])
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (ANDI [-1] x)
	// result: x
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != -1 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (ANDI [x] (MOVDconst [y]))
	// result: (MOVDconst [x & y])
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		y := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(x & y)
		return true
	}
	// match: (ANDI [x] (ANDI [y] z))
	// result: (ANDI [x & y] z)
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64ANDI {
			break
		}
		y := ssa.AuxIntToInt64(v_0.AuxInt)
		z := v_0.Args[0]
		v.Reset(ssaop.OpRISCV64ANDI)
		v.AuxInt = ssa.Int64ToAuxInt(x & y)
		v.AddArg(z)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64CZEROEQZ(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CZEROEQZ x (SNEZ y))
	// result: (CZEROEQZ x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64SNEZ {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpRISCV64CZEROEQZ)
		v.AddArg2(x, y)
		return true
	}
	// match: (CZEROEQZ x (SEQZ y))
	// result: (CZERONEZ x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64SEQZ {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpRISCV64CZERONEZ)
		v.AddArg2(x, y)
		return true
	}
	// match: (CZEROEQZ x (NEG y))
	// result: (CZEROEQZ x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64NEG {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpRISCV64CZEROEQZ)
		v.AddArg2(x, y)
		return true
	}
	// match: (CZEROEQZ x x)
	// result: x
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (CZEROEQZ (MOVDconst [0]) _)
	// result: (MOVDconst [0])
	for {
		if v_0.Op != ssaop.OpRISCV64MOVDconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64CZERONEZ(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CZERONEZ x (SNEZ y))
	// result: (CZERONEZ x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64SNEZ {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpRISCV64CZERONEZ)
		v.AddArg2(x, y)
		return true
	}
	// match: (CZERONEZ x (SEQZ y))
	// result: (CZEROEQZ x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64SEQZ {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpRISCV64CZEROEQZ)
		v.AddArg2(x, y)
		return true
	}
	// match: (CZERONEZ x (NEG y))
	// result: (CZERONEZ x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64NEG {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpRISCV64CZERONEZ)
		v.AddArg2(x, y)
		return true
	}
	// match: (CZERONEZ x x)
	// result: (MOVDconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (CZERONEZ (MOVDconst [0]) _)
	// result: (MOVDconst [0])
	for {
		if v_0.Op != ssaop.OpRISCV64MOVDconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64FADDD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FADDD a (FMULD x y))
	// cond: a.Block.Func.UseFMA(v)
	// result: (FMADDD x y a)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if v_1.Op != ssaop.OpRISCV64FMULD {
				continue
			}
			y := v_1.Args[1]
			x := v_1.Args[0]
			if !(a.Block.Func.UseFMA(v)) {
				continue
			}
			v.Reset(ssaop.OpRISCV64FMADDD)
			v.AddArg3(x, y, a)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpRISCV64FADDS(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FADDS a (FMULS x y))
	// cond: a.Block.Func.UseFMA(v)
	// result: (FMADDS x y a)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if v_1.Op != ssaop.OpRISCV64FMULS {
				continue
			}
			y := v_1.Args[1]
			x := v_1.Args[0]
			if !(a.Block.Func.UseFMA(v)) {
				continue
			}
			v.Reset(ssaop.OpRISCV64FMADDS)
			v.AddArg3(x, y, a)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpRISCV64FCVTSD(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (FCVTSD (FABSD (FCVTDS X)))
	// result: (FABSS X)
	for {
		if v_0.Op != ssaop.OpRISCV64FABSD {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpRISCV64FCVTDS {
			break
		}
		X := v_0_0.Args[0]
		v.Reset(ssaop.OpRISCV64FABSS)
		v.AddArg(X)
		return true
	}
	// match: (FCVTSD (FSQRTD (FCVTDS X)))
	// result: (FSQRTS X)
	for {
		if v_0.Op != ssaop.OpRISCV64FSQRTD {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpRISCV64FCVTDS {
			break
		}
		X := v_0_0.Args[0]
		v.Reset(ssaop.OpRISCV64FSQRTS)
		v.AddArg(X)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64FEQD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (FEQD x (FMOVDconst [math.Inf(-1)]))
	// result: (ANDI [0b00_0000_0001] (FCLASSD x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpRISCV64FMOVDconst || ssa.AuxIntToFloat64(v_1.AuxInt) != math.Inf(-1) {
				continue
			}
			v.Reset(ssaop.OpRISCV64ANDI)
			v.AuxInt = ssa.Int64ToAuxInt(0b00_0000_0001)
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (FEQD x (FMOVDconst [math.Inf(1)]))
	// result: (SNEZ (ANDI <typ.Int64> [0b00_1000_0000] (FCLASSD x)))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpRISCV64FMOVDconst || ssa.AuxIntToFloat64(v_1.AuxInt) != math.Inf(1) {
				continue
			}
			v.Reset(ssaop.OpRISCV64SNEZ)
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, typ.Int64)
			v0.AuxInt = ssa.Int64ToAuxInt(0b00_1000_0000)
			v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
			v1.AddArg(x)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpRISCV64FLED(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (FLED (FMOVDconst [-math.MaxFloat64]) x)
	// result: (SNEZ (ANDI <typ.Int64> [0b00_1111_1110] (FCLASSD x)))
	for {
		if v_0.Op != ssaop.OpRISCV64FMOVDconst || ssa.AuxIntToFloat64(v_0.AuxInt) != -math.MaxFloat64 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(0b00_1111_1110)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (FLED x (FMOVDconst [math.MaxFloat64]))
	// result: (SNEZ (ANDI <typ.Int64> [0b00_0111_1111] (FCLASSD x)))
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64FMOVDconst || ssa.AuxIntToFloat64(v_1.AuxInt) != math.MaxFloat64 {
			break
		}
		v.Reset(ssaop.OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(0b00_0111_1111)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (FLED (FMOVDconst [+0x1p-1022]) x)
	// result: (SNEZ (ANDI <typ.Int64> [0b00_1100_0000] (FCLASSD x)))
	for {
		if v_0.Op != ssaop.OpRISCV64FMOVDconst || ssa.AuxIntToFloat64(v_0.AuxInt) != +0x1p-1022 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(0b00_1100_0000)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (FLED x (FMOVDconst [-0x1p-1022]))
	// result: (SNEZ (ANDI <typ.Int64> [0b00_0000_0011] (FCLASSD x)))
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64FMOVDconst || ssa.AuxIntToFloat64(v_1.AuxInt) != -0x1p-1022 {
			break
		}
		v.Reset(ssaop.OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(0b00_0000_0011)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64FLTD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (FLTD x (FMOVDconst [-math.MaxFloat64]))
	// result: (ANDI [0b00_0000_0001] (FCLASSD x))
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64FMOVDconst || ssa.AuxIntToFloat64(v_1.AuxInt) != -math.MaxFloat64 {
			break
		}
		v.Reset(ssaop.OpRISCV64ANDI)
		v.AuxInt = ssa.Int64ToAuxInt(0b00_0000_0001)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (FLTD (FMOVDconst [math.MaxFloat64]) x)
	// result: (SNEZ (ANDI <typ.Int64> [0b00_1000_0000] (FCLASSD x)))
	for {
		if v_0.Op != ssaop.OpRISCV64FMOVDconst || ssa.AuxIntToFloat64(v_0.AuxInt) != math.MaxFloat64 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(0b00_1000_0000)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (FLTD x (FMOVDconst [+0x1p-1022]))
	// result: (SNEZ (ANDI <typ.Int64> [0b00_0011_1111] (FCLASSD x)))
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64FMOVDconst || ssa.AuxIntToFloat64(v_1.AuxInt) != +0x1p-1022 {
			break
		}
		v.Reset(ssaop.OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(0b00_0011_1111)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (FLTD (FMOVDconst [-0x1p-1022]) x)
	// result: (SNEZ (ANDI <typ.Int64> [0b00_1111_1100] (FCLASSD x)))
	for {
		if v_0.Op != ssaop.OpRISCV64FMOVDconst || ssa.AuxIntToFloat64(v_0.AuxInt) != -0x1p-1022 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(0b00_1111_1100)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64FMADDD(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMADDD neg:(FNEGD x) y z)
	// cond: neg.Uses == 1
	// result: (FNMSUBD x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			neg := v_0
			if neg.Op != ssaop.OpRISCV64FNEGD {
				continue
			}
			x := neg.Args[0]
			y := v_1
			z := v_2
			if !(neg.Uses == 1) {
				continue
			}
			v.Reset(ssaop.OpRISCV64FNMSUBD)
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
		if neg.Op != ssaop.OpRISCV64FNEGD {
			break
		}
		z := neg.Args[0]
		if !(neg.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMSUBD)
		v.AddArg3(x, y, z)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64FMADDS(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMADDS neg:(FNEGS x) y z)
	// cond: neg.Uses == 1
	// result: (FNMSUBS x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			neg := v_0
			if neg.Op != ssaop.OpRISCV64FNEGS {
				continue
			}
			x := neg.Args[0]
			y := v_1
			z := v_2
			if !(neg.Uses == 1) {
				continue
			}
			v.Reset(ssaop.OpRISCV64FNMSUBS)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (FMADDS x y neg:(FNEGS z))
	// cond: neg.Uses == 1
	// result: (FMSUBS x y z)
	for {
		x := v_0
		y := v_1
		neg := v_2
		if neg.Op != ssaop.OpRISCV64FNEGS {
			break
		}
		z := neg.Args[0]
		if !(neg.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMSUBS)
		v.AddArg3(x, y, z)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64FMOVDload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (FMOVDload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FMOVDload [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (FMOVDload [off1] {sym} (ADDI [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (FMOVDload [off1+int32(off2)] {sym} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (FMOVDload [off] {sym} ptr1 (MOVDstore [off] {sym} ptr2 x _))
	// cond: ssa.IsSamePtr(ptr1, ptr2)
	// result: (FMVDX x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr1 := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(ssa.IsSamePtr(ptr1, ptr2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMVDX)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64FMOVDstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (FMOVDstore [off1] {sym1} (MOVaddr [off2] {sym2} base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FMOVDstore [off1+off2] {ssa.MergeSym(sym1,sym2)} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (FMOVDstore [off1] {sym} (ADDI [off2] base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (FMOVDstore [off1+int32(off2)] {sym} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64FMOVWload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (FMOVWload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FMOVWload [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (FMOVWload [off1] {sym} (ADDI [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (FMOVWload [off1+int32(off2)] {sym} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (FMOVWload [off] {sym} ptr1 (MOVWstore [off] {sym} ptr2 x _))
	// cond: ssa.IsSamePtr(ptr1, ptr2)
	// result: (FMVSX x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr1 := v_0
		if v_1.Op != ssaop.OpRISCV64MOVWstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(ssa.IsSamePtr(ptr1, ptr2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMVSX)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64FMOVWstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (FMOVWstore [off1] {sym1} (MOVaddr [off2] {sym2} base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FMOVWstore [off1+off2] {ssa.MergeSym(sym1,sym2)} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (FMOVWstore [off1] {sym} (ADDI [off2] base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (FMOVWstore [off1+int32(off2)] {sym} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64FMSUBD(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMSUBD neg:(FNEGD x) y z)
	// cond: neg.Uses == 1
	// result: (FNMADDD x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			neg := v_0
			if neg.Op != ssaop.OpRISCV64FNEGD {
				continue
			}
			x := neg.Args[0]
			y := v_1
			z := v_2
			if !(neg.Uses == 1) {
				continue
			}
			v.Reset(ssaop.OpRISCV64FNMADDD)
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
		if neg.Op != ssaop.OpRISCV64FNEGD {
			break
		}
		z := neg.Args[0]
		if !(neg.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMADDD)
		v.AddArg3(x, y, z)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64FMSUBS(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMSUBS neg:(FNEGS x) y z)
	// cond: neg.Uses == 1
	// result: (FNMADDS x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			neg := v_0
			if neg.Op != ssaop.OpRISCV64FNEGS {
				continue
			}
			x := neg.Args[0]
			y := v_1
			z := v_2
			if !(neg.Uses == 1) {
				continue
			}
			v.Reset(ssaop.OpRISCV64FNMADDS)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (FMSUBS x y neg:(FNEGS z))
	// cond: neg.Uses == 1
	// result: (FMADDS x y z)
	for {
		x := v_0
		y := v_1
		neg := v_2
		if neg.Op != ssaop.OpRISCV64FNEGS {
			break
		}
		z := neg.Args[0]
		if !(neg.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMADDS)
		v.AddArg3(x, y, z)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64FNED(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (FNED x (FMOVDconst [math.Inf(-1)]))
	// result: (SEQZ (ANDI <typ.Int64> [0b00_0000_0001] (FCLASSD x)))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpRISCV64FMOVDconst || ssa.AuxIntToFloat64(v_1.AuxInt) != math.Inf(-1) {
				continue
			}
			v.Reset(ssaop.OpRISCV64SEQZ)
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, typ.Int64)
			v0.AuxInt = ssa.Int64ToAuxInt(0b00_0000_0001)
			v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
			v1.AddArg(x)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (FNED x (FMOVDconst [math.Inf(1)]))
	// result: (SEQZ (ANDI <typ.Int64> [0b00_1000_0000] (FCLASSD x)))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpRISCV64FMOVDconst || ssa.AuxIntToFloat64(v_1.AuxInt) != math.Inf(1) {
				continue
			}
			v.Reset(ssaop.OpRISCV64SEQZ)
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, typ.Int64)
			v0.AuxInt = ssa.Int64ToAuxInt(0b00_1000_0000)
			v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
			v1.AddArg(x)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpRISCV64FNMADDD(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FNMADDD neg:(FNEGD x) y z)
	// cond: neg.Uses == 1
	// result: (FMSUBD x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			neg := v_0
			if neg.Op != ssaop.OpRISCV64FNEGD {
				continue
			}
			x := neg.Args[0]
			y := v_1
			z := v_2
			if !(neg.Uses == 1) {
				continue
			}
			v.Reset(ssaop.OpRISCV64FMSUBD)
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
		if neg.Op != ssaop.OpRISCV64FNEGD {
			break
		}
		z := neg.Args[0]
		if !(neg.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpRISCV64FNMSUBD)
		v.AddArg3(x, y, z)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64FNMADDS(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FNMADDS neg:(FNEGS x) y z)
	// cond: neg.Uses == 1
	// result: (FMSUBS x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			neg := v_0
			if neg.Op != ssaop.OpRISCV64FNEGS {
				continue
			}
			x := neg.Args[0]
			y := v_1
			z := v_2
			if !(neg.Uses == 1) {
				continue
			}
			v.Reset(ssaop.OpRISCV64FMSUBS)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (FNMADDS x y neg:(FNEGS z))
	// cond: neg.Uses == 1
	// result: (FNMSUBS x y z)
	for {
		x := v_0
		y := v_1
		neg := v_2
		if neg.Op != ssaop.OpRISCV64FNEGS {
			break
		}
		z := neg.Args[0]
		if !(neg.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpRISCV64FNMSUBS)
		v.AddArg3(x, y, z)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64FNMSUBD(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FNMSUBD neg:(FNEGD x) y z)
	// cond: neg.Uses == 1
	// result: (FMADDD x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			neg := v_0
			if neg.Op != ssaop.OpRISCV64FNEGD {
				continue
			}
			x := neg.Args[0]
			y := v_1
			z := v_2
			if !(neg.Uses == 1) {
				continue
			}
			v.Reset(ssaop.OpRISCV64FMADDD)
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
		if neg.Op != ssaop.OpRISCV64FNEGD {
			break
		}
		z := neg.Args[0]
		if !(neg.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpRISCV64FNMADDD)
		v.AddArg3(x, y, z)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64FNMSUBS(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FNMSUBS neg:(FNEGS x) y z)
	// cond: neg.Uses == 1
	// result: (FMADDS x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			neg := v_0
			if neg.Op != ssaop.OpRISCV64FNEGS {
				continue
			}
			x := neg.Args[0]
			y := v_1
			z := v_2
			if !(neg.Uses == 1) {
				continue
			}
			v.Reset(ssaop.OpRISCV64FMADDS)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (FNMSUBS x y neg:(FNEGS z))
	// cond: neg.Uses == 1
	// result: (FNMADDS x y z)
	for {
		x := v_0
		y := v_1
		neg := v_2
		if neg.Op != ssaop.OpRISCV64FNEGS {
			break
		}
		z := neg.Args[0]
		if !(neg.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpRISCV64FNMADDS)
		v.AddArg3(x, y, z)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64FSUBD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FSUBD a (FMULD x y))
	// cond: a.Block.Func.UseFMA(v)
	// result: (FNMSUBD x y a)
	for {
		a := v_0
		if v_1.Op != ssaop.OpRISCV64FMULD {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64FNMSUBD)
		v.AddArg3(x, y, a)
		return true
	}
	// match: (FSUBD (FMULD x y) a)
	// cond: a.Block.Func.UseFMA(v)
	// result: (FMSUBD x y a)
	for {
		if v_0.Op != ssaop.OpRISCV64FMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		a := v_1
		if !(a.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMSUBD)
		v.AddArg3(x, y, a)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64FSUBS(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FSUBS a (FMULS x y))
	// cond: a.Block.Func.UseFMA(v)
	// result: (FNMSUBS x y a)
	for {
		a := v_0
		if v_1.Op != ssaop.OpRISCV64FMULS {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64FNMSUBS)
		v.AddArg3(x, y, a)
		return true
	}
	// match: (FSUBS (FMULS x y) a)
	// cond: a.Block.Func.UseFMA(v)
	// result: (FMSUBS x y a)
	for {
		if v_0.Op != ssaop.OpRISCV64FMULS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		a := v_1
		if !(a.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMSUBS)
		v.AddArg3(x, y, a)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64LoweredPanicBoundsCR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsCR [kind] {p} (MOVDconst [c]) mem)
	// result: (LoweredPanicBoundsCC [kind] {ssa.PanicBoundsCC{Cx:p.C, Cy:c}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		p := ssa.AuxToPanicBoundsC(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		mem := v_1
		v.Reset(ssaop.OpRISCV64LoweredPanicBoundsCC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCCToAux(ssa.PanicBoundsCC{Cx: p.C, Cy: c})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64LoweredPanicBoundsRC(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRC [kind] {p} (MOVDconst [c]) mem)
	// result: (LoweredPanicBoundsCC [kind] {ssa.PanicBoundsCC{Cx:c, Cy:p.C}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		p := ssa.AuxToPanicBoundsC(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		mem := v_1
		v.Reset(ssaop.OpRISCV64LoweredPanicBoundsCC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCCToAux(ssa.PanicBoundsCC{Cx: c, Cy: p.C})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64LoweredPanicBoundsRR(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRR [kind] x (MOVDconst [c]) mem)
	// result: (LoweredPanicBoundsRC [kind] x {ssa.PanicBoundsC{C:c}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.OpRISCV64LoweredPanicBoundsRC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCToAux(ssa.PanicBoundsC{C: c})
		v.AddArg2(x, mem)
		return true
	}
	// match: (LoweredPanicBoundsRR [kind] (MOVDconst [c]) y mem)
	// result: (LoweredPanicBoundsCR [kind] {ssa.PanicBoundsC{C:c}} y mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		y := v_1
		mem := v_2
		v.Reset(ssaop.OpRISCV64LoweredPanicBoundsCR)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCToAux(ssa.PanicBoundsC{C: c})
		v.AddArg2(y, mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVBUload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBUload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVBUload [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVBUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVBUload [off1] {sym} (ADDI [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (MOVBUload [off1+int32(off2)] {sym} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVBUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVBUload [off] {sym} ptr1 (MOVBstore [off] {sym} ptr2 x _))
	// cond: ssa.IsSamePtr(ptr1, ptr2)
	// result: (MOVBUreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr1 := v_0
		if v_1.Op != ssaop.OpRISCV64MOVBstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(ssa.IsSamePtr(ptr1, ptr2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVBUreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVBUreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVBUreg x:(FLES _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64FLES {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(FLTS _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64FLTS {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(FEQS _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64FEQS {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(FNES _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64FNES {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(FLED _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64FLED {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(FLTD _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64FLTD {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(FEQD _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64FEQD {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(FNED _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64FNED {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(SEQZ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64SEQZ {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(SNEZ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64SNEZ {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(SLT _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64SLT {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(SLTU _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64SLTU {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(ANDI [c] y))
	// cond: c >= 0 && int64(uint8(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64ANDI {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		if !(c >= 0 && int64(uint8(c)) == c) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg (ANDI [c] x))
	// cond: c < 0
	// result: (ANDI [int64(uint8(c))] x)
	for {
		if v_0.Op != ssaop.OpRISCV64ANDI {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c < 0) {
			break
		}
		v.Reset(ssaop.OpRISCV64ANDI)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint8(c)))
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (MOVDconst [c]))
	// result: (MOVDconst [int64(uint8(c))])
	for {
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint8(c)))
		return true
	}
	// match: (MOVBUreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVBUload {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(Select0 (LoweredAtomicLoad8 _ _)))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpSelect0 {
			break
		}
		x_0 := x.Args[0]
		if x_0.Op != ssaop.OpRISCV64LoweredAtomicLoad8 {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(Select0 (LoweredAtomicCas32 _ _ _ _)))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpSelect0 {
			break
		}
		x_0 := x.Args[0]
		if x_0.Op != ssaop.OpRISCV64LoweredAtomicCas32 {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(Select0 (LoweredAtomicCas64 _ _ _ _)))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpSelect0 {
			break
		}
		x_0 := x.Args[0]
		if x_0.Op != ssaop.OpRISCV64LoweredAtomicCas64 {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg <t> x:(MOVBload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && ssa.Clobber(x)
	// result: @x.Block (MOVBUload <t> [off] {sym} ptr mem)
	for {
		t := v.Type
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVBload {
			break
		}
		off := ssa.AuxIntToInt32(x.AuxInt)
		sym := ssa.AuxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1 && ssa.Clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, ssaop.OpRISCV64MOVBUload, t)
		v.CopyOf(v0)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVBload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVBload [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVBload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVBload [off1] {sym} (ADDI [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (MOVBload [off1+int32(off2)] {sym} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVBload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVBload [off] {sym} ptr1 (MOVBstore [off] {sym} ptr2 x _))
	// cond: ssa.IsSamePtr(ptr1, ptr2)
	// result: (MOVBreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr1 := v_0
		if v_1.Op != ssaop.OpRISCV64MOVBstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(ssa.IsSamePtr(ptr1, ptr2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVBreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVBreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVBreg x:(ANDI [c] y))
	// cond: c >= 0 && int64(int8(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64ANDI {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		if !(c >= 0 && int64(int8(c)) == c) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBreg (MOVDconst [c]))
	// result: (MOVDconst [int64(int8(c))])
	for {
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int8(c)))
		return true
	}
	// match: (MOVBreg x:(MOVBload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVBload {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVBreg {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg <t> x:(MOVBUload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && ssa.Clobber(x)
	// result: @x.Block (MOVBload <t> [off] {sym} ptr mem)
	for {
		t := v.Type
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVBUload {
			break
		}
		off := ssa.AuxIntToInt32(x.AuxInt)
		sym := ssa.AuxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1 && ssa.Clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, ssaop.OpRISCV64MOVBload, t)
		v.CopyOf(v0)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVBstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBstore [off1] {sym1} (MOVaddr [off2] {sym2} base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVBstore [off1+off2] {ssa.MergeSym(sym1,sym2)} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVBstore [off1] {sym} (ADDI [off2] base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (MOVBstore [off1+int32(off2)] {sym} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVDconst [0]) mem)
	// result: (MOVBstorezero [off] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVBstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpRISCV64MOVBreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVHreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpRISCV64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVWreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpRISCV64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBUreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpRISCV64MOVBUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVHUreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpRISCV64MOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVWUreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpRISCV64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVBstorezero(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBstorezero [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVBstorezero [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVBstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVBstorezero [off1] {sym} (ADDI [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (MOVBstorezero [off1+int32(off2)] {sym} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVBstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVDload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVDload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVDload [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVDload [off1] {sym} (ADDI [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (MOVDload [off1+int32(off2)] {sym} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVDload [off] {sym} ptr1 (MOVDstore [off] {sym} ptr2 x _))
	// cond: ssa.IsSamePtr(ptr1, ptr2)
	// result: (MOVDreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr1 := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(ssa.IsSamePtr(ptr1, ptr2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVDload [off] {sym} ptr1 (FMOVDstore [off] {sym} ptr2 x _))
	// cond: ssa.IsSamePtr(ptr1, ptr2)
	// result: (FMVXD x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr1 := v_0
		if v_1.Op != ssaop.OpRISCV64FMOVDstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(ssa.IsSamePtr(ptr1, ptr2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMVXD)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVDnop(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVDnop (MOVDconst [c]))
	// result: (MOVDconst [c])
	for {
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVDreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVDreg x)
	// cond: x.Uses == 1
	// result: (MOVDnop x)
	for {
		x := v_0
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDnop)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVDstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVDstore [off1] {sym1} (MOVaddr [off2] {sym2} base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVDstore [off1+off2] {ssa.MergeSym(sym1,sym2)} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym} (ADDI [off2] base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (MOVDstore [off1+int32(off2)] {sym} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVDstore [off] {sym} ptr (MOVDconst [0]) mem)
	// result: (MOVDstorezero [off] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVDstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVDstorezero(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVDstorezero [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVDstorezero [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVDstorezero [off1] {sym} (ADDI [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (MOVDstorezero [off1+int32(off2)] {sym} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVHUload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHUload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVHUload [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVHUload [off1] {sym} (ADDI [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (MOVHUload [off1+int32(off2)] {sym} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVHUload [off] {sym} ptr1 (MOVHstore [off] {sym} ptr2 x _))
	// cond: ssa.IsSamePtr(ptr1, ptr2)
	// result: (MOVHUreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr1 := v_0
		if v_1.Op != ssaop.OpRISCV64MOVHstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(ssa.IsSamePtr(ptr1, ptr2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHUreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVHUreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVHUreg x:(ANDI [c] y))
	// cond: c >= 0 && int64(uint16(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64ANDI {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		if !(c >= 0 && int64(uint16(c)) == c) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVHUreg (ANDI [c] x))
	// cond: c < 0
	// result: (ANDI [int64(uint16(c))] x)
	for {
		if v_0.Op != ssaop.OpRISCV64ANDI {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c < 0) {
			break
		}
		v.Reset(ssaop.OpRISCV64ANDI)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint16(c)))
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (MOVDconst [c]))
	// result: (MOVDconst [int64(uint16(c))])
	for {
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint16(c)))
		return true
	}
	// match: (MOVHUreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVBUload {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVHUload {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVHUreg {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg <t> x:(MOVHload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && ssa.Clobber(x)
	// result: @x.Block (MOVHUload <t> [off] {sym} ptr mem)
	for {
		t := v.Type
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVHload {
			break
		}
		off := ssa.AuxIntToInt32(x.AuxInt)
		sym := ssa.AuxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1 && ssa.Clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, ssaop.OpRISCV64MOVHUload, t)
		v.CopyOf(v0)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVHload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVHload [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVHload [off1] {sym} (ADDI [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (MOVHload [off1+int32(off2)] {sym} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVHload [off] {sym} ptr1 (MOVHstore [off] {sym} ptr2 x _))
	// cond: ssa.IsSamePtr(ptr1, ptr2)
	// result: (MOVHreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr1 := v_0
		if v_1.Op != ssaop.OpRISCV64MOVHstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(ssa.IsSamePtr(ptr1, ptr2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVHreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVHreg x:(ANDI [c] y))
	// cond: c >= 0 && int64(int16(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64ANDI {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		if !(c >= 0 && int64(int16(c)) == c) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVHreg (MOVDconst [c]))
	// result: (MOVDconst [int64(int16(c))])
	for {
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int16(c)))
		return true
	}
	// match: (MOVHreg x:(MOVBload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVBload {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVBUload {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVHload {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVBreg {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVHreg {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg <t> x:(MOVHUload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && ssa.Clobber(x)
	// result: @x.Block (MOVHload <t> [off] {sym} ptr mem)
	for {
		t := v.Type
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVHUload {
			break
		}
		off := ssa.AuxIntToInt32(x.AuxInt)
		sym := ssa.AuxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1 && ssa.Clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, ssaop.OpRISCV64MOVHload, t)
		v.CopyOf(v0)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVHstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHstore [off1] {sym1} (MOVaddr [off2] {sym2} base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVHstore [off1+off2] {ssa.MergeSym(sym1,sym2)} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVHstore [off1] {sym} (ADDI [off2] base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (MOVHstore [off1+int32(off2)] {sym} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVDconst [0]) mem)
	// result: (MOVHstorezero [off] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVHstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVHreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpRISCV64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVWreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpRISCV64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVHUreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpRISCV64MOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVWUreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpRISCV64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVHstorezero(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHstorezero [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVHstorezero [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVHstorezero [off1] {sym} (ADDI [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (MOVHstorezero [off1+int32(off2)] {sym} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVWUload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVWUload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVWUload [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVWUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVWUload [off1] {sym} (ADDI [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (MOVWUload [off1+int32(off2)] {sym} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVWUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVWUload [off] {sym} ptr1 (MOVWstore [off] {sym} ptr2 x _))
	// cond: ssa.IsSamePtr(ptr1, ptr2)
	// result: (MOVWUreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr1 := v_0
		if v_1.Op != ssaop.OpRISCV64MOVWstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(ssa.IsSamePtr(ptr1, ptr2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVWUreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUload [off] {sym} ptr1 (FMOVWstore [off] {sym} ptr2 x _))
	// cond: ssa.IsSamePtr(ptr1, ptr2)
	// result: (MOVWUreg (FMVXS x))
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr1 := v_0
		if v_1.Op != ssaop.OpRISCV64FMOVWstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(ssa.IsSamePtr(ptr1, ptr2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVWUreg)
		v0 := b.NewValue0(v_1.Pos, ssaop.OpRISCV64FMVXS, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVWUreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVWUreg x:(ANDI [c] y))
	// cond: c >= 0 && int64(uint32(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64ANDI {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		if !(c >= 0 && int64(uint32(c)) == c) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWUreg (ANDI [c] x))
	// cond: c < 0
	// result: (AND (MOVDconst [int64(uint32(c))]) x)
	for {
		if v_0.Op != ssaop.OpRISCV64ANDI {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c < 0) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(int64(uint32(c)))
		v.AddArg2(v0, x)
		return true
	}
	// match: (MOVWUreg (MOVDconst [c]))
	// result: (MOVDconst [int64(uint32(c))])
	for {
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint32(c)))
		return true
	}
	// match: (MOVWUreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVBUload {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVHUload {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVWUload {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVHUreg {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVWUreg {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg <t> x:(MOVWload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && ssa.Clobber(x)
	// result: @x.Block (MOVWUload <t> [off] {sym} ptr mem)
	for {
		t := v.Type
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVWload {
			break
		}
		off := ssa.AuxIntToInt32(x.AuxInt)
		sym := ssa.AuxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1 && ssa.Clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, ssaop.OpRISCV64MOVWUload, t)
		v.CopyOf(v0)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVWload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVWload [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVWload [off1] {sym} (ADDI [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (MOVWload [off1+int32(off2)] {sym} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVWload [off] {sym} ptr1 (MOVWstore [off] {sym} ptr2 x _))
	// cond: ssa.IsSamePtr(ptr1, ptr2)
	// result: (MOVWreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr1 := v_0
		if v_1.Op != ssaop.OpRISCV64MOVWstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(ssa.IsSamePtr(ptr1, ptr2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWload [off] {sym} ptr1 (FMOVWstore [off] {sym} ptr2 x _))
	// cond: ssa.IsSamePtr(ptr1, ptr2)
	// result: (FMVXS x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr1 := v_0
		if v_1.Op != ssaop.OpRISCV64FMOVWstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(ssa.IsSamePtr(ptr1, ptr2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMVXS)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVWreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVWreg x:(ANDI [c] y))
	// cond: c >= 0 && int64(int32(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64ANDI {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		if !(c >= 0 && int64(int32(c)) == c) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWreg (NEG x))
	// result: (NEGW x)
	for {
		if v_0.Op != ssaop.OpRISCV64NEG {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpRISCV64NEGW)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg (MOVDconst [c]))
	// result: (MOVDconst [int64(int32(c))])
	for {
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int32(c)))
		return true
	}
	// match: (MOVWreg x:(MOVBload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVBload {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVBUload {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVHload {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVHUload {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVWload {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(ADDIW _))
	// cond: (x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned()))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64ADDIW || !(x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned())) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(SUBW _ _))
	// cond: (x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned()))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64SUBW || !(x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned())) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(NEGW _))
	// cond: (x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned()))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64NEGW || !(x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned())) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MULW _ _))
	// cond: (x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned()))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MULW || !(x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned())) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(DIVW _ _))
	// cond: (x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned()))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64DIVW || !(x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned())) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(DIVUW _ _))
	// cond: (x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned()))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64DIVUW || !(x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned())) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(REMW _ _))
	// cond: (x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned()))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64REMW || !(x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned())) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(REMUW _ _))
	// cond: (x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned()))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64REMUW || !(x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned())) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(ROLW _ _))
	// cond: (x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned()))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64ROLW || !(x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned())) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(RORW _ _))
	// cond: (x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned()))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64RORW || !(x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned())) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(RORIW _))
	// cond: (x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned()))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64RORIW || !(x.Type.Size() == 8 || (x.Type.Size() == 4 && x.Type.IsSigned())) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVBreg {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVHreg {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVWreg {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg <t> x:(MOVWUload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && ssa.Clobber(x)
	// result: @x.Block (MOVWload <t> [off] {sym} ptr mem)
	for {
		t := v.Type
		x := v_0
		if x.Op != ssaop.OpRISCV64MOVWUload {
			break
		}
		off := ssa.AuxIntToInt32(x.AuxInt)
		sym := ssa.AuxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1 && ssa.Clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, ssaop.OpRISCV64MOVWload, t)
		v.CopyOf(v0)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVWstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWstore [off1] {sym1} (MOVaddr [off2] {sym2} base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVWstore [off1+off2] {ssa.MergeSym(sym1,sym2)} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym} (ADDI [off2] base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (MOVWstore [off1+int32(off2)] {sym} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVDconst [0]) mem)
	// result: (MOVWstorezero [off] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVWstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWreg x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpRISCV64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWUreg x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpRISCV64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpRISCV64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MOVWstorezero(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWstorezero [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVWstorezero [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64MOVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVWstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVWstorezero [off1] {sym} (ADDI [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2)
	// result: (MOVWstorezero [off1+int32(off2)] {sym} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpRISCV64ADDI {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1) + off2)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVWstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64MUL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MUL x (MOVDconst [c]))
	// cond: c%3 == 0 && ssa.IsPowerOfTwo(c/3) && buildcfg.GORISCV64 >= 22
	// result: (SLLI [ssa.Log64(c/3)] (SH1ADD <x.Type> x x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpRISCV64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(c%3 == 0 && ssa.IsPowerOfTwo(c/3) && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.Reset(ssaop.OpRISCV64SLLI)
			v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 3))
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SH1ADD, x.Type)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MUL x (MOVDconst [c]))
	// cond: c%5 == 0 && ssa.IsPowerOfTwo(c/5) && buildcfg.GORISCV64 >= 22
	// result: (SLLI [ssa.Log64(c/5)] (SH2ADD <x.Type> x x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpRISCV64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(c%5 == 0 && ssa.IsPowerOfTwo(c/5) && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.Reset(ssaop.OpRISCV64SLLI)
			v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 5))
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SH2ADD, x.Type)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MUL x (MOVDconst [c]))
	// cond: c%9 == 0 && ssa.IsPowerOfTwo(c/9) && buildcfg.GORISCV64 >= 22
	// result: (SLLI [ssa.Log64(c/9)] (SH3ADD <x.Type> x x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpRISCV64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(c%9 == 0 && ssa.IsPowerOfTwo(c/9) && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.Reset(ssaop.OpRISCV64SLLI)
			v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 9))
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SH3ADD, x.Type)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpRISCV64MULW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MULW x (MOVDconst [c]))
	// cond: c%3 == 0 && ssa.IsPowerOfTwo(c/3) && buildcfg.GORISCV64 >= 22
	// result: (SLLIW [ssa.Log64(c/3)] (SH1ADD <x.Type> x x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpRISCV64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(c%3 == 0 && ssa.IsPowerOfTwo(c/3) && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.Reset(ssaop.OpRISCV64SLLIW)
			v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 3))
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SH1ADD, x.Type)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: c%5 == 0 && ssa.IsPowerOfTwo(c/5) && buildcfg.GORISCV64 >= 22
	// result: (SLLIW [ssa.Log64(c/5)] (SH2ADD <x.Type> x x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpRISCV64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(c%5 == 0 && ssa.IsPowerOfTwo(c/5) && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.Reset(ssaop.OpRISCV64SLLIW)
			v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 5))
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SH2ADD, x.Type)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: c%9 == 0 && ssa.IsPowerOfTwo(c/9) && buildcfg.GORISCV64 >= 22
	// result: (SLLIW [ssa.Log64(c/9)] (SH3ADD <x.Type> x x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpRISCV64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(c%9 == 0 && ssa.IsPowerOfTwo(c/9) && buildcfg.GORISCV64 >= 22) {
				continue
			}
			v.Reset(ssaop.OpRISCV64SLLIW)
			v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 9))
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SH3ADD, x.Type)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpRISCV64NEG(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (NEG (SUB x y))
	// result: (SUB y x)
	for {
		if v_0.Op != ssaop.OpRISCV64SUB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpRISCV64SUB)
		v.AddArg2(y, x)
		return true
	}
	// match: (NEG <t> s:(ADDI [val] (SUB x y)))
	// cond: s.Uses == 1 && ssa.Is32Bit(-val)
	// result: (ADDI [-val] (SUB <t> y x))
	for {
		t := v.Type
		s := v_0
		if s.Op != ssaop.OpRISCV64ADDI {
			break
		}
		val := ssa.AuxIntToInt64(s.AuxInt)
		s_0 := s.Args[0]
		if s_0.Op != ssaop.OpRISCV64SUB {
			break
		}
		y := s_0.Args[1]
		x := s_0.Args[0]
		if !(s.Uses == 1 && ssa.Is32Bit(-val)) {
			break
		}
		v.Reset(ssaop.OpRISCV64ADDI)
		v.AuxInt = ssa.Int64ToAuxInt(-val)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SUB, t)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	// match: (NEG (NEG x))
	// result: x
	for {
		if v_0.Op != ssaop.OpRISCV64NEG {
			break
		}
		x := v_0.Args[0]
		v.CopyOf(x)
		return true
	}
	// match: (NEG <t> s:(ADDI [val] (NEG x)))
	// cond: s.Uses == 1 && ssa.Is32Bit(-val)
	// result: (ADDI [-val] x)
	for {
		s := v_0
		if s.Op != ssaop.OpRISCV64ADDI {
			break
		}
		val := ssa.AuxIntToInt64(s.AuxInt)
		s_0 := s.Args[0]
		if s_0.Op != ssaop.OpRISCV64NEG {
			break
		}
		x := s_0.Args[0]
		if !(s.Uses == 1 && ssa.Is32Bit(-val)) {
			break
		}
		v.Reset(ssaop.OpRISCV64ADDI)
		v.AuxInt = ssa.Int64ToAuxInt(-val)
		v.AddArg(x)
		return true
	}
	// match: (NEG (MOVDconst [x]))
	// result: (MOVDconst [-x])
	for {
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64NEGW(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (NEGW (MOVDconst [x]))
	// result: (MOVDconst [int64(int32(-x))])
	for {
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int32(-x)))
		return true
	}
	return false
}
func rewriteValue_OpRISCV64OR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (OR (MOVDconst [val]) x)
	// cond: ssa.Is32Bit(val)
	// result: (ORI [val] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64MOVDconst {
				continue
			}
			val := ssa.AuxIntToInt64(v_0.AuxInt)
			x := v_1
			if !(ssa.Is32Bit(val)) {
				continue
			}
			v.Reset(ssaop.OpRISCV64ORI)
			v.AuxInt = ssa.Int64ToAuxInt(val)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (OR x x)
	// result: x
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (OR (CZEROEQZ <t> x cond) (CZERONEZ <t> (ADD x y) cond))
	// result: (ADD x (CZERONEZ <t> y cond))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			cond := v_0.Args[1]
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != ssaop.OpRISCV64ADD {
				continue
			}
			_ = v_1_0.Args[1]
			v_1_0_0 := v_1_0.Args[0]
			v_1_0_1 := v_1_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0_0, v_1_0_1 = _i1+1, v_1_0_1, v_1_0_0 {
				if x != v_1_0_0 {
					continue
				}
				y := v_1_0_1
				if cond != v_1.Args[1] {
					continue
				}
				v.Reset(ssaop.OpRISCV64ADD)
				v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64CZERONEZ, t)
				v0.AddArg2(y, cond)
				v.AddArg2(x, v0)
				return true
			}
		}
		break
	}
	// match: (OR (CZEROEQZ <t> x cond) (CZERONEZ <t> (SUB x y) cond))
	// result: (SUB x (CZERONEZ <t> y cond))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			cond := v_0.Args[1]
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != ssaop.OpRISCV64SUB {
				continue
			}
			y := v_1_0.Args[1]
			if x != v_1_0.Args[0] || cond != v_1.Args[1] {
				continue
			}
			v.Reset(ssaop.OpRISCV64SUB)
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64CZERONEZ, t)
			v0.AddArg2(y, cond)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZEROEQZ <t> x cond) (CZERONEZ <t> (OR x y) cond))
	// result: (OR x (CZERONEZ <t> y cond))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			cond := v_0.Args[1]
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != ssaop.OpRISCV64OR {
				continue
			}
			_ = v_1_0.Args[1]
			v_1_0_0 := v_1_0.Args[0]
			v_1_0_1 := v_1_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0_0, v_1_0_1 = _i1+1, v_1_0_1, v_1_0_0 {
				if x != v_1_0_0 {
					continue
				}
				y := v_1_0_1
				if cond != v_1.Args[1] {
					continue
				}
				v.Reset(ssaop.OpRISCV64OR)
				v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64CZERONEZ, t)
				v0.AddArg2(y, cond)
				v.AddArg2(x, v0)
				return true
			}
		}
		break
	}
	// match: (OR (CZEROEQZ <t> x cond) (CZERONEZ <t> (XOR x y) cond))
	// result: (XOR x (CZERONEZ <t> y cond))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			cond := v_0.Args[1]
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != ssaop.OpRISCV64XOR {
				continue
			}
			_ = v_1_0.Args[1]
			v_1_0_0 := v_1_0.Args[0]
			v_1_0_1 := v_1_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0_0, v_1_0_1 = _i1+1, v_1_0_1, v_1_0_0 {
				if x != v_1_0_0 {
					continue
				}
				y := v_1_0_1
				if cond != v_1.Args[1] {
					continue
				}
				v.Reset(ssaop.OpRISCV64XOR)
				v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64CZERONEZ, t)
				v0.AddArg2(y, cond)
				v.AddArg2(x, v0)
				return true
			}
		}
		break
	}
	// match: (OR (CZEROEQZ <t> x cond) (CZERONEZ <t> (SUBW x y) cond))
	// result: (SUBW x (CZERONEZ <t> y cond))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			cond := v_0.Args[1]
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != ssaop.OpRISCV64SUBW {
				continue
			}
			y := v_1_0.Args[1]
			if x != v_1_0.Args[0] || cond != v_1.Args[1] {
				continue
			}
			v.Reset(ssaop.OpRISCV64SUBW)
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64CZERONEZ, t)
			v0.AddArg2(y, cond)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (ADD x y) cond) (CZERONEZ <t> x cond))
	// result: (ADD x (CZEROEQZ <t> y cond))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			cond := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpRISCV64ADD {
				continue
			}
			_ = v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0_0, v_0_0_1 = _i1+1, v_0_0_1, v_0_0_0 {
				x := v_0_0_0
				y := v_0_0_1
				if v_1.Op != ssaop.OpRISCV64CZERONEZ || v_1.Type != t {
					continue
				}
				_ = v_1.Args[1]
				if x != v_1.Args[0] || cond != v_1.Args[1] {
					continue
				}
				v.Reset(ssaop.OpRISCV64ADD)
				v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64CZEROEQZ, t)
				v0.AddArg2(y, cond)
				v.AddArg2(x, v0)
				return true
			}
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (SUB x y) cond) (CZERONEZ <t> x cond))
	// result: (SUB x (CZEROEQZ <t> y cond))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			cond := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpRISCV64SUB {
				continue
			}
			y := v_0_0.Args[1]
			x := v_0_0.Args[0]
			if v_1.Op != ssaop.OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || cond != v_1.Args[1] {
				continue
			}
			v.Reset(ssaop.OpRISCV64SUB)
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64CZEROEQZ, t)
			v0.AddArg2(y, cond)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (OR x y) cond) (CZERONEZ <t> x cond))
	// result: (OR x (CZEROEQZ <t> y cond))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			cond := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpRISCV64OR {
				continue
			}
			_ = v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0_0, v_0_0_1 = _i1+1, v_0_0_1, v_0_0_0 {
				x := v_0_0_0
				y := v_0_0_1
				if v_1.Op != ssaop.OpRISCV64CZERONEZ || v_1.Type != t {
					continue
				}
				_ = v_1.Args[1]
				if x != v_1.Args[0] || cond != v_1.Args[1] {
					continue
				}
				v.Reset(ssaop.OpRISCV64OR)
				v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64CZEROEQZ, t)
				v0.AddArg2(y, cond)
				v.AddArg2(x, v0)
				return true
			}
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (XOR x y) cond) (CZERONEZ <t> x cond))
	// result: (XOR x (CZEROEQZ <t> y cond))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			cond := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpRISCV64XOR {
				continue
			}
			_ = v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0_0, v_0_0_1 = _i1+1, v_0_0_1, v_0_0_0 {
				x := v_0_0_0
				y := v_0_0_1
				if v_1.Op != ssaop.OpRISCV64CZERONEZ || v_1.Type != t {
					continue
				}
				_ = v_1.Args[1]
				if x != v_1.Args[0] || cond != v_1.Args[1] {
					continue
				}
				v.Reset(ssaop.OpRISCV64XOR)
				v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64CZEROEQZ, t)
				v0.AddArg2(y, cond)
				v.AddArg2(x, v0)
				return true
			}
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (SUBW x y) cond) (CZERONEZ <t> x cond))
	// result: (SUBW x (CZEROEQZ <t> y cond))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			cond := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpRISCV64SUBW {
				continue
			}
			y := v_0_0.Args[1]
			x := v_0_0.Args[0]
			if v_1.Op != ssaop.OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || cond != v_1.Args[1] {
				continue
			}
			v.Reset(ssaop.OpRISCV64SUBW)
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64CZEROEQZ, t)
			v0.AddArg2(y, cond)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR x:(CZEROEQZ z cond) (CZERONEZ y:(AND z _) cond))
	// result: (OR y x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if x.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			cond := x.Args[1]
			z := x.Args[0]
			if v_1.Op != ssaop.OpRISCV64CZERONEZ {
				continue
			}
			_ = v_1.Args[1]
			y := v_1.Args[0]
			if y.Op != ssaop.OpRISCV64AND {
				continue
			}
			y_0 := y.Args[0]
			y_1 := y.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, y_0, y_1 = _i1+1, y_1, y_0 {
				if z != y_0 || cond != v_1.Args[1] {
					continue
				}
				v.Reset(ssaop.OpRISCV64OR)
				v.AddArg2(y, x)
				return true
			}
		}
		break
	}
	// match: (OR (CZEROEQZ x:(AND z _) cond) y:(CZERONEZ z cond))
	// result: (OR x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			cond := v_0.Args[1]
			x := v_0.Args[0]
			if x.Op != ssaop.OpRISCV64AND {
				continue
			}
			x_0 := x.Args[0]
			x_1 := x.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, x_0, x_1 = _i1+1, x_1, x_0 {
				z := x_0
				y := v_1
				if y.Op != ssaop.OpRISCV64CZERONEZ {
					continue
				}
				_ = y.Args[1]
				if z != y.Args[0] || cond != y.Args[1] {
					continue
				}
				v.Reset(ssaop.OpRISCV64OR)
				v.AddArg2(x, y)
				return true
			}
		}
		break
	}
	// match: (OR x:(CZEROEQZ z cond) (CZERONEZ y:(ANDI <t> [c] z) cond))
	// result: (OR y x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if x.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			cond := x.Args[1]
			z := x.Args[0]
			if v_1.Op != ssaop.OpRISCV64CZERONEZ {
				continue
			}
			_ = v_1.Args[1]
			y := v_1.Args[0]
			if y.Op != ssaop.OpRISCV64ANDI {
				continue
			}
			if z != y.Args[0] || cond != v_1.Args[1] {
				continue
			}
			v.Reset(ssaop.OpRISCV64OR)
			v.AddArg2(y, x)
			return true
		}
		break
	}
	// match: (OR (CZEROEQZ x:(ANDI <t> [c] z) cond) y:(CZERONEZ z cond))
	// result: (OR x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			cond := v_0.Args[1]
			x := v_0.Args[0]
			if x.Op != ssaop.OpRISCV64ANDI {
				continue
			}
			z := x.Args[0]
			y := v_1
			if y.Op != ssaop.OpRISCV64CZERONEZ {
				continue
			}
			_ = y.Args[1]
			if z != y.Args[0] || cond != y.Args[1] {
				continue
			}
			v.Reset(ssaop.OpRISCV64OR)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (OR (CZEROEQZ <t> x cond) (CZERONEZ <t> (ADDI [c] x) cond))
	// result: (ADD x (CZERONEZ <t> (MOVDconst [c]) cond))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			cond := v_0.Args[1]
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != ssaop.OpRISCV64ADDI {
				continue
			}
			c := ssa.AuxIntToInt64(v_1_0.AuxInt)
			if x != v_1_0.Args[0] || cond != v_1.Args[1] {
				continue
			}
			v.Reset(ssaop.OpRISCV64ADD)
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64CZERONEZ, t)
			v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
			v1.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg2(v1, cond)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZEROEQZ <t> x cond) (CZERONEZ <t> (ORI [c] x) cond))
	// result: (OR x (CZERONEZ <t> (MOVDconst [c]) cond))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			cond := v_0.Args[1]
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != ssaop.OpRISCV64ORI {
				continue
			}
			c := ssa.AuxIntToInt64(v_1_0.AuxInt)
			if x != v_1_0.Args[0] || cond != v_1.Args[1] {
				continue
			}
			v.Reset(ssaop.OpRISCV64OR)
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64CZERONEZ, t)
			v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
			v1.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg2(v1, cond)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZEROEQZ <t> x cond) (CZERONEZ <t> (XORI [c] x) cond))
	// result: (XOR x (CZERONEZ <t> (MOVDconst [c]) cond))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			cond := v_0.Args[1]
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != ssaop.OpRISCV64XORI {
				continue
			}
			c := ssa.AuxIntToInt64(v_1_0.AuxInt)
			if x != v_1_0.Args[0] || cond != v_1.Args[1] {
				continue
			}
			v.Reset(ssaop.OpRISCV64XOR)
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64CZERONEZ, t)
			v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
			v1.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg2(v1, cond)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (ADDI [c] x) cond) (CZERONEZ <t> x cond))
	// result: (ADD x (CZEROEQZ <t> (MOVDconst [c]) cond))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			cond := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpRISCV64ADDI {
				continue
			}
			c := ssa.AuxIntToInt64(v_0_0.AuxInt)
			x := v_0_0.Args[0]
			if v_1.Op != ssaop.OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || cond != v_1.Args[1] {
				continue
			}
			v.Reset(ssaop.OpRISCV64ADD)
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64CZEROEQZ, t)
			v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
			v1.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg2(v1, cond)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (ORI [c] x) cond) (CZERONEZ <t> x cond))
	// result: (OR x (CZEROEQZ <t> (MOVDconst [c]) cond))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			cond := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpRISCV64ORI {
				continue
			}
			c := ssa.AuxIntToInt64(v_0_0.AuxInt)
			x := v_0_0.Args[0]
			if v_1.Op != ssaop.OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || cond != v_1.Args[1] {
				continue
			}
			v.Reset(ssaop.OpRISCV64OR)
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64CZEROEQZ, t)
			v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
			v1.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg2(v1, cond)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (CZEROEQZ <t> (XORI [c] x) cond) (CZERONEZ <t> x cond))
	// result: (XOR x (CZEROEQZ <t> (MOVDconst [c]) cond))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64CZEROEQZ {
				continue
			}
			t := v_0.Type
			cond := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpRISCV64XORI {
				continue
			}
			c := ssa.AuxIntToInt64(v_0_0.AuxInt)
			x := v_0_0.Args[0]
			if v_1.Op != ssaop.OpRISCV64CZERONEZ || v_1.Type != t {
				continue
			}
			_ = v_1.Args[1]
			if x != v_1.Args[0] || cond != v_1.Args[1] {
				continue
			}
			v.Reset(ssaop.OpRISCV64XOR)
			v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64CZEROEQZ, t)
			v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
			v1.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg2(v1, cond)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpRISCV64ORI(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ORI [0] x)
	// result: x
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (ORI [-1] x)
	// result: (MOVDconst [-1])
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != -1 {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-1)
		return true
	}
	// match: (ORI [x] (MOVDconst [y]))
	// result: (MOVDconst [x | y])
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		y := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(x | y)
		return true
	}
	// match: (ORI [x] (ORI [y] z))
	// result: (ORI [x | y] z)
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64ORI {
			break
		}
		y := ssa.AuxIntToInt64(v_0.AuxInt)
		z := v_0.Args[0]
		v.Reset(ssaop.OpRISCV64ORI)
		v.AuxInt = ssa.Int64ToAuxInt(x | y)
		v.AddArg(z)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64ORN(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORN x x)
	// result: (MOVDconst [-1])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-1)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64ROL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROL x (MOVDconst [val]))
	// result: (RORI [-val&63] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		val := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpRISCV64RORI)
		v.AuxInt = ssa.Int64ToAuxInt(-val & 63)
		v.AddArg(x)
		return true
	}
	// match: (ROL x (NEG y))
	// result: (ROR x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64NEG {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpRISCV64ROR)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64ROLW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROLW x (MOVDconst [val]))
	// result: (RORIW [-val&31] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		val := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpRISCV64RORIW)
		v.AuxInt = ssa.Int64ToAuxInt(-val & 31)
		v.AddArg(x)
		return true
	}
	// match: (ROLW x (NEG y))
	// result: (RORW x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64NEG {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpRISCV64RORW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64ROR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROR x (MOVDconst [val]))
	// result: (RORI [val&63] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		val := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpRISCV64RORI)
		v.AuxInt = ssa.Int64ToAuxInt(val & 63)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64RORW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (RORW x (MOVDconst [val]))
	// result: (RORIW [val&31] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		val := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpRISCV64RORIW)
		v.AuxInt = ssa.Int64ToAuxInt(val & 31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SEQZ(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SEQZ (NEG x))
	// result: (SEQZ x)
	for {
		if v_0.Op != ssaop.OpRISCV64NEG {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpRISCV64SEQZ)
		v.AddArg(x)
		return true
	}
	// match: (SEQZ (SEQZ x))
	// result: (SNEZ x)
	for {
		if v_0.Op != ssaop.OpRISCV64SEQZ {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpRISCV64SNEZ)
		v.AddArg(x)
		return true
	}
	// match: (SEQZ (SNEZ x))
	// result: (SEQZ x)
	for {
		if v_0.Op != ssaop.OpRISCV64SNEZ {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpRISCV64SEQZ)
		v.AddArg(x)
		return true
	}
	// match: (SEQZ (ANDI [c] (FCLASSD (FNEGD x))))
	// result: (SEQZ (ANDI <typ.Int64> [(c&0b11_0000_0000)|int64(bits.Reverse8(uint8(c))&0b1111_1111)] (FCLASSD x)))
	for {
		if v_0.Op != ssaop.OpRISCV64ANDI {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpRISCV64FCLASSD {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != ssaop.OpRISCV64FNEGD {
			break
		}
		x := v_0_0_0.Args[0]
		v.Reset(ssaop.OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt((c & 0b11_0000_0000) | int64(bits.Reverse8(uint8(c))&0b1111_1111))
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (SEQZ (ANDI [c] (FCLASSD (FABSD x))))
	// result: (SEQZ (ANDI <typ.Int64> [(c&0b11_1111_0000)|int64(bits.Reverse8(uint8(c))&0b0000_1111)] (FCLASSD x)))
	for {
		if v_0.Op != ssaop.OpRISCV64ANDI {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpRISCV64FCLASSD {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != ssaop.OpRISCV64FABSD {
			break
		}
		x := v_0_0_0.Args[0]
		v.Reset(ssaop.OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt((c & 0b11_1111_0000) | int64(bits.Reverse8(uint8(c))&0b0000_1111))
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLL x (MOVDconst [val]))
	// result: (SLLI [val&63] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		val := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpRISCV64SLLI)
		v.AuxInt = ssa.Int64ToAuxInt(val & 63)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SLLI(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SLLI [x] (MOVDconst [y]))
	// cond: ssa.Is32Bit(y << uint32(x))
	// result: (MOVDconst [y << uint32(x)])
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		y := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(ssa.Is32Bit(y << uint32(x))) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(y << uint32(x))
		return true
	}
	// match: (SLLI <t> [c] (ADD x x))
	// cond: c < t.Size() * 8 - 1
	// result: (SLLI [c+1] x)
	for {
		t := v.Type
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64ADD {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] || !(c < t.Size()*8-1) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLLI)
		v.AuxInt = ssa.Int64ToAuxInt(c + 1)
		v.AddArg(x)
		return true
	}
	// match: (SLLI <t> [c] (ADD x x))
	// cond: c >= t.Size() * 8 - 1
	// result: (MOVDconst [0])
	for {
		t := v.Type
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64ADD {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] || !(c >= t.Size()*8-1) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SLLW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLLW x (MOVDconst [val]))
	// result: (SLLIW [val&31] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		val := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpRISCV64SLLIW)
		v.AuxInt = ssa.Int64ToAuxInt(val & 31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SLT(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLT x (MOVDconst [val]))
	// cond: ssa.Is12Bit(val)
	// result: (SLTI [val] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		val := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(ssa.Is12Bit(val)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLTI)
		v.AuxInt = ssa.Int64ToAuxInt(val)
		v.AddArg(x)
		return true
	}
	// match: (SLT x x)
	// result: (MOVDconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SLTI(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SLTI [x] (MOVDconst [y]))
	// result: (MOVDconst [ssa.B2i(int64(y) < int64(x))])
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		y := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.B2i(int64(y) < int64(x)))
		return true
	}
	// match: (SLTI [x] (ANDI [y] _))
	// cond: y >= 0 && int64(y) < int64(x)
	// result: (MOVDconst [1])
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64ANDI {
			break
		}
		y := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(y >= 0 && int64(y) < int64(x)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SLTIU(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SLTIU [x] (MOVDconst [y]))
	// result: (MOVDconst [ssa.B2i(uint64(y) < uint64(x))])
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		y := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.B2i(uint64(y) < uint64(x)))
		return true
	}
	// match: (SLTIU [x] (ANDI [y] _))
	// cond: y >= 0 && uint64(y) < uint64(x)
	// result: (MOVDconst [1])
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64ANDI {
			break
		}
		y := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(y >= 0 && uint64(y) < uint64(x)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SLTIU [x] (ORI [y] _))
	// cond: y >= 0 && uint64(y) >= uint64(x)
	// result: (MOVDconst [0])
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64ORI {
			break
		}
		y := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(y >= 0 && uint64(y) >= uint64(x)) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SLTU(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLTU x (MOVDconst [val]))
	// cond: ssa.Is12Bit(val)
	// result: (SLTIU [val] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		val := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(ssa.Is12Bit(val)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SLTIU)
		v.AuxInt = ssa.Int64ToAuxInt(val)
		v.AddArg(x)
		return true
	}
	// match: (SLTU x x)
	// result: (MOVDconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SNEZ(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SNEZ (NEG x))
	// result: (SNEZ x)
	for {
		if v_0.Op != ssaop.OpRISCV64NEG {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpRISCV64SNEZ)
		v.AddArg(x)
		return true
	}
	// match: (SNEZ (SEQZ x))
	// result: (SEQZ x)
	for {
		if v_0.Op != ssaop.OpRISCV64SEQZ {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpRISCV64SEQZ)
		v.AddArg(x)
		return true
	}
	// match: (SNEZ (SNEZ x))
	// result: (SNEZ x)
	for {
		if v_0.Op != ssaop.OpRISCV64SNEZ {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpRISCV64SNEZ)
		v.AddArg(x)
		return true
	}
	// match: (SNEZ (ANDI [c] (FCLASSD (FNEGD x))))
	// result: (SNEZ (ANDI <typ.Int64> [(c&0b11_0000_0000)|int64(bits.Reverse8(uint8(c))&0b1111_1111)] (FCLASSD x)))
	for {
		if v_0.Op != ssaop.OpRISCV64ANDI {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpRISCV64FCLASSD {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != ssaop.OpRISCV64FNEGD {
			break
		}
		x := v_0_0_0.Args[0]
		v.Reset(ssaop.OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt((c & 0b11_0000_0000) | int64(bits.Reverse8(uint8(c))&0b1111_1111))
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (SNEZ (ANDI [c] (FCLASSD (FABSD x))))
	// result: (SNEZ (ANDI <typ.Int64> [(c&0b11_1111_0000)|int64(bits.Reverse8(uint8(c))&0b0000_1111)] (FCLASSD x)))
	for {
		if v_0.Op != ssaop.OpRISCV64ANDI {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpRISCV64FCLASSD {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != ssaop.OpRISCV64FABSD {
			break
		}
		x := v_0_0_0.Args[0]
		v.Reset(ssaop.OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt((c & 0b11_1111_0000) | int64(bits.Reverse8(uint8(c))&0b0000_1111))
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRA x (MOVDconst [val]))
	// result: (SRAI [val&63] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		val := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpRISCV64SRAI)
		v.AuxInt = ssa.Int64ToAuxInt(val & 63)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SRAI(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (SRAI [x] (MOVWreg y))
	// cond: x >= 0 && x <= 31
	// result: (SRAIW [x] y)
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVWreg {
			break
		}
		y := v_0.Args[0]
		if !(x >= 0 && x <= 31) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRAIW)
		v.AuxInt = ssa.Int64ToAuxInt(x)
		v.AddArg(y)
		return true
	}
	// match: (SRAI <t> [x] (MOVBreg y))
	// cond: x >= 8
	// result: (SRAI [63] (SLLI <t> [56] y))
	for {
		t := v.Type
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVBreg {
			break
		}
		y := v_0.Args[0]
		if !(x >= 8) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRAI)
		v.AuxInt = ssa.Int64ToAuxInt(63)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, t)
		v0.AuxInt = ssa.Int64ToAuxInt(56)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SRAI <t> [x] (MOVHreg y))
	// cond: x >= 16
	// result: (SRAI [63] (SLLI <t> [48] y))
	for {
		t := v.Type
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVHreg {
			break
		}
		y := v_0.Args[0]
		if !(x >= 16) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRAI)
		v.AuxInt = ssa.Int64ToAuxInt(63)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLLI, t)
		v0.AuxInt = ssa.Int64ToAuxInt(48)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SRAI [x] (MOVWreg y))
	// cond: x >= 32
	// result: (SRAIW [31] y)
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVWreg {
			break
		}
		y := v_0.Args[0]
		if !(x >= 32) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRAIW)
		v.AuxInt = ssa.Int64ToAuxInt(31)
		v.AddArg(y)
		return true
	}
	// match: (SRAI [x] (MOVDconst [y]))
	// result: (MOVDconst [int64(y) >> uint32(x)])
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		y := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(y) >> uint32(x))
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SRAW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRAW x (MOVDconst [val]))
	// result: (SRAIW [val&31] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		val := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpRISCV64SRAIW)
		v.AuxInt = ssa.Int64ToAuxInt(val & 31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRL x (MOVDconst [val]))
	// result: (SRLI [val&63] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		val := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpRISCV64SRLI)
		v.AuxInt = ssa.Int64ToAuxInt(val & 63)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SRLI(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SRLI [x] (MOVWUreg y))
	// cond: x >= 0 && x <= 31
	// result: (SRLIW [x] y)
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVWUreg {
			break
		}
		y := v_0.Args[0]
		if !(x >= 0 && x <= 31) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRLIW)
		v.AuxInt = ssa.Int64ToAuxInt(x)
		v.AddArg(y)
		return true
	}
	// match: (SRLI [x] (MOVBUreg y))
	// cond: x >= 8
	// result: (MOVDconst [0])
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVBUreg {
			break
		}
		if !(x >= 8) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SRLI [x] (MOVHUreg y))
	// cond: x >= 16
	// result: (MOVDconst [0])
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVHUreg {
			break
		}
		if !(x >= 16) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SRLI [x] (MOVWUreg y))
	// cond: x >= 32
	// result: (MOVDconst [0])
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVWUreg {
			break
		}
		if !(x >= 32) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SRLI [x] (MOVDconst [y]))
	// result: (MOVDconst [int64(uint64(y) >> uint32(x))])
	for {
		x := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		y := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(y) >> uint32(x)))
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SRLW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRLW x (MOVDconst [val]))
	// result: (SRLIW [val&31] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		val := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpRISCV64SRLIW)
		v.AuxInt = ssa.Int64ToAuxInt(val & 31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SUB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUB x (NEG y))
	// result: (ADD x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64NEG {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpRISCV64ADD)
		v.AddArg2(x, y)
		return true
	}
	// match: (SUB x x)
	// result: (MOVDconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SUB x (MOVDconst [val]))
	// cond: ssa.Is32Bit(-val)
	// result: (ADDI [-val] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		val := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(ssa.Is32Bit(-val)) {
			break
		}
		v.Reset(ssaop.OpRISCV64ADDI)
		v.AuxInt = ssa.Int64ToAuxInt(-val)
		v.AddArg(x)
		return true
	}
	// match: (SUB <t> (MOVDconst [val]) y)
	// cond: ssa.Is32Bit(-val)
	// result: (NEG (ADDI <t> [-val] y))
	for {
		t := v.Type
		if v_0.Op != ssaop.OpRISCV64MOVDconst {
			break
		}
		val := ssa.AuxIntToInt64(v_0.AuxInt)
		y := v_1
		if !(ssa.Is32Bit(-val)) {
			break
		}
		v.Reset(ssaop.OpRISCV64NEG)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADDI, t)
		v0.AuxInt = ssa.Int64ToAuxInt(-val)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SUB x (MOVDconst [0]))
	// result: x
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (SUB (MOVDconst [0]) x)
	// result: (NEG x)
	for {
		if v_0.Op != ssaop.OpRISCV64MOVDconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpRISCV64NEG)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64SUBW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBW x (MOVDconst [0]))
	// result: (ADDIW [0] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpRISCV64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpRISCV64ADDIW)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg(x)
		return true
	}
	// match: (SUBW (MOVDconst [0]) x)
	// result: (NEGW x)
	for {
		if v_0.Op != ssaop.OpRISCV64MOVDconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpRISCV64NEGW)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRISCV64XOR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XOR (MOVDconst [val]) x)
	// cond: ssa.Is32Bit(val)
	// result: (XORI [val] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpRISCV64MOVDconst {
				continue
			}
			val := ssa.AuxIntToInt64(v_0.AuxInt)
			x := v_1
			if !(ssa.Is32Bit(val)) {
				continue
			}
			v.Reset(ssaop.OpRISCV64XORI)
			v.AuxInt = ssa.Int64ToAuxInt(val)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (XOR x x)
	// result: (MOVDconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpRotateLeft16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft16 <t> x y)
	// result: (OR (SLL <t> x (ANDI [15] <y.Type> y)) (SRL <t> (ZeroExt16to64 x) (ANDI [15] <y.Type> (NEG <y.Type> y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64OR)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, y.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(15)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRL, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(x)
		v4 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, y.Type)
		v4.AuxInt = ssa.Int64ToAuxInt(15)
		v5 := b.NewValue0(v.Pos, ssaop.OpRISCV64NEG, y.Type)
		v5.AddArg(y)
		v4.AddArg(v5)
		v2.AddArg2(v3, v4)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValue_OpRotateLeft8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft8 <t> x y)
	// result: (OR (SLL <t> x (ANDI [7] <y.Type> y)) (SRL <t> (ZeroExt8to64 x) (ANDI [7] <y.Type> (NEG <y.Type> y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRISCV64OR)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, y.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(7)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRL, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(x)
		v4 := b.NewValue0(v.Pos, ssaop.OpRISCV64ANDI, y.Type)
		v4.AuxInt = ssa.Int64ToAuxInt(7)
		v5 := b.NewValue0(v.Pos, ssaop.OpRISCV64NEG, y.Type)
		v5.AddArg(y)
		v4.AddArg(v5)
		v2.AddArg2(v3, v4)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValue_OpRsh16Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt16to64 x) y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpNeg16, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh16Ux16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL (ZeroExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh16Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt16to64 x) y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpNeg16, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh16Ux32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL (ZeroExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh16Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt16to64 x) y) (Neg16 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpNeg16, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh16Ux64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL (ZeroExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh16Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt16to64 x) y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpNeg16, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh16Ux8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL (ZeroExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh16x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRA <t> (SignExt16to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt16to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADDI, y.Type)
		v2.AuxInt = ssa.Int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, y.Type)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh16x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA (SignExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh16x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRA <t> (SignExt16to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt32to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADDI, y.Type)
		v2.AuxInt = ssa.Int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, y.Type)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh16x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA (SignExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh16x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRA <t> (SignExt16to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADDI, y.Type)
		v2.AuxInt = ssa.Int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, y.Type)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh16x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA (SignExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRA <t> (SignExt16to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt8to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADDI, y.Type)
		v2.AuxInt = ssa.Int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, y.Type)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh16x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA (SignExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh32Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SRLW <t> x y) (Neg32 <t> (SLTIU <t> [32] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRLW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg32, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(32)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh32Ux16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRLW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRLW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh32Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SRLW <t> x y) (Neg32 <t> (SLTIU <t> [32] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRLW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg32, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(32)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh32Ux32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRLW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRLW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh32Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32Ux64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SRLW <t> x y) (Neg32 <t> (SLTIU <t> [32] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRLW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg32, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh32Ux64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRLW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRLW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh32Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SRLW <t> x y) (Neg32 <t> (SLTIU <t> [32] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRLW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg32, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(32)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh32Ux8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRLW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRLW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh32x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRAW <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [32] (ZeroExt16to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRAW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64OR, y.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADDI, y.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(-1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, y.Type)
		v2.AuxInt = ssa.Int64ToAuxInt(32)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRAW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh32x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRAW <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [32] (ZeroExt32to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRAW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64OR, y.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADDI, y.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(-1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, y.Type)
		v2.AuxInt = ssa.Int64ToAuxInt(32)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRAW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh32x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32x64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRAW <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [32] y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRAW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64OR, y.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADDI, y.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(-1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, y.Type)
		v2.AuxInt = ssa.Int64ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRAW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh32x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRAW <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [32] (ZeroExt8to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRAW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64OR, y.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADDI, y.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(-1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, y.Type)
		v2.AuxInt = ssa.Int64ToAuxInt(32)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRAW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh64Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SRL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg64, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh64Ux16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh64Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SRL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg64, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh64Ux32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh64Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64Ux64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SRL <t> x y) (Neg64 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg64, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh64Ux64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh64Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SRL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpNeg64, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh64Ux8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh64x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRA <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt16to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64OR, y.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADDI, y.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(-1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, y.Type)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh64x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRA <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt32to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64OR, y.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADDI, y.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(-1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, y.Type)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh64x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64x64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRA <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64OR, y.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADDI, y.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(-1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, y.Type)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh64x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRA <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt8to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64OR, y.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADDI, y.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(-1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, y.Type)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh8Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt8to64 x) y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpNeg8, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh8Ux16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL (ZeroExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh8Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt8to64 x) y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpNeg8, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh8Ux32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL (ZeroExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh8Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt8to64 x) y) (Neg8 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpNeg8, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh8Ux64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL (ZeroExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh8Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (AND (SRL <t> (ZeroExt8to64 x) y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpNeg8, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, t)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Rsh8Ux8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL (ZeroExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRL)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRA <t> (SignExt8to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt16to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADDI, y.Type)
		v2.AuxInt = ssa.Int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, y.Type)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh8x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA (SignExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh8x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRA <t> (SignExt8to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt32to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADDI, y.Type)
		v2.AuxInt = ssa.Int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, y.Type)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh8x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA (SignExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh8x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRA <t> (SignExt8to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADDI, y.Type)
		v2.AuxInt = ssa.Int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, y.Type)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh8x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA (SignExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValue_OpRsh8x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRA <t> (SignExt8to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt8to64 y)))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64OR, y.Type)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADDI, y.Type)
		v2.AuxInt = ssa.Int64ToAuxInt(-1)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTIU, y.Type)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg2(y, v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh8x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA (SignExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRISCV64SRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	return false
}
func rewriteValue_OpSelect0(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select0 (Add64carry x y c))
	// result: (ADD (ADD <typ.UInt64> x y) c)
	for {
		if v_0.Op != ssaop.OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.Reset(ssaop.OpRISCV64ADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADD, typ.UInt64)
		v0.AddArg2(x, y)
		v.AddArg2(v0, c)
		return true
	}
	// match: (Select0 (Sub64borrow x y c))
	// result: (SUB (SUB <typ.UInt64> x y) c)
	for {
		if v_0.Op != ssaop.OpSub64borrow {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.Reset(ssaop.OpRISCV64SUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SUB, typ.UInt64)
		v0.AddArg2(x, y)
		v.AddArg2(v0, c)
		return true
	}
	// match: (Select0 m:(LoweredMuluhilo x y))
	// cond: m.Uses == 1
	// result: (MULHU x y)
	for {
		m := v_0
		if m.Op != ssaop.OpRISCV64LoweredMuluhilo {
			break
		}
		y := m.Args[1]
		x := m.Args[0]
		if !(m.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpRISCV64MULHU)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpSelect1(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select1 (Add64carry x y c))
	// result: (OR (SLTU <typ.UInt64> s:(ADD <typ.UInt64> x y) x) (SLTU <typ.UInt64> (ADD <typ.UInt64> s c) s))
	for {
		if v_0.Op != ssaop.OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.Reset(ssaop.OpRISCV64OR)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTU, typ.UInt64)
		s := b.NewValue0(v.Pos, ssaop.OpRISCV64ADD, typ.UInt64)
		s.AddArg2(x, y)
		v0.AddArg2(s, x)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTU, typ.UInt64)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64ADD, typ.UInt64)
		v3.AddArg2(s, c)
		v2.AddArg2(v3, s)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Select1 (Sub64borrow x y c))
	// result: (OR (SLTU <typ.UInt64> x s:(SUB <typ.UInt64> x y)) (SLTU <typ.UInt64> s (SUB <typ.UInt64> s c)))
	for {
		if v_0.Op != ssaop.OpSub64borrow {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.Reset(ssaop.OpRISCV64OR)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTU, typ.UInt64)
		s := b.NewValue0(v.Pos, ssaop.OpRISCV64SUB, typ.UInt64)
		s.AddArg2(x, y)
		v0.AddArg2(x, s)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64SLTU, typ.UInt64)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64SUB, typ.UInt64)
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
		if m.Op != ssaop.OpRISCV64LoweredMuluhilo {
			break
		}
		y := m.Args[1]
		x := m.Args[0]
		if !(m.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpRISCV64MUL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpSlicemask(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Slicemask <t> x)
	// result: (SRAI [63] (NEG <t> x))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpRISCV64SRAI)
		v.AuxInt = ssa.Int64ToAuxInt(63)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64NEG, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpStore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 1
	// result: (MOVBstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 1) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVBstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 2
	// result: (MOVHstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 2) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && !t.IsFloat()
	// result: (MOVWstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && !t.IsFloat()) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVWstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && !t.IsFloat()
	// result: (MOVDstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && !t.IsFloat()) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && t.IsFloat()
	// result: (FMOVWstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && t.IsFloat()) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMOVWstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && t.IsFloat()
	// result: (FMOVDstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && t.IsFloat()) {
			break
		}
		v.Reset(ssaop.OpRISCV64FMOVDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpZero(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Zero [0] _ mem)
	// result: mem
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		mem := v_1
		v.CopyOf(mem)
		return true
	}
	// match: (Zero [1] ptr mem)
	// result: (MOVBstore ptr (MOVDconst [0]) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpRISCV64MOVBstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [2] {t} ptr mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore ptr (MOVDconst [0]) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [2] ptr mem)
	// result: (MOVBstore [1] ptr (MOVDconst [0]) (MOVBstore ptr (MOVDconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpRISCV64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [4] {t} ptr mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore ptr (MOVDconst [0]) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVWstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [4] {t} ptr mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [2] ptr (MOVDconst [0]) (MOVHstore ptr (MOVDconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [4] ptr mem)
	// result: (MOVBstore [3] ptr (MOVDconst [0]) (MOVBstore [2] ptr (MOVDconst [0]) (MOVBstore [1] ptr (MOVDconst [0]) (MOVBstore ptr (MOVDconst [0]) mem))))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpRISCV64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(2)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBstore, types.TypeMem)
		v2.AuxInt = ssa.Int32ToAuxInt(1)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBstore, types.TypeMem)
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
		if ssa.AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%8 == 0) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVDstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [8] {t} ptr mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore [4] ptr (MOVDconst [0]) (MOVWstore ptr (MOVDconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVWstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [8] {t} ptr mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [6] ptr (MOVDconst [0]) (MOVHstore [4] ptr (MOVDconst [0]) (MOVHstore [2] ptr (MOVDconst [0]) (MOVHstore ptr (MOVDconst [0]) mem))))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(4)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHstore, types.TypeMem)
		v2.AuxInt = ssa.Int32ToAuxInt(2)
		v3 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHstore, types.TypeMem)
		v3.AddArg3(ptr, v0, mem)
		v2.AddArg3(ptr, v0, v3)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [3] ptr mem)
	// result: (MOVBstore [2] ptr (MOVDconst [0]) (MOVBstore [1] ptr (MOVDconst [0]) (MOVBstore ptr (MOVDconst [0]) mem)))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 3 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpRISCV64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVBstore, types.TypeMem)
		v2.AddArg3(ptr, v0, mem)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [6] {t} ptr mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [4] ptr (MOVDconst [0]) (MOVHstore [2] ptr (MOVDconst [0]) (MOVHstore ptr (MOVDconst [0]) mem)))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 6 {
			break
		}
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.Reset(ssaop.OpRISCV64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(2)
		v2 := b.NewValue0(v.Pos, ssaop.OpRISCV64MOVHstore, types.TypeMem)
		v2.AddArg3(ptr, v0, mem)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [s] {t} ptr mem)
	// cond: s <= 24*ssa.MoveSize(t.Alignment(), config)
	// result: (LoweredZero [ssa.MakeValAndOff(int32(s),int32(t.Alignment()))] ptr mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(s <= 24*ssa.MoveSize(t.Alignment(), config)) {
			break
		}
		v.Reset(ssaop.OpRISCV64LoweredZero)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(int32(s), int32(t.Alignment())))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Zero [s] {t} ptr mem)
	// cond: s > 24*ssa.MoveSize(t.Alignment(), config)
	// result: (LoweredZeroLoop [ssa.MakeValAndOff(int32(s),int32(t.Alignment()))] ptr mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(s > 24*ssa.MoveSize(t.Alignment(), config)) {
			break
		}
		v.Reset(ssaop.OpRISCV64LoweredZeroLoop)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(int32(s), int32(t.Alignment())))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func RewriteBlock(b *ssa.Block) bool {
	typ := &b.Func.Config.Types
	switch b.Kind {
	case block.BlockRISCV64BEQ:
		// match: (BEQ (MOVDconst [0]) cond yes no)
		// result: (BEQZ cond yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64MOVDconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			cond := b.Controls[1]
			b.ResetWithControl(block.BlockRISCV64BEQZ, cond)
			return true
		}
		// match: (BEQ cond (MOVDconst [0]) yes no)
		// result: (BEQZ cond yes no)
		for b.Controls[1].Op == ssaop.OpRISCV64MOVDconst {
			cond := b.Controls[0]
			v_1 := b.Controls[1]
			if ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockRISCV64BEQZ, cond)
			return true
		}
	case block.BlockRISCV64BEQZ:
		// match: (BEQZ (SEQZ x) yes no)
		// result: (BNEZ x yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64SEQZ {
			v_0 := b.Controls[0]
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockRISCV64BNEZ, x)
			return true
		}
		// match: (BEQZ (SNEZ x) yes no)
		// result: (BEQZ x yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64SNEZ {
			v_0 := b.Controls[0]
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockRISCV64BEQZ, x)
			return true
		}
		// match: (BEQZ (NEG x) yes no)
		// result: (BEQZ x yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64NEG {
			v_0 := b.Controls[0]
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockRISCV64BEQZ, x)
			return true
		}
		// match: (BEQZ (FNES <t> x y) yes no)
		// result: (BNEZ (FEQS <t> x y) yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64FNES {
			v_0 := b.Controls[0]
			t := v_0.Type
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				v0 := b.NewValue0(v_0.Pos, ssaop.OpRISCV64FEQS, t)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockRISCV64BNEZ, v0)
				return true
			}
		}
		// match: (BEQZ (FNED <t> x y) yes no)
		// result: (BNEZ (FEQD <t> x y) yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64FNED {
			v_0 := b.Controls[0]
			t := v_0.Type
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				v0 := b.NewValue0(v_0.Pos, ssaop.OpRISCV64FEQD, t)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockRISCV64BNEZ, v0)
				return true
			}
		}
		// match: (BEQZ (SUB x y) yes no)
		// result: (BEQ x y yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64SUB {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.ResetWithControl2(block.BlockRISCV64BEQ, x, y)
			return true
		}
		// match: (BEQZ (SLT x y) yes no)
		// result: (BGE x y yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64SLT {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.ResetWithControl2(block.BlockRISCV64BGE, x, y)
			return true
		}
		// match: (BEQZ (SLTU x y) yes no)
		// result: (BGEU x y yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64SLTU {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.ResetWithControl2(block.BlockRISCV64BGEU, x, y)
			return true
		}
		// match: (BEQZ (SLTI [x] y) yes no)
		// result: (BGE y (MOVDconst [x]) yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64SLTI {
			v_0 := b.Controls[0]
			x := ssa.AuxIntToInt64(v_0.AuxInt)
			y := v_0.Args[0]
			v0 := b.NewValue0(b.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
			v0.AuxInt = ssa.Int64ToAuxInt(x)
			b.ResetWithControl2(block.BlockRISCV64BGE, y, v0)
			return true
		}
		// match: (BEQZ (SLTIU [x] y) yes no)
		// result: (BGEU y (MOVDconst [x]) yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64SLTIU {
			v_0 := b.Controls[0]
			x := ssa.AuxIntToInt64(v_0.AuxInt)
			y := v_0.Args[0]
			v0 := b.NewValue0(b.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
			v0.AuxInt = ssa.Int64ToAuxInt(x)
			b.ResetWithControl2(block.BlockRISCV64BGEU, y, v0)
			return true
		}
		// match: (BEQZ (ANDI [c] (FCLASSD (FNEGD x))) yes no)
		// result: (BEQZ (ANDI <typ.Int64> [(c&0b11_0000_0000)|int64(bits.Reverse8(uint8(c))&0b1111_1111)] (FCLASSD x)) yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64ANDI {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpRISCV64FCLASSD {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != ssaop.OpRISCV64FNEGD {
				break
			}
			x := v_0_0_0.Args[0]
			v0 := b.NewValue0(v_0.Pos, ssaop.OpRISCV64ANDI, typ.Int64)
			v0.AuxInt = ssa.Int64ToAuxInt((c & 0b11_0000_0000) | int64(bits.Reverse8(uint8(c))&0b1111_1111))
			v1 := b.NewValue0(v_0.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
			v1.AddArg(x)
			v0.AddArg(v1)
			b.ResetWithControl(block.BlockRISCV64BEQZ, v0)
			return true
		}
		// match: (BEQZ (ANDI [c] (FCLASSD (FABSD x))) yes no)
		// result: (BEQZ (ANDI <typ.Int64> [(c&0b11_1111_0000)|int64(bits.Reverse8(uint8(c))&0b0000_1111)] (FCLASSD x)) yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64ANDI {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpRISCV64FCLASSD {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != ssaop.OpRISCV64FABSD {
				break
			}
			x := v_0_0_0.Args[0]
			v0 := b.NewValue0(v_0.Pos, ssaop.OpRISCV64ANDI, typ.Int64)
			v0.AuxInt = ssa.Int64ToAuxInt((c & 0b11_1111_0000) | int64(bits.Reverse8(uint8(c))&0b0000_1111))
			v1 := b.NewValue0(v_0.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
			v1.AddArg(x)
			v0.AddArg(v1)
			b.ResetWithControl(block.BlockRISCV64BEQZ, v0)
			return true
		}
	case block.BlockRISCV64BGE:
		// match: (BGE (MOVDconst [0]) cond yes no)
		// result: (BLEZ cond yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64MOVDconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			cond := b.Controls[1]
			b.ResetWithControl(block.BlockRISCV64BLEZ, cond)
			return true
		}
		// match: (BGE cond (MOVDconst [0]) yes no)
		// result: (BGEZ cond yes no)
		for b.Controls[1].Op == ssaop.OpRISCV64MOVDconst {
			cond := b.Controls[0]
			v_1 := b.Controls[1]
			if ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockRISCV64BGEZ, cond)
			return true
		}
	case block.BlockRISCV64BGEU:
		// match: (BGEU (MOVDconst [0]) cond yes no)
		// result: (BEQZ cond yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64MOVDconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			cond := b.Controls[1]
			b.ResetWithControl(block.BlockRISCV64BEQZ, cond)
			return true
		}
	case block.BlockRISCV64BLT:
		// match: (BLT (MOVDconst [0]) cond yes no)
		// result: (BGTZ cond yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64MOVDconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			cond := b.Controls[1]
			b.ResetWithControl(block.BlockRISCV64BGTZ, cond)
			return true
		}
		// match: (BLT cond (MOVDconst [0]) yes no)
		// result: (BLTZ cond yes no)
		for b.Controls[1].Op == ssaop.OpRISCV64MOVDconst {
			cond := b.Controls[0]
			v_1 := b.Controls[1]
			if ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockRISCV64BLTZ, cond)
			return true
		}
	case block.BlockRISCV64BLTU:
		// match: (BLTU (MOVDconst [0]) cond yes no)
		// result: (BNEZ cond yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64MOVDconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			cond := b.Controls[1]
			b.ResetWithControl(block.BlockRISCV64BNEZ, cond)
			return true
		}
	case block.BlockRISCV64BNE:
		// match: (BNE (MOVDconst [0]) cond yes no)
		// result: (BNEZ cond yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64MOVDconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			cond := b.Controls[1]
			b.ResetWithControl(block.BlockRISCV64BNEZ, cond)
			return true
		}
		// match: (BNE cond (MOVDconst [0]) yes no)
		// result: (BNEZ cond yes no)
		for b.Controls[1].Op == ssaop.OpRISCV64MOVDconst {
			cond := b.Controls[0]
			v_1 := b.Controls[1]
			if ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockRISCV64BNEZ, cond)
			return true
		}
	case block.BlockRISCV64BNEZ:
		// match: (BNEZ (SEQZ x) yes no)
		// result: (BEQZ x yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64SEQZ {
			v_0 := b.Controls[0]
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockRISCV64BEQZ, x)
			return true
		}
		// match: (BNEZ (SNEZ x) yes no)
		// result: (BNEZ x yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64SNEZ {
			v_0 := b.Controls[0]
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockRISCV64BNEZ, x)
			return true
		}
		// match: (BNEZ (NEG x) yes no)
		// result: (BNEZ x yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64NEG {
			v_0 := b.Controls[0]
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockRISCV64BNEZ, x)
			return true
		}
		// match: (BNEZ (FNES <t> x y) yes no)
		// result: (BEQZ (FEQS <t> x y) yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64FNES {
			v_0 := b.Controls[0]
			t := v_0.Type
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				v0 := b.NewValue0(v_0.Pos, ssaop.OpRISCV64FEQS, t)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockRISCV64BEQZ, v0)
				return true
			}
		}
		// match: (BNEZ (FNED <t> x y) yes no)
		// result: (BEQZ (FEQD <t> x y) yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64FNED {
			v_0 := b.Controls[0]
			t := v_0.Type
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				v0 := b.NewValue0(v_0.Pos, ssaop.OpRISCV64FEQD, t)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockRISCV64BEQZ, v0)
				return true
			}
		}
		// match: (BNEZ (SUB x y) yes no)
		// result: (BNE x y yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64SUB {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.ResetWithControl2(block.BlockRISCV64BNE, x, y)
			return true
		}
		// match: (BNEZ (SLT x y) yes no)
		// result: (BLT x y yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64SLT {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.ResetWithControl2(block.BlockRISCV64BLT, x, y)
			return true
		}
		// match: (BNEZ (SLTU x y) yes no)
		// result: (BLTU x y yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64SLTU {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.ResetWithControl2(block.BlockRISCV64BLTU, x, y)
			return true
		}
		// match: (BNEZ (SLTI [x] y) yes no)
		// result: (BLT y (MOVDconst [x]) yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64SLTI {
			v_0 := b.Controls[0]
			x := ssa.AuxIntToInt64(v_0.AuxInt)
			y := v_0.Args[0]
			v0 := b.NewValue0(b.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
			v0.AuxInt = ssa.Int64ToAuxInt(x)
			b.ResetWithControl2(block.BlockRISCV64BLT, y, v0)
			return true
		}
		// match: (BNEZ (SLTIU [x] y) yes no)
		// result: (BLTU y (MOVDconst [x]) yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64SLTIU {
			v_0 := b.Controls[0]
			x := ssa.AuxIntToInt64(v_0.AuxInt)
			y := v_0.Args[0]
			v0 := b.NewValue0(b.Pos, ssaop.OpRISCV64MOVDconst, typ.UInt64)
			v0.AuxInt = ssa.Int64ToAuxInt(x)
			b.ResetWithControl2(block.BlockRISCV64BLTU, y, v0)
			return true
		}
		// match: (BNEZ (ANDI [c] (FCLASSD (FNEGD x))) yes no)
		// result: (BNEZ (ANDI <typ.Int64> [(c&0b11_0000_0000)|int64(bits.Reverse8(uint8(c))&0b1111_1111)] (FCLASSD x)) yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64ANDI {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpRISCV64FCLASSD {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != ssaop.OpRISCV64FNEGD {
				break
			}
			x := v_0_0_0.Args[0]
			v0 := b.NewValue0(v_0.Pos, ssaop.OpRISCV64ANDI, typ.Int64)
			v0.AuxInt = ssa.Int64ToAuxInt((c & 0b11_0000_0000) | int64(bits.Reverse8(uint8(c))&0b1111_1111))
			v1 := b.NewValue0(v_0.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
			v1.AddArg(x)
			v0.AddArg(v1)
			b.ResetWithControl(block.BlockRISCV64BNEZ, v0)
			return true
		}
		// match: (BNEZ (ANDI [c] (FCLASSD (FABSD x))) yes no)
		// result: (BNEZ (ANDI <typ.Int64> [(c&0b11_1111_0000)|int64(bits.Reverse8(uint8(c))&0b0000_1111)] (FCLASSD x)) yes no)
		for b.Controls[0].Op == ssaop.OpRISCV64ANDI {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpRISCV64FCLASSD {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != ssaop.OpRISCV64FABSD {
				break
			}
			x := v_0_0_0.Args[0]
			v0 := b.NewValue0(v_0.Pos, ssaop.OpRISCV64ANDI, typ.Int64)
			v0.AuxInt = ssa.Int64ToAuxInt((c & 0b11_1111_0000) | int64(bits.Reverse8(uint8(c))&0b0000_1111))
			v1 := b.NewValue0(v_0.Pos, ssaop.OpRISCV64FCLASSD, typ.Int64)
			v1.AddArg(x)
			v0.AddArg(v1)
			b.ResetWithControl(block.BlockRISCV64BNEZ, v0)
			return true
		}
	case block.BlockIf:
		// match: (If cond yes no)
		// result: (BNEZ (MOVBUreg <typ.UInt64> cond) yes no)
		for {
			cond := b.Controls[0]
			v0 := b.NewValue0(cond.Pos, ssaop.OpRISCV64MOVBUreg, typ.UInt64)
			v0.AddArg(cond)
			b.ResetWithControl(block.BlockRISCV64BNEZ, v0)
			return true
		}
	case block.BlockJumpTable:
		// match: (JumpTable idx)
		// result: (JUMPTABLE {ssa.MakeJumpTableSym(b)} idx (MOVaddr <typ.Uintptr> {ssa.MakeJumpTableSym(b)} (SB)))
		for {
			idx := b.Controls[0]
			v0 := b.NewValue0(b.Pos, ssaop.OpRISCV64MOVaddr, typ.Uintptr)
			v0.Aux = ssa.SymToAux(ssa.MakeJumpTableSym(b))
			v1 := b.NewValue0(b.Pos, ssaop.OpSB, typ.Uintptr)
			v0.AddArg(v1)
			b.ResetWithControl2(block.BlockRISCV64JUMPTABLE, idx, v0)
			b.Aux = ssa.SymToAux(ssa.MakeJumpTableSym(b))
			return true
		}
	}
	return false
}
