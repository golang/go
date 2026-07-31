// Code generated from _gen/MIPS.rules using 'go generate'; DO NOT EDIT.

package rewritemips

import "cmd/compile/internal/types"
import "cmd/compile/internal/ssa/block"
import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa"

func RewriteValue(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.OpAbs:
		v.Op = ssaop.OpMIPSABSD
		return true
	case ssaop.OpAdd16:
		v.Op = ssaop.OpMIPSADD
		return true
	case ssaop.OpAdd32:
		v.Op = ssaop.OpMIPSADD
		return true
	case ssaop.OpAdd32F:
		v.Op = ssaop.OpMIPSADDF
		return true
	case ssaop.OpAdd32withcarry:
		return rewriteValue_OpAdd32withcarry(v)
	case ssaop.OpAdd64F:
		v.Op = ssaop.OpMIPSADDD
		return true
	case ssaop.OpAdd8:
		v.Op = ssaop.OpMIPSADD
		return true
	case ssaop.OpAddPtr:
		v.Op = ssaop.OpMIPSADD
		return true
	case ssaop.OpAddr:
		return rewriteValue_OpAddr(v)
	case ssaop.OpAnd16:
		v.Op = ssaop.OpMIPSAND
		return true
	case ssaop.OpAnd32:
		v.Op = ssaop.OpMIPSAND
		return true
	case ssaop.OpAnd8:
		v.Op = ssaop.OpMIPSAND
		return true
	case ssaop.OpAndB:
		v.Op = ssaop.OpMIPSAND
		return true
	case ssaop.OpAtomicAdd32:
		v.Op = ssaop.OpMIPSLoweredAtomicAdd
		return true
	case ssaop.OpAtomicAnd32:
		v.Op = ssaop.OpMIPSLoweredAtomicAnd
		return true
	case ssaop.OpAtomicAnd8:
		return rewriteValue_OpAtomicAnd8(v)
	case ssaop.OpAtomicCompareAndSwap32:
		v.Op = ssaop.OpMIPSLoweredAtomicCas
		return true
	case ssaop.OpAtomicExchange32:
		v.Op = ssaop.OpMIPSLoweredAtomicExchange
		return true
	case ssaop.OpAtomicLoad32:
		v.Op = ssaop.OpMIPSLoweredAtomicLoad32
		return true
	case ssaop.OpAtomicLoad8:
		v.Op = ssaop.OpMIPSLoweredAtomicLoad8
		return true
	case ssaop.OpAtomicLoadPtr:
		v.Op = ssaop.OpMIPSLoweredAtomicLoad32
		return true
	case ssaop.OpAtomicOr32:
		v.Op = ssaop.OpMIPSLoweredAtomicOr
		return true
	case ssaop.OpAtomicOr8:
		return rewriteValue_OpAtomicOr8(v)
	case ssaop.OpAtomicStore32:
		v.Op = ssaop.OpMIPSLoweredAtomicStore32
		return true
	case ssaop.OpAtomicStore8:
		v.Op = ssaop.OpMIPSLoweredAtomicStore8
		return true
	case ssaop.OpAtomicStorePtrNoWB:
		v.Op = ssaop.OpMIPSLoweredAtomicStore32
		return true
	case ssaop.OpAvg32u:
		return rewriteValue_OpAvg32u(v)
	case ssaop.OpBitLen16:
		return rewriteValue_OpBitLen16(v)
	case ssaop.OpBitLen32:
		return rewriteValue_OpBitLen32(v)
	case ssaop.OpBitLen8:
		return rewriteValue_OpBitLen8(v)
	case ssaop.OpClosureCall:
		v.Op = ssaop.OpMIPSCALLclosure
		return true
	case ssaop.OpCom16:
		return rewriteValue_OpCom16(v)
	case ssaop.OpCom32:
		return rewriteValue_OpCom32(v)
	case ssaop.OpCom8:
		return rewriteValue_OpCom8(v)
	case ssaop.OpConst16:
		return rewriteValue_OpConst16(v)
	case ssaop.OpConst32:
		return rewriteValue_OpConst32(v)
	case ssaop.OpConst32F:
		v.Op = ssaop.OpMIPSMOVFconst
		return true
	case ssaop.OpConst64F:
		v.Op = ssaop.OpMIPSMOVDconst
		return true
	case ssaop.OpConst8:
		return rewriteValue_OpConst8(v)
	case ssaop.OpConstBool:
		return rewriteValue_OpConstBool(v)
	case ssaop.OpConstNil:
		return rewriteValue_OpConstNil(v)
	case ssaop.OpCtz16:
		return rewriteValue_OpCtz16(v)
	case ssaop.OpCtz16NonZero:
		v.Op = ssaop.OpCtz32
		return true
	case ssaop.OpCtz32:
		return rewriteValue_OpCtz32(v)
	case ssaop.OpCtz32NonZero:
		v.Op = ssaop.OpCtz32
		return true
	case ssaop.OpCtz8:
		return rewriteValue_OpCtz8(v)
	case ssaop.OpCtz8NonZero:
		v.Op = ssaop.OpCtz32
		return true
	case ssaop.OpCvt32Fto32:
		v.Op = ssaop.OpMIPSTRUNCFW
		return true
	case ssaop.OpCvt32Fto64F:
		v.Op = ssaop.OpMIPSMOVFD
		return true
	case ssaop.OpCvt32to32F:
		v.Op = ssaop.OpMIPSMOVWF
		return true
	case ssaop.OpCvt32to64F:
		v.Op = ssaop.OpMIPSMOVWD
		return true
	case ssaop.OpCvt64Fto32:
		v.Op = ssaop.OpMIPSTRUNCDW
		return true
	case ssaop.OpCvt64Fto32F:
		v.Op = ssaop.OpMIPSMOVDF
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
		v.Op = ssaop.OpMIPSDIVF
		return true
	case ssaop.OpDiv32u:
		return rewriteValue_OpDiv32u(v)
	case ssaop.OpDiv64F:
		v.Op = ssaop.OpMIPSDIVD
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
		return rewriteValue_OpEq32F(v)
	case ssaop.OpEq64F:
		return rewriteValue_OpEq64F(v)
	case ssaop.OpEq8:
		return rewriteValue_OpEq8(v)
	case ssaop.OpEqB:
		return rewriteValue_OpEqB(v)
	case ssaop.OpEqPtr:
		return rewriteValue_OpEqPtr(v)
	case ssaop.OpGetCallerPC:
		v.Op = ssaop.OpMIPSLoweredGetCallerPC
		return true
	case ssaop.OpGetCallerSP:
		v.Op = ssaop.OpMIPSLoweredGetCallerSP
		return true
	case ssaop.OpGetClosurePtr:
		v.Op = ssaop.OpMIPSLoweredGetClosurePtr
		return true
	case ssaop.OpHmul32:
		return rewriteValue_OpHmul32(v)
	case ssaop.OpHmul32u:
		return rewriteValue_OpHmul32u(v)
	case ssaop.OpInterCall:
		v.Op = ssaop.OpMIPSCALLinter
		return true
	case ssaop.OpIsInBounds:
		return rewriteValue_OpIsInBounds(v)
	case ssaop.OpIsNonNil:
		return rewriteValue_OpIsNonNil(v)
	case ssaop.OpIsSliceInBounds:
		return rewriteValue_OpIsSliceInBounds(v)
	case ssaop.OpLeq16:
		return rewriteValue_OpLeq16(v)
	case ssaop.OpLeq16U:
		return rewriteValue_OpLeq16U(v)
	case ssaop.OpLeq32:
		return rewriteValue_OpLeq32(v)
	case ssaop.OpLeq32F:
		return rewriteValue_OpLeq32F(v)
	case ssaop.OpLeq32U:
		return rewriteValue_OpLeq32U(v)
	case ssaop.OpLeq64F:
		return rewriteValue_OpLeq64F(v)
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
		return rewriteValue_OpLess32F(v)
	case ssaop.OpLess32U:
		return rewriteValue_OpLess32U(v)
	case ssaop.OpLess64F:
		return rewriteValue_OpLess64F(v)
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
	case ssaop.OpLsh8x16:
		return rewriteValue_OpLsh8x16(v)
	case ssaop.OpLsh8x32:
		return rewriteValue_OpLsh8x32(v)
	case ssaop.OpLsh8x64:
		return rewriteValue_OpLsh8x64(v)
	case ssaop.OpLsh8x8:
		return rewriteValue_OpLsh8x8(v)
	case ssaop.OpMIPSADD:
		return rewriteValue_OpMIPSADD(v)
	case ssaop.OpMIPSADDconst:
		return rewriteValue_OpMIPSADDconst(v)
	case ssaop.OpMIPSAND:
		return rewriteValue_OpMIPSAND(v)
	case ssaop.OpMIPSANDconst:
		return rewriteValue_OpMIPSANDconst(v)
	case ssaop.OpMIPSCMOVZ:
		return rewriteValue_OpMIPSCMOVZ(v)
	case ssaop.OpMIPSCMOVZzero:
		return rewriteValue_OpMIPSCMOVZzero(v)
	case ssaop.OpMIPSLoweredAtomicAdd:
		return rewriteValue_OpMIPSLoweredAtomicAdd(v)
	case ssaop.OpMIPSLoweredAtomicStore32:
		return rewriteValue_OpMIPSLoweredAtomicStore32(v)
	case ssaop.OpMIPSLoweredPanicBoundsRC:
		return rewriteValue_OpMIPSLoweredPanicBoundsRC(v)
	case ssaop.OpMIPSLoweredPanicBoundsRR:
		return rewriteValue_OpMIPSLoweredPanicBoundsRR(v)
	case ssaop.OpMIPSLoweredPanicExtendRC:
		return rewriteValue_OpMIPSLoweredPanicExtendRC(v)
	case ssaop.OpMIPSLoweredPanicExtendRR:
		return rewriteValue_OpMIPSLoweredPanicExtendRR(v)
	case ssaop.OpMIPSMOVBUload:
		return rewriteValue_OpMIPSMOVBUload(v)
	case ssaop.OpMIPSMOVBUreg:
		return rewriteValue_OpMIPSMOVBUreg(v)
	case ssaop.OpMIPSMOVBload:
		return rewriteValue_OpMIPSMOVBload(v)
	case ssaop.OpMIPSMOVBreg:
		return rewriteValue_OpMIPSMOVBreg(v)
	case ssaop.OpMIPSMOVBstore:
		return rewriteValue_OpMIPSMOVBstore(v)
	case ssaop.OpMIPSMOVBstorezero:
		return rewriteValue_OpMIPSMOVBstorezero(v)
	case ssaop.OpMIPSMOVDload:
		return rewriteValue_OpMIPSMOVDload(v)
	case ssaop.OpMIPSMOVDstore:
		return rewriteValue_OpMIPSMOVDstore(v)
	case ssaop.OpMIPSMOVFload:
		return rewriteValue_OpMIPSMOVFload(v)
	case ssaop.OpMIPSMOVFstore:
		return rewriteValue_OpMIPSMOVFstore(v)
	case ssaop.OpMIPSMOVHUload:
		return rewriteValue_OpMIPSMOVHUload(v)
	case ssaop.OpMIPSMOVHUreg:
		return rewriteValue_OpMIPSMOVHUreg(v)
	case ssaop.OpMIPSMOVHload:
		return rewriteValue_OpMIPSMOVHload(v)
	case ssaop.OpMIPSMOVHreg:
		return rewriteValue_OpMIPSMOVHreg(v)
	case ssaop.OpMIPSMOVHstore:
		return rewriteValue_OpMIPSMOVHstore(v)
	case ssaop.OpMIPSMOVHstorezero:
		return rewriteValue_OpMIPSMOVHstorezero(v)
	case ssaop.OpMIPSMOVWload:
		return rewriteValue_OpMIPSMOVWload(v)
	case ssaop.OpMIPSMOVWnop:
		return rewriteValue_OpMIPSMOVWnop(v)
	case ssaop.OpMIPSMOVWreg:
		return rewriteValue_OpMIPSMOVWreg(v)
	case ssaop.OpMIPSMOVWstore:
		return rewriteValue_OpMIPSMOVWstore(v)
	case ssaop.OpMIPSMOVWstorezero:
		return rewriteValue_OpMIPSMOVWstorezero(v)
	case ssaop.OpMIPSMUL:
		return rewriteValue_OpMIPSMUL(v)
	case ssaop.OpMIPSNEG:
		return rewriteValue_OpMIPSNEG(v)
	case ssaop.OpMIPSOR:
		return rewriteValue_OpMIPSOR(v)
	case ssaop.OpMIPSORconst:
		return rewriteValue_OpMIPSORconst(v)
	case ssaop.OpMIPSSGT:
		return rewriteValue_OpMIPSSGT(v)
	case ssaop.OpMIPSSGTU:
		return rewriteValue_OpMIPSSGTU(v)
	case ssaop.OpMIPSSGTUconst:
		return rewriteValue_OpMIPSSGTUconst(v)
	case ssaop.OpMIPSSGTUzero:
		return rewriteValue_OpMIPSSGTUzero(v)
	case ssaop.OpMIPSSGTconst:
		return rewriteValue_OpMIPSSGTconst(v)
	case ssaop.OpMIPSSGTzero:
		return rewriteValue_OpMIPSSGTzero(v)
	case ssaop.OpMIPSSLL:
		return rewriteValue_OpMIPSSLL(v)
	case ssaop.OpMIPSSLLconst:
		return rewriteValue_OpMIPSSLLconst(v)
	case ssaop.OpMIPSSRA:
		return rewriteValue_OpMIPSSRA(v)
	case ssaop.OpMIPSSRAconst:
		return rewriteValue_OpMIPSSRAconst(v)
	case ssaop.OpMIPSSRL:
		return rewriteValue_OpMIPSSRL(v)
	case ssaop.OpMIPSSRLconst:
		return rewriteValue_OpMIPSSRLconst(v)
	case ssaop.OpMIPSSUB:
		return rewriteValue_OpMIPSSUB(v)
	case ssaop.OpMIPSSUBconst:
		return rewriteValue_OpMIPSSUBconst(v)
	case ssaop.OpMIPSXOR:
		return rewriteValue_OpMIPSXOR(v)
	case ssaop.OpMIPSXORconst:
		return rewriteValue_OpMIPSXORconst(v)
	case ssaop.OpMod16:
		return rewriteValue_OpMod16(v)
	case ssaop.OpMod16u:
		return rewriteValue_OpMod16u(v)
	case ssaop.OpMod32:
		return rewriteValue_OpMod32(v)
	case ssaop.OpMod32u:
		return rewriteValue_OpMod32u(v)
	case ssaop.OpMod8:
		return rewriteValue_OpMod8(v)
	case ssaop.OpMod8u:
		return rewriteValue_OpMod8u(v)
	case ssaop.OpMove:
		return rewriteValue_OpMove(v)
	case ssaop.OpMul16:
		v.Op = ssaop.OpMIPSMUL
		return true
	case ssaop.OpMul32:
		v.Op = ssaop.OpMIPSMUL
		return true
	case ssaop.OpMul32F:
		v.Op = ssaop.OpMIPSMULF
		return true
	case ssaop.OpMul32uhilo:
		v.Op = ssaop.OpMIPSMULTU
		return true
	case ssaop.OpMul64F:
		v.Op = ssaop.OpMIPSMULD
		return true
	case ssaop.OpMul8:
		v.Op = ssaop.OpMIPSMUL
		return true
	case ssaop.OpNeg16:
		v.Op = ssaop.OpMIPSNEG
		return true
	case ssaop.OpNeg32:
		v.Op = ssaop.OpMIPSNEG
		return true
	case ssaop.OpNeg32F:
		v.Op = ssaop.OpMIPSNEGF
		return true
	case ssaop.OpNeg64F:
		v.Op = ssaop.OpMIPSNEGD
		return true
	case ssaop.OpNeg8:
		v.Op = ssaop.OpMIPSNEG
		return true
	case ssaop.OpNeq16:
		return rewriteValue_OpNeq16(v)
	case ssaop.OpNeq32:
		return rewriteValue_OpNeq32(v)
	case ssaop.OpNeq32F:
		return rewriteValue_OpNeq32F(v)
	case ssaop.OpNeq64F:
		return rewriteValue_OpNeq64F(v)
	case ssaop.OpNeq8:
		return rewriteValue_OpNeq8(v)
	case ssaop.OpNeqB:
		v.Op = ssaop.OpMIPSXOR
		return true
	case ssaop.OpNeqPtr:
		return rewriteValue_OpNeqPtr(v)
	case ssaop.OpNilCheck:
		v.Op = ssaop.OpMIPSLoweredNilCheck
		return true
	case ssaop.OpNot:
		return rewriteValue_OpNot(v)
	case ssaop.OpOffPtr:
		return rewriteValue_OpOffPtr(v)
	case ssaop.OpOr16:
		v.Op = ssaop.OpMIPSOR
		return true
	case ssaop.OpOr32:
		v.Op = ssaop.OpMIPSOR
		return true
	case ssaop.OpOr8:
		v.Op = ssaop.OpMIPSOR
		return true
	case ssaop.OpOrB:
		v.Op = ssaop.OpMIPSOR
		return true
	case ssaop.OpPanicBounds:
		v.Op = ssaop.OpMIPSLoweredPanicBoundsRR
		return true
	case ssaop.OpPanicExtend:
		v.Op = ssaop.OpMIPSLoweredPanicExtendRR
		return true
	case ssaop.OpPubBarrier:
		v.Op = ssaop.OpMIPSLoweredPubBarrier
		return true
	case ssaop.OpRotateLeft16:
		return rewriteValue_OpRotateLeft16(v)
	case ssaop.OpRotateLeft32:
		return rewriteValue_OpRotateLeft32(v)
	case ssaop.OpRotateLeft64:
		return rewriteValue_OpRotateLeft64(v)
	case ssaop.OpRotateLeft8:
		return rewriteValue_OpRotateLeft8(v)
	case ssaop.OpRound32F:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpRound64F:
		v.Op = ssaop.OpCopy
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
		v.Op = ssaop.OpMIPSMOVHreg
		return true
	case ssaop.OpSignExt8to16:
		v.Op = ssaop.OpMIPSMOVBreg
		return true
	case ssaop.OpSignExt8to32:
		v.Op = ssaop.OpMIPSMOVBreg
		return true
	case ssaop.OpSignmask:
		return rewriteValue_OpSignmask(v)
	case ssaop.OpSlicemask:
		return rewriteValue_OpSlicemask(v)
	case ssaop.OpSqrt:
		v.Op = ssaop.OpMIPSSQRTD
		return true
	case ssaop.OpSqrt32:
		v.Op = ssaop.OpMIPSSQRTF
		return true
	case ssaop.OpStaticCall:
		v.Op = ssaop.OpMIPSCALLstatic
		return true
	case ssaop.OpStore:
		return rewriteValue_OpStore(v)
	case ssaop.OpSub16:
		v.Op = ssaop.OpMIPSSUB
		return true
	case ssaop.OpSub32:
		v.Op = ssaop.OpMIPSSUB
		return true
	case ssaop.OpSub32F:
		v.Op = ssaop.OpMIPSSUBF
		return true
	case ssaop.OpSub32withcarry:
		return rewriteValue_OpSub32withcarry(v)
	case ssaop.OpSub64F:
		v.Op = ssaop.OpMIPSSUBD
		return true
	case ssaop.OpSub8:
		v.Op = ssaop.OpMIPSSUB
		return true
	case ssaop.OpSubPtr:
		v.Op = ssaop.OpMIPSSUB
		return true
	case ssaop.OpTailCall:
		v.Op = ssaop.OpMIPSCALLtail
		return true
	case ssaop.OpTailCallInter:
		v.Op = ssaop.OpMIPSCALLtailinter
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
	case ssaop.OpWB:
		v.Op = ssaop.OpMIPSLoweredWB
		return true
	case ssaop.OpXor16:
		v.Op = ssaop.OpMIPSXOR
		return true
	case ssaop.OpXor32:
		v.Op = ssaop.OpMIPSXOR
		return true
	case ssaop.OpXor8:
		v.Op = ssaop.OpMIPSXOR
		return true
	case ssaop.OpZero:
		return rewriteValue_OpZero(v)
	case ssaop.OpZeroExt16to32:
		v.Op = ssaop.OpMIPSMOVHUreg
		return true
	case ssaop.OpZeroExt8to16:
		v.Op = ssaop.OpMIPSMOVBUreg
		return true
	case ssaop.OpZeroExt8to32:
		v.Op = ssaop.OpMIPSMOVBUreg
		return true
	case ssaop.OpZeromask:
		return rewriteValue_OpZeromask(v)
	}
	return false
}
func rewriteValue_OpAdd32withcarry(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Add32withcarry <t> x y c)
	// result: (ADD c (ADD <t> x y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		c := v_2
		v.Reset(ssaop.OpMIPSADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSADD, t)
		v0.AddArg2(x, y)
		v.AddArg2(c, v0)
		return true
	}
}
func rewriteValue_OpAddr(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Addr {sym} base)
	// result: (MOVWaddr {sym} base)
	for {
		sym := ssa.AuxToSym(v.Aux)
		base := v_0
		v.Reset(ssaop.OpMIPSMOVWaddr)
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
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (AtomicAnd8 ptr val mem)
	// cond: !config.BigEndian
	// result: (LoweredAtomicAnd (AND <typ.UInt32Ptr> (MOVWconst [^3]) ptr) (OR <typ.UInt32> (SLL <typ.UInt32> (ZeroExt8to32 val) (SLLconst <typ.UInt32> [3] (ANDconst <typ.UInt32> [3] ptr))) (NOR (MOVWconst [0]) <typ.UInt32> (SLL <typ.UInt32> (MOVWconst [0xff]) (SLLconst <typ.UInt32> [3] (ANDconst <typ.UInt32> [3] ptr))))) mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		if !(!config.BigEndian) {
			break
		}
		v.Reset(ssaop.OpMIPSLoweredAtomicAnd)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSAND, typ.UInt32Ptr)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(^3)
		v0.AddArg2(v1, ptr)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSOR, typ.UInt32)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSSLL, typ.UInt32)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v4.AddArg(val)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPSSLLconst, typ.UInt32)
		v5.AuxInt = ssa.Int32ToAuxInt(3)
		v6 := b.NewValue0(v.Pos, ssaop.OpMIPSANDconst, typ.UInt32)
		v6.AuxInt = ssa.Int32ToAuxInt(3)
		v6.AddArg(ptr)
		v5.AddArg(v6)
		v3.AddArg2(v4, v5)
		v7 := b.NewValue0(v.Pos, ssaop.OpMIPSNOR, typ.UInt32)
		v8 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v8.AuxInt = ssa.Int32ToAuxInt(0)
		v9 := b.NewValue0(v.Pos, ssaop.OpMIPSSLL, typ.UInt32)
		v10 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v10.AuxInt = ssa.Int32ToAuxInt(0xff)
		v9.AddArg2(v10, v5)
		v7.AddArg2(v8, v9)
		v2.AddArg2(v3, v7)
		v.AddArg3(v0, v2, mem)
		return true
	}
	// match: (AtomicAnd8 ptr val mem)
	// cond: config.BigEndian
	// result: (LoweredAtomicAnd (AND <typ.UInt32Ptr> (MOVWconst [^3]) ptr) (OR <typ.UInt32> (SLL <typ.UInt32> (ZeroExt8to32 val) (SLLconst <typ.UInt32> [3] (ANDconst <typ.UInt32> [3] (XORconst <typ.UInt32> [3] ptr)))) (NOR (MOVWconst [0]) <typ.UInt32> (SLL <typ.UInt32> (MOVWconst [0xff]) (SLLconst <typ.UInt32> [3] (ANDconst <typ.UInt32> [3] (XORconst <typ.UInt32> [3] ptr)))))) mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		if !(config.BigEndian) {
			break
		}
		v.Reset(ssaop.OpMIPSLoweredAtomicAnd)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSAND, typ.UInt32Ptr)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(^3)
		v0.AddArg2(v1, ptr)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSOR, typ.UInt32)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSSLL, typ.UInt32)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v4.AddArg(val)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPSSLLconst, typ.UInt32)
		v5.AuxInt = ssa.Int32ToAuxInt(3)
		v6 := b.NewValue0(v.Pos, ssaop.OpMIPSANDconst, typ.UInt32)
		v6.AuxInt = ssa.Int32ToAuxInt(3)
		v7 := b.NewValue0(v.Pos, ssaop.OpMIPSXORconst, typ.UInt32)
		v7.AuxInt = ssa.Int32ToAuxInt(3)
		v7.AddArg(ptr)
		v6.AddArg(v7)
		v5.AddArg(v6)
		v3.AddArg2(v4, v5)
		v8 := b.NewValue0(v.Pos, ssaop.OpMIPSNOR, typ.UInt32)
		v9 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v9.AuxInt = ssa.Int32ToAuxInt(0)
		v10 := b.NewValue0(v.Pos, ssaop.OpMIPSSLL, typ.UInt32)
		v11 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v11.AuxInt = ssa.Int32ToAuxInt(0xff)
		v10.AddArg2(v11, v5)
		v8.AddArg2(v9, v10)
		v2.AddArg2(v3, v8)
		v.AddArg3(v0, v2, mem)
		return true
	}
	return false
}
func rewriteValue_OpAtomicOr8(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (AtomicOr8 ptr val mem)
	// cond: !config.BigEndian
	// result: (LoweredAtomicOr (AND <typ.UInt32Ptr> (MOVWconst [^3]) ptr) (SLL <typ.UInt32> (ZeroExt8to32 val) (SLLconst <typ.UInt32> [3] (ANDconst <typ.UInt32> [3] ptr))) mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		if !(!config.BigEndian) {
			break
		}
		v.Reset(ssaop.OpMIPSLoweredAtomicOr)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSAND, typ.UInt32Ptr)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(^3)
		v0.AddArg2(v1, ptr)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSSLL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v3.AddArg(val)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPSSLLconst, typ.UInt32)
		v4.AuxInt = ssa.Int32ToAuxInt(3)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPSANDconst, typ.UInt32)
		v5.AuxInt = ssa.Int32ToAuxInt(3)
		v5.AddArg(ptr)
		v4.AddArg(v5)
		v2.AddArg2(v3, v4)
		v.AddArg3(v0, v2, mem)
		return true
	}
	// match: (AtomicOr8 ptr val mem)
	// cond: config.BigEndian
	// result: (LoweredAtomicOr (AND <typ.UInt32Ptr> (MOVWconst [^3]) ptr) (SLL <typ.UInt32> (ZeroExt8to32 val) (SLLconst <typ.UInt32> [3] (ANDconst <typ.UInt32> [3] (XORconst <typ.UInt32> [3] ptr)))) mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		if !(config.BigEndian) {
			break
		}
		v.Reset(ssaop.OpMIPSLoweredAtomicOr)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSAND, typ.UInt32Ptr)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(^3)
		v0.AddArg2(v1, ptr)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSSLL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v3.AddArg(val)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPSSLLconst, typ.UInt32)
		v4.AuxInt = ssa.Int32ToAuxInt(3)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPSANDconst, typ.UInt32)
		v5.AuxInt = ssa.Int32ToAuxInt(3)
		v6 := b.NewValue0(v.Pos, ssaop.OpMIPSXORconst, typ.UInt32)
		v6.AuxInt = ssa.Int32ToAuxInt(3)
		v6.AddArg(ptr)
		v5.AddArg(v6)
		v4.AddArg(v5)
		v2.AddArg2(v3, v4)
		v.AddArg3(v0, v2, mem)
		return true
	}
	return false
}
func rewriteValue_OpAvg32u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Avg32u <t> x y)
	// result: (ADD (SRLconst <t> (SUB <t> x y) [1]) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSRLconst, t)
		v0.AuxInt = ssa.Int32ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSSUB, t)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpBitLen16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen16 x)
	// result: (BitLen32 (ZeroExt16to32 x))
	for {
		x := v_0
		v.Reset(ssaop.OpBitLen32)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
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
	// result: (SUB (MOVWconst [32]) (CLZ <t> x))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpMIPSSUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(32)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSCLZ, t)
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
	// result: (BitLen32 (ZeroExt8to32 x))
	for {
		x := v_0
		v.Reset(ssaop.OpBitLen32)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpCom16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com16 x)
	// result: (NOR (MOVWconst [0]) x)
	for {
		x := v_0
		v.Reset(ssaop.OpMIPSNOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValue_OpCom32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com32 x)
	// result: (NOR (MOVWconst [0]) x)
	for {
		x := v_0
		v.Reset(ssaop.OpMIPSNOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValue_OpCom8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com8 x)
	// result: (NOR (MOVWconst [0]) x)
	for {
		x := v_0
		v.Reset(ssaop.OpMIPSNOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValue_OpConst16(v *ssa.Value) bool {
	// match: (Const16 [val])
	// result: (MOVWconst [int32(val)])
	for {
		val := ssa.AuxIntToInt16(v.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(val))
		return true
	}
}
func rewriteValue_OpConst32(v *ssa.Value) bool {
	// match: (Const32 [val])
	// result: (MOVWconst [int32(val)])
	for {
		val := ssa.AuxIntToInt32(v.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(val))
		return true
	}
}
func rewriteValue_OpConst8(v *ssa.Value) bool {
	// match: (Const8 [val])
	// result: (MOVWconst [int32(val)])
	for {
		val := ssa.AuxIntToInt8(v.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(val))
		return true
	}
}
func rewriteValue_OpConstBool(v *ssa.Value) bool {
	// match: (ConstBool [t])
	// result: (MOVWconst [ssa.B2i32(t)])
	for {
		t := ssa.AuxIntToBool(v.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(ssa.B2i32(t))
		return true
	}
}
func rewriteValue_OpConstNil(v *ssa.Value) bool {
	// match: (ConstNil)
	// result: (MOVWconst [0])
	for {
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
}
func rewriteValue_OpCtz16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz16 x)
	// result: (Ctz32 (Or32 <typ.UInt32> x (MOVWconst [1<<16])))
	for {
		x := v_0
		v.Reset(ssaop.OpCtz32)
		v0 := b.NewValue0(v.Pos, ssaop.OpOr32, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(1 << 16)
		v0.AddArg2(x, v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpCtz32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz32 <t> x)
	// result: (SUB (MOVWconst [32]) (CLZ <t> (SUBconst <t> [1] (AND <t> x (NEG <t> x)))))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpMIPSSUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(32)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSCLZ, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSSUBconst, t)
		v2.AuxInt = ssa.Int32ToAuxInt(1)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSAND, t)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPSNEG, t)
		v4.AddArg(x)
		v3.AddArg2(x, v4)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpCtz8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz8 x)
	// result: (Ctz32 (Or32 <typ.UInt32> x (MOVWconst [1<<8])))
	for {
		x := v_0
		v.Reset(ssaop.OpCtz32)
		v0 := b.NewValue0(v.Pos, ssaop.OpOr32, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(1 << 8)
		v0.AddArg2(x, v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpDiv16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 x y)
	// result: (Select1 (DIV (SignExt16to32 x) (SignExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSDIV, types.NewTuple(typ.Int32, typ.Int32))
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpDiv16u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16u x y)
	// result: (Select1 (DIVU (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSDIVU, types.NewTuple(typ.UInt32, typ.UInt32))
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpDiv32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div32 x y)
	// result: (Select1 (DIV x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSDIV, types.NewTuple(typ.Int32, typ.Int32))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpDiv32u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div32u x y)
	// result: (Select1 (DIVU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSDIVU, types.NewTuple(typ.UInt32, typ.UInt32))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpDiv8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// result: (Select1 (DIV (SignExt8to32 x) (SignExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSDIV, types.NewTuple(typ.Int32, typ.Int32))
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpDiv8u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// result: (Select1 (DIVU (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSDIVU, types.NewTuple(typ.UInt32, typ.UInt32))
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpEq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq16 x y)
	// result: (SGTUconst [1] (XOR (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSGTUconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSXOR, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
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
	// result: (SGTUconst [1] (XOR x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSGTUconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSXOR, typ.UInt32)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpEq32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32F x y)
	// result: (FPFlagTrue (CMPEQF x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSFPFlagTrue)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSCMPEQF, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpEq64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64F x y)
	// result: (FPFlagTrue (CMPEQD x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSFPFlagTrue)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSCMPEQD, types.TypeFlags)
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
	// result: (SGTUconst [1] (XOR (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSGTUconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSXOR, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
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
	// result: (XORconst [1] (XOR <typ.Bool> x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSXORconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSXOR, typ.Bool)
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
	// result: (SGTUconst [1] (XOR x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSGTUconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSXOR, typ.UInt32)
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
	// result: (Select0 (MULT x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMULT, types.NewTuple(typ.Int32, typ.Int32))
		v0.AddArg2(x, y)
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
	// result: (Select0 (MULTU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMULTU, types.NewTuple(typ.UInt32, typ.UInt32))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpIsInBounds(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IsInBounds idx len)
	// result: (SGTU len idx)
	for {
		idx := v_0
		len := v_1
		v.Reset(ssaop.OpMIPSSGTU)
		v.AddArg2(len, idx)
		return true
	}
}
func rewriteValue_OpIsNonNil(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (IsNonNil ptr)
	// result: (SGTU ptr (MOVWconst [0]))
	for {
		ptr := v_0
		v.Reset(ssaop.OpMIPSSGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(0)
		v.AddArg2(ptr, v0)
		return true
	}
}
func rewriteValue_OpIsSliceInBounds(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (IsSliceInBounds idx len)
	// result: (XORconst [1] (SGTU idx len))
	for {
		idx := v_0
		len := v_1
		v.Reset(ssaop.OpMIPSXORconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTU, typ.Bool)
		v0.AddArg2(idx, len)
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
	// result: (XORconst [1] (SGT (SignExt16to32 x) (SignExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSXORconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSGT, typ.Bool)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
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
	// result: (XORconst [1] (SGTU (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSXORconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTU, typ.Bool)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
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
	// result: (XORconst [1] (SGT x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSXORconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSGT, typ.Bool)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32F x y)
	// result: (FPFlagTrue (CMPGEF y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSFPFlagTrue)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSCMPGEF, types.TypeFlags)
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
	// result: (XORconst [1] (SGTU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSXORconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTU, typ.Bool)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64F x y)
	// result: (FPFlagTrue (CMPGED y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSFPFlagTrue)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSCMPGED, types.TypeFlags)
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
	// result: (XORconst [1] (SGT (SignExt8to32 x) (SignExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSXORconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSGT, typ.Bool)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
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
	// result: (XORconst [1] (SGTU (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSXORconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTU, typ.Bool)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
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
	// result: (SGT (SignExt16to32 y) (SignExt16to32 x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSGT)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
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
	// result: (SGTU (ZeroExt16to32 y) (ZeroExt16to32 x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLess32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Less32 x y)
	// result: (SGT y x)
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSGT)
		v.AddArg2(y, x)
		return true
	}
}
func rewriteValue_OpLess32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32F x y)
	// result: (FPFlagTrue (CMPGTF y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSFPFlagTrue)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSCMPGTF, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLess32U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Less32U x y)
	// result: (SGTU y x)
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSGTU)
		v.AddArg2(y, x)
		return true
	}
}
func rewriteValue_OpLess64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64F x y)
	// result: (FPFlagTrue (CMPGTD y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSFPFlagTrue)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSCMPGTD, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLess8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8 x y)
	// result: (SGT (SignExt8to32 y) (SignExt8to32 x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSGT)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v1.AddArg(x)
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
	// result: (SGTU (ZeroExt8to32 y) (ZeroExt8to32 x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
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
		v.Reset(ssaop.OpMIPSMOVBUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (ssa.Is8BitInt(t) && t.IsSigned())
	// result: (MOVBload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is8BitInt(t) && t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVBload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (ssa.Is8BitInt(t) && !t.IsSigned())
	// result: (MOVBUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is8BitInt(t) && !t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVBUload)
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
		v.Reset(ssaop.OpMIPSMOVHload)
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
		v.Reset(ssaop.OpMIPSMOVHUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (ssa.Is32BitInt(t) ||ssa.IsPtr(t))
	// result: (MOVWload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is32BitInt(t) || ssa.IsPtr(t)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ssa.Is32BitFloat(t)
	// result: (MOVFload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is32BitFloat(t)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVFload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ssa.Is64BitFloat(t)
	// result: (MOVDload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is64BitFloat(t)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVDload)
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
	// result: (MOVWaddr {sym} (SPanchored base mem))
	for {
		t := v.Type
		sym := ssa.AuxToSym(v.Aux)
		base := v_0
		mem := v_1
		if !(t.Elem().HasPointers()) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWaddr)
		v.Aux = ssa.SymToAux(sym)
		v0 := b.NewValue0(v.Pos, ssaop.OpSPanchored, typ.Uintptr)
		v0.AddArg2(base, mem)
		v.AddArg(v0)
		return true
	}
	// match: (LocalAddr <t> {sym} base _)
	// cond: !t.Elem().HasPointers()
	// result: (MOVWaddr {sym} base)
	for {
		t := v.Type
		sym := ssa.AuxToSym(v.Aux)
		base := v_0
		if !(!t.Elem().HasPointers()) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWaddr)
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
	// result: (CMOVZ (SLL <t> x (ZeroExt16to32 y) ) (MOVWconst [0]) (SGTUconst [32] (ZeroExt16to32 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSCMOVZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSLL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v3.AuxInt = ssa.Int32ToAuxInt(32)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValue_OpLsh16x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x32 <t> x y)
	// result: (CMOVZ (SLL <t> x y) (MOVWconst [0]) (SGTUconst [32] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSCMOVZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v2.AuxInt = ssa.Int32ToAuxInt(32)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValue_OpLsh16x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Lsh16x64 x (Const64 [c]))
	// cond: uint32(c) < 16
	// result: (SLLconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) < 16) {
			break
		}
		v.Reset(ssaop.OpMIPSSLLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Lsh16x64 _ (Const64 [c]))
	// cond: uint32(c) >= 16
	// result: (MOVWconst [0])
	for {
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) >= 16) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
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
	// result: (CMOVZ (SLL <t> x (ZeroExt8to32 y) ) (MOVWconst [0]) (SGTUconst [32] (ZeroExt8to32 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSCMOVZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSLL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v3.AuxInt = ssa.Int32ToAuxInt(32)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValue_OpLsh32x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x16 <t> x y)
	// result: (CMOVZ (SLL <t> x (ZeroExt16to32 y) ) (MOVWconst [0]) (SGTUconst [32] (ZeroExt16to32 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSCMOVZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSLL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v3.AuxInt = ssa.Int32ToAuxInt(32)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValue_OpLsh32x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x32 <t> x y)
	// result: (CMOVZ (SLL <t> x y) (MOVWconst [0]) (SGTUconst [32] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSCMOVZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v2.AuxInt = ssa.Int32ToAuxInt(32)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValue_OpLsh32x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Lsh32x64 x (Const64 [c]))
	// cond: uint32(c) < 32
	// result: (SLLconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) < 32) {
			break
		}
		v.Reset(ssaop.OpMIPSSLLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Lsh32x64 _ (Const64 [c]))
	// cond: uint32(c) >= 32
	// result: (MOVWconst [0])
	for {
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) >= 32) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
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
	// result: (CMOVZ (SLL <t> x (ZeroExt8to32 y) ) (MOVWconst [0]) (SGTUconst [32] (ZeroExt8to32 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSCMOVZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSLL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v3.AuxInt = ssa.Int32ToAuxInt(32)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValue_OpLsh8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x16 <t> x y)
	// result: (CMOVZ (SLL <t> x (ZeroExt16to32 y) ) (MOVWconst [0]) (SGTUconst [32] (ZeroExt16to32 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSCMOVZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSLL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v3.AuxInt = ssa.Int32ToAuxInt(32)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValue_OpLsh8x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x32 <t> x y)
	// result: (CMOVZ (SLL <t> x y) (MOVWconst [0]) (SGTUconst [32] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSCMOVZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v2.AuxInt = ssa.Int32ToAuxInt(32)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValue_OpLsh8x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Lsh8x64 x (Const64 [c]))
	// cond: uint32(c) < 8
	// result: (SLLconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) < 8) {
			break
		}
		v.Reset(ssaop.OpMIPSSLLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Lsh8x64 _ (Const64 [c]))
	// cond: uint32(c) >= 8
	// result: (MOVWconst [0])
	for {
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) >= 8) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
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
	// result: (CMOVZ (SLL <t> x (ZeroExt8to32 y) ) (MOVWconst [0]) (SGTUconst [32] (ZeroExt8to32 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSCMOVZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSLL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v3.AuxInt = ssa.Int32ToAuxInt(32)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValue_OpMIPSADD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADD x (MOVWconst <t> [c]))
	// cond: !t.IsPtr()
	// result: (ADDconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMIPSMOVWconst {
				continue
			}
			t := v_1.Type
			c := ssa.AuxIntToInt32(v_1.AuxInt)
			if !(!t.IsPtr()) {
				continue
			}
			v.Reset(ssaop.OpMIPSADDconst)
			v.AuxInt = ssa.Int32ToAuxInt(c)
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
			if v_1.Op != ssaop.OpMIPSNEG {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpMIPSSUB)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpMIPSADDconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ADDconst [off1] (MOVWaddr [off2] {sym} ptr))
	// result: (MOVWaddr [off1+off2] {sym} ptr)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVWaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		v.Reset(ssaop.OpMIPSMOVWaddr)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg(ptr)
		return true
	}
	// match: (ADDconst [0] x)
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (ADDconst [c] (MOVWconst [d]))
	// result: (MOVWconst [int32(c+d)])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c + d))
		return true
	}
	// match: (ADDconst [c] (ADDconst [d] x))
	// result: (ADDconst [c+d] x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSADDconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpMIPSADDconst)
		v.AuxInt = ssa.Int32ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [c] (SUBconst [d] x))
	// result: (ADDconst [c-d] x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSSUBconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpMIPSADDconst)
		v.AuxInt = ssa.Int32ToAuxInt(c - d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpMIPSAND(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (AND x (MOVWconst [c]))
	// result: (ANDconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMIPSMOVWconst {
				continue
			}
			c := ssa.AuxIntToInt32(v_1.AuxInt)
			v.Reset(ssaop.OpMIPSANDconst)
			v.AuxInt = ssa.Int32ToAuxInt(c)
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
	// match: (AND (SGTUconst [1] x) (SGTUconst [1] y))
	// result: (SGTUconst [1] (OR <x.Type> x y))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpMIPSSGTUconst || ssa.AuxIntToInt32(v_0.AuxInt) != 1 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpMIPSSGTUconst || ssa.AuxIntToInt32(v_1.AuxInt) != 1 {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpMIPSSGTUconst)
			v.AuxInt = ssa.Int32ToAuxInt(1)
			v0 := b.NewValue0(v.Pos, ssaop.OpMIPSOR, x.Type)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpMIPSANDconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ANDconst [0] _)
	// result: (MOVWconst [0])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (ANDconst [-1] x)
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != -1 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (ANDconst [c] (MOVWconst [d]))
	// result: (MOVWconst [c&d])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & d)
		return true
	}
	// match: (ANDconst [c] (ANDconst [d] x))
	// result: (ANDconst [c&d] x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSANDconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpMIPSANDconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpMIPSCMOVZ(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVZ _ f (MOVWconst [0]))
	// result: f
	for {
		f := v_1
		if v_2.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_2.AuxInt) != 0 {
			break
		}
		v.CopyOf(f)
		return true
	}
	// match: (CMOVZ a _ (MOVWconst [c]))
	// cond: c!=0
	// result: a
	for {
		a := v_0
		if v_2.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_2.AuxInt)
		if !(c != 0) {
			break
		}
		v.CopyOf(a)
		return true
	}
	// match: (CMOVZ a (MOVWconst [0]) c)
	// result: (CMOVZzero a c)
	for {
		a := v_0
		if v_1.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_1.AuxInt) != 0 {
			break
		}
		c := v_2
		v.Reset(ssaop.OpMIPSCMOVZzero)
		v.AddArg2(a, c)
		return true
	}
	return false
}
func rewriteValue_OpMIPSCMOVZzero(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVZzero _ (MOVWconst [0]))
	// result: (MOVWconst [0])
	for {
		if v_1.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_1.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (CMOVZzero a (MOVWconst [c]))
	// cond: c!=0
	// result: a
	for {
		a := v_0
		if v_1.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		if !(c != 0) {
			break
		}
		v.CopyOf(a)
		return true
	}
	return false
}
func rewriteValue_OpMIPSLoweredAtomicAdd(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredAtomicAdd ptr (MOVWconst [c]) mem)
	// cond: ssa.Is16Bit(int64(c))
	// result: (LoweredAtomicAddconst [c] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is16Bit(int64(c))) {
			break
		}
		v.Reset(ssaop.OpMIPSLoweredAtomicAddconst)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpMIPSLoweredAtomicStore32(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredAtomicStore32 ptr (MOVWconst [0]) mem)
	// result: (LoweredAtomicStorezero ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.Reset(ssaop.OpMIPSLoweredAtomicStorezero)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpMIPSLoweredPanicBoundsRC(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRC [kind] {p} (MOVWconst [c]) mem)
	// result: (LoweredPanicBoundsCC [kind] {ssa.PanicBoundsCC{Cx:int64(c), Cy:p.C}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		p := ssa.AuxToPanicBoundsC(v.Aux)
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		mem := v_1
		v.Reset(ssaop.OpMIPSLoweredPanicBoundsCC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCCToAux(ssa.PanicBoundsCC{Cx: int64(c), Cy: p.C})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValue_OpMIPSLoweredPanicBoundsRR(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRR [kind] x (MOVWconst [c]) mem)
	// result: (LoweredPanicBoundsRC [kind] x {ssa.PanicBoundsC{C:int64(c)}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.OpMIPSLoweredPanicBoundsRC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCToAux(ssa.PanicBoundsC{C: int64(c)})
		v.AddArg2(x, mem)
		return true
	}
	// match: (LoweredPanicBoundsRR [kind] (MOVWconst [c]) y mem)
	// result: (LoweredPanicBoundsCR [kind] {ssa.PanicBoundsC{C:int64(c)}} y mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		y := v_1
		mem := v_2
		v.Reset(ssaop.OpMIPSLoweredPanicBoundsCR)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCToAux(ssa.PanicBoundsC{C: int64(c)})
		v.AddArg2(y, mem)
		return true
	}
	return false
}
func rewriteValue_OpMIPSLoweredPanicExtendRC(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicExtendRC [kind] {p} (MOVWconst [hi]) (MOVWconst [lo]) mem)
	// result: (LoweredPanicBoundsCC [kind] {ssa.PanicBoundsCC{Cx:int64(hi)<<32+int64(uint32(lo)), Cy:p.C}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		p := ssa.AuxToPanicBoundsC(v.Aux)
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		hi := ssa.AuxIntToInt32(v_0.AuxInt)
		if v_1.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		lo := ssa.AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.OpMIPSLoweredPanicBoundsCC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCCToAux(ssa.PanicBoundsCC{Cx: int64(hi)<<32 + int64(uint32(lo)), Cy: p.C})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValue_OpMIPSLoweredPanicExtendRR(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicExtendRR [kind] hi lo (MOVWconst [c]) mem)
	// result: (LoweredPanicExtendRC [kind] hi lo {ssa.PanicBoundsC{C:int64(c)}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		hi := v_0
		lo := v_1
		if v_2.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_2.AuxInt)
		mem := v_3
		v.Reset(ssaop.OpMIPSLoweredPanicExtendRC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCToAux(ssa.PanicBoundsC{C: int64(c)})
		v.AddArg3(hi, lo, mem)
		return true
	}
	// match: (LoweredPanicExtendRR [kind] (MOVWconst [hi]) (MOVWconst [lo]) y mem)
	// result: (LoweredPanicBoundsCR [kind] {ssa.PanicBoundsC{C:int64(hi)<<32 + int64(uint32(lo))}} y mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		hi := ssa.AuxIntToInt32(v_0.AuxInt)
		if v_1.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		lo := ssa.AuxIntToInt32(v_1.AuxInt)
		y := v_2
		mem := v_3
		v.Reset(ssaop.OpMIPSLoweredPanicBoundsCR)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCToAux(ssa.PanicBoundsC{C: int64(hi)<<32 + int64(uint32(lo))})
		v.AddArg2(y, mem)
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVBUload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBUload [off] {sym} ptr (MOVBstore [off] {sym} ptr x _))
	// result: (MOVBUreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVBstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpMIPSMOVBUreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUload [off1] {sym} x:(ADDconst [off2] ptr) mem)
	// cond: (ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1)
	// result: (MOVBUload [off1+off2] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		x := v_0
		if x.Op != ssaop.OpMIPSADDconst {
			break
		}
		off2 := ssa.AuxIntToInt32(x.AuxInt)
		ptr := x.Args[0]
		mem := v_1
		if !(ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVBUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2)
	// result: (MOVBUload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPSMOVWaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVBUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUload [off] {sym} ptr (MOVBstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: (MOVBUreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVBstore {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVBUreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVBUreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVBUreg x:(MOVBUload _ _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPSMOVBUload {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUreg _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPSMOVBUreg {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg <t> x:(MOVBload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && ssa.Clobber(x)
	// result: @x.Block (MOVBUload <t> [off] {sym} ptr mem)
	for {
		t := v.Type
		x := v_0
		if x.Op != ssaop.OpMIPSMOVBload {
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
		v0 := b.NewValue0(x.Pos, ssaop.OpMIPSMOVBUload, t)
		v.CopyOf(v0)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUreg (ANDconst [c] x))
	// result: (ANDconst [c&0xff] x)
	for {
		if v_0.Op != ssaop.OpMIPSANDconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpMIPSANDconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & 0xff)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (MOVWconst [c]))
	// result: (MOVWconst [int32(uint8(c))])
	for {
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(uint8(c)))
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVBload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBload [off] {sym} ptr (MOVBstore [off] {sym} ptr x _))
	// result: (MOVBreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVBstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpMIPSMOVBreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBload [off1] {sym} x:(ADDconst [off2] ptr) mem)
	// cond: (ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1)
	// result: (MOVBload [off1+off2] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		x := v_0
		if x.Op != ssaop.OpMIPSADDconst {
			break
		}
		off2 := ssa.AuxIntToInt32(x.AuxInt)
		ptr := x.Args[0]
		mem := v_1
		if !(ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVBload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2)
	// result: (MOVBload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPSMOVWaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVBload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off] {sym} ptr (MOVBstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: (MOVBreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVBstore {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVBreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVBreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVBreg x:(MOVBload _ _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPSMOVBload {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBreg _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPSMOVBreg {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg <t> x:(MOVBUload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && ssa.Clobber(x)
	// result: @x.Block (MOVBload <t> [off] {sym} ptr mem)
	for {
		t := v.Type
		x := v_0
		if x.Op != ssaop.OpMIPSMOVBUload {
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
		v0 := b.NewValue0(x.Pos, ssaop.OpMIPSMOVBload, t)
		v.CopyOf(v0)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBreg (ANDconst [c] x))
	// cond: c & 0x80 == 0
	// result: (ANDconst [c&0x7f] x)
	for {
		if v_0.Op != ssaop.OpMIPSANDconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c&0x80 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPSANDconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & 0x7f)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (MOVWconst [c]))
	// result: (MOVWconst [int32(int8(c))])
	for {
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(int8(c)))
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVBstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBstore [off1] {sym} x:(ADDconst [off2] ptr) val mem)
	// cond: (ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1)
	// result: (MOVBstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		x := v_0
		if x.Op != ssaop.OpMIPSADDconst {
			break
		}
		off2 := ssa.AuxIntToInt32(x.AuxInt)
		ptr := x.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstore [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2)
	// result: (MOVBstore [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPSMOVWaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVWconst [0]) mem)
	// result: (MOVBstorezero [off] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.Reset(ssaop.OpMIPSMOVBstorezero)
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
		if v_1.Op != ssaop.OpMIPSMOVBreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPSMOVBstore)
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
		if v_1.Op != ssaop.OpMIPSMOVBUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPSMOVBstore)
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
		if v_1.Op != ssaop.OpMIPSMOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPSMOVBstore)
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
		if v_1.Op != ssaop.OpMIPSMOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPSMOVBstore)
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
		if v_1.Op != ssaop.OpMIPSMOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPSMOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVBstorezero(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBstorezero [off1] {sym} x:(ADDconst [off2] ptr) mem)
	// cond: (ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1)
	// result: (MOVBstorezero [off1+off2] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		x := v_0
		if x.Op != ssaop.OpMIPSADDconst {
			break
		}
		off2 := ssa.AuxIntToInt32(x.AuxInt)
		ptr := x.Args[0]
		mem := v_1
		if !(ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVBstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstorezero [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2)
	// result: (MOVBstorezero [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPSMOVWaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVBstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVDload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDload [off1] {sym} x:(ADDconst [off2] ptr) mem)
	// cond: (ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1)
	// result: (MOVDload [off1+off2] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		x := v_0
		if x.Op != ssaop.OpMIPSADDconst {
			break
		}
		off2 := ssa.AuxIntToInt32(x.AuxInt)
		ptr := x.Args[0]
		mem := v_1
		if !(ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2)
	// result: (MOVDload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPSMOVWaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDload [off] {sym} ptr (MOVDstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: x
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVDstore {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVDstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDstore [off1] {sym} x:(ADDconst [off2] ptr) val mem)
	// cond: (ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1)
	// result: (MOVDstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		x := v_0
		if x.Op != ssaop.OpMIPSADDconst {
			break
		}
		off2 := ssa.AuxIntToInt32(x.AuxInt)
		ptr := x.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2)
	// result: (MOVDstore [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPSMOVWaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVFload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVFload [off] {sym} ptr (MOVWstore [off] {sym} ptr val _))
	// result: (MOVWgpfp val)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVWstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWgpfp)
		v.AddArg(val)
		return true
	}
	// match: (MOVFload [off1] {sym} x:(ADDconst [off2] ptr) mem)
	// cond: (ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1)
	// result: (MOVFload [off1+off2] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		x := v_0
		if x.Op != ssaop.OpMIPSADDconst {
			break
		}
		off2 := ssa.AuxIntToInt32(x.AuxInt)
		ptr := x.Args[0]
		mem := v_1
		if !(ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVFload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVFload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2)
	// result: (MOVFload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPSMOVWaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVFload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVFload [off] {sym} ptr (MOVFstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: x
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVFstore {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVFstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVFstore [off] {sym} ptr (MOVWgpfp val) mem)
	// result: (MOVWstore [off] {sym} ptr val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVWgpfp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPSMOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVFstore [off1] {sym} x:(ADDconst [off2] ptr) val mem)
	// cond: (ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1)
	// result: (MOVFstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		x := v_0
		if x.Op != ssaop.OpMIPSADDconst {
			break
		}
		off2 := ssa.AuxIntToInt32(x.AuxInt)
		ptr := x.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVFstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVFstore [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2)
	// result: (MOVFstore [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPSMOVWaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVFstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVHUload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHUload [off] {sym} ptr (MOVHstore [off] {sym} ptr x _))
	// result: (MOVHUreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVHstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpMIPSMOVHUreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUload [off1] {sym} x:(ADDconst [off2] ptr) mem)
	// cond: (ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1)
	// result: (MOVHUload [off1+off2] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		x := v_0
		if x.Op != ssaop.OpMIPSADDconst {
			break
		}
		off2 := ssa.AuxIntToInt32(x.AuxInt)
		ptr := x.Args[0]
		mem := v_1
		if !(ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVHUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2)
	// result: (MOVHUload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPSMOVWaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVHUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUload [off] {sym} ptr (MOVHstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: (MOVHUreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVHstore {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVHUreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVHUreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVHUreg x:(MOVBUload _ _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPSMOVBUload {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUload _ _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPSMOVHUload {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUreg _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPSMOVBUreg {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUreg _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPSMOVHUreg {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg <t> x:(MOVHload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && ssa.Clobber(x)
	// result: @x.Block (MOVHUload <t> [off] {sym} ptr mem)
	for {
		t := v.Type
		x := v_0
		if x.Op != ssaop.OpMIPSMOVHload {
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
		v0 := b.NewValue0(x.Pos, ssaop.OpMIPSMOVHUload, t)
		v.CopyOf(v0)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUreg (ANDconst [c] x))
	// result: (ANDconst [c&0xffff] x)
	for {
		if v_0.Op != ssaop.OpMIPSANDconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpMIPSANDconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & 0xffff)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (MOVWconst [c]))
	// result: (MOVWconst [int32(uint16(c))])
	for {
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(uint16(c)))
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVHload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHload [off] {sym} ptr (MOVHstore [off] {sym} ptr x _))
	// result: (MOVHreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVHstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpMIPSMOVHreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHload [off1] {sym} x:(ADDconst [off2] ptr) mem)
	// cond: (ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1)
	// result: (MOVHload [off1+off2] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		x := v_0
		if x.Op != ssaop.OpMIPSADDconst {
			break
		}
		off2 := ssa.AuxIntToInt32(x.AuxInt)
		ptr := x.Args[0]
		mem := v_1
		if !(ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVHload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2)
	// result: (MOVHload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPSMOVWaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVHload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHload [off] {sym} ptr (MOVHstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: (MOVHreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVHstore {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVHreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVHreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVHreg x:(MOVBload _ _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPSMOVBload {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUload _ _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPSMOVBUload {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHload _ _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPSMOVHload {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBreg _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPSMOVBreg {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUreg _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPSMOVBUreg {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHreg _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPSMOVHreg {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg <t> x:(MOVHUload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && ssa.Clobber(x)
	// result: @x.Block (MOVHload <t> [off] {sym} ptr mem)
	for {
		t := v.Type
		x := v_0
		if x.Op != ssaop.OpMIPSMOVHUload {
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
		v0 := b.NewValue0(x.Pos, ssaop.OpMIPSMOVHload, t)
		v.CopyOf(v0)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHreg (ANDconst [c] x))
	// cond: c & 0x8000 == 0
	// result: (ANDconst [c&0x7fff] x)
	for {
		if v_0.Op != ssaop.OpMIPSANDconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c&0x8000 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPSANDconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & 0x7fff)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (MOVWconst [c]))
	// result: (MOVWconst [int32(int16(c))])
	for {
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(int16(c)))
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVHstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstore [off1] {sym} x:(ADDconst [off2] ptr) val mem)
	// cond: (ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1)
	// result: (MOVHstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		x := v_0
		if x.Op != ssaop.OpMIPSADDconst {
			break
		}
		off2 := ssa.AuxIntToInt32(x.AuxInt)
		ptr := x.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstore [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2)
	// result: (MOVHstore [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPSMOVWaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVWconst [0]) mem)
	// result: (MOVHstorezero [off] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.Reset(ssaop.OpMIPSMOVHstorezero)
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
		if v_1.Op != ssaop.OpMIPSMOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPSMOVHstore)
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
		if v_1.Op != ssaop.OpMIPSMOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPSMOVHstore)
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
		if v_1.Op != ssaop.OpMIPSMOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPSMOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVHstorezero(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstorezero [off1] {sym} x:(ADDconst [off2] ptr) mem)
	// cond: (ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1)
	// result: (MOVHstorezero [off1+off2] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		x := v_0
		if x.Op != ssaop.OpMIPSADDconst {
			break
		}
		off2 := ssa.AuxIntToInt32(x.AuxInt)
		ptr := x.Args[0]
		mem := v_1
		if !(ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVHstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHstorezero [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2)
	// result: (MOVHstorezero [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPSMOVWaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVHstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVWload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWload [off] {sym} ptr (MOVFstore [off] {sym} ptr val _))
	// result: (MOVWfpgp val)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVFstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWfpgp)
		v.AddArg(val)
		return true
	}
	// match: (MOVWload [off] {sym} ptr (MOVWstore [off] {sym} ptr x _))
	// result: (MOVWreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVWstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWload [off1] {sym} x:(ADDconst [off2] ptr) mem)
	// cond: (ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1)
	// result: (MOVWload [off1+off2] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		x := v_0
		if x.Op != ssaop.OpMIPSADDconst {
			break
		}
		off2 := ssa.AuxIntToInt32(x.AuxInt)
		ptr := x.Args[0]
		mem := v_1
		if !(ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2)
	// result: (MOVWload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPSMOVWaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off] {sym} ptr (MOVWstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: x
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVWstore {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVWnop(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWnop (MOVWconst [c]))
	// result: (MOVWconst [c])
	for {
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVWreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWreg x)
	// cond: x.Uses == 1
	// result: (MOVWnop x)
	for {
		x := v_0
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWnop)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg (MOVWconst [c]))
	// result: (MOVWconst [c])
	for {
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVWstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstore [off] {sym} ptr (MOVWfpgp val) mem)
	// result: (MOVFstore [off] {sym} ptr val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVWfpgp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPSMOVFstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym} x:(ADDconst [off2] ptr) val mem)
	// cond: (ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1)
	// result: (MOVWstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		x := v_0
		if x.Op != ssaop.OpMIPSADDconst {
			break
		}
		off2 := ssa.AuxIntToInt32(x.AuxInt)
		ptr := x.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2)
	// result: (MOVWstore [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPSMOVWaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWconst [0]) mem)
	// result: (MOVWstorezero [off] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.Reset(ssaop.OpMIPSMOVWstorezero)
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
		if v_1.Op != ssaop.OpMIPSMOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPSMOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValue_OpMIPSMOVWstorezero(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstorezero [off1] {sym} x:(ADDconst [off2] ptr) mem)
	// cond: (ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1)
	// result: (MOVWstorezero [off1+off2] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		x := v_0
		if x.Op != ssaop.OpMIPSADDconst {
			break
		}
		off2 := ssa.AuxIntToInt32(x.AuxInt)
		ptr := x.Args[0]
		mem := v_1
		if !(ssa.Is16Bit(int64(off1+off2)) || x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstorezero [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2)
	// result: (MOVWstorezero [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPSMOVWaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpMIPSMUL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MUL (MOVWconst [0]) _ )
	// result: (MOVWconst [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				continue
			}
			v.Reset(ssaop.OpMIPSMOVWconst)
			v.AuxInt = ssa.Int32ToAuxInt(0)
			return true
		}
		break
	}
	// match: (MUL (MOVWconst [1]) x )
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 1 {
				continue
			}
			x := v_1
			v.CopyOf(x)
			return true
		}
		break
	}
	// match: (MUL (MOVWconst [-1]) x )
	// result: (NEG x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_0.AuxInt) != -1 {
				continue
			}
			x := v_1
			v.Reset(ssaop.OpMIPSNEG)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (MUL (MOVWconst [c]) x )
	// cond: ssa.IsPowerOfTwo(uint32(c))
	// result: (SLLconst [int32(ssa.Log32u(uint32(c)))] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpMIPSMOVWconst {
				continue
			}
			c := ssa.AuxIntToInt32(v_0.AuxInt)
			x := v_1
			if !(ssa.IsPowerOfTwo(uint32(c))) {
				continue
			}
			v.Reset(ssaop.OpMIPSSLLconst)
			v.AuxInt = ssa.Int32ToAuxInt(int32(ssa.Log32u(uint32(c))))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (MUL (MOVWconst [c]) (MOVWconst [d]))
	// result: (MOVWconst [c*d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpMIPSMOVWconst {
				continue
			}
			c := ssa.AuxIntToInt32(v_0.AuxInt)
			if v_1.Op != ssaop.OpMIPSMOVWconst {
				continue
			}
			d := ssa.AuxIntToInt32(v_1.AuxInt)
			v.Reset(ssaop.OpMIPSMOVWconst)
			v.AuxInt = ssa.Int32ToAuxInt(c * d)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpMIPSNEG(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (NEG (SUB x y))
	// result: (SUB y x)
	for {
		if v_0.Op != ssaop.OpMIPSSUB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpMIPSSUB)
		v.AddArg2(y, x)
		return true
	}
	// match: (NEG (NEG x))
	// result: x
	for {
		if v_0.Op != ssaop.OpMIPSNEG {
			break
		}
		x := v_0.Args[0]
		v.CopyOf(x)
		return true
	}
	// match: (NEG (MOVWconst [c]))
	// result: (MOVWconst [-c])
	for {
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(-c)
		return true
	}
	return false
}
func rewriteValue_OpMIPSOR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (OR x (MOVWconst [c]))
	// result: (ORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMIPSMOVWconst {
				continue
			}
			c := ssa.AuxIntToInt32(v_1.AuxInt)
			v.Reset(ssaop.OpMIPSORconst)
			v.AuxInt = ssa.Int32ToAuxInt(c)
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
	// match: (OR (SGTUzero x) (SGTUzero y))
	// result: (SGTUzero (OR <x.Type> x y))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpMIPSSGTUzero {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpMIPSSGTUzero {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpMIPSSGTUzero)
			v0 := b.NewValue0(v.Pos, ssaop.OpMIPSOR, x.Type)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpMIPSORconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ORconst [0] x)
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (ORconst [-1] _)
	// result: (MOVWconst [-1])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != -1 {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(-1)
		return true
	}
	// match: (ORconst [c] (MOVWconst [d]))
	// result: (MOVWconst [c|d])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(c | d)
		return true
	}
	// match: (ORconst [c] (ORconst [d] x))
	// result: (ORconst [c|d] x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSORconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpMIPSORconst)
		v.AuxInt = ssa.Int32ToAuxInt(c | d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpMIPSSGT(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SGT (MOVWconst [c]) x)
	// result: (SGTconst [c] x)
	for {
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpMIPSSGTconst)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SGT x (MOVWconst [0]))
	// result: (SGTzero x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_1.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpMIPSSGTzero)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpMIPSSGTU(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SGTU (MOVWconst [c]) x)
	// result: (SGTUconst [c] x)
	for {
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpMIPSSGTUconst)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SGTU x (MOVWconst [0]))
	// result: (SGTUzero x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_1.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpMIPSSGTUzero)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpMIPSSGTUconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SGTUconst [c] (MOVWconst [d]))
	// cond: uint32(c) > uint32(d)
	// result: (MOVWconst [1])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(uint32(c) > uint32(d)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SGTUconst [c] (MOVWconst [d]))
	// cond: uint32(c) <= uint32(d)
	// result: (MOVWconst [0])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(uint32(c) <= uint32(d)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SGTUconst [c] (MOVBUreg _))
	// cond: 0xff < uint32(c)
	// result: (MOVWconst [1])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVBUreg || !(0xff < uint32(c)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SGTUconst [c] (MOVHUreg _))
	// cond: 0xffff < uint32(c)
	// result: (MOVWconst [1])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVHUreg || !(0xffff < uint32(c)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SGTUconst [c] (ANDconst [m] _))
	// cond: uint32(m) < uint32(c)
	// result: (MOVWconst [1])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSANDconst {
			break
		}
		m := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(uint32(m) < uint32(c)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SGTUconst [c] (SRLconst _ [d]))
	// cond: uint32(d) <= 31 && 0xffffffff>>uint32(d) < uint32(c)
	// result: (MOVWconst [1])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSSRLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(uint32(d) <= 31 && 0xffffffff>>uint32(d) < uint32(c)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValue_OpMIPSSGTUzero(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SGTUzero (MOVWconst [d]))
	// cond: d != 0
	// result: (MOVWconst [1])
	for {
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SGTUzero (MOVWconst [d]))
	// cond: d == 0
	// result: (MOVWconst [0])
	for {
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(d == 0) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpMIPSSGTconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SGTconst [c] (MOVWconst [d]))
	// cond: c > d
	// result: (MOVWconst [1])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(c > d) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVWconst [d]))
	// cond: c <= d
	// result: (MOVWconst [0])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(c <= d) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVBreg _))
	// cond: 0x7f < c
	// result: (MOVWconst [1])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVBreg || !(0x7f < c) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVBreg _))
	// cond: c <= -0x80
	// result: (MOVWconst [0])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVBreg || !(c <= -0x80) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVBUreg _))
	// cond: 0xff < c
	// result: (MOVWconst [1])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVBUreg || !(0xff < c) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVBUreg _))
	// cond: c < 0
	// result: (MOVWconst [0])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVBUreg || !(c < 0) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVHreg _))
	// cond: 0x7fff < c
	// result: (MOVWconst [1])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVHreg || !(0x7fff < c) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVHreg _))
	// cond: c <= -0x8000
	// result: (MOVWconst [0])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVHreg || !(c <= -0x8000) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVHUreg _))
	// cond: 0xffff < c
	// result: (MOVWconst [1])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVHUreg || !(0xffff < c) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVHUreg _))
	// cond: c < 0
	// result: (MOVWconst [0])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVHUreg || !(c < 0) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (ANDconst [m] _))
	// cond: 0 <= m && m < c
	// result: (MOVWconst [1])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSANDconst {
			break
		}
		m := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(0 <= m && m < c) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (SRLconst _ [d]))
	// cond: 0 <= c && uint32(d) <= 31 && 0xffffffff>>uint32(d) < uint32(c)
	// result: (MOVWconst [1])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSSRLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(0 <= c && uint32(d) <= 31 && 0xffffffff>>uint32(d) < uint32(c)) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValue_OpMIPSSGTzero(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SGTzero (MOVWconst [d]))
	// cond: d > 0
	// result: (MOVWconst [1])
	for {
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(d > 0) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SGTzero (MOVWconst [d]))
	// cond: d <= 0
	// result: (MOVWconst [0])
	for {
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(d <= 0) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpMIPSSLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLL x (MOVWconst [c]))
	// result: (SLLconst x [c&31])
	for {
		x := v_0
		if v_1.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpMIPSSLLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & 31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpMIPSSLLconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SLLconst [c] (MOVWconst [d]))
	// result: (MOVWconst [d<<uint32(c)])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(d << uint32(c))
		return true
	}
	return false
}
func rewriteValue_OpMIPSSRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRA x (MOVWconst [c]))
	// result: (SRAconst x [c&31])
	for {
		x := v_0
		if v_1.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpMIPSSRAconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & 31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpMIPSSRAconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SRAconst [c] (MOVWconst [d]))
	// result: (MOVWconst [d>>uint32(c)])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(d >> uint32(c))
		return true
	}
	return false
}
func rewriteValue_OpMIPSSRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRL x (MOVWconst [c]))
	// result: (SRLconst x [c&31])
	for {
		x := v_0
		if v_1.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpMIPSSRLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & 31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpMIPSSRLconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SRLconst [c] (MOVWconst [d]))
	// result: (MOVWconst [int32(uint32(d)>>uint32(c))])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(uint32(d) >> uint32(c)))
		return true
	}
	return false
}
func rewriteValue_OpMIPSSUB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUB x (MOVWconst [c]))
	// result: (SUBconst [c] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpMIPSSUBconst)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SUB x (NEG y))
	// result: (ADD x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpMIPSNEG {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpMIPSADD)
		v.AddArg2(x, y)
		return true
	}
	// match: (SUB x x)
	// result: (MOVWconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SUB (MOVWconst [0]) x)
	// result: (NEG x)
	for {
		if v_0.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpMIPSNEG)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpMIPSSUBconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SUBconst [0] x)
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (SUBconst [c] (MOVWconst [d]))
	// result: (MOVWconst [d-c])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(d - c)
		return true
	}
	// match: (SUBconst [c] (SUBconst [d] x))
	// result: (ADDconst [-c-d] x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSSUBconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpMIPSADDconst)
		v.AuxInt = ssa.Int32ToAuxInt(-c - d)
		v.AddArg(x)
		return true
	}
	// match: (SUBconst [c] (ADDconst [d] x))
	// result: (ADDconst [-c+d] x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSADDconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpMIPSADDconst)
		v.AuxInt = ssa.Int32ToAuxInt(-c + d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpMIPSXOR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XOR x (MOVWconst [c]))
	// result: (XORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMIPSMOVWconst {
				continue
			}
			c := ssa.AuxIntToInt32(v_1.AuxInt)
			v.Reset(ssaop.OpMIPSXORconst)
			v.AuxInt = ssa.Int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (XOR x x)
	// result: (MOVWconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpMIPSXORconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (XORconst [0] x)
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (XORconst [c] (MOVWconst [d]))
	// result: (MOVWconst [c^d])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(c ^ d)
		return true
	}
	// match: (XORconst [c] (XORconst [d] x))
	// result: (XORconst [c^d] x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpMIPSXORconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpMIPSXORconst)
		v.AuxInt = ssa.Int32ToAuxInt(c ^ d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpMod16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16 x y)
	// result: (Select0 (DIV (SignExt16to32 x) (SignExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSDIV, types.NewTuple(typ.Int32, typ.Int32))
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpMod16u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16u x y)
	// result: (Select0 (DIVU (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSDIVU, types.NewTuple(typ.UInt32, typ.UInt32))
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpMod32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32 x y)
	// result: (Select0 (DIV x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSDIV, types.NewTuple(typ.Int32, typ.Int32))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpMod32u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32u x y)
	// result: (Select0 (DIVU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSDIVU, types.NewTuple(typ.UInt32, typ.UInt32))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpMod8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8 x y)
	// result: (Select0 (DIV (SignExt8to32 x) (SignExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSDIV, types.NewTuple(typ.Int32, typ.Int32))
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpMod8u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8u x y)
	// result: (Select0 (DIVU (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSDIVU, types.NewTuple(typ.UInt32, typ.UInt32))
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
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
	// result: (MOVBstore dst (MOVBUload src mem) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpMIPSMOVBstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBUload, typ.UInt8)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [2] {t} dst src mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore dst (MOVHUload src mem) mem)
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
		v.Reset(ssaop.OpMIPSMOVHstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHUload, typ.UInt16)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [2] dst src mem)
	// result: (MOVBstore [1] dst (MOVBUload [1] src mem) (MOVBstore dst (MOVBUload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpMIPSMOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBUload, typ.UInt8)
		v0.AuxInt = ssa.Int32ToAuxInt(1)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBUload, typ.UInt8)
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
		v.Reset(ssaop.OpMIPSMOVWstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWload, typ.UInt32)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [4] {t} dst src mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [2] dst (MOVHUload [2] src mem) (MOVHstore dst (MOVHUload src mem) mem))
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
		v.Reset(ssaop.OpMIPSMOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHUload, typ.UInt16)
		v0.AuxInt = ssa.Int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHUload, typ.UInt16)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [4] dst src mem)
	// result: (MOVBstore [3] dst (MOVBUload [3] src mem) (MOVBstore [2] dst (MOVBUload [2] src mem) (MOVBstore [1] dst (MOVBUload [1] src mem) (MOVBstore dst (MOVBUload src mem) mem))))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpMIPSMOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBUload, typ.UInt8)
		v0.AuxInt = ssa.Int32ToAuxInt(3)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(2)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBUload, typ.UInt8)
		v2.AuxInt = ssa.Int32ToAuxInt(2)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBstore, types.TypeMem)
		v3.AuxInt = ssa.Int32ToAuxInt(1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBUload, typ.UInt8)
		v4.AuxInt = ssa.Int32ToAuxInt(1)
		v4.AddArg2(src, mem)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBstore, types.TypeMem)
		v6 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBUload, typ.UInt8)
		v6.AddArg2(src, mem)
		v5.AddArg3(dst, v6, mem)
		v3.AddArg3(dst, v4, v5)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [3] dst src mem)
	// result: (MOVBstore [2] dst (MOVBUload [2] src mem) (MOVBstore [1] dst (MOVBUload [1] src mem) (MOVBstore dst (MOVBUload src mem) mem)))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 3 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpMIPSMOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBUload, typ.UInt8)
		v0.AuxInt = ssa.Int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(1)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBUload, typ.UInt8)
		v2.AuxInt = ssa.Int32ToAuxInt(1)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBstore, types.TypeMem)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBUload, typ.UInt8)
		v4.AddArg2(src, mem)
		v3.AddArg3(dst, v4, mem)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
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
		v.Reset(ssaop.OpMIPSMOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWload, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWload, typ.UInt32)
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
		v.Reset(ssaop.OpMIPSMOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHload, typ.Int16)
		v0.AuxInt = ssa.Int32ToAuxInt(6)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(4)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHload, typ.Int16)
		v2.AuxInt = ssa.Int32ToAuxInt(4)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHstore, types.TypeMem)
		v3.AuxInt = ssa.Int32ToAuxInt(2)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHload, typ.Int16)
		v4.AuxInt = ssa.Int32ToAuxInt(2)
		v4.AddArg2(src, mem)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHstore, types.TypeMem)
		v6 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHload, typ.Int16)
		v6.AddArg2(src, mem)
		v5.AddArg3(dst, v6, mem)
		v3.AddArg3(dst, v4, v5)
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
		v.Reset(ssaop.OpMIPSMOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHload, typ.Int16)
		v0.AuxInt = ssa.Int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(2)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHload, typ.Int16)
		v2.AuxInt = ssa.Int32ToAuxInt(2)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHstore, types.TypeMem)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHload, typ.Int16)
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
		if ssa.AuxIntToInt64(v.AuxInt) != 12 {
			break
		}
		t := ssa.AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWload, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(4)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWload, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(4)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWstore, types.TypeMem)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWload, typ.UInt32)
		v4.AddArg2(src, mem)
		v3.AddArg3(dst, v4, mem)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [16] {t} dst src mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore [12] dst (MOVWload [12] src mem) (MOVWstore [8] dst (MOVWload [8] src mem) (MOVWstore [4] dst (MOVWload [4] src mem) (MOVWstore dst (MOVWload src mem) mem))))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 16 {
			break
		}
		t := ssa.AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(12)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWload, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(12)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(8)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWload, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(8)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWstore, types.TypeMem)
		v3.AuxInt = ssa.Int32ToAuxInt(4)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWload, typ.UInt32)
		v4.AuxInt = ssa.Int32ToAuxInt(4)
		v4.AddArg2(src, mem)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWstore, types.TypeMem)
		v6 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWload, typ.UInt32)
		v6.AddArg2(src, mem)
		v5.AddArg3(dst, v6, mem)
		v3.AddArg3(dst, v4, v5)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] {t} dst src mem)
	// cond: (s > 16 && ssa.LogLargeCopyValue(v, s) || t.Alignment()%4 != 0)
	// result: (LoweredMove [int32(t.Alignment())] dst src (ADDconst <src.Type> src [int32(s-ssa.MoveSize(t.Alignment(), config))]) mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		t := ssa.AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 16 && ssa.LogLargeCopyValue(v, s) || t.Alignment()%4 != 0) {
			break
		}
		v.Reset(ssaop.OpMIPSLoweredMove)
		v.AuxInt = ssa.Int32ToAuxInt(int32(t.Alignment()))
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSADDconst, src.Type)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(s - ssa.MoveSize(t.Alignment(), config)))
		v0.AddArg(src)
		v.AddArg4(dst, src, v0, mem)
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
	// result: (SGTU (XOR (ZeroExt16to32 x) (ZeroExt16to32 y)) (MOVWconst [0]))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSXOR, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(0)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValue_OpNeq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq32 x y)
	// result: (SGTU (XOR x y) (MOVWconst [0]))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSXOR, typ.UInt32)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(0)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpNeq32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32F x y)
	// result: (FPFlagFalse (CMPEQF x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSFPFlagFalse)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSCMPEQF, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpNeq64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64F x y)
	// result: (FPFlagFalse (CMPEQD x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSFPFlagFalse)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSCMPEQD, types.TypeFlags)
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
	// result: (SGTU (XOR (ZeroExt8to32 x) (ZeroExt8to32 y)) (MOVWconst [0]))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSXOR, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(0)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValue_OpNeqPtr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NeqPtr x y)
	// result: (SGTU (XOR x y) (MOVWconst [0]))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSXOR, typ.UInt32)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(0)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpNot(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Not x)
	// result: (XORconst [1] x)
	for {
		x := v_0
		v.Reset(ssaop.OpMIPSXORconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValue_OpOffPtr(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (OffPtr [off] ptr:(SP))
	// result: (MOVWaddr [int32(off)] ptr)
	for {
		off := ssa.AuxIntToInt64(v.AuxInt)
		ptr := v_0
		if ptr.Op != ssaop.OpSP {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWaddr)
		v.AuxInt = ssa.Int32ToAuxInt(int32(off))
		v.AddArg(ptr)
		return true
	}
	// match: (OffPtr [off] ptr)
	// result: (ADDconst [int32(off)] ptr)
	for {
		off := ssa.AuxIntToInt64(v.AuxInt)
		ptr := v_0
		v.Reset(ssaop.OpMIPSADDconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(off))
		v.AddArg(ptr)
		return true
	}
}
func rewriteValue_OpRotateLeft16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft16 <t> x (MOVWconst [c]))
	// result: (Or16 (Lsh16x32 <t> x (MOVWconst [c&15])) (Rsh16Ux32 <t> x (MOVWconst [-c&15])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpOr16)
		v0 := b.NewValue0(v.Pos, ssaop.OpLsh16x32, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(c & 15)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh16Ux32, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(-c & 15)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpRotateLeft32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft32 <t> x (MOVWconst [c]))
	// result: (Or32 (Lsh32x32 <t> x (MOVWconst [c&31])) (Rsh32Ux32 <t> x (MOVWconst [-c&31])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpOr32)
		v0 := b.NewValue0(v.Pos, ssaop.OpLsh32x32, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(c & 31)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh32Ux32, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(-c & 31)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpRotateLeft64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft64 <t> x (MOVWconst [c]))
	// result: (Or64 (Lsh64x32 <t> x (MOVWconst [c&63])) (Rsh64Ux32 <t> x (MOVWconst [-c&63])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpOr64)
		v0 := b.NewValue0(v.Pos, ssaop.OpLsh64x32, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(c & 63)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh64Ux32, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(-c & 63)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpRotateLeft8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft8 <t> x (MOVWconst [c]))
	// result: (Or8 (Lsh8x32 <t> x (MOVWconst [c&7])) (Rsh8Ux32 <t> x (MOVWconst [-c&7])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpOr8)
		v0 := b.NewValue0(v.Pos, ssaop.OpLsh8x32, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(c & 7)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh8Ux32, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(-c & 7)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpRsh16Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux16 <t> x y)
	// result: (CMOVZ (SRL <t> (ZeroExt16to32 x) (ZeroExt16to32 y) ) (MOVWconst [0]) (SGTUconst [32] (ZeroExt16to32 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSCMOVZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v4.AuxInt = ssa.Int32ToAuxInt(32)
		v4.AddArg(v2)
		v.AddArg3(v0, v3, v4)
		return true
	}
}
func rewriteValue_OpRsh16Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux32 <t> x y)
	// result: (CMOVZ (SRL <t> (ZeroExt16to32 x) y) (MOVWconst [0]) (SGTUconst [32] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSCMOVZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v3.AuxInt = ssa.Int32ToAuxInt(32)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValue_OpRsh16Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 x (Const64 [c]))
	// cond: uint32(c) < 16
	// result: (SRLconst (SLLconst <typ.UInt32> x [16]) [int32(c+16)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) < 16) {
			break
		}
		v.Reset(ssaop.OpMIPSSRLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c + 16))
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSLLconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(16)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16Ux64 _ (Const64 [c]))
	// cond: uint32(c) >= 16
	// result: (MOVWconst [0])
	for {
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) >= 16) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
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
	// result: (CMOVZ (SRL <t> (ZeroExt16to32 x) (ZeroExt8to32 y) ) (MOVWconst [0]) (SGTUconst [32] (ZeroExt8to32 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSCMOVZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v4.AuxInt = ssa.Int32ToAuxInt(32)
		v4.AddArg(v2)
		v.AddArg3(v0, v3, v4)
		return true
	}
}
func rewriteValue_OpRsh16x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x16 x y)
	// result: (SRA (SignExt16to32 x) ( CMOVZ <typ.UInt32> (ZeroExt16to32 y) (MOVWconst [31]) (SGTUconst [32] (ZeroExt16to32 y))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSCMOVZ, typ.UInt32)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(31)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v4.AuxInt = ssa.Int32ToAuxInt(32)
		v4.AddArg(v2)
		v1.AddArg3(v2, v3, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpRsh16x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x32 x y)
	// result: (SRA (SignExt16to32 x) ( CMOVZ <typ.UInt32> y (MOVWconst [31]) (SGTUconst [32] y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSCMOVZ, typ.UInt32)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(31)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v3.AuxInt = ssa.Int32ToAuxInt(32)
		v3.AddArg(y)
		v1.AddArg3(y, v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpRsh16x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x64 x (Const64 [c]))
	// cond: uint32(c) < 16
	// result: (SRAconst (SLLconst <typ.UInt32> x [16]) [int32(c+16)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) < 16) {
			break
		}
		v.Reset(ssaop.OpMIPSSRAconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c + 16))
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSLLconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(16)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16x64 x (Const64 [c]))
	// cond: uint32(c) >= 16
	// result: (SRAconst (SLLconst <typ.UInt32> x [16]) [31])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) >= 16) {
			break
		}
		v.Reset(ssaop.OpMIPSSRAconst)
		v.AuxInt = ssa.Int32ToAuxInt(31)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSLLconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(16)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue_OpRsh16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x8 x y)
	// result: (SRA (SignExt16to32 x) ( CMOVZ <typ.UInt32> (ZeroExt8to32 y) (MOVWconst [31]) (SGTUconst [32] (ZeroExt8to32 y))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSCMOVZ, typ.UInt32)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(31)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v4.AuxInt = ssa.Int32ToAuxInt(32)
		v4.AddArg(v2)
		v1.AddArg3(v2, v3, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpRsh32Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux16 <t> x y)
	// result: (CMOVZ (SRL <t> x (ZeroExt16to32 y) ) (MOVWconst [0]) (SGTUconst [32] (ZeroExt16to32 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSCMOVZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v3.AuxInt = ssa.Int32ToAuxInt(32)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValue_OpRsh32Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux32 <t> x y)
	// result: (CMOVZ (SRL <t> x y) (MOVWconst [0]) (SGTUconst [32] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSCMOVZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v2.AuxInt = ssa.Int32ToAuxInt(32)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValue_OpRsh32Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Rsh32Ux64 x (Const64 [c]))
	// cond: uint32(c) < 32
	// result: (SRLconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) < 32) {
			break
		}
		v.Reset(ssaop.OpMIPSSRLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Rsh32Ux64 _ (Const64 [c]))
	// cond: uint32(c) >= 32
	// result: (MOVWconst [0])
	for {
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) >= 32) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
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
	// result: (CMOVZ (SRL <t> x (ZeroExt8to32 y) ) (MOVWconst [0]) (SGTUconst [32] (ZeroExt8to32 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSCMOVZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v3.AuxInt = ssa.Int32ToAuxInt(32)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValue_OpRsh32x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x16 x y)
	// result: (SRA x ( CMOVZ <typ.UInt32> (ZeroExt16to32 y) (MOVWconst [31]) (SGTUconst [32] (ZeroExt16to32 y))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSCMOVZ, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(31)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v3.AuxInt = ssa.Int32ToAuxInt(32)
		v3.AddArg(v1)
		v0.AddArg3(v1, v2, v3)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpRsh32x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x32 x y)
	// result: (SRA x ( CMOVZ <typ.UInt32> y (MOVWconst [31]) (SGTUconst [32] y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSCMOVZ, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(31)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v2.AuxInt = ssa.Int32ToAuxInt(32)
		v2.AddArg(y)
		v0.AddArg3(y, v1, v2)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpRsh32x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Rsh32x64 x (Const64 [c]))
	// cond: uint32(c) < 32
	// result: (SRAconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) < 32) {
			break
		}
		v.Reset(ssaop.OpMIPSSRAconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Rsh32x64 x (Const64 [c]))
	// cond: uint32(c) >= 32
	// result: (SRAconst x [31])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) >= 32) {
			break
		}
		v.Reset(ssaop.OpMIPSSRAconst)
		v.AuxInt = ssa.Int32ToAuxInt(31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpRsh32x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x8 x y)
	// result: (SRA x ( CMOVZ <typ.UInt32> (ZeroExt8to32 y) (MOVWconst [31]) (SGTUconst [32] (ZeroExt8to32 y))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSCMOVZ, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(31)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v3.AuxInt = ssa.Int32ToAuxInt(32)
		v3.AddArg(v1)
		v0.AddArg3(v1, v2, v3)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpRsh8Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux16 <t> x y)
	// result: (CMOVZ (SRL <t> (ZeroExt8to32 x) (ZeroExt16to32 y) ) (MOVWconst [0]) (SGTUconst [32] (ZeroExt16to32 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSCMOVZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v4.AuxInt = ssa.Int32ToAuxInt(32)
		v4.AddArg(v2)
		v.AddArg3(v0, v3, v4)
		return true
	}
}
func rewriteValue_OpRsh8Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux32 <t> x y)
	// result: (CMOVZ (SRL <t> (ZeroExt8to32 x) y) (MOVWconst [0]) (SGTUconst [32] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSCMOVZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v3.AuxInt = ssa.Int32ToAuxInt(32)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValue_OpRsh8Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 x (Const64 [c]))
	// cond: uint32(c) < 8
	// result: (SRLconst (SLLconst <typ.UInt32> x [24]) [int32(c+24)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) < 8) {
			break
		}
		v.Reset(ssaop.OpMIPSSRLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c + 24))
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSLLconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(24)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8Ux64 _ (Const64 [c]))
	// cond: uint32(c) >= 8
	// result: (MOVWconst [0])
	for {
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) >= 8) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
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
	// result: (CMOVZ (SRL <t> (ZeroExt8to32 x) (ZeroExt8to32 y) ) (MOVWconst [0]) (SGTUconst [32] (ZeroExt8to32 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSCMOVZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v4.AuxInt = ssa.Int32ToAuxInt(32)
		v4.AddArg(v2)
		v.AddArg3(v0, v3, v4)
		return true
	}
}
func rewriteValue_OpRsh8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x16 x y)
	// result: (SRA (SignExt16to32 x) ( CMOVZ <typ.UInt32> (ZeroExt16to32 y) (MOVWconst [31]) (SGTUconst [32] (ZeroExt16to32 y))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSCMOVZ, typ.UInt32)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(31)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v4.AuxInt = ssa.Int32ToAuxInt(32)
		v4.AddArg(v2)
		v1.AddArg3(v2, v3, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpRsh8x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x32 x y)
	// result: (SRA (SignExt16to32 x) ( CMOVZ <typ.UInt32> y (MOVWconst [31]) (SGTUconst [32] y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSCMOVZ, typ.UInt32)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v2.AuxInt = ssa.Int32ToAuxInt(31)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v3.AuxInt = ssa.Int32ToAuxInt(32)
		v3.AddArg(y)
		v1.AddArg3(y, v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpRsh8x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x64 x (Const64 [c]))
	// cond: uint32(c) < 8
	// result: (SRAconst (SLLconst <typ.UInt32> x [24]) [int32(c+24)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) < 8) {
			break
		}
		v.Reset(ssaop.OpMIPSSRAconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c + 24))
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSLLconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(24)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8x64 x (Const64 [c]))
	// cond: uint32(c) >= 8
	// result: (SRAconst (SLLconst <typ.UInt32> x [24]) [31])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) >= 8) {
			break
		}
		v.Reset(ssaop.OpMIPSSRAconst)
		v.AuxInt = ssa.Int32ToAuxInt(31)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSLLconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(24)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue_OpRsh8x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x8 x y)
	// result: (SRA (SignExt16to32 x) ( CMOVZ <typ.UInt32> (ZeroExt8to32 y) (MOVWconst [31]) (SGTUconst [32] (ZeroExt8to32 y))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPSSRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSCMOVZ, typ.UInt32)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v3.AuxInt = ssa.Int32ToAuxInt(31)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTUconst, typ.Bool)
		v4.AuxInt = ssa.Int32ToAuxInt(32)
		v4.AddArg(v2)
		v1.AddArg3(v2, v3, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpSelect0(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select0 (Add32carry <t> x y))
	// result: (ADD <t.FieldType(0)> x y)
	for {
		if v_0.Op != ssaop.OpAdd32carry {
			break
		}
		t := v_0.Type
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpMIPSADD)
		v.Type = t.FieldType(0)
		v.AddArg2(x, y)
		return true
	}
	// match: (Select0 (Add32carrywithcarry <t> x y c))
	// result: (ADD <t.FieldType(0)> c (ADD <t.FieldType(0)> x y))
	for {
		if v_0.Op != ssaop.OpAdd32carrywithcarry {
			break
		}
		t := v_0.Type
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.Reset(ssaop.OpMIPSADD)
		v.Type = t.FieldType(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSADD, t.FieldType(0))
		v0.AddArg2(x, y)
		v.AddArg2(c, v0)
		return true
	}
	// match: (Select0 (Sub32carry <t> x y))
	// result: (SUB <t.FieldType(0)> x y)
	for {
		if v_0.Op != ssaop.OpSub32carry {
			break
		}
		t := v_0.Type
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpMIPSSUB)
		v.Type = t.FieldType(0)
		v.AddArg2(x, y)
		return true
	}
	// match: (Select0 (MULTU (MOVWconst [0]) _ ))
	// result: (MOVWconst [0])
	for {
		if v_0.Op != ssaop.OpMIPSMULTU {
			break
		}
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_0_0.AuxInt) != 0 {
				continue
			}
			v.Reset(ssaop.OpMIPSMOVWconst)
			v.AuxInt = ssa.Int32ToAuxInt(0)
			return true
		}
		break
	}
	// match: (Select0 (MULTU (MOVWconst [1]) _ ))
	// result: (MOVWconst [0])
	for {
		if v_0.Op != ssaop.OpMIPSMULTU {
			break
		}
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_0_0.AuxInt) != 1 {
				continue
			}
			v.Reset(ssaop.OpMIPSMOVWconst)
			v.AuxInt = ssa.Int32ToAuxInt(0)
			return true
		}
		break
	}
	// match: (Select0 (MULTU (MOVWconst [-1]) x ))
	// result: (CMOVZ (ADDconst <x.Type> [-1] x) (MOVWconst [0]) x)
	for {
		if v_0.Op != ssaop.OpMIPSMULTU {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_0_0.AuxInt) != -1 {
				continue
			}
			x := v_0_1
			v.Reset(ssaop.OpMIPSCMOVZ)
			v0 := b.NewValue0(v.Pos, ssaop.OpMIPSADDconst, x.Type)
			v0.AuxInt = ssa.Int32ToAuxInt(-1)
			v0.AddArg(x)
			v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
			v1.AuxInt = ssa.Int32ToAuxInt(0)
			v.AddArg3(v0, v1, x)
			return true
		}
		break
	}
	// match: (Select0 (MULTU (MOVWconst [c]) x ))
	// cond: ssa.IsPowerOfTwo(uint32(c))
	// result: (SRLconst [int32(32-ssa.Log32u(uint32(c)))] x)
	for {
		if v_0.Op != ssaop.OpMIPSMULTU {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != ssaop.OpMIPSMOVWconst {
				continue
			}
			c := ssa.AuxIntToInt32(v_0_0.AuxInt)
			x := v_0_1
			if !(ssa.IsPowerOfTwo(uint32(c))) {
				continue
			}
			v.Reset(ssaop.OpMIPSSRLconst)
			v.AuxInt = ssa.Int32ToAuxInt(int32(32 - ssa.Log32u(uint32(c))))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Select0 (MULTU (MOVWconst [c]) (MOVWconst [d])))
	// result: (MOVWconst [int32((int64(uint32(c))*int64(uint32(d)))>>32)])
	for {
		if v_0.Op != ssaop.OpMIPSMULTU {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != ssaop.OpMIPSMOVWconst {
				continue
			}
			c := ssa.AuxIntToInt32(v_0_0.AuxInt)
			if v_0_1.Op != ssaop.OpMIPSMOVWconst {
				continue
			}
			d := ssa.AuxIntToInt32(v_0_1.AuxInt)
			v.Reset(ssaop.OpMIPSMOVWconst)
			v.AuxInt = ssa.Int32ToAuxInt(int32((int64(uint32(c)) * int64(uint32(d))) >> 32))
			return true
		}
		break
	}
	// match: (Select0 (DIV (MOVWconst [c]) (MOVWconst [d])))
	// cond: d != 0
	// result: (MOVWconst [c%d])
	for {
		if v_0.Op != ssaop.OpMIPSDIV {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0_0.AuxInt)
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(c % d)
		return true
	}
	// match: (Select0 (DIVU (MOVWconst [c]) (MOVWconst [d])))
	// cond: d != 0
	// result: (MOVWconst [int32(uint32(c)%uint32(d))])
	for {
		if v_0.Op != ssaop.OpMIPSDIVU {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0_0.AuxInt)
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(uint32(c) % uint32(d)))
		return true
	}
	return false
}
func rewriteValue_OpSelect1(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select1 (Add32carry <t> x y))
	// result: (SGTU <typ.Bool> x (ADD <t.FieldType(0)> x y))
	for {
		if v_0.Op != ssaop.OpAdd32carry {
			break
		}
		t := v_0.Type
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpMIPSSGTU)
		v.Type = typ.Bool
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSADD, t.FieldType(0))
		v0.AddArg2(x, y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Select1 (Add32carrywithcarry <t> x y c))
	// result: (OR <typ.Bool> (SGTU <typ.Bool> x xy:(ADD <t.FieldType(0)> x y)) (SGTU <typ.Bool> xy (ADD <t.FieldType(0)> c xy)))
	for {
		if v_0.Op != ssaop.OpAdd32carrywithcarry {
			break
		}
		t := v_0.Type
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.Reset(ssaop.OpMIPSOR)
		v.Type = typ.Bool
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTU, typ.Bool)
		xy := b.NewValue0(v.Pos, ssaop.OpMIPSADD, t.FieldType(0))
		xy.AddArg2(x, y)
		v0.AddArg2(x, xy)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSADD, t.FieldType(0))
		v3.AddArg2(c, xy)
		v2.AddArg2(xy, v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Select1 (Sub32carry <t> x y))
	// result: (SGTU <typ.Bool> (SUB <t.FieldType(0)> x y) x)
	for {
		if v_0.Op != ssaop.OpSub32carry {
			break
		}
		t := v_0.Type
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpMIPSSGTU)
		v.Type = typ.Bool
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSUB, t.FieldType(0))
		v0.AddArg2(x, y)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Select1 (MULTU (MOVWconst [0]) _ ))
	// result: (MOVWconst [0])
	for {
		if v_0.Op != ssaop.OpMIPSMULTU {
			break
		}
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_0_0.AuxInt) != 0 {
				continue
			}
			v.Reset(ssaop.OpMIPSMOVWconst)
			v.AuxInt = ssa.Int32ToAuxInt(0)
			return true
		}
		break
	}
	// match: (Select1 (MULTU (MOVWconst [1]) x ))
	// result: x
	for {
		if v_0.Op != ssaop.OpMIPSMULTU {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_0_0.AuxInt) != 1 {
				continue
			}
			x := v_0_1
			v.CopyOf(x)
			return true
		}
		break
	}
	// match: (Select1 (MULTU (MOVWconst [-1]) x ))
	// result: (NEG <x.Type> x)
	for {
		if v_0.Op != ssaop.OpMIPSMULTU {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != ssaop.OpMIPSMOVWconst || ssa.AuxIntToInt32(v_0_0.AuxInt) != -1 {
				continue
			}
			x := v_0_1
			v.Reset(ssaop.OpMIPSNEG)
			v.Type = x.Type
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Select1 (MULTU (MOVWconst [c]) x ))
	// cond: ssa.IsPowerOfTwo(uint32(c))
	// result: (SLLconst [int32(ssa.Log32u(uint32(c)))] x)
	for {
		if v_0.Op != ssaop.OpMIPSMULTU {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != ssaop.OpMIPSMOVWconst {
				continue
			}
			c := ssa.AuxIntToInt32(v_0_0.AuxInt)
			x := v_0_1
			if !(ssa.IsPowerOfTwo(uint32(c))) {
				continue
			}
			v.Reset(ssaop.OpMIPSSLLconst)
			v.AuxInt = ssa.Int32ToAuxInt(int32(ssa.Log32u(uint32(c))))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Select1 (MULTU (MOVWconst [c]) (MOVWconst [d])))
	// result: (MOVWconst [int32(uint32(c)*uint32(d))])
	for {
		if v_0.Op != ssaop.OpMIPSMULTU {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != ssaop.OpMIPSMOVWconst {
				continue
			}
			c := ssa.AuxIntToInt32(v_0_0.AuxInt)
			if v_0_1.Op != ssaop.OpMIPSMOVWconst {
				continue
			}
			d := ssa.AuxIntToInt32(v_0_1.AuxInt)
			v.Reset(ssaop.OpMIPSMOVWconst)
			v.AuxInt = ssa.Int32ToAuxInt(int32(uint32(c) * uint32(d)))
			return true
		}
		break
	}
	// match: (Select1 (DIV (MOVWconst [c]) (MOVWconst [d])))
	// cond: d != 0
	// result: (MOVWconst [c/d])
	for {
		if v_0.Op != ssaop.OpMIPSDIV {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0_0.AuxInt)
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(c / d)
		return true
	}
	// match: (Select1 (DIVU (MOVWconst [c]) (MOVWconst [d])))
	// cond: d != 0
	// result: (MOVWconst [int32(uint32(c)/uint32(d))])
	for {
		if v_0.Op != ssaop.OpMIPSDIVU {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0_0.AuxInt)
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpMIPSMOVWconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(uint32(c) / uint32(d)))
		return true
	}
	return false
}
func rewriteValue_OpSignmask(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Signmask x)
	// result: (SRAconst x [31])
	for {
		x := v_0
		v.Reset(ssaop.OpMIPSSRAconst)
		v.AuxInt = ssa.Int32ToAuxInt(31)
		v.AddArg(x)
		return true
	}
}
func rewriteValue_OpSlicemask(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Slicemask <t> x)
	// result: (SRAconst (NEG <t> x) [31])
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpMIPSSRAconst)
		v.AuxInt = ssa.Int32ToAuxInt(31)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSNEG, t)
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
		v.Reset(ssaop.OpMIPSMOVBstore)
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
		v.Reset(ssaop.OpMIPSMOVHstore)
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
		v.Reset(ssaop.OpMIPSMOVWstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && t.IsFloat()
	// result: (MOVFstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && t.IsFloat()) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVFstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && t.IsFloat()
	// result: (MOVDstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && t.IsFloat()) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpSub32withcarry(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Sub32withcarry <t> x y c)
	// result: (SUB (SUB <t> x y) c)
	for {
		t := v.Type
		x := v_0
		y := v_1
		c := v_2
		v.Reset(ssaop.OpMIPSSUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSUB, t)
		v0.AddArg2(x, y)
		v.AddArg2(v0, c)
		return true
	}
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
	// result: (MOVBstore ptr (MOVWconst [0]) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpMIPSMOVBstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [2] {t} ptr mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore ptr (MOVWconst [0]) mem)
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
		v.Reset(ssaop.OpMIPSMOVHstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [2] ptr mem)
	// result: (MOVBstore [1] ptr (MOVWconst [0]) (MOVBstore [0] ptr (MOVWconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpMIPSMOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(0)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [4] {t} ptr mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore ptr (MOVWconst [0]) mem)
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
		v.Reset(ssaop.OpMIPSMOVWstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [4] {t} ptr mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [2] ptr (MOVWconst [0]) (MOVHstore [0] ptr (MOVWconst [0]) mem))
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
		v.Reset(ssaop.OpMIPSMOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(0)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [4] ptr mem)
	// result: (MOVBstore [3] ptr (MOVWconst [0]) (MOVBstore [2] ptr (MOVWconst [0]) (MOVBstore [1] ptr (MOVWconst [0]) (MOVBstore [0] ptr (MOVWconst [0]) mem))))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpMIPSMOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(2)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBstore, types.TypeMem)
		v2.AuxInt = ssa.Int32ToAuxInt(1)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBstore, types.TypeMem)
		v3.AuxInt = ssa.Int32ToAuxInt(0)
		v3.AddArg3(ptr, v0, mem)
		v2.AddArg3(ptr, v0, v3)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [3] ptr mem)
	// result: (MOVBstore [2] ptr (MOVWconst [0]) (MOVBstore [1] ptr (MOVWconst [0]) (MOVBstore [0] ptr (MOVWconst [0]) mem)))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 3 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpMIPSMOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(1)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVBstore, types.TypeMem)
		v2.AuxInt = ssa.Int32ToAuxInt(0)
		v2.AddArg3(ptr, v0, mem)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [6] {t} ptr mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [4] ptr (MOVWconst [0]) (MOVHstore [2] ptr (MOVWconst [0]) (MOVHstore [0] ptr (MOVWconst [0]) mem)))
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
		v.Reset(ssaop.OpMIPSMOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(2)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVHstore, types.TypeMem)
		v2.AuxInt = ssa.Int32ToAuxInt(0)
		v2.AddArg3(ptr, v0, mem)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [8] {t} ptr mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore [4] ptr (MOVWconst [0]) (MOVWstore [0] ptr (MOVWconst [0]) mem))
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
		v.Reset(ssaop.OpMIPSMOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(0)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [12] {t} ptr mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore [8] ptr (MOVWconst [0]) (MOVWstore [4] ptr (MOVWconst [0]) (MOVWstore [0] ptr (MOVWconst [0]) mem)))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 12 {
			break
		}
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(4)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWstore, types.TypeMem)
		v2.AuxInt = ssa.Int32ToAuxInt(0)
		v2.AddArg3(ptr, v0, mem)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [16] {t} ptr mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore [12] ptr (MOVWconst [0]) (MOVWstore [8] ptr (MOVWconst [0]) (MOVWstore [4] ptr (MOVWconst [0]) (MOVWstore [0] ptr (MOVWconst [0]) mem))))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 16 {
			break
		}
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPSMOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(12)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(8)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWstore, types.TypeMem)
		v2.AuxInt = ssa.Int32ToAuxInt(4)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWstore, types.TypeMem)
		v3.AuxInt = ssa.Int32ToAuxInt(0)
		v3.AddArg3(ptr, v0, mem)
		v2.AddArg3(ptr, v0, v3)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [s] {t} ptr mem)
	// cond: (s > 16 || t.Alignment()%4 != 0)
	// result: (LoweredZero [int32(t.Alignment())] ptr (ADDconst <ptr.Type> ptr [int32(s-ssa.MoveSize(t.Alignment(), config))]) mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(s > 16 || t.Alignment()%4 != 0) {
			break
		}
		v.Reset(ssaop.OpMIPSLoweredZero)
		v.AuxInt = ssa.Int32ToAuxInt(int32(t.Alignment()))
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSADDconst, ptr.Type)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(s - ssa.MoveSize(t.Alignment(), config)))
		v0.AddArg(ptr)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValue_OpZeromask(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Zeromask x)
	// result: (NEG (SGTU x (MOVWconst [0])))
	for {
		x := v_0
		v.Reset(ssaop.OpMIPSNEG)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPSSGTU, typ.Bool)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPSMOVWconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(0)
		v0.AddArg2(x, v1)
		v.AddArg(v0)
		return true
	}
}
func RewriteBlock(b *ssa.Block) bool {
	switch b.Kind {
	case block.BlockMIPSEQ:
		// match: (EQ (FPFlagTrue cmp) yes no)
		// result: (FPF cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPSFPFlagTrue {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPSFPF, cmp)
			return true
		}
		// match: (EQ (FPFlagFalse cmp) yes no)
		// result: (FPT cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPSFPFlagFalse {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPSFPT, cmp)
			return true
		}
		// match: (EQ (XORconst [1] cmp:(SGT _ _)) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPSXORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPSSGT {
				break
			}
			b.ResetWithControl(block.BlockMIPSNE, cmp)
			return true
		}
		// match: (EQ (XORconst [1] cmp:(SGTU _ _)) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPSXORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPSSGTU {
				break
			}
			b.ResetWithControl(block.BlockMIPSNE, cmp)
			return true
		}
		// match: (EQ (XORconst [1] cmp:(SGTconst _)) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPSXORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPSSGTconst {
				break
			}
			b.ResetWithControl(block.BlockMIPSNE, cmp)
			return true
		}
		// match: (EQ (XORconst [1] cmp:(SGTUconst _)) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPSXORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPSSGTUconst {
				break
			}
			b.ResetWithControl(block.BlockMIPSNE, cmp)
			return true
		}
		// match: (EQ (XORconst [1] cmp:(SGTzero _)) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPSXORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPSSGTzero {
				break
			}
			b.ResetWithControl(block.BlockMIPSNE, cmp)
			return true
		}
		// match: (EQ (XORconst [1] cmp:(SGTUzero _)) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPSXORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPSSGTUzero {
				break
			}
			b.ResetWithControl(block.BlockMIPSNE, cmp)
			return true
		}
		// match: (EQ (SGTUconst [1] x) yes no)
		// result: (NE x yes no)
		for b.Controls[0].Op == ssaop.OpMIPSSGTUconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 1 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPSNE, x)
			return true
		}
		// match: (EQ (SGTUzero x) yes no)
		// result: (EQ x yes no)
		for b.Controls[0].Op == ssaop.OpMIPSSGTUzero {
			v_0 := b.Controls[0]
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPSEQ, x)
			return true
		}
		// match: (EQ (SGTconst [0] x) yes no)
		// result: (GEZ x yes no)
		for b.Controls[0].Op == ssaop.OpMIPSSGTconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPSGEZ, x)
			return true
		}
		// match: (EQ (SGTzero x) yes no)
		// result: (LEZ x yes no)
		for b.Controls[0].Op == ssaop.OpMIPSSGTzero {
			v_0 := b.Controls[0]
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPSLEZ, x)
			return true
		}
		// match: (EQ (MOVWconst [0]) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpMIPSMOVWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (EQ (MOVWconst [c]) yes no)
		// cond: c != 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpMIPSMOVWconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt32(v_0.AuxInt)
			if !(c != 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockMIPSGEZ:
		// match: (GEZ (MOVWconst [c]) yes no)
		// cond: c >= 0
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpMIPSMOVWconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt32(v_0.AuxInt)
			if !(c >= 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GEZ (MOVWconst [c]) yes no)
		// cond: c < 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpMIPSMOVWconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt32(v_0.AuxInt)
			if !(c < 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockMIPSGTZ:
		// match: (GTZ (MOVWconst [c]) yes no)
		// cond: c > 0
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpMIPSMOVWconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt32(v_0.AuxInt)
			if !(c > 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GTZ (MOVWconst [c]) yes no)
		// cond: c <= 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpMIPSMOVWconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt32(v_0.AuxInt)
			if !(c <= 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockIf:
		// match: (If cond yes no)
		// result: (NE cond yes no)
		for {
			cond := b.Controls[0]
			b.ResetWithControl(block.BlockMIPSNE, cond)
			return true
		}
	case block.BlockMIPSLEZ:
		// match: (LEZ (MOVWconst [c]) yes no)
		// cond: c <= 0
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpMIPSMOVWconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt32(v_0.AuxInt)
			if !(c <= 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LEZ (MOVWconst [c]) yes no)
		// cond: c > 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpMIPSMOVWconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt32(v_0.AuxInt)
			if !(c > 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockMIPSLTZ:
		// match: (LTZ (MOVWconst [c]) yes no)
		// cond: c < 0
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpMIPSMOVWconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt32(v_0.AuxInt)
			if !(c < 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LTZ (MOVWconst [c]) yes no)
		// cond: c >= 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpMIPSMOVWconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt32(v_0.AuxInt)
			if !(c >= 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockMIPSNE:
		// match: (NE (FPFlagTrue cmp) yes no)
		// result: (FPT cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPSFPFlagTrue {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPSFPT, cmp)
			return true
		}
		// match: (NE (FPFlagFalse cmp) yes no)
		// result: (FPF cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPSFPFlagFalse {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPSFPF, cmp)
			return true
		}
		// match: (NE (XORconst [1] cmp:(SGT _ _)) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPSXORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPSSGT {
				break
			}
			b.ResetWithControl(block.BlockMIPSEQ, cmp)
			return true
		}
		// match: (NE (XORconst [1] cmp:(SGTU _ _)) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPSXORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPSSGTU {
				break
			}
			b.ResetWithControl(block.BlockMIPSEQ, cmp)
			return true
		}
		// match: (NE (XORconst [1] cmp:(SGTconst _)) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPSXORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPSSGTconst {
				break
			}
			b.ResetWithControl(block.BlockMIPSEQ, cmp)
			return true
		}
		// match: (NE (XORconst [1] cmp:(SGTUconst _)) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPSXORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPSSGTUconst {
				break
			}
			b.ResetWithControl(block.BlockMIPSEQ, cmp)
			return true
		}
		// match: (NE (XORconst [1] cmp:(SGTzero _)) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPSXORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPSSGTzero {
				break
			}
			b.ResetWithControl(block.BlockMIPSEQ, cmp)
			return true
		}
		// match: (NE (XORconst [1] cmp:(SGTUzero _)) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPSXORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPSSGTUzero {
				break
			}
			b.ResetWithControl(block.BlockMIPSEQ, cmp)
			return true
		}
		// match: (NE (SGTUconst [1] x) yes no)
		// result: (EQ x yes no)
		for b.Controls[0].Op == ssaop.OpMIPSSGTUconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 1 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPSEQ, x)
			return true
		}
		// match: (NE (SGTUzero x) yes no)
		// result: (NE x yes no)
		for b.Controls[0].Op == ssaop.OpMIPSSGTUzero {
			v_0 := b.Controls[0]
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPSNE, x)
			return true
		}
		// match: (NE (SGTconst [0] x) yes no)
		// result: (LTZ x yes no)
		for b.Controls[0].Op == ssaop.OpMIPSSGTconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPSLTZ, x)
			return true
		}
		// match: (NE (SGTzero x) yes no)
		// result: (GTZ x yes no)
		for b.Controls[0].Op == ssaop.OpMIPSSGTzero {
			v_0 := b.Controls[0]
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPSGTZ, x)
			return true
		}
		// match: (NE (MOVWconst [0]) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpMIPSMOVWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (NE (MOVWconst [c]) yes no)
		// cond: c != 0
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpMIPSMOVWconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt32(v_0.AuxInt)
			if !(c != 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
	}
	return false
}
