// Code generated from gen/386.rules; DO NOT EDIT.
// generated with: cd gen; go run *.go

package ssa

import "math"
import "cmd/compile/internal/types"

func rewriteValue386(v *Value) bool {
	switch v.Op {
	case Op386ADCL:
		return rewriteValue386_Op386ADCL(v)
	case Op386ADDL:
		return rewriteValue386_Op386ADDL(v)
	case Op386ADDLcarry:
		return rewriteValue386_Op386ADDLcarry(v)
	case Op386ADDLconst:
		return rewriteValue386_Op386ADDLconst(v)
	case Op386ADDLconstmodify:
		return rewriteValue386_Op386ADDLconstmodify(v)
	case Op386ADDLload:
		return rewriteValue386_Op386ADDLload(v)
	case Op386ADDLmodify:
		return rewriteValue386_Op386ADDLmodify(v)
	case Op386ADDSD:
		return rewriteValue386_Op386ADDSD(v)
	case Op386ADDSDload:
		return rewriteValue386_Op386ADDSDload(v)
	case Op386ADDSS:
		return rewriteValue386_Op386ADDSS(v)
	case Op386ADDSSload:
		return rewriteValue386_Op386ADDSSload(v)
	case Op386ANDL:
		return rewriteValue386_Op386ANDL(v)
	case Op386ANDLconst:
		return rewriteValue386_Op386ANDLconst(v)
	case Op386ANDLconstmodify:
		return rewriteValue386_Op386ANDLconstmodify(v)
	case Op386ANDLload:
		return rewriteValue386_Op386ANDLload(v)
	case Op386ANDLmodify:
		return rewriteValue386_Op386ANDLmodify(v)
	case Op386CMPB:
		return rewriteValue386_Op386CMPB(v)
	case Op386CMPBconst:
		return rewriteValue386_Op386CMPBconst(v)
	case Op386CMPBload:
		return rewriteValue386_Op386CMPBload(v)
	case Op386CMPL:
		return rewriteValue386_Op386CMPL(v)
	case Op386CMPLconst:
		return rewriteValue386_Op386CMPLconst(v)
	case Op386CMPLload:
		return rewriteValue386_Op386CMPLload(v)
	case Op386CMPW:
		return rewriteValue386_Op386CMPW(v)
	case Op386CMPWconst:
		return rewriteValue386_Op386CMPWconst(v)
	case Op386CMPWload:
		return rewriteValue386_Op386CMPWload(v)
	case Op386DIVSD:
		return rewriteValue386_Op386DIVSD(v)
	case Op386DIVSDload:
		return rewriteValue386_Op386DIVSDload(v)
	case Op386DIVSS:
		return rewriteValue386_Op386DIVSS(v)
	case Op386DIVSSload:
		return rewriteValue386_Op386DIVSSload(v)
	case Op386LEAL:
		return rewriteValue386_Op386LEAL(v)
	case Op386LEAL1:
		return rewriteValue386_Op386LEAL1(v)
	case Op386LEAL2:
		return rewriteValue386_Op386LEAL2(v)
	case Op386LEAL4:
		return rewriteValue386_Op386LEAL4(v)
	case Op386LEAL8:
		return rewriteValue386_Op386LEAL8(v)
	case Op386MOVBLSX:
		return rewriteValue386_Op386MOVBLSX(v)
	case Op386MOVBLSXload:
		return rewriteValue386_Op386MOVBLSXload(v)
	case Op386MOVBLZX:
		return rewriteValue386_Op386MOVBLZX(v)
	case Op386MOVBload:
		return rewriteValue386_Op386MOVBload(v)
	case Op386MOVBstore:
		return rewriteValue386_Op386MOVBstore(v)
	case Op386MOVBstoreconst:
		return rewriteValue386_Op386MOVBstoreconst(v)
	case Op386MOVLload:
		return rewriteValue386_Op386MOVLload(v)
	case Op386MOVLstore:
		return rewriteValue386_Op386MOVLstore(v)
	case Op386MOVLstoreconst:
		return rewriteValue386_Op386MOVLstoreconst(v)
	case Op386MOVSDconst:
		return rewriteValue386_Op386MOVSDconst(v)
	case Op386MOVSDload:
		return rewriteValue386_Op386MOVSDload(v)
	case Op386MOVSDstore:
		return rewriteValue386_Op386MOVSDstore(v)
	case Op386MOVSSconst:
		return rewriteValue386_Op386MOVSSconst(v)
	case Op386MOVSSload:
		return rewriteValue386_Op386MOVSSload(v)
	case Op386MOVSSstore:
		return rewriteValue386_Op386MOVSSstore(v)
	case Op386MOVWLSX:
		return rewriteValue386_Op386MOVWLSX(v)
	case Op386MOVWLSXload:
		return rewriteValue386_Op386MOVWLSXload(v)
	case Op386MOVWLZX:
		return rewriteValue386_Op386MOVWLZX(v)
	case Op386MOVWload:
		return rewriteValue386_Op386MOVWload(v)
	case Op386MOVWstore:
		return rewriteValue386_Op386MOVWstore(v)
	case Op386MOVWstoreconst:
		return rewriteValue386_Op386MOVWstoreconst(v)
	case Op386MULL:
		return rewriteValue386_Op386MULL(v)
	case Op386MULLconst:
		return rewriteValue386_Op386MULLconst(v)
	case Op386MULLload:
		return rewriteValue386_Op386MULLload(v)
	case Op386MULSD:
		return rewriteValue386_Op386MULSD(v)
	case Op386MULSDload:
		return rewriteValue386_Op386MULSDload(v)
	case Op386MULSS:
		return rewriteValue386_Op386MULSS(v)
	case Op386MULSSload:
		return rewriteValue386_Op386MULSSload(v)
	case Op386NEGL:
		return rewriteValue386_Op386NEGL(v)
	case Op386NOTL:
		return rewriteValue386_Op386NOTL(v)
	case Op386ORL:
		return rewriteValue386_Op386ORL(v)
	case Op386ORLconst:
		return rewriteValue386_Op386ORLconst(v)
	case Op386ORLconstmodify:
		return rewriteValue386_Op386ORLconstmodify(v)
	case Op386ORLload:
		return rewriteValue386_Op386ORLload(v)
	case Op386ORLmodify:
		return rewriteValue386_Op386ORLmodify(v)
	case Op386ROLBconst:
		return rewriteValue386_Op386ROLBconst(v)
	case Op386ROLLconst:
		return rewriteValue386_Op386ROLLconst(v)
	case Op386ROLWconst:
		return rewriteValue386_Op386ROLWconst(v)
	case Op386SARB:
		return rewriteValue386_Op386SARB(v)
	case Op386SARBconst:
		return rewriteValue386_Op386SARBconst(v)
	case Op386SARL:
		return rewriteValue386_Op386SARL(v)
	case Op386SARLconst:
		return rewriteValue386_Op386SARLconst(v)
	case Op386SARW:
		return rewriteValue386_Op386SARW(v)
	case Op386SARWconst:
		return rewriteValue386_Op386SARWconst(v)
	case Op386SBBL:
		return rewriteValue386_Op386SBBL(v)
	case Op386SBBLcarrymask:
		return rewriteValue386_Op386SBBLcarrymask(v)
	case Op386SETA:
		return rewriteValue386_Op386SETA(v)
	case Op386SETAE:
		return rewriteValue386_Op386SETAE(v)
	case Op386SETB:
		return rewriteValue386_Op386SETB(v)
	case Op386SETBE:
		return rewriteValue386_Op386SETBE(v)
	case Op386SETEQ:
		return rewriteValue386_Op386SETEQ(v)
	case Op386SETG:
		return rewriteValue386_Op386SETG(v)
	case Op386SETGE:
		return rewriteValue386_Op386SETGE(v)
	case Op386SETL:
		return rewriteValue386_Op386SETL(v)
	case Op386SETLE:
		return rewriteValue386_Op386SETLE(v)
	case Op386SETNE:
		return rewriteValue386_Op386SETNE(v)
	case Op386SHLL:
		return rewriteValue386_Op386SHLL(v)
	case Op386SHLLconst:
		return rewriteValue386_Op386SHLLconst(v)
	case Op386SHRB:
		return rewriteValue386_Op386SHRB(v)
	case Op386SHRBconst:
		return rewriteValue386_Op386SHRBconst(v)
	case Op386SHRL:
		return rewriteValue386_Op386SHRL(v)
	case Op386SHRLconst:
		return rewriteValue386_Op386SHRLconst(v)
	case Op386SHRW:
		return rewriteValue386_Op386SHRW(v)
	case Op386SHRWconst:
		return rewriteValue386_Op386SHRWconst(v)
	case Op386SUBL:
		return rewriteValue386_Op386SUBL(v)
	case Op386SUBLcarry:
		return rewriteValue386_Op386SUBLcarry(v)
	case Op386SUBLconst:
		return rewriteValue386_Op386SUBLconst(v)
	case Op386SUBLload:
		return rewriteValue386_Op386SUBLload(v)
	case Op386SUBLmodify:
		return rewriteValue386_Op386SUBLmodify(v)
	case Op386SUBSD:
		return rewriteValue386_Op386SUBSD(v)
	case Op386SUBSDload:
		return rewriteValue386_Op386SUBSDload(v)
	case Op386SUBSS:
		return rewriteValue386_Op386SUBSS(v)
	case Op386SUBSSload:
		return rewriteValue386_Op386SUBSSload(v)
	case Op386XORL:
		return rewriteValue386_Op386XORL(v)
	case Op386XORLconst:
		return rewriteValue386_Op386XORLconst(v)
	case Op386XORLconstmodify:
		return rewriteValue386_Op386XORLconstmodify(v)
	case Op386XORLload:
		return rewriteValue386_Op386XORLload(v)
	case Op386XORLmodify:
		return rewriteValue386_Op386XORLmodify(v)
	case OpAdd16:
		v.Op = Op386ADDL
		return true
	case OpAdd32:
		v.Op = Op386ADDL
		return true
	case OpAdd32F:
		v.Op = Op386ADDSS
		return true
	case OpAdd32carry:
		v.Op = Op386ADDLcarry
		return true
	case OpAdd32withcarry:
		v.Op = Op386ADCL
		return true
	case OpAdd64F:
		v.Op = Op386ADDSD
		return true
	case OpAdd8:
		v.Op = Op386ADDL
		return true
	case OpAddPtr:
		v.Op = Op386ADDL
		return true
	case OpAddr:
		return rewriteValue386_OpAddr(v)
	case OpAnd16:
		v.Op = Op386ANDL
		return true
	case OpAnd32:
		v.Op = Op386ANDL
		return true
	case OpAnd8:
		v.Op = Op386ANDL
		return true
	case OpAndB:
		v.Op = Op386ANDL
		return true
	case OpAvg32u:
		v.Op = Op386AVGLU
		return true
	case OpBswap32:
		v.Op = Op386BSWAPL
		return true
	case OpClosureCall:
		v.Op = Op386CALLclosure
		return true
	case OpCom16:
		v.Op = Op386NOTL
		return true
	case OpCom32:
		v.Op = Op386NOTL
		return true
	case OpCom8:
		v.Op = Op386NOTL
		return true
	case OpConst16:
		return rewriteValue386_OpConst16(v)
	case OpConst32:
		v.Op = Op386MOVLconst
		return true
	case OpConst32F:
		v.Op = Op386MOVSSconst
		return true
	case OpConst64F:
		v.Op = Op386MOVSDconst
		return true
	case OpConst8:
		return rewriteValue386_OpConst8(v)
	case OpConstBool:
		return rewriteValue386_OpConstBool(v)
	case OpConstNil:
		return rewriteValue386_OpConstNil(v)
	case OpCtz16:
		return rewriteValue386_OpCtz16(v)
	case OpCtz16NonZero:
		v.Op = Op386BSFL
		return true
	case OpCvt32Fto32:
		v.Op = Op386CVTTSS2SL
		return true
	case OpCvt32Fto64F:
		v.Op = Op386CVTSS2SD
		return true
	case OpCvt32to32F:
		v.Op = Op386CVTSL2SS
		return true
	case OpCvt32to64F:
		v.Op = Op386CVTSL2SD
		return true
	case OpCvt64Fto32:
		v.Op = Op386CVTTSD2SL
		return true
	case OpCvt64Fto32F:
		v.Op = Op386CVTSD2SS
		return true
	case OpCvtBoolToUint8:
		v.Op = OpCopy
		return true
	case OpDiv16:
		v.Op = Op386DIVW
		return true
	case OpDiv16u:
		v.Op = Op386DIVWU
		return true
	case OpDiv32:
		v.Op = Op386DIVL
		return true
	case OpDiv32F:
		v.Op = Op386DIVSS
		return true
	case OpDiv32u:
		v.Op = Op386DIVLU
		return true
	case OpDiv64F:
		v.Op = Op386DIVSD
		return true
	case OpDiv8:
		return rewriteValue386_OpDiv8(v)
	case OpDiv8u:
		return rewriteValue386_OpDiv8u(v)
	case OpEq16:
		return rewriteValue386_OpEq16(v)
	case OpEq32:
		return rewriteValue386_OpEq32(v)
	case OpEq32F:
		return rewriteValue386_OpEq32F(v)
	case OpEq64F:
		return rewriteValue386_OpEq64F(v)
	case OpEq8:
		return rewriteValue386_OpEq8(v)
	case OpEqB:
		return rewriteValue386_OpEqB(v)
	case OpEqPtr:
		return rewriteValue386_OpEqPtr(v)
	case OpGetCallerPC:
		v.Op = Op386LoweredGetCallerPC
		return true
	case OpGetCallerSP:
		v.Op = Op386LoweredGetCallerSP
		return true
	case OpGetClosurePtr:
		v.Op = Op386LoweredGetClosurePtr
		return true
	case OpGetG:
		v.Op = Op386LoweredGetG
		return true
	case OpHmul32:
		v.Op = Op386HMULL
		return true
	case OpHmul32u:
		v.Op = Op386HMULLU
		return true
	case OpInterCall:
		v.Op = Op386CALLinter
		return true
	case OpIsInBounds:
		return rewriteValue386_OpIsInBounds(v)
	case OpIsNonNil:
		return rewriteValue386_OpIsNonNil(v)
	case OpIsSliceInBounds:
		return rewriteValue386_OpIsSliceInBounds(v)
	case OpLeq16:
		return rewriteValue386_OpLeq16(v)
	case OpLeq16U:
		return rewriteValue386_OpLeq16U(v)
	case OpLeq32:
		return rewriteValue386_OpLeq32(v)
	case OpLeq32F:
		return rewriteValue386_OpLeq32F(v)
	case OpLeq32U:
		return rewriteValue386_OpLeq32U(v)
	case OpLeq64F:
		return rewriteValue386_OpLeq64F(v)
	case OpLeq8:
		return rewriteValue386_OpLeq8(v)
	case OpLeq8U:
		return rewriteValue386_OpLeq8U(v)
	case OpLess16:
		return rewriteValue386_OpLess16(v)
	case OpLess16U:
		return rewriteValue386_OpLess16U(v)
	case OpLess32:
		return rewriteValue386_OpLess32(v)
	case OpLess32F:
		return rewriteValue386_OpLess32F(v)
	case OpLess32U:
		return rewriteValue386_OpLess32U(v)
	case OpLess64F:
		return rewriteValue386_OpLess64F(v)
	case OpLess8:
		return rewriteValue386_OpLess8(v)
	case OpLess8U:
		return rewriteValue386_OpLess8U(v)
	case OpLoad:
		return rewriteValue386_OpLoad(v)
	case OpLocalAddr:
		return rewriteValue386_OpLocalAddr(v)
	case OpLsh16x16:
		return rewriteValue386_OpLsh16x16(v)
	case OpLsh16x32:
		return rewriteValue386_OpLsh16x32(v)
	case OpLsh16x64:
		return rewriteValue386_OpLsh16x64(v)
	case OpLsh16x8:
		return rewriteValue386_OpLsh16x8(v)
	case OpLsh32x16:
		return rewriteValue386_OpLsh32x16(v)
	case OpLsh32x32:
		return rewriteValue386_OpLsh32x32(v)
	case OpLsh32x64:
		return rewriteValue386_OpLsh32x64(v)
	case OpLsh32x8:
		return rewriteValue386_OpLsh32x8(v)
	case OpLsh8x16:
		return rewriteValue386_OpLsh8x16(v)
	case OpLsh8x32:
		return rewriteValue386_OpLsh8x32(v)
	case OpLsh8x64:
		return rewriteValue386_OpLsh8x64(v)
	case OpLsh8x8:
		return rewriteValue386_OpLsh8x8(v)
	case OpMod16:
		v.Op = Op386MODW
		return true
	case OpMod16u:
		v.Op = Op386MODWU
		return true
	case OpMod32:
		v.Op = Op386MODL
		return true
	case OpMod32u:
		v.Op = Op386MODLU
		return true
	case OpMod8:
		return rewriteValue386_OpMod8(v)
	case OpMod8u:
		return rewriteValue386_OpMod8u(v)
	case OpMove:
		return rewriteValue386_OpMove(v)
	case OpMul16:
		v.Op = Op386MULL
		return true
	case OpMul32:
		v.Op = Op386MULL
		return true
	case OpMul32F:
		v.Op = Op386MULSS
		return true
	case OpMul32uhilo:
		v.Op = Op386MULLQU
		return true
	case OpMul64F:
		v.Op = Op386MULSD
		return true
	case OpMul8:
		v.Op = Op386MULL
		return true
	case OpNeg16:
		v.Op = Op386NEGL
		return true
	case OpNeg32:
		v.Op = Op386NEGL
		return true
	case OpNeg32F:
		return rewriteValue386_OpNeg32F(v)
	case OpNeg64F:
		return rewriteValue386_OpNeg64F(v)
	case OpNeg8:
		v.Op = Op386NEGL
		return true
	case OpNeq16:
		return rewriteValue386_OpNeq16(v)
	case OpNeq32:
		return rewriteValue386_OpNeq32(v)
	case OpNeq32F:
		return rewriteValue386_OpNeq32F(v)
	case OpNeq64F:
		return rewriteValue386_OpNeq64F(v)
	case OpNeq8:
		return rewriteValue386_OpNeq8(v)
	case OpNeqB:
		return rewriteValue386_OpNeqB(v)
	case OpNeqPtr:
		return rewriteValue386_OpNeqPtr(v)
	case OpNilCheck:
		v.Op = Op386LoweredNilCheck
		return true
	case OpNot:
		return rewriteValue386_OpNot(v)
	case OpOffPtr:
		return rewriteValue386_OpOffPtr(v)
	case OpOr16:
		v.Op = Op386ORL
		return true
	case OpOr32:
		v.Op = Op386ORL
		return true
	case OpOr8:
		v.Op = Op386ORL
		return true
	case OpOrB:
		v.Op = Op386ORL
		return true
	case OpPanicBounds:
		return rewriteValue386_OpPanicBounds(v)
	case OpPanicExtend:
		return rewriteValue386_OpPanicExtend(v)
	case OpRotateLeft16:
		return rewriteValue386_OpRotateLeft16(v)
	case OpRotateLeft32:
		return rewriteValue386_OpRotateLeft32(v)
	case OpRotateLeft8:
		return rewriteValue386_OpRotateLeft8(v)
	case OpRound32F:
		v.Op = OpCopy
		return true
	case OpRound64F:
		v.Op = OpCopy
		return true
	case OpRsh16Ux16:
		return rewriteValue386_OpRsh16Ux16(v)
	case OpRsh16Ux32:
		return rewriteValue386_OpRsh16Ux32(v)
	case OpRsh16Ux64:
		return rewriteValue386_OpRsh16Ux64(v)
	case OpRsh16Ux8:
		return rewriteValue386_OpRsh16Ux8(v)
	case OpRsh16x16:
		return rewriteValue386_OpRsh16x16(v)
	case OpRsh16x32:
		return rewriteValue386_OpRsh16x32(v)
	case OpRsh16x64:
		return rewriteValue386_OpRsh16x64(v)
	case OpRsh16x8:
		return rewriteValue386_OpRsh16x8(v)
	case OpRsh32Ux16:
		return rewriteValue386_OpRsh32Ux16(v)
	case OpRsh32Ux32:
		return rewriteValue386_OpRsh32Ux32(v)
	case OpRsh32Ux64:
		return rewriteValue386_OpRsh32Ux64(v)
	case OpRsh32Ux8:
		return rewriteValue386_OpRsh32Ux8(v)
	case OpRsh32x16:
		return rewriteValue386_OpRsh32x16(v)
	case OpRsh32x32:
		return rewriteValue386_OpRsh32x32(v)
	case OpRsh32x64:
		return rewriteValue386_OpRsh32x64(v)
	case OpRsh32x8:
		return rewriteValue386_OpRsh32x8(v)
	case OpRsh8Ux16:
		return rewriteValue386_OpRsh8Ux16(v)
	case OpRsh8Ux32:
		return rewriteValue386_OpRsh8Ux32(v)
	case OpRsh8Ux64:
		return rewriteValue386_OpRsh8Ux64(v)
	case OpRsh8Ux8:
		return rewriteValue386_OpRsh8Ux8(v)
	case OpRsh8x16:
		return rewriteValue386_OpRsh8x16(v)
	case OpRsh8x32:
		return rewriteValue386_OpRsh8x32(v)
	case OpRsh8x64:
		return rewriteValue386_OpRsh8x64(v)
	case OpRsh8x8:
		return rewriteValue386_OpRsh8x8(v)
	case OpSelect0:
		return rewriteValue386_OpSelect0(v)
	case OpSelect1:
		return rewriteValue386_OpSelect1(v)
	case OpSignExt16to32:
		v.Op = Op386MOVWLSX
		return true
	case OpSignExt8to16:
		v.Op = Op386MOVBLSX
		return true
	case OpSignExt8to32:
		v.Op = Op386MOVBLSX
		return true
	case OpSignmask:
		return rewriteValue386_OpSignmask(v)
	case OpSlicemask:
		return rewriteValue386_OpSlicemask(v)
	case OpSqrt:
		v.Op = Op386SQRTSD
		return true
	case OpStaticCall:
		v.Op = Op386CALLstatic
		return true
	case OpStore:
		return rewriteValue386_OpStore(v)
	case OpSub16:
		v.Op = Op386SUBL
		return true
	case OpSub32:
		v.Op = Op386SUBL
		return true
	case OpSub32F:
		v.Op = Op386SUBSS
		return true
	case OpSub32carry:
		v.Op = Op386SUBLcarry
		return true
	case OpSub32withcarry:
		v.Op = Op386SBBL
		return true
	case OpSub64F:
		v.Op = Op386SUBSD
		return true
	case OpSub8:
		v.Op = Op386SUBL
		return true
	case OpSubPtr:
		v.Op = Op386SUBL
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
	case OpWB:
		v.Op = Op386LoweredWB
		return true
	case OpXor16:
		v.Op = Op386XORL
		return true
	case OpXor32:
		v.Op = Op386XORL
		return true
	case OpXor8:
		v.Op = Op386XORL
		return true
	case OpZero:
		return rewriteValue386_OpZero(v)
	case OpZeroExt16to32:
		v.Op = Op386MOVWLZX
		return true
	case OpZeroExt8to16:
		v.Op = Op386MOVBLZX
		return true
	case OpZeroExt8to32:
		v.Op = Op386MOVBLZX
		return true
	case OpZeromask:
		return rewriteValue386_OpZeromask(v)
	}
	return false
}
func rewriteValue386_Op386ADCL(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADCL x (MOVLconst [c]) f)
	// result: (ADCLconst [c] x f)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386MOVLconst {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			f := v_2
			v.reset(Op386ADCLconst)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg2(x, f)
			return true
		}
		break
	}
	return false
}
func rewriteValue386_Op386ADDL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDL x (MOVLconst [c]))
	// result: (ADDLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386MOVLconst {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			v.reset(Op386ADDLconst)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ADDL (SHLLconst [c] x) (SHRLconst [d] x))
	// cond: d == 32-c
	// result: (ROLLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != Op386SHLLconst {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if v_1.Op != Op386SHRLconst {
				continue
			}
			d := auxIntToInt32(v_1.AuxInt)
			if x != v_1.Args[0] || !(d == 32-c) {
				continue
			}
			v.reset(Op386ROLLconst)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ADDL <t> (SHLLconst x [c]) (SHRWconst x [d]))
	// cond: c < 16 && d == int16(16-c) && t.Size() == 2
	// result: (ROLWconst x [int16(c)])
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != Op386SHLLconst {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if v_1.Op != Op386SHRWconst {
				continue
			}
			d := auxIntToInt16(v_1.AuxInt)
			if x != v_1.Args[0] || !(c < 16 && d == int16(16-c) && t.Size() == 2) {
				continue
			}
			v.reset(Op386ROLWconst)
			v.AuxInt = int16ToAuxInt(int16(c))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ADDL <t> (SHLLconst x [c]) (SHRBconst x [d]))
	// cond: c < 8 && d == int8(8-c) && t.Size() == 1
	// result: (ROLBconst x [int8(c)])
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != Op386SHLLconst {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if v_1.Op != Op386SHRBconst {
				continue
			}
			d := auxIntToInt8(v_1.AuxInt)
			if x != v_1.Args[0] || !(c < 8 && d == int8(8-c) && t.Size() == 1) {
				continue
			}
			v.reset(Op386ROLBconst)
			v.AuxInt = int8ToAuxInt(int8(c))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ADDL x (SHLLconst [3] y))
	// result: (LEAL8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386SHLLconst || auxIntToInt32(v_1.AuxInt) != 3 {
				continue
			}
			y := v_1.Args[0]
			v.reset(Op386LEAL8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADDL x (SHLLconst [2] y))
	// result: (LEAL4 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386SHLLconst || auxIntToInt32(v_1.AuxInt) != 2 {
				continue
			}
			y := v_1.Args[0]
			v.reset(Op386LEAL4)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADDL x (SHLLconst [1] y))
	// result: (LEAL2 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386SHLLconst || auxIntToInt32(v_1.AuxInt) != 1 {
				continue
			}
			y := v_1.Args[0]
			v.reset(Op386LEAL2)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADDL x (ADDL y y))
	// result: (LEAL2 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386ADDL {
				continue
			}
			y := v_1.Args[1]
			if y != v_1.Args[0] {
				continue
			}
			v.reset(Op386LEAL2)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADDL x (ADDL x y))
	// result: (LEAL2 y x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386ADDL {
				continue
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				y := v_1_1
				v.reset(Op386LEAL2)
				v.AddArg2(y, x)
				return true
			}
		}
		break
	}
	// match: (ADDL (ADDLconst [c] x) y)
	// result: (LEAL1 [c] x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != Op386ADDLconst {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			y := v_1
			v.reset(Op386LEAL1)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADDL x (LEAL [c] {s} y))
	// cond: x.Op != OpSB && y.Op != OpSB
	// result: (LEAL1 [c] {s} x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386LEAL {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			s := auxToSym(v_1.Aux)
			y := v_1.Args[0]
			if !(x.Op != OpSB && y.Op != OpSB) {
				continue
			}
			v.reset(Op386LEAL1)
			v.AuxInt = int32ToAuxInt(c)
			v.Aux = symToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADDL x l:(MOVLload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (ADDLload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != Op386MOVLload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(Op386ADDLload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	// match: (ADDL x (NEGL y))
	// result: (SUBL x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386NEGL {
				continue
			}
			y := v_1.Args[0]
			v.reset(Op386SUBL)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue386_Op386ADDLcarry(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDLcarry x (MOVLconst [c]))
	// result: (ADDLconstcarry [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386MOVLconst {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			v.reset(Op386ADDLconstcarry)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValue386_Op386ADDLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ADDLconst [c] (ADDL x y))
	// result: (LEAL1 [c] x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386ADDL {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(Op386LEAL1)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (ADDLconst [c] (LEAL [d] {s} x))
	// cond: is32Bit(int64(c)+int64(d))
	// result: (LEAL [c+d] {s} x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386LEAL {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		s := auxToSym(v_0.Aux)
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(Op386LEAL)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg(x)
		return true
	}
	// match: (ADDLconst [c] x:(SP))
	// result: (LEAL [c] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if x.Op != OpSP {
			break
		}
		v.reset(Op386LEAL)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (ADDLconst [c] (LEAL1 [d] {s} x y))
	// cond: is32Bit(int64(c)+int64(d))
	// result: (LEAL1 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386LEAL1 {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		s := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(Op386LEAL1)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (ADDLconst [c] (LEAL2 [d] {s} x y))
	// cond: is32Bit(int64(c)+int64(d))
	// result: (LEAL2 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386LEAL2 {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		s := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(Op386LEAL2)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (ADDLconst [c] (LEAL4 [d] {s} x y))
	// cond: is32Bit(int64(c)+int64(d))
	// result: (LEAL4 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386LEAL4 {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		s := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(Op386LEAL4)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (ADDLconst [c] (LEAL8 [d] {s} x y))
	// cond: is32Bit(int64(c)+int64(d))
	// result: (LEAL8 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386LEAL8 {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		s := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(Op386LEAL8)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (ADDLconst [c] x)
	// cond: c==0
	// result: x
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(c == 0) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (ADDLconst [c] (MOVLconst [d]))
	// result: (MOVLconst [c+d])
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(c + d)
		return true
	}
	// match: (ADDLconst [c] (ADDLconst [d] x))
	// result: (ADDLconst [c+d] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386ADDLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(Op386ADDLconst)
		v.AuxInt = int32ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386ADDLconstmodify(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ADDLconstmodify [valoff1] {sym} (ADDLconst [off2] base) mem)
	// cond: valoff1.canAdd32(off2)
	// result: (ADDLconstmodify [valoff1.addOffset32(off2)] {sym} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(valoff1.canAdd32(off2)) {
			break
		}
		v.reset(Op386ADDLconstmodify)
		v.AuxInt = valAndOffToAuxInt(valoff1.addOffset32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (ADDLconstmodify [valoff1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: valoff1.canAdd32(off2) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (ADDLconstmodify [valoff1.addOffset32(off2)] {mergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(valoff1.canAdd32(off2) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386ADDLconstmodify)
		v.AuxInt = valAndOffToAuxInt(valoff1.addOffset32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ADDLload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ADDLload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ADDLload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386ADDLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ADDLload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (ADDLload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386ADDLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ADDLmodify(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ADDLmodify [off1] {sym} (ADDLconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ADDLmodify [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386ADDLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (ADDLmodify [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (ADDLmodify [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386ADDLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ADDSD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDSD x l:(MOVSDload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (ADDSDload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != Op386MOVSDload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(Op386ADDSDload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValue386_Op386ADDSDload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ADDSDload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ADDSDload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386ADDSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ADDSDload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (ADDSDload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386ADDSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ADDSS(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDSS x l:(MOVSSload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (ADDSSload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != Op386MOVSSload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(Op386ADDSSload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValue386_Op386ADDSSload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ADDSSload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ADDSSload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386ADDSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ADDSSload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (ADDSSload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386ADDSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ANDL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ANDL x (MOVLconst [c]))
	// result: (ANDLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386MOVLconst {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			v.reset(Op386ANDLconst)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ANDL x l:(MOVLload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (ANDLload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != Op386MOVLload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(Op386ANDLload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	// match: (ANDL x x)
	// result: x
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValue386_Op386ANDLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ANDLconst [c] (ANDLconst [d] x))
	// result: (ANDLconst [c & d] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386ANDLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(Op386ANDLconst)
		v.AuxInt = int32ToAuxInt(c & d)
		v.AddArg(x)
		return true
	}
	// match: (ANDLconst [c] _)
	// cond: c==0
	// result: (MOVLconst [0])
	for {
		c := auxIntToInt32(v.AuxInt)
		if !(c == 0) {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (ANDLconst [c] x)
	// cond: c==-1
	// result: x
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(c == -1) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (ANDLconst [c] (MOVLconst [d]))
	// result: (MOVLconst [c&d])
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(c & d)
		return true
	}
	return false
}
func rewriteValue386_Op386ANDLconstmodify(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ANDLconstmodify [valoff1] {sym} (ADDLconst [off2] base) mem)
	// cond: valoff1.canAdd32(off2)
	// result: (ANDLconstmodify [valoff1.addOffset32(off2)] {sym} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(valoff1.canAdd32(off2)) {
			break
		}
		v.reset(Op386ANDLconstmodify)
		v.AuxInt = valAndOffToAuxInt(valoff1.addOffset32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (ANDLconstmodify [valoff1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: valoff1.canAdd32(off2) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (ANDLconstmodify [valoff1.addOffset32(off2)] {mergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(valoff1.canAdd32(off2) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386ANDLconstmodify)
		v.AuxInt = valAndOffToAuxInt(valoff1.addOffset32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ANDLload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ANDLload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ANDLload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386ANDLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ANDLload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (ANDLload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386ANDLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ANDLmodify(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ANDLmodify [off1] {sym} (ADDLconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ANDLmodify [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386ANDLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (ANDLmodify [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (ANDLmodify [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386ANDLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386CMPB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPB x (MOVLconst [c]))
	// result: (CMPBconst x [int8(c)])
	for {
		x := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(Op386CMPBconst)
		v.AuxInt = int8ToAuxInt(int8(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPB (MOVLconst [c]) x)
	// result: (InvertFlags (CMPBconst x [int8(c)]))
	for {
		if v_0.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_1
		v.reset(Op386InvertFlags)
		v0 := b.NewValue0(v.Pos, Op386CMPBconst, types.TypeFlags)
		v0.AuxInt = int8ToAuxInt(int8(c))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPB x y)
	// cond: x.ID > y.ID
	// result: (InvertFlags (CMPB y x))
	for {
		x := v_0
		y := v_1
		if !(x.ID > y.ID) {
			break
		}
		v.reset(Op386InvertFlags)
		v0 := b.NewValue0(v.Pos, Op386CMPB, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPB l:(MOVBload {sym} [off] ptr mem) x)
	// cond: canMergeLoad(v, l) && clobber(l)
	// result: (CMPBload {sym} [off] ptr x mem)
	for {
		l := v_0
		if l.Op != Op386MOVBload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		x := v_1
		if !(canMergeLoad(v, l) && clobber(l)) {
			break
		}
		v.reset(Op386CMPBload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (CMPB x l:(MOVBload {sym} [off] ptr mem))
	// cond: canMergeLoad(v, l) && clobber(l)
	// result: (InvertFlags (CMPBload {sym} [off] ptr x mem))
	for {
		x := v_0
		l := v_1
		if l.Op != Op386MOVBload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(canMergeLoad(v, l) && clobber(l)) {
			break
		}
		v.reset(Op386InvertFlags)
		v0 := b.NewValue0(l.Pos, Op386CMPBload, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, x, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue386_Op386CMPBconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)==y
	// result: (FlagEQ)
	for {
		y := auxIntToInt8(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int8(x) == y) {
			break
		}
		v.reset(Op386FlagEQ)
		return true
	}
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)<y && uint8(x)<uint8(y)
	// result: (FlagLT_ULT)
	for {
		y := auxIntToInt8(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int8(x) < y && uint8(x) < uint8(y)) {
			break
		}
		v.reset(Op386FlagLT_ULT)
		return true
	}
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)<y && uint8(x)>uint8(y)
	// result: (FlagLT_UGT)
	for {
		y := auxIntToInt8(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int8(x) < y && uint8(x) > uint8(y)) {
			break
		}
		v.reset(Op386FlagLT_UGT)
		return true
	}
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)>y && uint8(x)<uint8(y)
	// result: (FlagGT_ULT)
	for {
		y := auxIntToInt8(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int8(x) > y && uint8(x) < uint8(y)) {
			break
		}
		v.reset(Op386FlagGT_ULT)
		return true
	}
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)>y && uint8(x)>uint8(y)
	// result: (FlagGT_UGT)
	for {
		y := auxIntToInt8(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int8(x) > y && uint8(x) > uint8(y)) {
			break
		}
		v.reset(Op386FlagGT_UGT)
		return true
	}
	// match: (CMPBconst (ANDLconst _ [m]) [n])
	// cond: 0 <= int8(m) && int8(m) < n
	// result: (FlagLT_ULT)
	for {
		n := auxIntToInt8(v.AuxInt)
		if v_0.Op != Op386ANDLconst {
			break
		}
		m := auxIntToInt32(v_0.AuxInt)
		if !(0 <= int8(m) && int8(m) < n) {
			break
		}
		v.reset(Op386FlagLT_ULT)
		return true
	}
	// match: (CMPBconst l:(ANDL x y) [0])
	// cond: l.Uses==1
	// result: (TESTB x y)
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		l := v_0
		if l.Op != Op386ANDL {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(l.Uses == 1) {
			break
		}
		v.reset(Op386TESTB)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMPBconst l:(ANDLconst [c] x) [0])
	// cond: l.Uses==1
	// result: (TESTBconst [int8(c)] x)
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		l := v_0
		if l.Op != Op386ANDLconst {
			break
		}
		c := auxIntToInt32(l.AuxInt)
		x := l.Args[0]
		if !(l.Uses == 1) {
			break
		}
		v.reset(Op386TESTBconst)
		v.AuxInt = int8ToAuxInt(int8(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPBconst x [0])
	// result: (TESTB x x)
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.reset(Op386TESTB)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPBconst l:(MOVBload {sym} [off] ptr mem) [c])
	// cond: l.Uses == 1 && validValAndOff(int64(c), int64(off)) && clobber(l)
	// result: @l.Block (CMPBconstload {sym} [makeValAndOff32(int32(c),int32(off))] ptr mem)
	for {
		c := auxIntToInt8(v.AuxInt)
		l := v_0
		if l.Op != Op386MOVBload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(l.Uses == 1 && validValAndOff(int64(c), int64(off)) && clobber(l)) {
			break
		}
		b = l.Block
		v0 := b.NewValue0(l.Pos, Op386CMPBconstload, types.TypeFlags)
		v.copyOf(v0)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff32(int32(c), int32(off)))
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386CMPBload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMPBload {sym} [off] ptr (MOVLconst [c]) mem)
	// cond: validValAndOff(int64(int8(c)),int64(off))
	// result: (CMPBconstload {sym} [makeValAndOff32(int32(int8(c)),off)] ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		mem := v_2
		if !(validValAndOff(int64(int8(c)), int64(off))) {
			break
		}
		v.reset(Op386CMPBconstload)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(int32(int8(c)), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386CMPL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPL x (MOVLconst [c]))
	// result: (CMPLconst x [c])
	for {
		x := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(Op386CMPLconst)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (CMPL (MOVLconst [c]) x)
	// result: (InvertFlags (CMPLconst x [c]))
	for {
		if v_0.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_1
		v.reset(Op386InvertFlags)
		v0 := b.NewValue0(v.Pos, Op386CMPLconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPL x y)
	// cond: x.ID > y.ID
	// result: (InvertFlags (CMPL y x))
	for {
		x := v_0
		y := v_1
		if !(x.ID > y.ID) {
			break
		}
		v.reset(Op386InvertFlags)
		v0 := b.NewValue0(v.Pos, Op386CMPL, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPL l:(MOVLload {sym} [off] ptr mem) x)
	// cond: canMergeLoad(v, l) && clobber(l)
	// result: (CMPLload {sym} [off] ptr x mem)
	for {
		l := v_0
		if l.Op != Op386MOVLload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		x := v_1
		if !(canMergeLoad(v, l) && clobber(l)) {
			break
		}
		v.reset(Op386CMPLload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (CMPL x l:(MOVLload {sym} [off] ptr mem))
	// cond: canMergeLoad(v, l) && clobber(l)
	// result: (InvertFlags (CMPLload {sym} [off] ptr x mem))
	for {
		x := v_0
		l := v_1
		if l.Op != Op386MOVLload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(canMergeLoad(v, l) && clobber(l)) {
			break
		}
		v.reset(Op386InvertFlags)
		v0 := b.NewValue0(l.Pos, Op386CMPLload, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, x, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue386_Op386CMPLconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: x==y
	// result: (FlagEQ)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(x == y) {
			break
		}
		v.reset(Op386FlagEQ)
		return true
	}
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: x<y && uint32(x)<uint32(y)
	// result: (FlagLT_ULT)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(x < y && uint32(x) < uint32(y)) {
			break
		}
		v.reset(Op386FlagLT_ULT)
		return true
	}
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: x<y && uint32(x)>uint32(y)
	// result: (FlagLT_UGT)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(x < y && uint32(x) > uint32(y)) {
			break
		}
		v.reset(Op386FlagLT_UGT)
		return true
	}
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: x>y && uint32(x)<uint32(y)
	// result: (FlagGT_ULT)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(x > y && uint32(x) < uint32(y)) {
			break
		}
		v.reset(Op386FlagGT_ULT)
		return true
	}
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: x>y && uint32(x)>uint32(y)
	// result: (FlagGT_UGT)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(x > y && uint32(x) > uint32(y)) {
			break
		}
		v.reset(Op386FlagGT_UGT)
		return true
	}
	// match: (CMPLconst (SHRLconst _ [c]) [n])
	// cond: 0 <= n && 0 < c && c <= 32 && (1<<uint64(32-c)) <= uint64(n)
	// result: (FlagLT_ULT)
	for {
		n := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386SHRLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		if !(0 <= n && 0 < c && c <= 32 && (1<<uint64(32-c)) <= uint64(n)) {
			break
		}
		v.reset(Op386FlagLT_ULT)
		return true
	}
	// match: (CMPLconst (ANDLconst _ [m]) [n])
	// cond: 0 <= m && m < n
	// result: (FlagLT_ULT)
	for {
		n := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386ANDLconst {
			break
		}
		m := auxIntToInt32(v_0.AuxInt)
		if !(0 <= m && m < n) {
			break
		}
		v.reset(Op386FlagLT_ULT)
		return true
	}
	// match: (CMPLconst l:(ANDL x y) [0])
	// cond: l.Uses==1
	// result: (TESTL x y)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		l := v_0
		if l.Op != Op386ANDL {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(l.Uses == 1) {
			break
		}
		v.reset(Op386TESTL)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMPLconst l:(ANDLconst [c] x) [0])
	// cond: l.Uses==1
	// result: (TESTLconst [c] x)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		l := v_0
		if l.Op != Op386ANDLconst {
			break
		}
		c := auxIntToInt32(l.AuxInt)
		x := l.Args[0]
		if !(l.Uses == 1) {
			break
		}
		v.reset(Op386TESTLconst)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (CMPLconst x [0])
	// result: (TESTL x x)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.reset(Op386TESTL)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPLconst l:(MOVLload {sym} [off] ptr mem) [c])
	// cond: l.Uses == 1 && validValAndOff(int64(c), int64(off)) && clobber(l)
	// result: @l.Block (CMPLconstload {sym} [makeValAndOff32(int32(c),int32(off))] ptr mem)
	for {
		c := auxIntToInt32(v.AuxInt)
		l := v_0
		if l.Op != Op386MOVLload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(l.Uses == 1 && validValAndOff(int64(c), int64(off)) && clobber(l)) {
			break
		}
		b = l.Block
		v0 := b.NewValue0(l.Pos, Op386CMPLconstload, types.TypeFlags)
		v.copyOf(v0)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff32(int32(c), int32(off)))
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386CMPLload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMPLload {sym} [off] ptr (MOVLconst [c]) mem)
	// cond: validValAndOff(int64(c),int64(off))
	// result: (CMPLconstload {sym} [makeValAndOff32(c,off)] ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		mem := v_2
		if !(validValAndOff(int64(c), int64(off))) {
			break
		}
		v.reset(Op386CMPLconstload)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(c, off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386CMPW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPW x (MOVLconst [c]))
	// result: (CMPWconst x [int16(c)])
	for {
		x := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(Op386CMPWconst)
		v.AuxInt = int16ToAuxInt(int16(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPW (MOVLconst [c]) x)
	// result: (InvertFlags (CMPWconst x [int16(c)]))
	for {
		if v_0.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_1
		v.reset(Op386InvertFlags)
		v0 := b.NewValue0(v.Pos, Op386CMPWconst, types.TypeFlags)
		v0.AuxInt = int16ToAuxInt(int16(c))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPW x y)
	// cond: x.ID > y.ID
	// result: (InvertFlags (CMPW y x))
	for {
		x := v_0
		y := v_1
		if !(x.ID > y.ID) {
			break
		}
		v.reset(Op386InvertFlags)
		v0 := b.NewValue0(v.Pos, Op386CMPW, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPW l:(MOVWload {sym} [off] ptr mem) x)
	// cond: canMergeLoad(v, l) && clobber(l)
	// result: (CMPWload {sym} [off] ptr x mem)
	for {
		l := v_0
		if l.Op != Op386MOVWload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		x := v_1
		if !(canMergeLoad(v, l) && clobber(l)) {
			break
		}
		v.reset(Op386CMPWload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (CMPW x l:(MOVWload {sym} [off] ptr mem))
	// cond: canMergeLoad(v, l) && clobber(l)
	// result: (InvertFlags (CMPWload {sym} [off] ptr x mem))
	for {
		x := v_0
		l := v_1
		if l.Op != Op386MOVWload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(canMergeLoad(v, l) && clobber(l)) {
			break
		}
		v.reset(Op386InvertFlags)
		v0 := b.NewValue0(l.Pos, Op386CMPWload, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, x, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue386_Op386CMPWconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)==y
	// result: (FlagEQ)
	for {
		y := auxIntToInt16(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int16(x) == y) {
			break
		}
		v.reset(Op386FlagEQ)
		return true
	}
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)<y && uint16(x)<uint16(y)
	// result: (FlagLT_ULT)
	for {
		y := auxIntToInt16(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int16(x) < y && uint16(x) < uint16(y)) {
			break
		}
		v.reset(Op386FlagLT_ULT)
		return true
	}
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)<y && uint16(x)>uint16(y)
	// result: (FlagLT_UGT)
	for {
		y := auxIntToInt16(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int16(x) < y && uint16(x) > uint16(y)) {
			break
		}
		v.reset(Op386FlagLT_UGT)
		return true
	}
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)>y && uint16(x)<uint16(y)
	// result: (FlagGT_ULT)
	for {
		y := auxIntToInt16(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int16(x) > y && uint16(x) < uint16(y)) {
			break
		}
		v.reset(Op386FlagGT_ULT)
		return true
	}
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)>y && uint16(x)>uint16(y)
	// result: (FlagGT_UGT)
	for {
		y := auxIntToInt16(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int16(x) > y && uint16(x) > uint16(y)) {
			break
		}
		v.reset(Op386FlagGT_UGT)
		return true
	}
	// match: (CMPWconst (ANDLconst _ [m]) [n])
	// cond: 0 <= int16(m) && int16(m) < n
	// result: (FlagLT_ULT)
	for {
		n := auxIntToInt16(v.AuxInt)
		if v_0.Op != Op386ANDLconst {
			break
		}
		m := auxIntToInt32(v_0.AuxInt)
		if !(0 <= int16(m) && int16(m) < n) {
			break
		}
		v.reset(Op386FlagLT_ULT)
		return true
	}
	// match: (CMPWconst l:(ANDL x y) [0])
	// cond: l.Uses==1
	// result: (TESTW x y)
	for {
		if auxIntToInt16(v.AuxInt) != 0 {
			break
		}
		l := v_0
		if l.Op != Op386ANDL {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(l.Uses == 1) {
			break
		}
		v.reset(Op386TESTW)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMPWconst l:(ANDLconst [c] x) [0])
	// cond: l.Uses==1
	// result: (TESTWconst [int16(c)] x)
	for {
		if auxIntToInt16(v.AuxInt) != 0 {
			break
		}
		l := v_0
		if l.Op != Op386ANDLconst {
			break
		}
		c := auxIntToInt32(l.AuxInt)
		x := l.Args[0]
		if !(l.Uses == 1) {
			break
		}
		v.reset(Op386TESTWconst)
		v.AuxInt = int16ToAuxInt(int16(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPWconst x [0])
	// result: (TESTW x x)
	for {
		if auxIntToInt16(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.reset(Op386TESTW)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPWconst l:(MOVWload {sym} [off] ptr mem) [c])
	// cond: l.Uses == 1 && validValAndOff(int64(c), int64(off)) && clobber(l)
	// result: @l.Block (CMPWconstload {sym} [makeValAndOff32(int32(c),int32(off))] ptr mem)
	for {
		c := auxIntToInt16(v.AuxInt)
		l := v_0
		if l.Op != Op386MOVWload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(l.Uses == 1 && validValAndOff(int64(c), int64(off)) && clobber(l)) {
			break
		}
		b = l.Block
		v0 := b.NewValue0(l.Pos, Op386CMPWconstload, types.TypeFlags)
		v.copyOf(v0)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff32(int32(c), int32(off)))
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386CMPWload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMPWload {sym} [off] ptr (MOVLconst [c]) mem)
	// cond: validValAndOff(int64(int16(c)),int64(off))
	// result: (CMPWconstload {sym} [makeValAndOff32(int32(int16(c)),off)] ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		mem := v_2
		if !(validValAndOff(int64(int16(c)), int64(off))) {
			break
		}
		v.reset(Op386CMPWconstload)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(int32(int16(c)), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386DIVSD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (DIVSD x l:(MOVSDload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (DIVSDload x [off] {sym} ptr mem)
	for {
		x := v_0
		l := v_1
		if l.Op != Op386MOVSDload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
			break
		}
		v.reset(Op386DIVSDload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(x, ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386DIVSDload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (DIVSDload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (DIVSDload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386DIVSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (DIVSDload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (DIVSDload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386DIVSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386DIVSS(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (DIVSS x l:(MOVSSload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (DIVSSload x [off] {sym} ptr mem)
	for {
		x := v_0
		l := v_1
		if l.Op != Op386MOVSSload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
			break
		}
		v.reset(Op386DIVSSload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(x, ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386DIVSSload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (DIVSSload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (DIVSSload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386DIVSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (DIVSSload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (DIVSSload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386DIVSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386LEAL(v *Value) bool {
	v_0 := v.Args[0]
	// match: (LEAL [c] {s} (ADDLconst [d] x))
	// cond: is32Bit(int64(c)+int64(d))
	// result: (LEAL [c+d] {s} x)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(Op386LEAL)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg(x)
		return true
	}
	// match: (LEAL [c] {s} (ADDL x y))
	// cond: x.Op != OpSB && y.Op != OpSB
	// result: (LEAL1 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != Op386ADDL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if !(x.Op != OpSB && y.Op != OpSB) {
				continue
			}
			v.reset(Op386LEAL1)
			v.AuxInt = int32ToAuxInt(c)
			v.Aux = symToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAL [off1] {sym1} (LEAL [off2] {sym2} x))
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (LEAL [off1+off2] {mergeSym(sym1,sym2)} x)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		x := v_0.Args[0]
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(Op386LEAL)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg(x)
		return true
	}
	// match: (LEAL [off1] {sym1} (LEAL1 [off2] {sym2} x y))
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (LEAL1 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL1 {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(Op386LEAL1)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL [off1] {sym1} (LEAL2 [off2] {sym2} x y))
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (LEAL2 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL2 {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(Op386LEAL2)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL [off1] {sym1} (LEAL4 [off2] {sym2} x y))
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (LEAL4 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL4 {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(Op386LEAL4)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL [off1] {sym1} (LEAL8 [off2] {sym2} x y))
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (LEAL8 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL8 {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(Op386LEAL8)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_Op386LEAL1(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LEAL1 [c] {s} (ADDLconst [d] x) y)
	// cond: is32Bit(int64(c)+int64(d)) && x.Op != OpSB
	// result: (LEAL1 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != Op386ADDLconst {
				continue
			}
			d := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			y := v_1
			if !(is32Bit(int64(c)+int64(d)) && x.Op != OpSB) {
				continue
			}
			v.reset(Op386LEAL1)
			v.AuxInt = int32ToAuxInt(c + d)
			v.Aux = symToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAL1 [c] {s} x (SHLLconst [1] y))
	// result: (LEAL2 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386SHLLconst || auxIntToInt32(v_1.AuxInt) != 1 {
				continue
			}
			y := v_1.Args[0]
			v.reset(Op386LEAL2)
			v.AuxInt = int32ToAuxInt(c)
			v.Aux = symToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAL1 [c] {s} x (SHLLconst [2] y))
	// result: (LEAL4 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386SHLLconst || auxIntToInt32(v_1.AuxInt) != 2 {
				continue
			}
			y := v_1.Args[0]
			v.reset(Op386LEAL4)
			v.AuxInt = int32ToAuxInt(c)
			v.Aux = symToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAL1 [c] {s} x (SHLLconst [3] y))
	// result: (LEAL8 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386SHLLconst || auxIntToInt32(v_1.AuxInt) != 3 {
				continue
			}
			y := v_1.Args[0]
			v.reset(Op386LEAL8)
			v.AuxInt = int32ToAuxInt(c)
			v.Aux = symToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAL1 [off1] {sym1} (LEAL [off2] {sym2} x) y)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && x.Op != OpSB
	// result: (LEAL1 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != Op386LEAL {
				continue
			}
			off2 := auxIntToInt32(v_0.AuxInt)
			sym2 := auxToSym(v_0.Aux)
			x := v_0.Args[0]
			y := v_1
			if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && x.Op != OpSB) {
				continue
			}
			v.reset(Op386LEAL1)
			v.AuxInt = int32ToAuxInt(off1 + off2)
			v.Aux = symToAux(mergeSym(sym1, sym2))
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAL1 [off1] {sym1} x (LEAL1 [off2] {sym2} y y))
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (LEAL2 [off1+off2] {mergeSym(sym1, sym2)} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386LEAL1 {
				continue
			}
			off2 := auxIntToInt32(v_1.AuxInt)
			sym2 := auxToSym(v_1.Aux)
			y := v_1.Args[1]
			if y != v_1.Args[0] || !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
				continue
			}
			v.reset(Op386LEAL2)
			v.AuxInt = int32ToAuxInt(off1 + off2)
			v.Aux = symToAux(mergeSym(sym1, sym2))
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAL1 [off1] {sym1} x (LEAL1 [off2] {sym2} x y))
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (LEAL2 [off1+off2] {mergeSym(sym1, sym2)} y x)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386LEAL1 {
				continue
			}
			off2 := auxIntToInt32(v_1.AuxInt)
			sym2 := auxToSym(v_1.Aux)
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				y := v_1_1
				if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
					continue
				}
				v.reset(Op386LEAL2)
				v.AuxInt = int32ToAuxInt(off1 + off2)
				v.Aux = symToAux(mergeSym(sym1, sym2))
				v.AddArg2(y, x)
				return true
			}
		}
		break
	}
	// match: (LEAL1 [0] {nil} x y)
	// result: (ADDL x y)
	for {
		if auxIntToInt32(v.AuxInt) != 0 || auxToSym(v.Aux) != nil {
			break
		}
		x := v_0
		y := v_1
		v.reset(Op386ADDL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_Op386LEAL2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LEAL2 [c] {s} (ADDLconst [d] x) y)
	// cond: is32Bit(int64(c)+int64(d)) && x.Op != OpSB
	// result: (LEAL2 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		y := v_1
		if !(is32Bit(int64(c)+int64(d)) && x.Op != OpSB) {
			break
		}
		v.reset(Op386LEAL2)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL2 [c] {s} x (ADDLconst [d] y))
	// cond: is32Bit(int64(c)+2*int64(d)) && y.Op != OpSB
	// result: (LEAL2 [c+2*d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != Op386ADDLconst {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(is32Bit(int64(c)+2*int64(d)) && y.Op != OpSB) {
			break
		}
		v.reset(Op386LEAL2)
		v.AuxInt = int32ToAuxInt(c + 2*d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL2 [c] {s} x (SHLLconst [1] y))
	// result: (LEAL4 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != Op386SHLLconst || auxIntToInt32(v_1.AuxInt) != 1 {
			break
		}
		y := v_1.Args[0]
		v.reset(Op386LEAL4)
		v.AuxInt = int32ToAuxInt(c)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL2 [c] {s} x (SHLLconst [2] y))
	// result: (LEAL8 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != Op386SHLLconst || auxIntToInt32(v_1.AuxInt) != 2 {
			break
		}
		y := v_1.Args[0]
		v.reset(Op386LEAL8)
		v.AuxInt = int32ToAuxInt(c)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL2 [off1] {sym1} (LEAL [off2] {sym2} x) y)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && x.Op != OpSB
	// result: (LEAL2 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		x := v_0.Args[0]
		y := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && x.Op != OpSB) {
			break
		}
		v.reset(Op386LEAL2)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL2 [off1] {sym} x (LEAL1 [off2] {nil} y y))
	// cond: is32Bit(int64(off1)+2*int64(off2))
	// result: (LEAL4 [off1+2*off2] {sym} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != Op386LEAL1 {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		if auxToSym(v_1.Aux) != nil {
			break
		}
		y := v_1.Args[1]
		if y != v_1.Args[0] || !(is32Bit(int64(off1) + 2*int64(off2))) {
			break
		}
		v.reset(Op386LEAL4)
		v.AuxInt = int32ToAuxInt(off1 + 2*off2)
		v.Aux = symToAux(sym)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_Op386LEAL4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LEAL4 [c] {s} (ADDLconst [d] x) y)
	// cond: is32Bit(int64(c)+int64(d)) && x.Op != OpSB
	// result: (LEAL4 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		y := v_1
		if !(is32Bit(int64(c)+int64(d)) && x.Op != OpSB) {
			break
		}
		v.reset(Op386LEAL4)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL4 [c] {s} x (ADDLconst [d] y))
	// cond: is32Bit(int64(c)+4*int64(d)) && y.Op != OpSB
	// result: (LEAL4 [c+4*d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != Op386ADDLconst {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(is32Bit(int64(c)+4*int64(d)) && y.Op != OpSB) {
			break
		}
		v.reset(Op386LEAL4)
		v.AuxInt = int32ToAuxInt(c + 4*d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL4 [c] {s} x (SHLLconst [1] y))
	// result: (LEAL8 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != Op386SHLLconst || auxIntToInt32(v_1.AuxInt) != 1 {
			break
		}
		y := v_1.Args[0]
		v.reset(Op386LEAL8)
		v.AuxInt = int32ToAuxInt(c)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL4 [off1] {sym1} (LEAL [off2] {sym2} x) y)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && x.Op != OpSB
	// result: (LEAL4 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		x := v_0.Args[0]
		y := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && x.Op != OpSB) {
			break
		}
		v.reset(Op386LEAL4)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL4 [off1] {sym} x (LEAL1 [off2] {nil} y y))
	// cond: is32Bit(int64(off1)+4*int64(off2))
	// result: (LEAL8 [off1+4*off2] {sym} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != Op386LEAL1 {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		if auxToSym(v_1.Aux) != nil {
			break
		}
		y := v_1.Args[1]
		if y != v_1.Args[0] || !(is32Bit(int64(off1) + 4*int64(off2))) {
			break
		}
		v.reset(Op386LEAL8)
		v.AuxInt = int32ToAuxInt(off1 + 4*off2)
		v.Aux = symToAux(sym)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_Op386LEAL8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LEAL8 [c] {s} (ADDLconst [d] x) y)
	// cond: is32Bit(int64(c)+int64(d)) && x.Op != OpSB
	// result: (LEAL8 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		y := v_1
		if !(is32Bit(int64(c)+int64(d)) && x.Op != OpSB) {
			break
		}
		v.reset(Op386LEAL8)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL8 [c] {s} x (ADDLconst [d] y))
	// cond: is32Bit(int64(c)+8*int64(d)) && y.Op != OpSB
	// result: (LEAL8 [c+8*d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != Op386ADDLconst {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(is32Bit(int64(c)+8*int64(d)) && y.Op != OpSB) {
			break
		}
		v.reset(Op386LEAL8)
		v.AuxInt = int32ToAuxInt(c + 8*d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL8 [off1] {sym1} (LEAL [off2] {sym2} x) y)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && x.Op != OpSB
	// result: (LEAL8 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		x := v_0.Args[0]
		y := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && x.Op != OpSB) {
			break
		}
		v.reset(Op386LEAL8)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVBLSX(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVBLSX x:(MOVBload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBLSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != Op386MOVBload {
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
		v0 := b.NewValue0(x.Pos, Op386MOVBLSXload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBLSX (ANDLconst [c] x))
	// cond: c & 0x80 == 0
	// result: (ANDLconst [c & 0x7f] x)
	for {
		if v_0.Op != Op386ANDLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c&0x80 == 0) {
			break
		}
		v.reset(Op386ANDLconst)
		v.AuxInt = int32ToAuxInt(c & 0x7f)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVBLSXload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBLSXload [off] {sym} ptr (MOVBstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVBLSX x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != Op386MOVBstore {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(Op386MOVBLSX)
		v.AddArg(x)
		return true
	}
	// match: (MOVBLSXload [off1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBLSXload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386MOVBLSXload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVBLZX(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVBLZX x:(MOVBload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != Op386MOVBload {
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
		v0 := b.NewValue0(x.Pos, Op386MOVBload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBLZX (ANDLconst [c] x))
	// result: (ANDLconst [c & 0xff] x)
	for {
		if v_0.Op != Op386ANDLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(Op386ANDLconst)
		v.AuxInt = int32ToAuxInt(c & 0xff)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVBload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBload [off] {sym} ptr (MOVBstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVBLZX x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != Op386MOVBstore {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(Op386MOVBLZX)
		v.AddArg(x)
		return true
	}
	// match: (MOVBload [off1] {sym} (ADDLconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVBload [off1+off2] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386MOVBload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386MOVBload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVBload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVLconst [int32(read8(sym, int64(off)))])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpSB || !(symIsRO(sym)) {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(int32(read8(sym, int64(off))))
		return true
	}
	return false
}
func rewriteValue386_Op386MOVBstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBstore [off] {sym} ptr (MOVBLSX x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != Op386MOVBLSX {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(Op386MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBLZX x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != Op386MOVBLZX {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(Op386MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off1] {sym} (ADDLconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVBstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386MOVBstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVLconst [c]) mem)
	// cond: validOff(int64(off))
	// result: (MOVBstoreconst [makeValAndOff32(c,off)] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		mem := v_2
		if !(validOff(int64(off))) {
			break
		}
		v.reset(Op386MOVBstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(c, off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstore [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386MOVBstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p (SHRWconst [8] w) x:(MOVBstore [i-1] {s} p w mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVWstore [i-1] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		if v_1.Op != Op386SHRWconst || auxIntToInt16(v_1.AuxInt) != 8 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != Op386MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] || w != x.Args[1] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(Op386MOVWstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v.AddArg3(p, w, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p (SHRLconst [8] w) x:(MOVBstore [i-1] {s} p w mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVWstore [i-1] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		if v_1.Op != Op386SHRLconst || auxIntToInt32(v_1.AuxInt) != 8 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != Op386MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] || w != x.Args[1] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(Op386MOVWstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v.AddArg3(p, w, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p w x:(MOVBstore {s} [i+1] p (SHRWconst [8] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVWstore [i] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		w := v_1
		x := v_2
		if x.Op != Op386MOVBstore || auxIntToInt32(x.AuxInt) != i+1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] {
			break
		}
		x_1 := x.Args[1]
		if x_1.Op != Op386SHRWconst || auxIntToInt16(x_1.AuxInt) != 8 || w != x_1.Args[0] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(Op386MOVWstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p, w, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p w x:(MOVBstore {s} [i+1] p (SHRLconst [8] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVWstore [i] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		w := v_1
		x := v_2
		if x.Op != Op386MOVBstore || auxIntToInt32(x.AuxInt) != i+1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] {
			break
		}
		x_1 := x.Args[1]
		if x_1.Op != Op386SHRLconst || auxIntToInt32(x_1.AuxInt) != 8 || w != x_1.Args[0] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(Op386MOVWstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p, w, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p (SHRLconst [j] w) x:(MOVBstore [i-1] {s} p w0:(SHRLconst [j-8] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVWstore [i-1] {s} p w0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		if v_1.Op != Op386SHRLconst {
			break
		}
		j := auxIntToInt32(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != Op386MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] {
			break
		}
		w0 := x.Args[1]
		if w0.Op != Op386SHRLconst || auxIntToInt32(w0.AuxInt) != j-8 || w != w0.Args[0] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(Op386MOVWstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v.AddArg3(p, w0, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p1 (SHRWconst [8] w) x:(MOVBstore [i] {s} p0 w mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)
	// result: (MOVWstore [i] {s} p0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p1 := v_0
		if v_1.Op != Op386SHRWconst || auxIntToInt16(v_1.AuxInt) != 8 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != Op386MOVBstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p0 := x.Args[0]
		if w != x.Args[1] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)) {
			break
		}
		v.reset(Op386MOVWstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p1 (SHRLconst [8] w) x:(MOVBstore [i] {s} p0 w mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)
	// result: (MOVWstore [i] {s} p0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p1 := v_0
		if v_1.Op != Op386SHRLconst || auxIntToInt32(v_1.AuxInt) != 8 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != Op386MOVBstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p0 := x.Args[0]
		if w != x.Args[1] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)) {
			break
		}
		v.reset(Op386MOVWstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p0 w x:(MOVBstore {s} [i] p1 (SHRWconst [8] w) mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)
	// result: (MOVWstore [i] {s} p0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p0 := v_0
		w := v_1
		x := v_2
		if x.Op != Op386MOVBstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p1 := x.Args[0]
		x_1 := x.Args[1]
		if x_1.Op != Op386SHRWconst || auxIntToInt16(x_1.AuxInt) != 8 || w != x_1.Args[0] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)) {
			break
		}
		v.reset(Op386MOVWstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p0 w x:(MOVBstore {s} [i] p1 (SHRLconst [8] w) mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)
	// result: (MOVWstore [i] {s} p0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p0 := v_0
		w := v_1
		x := v_2
		if x.Op != Op386MOVBstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p1 := x.Args[0]
		x_1 := x.Args[1]
		if x_1.Op != Op386SHRLconst || auxIntToInt32(x_1.AuxInt) != 8 || w != x_1.Args[0] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)) {
			break
		}
		v.reset(Op386MOVWstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p1 (SHRLconst [j] w) x:(MOVBstore [i] {s} p0 w0:(SHRLconst [j-8] w) mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)
	// result: (MOVWstore [i] {s} p0 w0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p1 := v_0
		if v_1.Op != Op386SHRLconst {
			break
		}
		j := auxIntToInt32(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != Op386MOVBstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p0 := x.Args[0]
		w0 := x.Args[1]
		if w0.Op != Op386SHRLconst || auxIntToInt32(w0.AuxInt) != j-8 || w != w0.Args[0] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)) {
			break
		}
		v.reset(Op386MOVWstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w0, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVBstoreconst(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBstoreconst [sc] {s} (ADDLconst [off] ptr) mem)
	// cond: sc.canAdd32(off)
	// result: (MOVBstoreconst [sc.addOffset32(off)] {s} ptr mem)
	for {
		sc := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(sc.canAdd32(off)) {
			break
		}
		v.reset(Op386MOVBstoreconst)
		v.AuxInt = valAndOffToAuxInt(sc.addOffset32(off))
		v.Aux = symToAux(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstoreconst [sc] {sym1} (LEAL [off] {sym2} ptr) mem)
	// cond: canMergeSym(sym1, sym2) && sc.canAdd32(off) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBstoreconst [sc.addOffset32(off)] {mergeSym(sym1, sym2)} ptr mem)
	for {
		sc := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && sc.canAdd32(off) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386MOVBstoreconst)
		v.AuxInt = valAndOffToAuxInt(sc.addOffset32(off))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstoreconst [c] {s} p x:(MOVBstoreconst [a] {s} p mem))
	// cond: x.Uses == 1 && a.Off() + 1 == c.Off() && clobber(x)
	// result: (MOVWstoreconst [makeValAndOff32(int32(a.Val()&0xff | c.Val()<<8), a.Off32())] {s} p mem)
	for {
		c := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		x := v_1
		if x.Op != Op386MOVBstoreconst {
			break
		}
		a := auxIntToValAndOff(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		if p != x.Args[0] || !(x.Uses == 1 && a.Off()+1 == c.Off() && clobber(x)) {
			break
		}
		v.reset(Op386MOVWstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(int32(a.Val()&0xff|c.Val()<<8), a.Off32()))
		v.Aux = symToAux(s)
		v.AddArg2(p, mem)
		return true
	}
	// match: (MOVBstoreconst [a] {s} p x:(MOVBstoreconst [c] {s} p mem))
	// cond: x.Uses == 1 && a.Off() + 1 == c.Off() && clobber(x)
	// result: (MOVWstoreconst [makeValAndOff32(int32(a.Val()&0xff | c.Val()<<8), a.Off32())] {s} p mem)
	for {
		a := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		x := v_1
		if x.Op != Op386MOVBstoreconst {
			break
		}
		c := auxIntToValAndOff(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		if p != x.Args[0] || !(x.Uses == 1 && a.Off()+1 == c.Off() && clobber(x)) {
			break
		}
		v.reset(Op386MOVWstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(int32(a.Val()&0xff|c.Val()<<8), a.Off32()))
		v.Aux = symToAux(s)
		v.AddArg2(p, mem)
		return true
	}
	// match: (MOVBstoreconst [c] {s} p1 x:(MOVBstoreconst [a] {s} p0 mem))
	// cond: x.Uses == 1 && a.Off() == c.Off() && sequentialAddresses(p0, p1, 1) && clobber(x)
	// result: (MOVWstoreconst [makeValAndOff32(int32(a.Val()&0xff | c.Val()<<8), a.Off32())] {s} p0 mem)
	for {
		c := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		p1 := v_0
		x := v_1
		if x.Op != Op386MOVBstoreconst {
			break
		}
		a := auxIntToValAndOff(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		p0 := x.Args[0]
		if !(x.Uses == 1 && a.Off() == c.Off() && sequentialAddresses(p0, p1, 1) && clobber(x)) {
			break
		}
		v.reset(Op386MOVWstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(int32(a.Val()&0xff|c.Val()<<8), a.Off32()))
		v.Aux = symToAux(s)
		v.AddArg2(p0, mem)
		return true
	}
	// match: (MOVBstoreconst [a] {s} p0 x:(MOVBstoreconst [c] {s} p1 mem))
	// cond: x.Uses == 1 && a.Off() == c.Off() && sequentialAddresses(p0, p1, 1) && clobber(x)
	// result: (MOVWstoreconst [makeValAndOff32(int32(a.Val()&0xff | c.Val()<<8), a.Off32())] {s} p0 mem)
	for {
		a := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		p0 := v_0
		x := v_1
		if x.Op != Op386MOVBstoreconst {
			break
		}
		c := auxIntToValAndOff(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		p1 := x.Args[0]
		if !(x.Uses == 1 && a.Off() == c.Off() && sequentialAddresses(p0, p1, 1) && clobber(x)) {
			break
		}
		v.reset(Op386MOVWstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(int32(a.Val()&0xff|c.Val()<<8), a.Off32()))
		v.Aux = symToAux(s)
		v.AddArg2(p0, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVLload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVLload [off] {sym} ptr (MOVLstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: x
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != Op386MOVLstore {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVLload [off1] {sym} (ADDLconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVLload [off1+off2] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386MOVLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLload [off1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVLload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386MOVLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVLload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVLconst [int32(read32(sym, int64(off), config.ctxt.Arch.ByteOrder))])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpSB || !(symIsRO(sym)) {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(int32(read32(sym, int64(off), config.ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValue386_Op386MOVLstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVLstore [off1] {sym} (ADDLconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVLstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386MOVLstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVLstore [off] {sym} ptr (MOVLconst [c]) mem)
	// cond: validOff(int64(off))
	// result: (MOVLstoreconst [makeValAndOff32(c,off)] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		mem := v_2
		if !(validOff(int64(off))) {
			break
		}
		v.reset(Op386MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(c, off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLstore [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVLstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386MOVLstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVLstore {sym} [off] ptr y:(ADDLload x [off] {sym} ptr mem) mem)
	// cond: y.Uses==1 && clobber(y)
	// result: (ADDLmodify [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != Op386ADDLload || auxIntToInt32(y.AuxInt) != off || auxToSym(y.Aux) != sym {
			break
		}
		mem := y.Args[2]
		x := y.Args[0]
		if ptr != y.Args[1] || mem != v_2 || !(y.Uses == 1 && clobber(y)) {
			break
		}
		v.reset(Op386ADDLmodify)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVLstore {sym} [off] ptr y:(ANDLload x [off] {sym} ptr mem) mem)
	// cond: y.Uses==1 && clobber(y)
	// result: (ANDLmodify [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != Op386ANDLload || auxIntToInt32(y.AuxInt) != off || auxToSym(y.Aux) != sym {
			break
		}
		mem := y.Args[2]
		x := y.Args[0]
		if ptr != y.Args[1] || mem != v_2 || !(y.Uses == 1 && clobber(y)) {
			break
		}
		v.reset(Op386ANDLmodify)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVLstore {sym} [off] ptr y:(ORLload x [off] {sym} ptr mem) mem)
	// cond: y.Uses==1 && clobber(y)
	// result: (ORLmodify [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != Op386ORLload || auxIntToInt32(y.AuxInt) != off || auxToSym(y.Aux) != sym {
			break
		}
		mem := y.Args[2]
		x := y.Args[0]
		if ptr != y.Args[1] || mem != v_2 || !(y.Uses == 1 && clobber(y)) {
			break
		}
		v.reset(Op386ORLmodify)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVLstore {sym} [off] ptr y:(XORLload x [off] {sym} ptr mem) mem)
	// cond: y.Uses==1 && clobber(y)
	// result: (XORLmodify [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != Op386XORLload || auxIntToInt32(y.AuxInt) != off || auxToSym(y.Aux) != sym {
			break
		}
		mem := y.Args[2]
		x := y.Args[0]
		if ptr != y.Args[1] || mem != v_2 || !(y.Uses == 1 && clobber(y)) {
			break
		}
		v.reset(Op386XORLmodify)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVLstore {sym} [off] ptr y:(ADDL l:(MOVLload [off] {sym} ptr mem) x) mem)
	// cond: y.Uses==1 && l.Uses==1 && clobber(y, l)
	// result: (ADDLmodify [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != Op386ADDL {
			break
		}
		_ = y.Args[1]
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			l := y_0
			if l.Op != Op386MOVLload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
				continue
			}
			mem := l.Args[1]
			if ptr != l.Args[0] {
				continue
			}
			x := y_1
			if mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && clobber(y, l)) {
				continue
			}
			v.reset(Op386ADDLmodify)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(ptr, x, mem)
			return true
		}
		break
	}
	// match: (MOVLstore {sym} [off] ptr y:(SUBL l:(MOVLload [off] {sym} ptr mem) x) mem)
	// cond: y.Uses==1 && l.Uses==1 && clobber(y, l)
	// result: (SUBLmodify [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != Op386SUBL {
			break
		}
		x := y.Args[1]
		l := y.Args[0]
		if l.Op != Op386MOVLload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		if ptr != l.Args[0] || mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && clobber(y, l)) {
			break
		}
		v.reset(Op386SUBLmodify)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVLstore {sym} [off] ptr y:(ANDL l:(MOVLload [off] {sym} ptr mem) x) mem)
	// cond: y.Uses==1 && l.Uses==1 && clobber(y, l)
	// result: (ANDLmodify [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != Op386ANDL {
			break
		}
		_ = y.Args[1]
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			l := y_0
			if l.Op != Op386MOVLload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
				continue
			}
			mem := l.Args[1]
			if ptr != l.Args[0] {
				continue
			}
			x := y_1
			if mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && clobber(y, l)) {
				continue
			}
			v.reset(Op386ANDLmodify)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(ptr, x, mem)
			return true
		}
		break
	}
	// match: (MOVLstore {sym} [off] ptr y:(ORL l:(MOVLload [off] {sym} ptr mem) x) mem)
	// cond: y.Uses==1 && l.Uses==1 && clobber(y, l)
	// result: (ORLmodify [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != Op386ORL {
			break
		}
		_ = y.Args[1]
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			l := y_0
			if l.Op != Op386MOVLload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
				continue
			}
			mem := l.Args[1]
			if ptr != l.Args[0] {
				continue
			}
			x := y_1
			if mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && clobber(y, l)) {
				continue
			}
			v.reset(Op386ORLmodify)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(ptr, x, mem)
			return true
		}
		break
	}
	// match: (MOVLstore {sym} [off] ptr y:(XORL l:(MOVLload [off] {sym} ptr mem) x) mem)
	// cond: y.Uses==1 && l.Uses==1 && clobber(y, l)
	// result: (XORLmodify [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != Op386XORL {
			break
		}
		_ = y.Args[1]
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			l := y_0
			if l.Op != Op386MOVLload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
				continue
			}
			mem := l.Args[1]
			if ptr != l.Args[0] {
				continue
			}
			x := y_1
			if mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && clobber(y, l)) {
				continue
			}
			v.reset(Op386XORLmodify)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(ptr, x, mem)
			return true
		}
		break
	}
	// match: (MOVLstore {sym} [off] ptr y:(ADDLconst [c] l:(MOVLload [off] {sym} ptr mem)) mem)
	// cond: y.Uses==1 && l.Uses==1 && clobber(y, l) && validValAndOff(int64(c),int64(off))
	// result: (ADDLconstmodify [makeValAndOff32(c,off)] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != Op386ADDLconst {
			break
		}
		c := auxIntToInt32(y.AuxInt)
		l := y.Args[0]
		if l.Op != Op386MOVLload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		if ptr != l.Args[0] || mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && clobber(y, l) && validValAndOff(int64(c), int64(off))) {
			break
		}
		v.reset(Op386ADDLconstmodify)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(c, off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLstore {sym} [off] ptr y:(ANDLconst [c] l:(MOVLload [off] {sym} ptr mem)) mem)
	// cond: y.Uses==1 && l.Uses==1 && clobber(y, l) && validValAndOff(int64(c),int64(off))
	// result: (ANDLconstmodify [makeValAndOff32(c,off)] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != Op386ANDLconst {
			break
		}
		c := auxIntToInt32(y.AuxInt)
		l := y.Args[0]
		if l.Op != Op386MOVLload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		if ptr != l.Args[0] || mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && clobber(y, l) && validValAndOff(int64(c), int64(off))) {
			break
		}
		v.reset(Op386ANDLconstmodify)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(c, off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLstore {sym} [off] ptr y:(ORLconst [c] l:(MOVLload [off] {sym} ptr mem)) mem)
	// cond: y.Uses==1 && l.Uses==1 && clobber(y, l) && validValAndOff(int64(c),int64(off))
	// result: (ORLconstmodify [makeValAndOff32(c,off)] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != Op386ORLconst {
			break
		}
		c := auxIntToInt32(y.AuxInt)
		l := y.Args[0]
		if l.Op != Op386MOVLload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		if ptr != l.Args[0] || mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && clobber(y, l) && validValAndOff(int64(c), int64(off))) {
			break
		}
		v.reset(Op386ORLconstmodify)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(c, off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLstore {sym} [off] ptr y:(XORLconst [c] l:(MOVLload [off] {sym} ptr mem)) mem)
	// cond: y.Uses==1 && l.Uses==1 && clobber(y, l) && validValAndOff(int64(c),int64(off))
	// result: (XORLconstmodify [makeValAndOff32(c,off)] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != Op386XORLconst {
			break
		}
		c := auxIntToInt32(y.AuxInt)
		l := y.Args[0]
		if l.Op != Op386MOVLload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		if ptr != l.Args[0] || mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && clobber(y, l) && validValAndOff(int64(c), int64(off))) {
			break
		}
		v.reset(Op386XORLconstmodify)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(c, off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVLstoreconst(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVLstoreconst [sc] {s} (ADDLconst [off] ptr) mem)
	// cond: sc.canAdd32(off)
	// result: (MOVLstoreconst [sc.addOffset32(off)] {s} ptr mem)
	for {
		sc := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(sc.canAdd32(off)) {
			break
		}
		v.reset(Op386MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(sc.addOffset32(off))
		v.Aux = symToAux(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLstoreconst [sc] {sym1} (LEAL [off] {sym2} ptr) mem)
	// cond: canMergeSym(sym1, sym2) && sc.canAdd32(off) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVLstoreconst [sc.addOffset32(off)] {mergeSym(sym1, sym2)} ptr mem)
	for {
		sc := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && sc.canAdd32(off) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(sc.addOffset32(off))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVSDconst(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVSDconst [c])
	// cond: config.ctxt.Flag_shared
	// result: (MOVSDconst2 (MOVSDconst1 [c]))
	for {
		c := auxIntToFloat64(v.AuxInt)
		if !(config.ctxt.Flag_shared) {
			break
		}
		v.reset(Op386MOVSDconst2)
		v0 := b.NewValue0(v.Pos, Op386MOVSDconst1, typ.UInt32)
		v0.AuxInt = float64ToAuxInt(c)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVSDload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVSDload [off1] {sym} (ADDLconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVSDload [off1+off2] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386MOVSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVSDload [off1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVSDload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386MOVSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVSDstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVSDstore [off1] {sym} (ADDLconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVSDstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386MOVSDstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVSDstore [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVSDstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386MOVSDstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVSSconst(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVSSconst [c])
	// cond: config.ctxt.Flag_shared
	// result: (MOVSSconst2 (MOVSSconst1 [c]))
	for {
		c := auxIntToFloat32(v.AuxInt)
		if !(config.ctxt.Flag_shared) {
			break
		}
		v.reset(Op386MOVSSconst2)
		v0 := b.NewValue0(v.Pos, Op386MOVSSconst1, typ.UInt32)
		v0.AuxInt = float32ToAuxInt(c)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVSSload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVSSload [off1] {sym} (ADDLconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVSSload [off1+off2] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386MOVSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVSSload [off1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVSSload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386MOVSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVSSstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVSSstore [off1] {sym} (ADDLconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVSSstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386MOVSSstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVSSstore [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVSSstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386MOVSSstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVWLSX(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVWLSX x:(MOVWload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVWLSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != Op386MOVWload {
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
		v0 := b.NewValue0(x.Pos, Op386MOVWLSXload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWLSX (ANDLconst [c] x))
	// cond: c & 0x8000 == 0
	// result: (ANDLconst [c & 0x7fff] x)
	for {
		if v_0.Op != Op386ANDLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c&0x8000 == 0) {
			break
		}
		v.reset(Op386ANDLconst)
		v.AuxInt = int32ToAuxInt(c & 0x7fff)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVWLSXload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWLSXload [off] {sym} ptr (MOVWstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVWLSX x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != Op386MOVWstore {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(Op386MOVWLSX)
		v.AddArg(x)
		return true
	}
	// match: (MOVWLSXload [off1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWLSXload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386MOVWLSXload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVWLZX(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVWLZX x:(MOVWload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVWload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != Op386MOVWload {
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
		v0 := b.NewValue0(x.Pos, Op386MOVWload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWLZX (ANDLconst [c] x))
	// result: (ANDLconst [c & 0xffff] x)
	for {
		if v_0.Op != Op386ANDLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(Op386ANDLconst)
		v.AuxInt = int32ToAuxInt(c & 0xffff)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVWload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWload [off] {sym} ptr (MOVWstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVWLZX x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != Op386MOVWstore {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(Op386MOVWLZX)
		v.AddArg(x)
		return true
	}
	// match: (MOVWload [off1] {sym} (ADDLconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVWload [off1+off2] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386MOVWload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386MOVWload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVWload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVLconst [int32(read16(sym, int64(off), config.ctxt.Arch.ByteOrder))])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpSB || !(symIsRO(sym)) {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(int32(read16(sym, int64(off), config.ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValue386_Op386MOVWstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWstore [off] {sym} ptr (MOVWLSX x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != Op386MOVWLSX {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(Op386MOVWstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWLZX x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != Op386MOVWLZX {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(Op386MOVWstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym} (ADDLconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVWstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386MOVWstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVLconst [c]) mem)
	// cond: validOff(int64(off))
	// result: (MOVWstoreconst [makeValAndOff32(c,off)] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		mem := v_2
		if !(validOff(int64(off))) {
			break
		}
		v.reset(Op386MOVWstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(c, off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386MOVWstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVWstore [i] {s} p (SHRLconst [16] w) x:(MOVWstore [i-2] {s} p w mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVLstore [i-2] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		if v_1.Op != Op386SHRLconst || auxIntToInt32(v_1.AuxInt) != 16 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != Op386MOVWstore || auxIntToInt32(x.AuxInt) != i-2 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] || w != x.Args[1] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(Op386MOVLstore)
		v.AuxInt = int32ToAuxInt(i - 2)
		v.Aux = symToAux(s)
		v.AddArg3(p, w, mem)
		return true
	}
	// match: (MOVWstore [i] {s} p (SHRLconst [j] w) x:(MOVWstore [i-2] {s} p w0:(SHRLconst [j-16] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVLstore [i-2] {s} p w0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		if v_1.Op != Op386SHRLconst {
			break
		}
		j := auxIntToInt32(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != Op386MOVWstore || auxIntToInt32(x.AuxInt) != i-2 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] {
			break
		}
		w0 := x.Args[1]
		if w0.Op != Op386SHRLconst || auxIntToInt32(w0.AuxInt) != j-16 || w != w0.Args[0] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(Op386MOVLstore)
		v.AuxInt = int32ToAuxInt(i - 2)
		v.Aux = symToAux(s)
		v.AddArg3(p, w0, mem)
		return true
	}
	// match: (MOVWstore [i] {s} p1 (SHRLconst [16] w) x:(MOVWstore [i] {s} p0 w mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, 2) && clobber(x)
	// result: (MOVLstore [i] {s} p0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p1 := v_0
		if v_1.Op != Op386SHRLconst || auxIntToInt32(v_1.AuxInt) != 16 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != Op386MOVWstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p0 := x.Args[0]
		if w != x.Args[1] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 2) && clobber(x)) {
			break
		}
		v.reset(Op386MOVLstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w, mem)
		return true
	}
	// match: (MOVWstore [i] {s} p1 (SHRLconst [j] w) x:(MOVWstore [i] {s} p0 w0:(SHRLconst [j-16] w) mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, 2) && clobber(x)
	// result: (MOVLstore [i] {s} p0 w0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p1 := v_0
		if v_1.Op != Op386SHRLconst {
			break
		}
		j := auxIntToInt32(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != Op386MOVWstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p0 := x.Args[0]
		w0 := x.Args[1]
		if w0.Op != Op386SHRLconst || auxIntToInt32(w0.AuxInt) != j-16 || w != w0.Args[0] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 2) && clobber(x)) {
			break
		}
		v.reset(Op386MOVLstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w0, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVWstoreconst(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWstoreconst [sc] {s} (ADDLconst [off] ptr) mem)
	// cond: sc.canAdd32(off)
	// result: (MOVWstoreconst [sc.addOffset32(off)] {s} ptr mem)
	for {
		sc := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(sc.canAdd32(off)) {
			break
		}
		v.reset(Op386MOVWstoreconst)
		v.AuxInt = valAndOffToAuxInt(sc.addOffset32(off))
		v.Aux = symToAux(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstoreconst [sc] {sym1} (LEAL [off] {sym2} ptr) mem)
	// cond: canMergeSym(sym1, sym2) && sc.canAdd32(off) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWstoreconst [sc.addOffset32(off)] {mergeSym(sym1, sym2)} ptr mem)
	for {
		sc := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && sc.canAdd32(off) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386MOVWstoreconst)
		v.AuxInt = valAndOffToAuxInt(sc.addOffset32(off))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstoreconst [c] {s} p x:(MOVWstoreconst [a] {s} p mem))
	// cond: x.Uses == 1 && a.Off() + 2 == c.Off() && clobber(x)
	// result: (MOVLstoreconst [makeValAndOff32(int32(a.Val()&0xffff | c.Val()<<16), a.Off32())] {s} p mem)
	for {
		c := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		x := v_1
		if x.Op != Op386MOVWstoreconst {
			break
		}
		a := auxIntToValAndOff(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		if p != x.Args[0] || !(x.Uses == 1 && a.Off()+2 == c.Off() && clobber(x)) {
			break
		}
		v.reset(Op386MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(int32(a.Val()&0xffff|c.Val()<<16), a.Off32()))
		v.Aux = symToAux(s)
		v.AddArg2(p, mem)
		return true
	}
	// match: (MOVWstoreconst [a] {s} p x:(MOVWstoreconst [c] {s} p mem))
	// cond: x.Uses == 1 && ValAndOff(a).Off() + 2 == ValAndOff(c).Off() && clobber(x)
	// result: (MOVLstoreconst [makeValAndOff32(int32(a.Val()&0xffff | c.Val()<<16), a.Off32())] {s} p mem)
	for {
		a := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		x := v_1
		if x.Op != Op386MOVWstoreconst {
			break
		}
		c := auxIntToValAndOff(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		if p != x.Args[0] || !(x.Uses == 1 && ValAndOff(a).Off()+2 == ValAndOff(c).Off() && clobber(x)) {
			break
		}
		v.reset(Op386MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(int32(a.Val()&0xffff|c.Val()<<16), a.Off32()))
		v.Aux = symToAux(s)
		v.AddArg2(p, mem)
		return true
	}
	// match: (MOVWstoreconst [c] {s} p1 x:(MOVWstoreconst [a] {s} p0 mem))
	// cond: x.Uses == 1 && a.Off() == c.Off() && sequentialAddresses(p0, p1, 2) && clobber(x)
	// result: (MOVLstoreconst [makeValAndOff32(int32(a.Val()&0xffff | c.Val()<<16), a.Off32())] {s} p0 mem)
	for {
		c := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		p1 := v_0
		x := v_1
		if x.Op != Op386MOVWstoreconst {
			break
		}
		a := auxIntToValAndOff(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		p0 := x.Args[0]
		if !(x.Uses == 1 && a.Off() == c.Off() && sequentialAddresses(p0, p1, 2) && clobber(x)) {
			break
		}
		v.reset(Op386MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(int32(a.Val()&0xffff|c.Val()<<16), a.Off32()))
		v.Aux = symToAux(s)
		v.AddArg2(p0, mem)
		return true
	}
	// match: (MOVWstoreconst [a] {s} p0 x:(MOVWstoreconst [c] {s} p1 mem))
	// cond: x.Uses == 1 && a.Off() == c.Off() && sequentialAddresses(p0, p1, 2) && clobber(x)
	// result: (MOVLstoreconst [makeValAndOff32(int32(a.Val()&0xffff | c.Val()<<16), a.Off32())] {s} p0 mem)
	for {
		a := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		p0 := v_0
		x := v_1
		if x.Op != Op386MOVWstoreconst {
			break
		}
		c := auxIntToValAndOff(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		p1 := x.Args[0]
		if !(x.Uses == 1 && a.Off() == c.Off() && sequentialAddresses(p0, p1, 2) && clobber(x)) {
			break
		}
		v.reset(Op386MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(int32(a.Val()&0xffff|c.Val()<<16), a.Off32()))
		v.Aux = symToAux(s)
		v.AddArg2(p0, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MULL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MULL x (MOVLconst [c]))
	// result: (MULLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386MOVLconst {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			v.reset(Op386MULLconst)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (MULL x l:(MOVLload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (MULLload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != Op386MOVLload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(Op386MULLload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValue386_Op386MULLconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MULLconst [c] (MULLconst [d] x))
	// result: (MULLconst [c * d] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386MULLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(Op386MULLconst)
		v.AuxInt = int32ToAuxInt(c * d)
		v.AddArg(x)
		return true
	}
	// match: (MULLconst [-9] x)
	// result: (NEGL (LEAL8 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != -9 {
			break
		}
		x := v_0
		v.reset(Op386NEGL)
		v0 := b.NewValue0(v.Pos, Op386LEAL8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULLconst [-5] x)
	// result: (NEGL (LEAL4 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != -5 {
			break
		}
		x := v_0
		v.reset(Op386NEGL)
		v0 := b.NewValue0(v.Pos, Op386LEAL4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULLconst [-3] x)
	// result: (NEGL (LEAL2 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != -3 {
			break
		}
		x := v_0
		v.reset(Op386NEGL)
		v0 := b.NewValue0(v.Pos, Op386LEAL2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULLconst [-1] x)
	// result: (NEGL x)
	for {
		if auxIntToInt32(v.AuxInt) != -1 {
			break
		}
		x := v_0
		v.reset(Op386NEGL)
		v.AddArg(x)
		return true
	}
	// match: (MULLconst [0] _)
	// result: (MOVLconst [0])
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (MULLconst [1] x)
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (MULLconst [3] x)
	// result: (LEAL2 x x)
	for {
		if auxIntToInt32(v.AuxInt) != 3 {
			break
		}
		x := v_0
		v.reset(Op386LEAL2)
		v.AddArg2(x, x)
		return true
	}
	// match: (MULLconst [5] x)
	// result: (LEAL4 x x)
	for {
		if auxIntToInt32(v.AuxInt) != 5 {
			break
		}
		x := v_0
		v.reset(Op386LEAL4)
		v.AddArg2(x, x)
		return true
	}
	// match: (MULLconst [7] x)
	// result: (LEAL2 x (LEAL2 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 7 {
			break
		}
		x := v_0
		v.reset(Op386LEAL2)
		v0 := b.NewValue0(v.Pos, Op386LEAL2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [9] x)
	// result: (LEAL8 x x)
	for {
		if auxIntToInt32(v.AuxInt) != 9 {
			break
		}
		x := v_0
		v.reset(Op386LEAL8)
		v.AddArg2(x, x)
		return true
	}
	// match: (MULLconst [11] x)
	// result: (LEAL2 x (LEAL4 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 11 {
			break
		}
		x := v_0
		v.reset(Op386LEAL2)
		v0 := b.NewValue0(v.Pos, Op386LEAL4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [13] x)
	// result: (LEAL4 x (LEAL2 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 13 {
			break
		}
		x := v_0
		v.reset(Op386LEAL4)
		v0 := b.NewValue0(v.Pos, Op386LEAL2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [19] x)
	// result: (LEAL2 x (LEAL8 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 19 {
			break
		}
		x := v_0
		v.reset(Op386LEAL2)
		v0 := b.NewValue0(v.Pos, Op386LEAL8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [21] x)
	// result: (LEAL4 x (LEAL4 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 21 {
			break
		}
		x := v_0
		v.reset(Op386LEAL4)
		v0 := b.NewValue0(v.Pos, Op386LEAL4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [25] x)
	// result: (LEAL8 x (LEAL2 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 25 {
			break
		}
		x := v_0
		v.reset(Op386LEAL8)
		v0 := b.NewValue0(v.Pos, Op386LEAL2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [27] x)
	// result: (LEAL8 (LEAL2 <v.Type> x x) (LEAL2 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 27 {
			break
		}
		x := v_0
		v.reset(Op386LEAL8)
		v0 := b.NewValue0(v.Pos, Op386LEAL2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(v0, v0)
		return true
	}
	// match: (MULLconst [37] x)
	// result: (LEAL4 x (LEAL8 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 37 {
			break
		}
		x := v_0
		v.reset(Op386LEAL4)
		v0 := b.NewValue0(v.Pos, Op386LEAL8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [41] x)
	// result: (LEAL8 x (LEAL4 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 41 {
			break
		}
		x := v_0
		v.reset(Op386LEAL8)
		v0 := b.NewValue0(v.Pos, Op386LEAL4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [45] x)
	// result: (LEAL8 (LEAL4 <v.Type> x x) (LEAL4 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 45 {
			break
		}
		x := v_0
		v.reset(Op386LEAL8)
		v0 := b.NewValue0(v.Pos, Op386LEAL4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(v0, v0)
		return true
	}
	// match: (MULLconst [73] x)
	// result: (LEAL8 x (LEAL8 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 73 {
			break
		}
		x := v_0
		v.reset(Op386LEAL8)
		v0 := b.NewValue0(v.Pos, Op386LEAL8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [81] x)
	// result: (LEAL8 (LEAL8 <v.Type> x x) (LEAL8 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 81 {
			break
		}
		x := v_0
		v.reset(Op386LEAL8)
		v0 := b.NewValue0(v.Pos, Op386LEAL8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(v0, v0)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: isPowerOfTwo32(c+1) && c >= 15
	// result: (SUBL (SHLLconst <v.Type> [int32(log32(c+1))] x) x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isPowerOfTwo32(c+1) && c >= 15) {
			break
		}
		v.reset(Op386SUBL)
		v0 := b.NewValue0(v.Pos, Op386SHLLconst, v.Type)
		v0.AuxInt = int32ToAuxInt(int32(log32(c + 1)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: isPowerOfTwo32(c-1) && c >= 17
	// result: (LEAL1 (SHLLconst <v.Type> [int32(log32(c-1))] x) x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isPowerOfTwo32(c-1) && c >= 17) {
			break
		}
		v.reset(Op386LEAL1)
		v0 := b.NewValue0(v.Pos, Op386SHLLconst, v.Type)
		v0.AuxInt = int32ToAuxInt(int32(log32(c - 1)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: isPowerOfTwo32(c-2) && c >= 34
	// result: (LEAL2 (SHLLconst <v.Type> [int32(log32(c-2))] x) x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isPowerOfTwo32(c-2) && c >= 34) {
			break
		}
		v.reset(Op386LEAL2)
		v0 := b.NewValue0(v.Pos, Op386SHLLconst, v.Type)
		v0.AuxInt = int32ToAuxInt(int32(log32(c - 2)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: isPowerOfTwo32(c-4) && c >= 68
	// result: (LEAL4 (SHLLconst <v.Type> [int32(log32(c-4))] x) x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isPowerOfTwo32(c-4) && c >= 68) {
			break
		}
		v.reset(Op386LEAL4)
		v0 := b.NewValue0(v.Pos, Op386SHLLconst, v.Type)
		v0.AuxInt = int32ToAuxInt(int32(log32(c - 4)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: isPowerOfTwo32(c-8) && c >= 136
	// result: (LEAL8 (SHLLconst <v.Type> [int32(log32(c-8))] x) x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isPowerOfTwo32(c-8) && c >= 136) {
			break
		}
		v.reset(Op386LEAL8)
		v0 := b.NewValue0(v.Pos, Op386SHLLconst, v.Type)
		v0.AuxInt = int32ToAuxInt(int32(log32(c - 8)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: c%3 == 0 && isPowerOfTwo32(c/3)
	// result: (SHLLconst [int32(log32(c/3))] (LEAL2 <v.Type> x x))
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(c%3 == 0 && isPowerOfTwo32(c/3)) {
			break
		}
		v.reset(Op386SHLLconst)
		v.AuxInt = int32ToAuxInt(int32(log32(c / 3)))
		v0 := b.NewValue0(v.Pos, Op386LEAL2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: c%5 == 0 && isPowerOfTwo32(c/5)
	// result: (SHLLconst [int32(log32(c/5))] (LEAL4 <v.Type> x x))
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(c%5 == 0 && isPowerOfTwo32(c/5)) {
			break
		}
		v.reset(Op386SHLLconst)
		v.AuxInt = int32ToAuxInt(int32(log32(c / 5)))
		v0 := b.NewValue0(v.Pos, Op386LEAL4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: c%9 == 0 && isPowerOfTwo32(c/9)
	// result: (SHLLconst [int32(log32(c/9))] (LEAL8 <v.Type> x x))
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(c%9 == 0 && isPowerOfTwo32(c/9)) {
			break
		}
		v.reset(Op386SHLLconst)
		v.AuxInt = int32ToAuxInt(int32(log32(c / 9)))
		v0 := b.NewValue0(v.Pos, Op386LEAL8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULLconst [c] (MOVLconst [d]))
	// result: (MOVLconst [c*d])
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(c * d)
		return true
	}
	return false
}
func rewriteValue386_Op386MULLload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MULLload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MULLload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386MULLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (MULLload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MULLload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386MULLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MULSD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MULSD x l:(MOVSDload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (MULSDload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != Op386MOVSDload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(Op386MULSDload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValue386_Op386MULSDload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MULSDload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MULSDload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386MULSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (MULSDload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MULSDload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386MULSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MULSS(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MULSS x l:(MOVSSload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (MULSSload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != Op386MOVSSload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(Op386MULSSload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValue386_Op386MULSSload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MULSSload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MULSSload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386MULSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (MULSSload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MULSSload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386MULSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386NEGL(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NEGL (MOVLconst [c]))
	// result: (MOVLconst [-c])
	for {
		if v_0.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(-c)
		return true
	}
	return false
}
func rewriteValue386_Op386NOTL(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NOTL (MOVLconst [c]))
	// result: (MOVLconst [^c])
	for {
		if v_0.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(^c)
		return true
	}
	return false
}
func rewriteValue386_Op386ORL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ORL x (MOVLconst [c]))
	// result: (ORLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386MOVLconst {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			v.reset(Op386ORLconst)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: ( ORL (SHLLconst [c] x) (SHRLconst [d] x))
	// cond: d == 32-c
	// result: (ROLLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != Op386SHLLconst {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if v_1.Op != Op386SHRLconst {
				continue
			}
			d := auxIntToInt32(v_1.AuxInt)
			if x != v_1.Args[0] || !(d == 32-c) {
				continue
			}
			v.reset(Op386ROLLconst)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: ( ORL <t> (SHLLconst x [c]) (SHRWconst x [d]))
	// cond: c < 16 && d == int16(16-c) && t.Size() == 2
	// result: (ROLWconst x [int16(c)])
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != Op386SHLLconst {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if v_1.Op != Op386SHRWconst {
				continue
			}
			d := auxIntToInt16(v_1.AuxInt)
			if x != v_1.Args[0] || !(c < 16 && d == int16(16-c) && t.Size() == 2) {
				continue
			}
			v.reset(Op386ROLWconst)
			v.AuxInt = int16ToAuxInt(int16(c))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: ( ORL <t> (SHLLconst x [c]) (SHRBconst x [d]))
	// cond: c < 8 && d == int8(8-c) && t.Size() == 1
	// result: (ROLBconst x [int8(c)])
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != Op386SHLLconst {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if v_1.Op != Op386SHRBconst {
				continue
			}
			d := auxIntToInt8(v_1.AuxInt)
			if x != v_1.Args[0] || !(c < 8 && d == int8(8-c) && t.Size() == 1) {
				continue
			}
			v.reset(Op386ROLBconst)
			v.AuxInt = int8ToAuxInt(int8(c))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ORL x l:(MOVLload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (ORLload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != Op386MOVLload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(Op386ORLload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	// match: (ORL x x)
	// result: x
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (ORL x0:(MOVBload [i0] {s} p mem) s0:(SHLLconst [8] x1:(MOVBload [i1] {s} p mem)))
	// cond: i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0, x1, s0)
	// result: @mergePoint(b,x0,x1) (MOVWload [i0] {s} p mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			if x0.Op != Op386MOVBload {
				continue
			}
			i0 := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p := x0.Args[0]
			s0 := v_1
			if s0.Op != Op386SHLLconst || auxIntToInt32(s0.AuxInt) != 8 {
				continue
			}
			x1 := s0.Args[0]
			if x1.Op != Op386MOVBload {
				continue
			}
			i1 := auxIntToInt32(x1.AuxInt)
			if auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			if p != x1.Args[0] || mem != x1.Args[1] || !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0, x1, s0)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x1.Pos, Op386MOVWload, typ.UInt16)
			v.copyOf(v0)
			v0.AuxInt = int32ToAuxInt(i0)
			v0.Aux = symToAux(s)
			v0.AddArg2(p, mem)
			return true
		}
		break
	}
	// match: (ORL x0:(MOVBload [i] {s} p0 mem) s0:(SHLLconst [8] x1:(MOVBload [i] {s} p1 mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && sequentialAddresses(p0, p1, 1) && mergePoint(b,x0,x1) != nil && clobber(x0, x1, s0)
	// result: @mergePoint(b,x0,x1) (MOVWload [i] {s} p0 mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			if x0.Op != Op386MOVBload {
				continue
			}
			i := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p0 := x0.Args[0]
			s0 := v_1
			if s0.Op != Op386SHLLconst || auxIntToInt32(s0.AuxInt) != 8 {
				continue
			}
			x1 := s0.Args[0]
			if x1.Op != Op386MOVBload || auxIntToInt32(x1.AuxInt) != i || auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			p1 := x1.Args[0]
			if mem != x1.Args[1] || !(x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && sequentialAddresses(p0, p1, 1) && mergePoint(b, x0, x1) != nil && clobber(x0, x1, s0)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x1.Pos, Op386MOVWload, typ.UInt16)
			v.copyOf(v0)
			v0.AuxInt = int32ToAuxInt(i)
			v0.Aux = symToAux(s)
			v0.AddArg2(p0, mem)
			return true
		}
		break
	}
	// match: (ORL o0:(ORL x0:(MOVWload [i0] {s} p mem) s0:(SHLLconst [16] x1:(MOVBload [i2] {s} p mem))) s1:(SHLLconst [24] x2:(MOVBload [i3] {s} p mem)))
	// cond: i2 == i0+2 && i3 == i0+3 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && o0.Uses == 1 && mergePoint(b,x0,x1,x2) != nil && clobber(x0, x1, x2, s0, s1, o0)
	// result: @mergePoint(b,x0,x1,x2) (MOVLload [i0] {s} p mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			o0 := v_0
			if o0.Op != Op386ORL {
				continue
			}
			_ = o0.Args[1]
			o0_0 := o0.Args[0]
			o0_1 := o0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, o0_0, o0_1 = _i1+1, o0_1, o0_0 {
				x0 := o0_0
				if x0.Op != Op386MOVWload {
					continue
				}
				i0 := auxIntToInt32(x0.AuxInt)
				s := auxToSym(x0.Aux)
				mem := x0.Args[1]
				p := x0.Args[0]
				s0 := o0_1
				if s0.Op != Op386SHLLconst || auxIntToInt32(s0.AuxInt) != 16 {
					continue
				}
				x1 := s0.Args[0]
				if x1.Op != Op386MOVBload {
					continue
				}
				i2 := auxIntToInt32(x1.AuxInt)
				if auxToSym(x1.Aux) != s {
					continue
				}
				_ = x1.Args[1]
				if p != x1.Args[0] || mem != x1.Args[1] {
					continue
				}
				s1 := v_1
				if s1.Op != Op386SHLLconst || auxIntToInt32(s1.AuxInt) != 24 {
					continue
				}
				x2 := s1.Args[0]
				if x2.Op != Op386MOVBload {
					continue
				}
				i3 := auxIntToInt32(x2.AuxInt)
				if auxToSym(x2.Aux) != s {
					continue
				}
				_ = x2.Args[1]
				if p != x2.Args[0] || mem != x2.Args[1] || !(i2 == i0+2 && i3 == i0+3 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && o0.Uses == 1 && mergePoint(b, x0, x1, x2) != nil && clobber(x0, x1, x2, s0, s1, o0)) {
					continue
				}
				b = mergePoint(b, x0, x1, x2)
				v0 := b.NewValue0(x2.Pos, Op386MOVLload, typ.UInt32)
				v.copyOf(v0)
				v0.AuxInt = int32ToAuxInt(i0)
				v0.Aux = symToAux(s)
				v0.AddArg2(p, mem)
				return true
			}
		}
		break
	}
	// match: (ORL o0:(ORL x0:(MOVWload [i] {s} p0 mem) s0:(SHLLconst [16] x1:(MOVBload [i] {s} p1 mem))) s1:(SHLLconst [24] x2:(MOVBload [i] {s} p2 mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && o0.Uses == 1 && sequentialAddresses(p0, p1, 2) && sequentialAddresses(p1, p2, 1) && mergePoint(b,x0,x1,x2) != nil && clobber(x0, x1, x2, s0, s1, o0)
	// result: @mergePoint(b,x0,x1,x2) (MOVLload [i] {s} p0 mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			o0 := v_0
			if o0.Op != Op386ORL {
				continue
			}
			_ = o0.Args[1]
			o0_0 := o0.Args[0]
			o0_1 := o0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, o0_0, o0_1 = _i1+1, o0_1, o0_0 {
				x0 := o0_0
				if x0.Op != Op386MOVWload {
					continue
				}
				i := auxIntToInt32(x0.AuxInt)
				s := auxToSym(x0.Aux)
				mem := x0.Args[1]
				p0 := x0.Args[0]
				s0 := o0_1
				if s0.Op != Op386SHLLconst || auxIntToInt32(s0.AuxInt) != 16 {
					continue
				}
				x1 := s0.Args[0]
				if x1.Op != Op386MOVBload || auxIntToInt32(x1.AuxInt) != i || auxToSym(x1.Aux) != s {
					continue
				}
				_ = x1.Args[1]
				p1 := x1.Args[0]
				if mem != x1.Args[1] {
					continue
				}
				s1 := v_1
				if s1.Op != Op386SHLLconst || auxIntToInt32(s1.AuxInt) != 24 {
					continue
				}
				x2 := s1.Args[0]
				if x2.Op != Op386MOVBload || auxIntToInt32(x2.AuxInt) != i || auxToSym(x2.Aux) != s {
					continue
				}
				_ = x2.Args[1]
				p2 := x2.Args[0]
				if mem != x2.Args[1] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && o0.Uses == 1 && sequentialAddresses(p0, p1, 2) && sequentialAddresses(p1, p2, 1) && mergePoint(b, x0, x1, x2) != nil && clobber(x0, x1, x2, s0, s1, o0)) {
					continue
				}
				b = mergePoint(b, x0, x1, x2)
				v0 := b.NewValue0(x2.Pos, Op386MOVLload, typ.UInt32)
				v.copyOf(v0)
				v0.AuxInt = int32ToAuxInt(i)
				v0.Aux = symToAux(s)
				v0.AddArg2(p0, mem)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValue386_Op386ORLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ORLconst [c] x)
	// cond: c==0
	// result: x
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(c == 0) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (ORLconst [c] _)
	// cond: c==-1
	// result: (MOVLconst [-1])
	for {
		c := auxIntToInt32(v.AuxInt)
		if !(c == -1) {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(-1)
		return true
	}
	// match: (ORLconst [c] (MOVLconst [d]))
	// result: (MOVLconst [c|d])
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(c | d)
		return true
	}
	return false
}
func rewriteValue386_Op386ORLconstmodify(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ORLconstmodify [valoff1] {sym} (ADDLconst [off2] base) mem)
	// cond: valoff1.canAdd32(off2)
	// result: (ORLconstmodify [valoff1.addOffset32(off2)] {sym} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(valoff1.canAdd32(off2)) {
			break
		}
		v.reset(Op386ORLconstmodify)
		v.AuxInt = valAndOffToAuxInt(valoff1.addOffset32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (ORLconstmodify [valoff1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: valoff1.canAdd32(off2) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (ORLconstmodify [valoff1.addOffset32(off2)] {mergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(valoff1.canAdd32(off2) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386ORLconstmodify)
		v.AuxInt = valAndOffToAuxInt(valoff1.addOffset32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ORLload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ORLload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ORLload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386ORLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ORLload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (ORLload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386ORLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ORLmodify(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ORLmodify [off1] {sym} (ADDLconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ORLmodify [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386ORLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (ORLmodify [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (ORLmodify [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386ORLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ROLBconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ROLBconst [c] (ROLBconst [d] x))
	// result: (ROLBconst [(c+d)& 7] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != Op386ROLBconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(Op386ROLBconst)
		v.AuxInt = int8ToAuxInt((c + d) & 7)
		v.AddArg(x)
		return true
	}
	// match: (ROLBconst [0] x)
	// result: x
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValue386_Op386ROLLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ROLLconst [c] (ROLLconst [d] x))
	// result: (ROLLconst [(c+d)&31] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386ROLLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(Op386ROLLconst)
		v.AuxInt = int32ToAuxInt((c + d) & 31)
		v.AddArg(x)
		return true
	}
	// match: (ROLLconst [0] x)
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValue386_Op386ROLWconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ROLWconst [c] (ROLWconst [d] x))
	// result: (ROLWconst [(c+d)&15] x)
	for {
		c := auxIntToInt16(v.AuxInt)
		if v_0.Op != Op386ROLWconst {
			break
		}
		d := auxIntToInt16(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(Op386ROLWconst)
		v.AuxInt = int16ToAuxInt((c + d) & 15)
		v.AddArg(x)
		return true
	}
	// match: (ROLWconst [0] x)
	// result: x
	for {
		if auxIntToInt16(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValue386_Op386SARB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SARB x (MOVLconst [c]))
	// result: (SARBconst [int8(min(int64(c&31),7))] x)
	for {
		x := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(Op386SARBconst)
		v.AuxInt = int8ToAuxInt(int8(min(int64(c&31), 7)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386SARBconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SARBconst x [0])
	// result: x
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (SARBconst [c] (MOVLconst [d]))
	// result: (MOVLconst [d>>uint64(c)])
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(d >> uint64(c))
		return true
	}
	return false
}
func rewriteValue386_Op386SARL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SARL x (MOVLconst [c]))
	// result: (SARLconst [c&31] x)
	for {
		x := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(Op386SARLconst)
		v.AuxInt = int32ToAuxInt(c & 31)
		v.AddArg(x)
		return true
	}
	// match: (SARL x (ANDLconst [31] y))
	// result: (SARL x y)
	for {
		x := v_0
		if v_1.Op != Op386ANDLconst || auxIntToInt32(v_1.AuxInt) != 31 {
			break
		}
		y := v_1.Args[0]
		v.reset(Op386SARL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_Op386SARLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SARLconst x [0])
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (SARLconst [c] (MOVLconst [d]))
	// result: (MOVLconst [d>>uint64(c)])
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(d >> uint64(c))
		return true
	}
	return false
}
func rewriteValue386_Op386SARW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SARW x (MOVLconst [c]))
	// result: (SARWconst [int16(min(int64(c&31),15))] x)
	for {
		x := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(Op386SARWconst)
		v.AuxInt = int16ToAuxInt(int16(min(int64(c&31), 15)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386SARWconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SARWconst x [0])
	// result: x
	for {
		if auxIntToInt16(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (SARWconst [c] (MOVLconst [d]))
	// result: (MOVLconst [d>>uint64(c)])
	for {
		c := auxIntToInt16(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(d >> uint64(c))
		return true
	}
	return false
}
func rewriteValue386_Op386SBBL(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SBBL x (MOVLconst [c]) f)
	// result: (SBBLconst [c] x f)
	for {
		x := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		f := v_2
		v.reset(Op386SBBLconst)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg2(x, f)
		return true
	}
	return false
}
func rewriteValue386_Op386SBBLcarrymask(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SBBLcarrymask (FlagEQ))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagEQ {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SBBLcarrymask (FlagLT_ULT))
	// result: (MOVLconst [-1])
	for {
		if v_0.Op != Op386FlagLT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(-1)
		return true
	}
	// match: (SBBLcarrymask (FlagLT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagLT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SBBLcarrymask (FlagGT_ULT))
	// result: (MOVLconst [-1])
	for {
		if v_0.Op != Op386FlagGT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(-1)
		return true
	}
	// match: (SBBLcarrymask (FlagGT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagGT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386SETA(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SETA (InvertFlags x))
	// result: (SETB x)
	for {
		if v_0.Op != Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(Op386SETB)
		v.AddArg(x)
		return true
	}
	// match: (SETA (FlagEQ))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagEQ {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETA (FlagLT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagLT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETA (FlagLT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagLT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETA (FlagGT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagGT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETA (FlagGT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagGT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValue386_Op386SETAE(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SETAE (InvertFlags x))
	// result: (SETBE x)
	for {
		if v_0.Op != Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(Op386SETBE)
		v.AddArg(x)
		return true
	}
	// match: (SETAE (FlagEQ))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagEQ {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETAE (FlagLT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagLT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETAE (FlagLT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagLT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETAE (FlagGT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagGT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETAE (FlagGT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagGT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValue386_Op386SETB(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SETB (InvertFlags x))
	// result: (SETA x)
	for {
		if v_0.Op != Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(Op386SETA)
		v.AddArg(x)
		return true
	}
	// match: (SETB (FlagEQ))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagEQ {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETB (FlagLT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagLT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETB (FlagLT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagLT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETB (FlagGT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagGT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETB (FlagGT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagGT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386SETBE(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SETBE (InvertFlags x))
	// result: (SETAE x)
	for {
		if v_0.Op != Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(Op386SETAE)
		v.AddArg(x)
		return true
	}
	// match: (SETBE (FlagEQ))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagEQ {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETBE (FlagLT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagLT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETBE (FlagLT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagLT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETBE (FlagGT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagGT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETBE (FlagGT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagGT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386SETEQ(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SETEQ (InvertFlags x))
	// result: (SETEQ x)
	for {
		if v_0.Op != Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(Op386SETEQ)
		v.AddArg(x)
		return true
	}
	// match: (SETEQ (FlagEQ))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagEQ {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETEQ (FlagLT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagLT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETEQ (FlagLT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagLT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETEQ (FlagGT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagGT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETEQ (FlagGT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagGT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386SETG(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SETG (InvertFlags x))
	// result: (SETL x)
	for {
		if v_0.Op != Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(Op386SETL)
		v.AddArg(x)
		return true
	}
	// match: (SETG (FlagEQ))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagEQ {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETG (FlagLT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagLT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETG (FlagLT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagLT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETG (FlagGT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagGT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETG (FlagGT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagGT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValue386_Op386SETGE(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SETGE (InvertFlags x))
	// result: (SETLE x)
	for {
		if v_0.Op != Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(Op386SETLE)
		v.AddArg(x)
		return true
	}
	// match: (SETGE (FlagEQ))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagEQ {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETGE (FlagLT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagLT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETGE (FlagLT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagLT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETGE (FlagGT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagGT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETGE (FlagGT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagGT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValue386_Op386SETL(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SETL (InvertFlags x))
	// result: (SETG x)
	for {
		if v_0.Op != Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(Op386SETG)
		v.AddArg(x)
		return true
	}
	// match: (SETL (FlagEQ))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagEQ {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETL (FlagLT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagLT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETL (FlagLT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagLT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETL (FlagGT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagGT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETL (FlagGT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagGT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386SETLE(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SETLE (InvertFlags x))
	// result: (SETGE x)
	for {
		if v_0.Op != Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(Op386SETGE)
		v.AddArg(x)
		return true
	}
	// match: (SETLE (FlagEQ))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagEQ {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETLE (FlagLT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagLT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETLE (FlagLT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagLT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETLE (FlagGT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagGT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETLE (FlagGT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagGT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386SETNE(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SETNE (InvertFlags x))
	// result: (SETNE x)
	for {
		if v_0.Op != Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(Op386SETNE)
		v.AddArg(x)
		return true
	}
	// match: (SETNE (FlagEQ))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != Op386FlagEQ {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETNE (FlagLT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagLT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETNE (FlagLT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagLT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETNE (FlagGT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagGT_ULT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETNE (FlagGT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != Op386FlagGT_UGT {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValue386_Op386SHLL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SHLL x (MOVLconst [c]))
	// result: (SHLLconst [c&31] x)
	for {
		x := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(Op386SHLLconst)
		v.AuxInt = int32ToAuxInt(c & 31)
		v.AddArg(x)
		return true
	}
	// match: (SHLL x (ANDLconst [31] y))
	// result: (SHLL x y)
	for {
		x := v_0
		if v_1.Op != Op386ANDLconst || auxIntToInt32(v_1.AuxInt) != 31 {
			break
		}
		y := v_1.Args[0]
		v.reset(Op386SHLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_Op386SHLLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SHLLconst x [0])
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValue386_Op386SHRB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SHRB x (MOVLconst [c]))
	// cond: c&31 < 8
	// result: (SHRBconst [int8(c&31)] x)
	for {
		x := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(c&31 < 8) {
			break
		}
		v.reset(Op386SHRBconst)
		v.AuxInt = int8ToAuxInt(int8(c & 31))
		v.AddArg(x)
		return true
	}
	// match: (SHRB _ (MOVLconst [c]))
	// cond: c&31 >= 8
	// result: (MOVLconst [0])
	for {
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(c&31 >= 8) {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386SHRBconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SHRBconst x [0])
	// result: x
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValue386_Op386SHRL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SHRL x (MOVLconst [c]))
	// result: (SHRLconst [c&31] x)
	for {
		x := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(Op386SHRLconst)
		v.AuxInt = int32ToAuxInt(c & 31)
		v.AddArg(x)
		return true
	}
	// match: (SHRL x (ANDLconst [31] y))
	// result: (SHRL x y)
	for {
		x := v_0
		if v_1.Op != Op386ANDLconst || auxIntToInt32(v_1.AuxInt) != 31 {
			break
		}
		y := v_1.Args[0]
		v.reset(Op386SHRL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_Op386SHRLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SHRLconst x [0])
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValue386_Op386SHRW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SHRW x (MOVLconst [c]))
	// cond: c&31 < 16
	// result: (SHRWconst [int16(c&31)] x)
	for {
		x := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(c&31 < 16) {
			break
		}
		v.reset(Op386SHRWconst)
		v.AuxInt = int16ToAuxInt(int16(c & 31))
		v.AddArg(x)
		return true
	}
	// match: (SHRW _ (MOVLconst [c]))
	// cond: c&31 >= 16
	// result: (MOVLconst [0])
	for {
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(c&31 >= 16) {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386SHRWconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SHRWconst x [0])
	// result: x
	for {
		if auxIntToInt16(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValue386_Op386SUBL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUBL x (MOVLconst [c]))
	// result: (SUBLconst x [c])
	for {
		x := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(Op386SUBLconst)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SUBL (MOVLconst [c]) x)
	// result: (NEGL (SUBLconst <v.Type> x [c]))
	for {
		if v_0.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_1
		v.reset(Op386NEGL)
		v0 := b.NewValue0(v.Pos, Op386SUBLconst, v.Type)
		v0.AuxInt = int32ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SUBL x l:(MOVLload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (SUBLload x [off] {sym} ptr mem)
	for {
		x := v_0
		l := v_1
		if l.Op != Op386MOVLload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
			break
		}
		v.reset(Op386SUBLload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(x, ptr, mem)
		return true
	}
	// match: (SUBL x x)
	// result: (MOVLconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386SUBLcarry(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBLcarry x (MOVLconst [c]))
	// result: (SUBLconstcarry [c] x)
	for {
		x := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(Op386SUBLconstcarry)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386SUBLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SUBLconst [c] x)
	// cond: c==0
	// result: x
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(c == 0) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (SUBLconst [c] x)
	// result: (ADDLconst [-c] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		v.reset(Op386ADDLconst)
		v.AuxInt = int32ToAuxInt(-c)
		v.AddArg(x)
		return true
	}
}
func rewriteValue386_Op386SUBLload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (SUBLload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SUBLload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386SUBLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (SUBLload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (SUBLload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386SUBLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386SUBLmodify(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (SUBLmodify [off1] {sym} (ADDLconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SUBLmodify [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386SUBLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SUBLmodify [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (SUBLmodify [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386SUBLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386SUBSD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBSD x l:(MOVSDload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (SUBSDload x [off] {sym} ptr mem)
	for {
		x := v_0
		l := v_1
		if l.Op != Op386MOVSDload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
			break
		}
		v.reset(Op386SUBSDload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(x, ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386SUBSDload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (SUBSDload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SUBSDload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386SUBSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (SUBSDload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (SUBSDload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386SUBSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386SUBSS(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBSS x l:(MOVSSload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (SUBSSload x [off] {sym} ptr mem)
	for {
		x := v_0
		l := v_1
		if l.Op != Op386MOVSSload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
			break
		}
		v.reset(Op386SUBSSload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(x, ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386SUBSSload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (SUBSSload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SUBSSload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386SUBSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (SUBSSload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (SUBSSload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386SUBSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386XORL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XORL x (MOVLconst [c]))
	// result: (XORLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != Op386MOVLconst {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			v.reset(Op386XORLconst)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (XORL (SHLLconst [c] x) (SHRLconst [d] x))
	// cond: d == 32-c
	// result: (ROLLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != Op386SHLLconst {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if v_1.Op != Op386SHRLconst {
				continue
			}
			d := auxIntToInt32(v_1.AuxInt)
			if x != v_1.Args[0] || !(d == 32-c) {
				continue
			}
			v.reset(Op386ROLLconst)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (XORL <t> (SHLLconst x [c]) (SHRWconst x [d]))
	// cond: c < 16 && d == int16(16-c) && t.Size() == 2
	// result: (ROLWconst x [int16(c)])
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != Op386SHLLconst {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if v_1.Op != Op386SHRWconst {
				continue
			}
			d := auxIntToInt16(v_1.AuxInt)
			if x != v_1.Args[0] || !(c < 16 && d == int16(16-c) && t.Size() == 2) {
				continue
			}
			v.reset(Op386ROLWconst)
			v.AuxInt = int16ToAuxInt(int16(c))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (XORL <t> (SHLLconst x [c]) (SHRBconst x [d]))
	// cond: c < 8 && d == int8(8-c) && t.Size() == 1
	// result: (ROLBconst x [int8(c)])
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != Op386SHLLconst {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if v_1.Op != Op386SHRBconst {
				continue
			}
			d := auxIntToInt8(v_1.AuxInt)
			if x != v_1.Args[0] || !(c < 8 && d == int8(8-c) && t.Size() == 1) {
				continue
			}
			v.reset(Op386ROLBconst)
			v.AuxInt = int8ToAuxInt(int8(c))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (XORL x l:(MOVLload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (XORLload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != Op386MOVLload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(Op386XORLload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	// match: (XORL x x)
	// result: (MOVLconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386XORLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (XORLconst [c] (XORLconst [d] x))
	// result: (XORLconst [c ^ d] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386XORLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(Op386XORLconst)
		v.AuxInt = int32ToAuxInt(c ^ d)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [c] x)
	// cond: c==0
	// result: x
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(c == 0) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (XORLconst [c] (MOVLconst [d]))
	// result: (MOVLconst [c^d])
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != Op386MOVLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(c ^ d)
		return true
	}
	return false
}
func rewriteValue386_Op386XORLconstmodify(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (XORLconstmodify [valoff1] {sym} (ADDLconst [off2] base) mem)
	// cond: valoff1.canAdd32(off2)
	// result: (XORLconstmodify [valoff1.addOffset32(off2)] {sym} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(valoff1.canAdd32(off2)) {
			break
		}
		v.reset(Op386XORLconstmodify)
		v.AuxInt = valAndOffToAuxInt(valoff1.addOffset32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (XORLconstmodify [valoff1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: valoff1.canAdd32(off2) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (XORLconstmodify [valoff1.addOffset32(off2)] {mergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(valoff1.canAdd32(off2) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386XORLconstmodify)
		v.AuxInt = valAndOffToAuxInt(valoff1.addOffset32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386XORLload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (XORLload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (XORLload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386XORLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (XORLload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (XORLload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386XORLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386XORLmodify(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (XORLmodify [off1] {sym} (ADDLconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (XORLmodify [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != Op386ADDLconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(Op386XORLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (XORLmodify [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (XORLmodify [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != Op386LEAL {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && (base.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(Op386XORLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValue386_OpAddr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Addr {sym} base)
	// result: (LEAL {sym} base)
	for {
		sym := auxToSym(v.Aux)
		base := v_0
		v.reset(Op386LEAL)
		v.Aux = symToAux(sym)
		v.AddArg(base)
		return true
	}
}
func rewriteValue386_OpConst16(v *Value) bool {
	// match: (Const16 [c])
	// result: (MOVLconst [int32(c)])
	for {
		c := auxIntToInt16(v.AuxInt)
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(int32(c))
		return true
	}
}
func rewriteValue386_OpConst8(v *Value) bool {
	// match: (Const8 [c])
	// result: (MOVLconst [int32(c)])
	for {
		c := auxIntToInt8(v.AuxInt)
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(int32(c))
		return true
	}
}
func rewriteValue386_OpConstBool(v *Value) bool {
	// match: (ConstBool [c])
	// result: (MOVLconst [b2i32(c)])
	for {
		c := auxIntToBool(v.AuxInt)
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(b2i32(c))
		return true
	}
}
func rewriteValue386_OpConstNil(v *Value) bool {
	// match: (ConstNil)
	// result: (MOVLconst [0])
	for {
		v.reset(Op386MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
}
func rewriteValue386_OpCtz16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz16 x)
	// result: (BSFL (ORLconst <typ.UInt32> [0x10000] x))
	for {
		x := v_0
		v.reset(Op386BSFL)
		v0 := b.NewValue0(v.Pos, Op386ORLconst, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(0x10000)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpDiv8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// result: (DIVW (SignExt8to16 x) (SignExt8to16 y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386DIVW)
		v0 := b.NewValue0(v.Pos, OpSignExt8to16, typ.Int16)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt8to16, typ.Int16)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue386_OpDiv8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// result: (DIVWU (ZeroExt8to16 x) (ZeroExt8to16 y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386DIVWU)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to16, typ.UInt16)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to16, typ.UInt16)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue386_OpEq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq16 x y)
	// result: (SETEQ (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETEQ)
		v0 := b.NewValue0(v.Pos, Op386CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpEq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32 x y)
	// result: (SETEQ (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETEQ)
		v0 := b.NewValue0(v.Pos, Op386CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpEq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32F x y)
	// result: (SETEQF (UCOMISS x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETEQF)
		v0 := b.NewValue0(v.Pos, Op386UCOMISS, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpEq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64F x y)
	// result: (SETEQF (UCOMISD x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETEQF)
		v0 := b.NewValue0(v.Pos, Op386UCOMISD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpEq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq8 x y)
	// result: (SETEQ (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETEQ)
		v0 := b.NewValue0(v.Pos, Op386CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpEqB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (EqB x y)
	// result: (SETEQ (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETEQ)
		v0 := b.NewValue0(v.Pos, Op386CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpEqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (EqPtr x y)
	// result: (SETEQ (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETEQ)
		v0 := b.NewValue0(v.Pos, Op386CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpIsInBounds(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsInBounds idx len)
	// result: (SETB (CMPL idx len))
	for {
		idx := v_0
		len := v_1
		v.reset(Op386SETB)
		v0 := b.NewValue0(v.Pos, Op386CMPL, types.TypeFlags)
		v0.AddArg2(idx, len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpIsNonNil(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsNonNil p)
	// result: (SETNE (TESTL p p))
	for {
		p := v_0
		v.reset(Op386SETNE)
		v0 := b.NewValue0(v.Pos, Op386TESTL, types.TypeFlags)
		v0.AddArg2(p, p)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpIsSliceInBounds(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsSliceInBounds idx len)
	// result: (SETBE (CMPL idx len))
	for {
		idx := v_0
		len := v_1
		v.reset(Op386SETBE)
		v0 := b.NewValue0(v.Pos, Op386CMPL, types.TypeFlags)
		v0.AddArg2(idx, len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq16 x y)
	// result: (SETLE (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETLE)
		v0 := b.NewValue0(v.Pos, Op386CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLeq16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq16U x y)
	// result: (SETBE (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETBE)
		v0 := b.NewValue0(v.Pos, Op386CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32 x y)
	// result: (SETLE (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETLE)
		v0 := b.NewValue0(v.Pos, Op386CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32F x y)
	// result: (SETGEF (UCOMISS y x))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETGEF)
		v0 := b.NewValue0(v.Pos, Op386UCOMISS, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLeq32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32U x y)
	// result: (SETBE (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETBE)
		v0 := b.NewValue0(v.Pos, Op386CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64F x y)
	// result: (SETGEF (UCOMISD y x))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETGEF)
		v0 := b.NewValue0(v.Pos, Op386UCOMISD, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq8 x y)
	// result: (SETLE (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETLE)
		v0 := b.NewValue0(v.Pos, Op386CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLeq8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq8U x y)
	// result: (SETBE (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETBE)
		v0 := b.NewValue0(v.Pos, Op386CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLess16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less16 x y)
	// result: (SETL (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETL)
		v0 := b.NewValue0(v.Pos, Op386CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLess16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less16U x y)
	// result: (SETB (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETB)
		v0 := b.NewValue0(v.Pos, Op386CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLess32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32 x y)
	// result: (SETL (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETL)
		v0 := b.NewValue0(v.Pos, Op386CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLess32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32F x y)
	// result: (SETGF (UCOMISS y x))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETGF)
		v0 := b.NewValue0(v.Pos, Op386UCOMISS, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLess32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32U x y)
	// result: (SETB (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETB)
		v0 := b.NewValue0(v.Pos, Op386CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLess64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64F x y)
	// result: (SETGF (UCOMISD y x))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETGF)
		v0 := b.NewValue0(v.Pos, Op386UCOMISD, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLess8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less8 x y)
	// result: (SETL (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETL)
		v0 := b.NewValue0(v.Pos, Op386CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLess8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less8U x y)
	// result: (SETB (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETB)
		v0 := b.NewValue0(v.Pos, Op386CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLoad(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Load <t> ptr mem)
	// cond: (is32BitInt(t) || isPtr(t))
	// result: (MOVLload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is32BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(Op386MOVLload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is16BitInt(t)
	// result: (MOVWload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is16BitInt(t)) {
			break
		}
		v.reset(Op386MOVWload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (t.IsBoolean() || is8BitInt(t))
	// result: (MOVBload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.IsBoolean() || is8BitInt(t)) {
			break
		}
		v.reset(Op386MOVBload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is32BitFloat(t)
	// result: (MOVSSload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is32BitFloat(t)) {
			break
		}
		v.reset(Op386MOVSSload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is64BitFloat(t)
	// result: (MOVSDload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is64BitFloat(t)) {
			break
		}
		v.reset(Op386MOVSDload)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_OpLocalAddr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (LocalAddr {sym} base _)
	// result: (LEAL {sym} base)
	for {
		sym := auxToSym(v.Aux)
		base := v_0
		v.reset(Op386LEAL)
		v.Aux = symToAux(sym)
		v.AddArg(base)
		return true
	}
}
func rewriteValue386_OpLsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh16x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPWconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386ANDL)
		v0 := b.NewValue0(v.Pos, Op386SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, Op386CMPWconst, types.TypeFlags)
		v2.AuxInt = int16ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh16x16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SHLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpLsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh16x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPLconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386ANDL)
		v0 := b.NewValue0(v.Pos, Op386SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, Op386CMPLconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh16x32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SHLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpLsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Lsh16x64 x (Const64 [c]))
	// cond: uint64(c) < 16
	// result: (SHLLconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.reset(Op386SHLLconst)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Lsh16x64 _ (Const64 [c]))
	// cond: uint64(c) >= 16
	// result: (Const16 [0])
	for {
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 16) {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_OpLsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh16x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPBconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386ANDL)
		v0 := b.NewValue0(v.Pos, Op386SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, Op386CMPBconst, types.TypeFlags)
		v2.AuxInt = int8ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh16x8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SHLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpLsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh32x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPWconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386ANDL)
		v0 := b.NewValue0(v.Pos, Op386SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, Op386CMPWconst, types.TypeFlags)
		v2.AuxInt = int16ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh32x16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SHLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpLsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh32x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPLconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386ANDL)
		v0 := b.NewValue0(v.Pos, Op386SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, Op386CMPLconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh32x32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SHLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpLsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Lsh32x64 x (Const64 [c]))
	// cond: uint64(c) < 32
	// result: (SHLLconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.reset(Op386SHLLconst)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Lsh32x64 _ (Const64 [c]))
	// cond: uint64(c) >= 32
	// result: (Const32 [0])
	for {
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 32) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_OpLsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh32x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPBconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386ANDL)
		v0 := b.NewValue0(v.Pos, Op386SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, Op386CMPBconst, types.TypeFlags)
		v2.AuxInt = int8ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh32x8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SHLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpLsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh8x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPWconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386ANDL)
		v0 := b.NewValue0(v.Pos, Op386SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, Op386CMPWconst, types.TypeFlags)
		v2.AuxInt = int16ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh8x16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SHLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpLsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh8x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPLconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386ANDL)
		v0 := b.NewValue0(v.Pos, Op386SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, Op386CMPLconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh8x32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SHLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpLsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Lsh8x64 x (Const64 [c]))
	// cond: uint64(c) < 8
	// result: (SHLLconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.reset(Op386SHLLconst)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Lsh8x64 _ (Const64 [c]))
	// cond: uint64(c) >= 8
	// result: (Const8 [0])
	for {
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 8) {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_OpLsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh8x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPBconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386ANDL)
		v0 := b.NewValue0(v.Pos, Op386SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, Op386CMPBconst, types.TypeFlags)
		v2.AuxInt = int8ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh8x8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SHLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpMod8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8 x y)
	// result: (MODW (SignExt8to16 x) (SignExt8to16 y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386MODW)
		v0 := b.NewValue0(v.Pos, OpSignExt8to16, typ.Int16)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt8to16, typ.Int16)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue386_OpMod8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8u x y)
	// result: (MODWU (ZeroExt8to16 x) (ZeroExt8to16 y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386MODWU)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to16, typ.UInt16)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to16, typ.UInt16)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue386_OpMove(v *Value) bool {
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
		v.reset(Op386MOVBstore)
		v0 := b.NewValue0(v.Pos, Op386MOVBload, typ.UInt8)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [2] dst src mem)
	// result: (MOVWstore dst (MOVWload src mem) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 2 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(Op386MOVWstore)
		v0 := b.NewValue0(v.Pos, Op386MOVWload, typ.UInt16)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [4] dst src mem)
	// result: (MOVLstore dst (MOVLload src mem) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 4 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(Op386MOVLstore)
		v0 := b.NewValue0(v.Pos, Op386MOVLload, typ.UInt32)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [3] dst src mem)
	// result: (MOVBstore [2] dst (MOVBload [2] src mem) (MOVWstore dst (MOVWload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 3 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(Op386MOVBstore)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, Op386MOVBload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, Op386MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, Op386MOVWload, typ.UInt16)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [5] dst src mem)
	// result: (MOVBstore [4] dst (MOVBload [4] src mem) (MOVLstore dst (MOVLload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 5 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(Op386MOVBstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, Op386MOVBload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, Op386MOVLstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, Op386MOVLload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [6] dst src mem)
	// result: (MOVWstore [4] dst (MOVWload [4] src mem) (MOVLstore dst (MOVLload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 6 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(Op386MOVWstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, Op386MOVWload, typ.UInt16)
		v0.AuxInt = int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, Op386MOVLstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, Op386MOVLload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [7] dst src mem)
	// result: (MOVLstore [3] dst (MOVLload [3] src mem) (MOVLstore dst (MOVLload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 7 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(Op386MOVLstore)
		v.AuxInt = int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, Op386MOVLload, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(3)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, Op386MOVLstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, Op386MOVLload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [8] dst src mem)
	// result: (MOVLstore [4] dst (MOVLload [4] src mem) (MOVLstore dst (MOVLload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(Op386MOVLstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, Op386MOVLload, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, Op386MOVLstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, Op386MOVLload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 8 && s%4 != 0
	// result: (Move [s-s%4] (ADDLconst <dst.Type> dst [int32(s%4)]) (ADDLconst <src.Type> src [int32(s%4)]) (MOVLstore dst (MOVLload src mem) mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 8 && s%4 != 0) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(s - s%4)
		v0 := b.NewValue0(v.Pos, Op386ADDLconst, dst.Type)
		v0.AuxInt = int32ToAuxInt(int32(s % 4))
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, Op386ADDLconst, src.Type)
		v1.AuxInt = int32ToAuxInt(int32(s % 4))
		v1.AddArg(src)
		v2 := b.NewValue0(v.Pos, Op386MOVLstore, types.TypeMem)
		v3 := b.NewValue0(v.Pos, Op386MOVLload, typ.UInt32)
		v3.AddArg2(src, mem)
		v2.AddArg3(dst, v3, mem)
		v.AddArg3(v0, v1, v2)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 8 && s <= 4*128 && s%4 == 0 && !config.noDuffDevice && logLargeCopy(v, s)
	// result: (DUFFCOPY [10*(128-s/4)] dst src mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 8 && s <= 4*128 && s%4 == 0 && !config.noDuffDevice && logLargeCopy(v, s)) {
			break
		}
		v.reset(Op386DUFFCOPY)
		v.AuxInt = int64ToAuxInt(10 * (128 - s/4))
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: (s > 4*128 || config.noDuffDevice) && s%4 == 0 && logLargeCopy(v, s)
	// result: (REPMOVSL dst src (MOVLconst [int32(s/4)]) mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !((s > 4*128 || config.noDuffDevice) && s%4 == 0 && logLargeCopy(v, s)) {
			break
		}
		v.reset(Op386REPMOVSL)
		v0 := b.NewValue0(v.Pos, Op386MOVLconst, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(int32(s / 4))
		v.AddArg4(dst, src, v0, mem)
		return true
	}
	return false
}
func rewriteValue386_OpNeg32F(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg32F x)
	// result: (PXOR x (MOVSSconst <typ.Float32> [float32(math.Copysign(0, -1))]))
	for {
		x := v_0
		v.reset(Op386PXOR)
		v0 := b.NewValue0(v.Pos, Op386MOVSSconst, typ.Float32)
		v0.AuxInt = float32ToAuxInt(float32(math.Copysign(0, -1)))
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue386_OpNeg64F(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg64F x)
	// result: (PXOR x (MOVSDconst <typ.Float64> [math.Copysign(0, -1)]))
	for {
		x := v_0
		v.reset(Op386PXOR)
		v0 := b.NewValue0(v.Pos, Op386MOVSDconst, typ.Float64)
		v0.AuxInt = float64ToAuxInt(math.Copysign(0, -1))
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue386_OpNeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq16 x y)
	// result: (SETNE (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETNE)
		v0 := b.NewValue0(v.Pos, Op386CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpNeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32 x y)
	// result: (SETNE (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETNE)
		v0 := b.NewValue0(v.Pos, Op386CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpNeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32F x y)
	// result: (SETNEF (UCOMISS x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETNEF)
		v0 := b.NewValue0(v.Pos, Op386UCOMISS, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpNeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64F x y)
	// result: (SETNEF (UCOMISD x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETNEF)
		v0 := b.NewValue0(v.Pos, Op386UCOMISD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpNeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq8 x y)
	// result: (SETNE (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETNE)
		v0 := b.NewValue0(v.Pos, Op386CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpNeqB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (NeqB x y)
	// result: (SETNE (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETNE)
		v0 := b.NewValue0(v.Pos, Op386CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpNeqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (NeqPtr x y)
	// result: (SETNE (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.reset(Op386SETNE)
		v0 := b.NewValue0(v.Pos, Op386CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpNot(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Not x)
	// result: (XORLconst [1] x)
	for {
		x := v_0
		v.reset(Op386XORLconst)
		v.AuxInt = int32ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValue386_OpOffPtr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (OffPtr [off] ptr)
	// result: (ADDLconst [int32(off)] ptr)
	for {
		off := auxIntToInt64(v.AuxInt)
		ptr := v_0
		v.reset(Op386ADDLconst)
		v.AuxInt = int32ToAuxInt(int32(off))
		v.AddArg(ptr)
		return true
	}
}
func rewriteValue386_OpPanicBounds(v *Value) bool {
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
		v.reset(Op386LoweredPanicBoundsA)
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
		v.reset(Op386LoweredPanicBoundsB)
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
		v.reset(Op386LoweredPanicBoundsC)
		v.AuxInt = int64ToAuxInt(kind)
		v.AddArg3(x, y, mem)
		return true
	}
	return false
}
func rewriteValue386_OpPanicExtend(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (PanicExtend [kind] hi lo y mem)
	// cond: boundsABI(kind) == 0
	// result: (LoweredPanicExtendA [kind] hi lo y mem)
	for {
		kind := auxIntToInt64(v.AuxInt)
		hi := v_0
		lo := v_1
		y := v_2
		mem := v_3
		if !(boundsABI(kind) == 0) {
			break
		}
		v.reset(Op386LoweredPanicExtendA)
		v.AuxInt = int64ToAuxInt(kind)
		v.AddArg4(hi, lo, y, mem)
		return true
	}
	// match: (PanicExtend [kind] hi lo y mem)
	// cond: boundsABI(kind) == 1
	// result: (LoweredPanicExtendB [kind] hi lo y mem)
	for {
		kind := auxIntToInt64(v.AuxInt)
		hi := v_0
		lo := v_1
		y := v_2
		mem := v_3
		if !(boundsABI(kind) == 1) {
			break
		}
		v.reset(Op386LoweredPanicExtendB)
		v.AuxInt = int64ToAuxInt(kind)
		v.AddArg4(hi, lo, y, mem)
		return true
	}
	// match: (PanicExtend [kind] hi lo y mem)
	// cond: boundsABI(kind) == 2
	// result: (LoweredPanicExtendC [kind] hi lo y mem)
	for {
		kind := auxIntToInt64(v.AuxInt)
		hi := v_0
		lo := v_1
		y := v_2
		mem := v_3
		if !(boundsABI(kind) == 2) {
			break
		}
		v.reset(Op386LoweredPanicExtendC)
		v.AuxInt = int64ToAuxInt(kind)
		v.AddArg4(hi, lo, y, mem)
		return true
	}
	return false
}
func rewriteValue386_OpRotateLeft16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (RotateLeft16 x (MOVLconst [c]))
	// result: (ROLWconst [int16(c&15)] x)
	for {
		x := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(Op386ROLWconst)
		v.AuxInt = int16ToAuxInt(int16(c & 15))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_OpRotateLeft32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (RotateLeft32 x (MOVLconst [c]))
	// result: (ROLLconst [c&31] x)
	for {
		x := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(Op386ROLLconst)
		v.AuxInt = int32ToAuxInt(c & 31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_OpRotateLeft8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (RotateLeft8 x (MOVLconst [c]))
	// result: (ROLBconst [int8(c&7)] x)
	for {
		x := v_0
		if v_1.Op != Op386MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(Op386ROLBconst)
		v.AuxInt = int8ToAuxInt(int8(c & 7))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_OpRsh16Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16Ux16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHRW <t> x y) (SBBLcarrymask <t> (CMPWconst y [16])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386ANDL)
		v0 := b.NewValue0(v.Pos, Op386SHRW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, Op386CMPWconst, types.TypeFlags)
		v2.AuxInt = int16ToAuxInt(16)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh16Ux16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SHRW <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SHRW)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh16Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16Ux32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHRW <t> x y) (SBBLcarrymask <t> (CMPLconst y [16])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386ANDL)
		v0 := b.NewValue0(v.Pos, Op386SHRW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, Op386CMPLconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(16)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh16Ux32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SHRW <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SHRW)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh16Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Rsh16Ux64 x (Const64 [c]))
	// cond: uint64(c) < 16
	// result: (SHRWconst x [int16(c)])
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.reset(Op386SHRWconst)
		v.AuxInt = int16ToAuxInt(int16(c))
		v.AddArg(x)
		return true
	}
	// match: (Rsh16Ux64 _ (Const64 [c]))
	// cond: uint64(c) >= 16
	// result: (Const16 [0])
	for {
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 16) {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int16ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_OpRsh16Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16Ux8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHRW <t> x y) (SBBLcarrymask <t> (CMPBconst y [16])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386ANDL)
		v0 := b.NewValue0(v.Pos, Op386SHRW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, Op386CMPBconst, types.TypeFlags)
		v2.AuxInt = int8ToAuxInt(16)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh16Ux8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SHRW <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SHRW)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SARW <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPWconst y [16])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SARW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, Op386ORL, y.Type)
		v1 := b.NewValue0(v.Pos, Op386NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, Op386SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, Op386CMPWconst, types.TypeFlags)
		v3.AuxInt = int16ToAuxInt(16)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16x16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SARW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SARW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SARW <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPLconst y [16])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SARW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, Op386ORL, y.Type)
		v1 := b.NewValue0(v.Pos, Op386NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, Op386SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, Op386CMPLconst, types.TypeFlags)
		v3.AuxInt = int32ToAuxInt(16)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16x32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SARW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SARW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Rsh16x64 x (Const64 [c]))
	// cond: uint64(c) < 16
	// result: (SARWconst x [int16(c)])
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.reset(Op386SARWconst)
		v.AuxInt = int16ToAuxInt(int16(c))
		v.AddArg(x)
		return true
	}
	// match: (Rsh16x64 x (Const64 [c]))
	// cond: uint64(c) >= 16
	// result: (SARWconst x [15])
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 16) {
			break
		}
		v.reset(Op386SARWconst)
		v.AuxInt = int16ToAuxInt(15)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_OpRsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SARW <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPBconst y [16])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SARW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, Op386ORL, y.Type)
		v1 := b.NewValue0(v.Pos, Op386NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, Op386SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, Op386CMPBconst, types.TypeFlags)
		v3.AuxInt = int8ToAuxInt(16)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16x8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SARW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SARW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh32Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32Ux16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHRL <t> x y) (SBBLcarrymask <t> (CMPWconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386ANDL)
		v0 := b.NewValue0(v.Pos, Op386SHRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, Op386CMPWconst, types.TypeFlags)
		v2.AuxInt = int16ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh32Ux16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SHRL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SHRL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh32Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32Ux32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHRL <t> x y) (SBBLcarrymask <t> (CMPLconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386ANDL)
		v0 := b.NewValue0(v.Pos, Op386SHRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, Op386CMPLconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh32Ux32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SHRL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SHRL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh32Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Rsh32Ux64 x (Const64 [c]))
	// cond: uint64(c) < 32
	// result: (SHRLconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.reset(Op386SHRLconst)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Rsh32Ux64 _ (Const64 [c]))
	// cond: uint64(c) >= 32
	// result: (Const32 [0])
	for {
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 32) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_OpRsh32Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32Ux8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHRL <t> x y) (SBBLcarrymask <t> (CMPBconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386ANDL)
		v0 := b.NewValue0(v.Pos, Op386SHRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, Op386CMPBconst, types.TypeFlags)
		v2.AuxInt = int8ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh32Ux8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SHRL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SHRL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SARL <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPWconst y [32])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SARL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, Op386ORL, y.Type)
		v1 := b.NewValue0(v.Pos, Op386NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, Op386SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, Op386CMPWconst, types.TypeFlags)
		v3.AuxInt = int16ToAuxInt(32)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32x16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SARL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SARL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SARL <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPLconst y [32])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SARL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, Op386ORL, y.Type)
		v1 := b.NewValue0(v.Pos, Op386NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, Op386SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, Op386CMPLconst, types.TypeFlags)
		v3.AuxInt = int32ToAuxInt(32)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32x32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SARL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SARL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Rsh32x64 x (Const64 [c]))
	// cond: uint64(c) < 32
	// result: (SARLconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.reset(Op386SARLconst)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Rsh32x64 x (Const64 [c]))
	// cond: uint64(c) >= 32
	// result: (SARLconst x [31])
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 32) {
			break
		}
		v.reset(Op386SARLconst)
		v.AuxInt = int32ToAuxInt(31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_OpRsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SARL <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPBconst y [32])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SARL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, Op386ORL, y.Type)
		v1 := b.NewValue0(v.Pos, Op386NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, Op386SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, Op386CMPBconst, types.TypeFlags)
		v3.AuxInt = int8ToAuxInt(32)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32x8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SARL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SARL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh8Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8Ux16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHRB <t> x y) (SBBLcarrymask <t> (CMPWconst y [8])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386ANDL)
		v0 := b.NewValue0(v.Pos, Op386SHRB, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, Op386CMPWconst, types.TypeFlags)
		v2.AuxInt = int16ToAuxInt(8)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh8Ux16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SHRB <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SHRB)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh8Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8Ux32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHRB <t> x y) (SBBLcarrymask <t> (CMPLconst y [8])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386ANDL)
		v0 := b.NewValue0(v.Pos, Op386SHRB, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, Op386CMPLconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(8)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh8Ux32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SHRB <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SHRB)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh8Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Rsh8Ux64 x (Const64 [c]))
	// cond: uint64(c) < 8
	// result: (SHRBconst x [int8(c)])
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.reset(Op386SHRBconst)
		v.AuxInt = int8ToAuxInt(int8(c))
		v.AddArg(x)
		return true
	}
	// match: (Rsh8Ux64 _ (Const64 [c]))
	// cond: uint64(c) >= 8
	// result: (Const8 [0])
	for {
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 8) {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int8ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_OpRsh8Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8Ux8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHRB <t> x y) (SBBLcarrymask <t> (CMPBconst y [8])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386ANDL)
		v0 := b.NewValue0(v.Pos, Op386SHRB, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, Op386CMPBconst, types.TypeFlags)
		v2.AuxInt = int8ToAuxInt(8)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh8Ux8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SHRB <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SHRB)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SARB <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPWconst y [8])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SARB)
		v.Type = t
		v0 := b.NewValue0(v.Pos, Op386ORL, y.Type)
		v1 := b.NewValue0(v.Pos, Op386NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, Op386SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, Op386CMPWconst, types.TypeFlags)
		v3.AuxInt = int16ToAuxInt(8)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8x16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SARB x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SARB)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SARB <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPLconst y [8])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SARB)
		v.Type = t
		v0 := b.NewValue0(v.Pos, Op386ORL, y.Type)
		v1 := b.NewValue0(v.Pos, Op386NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, Op386SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, Op386CMPLconst, types.TypeFlags)
		v3.AuxInt = int32ToAuxInt(8)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8x32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SARB x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SARB)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Rsh8x64 x (Const64 [c]))
	// cond: uint64(c) < 8
	// result: (SARBconst x [int8(c)])
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.reset(Op386SARBconst)
		v.AuxInt = int8ToAuxInt(int8(c))
		v.AddArg(x)
		return true
	}
	// match: (Rsh8x64 x (Const64 [c]))
	// cond: uint64(c) >= 8
	// result: (SARBconst x [7])
	for {
		x := v_0
		if v_1.Op != OpConst64 {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 8) {
			break
		}
		v.reset(Op386SARBconst)
		v.AuxInt = int8ToAuxInt(7)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_OpRsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SARB <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPBconst y [8])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SARB)
		v.Type = t
		v0 := b.NewValue0(v.Pos, Op386ORL, y.Type)
		v1 := b.NewValue0(v.Pos, Op386NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, Op386SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, Op386CMPBconst, types.TypeFlags)
		v3.AuxInt = int8ToAuxInt(8)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8x8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SARB x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(Op386SARB)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpSelect0(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select0 (Mul32uover x y))
	// result: (Select0 <typ.UInt32> (MULLU x y))
	for {
		if v_0.Op != OpMul32uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpSelect0)
		v.Type = typ.UInt32
		v0 := b.NewValue0(v.Pos, Op386MULLU, types.NewTuple(typ.UInt32, types.TypeFlags))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue386_OpSelect1(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select1 (Mul32uover x y))
	// result: (SETO (Select1 <types.TypeFlags> (MULLU x y)))
	for {
		if v_0.Op != OpMul32uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(Op386SETO)
		v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, Op386MULLU, types.NewTuple(typ.UInt32, types.TypeFlags))
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue386_OpSignmask(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Signmask x)
	// result: (SARLconst x [31])
	for {
		x := v_0
		v.reset(Op386SARLconst)
		v.AuxInt = int32ToAuxInt(31)
		v.AddArg(x)
		return true
	}
}
func rewriteValue386_OpSlicemask(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Slicemask <t> x)
	// result: (SARLconst (NEGL <t> x) [31])
	for {
		t := v.Type
		x := v_0
		v.reset(Op386SARLconst)
		v.AuxInt = int32ToAuxInt(31)
		v0 := b.NewValue0(v.Pos, Op386NEGL, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpStore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && is64BitFloat(val.Type)
	// result: (MOVSDstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && is64BitFloat(val.Type)) {
			break
		}
		v.reset(Op386MOVSDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && is32BitFloat(val.Type)
	// result: (MOVSSstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && is32BitFloat(val.Type)) {
			break
		}
		v.reset(Op386MOVSSstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4
	// result: (MOVLstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4) {
			break
		}
		v.reset(Op386MOVLstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 2
	// result: (MOVWstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 2) {
			break
		}
		v.reset(Op386MOVWstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
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
		v.reset(Op386MOVBstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValue386_OpZero(v *Value) bool {
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
	// match: (Zero [1] destptr mem)
	// result: (MOVBstoreconst [0] destptr mem)
	for {
		if auxIntToInt64(v.AuxInt) != 1 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(Op386MOVBstoreconst)
		v.AuxInt = valAndOffToAuxInt(0)
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [2] destptr mem)
	// result: (MOVWstoreconst [0] destptr mem)
	for {
		if auxIntToInt64(v.AuxInt) != 2 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(Op386MOVWstoreconst)
		v.AuxInt = valAndOffToAuxInt(0)
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [4] destptr mem)
	// result: (MOVLstoreconst [0] destptr mem)
	for {
		if auxIntToInt64(v.AuxInt) != 4 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(Op386MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(0)
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [3] destptr mem)
	// result: (MOVBstoreconst [makeValAndOff32(0,2)] destptr (MOVWstoreconst [makeValAndOff32(0,0)] destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 3 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(Op386MOVBstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(0, 2))
		v0 := b.NewValue0(v.Pos, Op386MOVWstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff32(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [5] destptr mem)
	// result: (MOVBstoreconst [makeValAndOff32(0,4)] destptr (MOVLstoreconst [makeValAndOff32(0,0)] destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 5 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(Op386MOVBstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(0, 4))
		v0 := b.NewValue0(v.Pos, Op386MOVLstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff32(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [6] destptr mem)
	// result: (MOVWstoreconst [makeValAndOff32(0,4)] destptr (MOVLstoreconst [makeValAndOff32(0,0)] destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 6 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(Op386MOVWstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(0, 4))
		v0 := b.NewValue0(v.Pos, Op386MOVLstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff32(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [7] destptr mem)
	// result: (MOVLstoreconst [makeValAndOff32(0,3)] destptr (MOVLstoreconst [makeValAndOff32(0,0)] destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 7 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(Op386MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(0, 3))
		v0 := b.NewValue0(v.Pos, Op386MOVLstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff32(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s%4 != 0 && s > 4
	// result: (Zero [s-s%4] (ADDLconst destptr [int32(s%4)]) (MOVLstoreconst [0] destptr mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		if !(s%4 != 0 && s > 4) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = int64ToAuxInt(s - s%4)
		v0 := b.NewValue0(v.Pos, Op386ADDLconst, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(int32(s % 4))
		v0.AddArg(destptr)
		v1 := b.NewValue0(v.Pos, Op386MOVLstoreconst, types.TypeMem)
		v1.AuxInt = valAndOffToAuxInt(0)
		v1.AddArg2(destptr, mem)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Zero [8] destptr mem)
	// result: (MOVLstoreconst [makeValAndOff32(0,4)] destptr (MOVLstoreconst [makeValAndOff32(0,0)] destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(Op386MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(0, 4))
		v0 := b.NewValue0(v.Pos, Op386MOVLstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff32(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [12] destptr mem)
	// result: (MOVLstoreconst [makeValAndOff32(0,8)] destptr (MOVLstoreconst [makeValAndOff32(0,4)] destptr (MOVLstoreconst [makeValAndOff32(0,0)] destptr mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 12 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(Op386MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(0, 8))
		v0 := b.NewValue0(v.Pos, Op386MOVLstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff32(0, 4))
		v1 := b.NewValue0(v.Pos, Op386MOVLstoreconst, types.TypeMem)
		v1.AuxInt = valAndOffToAuxInt(makeValAndOff32(0, 0))
		v1.AddArg2(destptr, mem)
		v0.AddArg2(destptr, v1)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [16] destptr mem)
	// result: (MOVLstoreconst [makeValAndOff32(0,12)] destptr (MOVLstoreconst [makeValAndOff32(0,8)] destptr (MOVLstoreconst [makeValAndOff32(0,4)] destptr (MOVLstoreconst [makeValAndOff32(0,0)] destptr mem))))
	for {
		if auxIntToInt64(v.AuxInt) != 16 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(Op386MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff32(0, 12))
		v0 := b.NewValue0(v.Pos, Op386MOVLstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff32(0, 8))
		v1 := b.NewValue0(v.Pos, Op386MOVLstoreconst, types.TypeMem)
		v1.AuxInt = valAndOffToAuxInt(makeValAndOff32(0, 4))
		v2 := b.NewValue0(v.Pos, Op386MOVLstoreconst, types.TypeMem)
		v2.AuxInt = valAndOffToAuxInt(makeValAndOff32(0, 0))
		v2.AddArg2(destptr, mem)
		v1.AddArg2(destptr, v2)
		v0.AddArg2(destptr, v1)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s > 16 && s <= 4*128 && s%4 == 0 && !config.noDuffDevice
	// result: (DUFFZERO [1*(128-s/4)] destptr (MOVLconst [0]) mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		if !(s > 16 && s <= 4*128 && s%4 == 0 && !config.noDuffDevice) {
			break
		}
		v.reset(Op386DUFFZERO)
		v.AuxInt = int64ToAuxInt(1 * (128 - s/4))
		v0 := b.NewValue0(v.Pos, Op386MOVLconst, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(destptr, v0, mem)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: (s > 4*128 || (config.noDuffDevice && s > 16)) && s%4 == 0
	// result: (REPSTOSL destptr (MOVLconst [int32(s/4)]) (MOVLconst [0]) mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		if !((s > 4*128 || (config.noDuffDevice && s > 16)) && s%4 == 0) {
			break
		}
		v.reset(Op386REPSTOSL)
		v0 := b.NewValue0(v.Pos, Op386MOVLconst, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(int32(s / 4))
		v1 := b.NewValue0(v.Pos, Op386MOVLconst, typ.UInt32)
		v1.AuxInt = int32ToAuxInt(0)
		v.AddArg4(destptr, v0, v1, mem)
		return true
	}
	return false
}
func rewriteValue386_OpZeromask(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Zeromask <t> x)
	// result: (XORLconst [-1] (SBBLcarrymask <t> (CMPLconst x [1])))
	for {
		t := v.Type
		x := v_0
		v.reset(Op386XORLconst)
		v.AuxInt = int32ToAuxInt(-1)
		v0 := b.NewValue0(v.Pos, Op386SBBLcarrymask, t)
		v1 := b.NewValue0(v.Pos, Op386CMPLconst, types.TypeFlags)
		v1.AuxInt = int32ToAuxInt(1)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteBlock386(b *Block) bool {
	switch b.Kind {
	case Block386EQ:
		// match: (EQ (InvertFlags cmp) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386EQ, cmp)
			return true
		}
		// match: (EQ (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagEQ {
			b.Reset(BlockFirst)
			return true
		}
		// match: (EQ (FlagLT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagLT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (EQ (FlagLT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagLT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (EQ (FlagGT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagGT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (EQ (FlagGT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagGT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
	case Block386GE:
		// match: (GE (InvertFlags cmp) yes no)
		// result: (LE cmp yes no)
		for b.Controls[0].Op == Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386LE, cmp)
			return true
		}
		// match: (GE (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagEQ {
			b.Reset(BlockFirst)
			return true
		}
		// match: (GE (FlagLT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagLT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (GE (FlagLT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagLT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (GE (FlagGT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagGT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (GE (FlagGT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagGT_UGT {
			b.Reset(BlockFirst)
			return true
		}
	case Block386GT:
		// match: (GT (InvertFlags cmp) yes no)
		// result: (LT cmp yes no)
		for b.Controls[0].Op == Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386LT, cmp)
			return true
		}
		// match: (GT (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagEQ {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (GT (FlagLT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagLT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (GT (FlagLT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagLT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (GT (FlagGT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagGT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (GT (FlagGT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagGT_UGT {
			b.Reset(BlockFirst)
			return true
		}
	case BlockIf:
		// match: (If (SETL cmp) yes no)
		// result: (LT cmp yes no)
		for b.Controls[0].Op == Op386SETL {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386LT, cmp)
			return true
		}
		// match: (If (SETLE cmp) yes no)
		// result: (LE cmp yes no)
		for b.Controls[0].Op == Op386SETLE {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386LE, cmp)
			return true
		}
		// match: (If (SETG cmp) yes no)
		// result: (GT cmp yes no)
		for b.Controls[0].Op == Op386SETG {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386GT, cmp)
			return true
		}
		// match: (If (SETGE cmp) yes no)
		// result: (GE cmp yes no)
		for b.Controls[0].Op == Op386SETGE {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386GE, cmp)
			return true
		}
		// match: (If (SETEQ cmp) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == Op386SETEQ {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386EQ, cmp)
			return true
		}
		// match: (If (SETNE cmp) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == Op386SETNE {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386NE, cmp)
			return true
		}
		// match: (If (SETB cmp) yes no)
		// result: (ULT cmp yes no)
		for b.Controls[0].Op == Op386SETB {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386ULT, cmp)
			return true
		}
		// match: (If (SETBE cmp) yes no)
		// result: (ULE cmp yes no)
		for b.Controls[0].Op == Op386SETBE {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386ULE, cmp)
			return true
		}
		// match: (If (SETA cmp) yes no)
		// result: (UGT cmp yes no)
		for b.Controls[0].Op == Op386SETA {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386UGT, cmp)
			return true
		}
		// match: (If (SETAE cmp) yes no)
		// result: (UGE cmp yes no)
		for b.Controls[0].Op == Op386SETAE {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386UGE, cmp)
			return true
		}
		// match: (If (SETO cmp) yes no)
		// result: (OS cmp yes no)
		for b.Controls[0].Op == Op386SETO {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386OS, cmp)
			return true
		}
		// match: (If (SETGF cmp) yes no)
		// result: (UGT cmp yes no)
		for b.Controls[0].Op == Op386SETGF {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386UGT, cmp)
			return true
		}
		// match: (If (SETGEF cmp) yes no)
		// result: (UGE cmp yes no)
		for b.Controls[0].Op == Op386SETGEF {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386UGE, cmp)
			return true
		}
		// match: (If (SETEQF cmp) yes no)
		// result: (EQF cmp yes no)
		for b.Controls[0].Op == Op386SETEQF {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386EQF, cmp)
			return true
		}
		// match: (If (SETNEF cmp) yes no)
		// result: (NEF cmp yes no)
		for b.Controls[0].Op == Op386SETNEF {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386NEF, cmp)
			return true
		}
		// match: (If cond yes no)
		// result: (NE (TESTB cond cond) yes no)
		for {
			cond := b.Controls[0]
			v0 := b.NewValue0(cond.Pos, Op386TESTB, types.TypeFlags)
			v0.AddArg2(cond, cond)
			b.resetWithControl(Block386NE, v0)
			return true
		}
	case Block386LE:
		// match: (LE (InvertFlags cmp) yes no)
		// result: (GE cmp yes no)
		for b.Controls[0].Op == Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386GE, cmp)
			return true
		}
		// match: (LE (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagEQ {
			b.Reset(BlockFirst)
			return true
		}
		// match: (LE (FlagLT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagLT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (LE (FlagLT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagLT_UGT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (LE (FlagGT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagGT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (LE (FlagGT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagGT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
	case Block386LT:
		// match: (LT (InvertFlags cmp) yes no)
		// result: (GT cmp yes no)
		for b.Controls[0].Op == Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386GT, cmp)
			return true
		}
		// match: (LT (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagEQ {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (LT (FlagLT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagLT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (LT (FlagLT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagLT_UGT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (LT (FlagGT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagGT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (LT (FlagGT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagGT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
	case Block386NE:
		// match: (NE (TESTB (SETL cmp) (SETL cmp)) yes no)
		// result: (LT cmp yes no)
		for b.Controls[0].Op == Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != Op386SETL {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != Op386SETL || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(Block386LT, cmp)
			return true
		}
		// match: (NE (TESTB (SETLE cmp) (SETLE cmp)) yes no)
		// result: (LE cmp yes no)
		for b.Controls[0].Op == Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != Op386SETLE {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != Op386SETLE || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(Block386LE, cmp)
			return true
		}
		// match: (NE (TESTB (SETG cmp) (SETG cmp)) yes no)
		// result: (GT cmp yes no)
		for b.Controls[0].Op == Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != Op386SETG {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != Op386SETG || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(Block386GT, cmp)
			return true
		}
		// match: (NE (TESTB (SETGE cmp) (SETGE cmp)) yes no)
		// result: (GE cmp yes no)
		for b.Controls[0].Op == Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != Op386SETGE {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != Op386SETGE || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(Block386GE, cmp)
			return true
		}
		// match: (NE (TESTB (SETEQ cmp) (SETEQ cmp)) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != Op386SETEQ {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != Op386SETEQ || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(Block386EQ, cmp)
			return true
		}
		// match: (NE (TESTB (SETNE cmp) (SETNE cmp)) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != Op386SETNE {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != Op386SETNE || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(Block386NE, cmp)
			return true
		}
		// match: (NE (TESTB (SETB cmp) (SETB cmp)) yes no)
		// result: (ULT cmp yes no)
		for b.Controls[0].Op == Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != Op386SETB {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != Op386SETB || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(Block386ULT, cmp)
			return true
		}
		// match: (NE (TESTB (SETBE cmp) (SETBE cmp)) yes no)
		// result: (ULE cmp yes no)
		for b.Controls[0].Op == Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != Op386SETBE {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != Op386SETBE || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(Block386ULE, cmp)
			return true
		}
		// match: (NE (TESTB (SETA cmp) (SETA cmp)) yes no)
		// result: (UGT cmp yes no)
		for b.Controls[0].Op == Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != Op386SETA {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != Op386SETA || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(Block386UGT, cmp)
			return true
		}
		// match: (NE (TESTB (SETAE cmp) (SETAE cmp)) yes no)
		// result: (UGE cmp yes no)
		for b.Controls[0].Op == Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != Op386SETAE {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != Op386SETAE || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(Block386UGE, cmp)
			return true
		}
		// match: (NE (TESTB (SETO cmp) (SETO cmp)) yes no)
		// result: (OS cmp yes no)
		for b.Controls[0].Op == Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != Op386SETO {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != Op386SETO || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(Block386OS, cmp)
			return true
		}
		// match: (NE (TESTB (SETGF cmp) (SETGF cmp)) yes no)
		// result: (UGT cmp yes no)
		for b.Controls[0].Op == Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != Op386SETGF {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != Op386SETGF || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(Block386UGT, cmp)
			return true
		}
		// match: (NE (TESTB (SETGEF cmp) (SETGEF cmp)) yes no)
		// result: (UGE cmp yes no)
		for b.Controls[0].Op == Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != Op386SETGEF {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != Op386SETGEF || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(Block386UGE, cmp)
			return true
		}
		// match: (NE (TESTB (SETEQF cmp) (SETEQF cmp)) yes no)
		// result: (EQF cmp yes no)
		for b.Controls[0].Op == Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != Op386SETEQF {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != Op386SETEQF || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(Block386EQF, cmp)
			return true
		}
		// match: (NE (TESTB (SETNEF cmp) (SETNEF cmp)) yes no)
		// result: (NEF cmp yes no)
		for b.Controls[0].Op == Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != Op386SETNEF {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != Op386SETNEF || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(Block386NEF, cmp)
			return true
		}
		// match: (NE (InvertFlags cmp) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386NE, cmp)
			return true
		}
		// match: (NE (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagEQ {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (NE (FlagLT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagLT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (NE (FlagLT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagLT_UGT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (NE (FlagGT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagGT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (NE (FlagGT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagGT_UGT {
			b.Reset(BlockFirst)
			return true
		}
	case Block386UGE:
		// match: (UGE (InvertFlags cmp) yes no)
		// result: (ULE cmp yes no)
		for b.Controls[0].Op == Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386ULE, cmp)
			return true
		}
		// match: (UGE (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagEQ {
			b.Reset(BlockFirst)
			return true
		}
		// match: (UGE (FlagLT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagLT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (UGE (FlagLT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagLT_UGT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (UGE (FlagGT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagGT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (UGE (FlagGT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagGT_UGT {
			b.Reset(BlockFirst)
			return true
		}
	case Block386UGT:
		// match: (UGT (InvertFlags cmp) yes no)
		// result: (ULT cmp yes no)
		for b.Controls[0].Op == Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386ULT, cmp)
			return true
		}
		// match: (UGT (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagEQ {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (UGT (FlagLT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagLT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (UGT (FlagLT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagLT_UGT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (UGT (FlagGT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagGT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (UGT (FlagGT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagGT_UGT {
			b.Reset(BlockFirst)
			return true
		}
	case Block386ULE:
		// match: (ULE (InvertFlags cmp) yes no)
		// result: (UGE cmp yes no)
		for b.Controls[0].Op == Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386UGE, cmp)
			return true
		}
		// match: (ULE (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagEQ {
			b.Reset(BlockFirst)
			return true
		}
		// match: (ULE (FlagLT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagLT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (ULE (FlagLT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagLT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (ULE (FlagGT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagGT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (ULE (FlagGT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagGT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
	case Block386ULT:
		// match: (ULT (InvertFlags cmp) yes no)
		// result: (UGT cmp yes no)
		for b.Controls[0].Op == Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(Block386UGT, cmp)
			return true
		}
		// match: (ULT (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagEQ {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (ULT (FlagLT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagLT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (ULT (FlagLT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagLT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (ULT (FlagGT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == Op386FlagGT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (ULT (FlagGT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == Op386FlagGT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
	}
	return false
}
