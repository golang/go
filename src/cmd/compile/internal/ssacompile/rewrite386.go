// Code generated from _gen/386.rules using 'go generate'; DO NOT EDIT.

package ssacompile

import "math"
import "cmd/compile/internal/types"
import "cmd/compile/internal/ssa/block"
import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa"

func rewriteValue386(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.Op386ADCL:
		return rewriteValue386_Op386ADCL(v)
	case ssaop.Op386ADDL:
		return rewriteValue386_Op386ADDL(v)
	case ssaop.Op386ADDLcarry:
		return rewriteValue386_Op386ADDLcarry(v)
	case ssaop.Op386ADDLconst:
		return rewriteValue386_Op386ADDLconst(v)
	case ssaop.Op386ADDLconstmodify:
		return rewriteValue386_Op386ADDLconstmodify(v)
	case ssaop.Op386ADDLload:
		return rewriteValue386_Op386ADDLload(v)
	case ssaop.Op386ADDLmodify:
		return rewriteValue386_Op386ADDLmodify(v)
	case ssaop.Op386ADDSD:
		return rewriteValue386_Op386ADDSD(v)
	case ssaop.Op386ADDSDload:
		return rewriteValue386_Op386ADDSDload(v)
	case ssaop.Op386ADDSS:
		return rewriteValue386_Op386ADDSS(v)
	case ssaop.Op386ADDSSload:
		return rewriteValue386_Op386ADDSSload(v)
	case ssaop.Op386ANDL:
		return rewriteValue386_Op386ANDL(v)
	case ssaop.Op386ANDLconst:
		return rewriteValue386_Op386ANDLconst(v)
	case ssaop.Op386ANDLconstmodify:
		return rewriteValue386_Op386ANDLconstmodify(v)
	case ssaop.Op386ANDLload:
		return rewriteValue386_Op386ANDLload(v)
	case ssaop.Op386ANDLmodify:
		return rewriteValue386_Op386ANDLmodify(v)
	case ssaop.Op386CMPB:
		return rewriteValue386_Op386CMPB(v)
	case ssaop.Op386CMPBconst:
		return rewriteValue386_Op386CMPBconst(v)
	case ssaop.Op386CMPBload:
		return rewriteValue386_Op386CMPBload(v)
	case ssaop.Op386CMPL:
		return rewriteValue386_Op386CMPL(v)
	case ssaop.Op386CMPLconst:
		return rewriteValue386_Op386CMPLconst(v)
	case ssaop.Op386CMPLload:
		return rewriteValue386_Op386CMPLload(v)
	case ssaop.Op386CMPW:
		return rewriteValue386_Op386CMPW(v)
	case ssaop.Op386CMPWconst:
		return rewriteValue386_Op386CMPWconst(v)
	case ssaop.Op386CMPWload:
		return rewriteValue386_Op386CMPWload(v)
	case ssaop.Op386DIVSD:
		return rewriteValue386_Op386DIVSD(v)
	case ssaop.Op386DIVSDload:
		return rewriteValue386_Op386DIVSDload(v)
	case ssaop.Op386DIVSS:
		return rewriteValue386_Op386DIVSS(v)
	case ssaop.Op386DIVSSload:
		return rewriteValue386_Op386DIVSSload(v)
	case ssaop.Op386LEAL:
		return rewriteValue386_Op386LEAL(v)
	case ssaop.Op386LEAL1:
		return rewriteValue386_Op386LEAL1(v)
	case ssaop.Op386LEAL2:
		return rewriteValue386_Op386LEAL2(v)
	case ssaop.Op386LEAL4:
		return rewriteValue386_Op386LEAL4(v)
	case ssaop.Op386LEAL8:
		return rewriteValue386_Op386LEAL8(v)
	case ssaop.Op386LoweredPanicBoundsRC:
		return rewriteValue386_Op386LoweredPanicBoundsRC(v)
	case ssaop.Op386LoweredPanicBoundsRR:
		return rewriteValue386_Op386LoweredPanicBoundsRR(v)
	case ssaop.Op386LoweredPanicExtendRC:
		return rewriteValue386_Op386LoweredPanicExtendRC(v)
	case ssaop.Op386LoweredPanicExtendRR:
		return rewriteValue386_Op386LoweredPanicExtendRR(v)
	case ssaop.Op386MOVBLSX:
		return rewriteValue386_Op386MOVBLSX(v)
	case ssaop.Op386MOVBLSXload:
		return rewriteValue386_Op386MOVBLSXload(v)
	case ssaop.Op386MOVBLZX:
		return rewriteValue386_Op386MOVBLZX(v)
	case ssaop.Op386MOVBload:
		return rewriteValue386_Op386MOVBload(v)
	case ssaop.Op386MOVBstore:
		return rewriteValue386_Op386MOVBstore(v)
	case ssaop.Op386MOVBstoreconst:
		return rewriteValue386_Op386MOVBstoreconst(v)
	case ssaop.Op386MOVLload:
		return rewriteValue386_Op386MOVLload(v)
	case ssaop.Op386MOVLstore:
		return rewriteValue386_Op386MOVLstore(v)
	case ssaop.Op386MOVLstoreconst:
		return rewriteValue386_Op386MOVLstoreconst(v)
	case ssaop.Op386MOVSDconst:
		return rewriteValue386_Op386MOVSDconst(v)
	case ssaop.Op386MOVSDload:
		return rewriteValue386_Op386MOVSDload(v)
	case ssaop.Op386MOVSDstore:
		return rewriteValue386_Op386MOVSDstore(v)
	case ssaop.Op386MOVSSconst:
		return rewriteValue386_Op386MOVSSconst(v)
	case ssaop.Op386MOVSSload:
		return rewriteValue386_Op386MOVSSload(v)
	case ssaop.Op386MOVSSstore:
		return rewriteValue386_Op386MOVSSstore(v)
	case ssaop.Op386MOVWLSX:
		return rewriteValue386_Op386MOVWLSX(v)
	case ssaop.Op386MOVWLSXload:
		return rewriteValue386_Op386MOVWLSXload(v)
	case ssaop.Op386MOVWLZX:
		return rewriteValue386_Op386MOVWLZX(v)
	case ssaop.Op386MOVWload:
		return rewriteValue386_Op386MOVWload(v)
	case ssaop.Op386MOVWstore:
		return rewriteValue386_Op386MOVWstore(v)
	case ssaop.Op386MOVWstoreconst:
		return rewriteValue386_Op386MOVWstoreconst(v)
	case ssaop.Op386MULL:
		return rewriteValue386_Op386MULL(v)
	case ssaop.Op386MULLconst:
		return rewriteValue386_Op386MULLconst(v)
	case ssaop.Op386MULLload:
		return rewriteValue386_Op386MULLload(v)
	case ssaop.Op386MULSD:
		return rewriteValue386_Op386MULSD(v)
	case ssaop.Op386MULSDload:
		return rewriteValue386_Op386MULSDload(v)
	case ssaop.Op386MULSS:
		return rewriteValue386_Op386MULSS(v)
	case ssaop.Op386MULSSload:
		return rewriteValue386_Op386MULSSload(v)
	case ssaop.Op386NEGL:
		return rewriteValue386_Op386NEGL(v)
	case ssaop.Op386NOTL:
		return rewriteValue386_Op386NOTL(v)
	case ssaop.Op386ORL:
		return rewriteValue386_Op386ORL(v)
	case ssaop.Op386ORLconst:
		return rewriteValue386_Op386ORLconst(v)
	case ssaop.Op386ORLconstmodify:
		return rewriteValue386_Op386ORLconstmodify(v)
	case ssaop.Op386ORLload:
		return rewriteValue386_Op386ORLload(v)
	case ssaop.Op386ORLmodify:
		return rewriteValue386_Op386ORLmodify(v)
	case ssaop.Op386ROLB:
		return rewriteValue386_Op386ROLB(v)
	case ssaop.Op386ROLBconst:
		return rewriteValue386_Op386ROLBconst(v)
	case ssaop.Op386ROLL:
		return rewriteValue386_Op386ROLL(v)
	case ssaop.Op386ROLLconst:
		return rewriteValue386_Op386ROLLconst(v)
	case ssaop.Op386ROLW:
		return rewriteValue386_Op386ROLW(v)
	case ssaop.Op386ROLWconst:
		return rewriteValue386_Op386ROLWconst(v)
	case ssaop.Op386SARB:
		return rewriteValue386_Op386SARB(v)
	case ssaop.Op386SARBconst:
		return rewriteValue386_Op386SARBconst(v)
	case ssaop.Op386SARL:
		return rewriteValue386_Op386SARL(v)
	case ssaop.Op386SARLconst:
		return rewriteValue386_Op386SARLconst(v)
	case ssaop.Op386SARW:
		return rewriteValue386_Op386SARW(v)
	case ssaop.Op386SARWconst:
		return rewriteValue386_Op386SARWconst(v)
	case ssaop.Op386SBBL:
		return rewriteValue386_Op386SBBL(v)
	case ssaop.Op386SBBLcarrymask:
		return rewriteValue386_Op386SBBLcarrymask(v)
	case ssaop.Op386SETA:
		return rewriteValue386_Op386SETA(v)
	case ssaop.Op386SETAE:
		return rewriteValue386_Op386SETAE(v)
	case ssaop.Op386SETB:
		return rewriteValue386_Op386SETB(v)
	case ssaop.Op386SETBE:
		return rewriteValue386_Op386SETBE(v)
	case ssaop.Op386SETEQ:
		return rewriteValue386_Op386SETEQ(v)
	case ssaop.Op386SETG:
		return rewriteValue386_Op386SETG(v)
	case ssaop.Op386SETGE:
		return rewriteValue386_Op386SETGE(v)
	case ssaop.Op386SETL:
		return rewriteValue386_Op386SETL(v)
	case ssaop.Op386SETLE:
		return rewriteValue386_Op386SETLE(v)
	case ssaop.Op386SETNE:
		return rewriteValue386_Op386SETNE(v)
	case ssaop.Op386SHLL:
		return rewriteValue386_Op386SHLL(v)
	case ssaop.Op386SHLLconst:
		return rewriteValue386_Op386SHLLconst(v)
	case ssaop.Op386SHRB:
		return rewriteValue386_Op386SHRB(v)
	case ssaop.Op386SHRBconst:
		return rewriteValue386_Op386SHRBconst(v)
	case ssaop.Op386SHRL:
		return rewriteValue386_Op386SHRL(v)
	case ssaop.Op386SHRLconst:
		return rewriteValue386_Op386SHRLconst(v)
	case ssaop.Op386SHRW:
		return rewriteValue386_Op386SHRW(v)
	case ssaop.Op386SHRWconst:
		return rewriteValue386_Op386SHRWconst(v)
	case ssaop.Op386SUBL:
		return rewriteValue386_Op386SUBL(v)
	case ssaop.Op386SUBLcarry:
		return rewriteValue386_Op386SUBLcarry(v)
	case ssaop.Op386SUBLconst:
		return rewriteValue386_Op386SUBLconst(v)
	case ssaop.Op386SUBLload:
		return rewriteValue386_Op386SUBLload(v)
	case ssaop.Op386SUBLmodify:
		return rewriteValue386_Op386SUBLmodify(v)
	case ssaop.Op386SUBSD:
		return rewriteValue386_Op386SUBSD(v)
	case ssaop.Op386SUBSDload:
		return rewriteValue386_Op386SUBSDload(v)
	case ssaop.Op386SUBSS:
		return rewriteValue386_Op386SUBSS(v)
	case ssaop.Op386SUBSSload:
		return rewriteValue386_Op386SUBSSload(v)
	case ssaop.Op386XORL:
		return rewriteValue386_Op386XORL(v)
	case ssaop.Op386XORLconst:
		return rewriteValue386_Op386XORLconst(v)
	case ssaop.Op386XORLconstmodify:
		return rewriteValue386_Op386XORLconstmodify(v)
	case ssaop.Op386XORLload:
		return rewriteValue386_Op386XORLload(v)
	case ssaop.Op386XORLmodify:
		return rewriteValue386_Op386XORLmodify(v)
	case ssaop.OpAdd16:
		v.Op = ssaop.Op386ADDL
		return true
	case ssaop.OpAdd32:
		v.Op = ssaop.Op386ADDL
		return true
	case ssaop.OpAdd32F:
		v.Op = ssaop.Op386ADDSS
		return true
	case ssaop.OpAdd32carry:
		v.Op = ssaop.Op386ADDLcarry
		return true
	case ssaop.OpAdd32carrywithcarry:
		v.Op = ssaop.Op386ADCLcarry
		return true
	case ssaop.OpAdd32withcarry:
		v.Op = ssaop.Op386ADCL
		return true
	case ssaop.OpAdd64F:
		v.Op = ssaop.Op386ADDSD
		return true
	case ssaop.OpAdd8:
		v.Op = ssaop.Op386ADDL
		return true
	case ssaop.OpAddPtr:
		v.Op = ssaop.Op386ADDL
		return true
	case ssaop.OpAddr:
		return rewriteValue386_OpAddr(v)
	case ssaop.OpAnd16:
		v.Op = ssaop.Op386ANDL
		return true
	case ssaop.OpAnd32:
		v.Op = ssaop.Op386ANDL
		return true
	case ssaop.OpAnd8:
		v.Op = ssaop.Op386ANDL
		return true
	case ssaop.OpAndB:
		v.Op = ssaop.Op386ANDL
		return true
	case ssaop.OpAvg32u:
		v.Op = ssaop.Op386AVGLU
		return true
	case ssaop.OpBswap16:
		return rewriteValue386_OpBswap16(v)
	case ssaop.OpBswap32:
		v.Op = ssaop.Op386BSWAPL
		return true
	case ssaop.OpClosureCall:
		v.Op = ssaop.Op386CALLclosure
		return true
	case ssaop.OpCom16:
		v.Op = ssaop.Op386NOTL
		return true
	case ssaop.OpCom32:
		v.Op = ssaop.Op386NOTL
		return true
	case ssaop.OpCom8:
		v.Op = ssaop.Op386NOTL
		return true
	case ssaop.OpConst16:
		return rewriteValue386_OpConst16(v)
	case ssaop.OpConst32:
		v.Op = ssaop.Op386MOVLconst
		return true
	case ssaop.OpConst32F:
		v.Op = ssaop.Op386MOVSSconst
		return true
	case ssaop.OpConst64F:
		v.Op = ssaop.Op386MOVSDconst
		return true
	case ssaop.OpConst8:
		return rewriteValue386_OpConst8(v)
	case ssaop.OpConstBool:
		return rewriteValue386_OpConstBool(v)
	case ssaop.OpConstNil:
		return rewriteValue386_OpConstNil(v)
	case ssaop.OpCtz16:
		return rewriteValue386_OpCtz16(v)
	case ssaop.OpCtz16NonZero:
		v.Op = ssaop.Op386BSFL
		return true
	case ssaop.OpCtz32:
		v.Op = ssaop.Op386LoweredCtz32
		return true
	case ssaop.OpCtz32NonZero:
		v.Op = ssaop.Op386BSFL
		return true
	case ssaop.OpCtz64On32:
		v.Op = ssaop.Op386LoweredCtz64
		return true
	case ssaop.OpCtz8:
		return rewriteValue386_OpCtz8(v)
	case ssaop.OpCtz8NonZero:
		v.Op = ssaop.Op386BSFL
		return true
	case ssaop.OpCvt32Fto32:
		v.Op = ssaop.Op386CVTTSS2SL
		return true
	case ssaop.OpCvt32Fto64F:
		v.Op = ssaop.Op386CVTSS2SD
		return true
	case ssaop.OpCvt32to32F:
		v.Op = ssaop.Op386CVTSL2SS
		return true
	case ssaop.OpCvt32to64F:
		v.Op = ssaop.Op386CVTSL2SD
		return true
	case ssaop.OpCvt64Fto32:
		v.Op = ssaop.Op386CVTTSD2SL
		return true
	case ssaop.OpCvt64Fto32F:
		v.Op = ssaop.Op386CVTSD2SS
		return true
	case ssaop.OpCvtBoolToUint8:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpDiv16:
		v.Op = ssaop.Op386DIVW
		return true
	case ssaop.OpDiv16u:
		v.Op = ssaop.Op386DIVWU
		return true
	case ssaop.OpDiv32:
		v.Op = ssaop.Op386DIVL
		return true
	case ssaop.OpDiv32F:
		v.Op = ssaop.Op386DIVSS
		return true
	case ssaop.OpDiv32u:
		v.Op = ssaop.Op386DIVLU
		return true
	case ssaop.OpDiv64F:
		v.Op = ssaop.Op386DIVSD
		return true
	case ssaop.OpDiv8:
		return rewriteValue386_OpDiv8(v)
	case ssaop.OpDiv8u:
		return rewriteValue386_OpDiv8u(v)
	case ssaop.OpEq16:
		return rewriteValue386_OpEq16(v)
	case ssaop.OpEq32:
		return rewriteValue386_OpEq32(v)
	case ssaop.OpEq32F:
		return rewriteValue386_OpEq32F(v)
	case ssaop.OpEq64F:
		return rewriteValue386_OpEq64F(v)
	case ssaop.OpEq8:
		return rewriteValue386_OpEq8(v)
	case ssaop.OpEqB:
		return rewriteValue386_OpEqB(v)
	case ssaop.OpEqPtr:
		return rewriteValue386_OpEqPtr(v)
	case ssaop.OpGetCallerPC:
		v.Op = ssaop.Op386LoweredGetCallerPC
		return true
	case ssaop.OpGetCallerSP:
		v.Op = ssaop.Op386LoweredGetCallerSP
		return true
	case ssaop.OpGetClosurePtr:
		v.Op = ssaop.Op386LoweredGetClosurePtr
		return true
	case ssaop.OpGetG:
		v.Op = ssaop.Op386LoweredGetG
		return true
	case ssaop.OpHmul32:
		v.Op = ssaop.Op386HMULL
		return true
	case ssaop.OpHmul32u:
		v.Op = ssaop.Op386HMULLU
		return true
	case ssaop.OpInterCall:
		v.Op = ssaop.Op386CALLinter
		return true
	case ssaop.OpIsInBounds:
		return rewriteValue386_OpIsInBounds(v)
	case ssaop.OpIsNonNil:
		return rewriteValue386_OpIsNonNil(v)
	case ssaop.OpIsSliceInBounds:
		return rewriteValue386_OpIsSliceInBounds(v)
	case ssaop.OpLeq16:
		return rewriteValue386_OpLeq16(v)
	case ssaop.OpLeq16U:
		return rewriteValue386_OpLeq16U(v)
	case ssaop.OpLeq32:
		return rewriteValue386_OpLeq32(v)
	case ssaop.OpLeq32F:
		return rewriteValue386_OpLeq32F(v)
	case ssaop.OpLeq32U:
		return rewriteValue386_OpLeq32U(v)
	case ssaop.OpLeq64F:
		return rewriteValue386_OpLeq64F(v)
	case ssaop.OpLeq8:
		return rewriteValue386_OpLeq8(v)
	case ssaop.OpLeq8U:
		return rewriteValue386_OpLeq8U(v)
	case ssaop.OpLess16:
		return rewriteValue386_OpLess16(v)
	case ssaop.OpLess16U:
		return rewriteValue386_OpLess16U(v)
	case ssaop.OpLess32:
		return rewriteValue386_OpLess32(v)
	case ssaop.OpLess32F:
		return rewriteValue386_OpLess32F(v)
	case ssaop.OpLess32U:
		return rewriteValue386_OpLess32U(v)
	case ssaop.OpLess64F:
		return rewriteValue386_OpLess64F(v)
	case ssaop.OpLess8:
		return rewriteValue386_OpLess8(v)
	case ssaop.OpLess8U:
		return rewriteValue386_OpLess8U(v)
	case ssaop.OpLoad:
		return rewriteValue386_OpLoad(v)
	case ssaop.OpLocalAddr:
		return rewriteValue386_OpLocalAddr(v)
	case ssaop.OpLsh16x16:
		return rewriteValue386_OpLsh16x16(v)
	case ssaop.OpLsh16x32:
		return rewriteValue386_OpLsh16x32(v)
	case ssaop.OpLsh16x64:
		return rewriteValue386_OpLsh16x64(v)
	case ssaop.OpLsh16x8:
		return rewriteValue386_OpLsh16x8(v)
	case ssaop.OpLsh32x16:
		return rewriteValue386_OpLsh32x16(v)
	case ssaop.OpLsh32x32:
		return rewriteValue386_OpLsh32x32(v)
	case ssaop.OpLsh32x64:
		return rewriteValue386_OpLsh32x64(v)
	case ssaop.OpLsh32x8:
		return rewriteValue386_OpLsh32x8(v)
	case ssaop.OpLsh8x16:
		return rewriteValue386_OpLsh8x16(v)
	case ssaop.OpLsh8x32:
		return rewriteValue386_OpLsh8x32(v)
	case ssaop.OpLsh8x64:
		return rewriteValue386_OpLsh8x64(v)
	case ssaop.OpLsh8x8:
		return rewriteValue386_OpLsh8x8(v)
	case ssaop.OpMod16:
		v.Op = ssaop.Op386MODW
		return true
	case ssaop.OpMod16u:
		v.Op = ssaop.Op386MODWU
		return true
	case ssaop.OpMod32:
		v.Op = ssaop.Op386MODL
		return true
	case ssaop.OpMod32u:
		v.Op = ssaop.Op386MODLU
		return true
	case ssaop.OpMod8:
		return rewriteValue386_OpMod8(v)
	case ssaop.OpMod8u:
		return rewriteValue386_OpMod8u(v)
	case ssaop.OpMove:
		return rewriteValue386_OpMove(v)
	case ssaop.OpMul16:
		v.Op = ssaop.Op386MULL
		return true
	case ssaop.OpMul32:
		v.Op = ssaop.Op386MULL
		return true
	case ssaop.OpMul32F:
		v.Op = ssaop.Op386MULSS
		return true
	case ssaop.OpMul32uhilo:
		v.Op = ssaop.Op386MULLQU
		return true
	case ssaop.OpMul64F:
		v.Op = ssaop.Op386MULSD
		return true
	case ssaop.OpMul8:
		v.Op = ssaop.Op386MULL
		return true
	case ssaop.OpNeg16:
		v.Op = ssaop.Op386NEGL
		return true
	case ssaop.OpNeg32:
		v.Op = ssaop.Op386NEGL
		return true
	case ssaop.OpNeg32F:
		return rewriteValue386_OpNeg32F(v)
	case ssaop.OpNeg64F:
		return rewriteValue386_OpNeg64F(v)
	case ssaop.OpNeg8:
		v.Op = ssaop.Op386NEGL
		return true
	case ssaop.OpNeq16:
		return rewriteValue386_OpNeq16(v)
	case ssaop.OpNeq32:
		return rewriteValue386_OpNeq32(v)
	case ssaop.OpNeq32F:
		return rewriteValue386_OpNeq32F(v)
	case ssaop.OpNeq64F:
		return rewriteValue386_OpNeq64F(v)
	case ssaop.OpNeq8:
		return rewriteValue386_OpNeq8(v)
	case ssaop.OpNeqB:
		return rewriteValue386_OpNeqB(v)
	case ssaop.OpNeqPtr:
		return rewriteValue386_OpNeqPtr(v)
	case ssaop.OpNilCheck:
		v.Op = ssaop.Op386LoweredNilCheck
		return true
	case ssaop.OpNot:
		return rewriteValue386_OpNot(v)
	case ssaop.OpOffPtr:
		return rewriteValue386_OpOffPtr(v)
	case ssaop.OpOr16:
		v.Op = ssaop.Op386ORL
		return true
	case ssaop.OpOr32:
		v.Op = ssaop.Op386ORL
		return true
	case ssaop.OpOr8:
		v.Op = ssaop.Op386ORL
		return true
	case ssaop.OpOrB:
		v.Op = ssaop.Op386ORL
		return true
	case ssaop.OpPanicBounds:
		v.Op = ssaop.Op386LoweredPanicBoundsRR
		return true
	case ssaop.OpPanicExtend:
		v.Op = ssaop.Op386LoweredPanicExtendRR
		return true
	case ssaop.OpRotateLeft16:
		v.Op = ssaop.Op386ROLW
		return true
	case ssaop.OpRotateLeft32:
		v.Op = ssaop.Op386ROLL
		return true
	case ssaop.OpRotateLeft8:
		v.Op = ssaop.Op386ROLB
		return true
	case ssaop.OpRound32F:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpRound64F:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpRsh16Ux16:
		return rewriteValue386_OpRsh16Ux16(v)
	case ssaop.OpRsh16Ux32:
		return rewriteValue386_OpRsh16Ux32(v)
	case ssaop.OpRsh16Ux64:
		return rewriteValue386_OpRsh16Ux64(v)
	case ssaop.OpRsh16Ux8:
		return rewriteValue386_OpRsh16Ux8(v)
	case ssaop.OpRsh16x16:
		return rewriteValue386_OpRsh16x16(v)
	case ssaop.OpRsh16x32:
		return rewriteValue386_OpRsh16x32(v)
	case ssaop.OpRsh16x64:
		return rewriteValue386_OpRsh16x64(v)
	case ssaop.OpRsh16x8:
		return rewriteValue386_OpRsh16x8(v)
	case ssaop.OpRsh32Ux16:
		return rewriteValue386_OpRsh32Ux16(v)
	case ssaop.OpRsh32Ux32:
		return rewriteValue386_OpRsh32Ux32(v)
	case ssaop.OpRsh32Ux64:
		return rewriteValue386_OpRsh32Ux64(v)
	case ssaop.OpRsh32Ux8:
		return rewriteValue386_OpRsh32Ux8(v)
	case ssaop.OpRsh32x16:
		return rewriteValue386_OpRsh32x16(v)
	case ssaop.OpRsh32x32:
		return rewriteValue386_OpRsh32x32(v)
	case ssaop.OpRsh32x64:
		return rewriteValue386_OpRsh32x64(v)
	case ssaop.OpRsh32x8:
		return rewriteValue386_OpRsh32x8(v)
	case ssaop.OpRsh8Ux16:
		return rewriteValue386_OpRsh8Ux16(v)
	case ssaop.OpRsh8Ux32:
		return rewriteValue386_OpRsh8Ux32(v)
	case ssaop.OpRsh8Ux64:
		return rewriteValue386_OpRsh8Ux64(v)
	case ssaop.OpRsh8Ux8:
		return rewriteValue386_OpRsh8Ux8(v)
	case ssaop.OpRsh8x16:
		return rewriteValue386_OpRsh8x16(v)
	case ssaop.OpRsh8x32:
		return rewriteValue386_OpRsh8x32(v)
	case ssaop.OpRsh8x64:
		return rewriteValue386_OpRsh8x64(v)
	case ssaop.OpRsh8x8:
		return rewriteValue386_OpRsh8x8(v)
	case ssaop.OpSelect0:
		return rewriteValue386_OpSelect0(v)
	case ssaop.OpSelect1:
		return rewriteValue386_OpSelect1(v)
	case ssaop.OpSignExt16to32:
		v.Op = ssaop.Op386MOVWLSX
		return true
	case ssaop.OpSignExt8to16:
		v.Op = ssaop.Op386MOVBLSX
		return true
	case ssaop.OpSignExt8to32:
		v.Op = ssaop.Op386MOVBLSX
		return true
	case ssaop.OpSignmask:
		return rewriteValue386_OpSignmask(v)
	case ssaop.OpSlicemask:
		return rewriteValue386_OpSlicemask(v)
	case ssaop.OpSqrt:
		v.Op = ssaop.Op386SQRTSD
		return true
	case ssaop.OpSqrt32:
		v.Op = ssaop.Op386SQRTSS
		return true
	case ssaop.OpStaticCall:
		v.Op = ssaop.Op386CALLstatic
		return true
	case ssaop.OpStore:
		return rewriteValue386_OpStore(v)
	case ssaop.OpSub16:
		v.Op = ssaop.Op386SUBL
		return true
	case ssaop.OpSub32:
		v.Op = ssaop.Op386SUBL
		return true
	case ssaop.OpSub32F:
		v.Op = ssaop.Op386SUBSS
		return true
	case ssaop.OpSub32carry:
		v.Op = ssaop.Op386SUBLcarry
		return true
	case ssaop.OpSub32withcarry:
		v.Op = ssaop.Op386SBBL
		return true
	case ssaop.OpSub64F:
		v.Op = ssaop.Op386SUBSD
		return true
	case ssaop.OpSub8:
		v.Op = ssaop.Op386SUBL
		return true
	case ssaop.OpSubPtr:
		v.Op = ssaop.Op386SUBL
		return true
	case ssaop.OpTailCall:
		v.Op = ssaop.Op386CALLtail
		return true
	case ssaop.OpTailCallInter:
		v.Op = ssaop.Op386CALLtailinter
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
		v.Op = ssaop.Op386LoweredWB
		return true
	case ssaop.OpXor16:
		v.Op = ssaop.Op386XORL
		return true
	case ssaop.OpXor32:
		v.Op = ssaop.Op386XORL
		return true
	case ssaop.OpXor8:
		v.Op = ssaop.Op386XORL
		return true
	case ssaop.OpZero:
		return rewriteValue386_OpZero(v)
	case ssaop.OpZeroExt16to32:
		v.Op = ssaop.Op386MOVWLZX
		return true
	case ssaop.OpZeroExt8to16:
		v.Op = ssaop.Op386MOVBLZX
		return true
	case ssaop.OpZeroExt8to32:
		v.Op = ssaop.Op386MOVBLZX
		return true
	case ssaop.OpZeromask:
		return rewriteValue386_OpZeromask(v)
	}
	return false
}
func rewriteValue386_Op386ADCL(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADCL x (MOVLconst [c]) f)
	// result: (ADCLconst [c] x f)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.Op386MOVLconst {
				continue
			}
			c := ssa.AuxIntToInt32(v_1.AuxInt)
			f := v_2
			v.Reset(ssaop.Op386ADCLconst)
			v.AuxInt = ssa.Int32ToAuxInt(c)
			v.AddArg2(x, f)
			return true
		}
		break
	}
	return false
}
func rewriteValue386_Op386ADDL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDL x (MOVLconst <t> [c]))
	// cond: !t.IsPtr()
	// result: (ADDLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.Op386MOVLconst {
				continue
			}
			t := v_1.Type
			c := ssa.AuxIntToInt32(v_1.AuxInt)
			if !(!t.IsPtr()) {
				continue
			}
			v.Reset(ssaop.Op386ADDLconst)
			v.AuxInt = ssa.Int32ToAuxInt(c)
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
			if v_1.Op != ssaop.Op386SHLLconst || ssa.AuxIntToInt32(v_1.AuxInt) != 3 {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.Op386LEAL8)
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
			if v_1.Op != ssaop.Op386SHLLconst || ssa.AuxIntToInt32(v_1.AuxInt) != 2 {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.Op386LEAL4)
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
			if v_1.Op != ssaop.Op386SHLLconst || ssa.AuxIntToInt32(v_1.AuxInt) != 1 {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.Op386LEAL2)
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
			if v_1.Op != ssaop.Op386ADDL {
				continue
			}
			y := v_1.Args[1]
			if y != v_1.Args[0] {
				continue
			}
			v.Reset(ssaop.Op386LEAL2)
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
			if v_1.Op != ssaop.Op386ADDL {
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
				v.Reset(ssaop.Op386LEAL2)
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
			if v_0.Op != ssaop.Op386ADDLconst {
				continue
			}
			c := ssa.AuxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			y := v_1
			v.Reset(ssaop.Op386LEAL1)
			v.AuxInt = ssa.Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADDL x (LEAL [c] {s} y))
	// cond: x.Op != ssaop.OpSB && y.Op != ssaop.OpSB
	// result: (LEAL1 [c] {s} x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.Op386LEAL {
				continue
			}
			c := ssa.AuxIntToInt32(v_1.AuxInt)
			s := ssa.AuxToSym(v_1.Aux)
			y := v_1.Args[0]
			if !(x.Op != ssaop.OpSB && y.Op != ssaop.OpSB) {
				continue
			}
			v.Reset(ssaop.Op386LEAL1)
			v.AuxInt = ssa.Int32ToAuxInt(c)
			v.Aux = ssa.SymToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADDL x l:(MOVLload [off] {sym} ptr mem))
	// cond: ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)
	// result: (ADDLload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != ssaop.Op386MOVLload {
				continue
			}
			off := ssa.AuxIntToInt32(l.AuxInt)
			sym := ssa.AuxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)) {
				continue
			}
			v.Reset(ssaop.Op386ADDLload)
			v.AuxInt = ssa.Int32ToAuxInt(off)
			v.Aux = ssa.SymToAux(sym)
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
			if v_1.Op != ssaop.Op386NEGL {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.Op386SUBL)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue386_Op386ADDLcarry(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDLcarry x (MOVLconst [c]))
	// result: (ADDLconstcarry [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.Op386MOVLconst {
				continue
			}
			c := ssa.AuxIntToInt32(v_1.AuxInt)
			v.Reset(ssaop.Op386ADDLconstcarry)
			v.AuxInt = ssa.Int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValue386_Op386ADDLconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ADDLconst [c] (ADDL x y))
	// result: (LEAL1 [c] x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386ADDL {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.Op386LEAL1)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (ADDLconst [c] (LEAL [d] {s} x))
	// cond: ssa.Is32Bit(int64(c)+int64(d))
	// result: (LEAL [c+d] {s} x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		s := ssa.AuxToSym(v_0.Aux)
		x := v_0.Args[0]
		if !(ssa.Is32Bit(int64(c) + int64(d))) {
			break
		}
		v.Reset(ssaop.Op386LEAL)
		v.AuxInt = ssa.Int32ToAuxInt(c + d)
		v.Aux = ssa.SymToAux(s)
		v.AddArg(x)
		return true
	}
	// match: (ADDLconst [c] x:(SP))
	// result: (LEAL [c] x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		x := v_0
		if x.Op != ssaop.OpSP {
			break
		}
		v.Reset(ssaop.Op386LEAL)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (ADDLconst [c] (LEAL1 [d] {s} x y))
	// cond: ssa.Is32Bit(int64(c)+int64(d))
	// result: (LEAL1 [c+d] {s} x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386LEAL1 {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		s := ssa.AuxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(ssa.Is32Bit(int64(c) + int64(d))) {
			break
		}
		v.Reset(ssaop.Op386LEAL1)
		v.AuxInt = ssa.Int32ToAuxInt(c + d)
		v.Aux = ssa.SymToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (ADDLconst [c] (LEAL2 [d] {s} x y))
	// cond: ssa.Is32Bit(int64(c)+int64(d))
	// result: (LEAL2 [c+d] {s} x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386LEAL2 {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		s := ssa.AuxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(ssa.Is32Bit(int64(c) + int64(d))) {
			break
		}
		v.Reset(ssaop.Op386LEAL2)
		v.AuxInt = ssa.Int32ToAuxInt(c + d)
		v.Aux = ssa.SymToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (ADDLconst [c] (LEAL4 [d] {s} x y))
	// cond: ssa.Is32Bit(int64(c)+int64(d))
	// result: (LEAL4 [c+d] {s} x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386LEAL4 {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		s := ssa.AuxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(ssa.Is32Bit(int64(c) + int64(d))) {
			break
		}
		v.Reset(ssaop.Op386LEAL4)
		v.AuxInt = ssa.Int32ToAuxInt(c + d)
		v.Aux = ssa.SymToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (ADDLconst [c] (LEAL8 [d] {s} x y))
	// cond: ssa.Is32Bit(int64(c)+int64(d))
	// result: (LEAL8 [c+d] {s} x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386LEAL8 {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		s := ssa.AuxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(ssa.Is32Bit(int64(c) + int64(d))) {
			break
		}
		v.Reset(ssaop.Op386LEAL8)
		v.AuxInt = ssa.Int32ToAuxInt(c + d)
		v.Aux = ssa.SymToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (ADDLconst [c] x)
	// cond: c==0
	// result: x
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(c == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ADDLconst [c] (MOVLconst [d]))
	// result: (MOVLconst [c+d])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c + d)
		return true
	}
	// match: (ADDLconst [c] (ADDLconst [d] x))
	// result: (ADDLconst [c+d] x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.Op386ADDLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386ADDLconstmodify(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ADDLconstmodify [valoff1] {sym} (ADDLconst [off2] base) mem)
	// cond: valoff1.CanAdd32(off2)
	// result: (ADDLconstmodify [valoff1.AddOffset32(off2)] {sym} base mem)
	for {
		valoff1 := ssa.AuxIntToValAndOff(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(valoff1.CanAdd32(off2)) {
			break
		}
		v.Reset(ssaop.Op386ADDLconstmodify)
		v.AuxInt = ssa.ValAndOffToAuxInt(valoff1.AddOffset32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (ADDLconstmodify [valoff1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: valoff1.CanAdd32(off2) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (ADDLconstmodify [valoff1.AddOffset32(off2)] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := ssa.AuxIntToValAndOff(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(valoff1.CanAdd32(off2) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386ADDLconstmodify)
		v.AuxInt = ssa.ValAndOffToAuxInt(valoff1.AddOffset32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ADDLload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ADDLload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (ADDLload [off1+off2] {sym} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386ADDLload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ADDLload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (ADDLload [off1+off2] {ssa.MergeSym(sym1,sym2)} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386ADDLload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ADDLmodify(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ADDLmodify [off1] {sym} (ADDLconst [off2] base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (ADDLmodify [off1+off2] {sym} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386ADDLmodify)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (ADDLmodify [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (ADDLmodify [off1+off2] {ssa.MergeSym(sym1,sym2)} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386ADDLmodify)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ADDSD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDSD x l:(MOVSDload [off] {sym} ptr mem))
	// cond: ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)
	// result: (ADDSDload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != ssaop.Op386MOVSDload {
				continue
			}
			off := ssa.AuxIntToInt32(l.AuxInt)
			sym := ssa.AuxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)) {
				continue
			}
			v.Reset(ssaop.Op386ADDSDload)
			v.AuxInt = ssa.Int32ToAuxInt(off)
			v.Aux = ssa.SymToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValue386_Op386ADDSDload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ADDSDload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (ADDSDload [off1+off2] {sym} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386ADDSDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ADDSDload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (ADDSDload [off1+off2] {ssa.MergeSym(sym1,sym2)} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386ADDSDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ADDSS(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDSS x l:(MOVSSload [off] {sym} ptr mem))
	// cond: ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)
	// result: (ADDSSload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != ssaop.Op386MOVSSload {
				continue
			}
			off := ssa.AuxIntToInt32(l.AuxInt)
			sym := ssa.AuxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)) {
				continue
			}
			v.Reset(ssaop.Op386ADDSSload)
			v.AuxInt = ssa.Int32ToAuxInt(off)
			v.Aux = ssa.SymToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValue386_Op386ADDSSload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ADDSSload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (ADDSSload [off1+off2] {sym} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386ADDSSload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ADDSSload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (ADDSSload [off1+off2] {ssa.MergeSym(sym1,sym2)} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386ADDSSload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ANDL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ANDL x (MOVLconst [c]))
	// result: (ANDLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.Op386MOVLconst {
				continue
			}
			c := ssa.AuxIntToInt32(v_1.AuxInt)
			v.Reset(ssaop.Op386ANDLconst)
			v.AuxInt = ssa.Int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ANDL x l:(MOVLload [off] {sym} ptr mem))
	// cond: ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)
	// result: (ANDLload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != ssaop.Op386MOVLload {
				continue
			}
			off := ssa.AuxIntToInt32(l.AuxInt)
			sym := ssa.AuxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)) {
				continue
			}
			v.Reset(ssaop.Op386ANDLload)
			v.AuxInt = ssa.Int32ToAuxInt(off)
			v.Aux = ssa.SymToAux(sym)
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
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue386_Op386ANDLconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ANDLconst [c] (ANDLconst [d] x))
	// result: (ANDLconst [c & d] x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386ANDLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.Op386ANDLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & d)
		v.AddArg(x)
		return true
	}
	// match: (ANDLconst [c] _)
	// cond: c==0
	// result: (MOVLconst [0])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if !(c == 0) {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (ANDLconst [c] x)
	// cond: c==-1
	// result: x
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(c == -1) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ANDLconst [c] (MOVLconst [d]))
	// result: (MOVLconst [c&d])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & d)
		return true
	}
	return false
}
func rewriteValue386_Op386ANDLconstmodify(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ANDLconstmodify [valoff1] {sym} (ADDLconst [off2] base) mem)
	// cond: valoff1.CanAdd32(off2)
	// result: (ANDLconstmodify [valoff1.AddOffset32(off2)] {sym} base mem)
	for {
		valoff1 := ssa.AuxIntToValAndOff(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(valoff1.CanAdd32(off2)) {
			break
		}
		v.Reset(ssaop.Op386ANDLconstmodify)
		v.AuxInt = ssa.ValAndOffToAuxInt(valoff1.AddOffset32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (ANDLconstmodify [valoff1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: valoff1.CanAdd32(off2) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (ANDLconstmodify [valoff1.AddOffset32(off2)] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := ssa.AuxIntToValAndOff(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(valoff1.CanAdd32(off2) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386ANDLconstmodify)
		v.AuxInt = ssa.ValAndOffToAuxInt(valoff1.AddOffset32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ANDLload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ANDLload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (ANDLload [off1+off2] {sym} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386ANDLload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ANDLload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (ANDLload [off1+off2] {ssa.MergeSym(sym1,sym2)} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386ANDLload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ANDLmodify(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ANDLmodify [off1] {sym} (ADDLconst [off2] base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (ANDLmodify [off1+off2] {sym} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386ANDLmodify)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (ANDLmodify [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (ANDLmodify [off1+off2] {ssa.MergeSym(sym1,sym2)} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386ANDLmodify)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386CMPB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPB x (MOVLconst [c]))
	// result: (CMPBconst x [int8(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.Op386CMPBconst)
		v.AuxInt = ssa.Int8ToAuxInt(int8(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPB (MOVLconst [c]) x)
	// result: (InvertFlags (CMPBconst x [int8(c)]))
	for {
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.Op386InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPBconst, types.TypeFlags)
		v0.AuxInt = ssa.Int8ToAuxInt(int8(c))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPB x y)
	// cond: ssa.CanonLessThan(x,y)
	// result: (InvertFlags (CMPB y x))
	for {
		x := v_0
		y := v_1
		if !(ssa.CanonLessThan(x, y)) {
			break
		}
		v.Reset(ssaop.Op386InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPB, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPB l:(MOVBload {sym} [off] ptr mem) x)
	// cond: ssa.CanMergeLoad(v, l) && ssa.Clobber(l)
	// result: (CMPBload {sym} [off] ptr x mem)
	for {
		l := v_0
		if l.Op != ssaop.Op386MOVBload {
			break
		}
		off := ssa.AuxIntToInt32(l.AuxInt)
		sym := ssa.AuxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		x := v_1
		if !(ssa.CanMergeLoad(v, l) && ssa.Clobber(l)) {
			break
		}
		v.Reset(ssaop.Op386CMPBload)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (CMPB x l:(MOVBload {sym} [off] ptr mem))
	// cond: ssa.CanMergeLoad(v, l) && ssa.Clobber(l)
	// result: (InvertFlags (CMPBload {sym} [off] ptr x mem))
	for {
		x := v_0
		l := v_1
		if l.Op != ssaop.Op386MOVBload {
			break
		}
		off := ssa.AuxIntToInt32(l.AuxInt)
		sym := ssa.AuxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(ssa.CanMergeLoad(v, l) && ssa.Clobber(l)) {
			break
		}
		v.Reset(ssaop.Op386InvertFlags)
		v0 := b.NewValue0(l.Pos, ssaop.Op386CMPBload, types.TypeFlags)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg3(ptr, x, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue386_Op386CMPBconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)==y
	// result: (FlagEQ)
	for {
		y := ssa.AuxIntToInt8(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		x := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(int8(x) == y) {
			break
		}
		v.Reset(ssaop.Op386FlagEQ)
		return true
	}
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)<y && uint8(x)<uint8(y)
	// result: (FlagLT_ULT)
	for {
		y := ssa.AuxIntToInt8(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		x := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(int8(x) < y && uint8(x) < uint8(y)) {
			break
		}
		v.Reset(ssaop.Op386FlagLT_ULT)
		return true
	}
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)<y && uint8(x)>uint8(y)
	// result: (FlagLT_UGT)
	for {
		y := ssa.AuxIntToInt8(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		x := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(int8(x) < y && uint8(x) > uint8(y)) {
			break
		}
		v.Reset(ssaop.Op386FlagLT_UGT)
		return true
	}
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)>y && uint8(x)<uint8(y)
	// result: (FlagGT_ULT)
	for {
		y := ssa.AuxIntToInt8(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		x := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(int8(x) > y && uint8(x) < uint8(y)) {
			break
		}
		v.Reset(ssaop.Op386FlagGT_ULT)
		return true
	}
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)>y && uint8(x)>uint8(y)
	// result: (FlagGT_UGT)
	for {
		y := ssa.AuxIntToInt8(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		x := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(int8(x) > y && uint8(x) > uint8(y)) {
			break
		}
		v.Reset(ssaop.Op386FlagGT_UGT)
		return true
	}
	// match: (CMPBconst (ANDLconst _ [m]) [n])
	// cond: 0 <= int8(m) && int8(m) < n
	// result: (FlagLT_ULT)
	for {
		n := ssa.AuxIntToInt8(v.AuxInt)
		if v_0.Op != ssaop.Op386ANDLconst {
			break
		}
		m := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(0 <= int8(m) && int8(m) < n) {
			break
		}
		v.Reset(ssaop.Op386FlagLT_ULT)
		return true
	}
	// match: (CMPBconst l:(ANDL x y) [0])
	// cond: l.Uses==1
	// result: (TESTB x y)
	for {
		if ssa.AuxIntToInt8(v.AuxInt) != 0 {
			break
		}
		l := v_0
		if l.Op != ssaop.Op386ANDL {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(l.Uses == 1) {
			break
		}
		v.Reset(ssaop.Op386TESTB)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMPBconst l:(ANDLconst [c] x) [0])
	// cond: l.Uses==1
	// result: (TESTBconst [int8(c)] x)
	for {
		if ssa.AuxIntToInt8(v.AuxInt) != 0 {
			break
		}
		l := v_0
		if l.Op != ssaop.Op386ANDLconst {
			break
		}
		c := ssa.AuxIntToInt32(l.AuxInt)
		x := l.Args[0]
		if !(l.Uses == 1) {
			break
		}
		v.Reset(ssaop.Op386TESTBconst)
		v.AuxInt = ssa.Int8ToAuxInt(int8(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPBconst x [0])
	// result: (TESTB x x)
	for {
		if ssa.AuxIntToInt8(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386TESTB)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPBconst l:(MOVBload {sym} [off] ptr mem) [c])
	// cond: l.Uses == 1 && ssa.Clobber(l)
	// result: @l.Block (CMPBconstload {sym} [ssa.MakeValAndOff(int32(c),off)] ptr mem)
	for {
		c := ssa.AuxIntToInt8(v.AuxInt)
		l := v_0
		if l.Op != ssaop.Op386MOVBload {
			break
		}
		off := ssa.AuxIntToInt32(l.AuxInt)
		sym := ssa.AuxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(l.Uses == 1 && ssa.Clobber(l)) {
			break
		}
		b = l.Block
		v0 := b.NewValue0(l.Pos, ssaop.Op386CMPBconstload, types.TypeFlags)
		v.CopyOf(v0)
		v0.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(int32(c), off))
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386CMPBload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMPBload {sym} [off] ptr (MOVLconst [c]) mem)
	// result: (CMPBconstload {sym} [ssa.MakeValAndOff(int32(int8(c)),off)] ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.Op386CMPBconstload)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(int32(int8(c)), off))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386CMPL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPL x (MOVLconst [c]))
	// result: (CMPLconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.Op386CMPLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (CMPL (MOVLconst [c]) x)
	// result: (InvertFlags (CMPLconst x [c]))
	for {
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.Op386InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPLconst, types.TypeFlags)
		v0.AuxInt = ssa.Int32ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPL x y)
	// cond: ssa.CanonLessThan(x,y)
	// result: (InvertFlags (CMPL y x))
	for {
		x := v_0
		y := v_1
		if !(ssa.CanonLessThan(x, y)) {
			break
		}
		v.Reset(ssaop.Op386InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPL, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPL l:(MOVLload {sym} [off] ptr mem) x)
	// cond: ssa.CanMergeLoad(v, l) && ssa.Clobber(l)
	// result: (CMPLload {sym} [off] ptr x mem)
	for {
		l := v_0
		if l.Op != ssaop.Op386MOVLload {
			break
		}
		off := ssa.AuxIntToInt32(l.AuxInt)
		sym := ssa.AuxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		x := v_1
		if !(ssa.CanMergeLoad(v, l) && ssa.Clobber(l)) {
			break
		}
		v.Reset(ssaop.Op386CMPLload)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (CMPL x l:(MOVLload {sym} [off] ptr mem))
	// cond: ssa.CanMergeLoad(v, l) && ssa.Clobber(l)
	// result: (InvertFlags (CMPLload {sym} [off] ptr x mem))
	for {
		x := v_0
		l := v_1
		if l.Op != ssaop.Op386MOVLload {
			break
		}
		off := ssa.AuxIntToInt32(l.AuxInt)
		sym := ssa.AuxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(ssa.CanMergeLoad(v, l) && ssa.Clobber(l)) {
			break
		}
		v.Reset(ssaop.Op386InvertFlags)
		v0 := b.NewValue0(l.Pos, ssaop.Op386CMPLload, types.TypeFlags)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg3(ptr, x, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue386_Op386CMPLconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: x==y
	// result: (FlagEQ)
	for {
		y := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		x := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(x == y) {
			break
		}
		v.Reset(ssaop.Op386FlagEQ)
		return true
	}
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: x<y && uint32(x)<uint32(y)
	// result: (FlagLT_ULT)
	for {
		y := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		x := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(x < y && uint32(x) < uint32(y)) {
			break
		}
		v.Reset(ssaop.Op386FlagLT_ULT)
		return true
	}
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: x<y && uint32(x)>uint32(y)
	// result: (FlagLT_UGT)
	for {
		y := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		x := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(x < y && uint32(x) > uint32(y)) {
			break
		}
		v.Reset(ssaop.Op386FlagLT_UGT)
		return true
	}
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: x>y && uint32(x)<uint32(y)
	// result: (FlagGT_ULT)
	for {
		y := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		x := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(x > y && uint32(x) < uint32(y)) {
			break
		}
		v.Reset(ssaop.Op386FlagGT_ULT)
		return true
	}
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: x>y && uint32(x)>uint32(y)
	// result: (FlagGT_UGT)
	for {
		y := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		x := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(x > y && uint32(x) > uint32(y)) {
			break
		}
		v.Reset(ssaop.Op386FlagGT_UGT)
		return true
	}
	// match: (CMPLconst (SHRLconst _ [c]) [n])
	// cond: 0 <= n && 0 < c && c <= 32 && (1<<uint64(32-c)) <= uint64(n)
	// result: (FlagLT_ULT)
	for {
		n := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386SHRLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(0 <= n && 0 < c && c <= 32 && (1<<uint64(32-c)) <= uint64(n)) {
			break
		}
		v.Reset(ssaop.Op386FlagLT_ULT)
		return true
	}
	// match: (CMPLconst (ANDLconst _ [m]) [n])
	// cond: 0 <= m && m < n
	// result: (FlagLT_ULT)
	for {
		n := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386ANDLconst {
			break
		}
		m := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(0 <= m && m < n) {
			break
		}
		v.Reset(ssaop.Op386FlagLT_ULT)
		return true
	}
	// match: (CMPLconst l:(ANDL x y) [0])
	// cond: l.Uses==1
	// result: (TESTL x y)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		l := v_0
		if l.Op != ssaop.Op386ANDL {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(l.Uses == 1) {
			break
		}
		v.Reset(ssaop.Op386TESTL)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMPLconst l:(ANDLconst [c] x) [0])
	// cond: l.Uses==1
	// result: (TESTLconst [c] x)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		l := v_0
		if l.Op != ssaop.Op386ANDLconst {
			break
		}
		c := ssa.AuxIntToInt32(l.AuxInt)
		x := l.Args[0]
		if !(l.Uses == 1) {
			break
		}
		v.Reset(ssaop.Op386TESTLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (CMPLconst x [0])
	// result: (TESTL x x)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386TESTL)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPLconst l:(MOVLload {sym} [off] ptr mem) [c])
	// cond: l.Uses == 1 && ssa.Clobber(l)
	// result: @l.Block (CMPLconstload {sym} [ssa.MakeValAndOff(int32(c),off)] ptr mem)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		l := v_0
		if l.Op != ssaop.Op386MOVLload {
			break
		}
		off := ssa.AuxIntToInt32(l.AuxInt)
		sym := ssa.AuxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(l.Uses == 1 && ssa.Clobber(l)) {
			break
		}
		b = l.Block
		v0 := b.NewValue0(l.Pos, ssaop.Op386CMPLconstload, types.TypeFlags)
		v.CopyOf(v0)
		v0.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(int32(c), off))
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386CMPLload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMPLload {sym} [off] ptr (MOVLconst [c]) mem)
	// result: (CMPLconstload {sym} [ssa.MakeValAndOff(c,off)] ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.Op386CMPLconstload)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(c, off))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386CMPW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPW x (MOVLconst [c]))
	// result: (CMPWconst x [int16(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.Op386CMPWconst)
		v.AuxInt = ssa.Int16ToAuxInt(int16(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPW (MOVLconst [c]) x)
	// result: (InvertFlags (CMPWconst x [int16(c)]))
	for {
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.Op386InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPWconst, types.TypeFlags)
		v0.AuxInt = ssa.Int16ToAuxInt(int16(c))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPW x y)
	// cond: ssa.CanonLessThan(x,y)
	// result: (InvertFlags (CMPW y x))
	for {
		x := v_0
		y := v_1
		if !(ssa.CanonLessThan(x, y)) {
			break
		}
		v.Reset(ssaop.Op386InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPW, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPW l:(MOVWload {sym} [off] ptr mem) x)
	// cond: ssa.CanMergeLoad(v, l) && ssa.Clobber(l)
	// result: (CMPWload {sym} [off] ptr x mem)
	for {
		l := v_0
		if l.Op != ssaop.Op386MOVWload {
			break
		}
		off := ssa.AuxIntToInt32(l.AuxInt)
		sym := ssa.AuxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		x := v_1
		if !(ssa.CanMergeLoad(v, l) && ssa.Clobber(l)) {
			break
		}
		v.Reset(ssaop.Op386CMPWload)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (CMPW x l:(MOVWload {sym} [off] ptr mem))
	// cond: ssa.CanMergeLoad(v, l) && ssa.Clobber(l)
	// result: (InvertFlags (CMPWload {sym} [off] ptr x mem))
	for {
		x := v_0
		l := v_1
		if l.Op != ssaop.Op386MOVWload {
			break
		}
		off := ssa.AuxIntToInt32(l.AuxInt)
		sym := ssa.AuxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(ssa.CanMergeLoad(v, l) && ssa.Clobber(l)) {
			break
		}
		v.Reset(ssaop.Op386InvertFlags)
		v0 := b.NewValue0(l.Pos, ssaop.Op386CMPWload, types.TypeFlags)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg3(ptr, x, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue386_Op386CMPWconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)==y
	// result: (FlagEQ)
	for {
		y := ssa.AuxIntToInt16(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		x := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(int16(x) == y) {
			break
		}
		v.Reset(ssaop.Op386FlagEQ)
		return true
	}
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)<y && uint16(x)<uint16(y)
	// result: (FlagLT_ULT)
	for {
		y := ssa.AuxIntToInt16(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		x := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(int16(x) < y && uint16(x) < uint16(y)) {
			break
		}
		v.Reset(ssaop.Op386FlagLT_ULT)
		return true
	}
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)<y && uint16(x)>uint16(y)
	// result: (FlagLT_UGT)
	for {
		y := ssa.AuxIntToInt16(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		x := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(int16(x) < y && uint16(x) > uint16(y)) {
			break
		}
		v.Reset(ssaop.Op386FlagLT_UGT)
		return true
	}
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)>y && uint16(x)<uint16(y)
	// result: (FlagGT_ULT)
	for {
		y := ssa.AuxIntToInt16(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		x := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(int16(x) > y && uint16(x) < uint16(y)) {
			break
		}
		v.Reset(ssaop.Op386FlagGT_ULT)
		return true
	}
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)>y && uint16(x)>uint16(y)
	// result: (FlagGT_UGT)
	for {
		y := ssa.AuxIntToInt16(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		x := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(int16(x) > y && uint16(x) > uint16(y)) {
			break
		}
		v.Reset(ssaop.Op386FlagGT_UGT)
		return true
	}
	// match: (CMPWconst (ANDLconst _ [m]) [n])
	// cond: 0 <= int16(m) && int16(m) < n
	// result: (FlagLT_ULT)
	for {
		n := ssa.AuxIntToInt16(v.AuxInt)
		if v_0.Op != ssaop.Op386ANDLconst {
			break
		}
		m := ssa.AuxIntToInt32(v_0.AuxInt)
		if !(0 <= int16(m) && int16(m) < n) {
			break
		}
		v.Reset(ssaop.Op386FlagLT_ULT)
		return true
	}
	// match: (CMPWconst l:(ANDL x y) [0])
	// cond: l.Uses==1
	// result: (TESTW x y)
	for {
		if ssa.AuxIntToInt16(v.AuxInt) != 0 {
			break
		}
		l := v_0
		if l.Op != ssaop.Op386ANDL {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(l.Uses == 1) {
			break
		}
		v.Reset(ssaop.Op386TESTW)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMPWconst l:(ANDLconst [c] x) [0])
	// cond: l.Uses==1
	// result: (TESTWconst [int16(c)] x)
	for {
		if ssa.AuxIntToInt16(v.AuxInt) != 0 {
			break
		}
		l := v_0
		if l.Op != ssaop.Op386ANDLconst {
			break
		}
		c := ssa.AuxIntToInt32(l.AuxInt)
		x := l.Args[0]
		if !(l.Uses == 1) {
			break
		}
		v.Reset(ssaop.Op386TESTWconst)
		v.AuxInt = ssa.Int16ToAuxInt(int16(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPWconst x [0])
	// result: (TESTW x x)
	for {
		if ssa.AuxIntToInt16(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386TESTW)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPWconst l:(MOVWload {sym} [off] ptr mem) [c])
	// cond: l.Uses == 1 && ssa.Clobber(l)
	// result: @l.Block (CMPWconstload {sym} [ssa.MakeValAndOff(int32(c),off)] ptr mem)
	for {
		c := ssa.AuxIntToInt16(v.AuxInt)
		l := v_0
		if l.Op != ssaop.Op386MOVWload {
			break
		}
		off := ssa.AuxIntToInt32(l.AuxInt)
		sym := ssa.AuxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(l.Uses == 1 && ssa.Clobber(l)) {
			break
		}
		b = l.Block
		v0 := b.NewValue0(l.Pos, ssaop.Op386CMPWconstload, types.TypeFlags)
		v.CopyOf(v0)
		v0.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(int32(c), off))
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386CMPWload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMPWload {sym} [off] ptr (MOVLconst [c]) mem)
	// result: (CMPWconstload {sym} [ssa.MakeValAndOff(int32(int16(c)),off)] ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.Op386CMPWconstload)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(int32(int16(c)), off))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386DIVSD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (DIVSD x l:(MOVSDload [off] {sym} ptr mem))
	// cond: ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)
	// result: (DIVSDload x [off] {sym} ptr mem)
	for {
		x := v_0
		l := v_1
		if l.Op != ssaop.Op386MOVSDload {
			break
		}
		off := ssa.AuxIntToInt32(l.AuxInt)
		sym := ssa.AuxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)) {
			break
		}
		v.Reset(ssaop.Op386DIVSDload)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(x, ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386DIVSDload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (DIVSDload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (DIVSDload [off1+off2] {sym} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386DIVSDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (DIVSDload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (DIVSDload [off1+off2] {ssa.MergeSym(sym1,sym2)} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386DIVSDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386DIVSS(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (DIVSS x l:(MOVSSload [off] {sym} ptr mem))
	// cond: ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)
	// result: (DIVSSload x [off] {sym} ptr mem)
	for {
		x := v_0
		l := v_1
		if l.Op != ssaop.Op386MOVSSload {
			break
		}
		off := ssa.AuxIntToInt32(l.AuxInt)
		sym := ssa.AuxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)) {
			break
		}
		v.Reset(ssaop.Op386DIVSSload)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(x, ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386DIVSSload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (DIVSSload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (DIVSSload [off1+off2] {sym} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386DIVSSload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (DIVSSload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (DIVSSload [off1+off2] {ssa.MergeSym(sym1,sym2)} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386DIVSSload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386LEAL(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (LEAL [c] {s} (ADDLconst [d] x))
	// cond: ssa.Is32Bit(int64(c)+int64(d))
	// result: (LEAL [c+d] {s} x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		s := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.Is32Bit(int64(c) + int64(d))) {
			break
		}
		v.Reset(ssaop.Op386LEAL)
		v.AuxInt = ssa.Int32ToAuxInt(c + d)
		v.Aux = ssa.SymToAux(s)
		v.AddArg(x)
		return true
	}
	// match: (LEAL [c] {s} (ADDL x y))
	// cond: x.Op != ssaop.OpSB && y.Op != ssaop.OpSB
	// result: (LEAL1 [c] {s} x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		s := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if !(x.Op != ssaop.OpSB && y.Op != ssaop.OpSB) {
				continue
			}
			v.Reset(ssaop.Op386LEAL1)
			v.AuxInt = ssa.Int32ToAuxInt(c)
			v.Aux = ssa.SymToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAL [off1] {sym1} (LEAL [off2] {sym2} x))
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2)
	// result: (LEAL [off1+off2] {ssa.MergeSym(sym1,sym2)} x)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		x := v_0.Args[0]
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.Op386LEAL)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg(x)
		return true
	}
	// match: (LEAL [off1] {sym1} (LEAL1 [off2] {sym2} x y))
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2)
	// result: (LEAL1 [off1+off2] {ssa.MergeSym(sym1,sym2)} x y)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL1 {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.Op386LEAL1)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL [off1] {sym1} (LEAL2 [off2] {sym2} x y))
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2)
	// result: (LEAL2 [off1+off2] {ssa.MergeSym(sym1,sym2)} x y)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL2 {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.Op386LEAL2)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL [off1] {sym1} (LEAL4 [off2] {sym2} x y))
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2)
	// result: (LEAL4 [off1+off2] {ssa.MergeSym(sym1,sym2)} x y)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL4 {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.Op386LEAL4)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL [off1] {sym1} (LEAL8 [off2] {sym2} x y))
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2)
	// result: (LEAL8 [off1+off2] {ssa.MergeSym(sym1,sym2)} x y)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL8 {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.Op386LEAL8)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_Op386LEAL1(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LEAL1 [c] {s} (ADDLconst [d] x) y)
	// cond: ssa.Is32Bit(int64(c)+int64(d)) && x.Op != ssaop.OpSB
	// result: (LEAL1 [c+d] {s} x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		s := ssa.AuxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.Op386ADDLconst {
				continue
			}
			d := ssa.AuxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			y := v_1
			if !(ssa.Is32Bit(int64(c)+int64(d)) && x.Op != ssaop.OpSB) {
				continue
			}
			v.Reset(ssaop.Op386LEAL1)
			v.AuxInt = ssa.Int32ToAuxInt(c + d)
			v.Aux = ssa.SymToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAL1 [c] {s} x (SHLLconst [1] y))
	// result: (LEAL2 [c] {s} x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		s := ssa.AuxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.Op386SHLLconst || ssa.AuxIntToInt32(v_1.AuxInt) != 1 {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.Op386LEAL2)
			v.AuxInt = ssa.Int32ToAuxInt(c)
			v.Aux = ssa.SymToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAL1 [c] {s} x (SHLLconst [2] y))
	// result: (LEAL4 [c] {s} x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		s := ssa.AuxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.Op386SHLLconst || ssa.AuxIntToInt32(v_1.AuxInt) != 2 {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.Op386LEAL4)
			v.AuxInt = ssa.Int32ToAuxInt(c)
			v.Aux = ssa.SymToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAL1 [c] {s} x (SHLLconst [3] y))
	// result: (LEAL8 [c] {s} x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		s := ssa.AuxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.Op386SHLLconst || ssa.AuxIntToInt32(v_1.AuxInt) != 3 {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.Op386LEAL8)
			v.AuxInt = ssa.Int32ToAuxInt(c)
			v.Aux = ssa.SymToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAL1 [off1] {sym1} (LEAL [off2] {sym2} x) y)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && x.Op != ssaop.OpSB
	// result: (LEAL1 [off1+off2] {ssa.MergeSym(sym1,sym2)} x y)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.Op386LEAL {
				continue
			}
			off2 := ssa.AuxIntToInt32(v_0.AuxInt)
			sym2 := ssa.AuxToSym(v_0.Aux)
			x := v_0.Args[0]
			y := v_1
			if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && x.Op != ssaop.OpSB) {
				continue
			}
			v.Reset(ssaop.Op386LEAL1)
			v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
			v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAL1 [off1] {sym1} x (LEAL1 [off2] {sym2} y y))
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2)
	// result: (LEAL2 [off1+off2] {ssa.MergeSym(sym1, sym2)} x y)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.Op386LEAL1 {
				continue
			}
			off2 := ssa.AuxIntToInt32(v_1.AuxInt)
			sym2 := ssa.AuxToSym(v_1.Aux)
			y := v_1.Args[1]
			if y != v_1.Args[0] || !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2)) {
				continue
			}
			v.Reset(ssaop.Op386LEAL2)
			v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
			v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAL1 [off1] {sym1} x (LEAL1 [off2] {sym2} x y))
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2)
	// result: (LEAL2 [off1+off2] {ssa.MergeSym(sym1, sym2)} y x)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.Op386LEAL1 {
				continue
			}
			off2 := ssa.AuxIntToInt32(v_1.AuxInt)
			sym2 := ssa.AuxToSym(v_1.Aux)
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_1_0, v_1_1 = _i1+1, v_1_1, v_1_0 {
				if x != v_1_0 {
					continue
				}
				y := v_1_1
				if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2)) {
					continue
				}
				v.Reset(ssaop.Op386LEAL2)
				v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
				v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
				v.AddArg2(y, x)
				return true
			}
		}
		break
	}
	// match: (LEAL1 [0] {nil} x y)
	// result: (ADDL x y)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 || ssa.AuxToSym(v.Aux) != nil {
			break
		}
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386ADDL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_Op386LEAL2(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LEAL2 [c] {s} (ADDLconst [d] x) y)
	// cond: ssa.Is32Bit(int64(c)+int64(d)) && x.Op != ssaop.OpSB
	// result: (LEAL2 [c+d] {s} x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		s := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		y := v_1
		if !(ssa.Is32Bit(int64(c)+int64(d)) && x.Op != ssaop.OpSB) {
			break
		}
		v.Reset(ssaop.Op386LEAL2)
		v.AuxInt = ssa.Int32ToAuxInt(c + d)
		v.Aux = ssa.SymToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL2 [c] {s} x (ADDLconst [d] y))
	// cond: ssa.Is32Bit(int64(c)+2*int64(d)) && y.Op != ssaop.OpSB
	// result: (LEAL2 [c+2*d] {s} x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		s := ssa.AuxToSym(v.Aux)
		x := v_0
		if v_1.Op != ssaop.Op386ADDLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(ssa.Is32Bit(int64(c)+2*int64(d)) && y.Op != ssaop.OpSB) {
			break
		}
		v.Reset(ssaop.Op386LEAL2)
		v.AuxInt = ssa.Int32ToAuxInt(c + 2*d)
		v.Aux = ssa.SymToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL2 [c] {s} x (SHLLconst [1] y))
	// result: (LEAL4 [c] {s} x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		s := ssa.AuxToSym(v.Aux)
		x := v_0
		if v_1.Op != ssaop.Op386SHLLconst || ssa.AuxIntToInt32(v_1.AuxInt) != 1 {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.Op386LEAL4)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		v.Aux = ssa.SymToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL2 [c] {s} x (SHLLconst [2] y))
	// result: (LEAL8 [c] {s} x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		s := ssa.AuxToSym(v.Aux)
		x := v_0
		if v_1.Op != ssaop.Op386SHLLconst || ssa.AuxIntToInt32(v_1.AuxInt) != 2 {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.Op386LEAL8)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		v.Aux = ssa.SymToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL2 [off1] {sym1} (LEAL [off2] {sym2} x) y)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && x.Op != ssaop.OpSB
	// result: (LEAL2 [off1+off2] {ssa.MergeSym(sym1,sym2)} x y)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		x := v_0.Args[0]
		y := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && x.Op != ssaop.OpSB) {
			break
		}
		v.Reset(ssaop.Op386LEAL2)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL2 [off1] {sym} x (LEAL1 [off2] {nil} y y))
	// cond: ssa.Is32Bit(int64(off1)+2*int64(off2))
	// result: (LEAL4 [off1+2*off2] {sym} x y)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		x := v_0
		if v_1.Op != ssaop.Op386LEAL1 {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		if ssa.AuxToSym(v_1.Aux) != nil {
			break
		}
		y := v_1.Args[1]
		if y != v_1.Args[0] || !(ssa.Is32Bit(int64(off1) + 2*int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386LEAL4)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + 2*off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_Op386LEAL4(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LEAL4 [c] {s} (ADDLconst [d] x) y)
	// cond: ssa.Is32Bit(int64(c)+int64(d)) && x.Op != ssaop.OpSB
	// result: (LEAL4 [c+d] {s} x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		s := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		y := v_1
		if !(ssa.Is32Bit(int64(c)+int64(d)) && x.Op != ssaop.OpSB) {
			break
		}
		v.Reset(ssaop.Op386LEAL4)
		v.AuxInt = ssa.Int32ToAuxInt(c + d)
		v.Aux = ssa.SymToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL4 [c] {s} x (ADDLconst [d] y))
	// cond: ssa.Is32Bit(int64(c)+4*int64(d)) && y.Op != ssaop.OpSB
	// result: (LEAL4 [c+4*d] {s} x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		s := ssa.AuxToSym(v.Aux)
		x := v_0
		if v_1.Op != ssaop.Op386ADDLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(ssa.Is32Bit(int64(c)+4*int64(d)) && y.Op != ssaop.OpSB) {
			break
		}
		v.Reset(ssaop.Op386LEAL4)
		v.AuxInt = ssa.Int32ToAuxInt(c + 4*d)
		v.Aux = ssa.SymToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL4 [c] {s} x (SHLLconst [1] y))
	// result: (LEAL8 [c] {s} x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		s := ssa.AuxToSym(v.Aux)
		x := v_0
		if v_1.Op != ssaop.Op386SHLLconst || ssa.AuxIntToInt32(v_1.AuxInt) != 1 {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.Op386LEAL8)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		v.Aux = ssa.SymToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL4 [off1] {sym1} (LEAL [off2] {sym2} x) y)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && x.Op != ssaop.OpSB
	// result: (LEAL4 [off1+off2] {ssa.MergeSym(sym1,sym2)} x y)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		x := v_0.Args[0]
		y := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && x.Op != ssaop.OpSB) {
			break
		}
		v.Reset(ssaop.Op386LEAL4)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL4 [off1] {sym} x (LEAL1 [off2] {nil} y y))
	// cond: ssa.Is32Bit(int64(off1)+4*int64(off2))
	// result: (LEAL8 [off1+4*off2] {sym} x y)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		x := v_0
		if v_1.Op != ssaop.Op386LEAL1 {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		if ssa.AuxToSym(v_1.Aux) != nil {
			break
		}
		y := v_1.Args[1]
		if y != v_1.Args[0] || !(ssa.Is32Bit(int64(off1) + 4*int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386LEAL8)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + 4*off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_Op386LEAL8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LEAL8 [c] {s} (ADDLconst [d] x) y)
	// cond: ssa.Is32Bit(int64(c)+int64(d)) && x.Op != ssaop.OpSB
	// result: (LEAL8 [c+d] {s} x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		s := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		y := v_1
		if !(ssa.Is32Bit(int64(c)+int64(d)) && x.Op != ssaop.OpSB) {
			break
		}
		v.Reset(ssaop.Op386LEAL8)
		v.AuxInt = ssa.Int32ToAuxInt(c + d)
		v.Aux = ssa.SymToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL8 [c] {s} x (ADDLconst [d] y))
	// cond: ssa.Is32Bit(int64(c)+8*int64(d)) && y.Op != ssaop.OpSB
	// result: (LEAL8 [c+8*d] {s} x y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		s := ssa.AuxToSym(v.Aux)
		x := v_0
		if v_1.Op != ssaop.Op386ADDLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(ssa.Is32Bit(int64(c)+8*int64(d)) && y.Op != ssaop.OpSB) {
			break
		}
		v.Reset(ssaop.Op386LEAL8)
		v.AuxInt = ssa.Int32ToAuxInt(c + 8*d)
		v.Aux = ssa.SymToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL8 [off1] {sym1} (LEAL [off2] {sym2} x) y)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && x.Op != ssaop.OpSB
	// result: (LEAL8 [off1+off2] {ssa.MergeSym(sym1,sym2)} x y)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		x := v_0.Args[0]
		y := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && x.Op != ssaop.OpSB) {
			break
		}
		v.Reset(ssaop.Op386LEAL8)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_Op386LoweredPanicBoundsRC(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRC [kind] {p} (MOVLconst [c]) mem)
	// result: (LoweredPanicBoundsCC [kind] {ssa.PanicBoundsCC{Cx:int64(c), Cy:p.C}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		p := ssa.AuxToPanicBoundsC(v.Aux)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		mem := v_1
		v.Reset(ssaop.Op386LoweredPanicBoundsCC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCCToAux(ssa.PanicBoundsCC{Cx: int64(c), Cy: p.C})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValue386_Op386LoweredPanicBoundsRR(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRR [kind] x (MOVLconst [c]) mem)
	// result: (LoweredPanicBoundsRC [kind] x {ssa.PanicBoundsC{C:int64(c)}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.Op386LoweredPanicBoundsRC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCToAux(ssa.PanicBoundsC{C: int64(c)})
		v.AddArg2(x, mem)
		return true
	}
	// match: (LoweredPanicBoundsRR [kind] (MOVLconst [c]) y mem)
	// result: (LoweredPanicBoundsCR [kind] {ssa.PanicBoundsC{C:int64(c)}} y mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		y := v_1
		mem := v_2
		v.Reset(ssaop.Op386LoweredPanicBoundsCR)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCToAux(ssa.PanicBoundsC{C: int64(c)})
		v.AddArg2(y, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386LoweredPanicExtendRC(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicExtendRC [kind] {p} (MOVLconst [hi]) (MOVLconst [lo]) mem)
	// result: (LoweredPanicBoundsCC [kind] {ssa.PanicBoundsCC{Cx:int64(hi)<<32+int64(uint32(lo)), Cy:p.C}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		p := ssa.AuxToPanicBoundsC(v.Aux)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		hi := ssa.AuxIntToInt32(v_0.AuxInt)
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		lo := ssa.AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.Op386LoweredPanicBoundsCC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCCToAux(ssa.PanicBoundsCC{Cx: int64(hi)<<32 + int64(uint32(lo)), Cy: p.C})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValue386_Op386LoweredPanicExtendRR(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicExtendRR [kind] hi lo (MOVLconst [c]) mem)
	// result: (LoweredPanicExtendRC [kind] hi lo {ssa.PanicBoundsC{C:int64(c)}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		hi := v_0
		lo := v_1
		if v_2.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_2.AuxInt)
		mem := v_3
		v.Reset(ssaop.Op386LoweredPanicExtendRC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCToAux(ssa.PanicBoundsC{C: int64(c)})
		v.AddArg3(hi, lo, mem)
		return true
	}
	// match: (LoweredPanicExtendRR [kind] (MOVLconst [hi]) (MOVLconst [lo]) y mem)
	// result: (LoweredPanicBoundsCR [kind] {ssa.PanicBoundsC{C:int64(hi)<<32 + int64(uint32(lo))}} y mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		hi := ssa.AuxIntToInt32(v_0.AuxInt)
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		lo := ssa.AuxIntToInt32(v_1.AuxInt)
		y := v_2
		mem := v_3
		v.Reset(ssaop.Op386LoweredPanicBoundsCR)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCToAux(ssa.PanicBoundsC{C: int64(hi)<<32 + int64(uint32(lo))})
		v.AddArg2(y, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVBLSX(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVBLSX x:(MOVBload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && ssa.Clobber(x)
	// result: @x.Block (MOVBLSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != ssaop.Op386MOVBload {
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
		v0 := b.NewValue0(x.Pos, ssaop.Op386MOVBLSXload, v.Type)
		v.CopyOf(v0)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBLSX (ANDLconst [c] x))
	// cond: c & 0x80 == 0
	// result: (ANDLconst [c & 0x7f] x)
	for {
		if v_0.Op != ssaop.Op386ANDLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c&0x80 == 0) {
			break
		}
		v.Reset(ssaop.Op386ANDLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & 0x7f)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVBLSXload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBLSXload [off] {sym} ptr (MOVBstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: (MOVBLSX x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.Op386MOVBstore {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.Reset(ssaop.Op386MOVBLSX)
		v.AddArg(x)
		return true
	}
	// match: (MOVBLSXload [off1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVBLSXload [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386MOVBLSXload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVBLSXload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVLconst [int32(int8(ssa.Read8(sym, int64(off))))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(int8(ssa.Read8(sym, int64(off)))))
		return true
	}
	return false
}
func rewriteValue386_Op386MOVBLZX(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVBLZX x:(MOVBload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && ssa.Clobber(x)
	// result: @x.Block (MOVBload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != ssaop.Op386MOVBload {
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
		v0 := b.NewValue0(x.Pos, ssaop.Op386MOVBload, v.Type)
		v.CopyOf(v0)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBLZX (ANDLconst [c] x))
	// result: (ANDLconst [c & 0xff] x)
	for {
		if v_0.Op != ssaop.Op386ANDLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.Op386ANDLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & 0xff)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVBload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBload [off] {sym} ptr (MOVBstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: (MOVBLZX x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.Op386MOVBstore {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.Reset(ssaop.Op386MOVBLZX)
		v.AddArg(x)
		return true
	}
	// match: (MOVBload [off1] {sym} (ADDLconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (MOVBload [off1+off2] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386MOVBload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVBload [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386MOVBload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVBload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVLconst [int32(ssa.Read8(sym, int64(off)))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(ssa.Read8(sym, int64(off))))
		return true
	}
	return false
}
func rewriteValue386_Op386MOVBstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBstore [off] {sym} ptr (MOVBLSX x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.Op386MOVBLSX {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.Op386MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBLZX x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.Op386MOVBLZX {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.Op386MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off1] {sym} (ADDLconst [off2] ptr) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (MOVBstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVLconst [c]) mem)
	// result: (MOVBstoreconst [ssa.MakeValAndOff(c,off)] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.Op386MOVBstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(c, off))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstore [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVBstore [off1+off2] {ssa.MergeSym(sym1,sym2)} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVBstoreconst(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBstoreconst [sc] {s} (ADDLconst [off] ptr) mem)
	// cond: sc.CanAdd32(off)
	// result: (MOVBstoreconst [sc.AddOffset32(off)] {s} ptr mem)
	for {
		sc := ssa.AuxIntToValAndOff(v.AuxInt)
		s := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off := ssa.AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(sc.CanAdd32(off)) {
			break
		}
		v.Reset(ssaop.Op386MOVBstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(sc.AddOffset32(off))
		v.Aux = ssa.SymToAux(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstoreconst [sc] {sym1} (LEAL [off] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1, sym2) && sc.CanAdd32(off) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVBstoreconst [sc.AddOffset32(off)] {ssa.MergeSym(sym1, sym2)} ptr mem)
	for {
		sc := ssa.AuxIntToValAndOff(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && sc.CanAdd32(off) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386MOVBstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(sc.AddOffset32(off))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVLload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVLload [off] {sym} ptr (MOVLstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: x
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.Op386MOVLstore {
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
	// match: (MOVLload [off1] {sym} (ADDLconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (MOVLload [off1+off2] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386MOVLload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLload [off1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVLload [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386MOVLload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVLload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVLconst [int32(ssa.Read32(sym, int64(off), config.Ctxt.Arch.ByteOrder))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(ssa.Read32(sym, int64(off), config.Ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValue386_Op386MOVLstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVLstore [off1] {sym} (ADDLconst [off2] ptr) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (MOVLstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386MOVLstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVLstore [off] {sym} ptr (MOVLconst [c]) mem)
	// result: (MOVLstoreconst [ssa.MakeValAndOff(c,off)] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.Op386MOVLstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(c, off))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLstore [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVLstore [off1+off2] {ssa.MergeSym(sym1,sym2)} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386MOVLstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVLstore {sym} [off] ptr y:(ADDLload x [off] {sym} ptr mem) mem)
	// cond: y.Uses==1 && ssa.Clobber(y)
	// result: (ADDLmodify [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != ssaop.Op386ADDLload || ssa.AuxIntToInt32(y.AuxInt) != off || ssa.AuxToSym(y.Aux) != sym {
			break
		}
		mem := y.Args[2]
		x := y.Args[0]
		if ptr != y.Args[1] || mem != v_2 || !(y.Uses == 1 && ssa.Clobber(y)) {
			break
		}
		v.Reset(ssaop.Op386ADDLmodify)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVLstore {sym} [off] ptr y:(ANDLload x [off] {sym} ptr mem) mem)
	// cond: y.Uses==1 && ssa.Clobber(y)
	// result: (ANDLmodify [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != ssaop.Op386ANDLload || ssa.AuxIntToInt32(y.AuxInt) != off || ssa.AuxToSym(y.Aux) != sym {
			break
		}
		mem := y.Args[2]
		x := y.Args[0]
		if ptr != y.Args[1] || mem != v_2 || !(y.Uses == 1 && ssa.Clobber(y)) {
			break
		}
		v.Reset(ssaop.Op386ANDLmodify)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVLstore {sym} [off] ptr y:(ORLload x [off] {sym} ptr mem) mem)
	// cond: y.Uses==1 && ssa.Clobber(y)
	// result: (ORLmodify [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != ssaop.Op386ORLload || ssa.AuxIntToInt32(y.AuxInt) != off || ssa.AuxToSym(y.Aux) != sym {
			break
		}
		mem := y.Args[2]
		x := y.Args[0]
		if ptr != y.Args[1] || mem != v_2 || !(y.Uses == 1 && ssa.Clobber(y)) {
			break
		}
		v.Reset(ssaop.Op386ORLmodify)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVLstore {sym} [off] ptr y:(XORLload x [off] {sym} ptr mem) mem)
	// cond: y.Uses==1 && ssa.Clobber(y)
	// result: (XORLmodify [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != ssaop.Op386XORLload || ssa.AuxIntToInt32(y.AuxInt) != off || ssa.AuxToSym(y.Aux) != sym {
			break
		}
		mem := y.Args[2]
		x := y.Args[0]
		if ptr != y.Args[1] || mem != v_2 || !(y.Uses == 1 && ssa.Clobber(y)) {
			break
		}
		v.Reset(ssaop.Op386XORLmodify)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVLstore {sym} [off] ptr y:(ADDL l:(MOVLload [off] {sym} ptr mem) x) mem)
	// cond: y.Uses==1 && l.Uses==1 && ssa.Clobber(y, l)
	// result: (ADDLmodify [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != ssaop.Op386ADDL {
			break
		}
		_ = y.Args[1]
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			l := y_0
			if l.Op != ssaop.Op386MOVLload || ssa.AuxIntToInt32(l.AuxInt) != off || ssa.AuxToSym(l.Aux) != sym {
				continue
			}
			mem := l.Args[1]
			if ptr != l.Args[0] {
				continue
			}
			x := y_1
			if mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && ssa.Clobber(y, l)) {
				continue
			}
			v.Reset(ssaop.Op386ADDLmodify)
			v.AuxInt = ssa.Int32ToAuxInt(off)
			v.Aux = ssa.SymToAux(sym)
			v.AddArg3(ptr, x, mem)
			return true
		}
		break
	}
	// match: (MOVLstore {sym} [off] ptr y:(SUBL l:(MOVLload [off] {sym} ptr mem) x) mem)
	// cond: y.Uses==1 && l.Uses==1 && ssa.Clobber(y, l)
	// result: (SUBLmodify [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != ssaop.Op386SUBL {
			break
		}
		x := y.Args[1]
		l := y.Args[0]
		if l.Op != ssaop.Op386MOVLload || ssa.AuxIntToInt32(l.AuxInt) != off || ssa.AuxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		if ptr != l.Args[0] || mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && ssa.Clobber(y, l)) {
			break
		}
		v.Reset(ssaop.Op386SUBLmodify)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVLstore {sym} [off] ptr y:(ANDL l:(MOVLload [off] {sym} ptr mem) x) mem)
	// cond: y.Uses==1 && l.Uses==1 && ssa.Clobber(y, l)
	// result: (ANDLmodify [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != ssaop.Op386ANDL {
			break
		}
		_ = y.Args[1]
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			l := y_0
			if l.Op != ssaop.Op386MOVLload || ssa.AuxIntToInt32(l.AuxInt) != off || ssa.AuxToSym(l.Aux) != sym {
				continue
			}
			mem := l.Args[1]
			if ptr != l.Args[0] {
				continue
			}
			x := y_1
			if mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && ssa.Clobber(y, l)) {
				continue
			}
			v.Reset(ssaop.Op386ANDLmodify)
			v.AuxInt = ssa.Int32ToAuxInt(off)
			v.Aux = ssa.SymToAux(sym)
			v.AddArg3(ptr, x, mem)
			return true
		}
		break
	}
	// match: (MOVLstore {sym} [off] ptr y:(ORL l:(MOVLload [off] {sym} ptr mem) x) mem)
	// cond: y.Uses==1 && l.Uses==1 && ssa.Clobber(y, l)
	// result: (ORLmodify [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != ssaop.Op386ORL {
			break
		}
		_ = y.Args[1]
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			l := y_0
			if l.Op != ssaop.Op386MOVLload || ssa.AuxIntToInt32(l.AuxInt) != off || ssa.AuxToSym(l.Aux) != sym {
				continue
			}
			mem := l.Args[1]
			if ptr != l.Args[0] {
				continue
			}
			x := y_1
			if mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && ssa.Clobber(y, l)) {
				continue
			}
			v.Reset(ssaop.Op386ORLmodify)
			v.AuxInt = ssa.Int32ToAuxInt(off)
			v.Aux = ssa.SymToAux(sym)
			v.AddArg3(ptr, x, mem)
			return true
		}
		break
	}
	// match: (MOVLstore {sym} [off] ptr y:(XORL l:(MOVLload [off] {sym} ptr mem) x) mem)
	// cond: y.Uses==1 && l.Uses==1 && ssa.Clobber(y, l)
	// result: (XORLmodify [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != ssaop.Op386XORL {
			break
		}
		_ = y.Args[1]
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			l := y_0
			if l.Op != ssaop.Op386MOVLload || ssa.AuxIntToInt32(l.AuxInt) != off || ssa.AuxToSym(l.Aux) != sym {
				continue
			}
			mem := l.Args[1]
			if ptr != l.Args[0] {
				continue
			}
			x := y_1
			if mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && ssa.Clobber(y, l)) {
				continue
			}
			v.Reset(ssaop.Op386XORLmodify)
			v.AuxInt = ssa.Int32ToAuxInt(off)
			v.Aux = ssa.SymToAux(sym)
			v.AddArg3(ptr, x, mem)
			return true
		}
		break
	}
	// match: (MOVLstore {sym} [off] ptr y:(ADDLconst [c] l:(MOVLload [off] {sym} ptr mem)) mem)
	// cond: y.Uses==1 && l.Uses==1 && ssa.Clobber(y, l)
	// result: (ADDLconstmodify [ssa.MakeValAndOff(c,off)] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != ssaop.Op386ADDLconst {
			break
		}
		c := ssa.AuxIntToInt32(y.AuxInt)
		l := y.Args[0]
		if l.Op != ssaop.Op386MOVLload || ssa.AuxIntToInt32(l.AuxInt) != off || ssa.AuxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		if ptr != l.Args[0] || mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && ssa.Clobber(y, l)) {
			break
		}
		v.Reset(ssaop.Op386ADDLconstmodify)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(c, off))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLstore {sym} [off] ptr y:(ANDLconst [c] l:(MOVLload [off] {sym} ptr mem)) mem)
	// cond: y.Uses==1 && l.Uses==1 && ssa.Clobber(y, l)
	// result: (ANDLconstmodify [ssa.MakeValAndOff(c,off)] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != ssaop.Op386ANDLconst {
			break
		}
		c := ssa.AuxIntToInt32(y.AuxInt)
		l := y.Args[0]
		if l.Op != ssaop.Op386MOVLload || ssa.AuxIntToInt32(l.AuxInt) != off || ssa.AuxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		if ptr != l.Args[0] || mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && ssa.Clobber(y, l)) {
			break
		}
		v.Reset(ssaop.Op386ANDLconstmodify)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(c, off))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLstore {sym} [off] ptr y:(ORLconst [c] l:(MOVLload [off] {sym} ptr mem)) mem)
	// cond: y.Uses==1 && l.Uses==1 && ssa.Clobber(y, l)
	// result: (ORLconstmodify [ssa.MakeValAndOff(c,off)] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != ssaop.Op386ORLconst {
			break
		}
		c := ssa.AuxIntToInt32(y.AuxInt)
		l := y.Args[0]
		if l.Op != ssaop.Op386MOVLload || ssa.AuxIntToInt32(l.AuxInt) != off || ssa.AuxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		if ptr != l.Args[0] || mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && ssa.Clobber(y, l)) {
			break
		}
		v.Reset(ssaop.Op386ORLconstmodify)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(c, off))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLstore {sym} [off] ptr y:(XORLconst [c] l:(MOVLload [off] {sym} ptr mem)) mem)
	// cond: y.Uses==1 && l.Uses==1 && ssa.Clobber(y, l)
	// result: (XORLconstmodify [ssa.MakeValAndOff(c,off)] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != ssaop.Op386XORLconst {
			break
		}
		c := ssa.AuxIntToInt32(y.AuxInt)
		l := y.Args[0]
		if l.Op != ssaop.Op386MOVLload || ssa.AuxIntToInt32(l.AuxInt) != off || ssa.AuxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		if ptr != l.Args[0] || mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && ssa.Clobber(y, l)) {
			break
		}
		v.Reset(ssaop.Op386XORLconstmodify)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(c, off))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVLstoreconst(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVLstoreconst [sc] {s} (ADDLconst [off] ptr) mem)
	// cond: sc.CanAdd32(off)
	// result: (MOVLstoreconst [sc.AddOffset32(off)] {s} ptr mem)
	for {
		sc := ssa.AuxIntToValAndOff(v.AuxInt)
		s := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off := ssa.AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(sc.CanAdd32(off)) {
			break
		}
		v.Reset(ssaop.Op386MOVLstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(sc.AddOffset32(off))
		v.Aux = ssa.SymToAux(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLstoreconst [sc] {sym1} (LEAL [off] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1, sym2) && sc.CanAdd32(off) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVLstoreconst [sc.AddOffset32(off)] {ssa.MergeSym(sym1, sym2)} ptr mem)
	for {
		sc := ssa.AuxIntToValAndOff(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && sc.CanAdd32(off) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386MOVLstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(sc.AddOffset32(off))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVSDconst(v *ssa.Value) bool {
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVSDconst [c])
	// cond: config.Ctxt.Flag_shared
	// result: (MOVSDconst2 (MOVSDconst1 [c]))
	for {
		c := ssa.AuxIntToFloat64(v.AuxInt)
		if !(config.Ctxt.Flag_shared) {
			break
		}
		v.Reset(ssaop.Op386MOVSDconst2)
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVSDconst1, typ.UInt32)
		v0.AuxInt = ssa.Float64ToAuxInt(c)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVSDload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVSDload [off1] {sym} (ADDLconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (MOVSDload [off1+off2] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386MOVSDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVSDload [off1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVSDload [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386MOVSDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVSDstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVSDstore [off1] {sym} (ADDLconst [off2] ptr) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (MOVSDstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386MOVSDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVSDstore [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVSDstore [off1+off2] {ssa.MergeSym(sym1,sym2)} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386MOVSDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVSSconst(v *ssa.Value) bool {
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVSSconst [c])
	// cond: config.Ctxt.Flag_shared
	// result: (MOVSSconst2 (MOVSSconst1 [c]))
	for {
		c := ssa.AuxIntToFloat32(v.AuxInt)
		if !(config.Ctxt.Flag_shared) {
			break
		}
		v.Reset(ssaop.Op386MOVSSconst2)
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVSSconst1, typ.UInt32)
		v0.AuxInt = ssa.Float32ToAuxInt(c)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVSSload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVSSload [off1] {sym} (ADDLconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (MOVSSload [off1+off2] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386MOVSSload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVSSload [off1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVSSload [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386MOVSSload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVSSstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVSSstore [off1] {sym} (ADDLconst [off2] ptr) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (MOVSSstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386MOVSSstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVSSstore [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVSSstore [off1+off2] {ssa.MergeSym(sym1,sym2)} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386MOVSSstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVWLSX(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVWLSX x:(MOVWload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && ssa.Clobber(x)
	// result: @x.Block (MOVWLSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != ssaop.Op386MOVWload {
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
		v0 := b.NewValue0(x.Pos, ssaop.Op386MOVWLSXload, v.Type)
		v.CopyOf(v0)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWLSX (ANDLconst [c] x))
	// cond: c & 0x8000 == 0
	// result: (ANDLconst [c & 0x7fff] x)
	for {
		if v_0.Op != ssaop.Op386ANDLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c&0x8000 == 0) {
			break
		}
		v.Reset(ssaop.Op386ANDLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & 0x7fff)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVWLSXload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWLSXload [off] {sym} ptr (MOVWstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: (MOVWLSX x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.Op386MOVWstore {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.Reset(ssaop.Op386MOVWLSX)
		v.AddArg(x)
		return true
	}
	// match: (MOVWLSXload [off1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVWLSXload [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386MOVWLSXload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVWLSXload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVLconst [int32(int16(ssa.Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder)))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(int16(ssa.Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder))))
		return true
	}
	return false
}
func rewriteValue386_Op386MOVWLZX(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVWLZX x:(MOVWload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && ssa.Clobber(x)
	// result: @x.Block (MOVWload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != ssaop.Op386MOVWload {
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
		v0 := b.NewValue0(x.Pos, ssaop.Op386MOVWload, v.Type)
		v.CopyOf(v0)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWLZX (ANDLconst [c] x))
	// result: (ANDLconst [c & 0xffff] x)
	for {
		if v_0.Op != ssaop.Op386ANDLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.Op386ANDLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & 0xffff)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVWload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWload [off] {sym} ptr (MOVWstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: (MOVWLZX x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.Op386MOVWstore {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.Reset(ssaop.Op386MOVWLZX)
		v.AddArg(x)
		return true
	}
	// match: (MOVWload [off1] {sym} (ADDLconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (MOVWload [off1+off2] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386MOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVWload [off1+off2] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386MOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVWload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVLconst [int32(ssa.Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(ssa.Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValue386_Op386MOVWstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWstore [off] {sym} ptr (MOVWLSX x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.Op386MOVWLSX {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.Op386MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWLZX x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.Op386MOVWLZX {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.Op386MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym} (ADDLconst [off2] ptr) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (MOVWstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVLconst [c]) mem)
	// result: (MOVWstoreconst [ssa.MakeValAndOff(c,off)] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.Op386MOVWstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(c, off))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVWstore [off1+off2] {ssa.MergeSym(sym1,sym2)} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MOVWstoreconst(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWstoreconst [sc] {s} (ADDLconst [off] ptr) mem)
	// cond: sc.CanAdd32(off)
	// result: (MOVWstoreconst [sc.AddOffset32(off)] {s} ptr mem)
	for {
		sc := ssa.AuxIntToValAndOff(v.AuxInt)
		s := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off := ssa.AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(sc.CanAdd32(off)) {
			break
		}
		v.Reset(ssaop.Op386MOVWstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(sc.AddOffset32(off))
		v.Aux = ssa.SymToAux(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstoreconst [sc] {sym1} (LEAL [off] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1, sym2) && sc.CanAdd32(off) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVWstoreconst [sc.AddOffset32(off)] {ssa.MergeSym(sym1, sym2)} ptr mem)
	for {
		sc := ssa.AuxIntToValAndOff(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && sc.CanAdd32(off) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386MOVWstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(sc.AddOffset32(off))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MULL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MULL x (MOVLconst [c]))
	// result: (MULLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.Op386MOVLconst {
				continue
			}
			c := ssa.AuxIntToInt32(v_1.AuxInt)
			v.Reset(ssaop.Op386MULLconst)
			v.AuxInt = ssa.Int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (MULL x l:(MOVLload [off] {sym} ptr mem))
	// cond: ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)
	// result: (MULLload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != ssaop.Op386MOVLload {
				continue
			}
			off := ssa.AuxIntToInt32(l.AuxInt)
			sym := ssa.AuxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)) {
				continue
			}
			v.Reset(ssaop.Op386MULLload)
			v.AuxInt = ssa.Int32ToAuxInt(off)
			v.Aux = ssa.SymToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValue386_Op386MULLconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MULLconst [c] (MULLconst [d] x))
	// result: (MULLconst [c * d] x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386MULLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.Op386MULLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c * d)
		v.AddArg(x)
		return true
	}
	// match: (MULLconst [-9] x)
	// result: (NEGL (LEAL8 <v.Type> x x))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != -9 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386NEGL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386LEAL8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULLconst [-5] x)
	// result: (NEGL (LEAL4 <v.Type> x x))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != -5 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386NEGL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386LEAL4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULLconst [-3] x)
	// result: (NEGL (LEAL2 <v.Type> x x))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != -3 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386NEGL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386LEAL2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULLconst [-1] x)
	// result: (NEGL x)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != -1 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386NEGL)
		v.AddArg(x)
		return true
	}
	// match: (MULLconst [0] _)
	// result: (MOVLconst [0])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (MULLconst [1] x)
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 1 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (MULLconst [3] x)
	// result: (LEAL2 x x)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 3 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386LEAL2)
		v.AddArg2(x, x)
		return true
	}
	// match: (MULLconst [5] x)
	// result: (LEAL4 x x)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 5 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386LEAL4)
		v.AddArg2(x, x)
		return true
	}
	// match: (MULLconst [7] x)
	// result: (LEAL2 x (LEAL2 <v.Type> x x))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 7 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386LEAL2)
		v0 := b.NewValue0(v.Pos, ssaop.Op386LEAL2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [9] x)
	// result: (LEAL8 x x)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 9 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386LEAL8)
		v.AddArg2(x, x)
		return true
	}
	// match: (MULLconst [11] x)
	// result: (LEAL2 x (LEAL4 <v.Type> x x))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 11 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386LEAL2)
		v0 := b.NewValue0(v.Pos, ssaop.Op386LEAL4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [13] x)
	// result: (LEAL4 x (LEAL2 <v.Type> x x))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 13 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386LEAL4)
		v0 := b.NewValue0(v.Pos, ssaop.Op386LEAL2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [19] x)
	// result: (LEAL2 x (LEAL8 <v.Type> x x))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 19 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386LEAL2)
		v0 := b.NewValue0(v.Pos, ssaop.Op386LEAL8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [21] x)
	// result: (LEAL4 x (LEAL4 <v.Type> x x))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 21 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386LEAL4)
		v0 := b.NewValue0(v.Pos, ssaop.Op386LEAL4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [25] x)
	// result: (LEAL8 x (LEAL2 <v.Type> x x))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 25 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386LEAL8)
		v0 := b.NewValue0(v.Pos, ssaop.Op386LEAL2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [27] x)
	// result: (LEAL8 (LEAL2 <v.Type> x x) (LEAL2 <v.Type> x x))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 27 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386LEAL8)
		v0 := b.NewValue0(v.Pos, ssaop.Op386LEAL2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(v0, v0)
		return true
	}
	// match: (MULLconst [37] x)
	// result: (LEAL4 x (LEAL8 <v.Type> x x))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 37 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386LEAL4)
		v0 := b.NewValue0(v.Pos, ssaop.Op386LEAL8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [41] x)
	// result: (LEAL8 x (LEAL4 <v.Type> x x))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 41 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386LEAL8)
		v0 := b.NewValue0(v.Pos, ssaop.Op386LEAL4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [45] x)
	// result: (LEAL8 (LEAL4 <v.Type> x x) (LEAL4 <v.Type> x x))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 45 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386LEAL8)
		v0 := b.NewValue0(v.Pos, ssaop.Op386LEAL4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(v0, v0)
		return true
	}
	// match: (MULLconst [73] x)
	// result: (LEAL8 x (LEAL8 <v.Type> x x))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 73 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386LEAL8)
		v0 := b.NewValue0(v.Pos, ssaop.Op386LEAL8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [81] x)
	// result: (LEAL8 (LEAL8 <v.Type> x x) (LEAL8 <v.Type> x x))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 81 {
			break
		}
		x := v_0
		v.Reset(ssaop.Op386LEAL8)
		v0 := b.NewValue0(v.Pos, ssaop.Op386LEAL8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(v0, v0)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: ssa.IsPowerOfTwo(c+1) && c >= 15
	// result: (SUBL (SHLLconst <v.Type> [int32(ssa.Log32(c+1))] x) x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(ssa.IsPowerOfTwo(c+1) && c >= 15) {
			break
		}
		v.Reset(ssaop.Op386SUBL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHLLconst, v.Type)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(ssa.Log32(c + 1)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: ssa.IsPowerOfTwo(c-1) && c >= 17
	// result: (LEAL1 (SHLLconst <v.Type> [int32(ssa.Log32(c-1))] x) x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(ssa.IsPowerOfTwo(c-1) && c >= 17) {
			break
		}
		v.Reset(ssaop.Op386LEAL1)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHLLconst, v.Type)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(ssa.Log32(c - 1)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: ssa.IsPowerOfTwo(c-2) && c >= 34
	// result: (LEAL2 (SHLLconst <v.Type> [int32(ssa.Log32(c-2))] x) x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(ssa.IsPowerOfTwo(c-2) && c >= 34) {
			break
		}
		v.Reset(ssaop.Op386LEAL2)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHLLconst, v.Type)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(ssa.Log32(c - 2)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: ssa.IsPowerOfTwo(c-4) && c >= 68
	// result: (LEAL4 (SHLLconst <v.Type> [int32(ssa.Log32(c-4))] x) x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(ssa.IsPowerOfTwo(c-4) && c >= 68) {
			break
		}
		v.Reset(ssaop.Op386LEAL4)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHLLconst, v.Type)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(ssa.Log32(c - 4)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: ssa.IsPowerOfTwo(c-8) && c >= 136
	// result: (LEAL8 (SHLLconst <v.Type> [int32(ssa.Log32(c-8))] x) x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(ssa.IsPowerOfTwo(c-8) && c >= 136) {
			break
		}
		v.Reset(ssaop.Op386LEAL8)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHLLconst, v.Type)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(ssa.Log32(c - 8)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: c%3 == 0 && ssa.IsPowerOfTwo(c/3)
	// result: (SHLLconst [int32(ssa.Log32(c/3))] (LEAL2 <v.Type> x x))
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(c%3 == 0 && ssa.IsPowerOfTwo(c/3)) {
			break
		}
		v.Reset(ssaop.Op386SHLLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(ssa.Log32(c / 3)))
		v0 := b.NewValue0(v.Pos, ssaop.Op386LEAL2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: c%5 == 0 && ssa.IsPowerOfTwo(c/5)
	// result: (SHLLconst [int32(ssa.Log32(c/5))] (LEAL4 <v.Type> x x))
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(c%5 == 0 && ssa.IsPowerOfTwo(c/5)) {
			break
		}
		v.Reset(ssaop.Op386SHLLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(ssa.Log32(c / 5)))
		v0 := b.NewValue0(v.Pos, ssaop.Op386LEAL4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: c%9 == 0 && ssa.IsPowerOfTwo(c/9)
	// result: (SHLLconst [int32(ssa.Log32(c/9))] (LEAL8 <v.Type> x x))
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(c%9 == 0 && ssa.IsPowerOfTwo(c/9)) {
			break
		}
		v.Reset(ssaop.Op386SHLLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(ssa.Log32(c / 9)))
		v0 := b.NewValue0(v.Pos, ssaop.Op386LEAL8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULLconst [c] (MOVLconst [d]))
	// result: (MOVLconst [c*d])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c * d)
		return true
	}
	return false
}
func rewriteValue386_Op386MULLload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MULLload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (MULLload [off1+off2] {sym} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386MULLload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (MULLload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MULLload [off1+off2] {ssa.MergeSym(sym1,sym2)} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386MULLload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MULSD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MULSD x l:(MOVSDload [off] {sym} ptr mem))
	// cond: ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)
	// result: (MULSDload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != ssaop.Op386MOVSDload {
				continue
			}
			off := ssa.AuxIntToInt32(l.AuxInt)
			sym := ssa.AuxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)) {
				continue
			}
			v.Reset(ssaop.Op386MULSDload)
			v.AuxInt = ssa.Int32ToAuxInt(off)
			v.Aux = ssa.SymToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValue386_Op386MULSDload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MULSDload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (MULSDload [off1+off2] {sym} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386MULSDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (MULSDload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MULSDload [off1+off2] {ssa.MergeSym(sym1,sym2)} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386MULSDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386MULSS(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MULSS x l:(MOVSSload [off] {sym} ptr mem))
	// cond: ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)
	// result: (MULSSload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != ssaop.Op386MOVSSload {
				continue
			}
			off := ssa.AuxIntToInt32(l.AuxInt)
			sym := ssa.AuxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)) {
				continue
			}
			v.Reset(ssaop.Op386MULSSload)
			v.AuxInt = ssa.Int32ToAuxInt(off)
			v.Aux = ssa.SymToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValue386_Op386MULSSload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MULSSload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (MULSSload [off1+off2] {sym} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386MULSSload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (MULSSload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MULSSload [off1+off2] {ssa.MergeSym(sym1,sym2)} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386MULSSload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386NEGL(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (NEGL (MOVLconst [c]))
	// result: (MOVLconst [-c])
	for {
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(-c)
		return true
	}
	return false
}
func rewriteValue386_Op386NOTL(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (NOTL (MOVLconst [c]))
	// result: (MOVLconst [^c])
	for {
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(^c)
		return true
	}
	return false
}
func rewriteValue386_Op386ORL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORL x (MOVLconst [c]))
	// result: (ORLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.Op386MOVLconst {
				continue
			}
			c := ssa.AuxIntToInt32(v_1.AuxInt)
			v.Reset(ssaop.Op386ORLconst)
			v.AuxInt = ssa.Int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ORL x l:(MOVLload [off] {sym} ptr mem))
	// cond: ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)
	// result: (ORLload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != ssaop.Op386MOVLload {
				continue
			}
			off := ssa.AuxIntToInt32(l.AuxInt)
			sym := ssa.AuxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)) {
				continue
			}
			v.Reset(ssaop.Op386ORLload)
			v.AuxInt = ssa.Int32ToAuxInt(off)
			v.Aux = ssa.SymToAux(sym)
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
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue386_Op386ORLconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ORLconst [c] x)
	// cond: c==0
	// result: x
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(c == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ORLconst [c] _)
	// cond: c==-1
	// result: (MOVLconst [-1])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if !(c == -1) {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(-1)
		return true
	}
	// match: (ORLconst [c] (MOVLconst [d]))
	// result: (MOVLconst [c|d])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c | d)
		return true
	}
	return false
}
func rewriteValue386_Op386ORLconstmodify(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ORLconstmodify [valoff1] {sym} (ADDLconst [off2] base) mem)
	// cond: valoff1.CanAdd32(off2)
	// result: (ORLconstmodify [valoff1.AddOffset32(off2)] {sym} base mem)
	for {
		valoff1 := ssa.AuxIntToValAndOff(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(valoff1.CanAdd32(off2)) {
			break
		}
		v.Reset(ssaop.Op386ORLconstmodify)
		v.AuxInt = ssa.ValAndOffToAuxInt(valoff1.AddOffset32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (ORLconstmodify [valoff1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: valoff1.CanAdd32(off2) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (ORLconstmodify [valoff1.AddOffset32(off2)] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := ssa.AuxIntToValAndOff(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(valoff1.CanAdd32(off2) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386ORLconstmodify)
		v.AuxInt = ssa.ValAndOffToAuxInt(valoff1.AddOffset32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ORLload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ORLload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (ORLload [off1+off2] {sym} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386ORLload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ORLload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (ORLload [off1+off2] {ssa.MergeSym(sym1,sym2)} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386ORLload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ORLmodify(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (ORLmodify [off1] {sym} (ADDLconst [off2] base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (ORLmodify [off1+off2] {sym} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386ORLmodify)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (ORLmodify [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (ORLmodify [off1+off2] {ssa.MergeSym(sym1,sym2)} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386ORLmodify)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386ROLB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROLB x (MOVLconst [c]))
	// result: (ROLBconst [int8(c&7)] x)
	for {
		x := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.Op386ROLBconst)
		v.AuxInt = ssa.Int8ToAuxInt(int8(c & 7))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386ROLBconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ROLBconst [0] x)
	// result: x
	for {
		if ssa.AuxIntToInt8(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue386_Op386ROLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROLL x (MOVLconst [c]))
	// result: (ROLLconst [c&31] x)
	for {
		x := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.Op386ROLLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & 31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386ROLLconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ROLLconst [0] x)
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue386_Op386ROLW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROLW x (MOVLconst [c]))
	// result: (ROLWconst [int16(c&15)] x)
	for {
		x := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.Op386ROLWconst)
		v.AuxInt = ssa.Int16ToAuxInt(int16(c & 15))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386ROLWconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ROLWconst [0] x)
	// result: x
	for {
		if ssa.AuxIntToInt16(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue386_Op386SARB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SARB x (MOVLconst [c]))
	// result: (SARBconst [int8(min(int64(c&31),7))] x)
	for {
		x := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.Op386SARBconst)
		v.AuxInt = ssa.Int8ToAuxInt(int8(min(int64(c&31), 7)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386SARBconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SARBconst x [0])
	// result: x
	for {
		if ssa.AuxIntToInt8(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (SARBconst [c] (MOVLconst [d]))
	// result: (MOVLconst [d>>uint64(c)])
	for {
		c := ssa.AuxIntToInt8(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(d >> uint64(c))
		return true
	}
	return false
}
func rewriteValue386_Op386SARL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SARL x (MOVLconst [c]))
	// result: (SARLconst [c&31] x)
	for {
		x := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.Op386SARLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & 31)
		v.AddArg(x)
		return true
	}
	// match: (SARL x (ANDLconst [31] y))
	// result: (SARL x y)
	for {
		x := v_0
		if v_1.Op != ssaop.Op386ANDLconst || ssa.AuxIntToInt32(v_1.AuxInt) != 31 {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.Op386SARL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_Op386SARLconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SARLconst x [0])
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (SARLconst [c] (MOVLconst [d]))
	// result: (MOVLconst [d>>uint64(c)])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(d >> uint64(c))
		return true
	}
	return false
}
func rewriteValue386_Op386SARW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SARW x (MOVLconst [c]))
	// result: (SARWconst [int16(min(int64(c&31),15))] x)
	for {
		x := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.Op386SARWconst)
		v.AuxInt = ssa.Int16ToAuxInt(int16(min(int64(c&31), 15)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386SARWconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SARWconst x [0])
	// result: x
	for {
		if ssa.AuxIntToInt16(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (SARWconst [c] (MOVLconst [d]))
	// result: (MOVLconst [d>>uint64(c)])
	for {
		c := ssa.AuxIntToInt16(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(d >> uint64(c))
		return true
	}
	return false
}
func rewriteValue386_Op386SBBL(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SBBL x (MOVLconst [c]) f)
	// result: (SBBLconst [c] x f)
	for {
		x := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		f := v_2
		v.Reset(ssaop.Op386SBBLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		v.AddArg2(x, f)
		return true
	}
	return false
}
func rewriteValue386_Op386SBBLcarrymask(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SBBLcarrymask (FlagEQ))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagEQ {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SBBLcarrymask (FlagLT_ULT))
	// result: (MOVLconst [-1])
	for {
		if v_0.Op != ssaop.Op386FlagLT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(-1)
		return true
	}
	// match: (SBBLcarrymask (FlagLT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagLT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SBBLcarrymask (FlagGT_ULT))
	// result: (MOVLconst [-1])
	for {
		if v_0.Op != ssaop.Op386FlagGT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(-1)
		return true
	}
	// match: (SBBLcarrymask (FlagGT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagGT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386SETA(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SETA (InvertFlags x))
	// result: (SETB x)
	for {
		if v_0.Op != ssaop.Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.Op386SETB)
		v.AddArg(x)
		return true
	}
	// match: (SETA (FlagEQ))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagEQ {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETA (FlagLT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagLT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETA (FlagLT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagLT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETA (FlagGT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagGT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETA (FlagGT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagGT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValue386_Op386SETAE(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SETAE (InvertFlags x))
	// result: (SETBE x)
	for {
		if v_0.Op != ssaop.Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.Op386SETBE)
		v.AddArg(x)
		return true
	}
	// match: (SETAE (FlagEQ))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagEQ {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETAE (FlagLT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagLT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETAE (FlagLT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagLT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETAE (FlagGT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagGT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETAE (FlagGT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagGT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValue386_Op386SETB(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SETB (InvertFlags x))
	// result: (SETA x)
	for {
		if v_0.Op != ssaop.Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.Op386SETA)
		v.AddArg(x)
		return true
	}
	// match: (SETB (FlagEQ))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagEQ {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETB (FlagLT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagLT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETB (FlagLT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagLT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETB (FlagGT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagGT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETB (FlagGT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagGT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386SETBE(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SETBE (InvertFlags x))
	// result: (SETAE x)
	for {
		if v_0.Op != ssaop.Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.Op386SETAE)
		v.AddArg(x)
		return true
	}
	// match: (SETBE (FlagEQ))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagEQ {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETBE (FlagLT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagLT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETBE (FlagLT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagLT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETBE (FlagGT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagGT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETBE (FlagGT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagGT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386SETEQ(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SETEQ (InvertFlags x))
	// result: (SETEQ x)
	for {
		if v_0.Op != ssaop.Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.Op386SETEQ)
		v.AddArg(x)
		return true
	}
	// match: (SETEQ (FlagEQ))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagEQ {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETEQ (FlagLT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagLT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETEQ (FlagLT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagLT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETEQ (FlagGT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagGT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETEQ (FlagGT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagGT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386SETG(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SETG (InvertFlags x))
	// result: (SETL x)
	for {
		if v_0.Op != ssaop.Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.Op386SETL)
		v.AddArg(x)
		return true
	}
	// match: (SETG (FlagEQ))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagEQ {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETG (FlagLT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagLT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETG (FlagLT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagLT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETG (FlagGT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagGT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETG (FlagGT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagGT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValue386_Op386SETGE(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SETGE (InvertFlags x))
	// result: (SETLE x)
	for {
		if v_0.Op != ssaop.Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.Op386SETLE)
		v.AddArg(x)
		return true
	}
	// match: (SETGE (FlagEQ))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagEQ {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETGE (FlagLT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagLT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETGE (FlagLT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagLT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETGE (FlagGT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagGT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETGE (FlagGT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagGT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValue386_Op386SETL(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SETL (InvertFlags x))
	// result: (SETG x)
	for {
		if v_0.Op != ssaop.Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.Op386SETG)
		v.AddArg(x)
		return true
	}
	// match: (SETL (FlagEQ))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagEQ {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETL (FlagLT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagLT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETL (FlagLT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagLT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETL (FlagGT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagGT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETL (FlagGT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagGT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386SETLE(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SETLE (InvertFlags x))
	// result: (SETGE x)
	for {
		if v_0.Op != ssaop.Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.Op386SETGE)
		v.AddArg(x)
		return true
	}
	// match: (SETLE (FlagEQ))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagEQ {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETLE (FlagLT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagLT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETLE (FlagLT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagLT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETLE (FlagGT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagGT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETLE (FlagGT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagGT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386SETNE(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SETNE (InvertFlags x))
	// result: (SETNE x)
	for {
		if v_0.Op != ssaop.Op386InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.Op386SETNE)
		v.AddArg(x)
		return true
	}
	// match: (SETNE (FlagEQ))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != ssaop.Op386FlagEQ {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	// match: (SETNE (FlagLT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagLT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETNE (FlagLT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagLT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETNE (FlagGT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagGT_ULT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	// match: (SETNE (FlagGT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != ssaop.Op386FlagGT_UGT {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValue386_Op386SHLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SHLL x (MOVLconst [c]))
	// result: (SHLLconst [c&31] x)
	for {
		x := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.Op386SHLLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & 31)
		v.AddArg(x)
		return true
	}
	// match: (SHLL x (ANDLconst [31] y))
	// result: (SHLL x y)
	for {
		x := v_0
		if v_1.Op != ssaop.Op386ANDLconst || ssa.AuxIntToInt32(v_1.AuxInt) != 31 {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.Op386SHLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_Op386SHLLconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SHLLconst x [0])
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue386_Op386SHRB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SHRB x (MOVLconst [c]))
	// cond: c&31 < 8
	// result: (SHRBconst [int8(c&31)] x)
	for {
		x := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		if !(c&31 < 8) {
			break
		}
		v.Reset(ssaop.Op386SHRBconst)
		v.AuxInt = ssa.Int8ToAuxInt(int8(c & 31))
		v.AddArg(x)
		return true
	}
	// match: (SHRB _ (MOVLconst [c]))
	// cond: c&31 >= 8
	// result: (MOVLconst [0])
	for {
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		if !(c&31 >= 8) {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386SHRBconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SHRBconst x [0])
	// result: x
	for {
		if ssa.AuxIntToInt8(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue386_Op386SHRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SHRL x (MOVLconst [c]))
	// result: (SHRLconst [c&31] x)
	for {
		x := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.Op386SHRLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c & 31)
		v.AddArg(x)
		return true
	}
	// match: (SHRL x (ANDLconst [31] y))
	// result: (SHRL x y)
	for {
		x := v_0
		if v_1.Op != ssaop.Op386ANDLconst || ssa.AuxIntToInt32(v_1.AuxInt) != 31 {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.Op386SHRL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_Op386SHRLconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SHRLconst x [0])
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue386_Op386SHRW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SHRW x (MOVLconst [c]))
	// cond: c&31 < 16
	// result: (SHRWconst [int16(c&31)] x)
	for {
		x := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		if !(c&31 < 16) {
			break
		}
		v.Reset(ssaop.Op386SHRWconst)
		v.AuxInt = ssa.Int16ToAuxInt(int16(c & 31))
		v.AddArg(x)
		return true
	}
	// match: (SHRW _ (MOVLconst [c]))
	// cond: c&31 >= 16
	// result: (MOVLconst [0])
	for {
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		if !(c&31 >= 16) {
			break
		}
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386SHRWconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SHRWconst x [0])
	// result: x
	for {
		if ssa.AuxIntToInt16(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue386_Op386SUBL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUBL x (MOVLconst [c]))
	// result: (SUBLconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.Op386SUBLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SUBL (MOVLconst [c]) x)
	// result: (NEGL (SUBLconst <v.Type> x [c]))
	for {
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.Op386NEGL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SUBLconst, v.Type)
		v0.AuxInt = ssa.Int32ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SUBL x l:(MOVLload [off] {sym} ptr mem))
	// cond: ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)
	// result: (SUBLload x [off] {sym} ptr mem)
	for {
		x := v_0
		l := v_1
		if l.Op != ssaop.Op386MOVLload {
			break
		}
		off := ssa.AuxIntToInt32(l.AuxInt)
		sym := ssa.AuxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)) {
			break
		}
		v.Reset(ssaop.Op386SUBLload)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
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
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386SUBLcarry(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBLcarry x (MOVLconst [c]))
	// result: (SUBLconstcarry [c] x)
	for {
		x := v_0
		if v_1.Op != ssaop.Op386MOVLconst {
			break
		}
		c := ssa.AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.Op386SUBLconstcarry)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_Op386SUBLconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SUBLconst [c] x)
	// cond: c==0
	// result: x
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(c == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (SUBLconst [c] x)
	// result: (ADDLconst [-c] x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		x := v_0
		v.Reset(ssaop.Op386ADDLconst)
		v.AuxInt = ssa.Int32ToAuxInt(-c)
		v.AddArg(x)
		return true
	}
}
func rewriteValue386_Op386SUBLload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (SUBLload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (SUBLload [off1+off2] {sym} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386SUBLload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (SUBLload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (SUBLload [off1+off2] {ssa.MergeSym(sym1,sym2)} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386SUBLload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386SUBLmodify(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (SUBLmodify [off1] {sym} (ADDLconst [off2] base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (SUBLmodify [off1+off2] {sym} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386SUBLmodify)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SUBLmodify [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (SUBLmodify [off1+off2] {ssa.MergeSym(sym1,sym2)} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386SUBLmodify)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386SUBSD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBSD x l:(MOVSDload [off] {sym} ptr mem))
	// cond: ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)
	// result: (SUBSDload x [off] {sym} ptr mem)
	for {
		x := v_0
		l := v_1
		if l.Op != ssaop.Op386MOVSDload {
			break
		}
		off := ssa.AuxIntToInt32(l.AuxInt)
		sym := ssa.AuxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)) {
			break
		}
		v.Reset(ssaop.Op386SUBSDload)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(x, ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386SUBSDload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (SUBSDload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (SUBSDload [off1+off2] {sym} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386SUBSDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (SUBSDload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (SUBSDload [off1+off2] {ssa.MergeSym(sym1,sym2)} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386SUBSDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386SUBSS(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBSS x l:(MOVSSload [off] {sym} ptr mem))
	// cond: ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)
	// result: (SUBSSload x [off] {sym} ptr mem)
	for {
		x := v_0
		l := v_1
		if l.Op != ssaop.Op386MOVSSload {
			break
		}
		off := ssa.AuxIntToInt32(l.AuxInt)
		sym := ssa.AuxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)) {
			break
		}
		v.Reset(ssaop.Op386SUBSSload)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(x, ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386SUBSSload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (SUBSSload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (SUBSSload [off1+off2] {sym} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386SUBSSload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (SUBSSload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (SUBSSload [off1+off2] {ssa.MergeSym(sym1,sym2)} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386SUBSSload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386XORL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XORL x (MOVLconst [c]))
	// result: (XORLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.Op386MOVLconst {
				continue
			}
			c := ssa.AuxIntToInt32(v_1.AuxInt)
			v.Reset(ssaop.Op386XORLconst)
			v.AuxInt = ssa.Int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (XORL x l:(MOVLload [off] {sym} ptr mem))
	// cond: ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)
	// result: (XORLload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != ssaop.Op386MOVLload {
				continue
			}
			off := ssa.AuxIntToInt32(l.AuxInt)
			sym := ssa.AuxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(ssa.CanMergeLoadClobber(v, l, x) && ssa.Clobber(l)) {
				continue
			}
			v.Reset(ssaop.Op386XORLload)
			v.AuxInt = ssa.Int32ToAuxInt(off)
			v.Aux = ssa.SymToAux(sym)
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
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_Op386XORLconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (XORLconst [c] (XORLconst [d] x))
	// result: (XORLconst [c ^ d] x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386XORLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.Op386XORLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c ^ d)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [c] x)
	// cond: c==0
	// result: x
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(c == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (XORLconst [c] (MOVLconst [d]))
	// result: (MOVLconst [c^d])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.Op386MOVLconst {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(c ^ d)
		return true
	}
	return false
}
func rewriteValue386_Op386XORLconstmodify(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (XORLconstmodify [valoff1] {sym} (ADDLconst [off2] base) mem)
	// cond: valoff1.CanAdd32(off2)
	// result: (XORLconstmodify [valoff1.AddOffset32(off2)] {sym} base mem)
	for {
		valoff1 := ssa.AuxIntToValAndOff(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(valoff1.CanAdd32(off2)) {
			break
		}
		v.Reset(ssaop.Op386XORLconstmodify)
		v.AuxInt = ssa.ValAndOffToAuxInt(valoff1.AddOffset32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (XORLconstmodify [valoff1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: valoff1.CanAdd32(off2) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (XORLconstmodify [valoff1.AddOffset32(off2)] {ssa.MergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := ssa.AuxIntToValAndOff(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(valoff1.CanAdd32(off2) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386XORLconstmodify)
		v.AuxInt = ssa.ValAndOffToAuxInt(valoff1.AddOffset32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386XORLload(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (XORLload [off1] {sym} val (ADDLconst [off2] base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (XORLload [off1+off2] {sym} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386XORLload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (XORLload [off1] {sym1} val (LEAL [off2] {sym2} base) mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (XORLload [off1+off2] {ssa.MergeSym(sym1,sym2)} val base mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		val := v_0
		if v_1.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_1.AuxInt)
		sym2 := ssa.AuxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386XORLload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValue386_Op386XORLmodify(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (XORLmodify [off1] {sym} (ADDLconst [off2] base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2))
	// result: (XORLmodify [off1+off2] {sym} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386ADDLconst {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.Reset(ssaop.Op386XORLmodify)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (XORLmodify [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (XORLmodify [off1+off2] {ssa.MergeSym(sym1,sym2)} base val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.Op386LEAL {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+int64(off2)) && ssa.CanMergeSym(sym1, sym2) && (base.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.Op386XORLmodify)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValue386_OpAddr(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Addr {sym} base)
	// result: (LEAL {sym} base)
	for {
		sym := ssa.AuxToSym(v.Aux)
		base := v_0
		v.Reset(ssaop.Op386LEAL)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg(base)
		return true
	}
}
func rewriteValue386_OpBswap16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Bswap16 x)
	// result: (ROLWconst [8] x)
	for {
		x := v_0
		v.Reset(ssaop.Op386ROLWconst)
		v.AuxInt = ssa.Int16ToAuxInt(8)
		v.AddArg(x)
		return true
	}
}
func rewriteValue386_OpConst16(v *ssa.Value) bool {
	// match: (Const16 [c])
	// result: (MOVLconst [int32(c)])
	for {
		c := ssa.AuxIntToInt16(v.AuxInt)
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		return true
	}
}
func rewriteValue386_OpConst8(v *ssa.Value) bool {
	// match: (Const8 [c])
	// result: (MOVLconst [int32(c)])
	for {
		c := ssa.AuxIntToInt8(v.AuxInt)
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		return true
	}
}
func rewriteValue386_OpConstBool(v *ssa.Value) bool {
	// match: (ConstBool [c])
	// result: (MOVLconst [ssa.B2i32(c)])
	for {
		c := ssa.AuxIntToBool(v.AuxInt)
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(ssa.B2i32(c))
		return true
	}
}
func rewriteValue386_OpConstNil(v *ssa.Value) bool {
	// match: (ConstNil)
	// result: (MOVLconst [0])
	for {
		v.Reset(ssaop.Op386MOVLconst)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
}
func rewriteValue386_OpCtz16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz16 x)
	// result: (BSFL (ORLconst <typ.UInt32> [0x10000] x))
	for {
		x := v_0
		v.Reset(ssaop.Op386BSFL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386ORLconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(0x10000)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpCtz8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz8 x)
	// result: (BSFL (ORLconst <typ.UInt32> [0x100] x))
	for {
		x := v_0
		v.Reset(ssaop.Op386BSFL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386ORLconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(0x100)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpDiv8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// result: (DIVW (SignExt8to16 x) (SignExt8to16 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386DIVW)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to16, typ.Int16)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to16, typ.Int16)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue386_OpDiv8u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// result: (DIVWU (ZeroExt8to16 x) (ZeroExt8to16 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386DIVWU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to16, typ.UInt16)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to16, typ.UInt16)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue386_OpEq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq16 x y)
	// result: (SETEQ (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETEQ)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpEq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32 x y)
	// result: (SETEQ (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETEQ)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpEq32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32F x y)
	// result: (SETEQF (UCOMISS x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETEQF)
		v0 := b.NewValue0(v.Pos, ssaop.Op386UCOMISS, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpEq64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64F x y)
	// result: (SETEQF (UCOMISD x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETEQF)
		v0 := b.NewValue0(v.Pos, ssaop.Op386UCOMISD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpEq8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq8 x y)
	// result: (SETEQ (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETEQ)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpEqB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (EqB x y)
	// result: (SETEQ (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETEQ)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpEqPtr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (EqPtr x y)
	// result: (SETEQ (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETEQ)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpIsInBounds(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsInBounds idx len)
	// result: (SETB (CMPL idx len))
	for {
		idx := v_0
		len := v_1
		v.Reset(ssaop.Op386SETB)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPL, types.TypeFlags)
		v0.AddArg2(idx, len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpIsNonNil(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsNonNil p)
	// result: (SETNE (TESTL p p))
	for {
		p := v_0
		v.Reset(ssaop.Op386SETNE)
		v0 := b.NewValue0(v.Pos, ssaop.Op386TESTL, types.TypeFlags)
		v0.AddArg2(p, p)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpIsSliceInBounds(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsSliceInBounds idx len)
	// result: (SETBE (CMPL idx len))
	for {
		idx := v_0
		len := v_1
		v.Reset(ssaop.Op386SETBE)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPL, types.TypeFlags)
		v0.AddArg2(idx, len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLeq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq16 x y)
	// result: (SETLE (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETLE)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLeq16U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq16U x y)
	// result: (SETBE (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETBE)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLeq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32 x y)
	// result: (SETLE (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETLE)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLeq32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32F x y)
	// result: (SETGEF (UCOMISS y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETGEF)
		v0 := b.NewValue0(v.Pos, ssaop.Op386UCOMISS, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLeq32U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32U x y)
	// result: (SETBE (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETBE)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLeq64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64F x y)
	// result: (SETGEF (UCOMISD y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETGEF)
		v0 := b.NewValue0(v.Pos, ssaop.Op386UCOMISD, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLeq8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq8 x y)
	// result: (SETLE (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETLE)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLeq8U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq8U x y)
	// result: (SETBE (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETBE)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLess16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less16 x y)
	// result: (SETL (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLess16U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less16U x y)
	// result: (SETB (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETB)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLess32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32 x y)
	// result: (SETL (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLess32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32F x y)
	// result: (SETGF (UCOMISS y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETGF)
		v0 := b.NewValue0(v.Pos, ssaop.Op386UCOMISS, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLess32U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32U x y)
	// result: (SETB (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETB)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLess64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64F x y)
	// result: (SETGF (UCOMISD y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETGF)
		v0 := b.NewValue0(v.Pos, ssaop.Op386UCOMISD, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLess8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less8 x y)
	// result: (SETL (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLess8U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less8U x y)
	// result: (SETB (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETB)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpLoad(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Load <t> ptr mem)
	// cond: (ssa.Is32BitInt(t) ||ssa.IsPtr(t))
	// result: (MOVLload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is32BitInt(t) || ssa.IsPtr(t)) {
			break
		}
		v.Reset(ssaop.Op386MOVLload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ssa.Is16BitInt(t)
	// result: (MOVWload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is16BitInt(t)) {
			break
		}
		v.Reset(ssaop.Op386MOVWload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (t.IsBoolean() || ssa.Is8BitInt(t))
	// result: (MOVBload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.IsBoolean() || ssa.Is8BitInt(t)) {
			break
		}
		v.Reset(ssaop.Op386MOVBload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ssa.Is32BitFloat(t)
	// result: (MOVSSload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is32BitFloat(t)) {
			break
		}
		v.Reset(ssaop.Op386MOVSSload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ssa.Is64BitFloat(t)
	// result: (MOVSDload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is64BitFloat(t)) {
			break
		}
		v.Reset(ssaop.Op386MOVSDload)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue386_OpLocalAddr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LocalAddr <t> {sym} base mem)
	// cond: t.Elem().HasPointers()
	// result: (LEAL {sym} (SPanchored base mem))
	for {
		t := v.Type
		sym := ssa.AuxToSym(v.Aux)
		base := v_0
		mem := v_1
		if !(t.Elem().HasPointers()) {
			break
		}
		v.Reset(ssaop.Op386LEAL)
		v.Aux = ssa.SymToAux(sym)
		v0 := b.NewValue0(v.Pos, ssaop.OpSPanchored, typ.Uintptr)
		v0.AddArg2(base, mem)
		v.AddArg(v0)
		return true
	}
	// match: (LocalAddr <t> {sym} base _)
	// cond: !t.Elem().HasPointers()
	// result: (LEAL {sym} base)
	for {
		t := v.Type
		sym := ssa.AuxToSym(v.Aux)
		base := v_0
		if !(!t.Elem().HasPointers()) {
			break
		}
		v.Reset(ssaop.Op386LEAL)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg(base)
		return true
	}
	return false
}
func rewriteValue386_OpLsh16x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh16x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPWconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386ANDL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, ssaop.Op386CMPWconst, types.TypeFlags)
		v2.AuxInt = ssa.Int16ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh16x16 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SHLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SHLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpLsh16x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh16x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPLconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386ANDL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, ssaop.Op386CMPLconst, types.TypeFlags)
		v2.AuxInt = ssa.Int32ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh16x32 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SHLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SHLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpLsh16x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Lsh16x64 x (Const64 [c]))
	// cond: uint64(c) < 16
	// result: (SHLLconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.Reset(ssaop.Op386SHLLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Lsh16x64 _ (Const64 [c]))
	// cond: uint64(c) >= 16
	// result: (Const16 [0])
	for {
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 16) {
			break
		}
		v.Reset(ssaop.OpConst16)
		v.AuxInt = ssa.Int16ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_OpLsh16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh16x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPBconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386ANDL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, ssaop.Op386CMPBconst, types.TypeFlags)
		v2.AuxInt = ssa.Int8ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh16x8 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SHLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SHLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpLsh32x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh32x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPWconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386ANDL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, ssaop.Op386CMPWconst, types.TypeFlags)
		v2.AuxInt = ssa.Int16ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh32x16 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SHLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SHLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpLsh32x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh32x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPLconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386ANDL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, ssaop.Op386CMPLconst, types.TypeFlags)
		v2.AuxInt = ssa.Int32ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh32x32 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SHLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SHLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpLsh32x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Lsh32x64 x (Const64 [c]))
	// cond: uint64(c) < 32
	// result: (SHLLconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.Reset(ssaop.Op386SHLLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Lsh32x64 _ (Const64 [c]))
	// cond: uint64(c) >= 32
	// result: (Const32 [0])
	for {
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 32) {
			break
		}
		v.Reset(ssaop.OpConst32)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_OpLsh32x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh32x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPBconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386ANDL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, ssaop.Op386CMPBconst, types.TypeFlags)
		v2.AuxInt = ssa.Int8ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh32x8 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SHLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SHLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpLsh8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh8x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPWconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386ANDL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, ssaop.Op386CMPWconst, types.TypeFlags)
		v2.AuxInt = ssa.Int16ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh8x16 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SHLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SHLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpLsh8x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh8x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPLconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386ANDL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, ssaop.Op386CMPLconst, types.TypeFlags)
		v2.AuxInt = ssa.Int32ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh8x32 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SHLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SHLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpLsh8x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Lsh8x64 x (Const64 [c]))
	// cond: uint64(c) < 8
	// result: (SHLLconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.Reset(ssaop.Op386SHLLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Lsh8x64 _ (Const64 [c]))
	// cond: uint64(c) >= 8
	// result: (Const8 [0])
	for {
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 8) {
			break
		}
		v.Reset(ssaop.OpConst8)
		v.AuxInt = ssa.Int8ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_OpLsh8x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh8x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPBconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386ANDL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, ssaop.Op386CMPBconst, types.TypeFlags)
		v2.AuxInt = ssa.Int8ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh8x8 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SHLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SHLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpMod8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8 x y)
	// result: (MODW (SignExt8to16 x) (SignExt8to16 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386MODW)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to16, typ.Int16)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to16, typ.Int16)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue386_OpMod8u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8u x y)
	// result: (MODWU (ZeroExt8to16 x) (ZeroExt8to16 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386MODWU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to16, typ.UInt16)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to16, typ.UInt16)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue386_OpMove(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
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
		v.Reset(ssaop.Op386MOVBstore)
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVBload, typ.UInt8)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [2] dst src mem)
	// result: (MOVWstore dst (MOVWload src mem) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.Op386MOVWstore)
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVWload, typ.UInt16)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [4] dst src mem)
	// result: (MOVLstore dst (MOVLload src mem) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.Op386MOVLstore)
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVLload, typ.UInt32)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [3] dst src mem)
	// result: (MOVBstore [2] dst (MOVBload [2] src mem) (MOVWstore dst (MOVWload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 3 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.Op386MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVBload, typ.UInt8)
		v0.AuxInt = ssa.Int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.Op386MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.Op386MOVWload, typ.UInt16)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [5] dst src mem)
	// result: (MOVBstore [4] dst (MOVBload [4] src mem) (MOVLstore dst (MOVLload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 5 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.Op386MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVBload, typ.UInt8)
		v0.AuxInt = ssa.Int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.Op386MOVLstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.Op386MOVLload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [6] dst src mem)
	// result: (MOVWstore [4] dst (MOVWload [4] src mem) (MOVLstore dst (MOVLload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 6 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.Op386MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVWload, typ.UInt16)
		v0.AuxInt = ssa.Int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.Op386MOVLstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.Op386MOVLload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [7] dst src mem)
	// result: (MOVLstore [3] dst (MOVLload [3] src mem) (MOVLstore dst (MOVLload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 7 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.Op386MOVLstore)
		v.AuxInt = ssa.Int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVLload, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(3)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.Op386MOVLstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.Op386MOVLload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [8] dst src mem)
	// result: (MOVLstore [4] dst (MOVLload [4] src mem) (MOVLstore dst (MOVLload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.Op386MOVLstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVLload, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.Op386MOVLstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.Op386MOVLload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 8 && s%4 != 0
	// result: (Move [s-s%4] (ADDLconst <dst.Type> dst [int32(s%4)]) (ADDLconst <src.Type> src [int32(s%4)]) (MOVLstore dst (MOVLload src mem) mem))
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 8 && s%4 != 0) {
			break
		}
		v.Reset(ssaop.OpMove)
		v.AuxInt = ssa.Int64ToAuxInt(s - s%4)
		v0 := b.NewValue0(v.Pos, ssaop.Op386ADDLconst, dst.Type)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(s % 4))
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, ssaop.Op386ADDLconst, src.Type)
		v1.AuxInt = ssa.Int32ToAuxInt(int32(s % 4))
		v1.AddArg(src)
		v2 := b.NewValue0(v.Pos, ssaop.Op386MOVLstore, types.TypeMem)
		v3 := b.NewValue0(v.Pos, ssaop.Op386MOVLload, typ.UInt32)
		v3.AddArg2(src, mem)
		v2.AddArg3(dst, v3, mem)
		v.AddArg3(v0, v1, v2)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 8 && s <= 4*128 && s%4 == 0 && ssa.LogLargeCopyValue(v, s)
	// result: (DUFFCOPY [10*(128-s/4)] dst src mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 8 && s <= 4*128 && s%4 == 0 && ssa.LogLargeCopyValue(v, s)) {
			break
		}
		v.Reset(ssaop.Op386DUFFCOPY)
		v.AuxInt = ssa.Int64ToAuxInt(10 * (128 - s/4))
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 4*128 && s%4 == 0 && ssa.LogLargeCopyValue(v, s)
	// result: (REPMOVSL dst src (MOVLconst [int32(s/4)]) mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 4*128 && s%4 == 0 && ssa.LogLargeCopyValue(v, s)) {
			break
		}
		v.Reset(ssaop.Op386REPMOVSL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVLconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(s / 4))
		v.AddArg4(dst, src, v0, mem)
		return true
	}
	return false
}
func rewriteValue386_OpNeg32F(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg32F x)
	// result: (PXOR x (MOVSSconst <typ.Float32> [float32(math.Copysign(0, -1))]))
	for {
		x := v_0
		v.Reset(ssaop.Op386PXOR)
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVSSconst, typ.Float32)
		v0.AuxInt = ssa.Float32ToAuxInt(float32(math.Copysign(0, -1)))
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue386_OpNeg64F(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg64F x)
	// result: (PXOR x (MOVSDconst <typ.Float64> [math.Copysign(0, -1)]))
	for {
		x := v_0
		v.Reset(ssaop.Op386PXOR)
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVSDconst, typ.Float64)
		v0.AuxInt = ssa.Float64ToAuxInt(math.Copysign(0, -1))
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue386_OpNeq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq16 x y)
	// result: (SETNE (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETNE)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpNeq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32 x y)
	// result: (SETNE (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETNE)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpNeq32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32F x y)
	// result: (SETNEF (UCOMISS x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETNEF)
		v0 := b.NewValue0(v.Pos, ssaop.Op386UCOMISS, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpNeq64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64F x y)
	// result: (SETNEF (UCOMISD x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETNEF)
		v0 := b.NewValue0(v.Pos, ssaop.Op386UCOMISD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpNeq8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq8 x y)
	// result: (SETNE (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETNE)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpNeqB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (NeqB x y)
	// result: (SETNE (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETNE)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpNeqPtr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (NeqPtr x y)
	// result: (SETNE (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.Op386SETNE)
		v0 := b.NewValue0(v.Pos, ssaop.Op386CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpNot(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Not x)
	// result: (XORLconst [1] x)
	for {
		x := v_0
		v.Reset(ssaop.Op386XORLconst)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValue386_OpOffPtr(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (OffPtr [off] ptr)
	// result: (ADDLconst [int32(off)] ptr)
	for {
		off := ssa.AuxIntToInt64(v.AuxInt)
		ptr := v_0
		v.Reset(ssaop.Op386ADDLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(off))
		v.AddArg(ptr)
		return true
	}
}
func rewriteValue386_OpRsh16Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16Ux16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (ANDL (SHRW <t> x y) (SBBLcarrymask <t> (CMPWconst y [16])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386ANDL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHRW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, ssaop.Op386CMPWconst, types.TypeFlags)
		v2.AuxInt = ssa.Int16ToAuxInt(16)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh16Ux16 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SHRW <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SHRW)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh16Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16Ux32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (ANDL (SHRW <t> x y) (SBBLcarrymask <t> (CMPLconst y [16])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386ANDL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHRW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, ssaop.Op386CMPLconst, types.TypeFlags)
		v2.AuxInt = ssa.Int32ToAuxInt(16)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh16Ux32 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SHRW <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SHRW)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh16Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Rsh16Ux64 x (Const64 [c]))
	// cond: uint64(c) < 16
	// result: (SHRWconst x [int16(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.Reset(ssaop.Op386SHRWconst)
		v.AuxInt = ssa.Int16ToAuxInt(int16(c))
		v.AddArg(x)
		return true
	}
	// match: (Rsh16Ux64 _ (Const64 [c]))
	// cond: uint64(c) >= 16
	// result: (Const16 [0])
	for {
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 16) {
			break
		}
		v.Reset(ssaop.OpConst16)
		v.AuxInt = ssa.Int16ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_OpRsh16Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16Ux8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (ANDL (SHRW <t> x y) (SBBLcarrymask <t> (CMPBconst y [16])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386ANDL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHRW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, ssaop.Op386CMPBconst, types.TypeFlags)
		v2.AuxInt = ssa.Int8ToAuxInt(16)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh16Ux8 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SHRW <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SHRW)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh16x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SARW <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPWconst y [16])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SARW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.Op386ORL, y.Type)
		v1 := b.NewValue0(v.Pos, ssaop.Op386NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, ssaop.Op386CMPWconst, types.TypeFlags)
		v3.AuxInt = ssa.Int16ToAuxInt(16)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16x16 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SARW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SARW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh16x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SARW <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPLconst y [16])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SARW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.Op386ORL, y.Type)
		v1 := b.NewValue0(v.Pos, ssaop.Op386NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, ssaop.Op386CMPLconst, types.TypeFlags)
		v3.AuxInt = ssa.Int32ToAuxInt(16)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16x32 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SARW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SARW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh16x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Rsh16x64 x (Const64 [c]))
	// cond: uint64(c) < 16
	// result: (SARWconst x [int16(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.Reset(ssaop.Op386SARWconst)
		v.AuxInt = ssa.Int16ToAuxInt(int16(c))
		v.AddArg(x)
		return true
	}
	// match: (Rsh16x64 x (Const64 [c]))
	// cond: uint64(c) >= 16
	// result: (SARWconst x [15])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 16) {
			break
		}
		v.Reset(ssaop.Op386SARWconst)
		v.AuxInt = ssa.Int16ToAuxInt(15)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_OpRsh16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SARW <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPBconst y [16])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SARW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.Op386ORL, y.Type)
		v1 := b.NewValue0(v.Pos, ssaop.Op386NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, ssaop.Op386CMPBconst, types.TypeFlags)
		v3.AuxInt = ssa.Int8ToAuxInt(16)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16x8 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SARW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SARW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh32Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32Ux16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (ANDL (SHRL <t> x y) (SBBLcarrymask <t> (CMPWconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386ANDL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, ssaop.Op386CMPWconst, types.TypeFlags)
		v2.AuxInt = ssa.Int16ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh32Ux16 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SHRL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SHRL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh32Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32Ux32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (ANDL (SHRL <t> x y) (SBBLcarrymask <t> (CMPLconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386ANDL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, ssaop.Op386CMPLconst, types.TypeFlags)
		v2.AuxInt = ssa.Int32ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh32Ux32 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SHRL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SHRL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh32Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Rsh32Ux64 x (Const64 [c]))
	// cond: uint64(c) < 32
	// result: (SHRLconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.Reset(ssaop.Op386SHRLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Rsh32Ux64 _ (Const64 [c]))
	// cond: uint64(c) >= 32
	// result: (Const32 [0])
	for {
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 32) {
			break
		}
		v.Reset(ssaop.OpConst32)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_OpRsh32Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32Ux8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (ANDL (SHRL <t> x y) (SBBLcarrymask <t> (CMPBconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386ANDL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, ssaop.Op386CMPBconst, types.TypeFlags)
		v2.AuxInt = ssa.Int8ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh32Ux8 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SHRL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SHRL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh32x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SARL <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPWconst y [32])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SARL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.Op386ORL, y.Type)
		v1 := b.NewValue0(v.Pos, ssaop.Op386NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, ssaop.Op386CMPWconst, types.TypeFlags)
		v3.AuxInt = ssa.Int16ToAuxInt(32)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32x16 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SARL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SARL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh32x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SARL <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPLconst y [32])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SARL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.Op386ORL, y.Type)
		v1 := b.NewValue0(v.Pos, ssaop.Op386NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, ssaop.Op386CMPLconst, types.TypeFlags)
		v3.AuxInt = ssa.Int32ToAuxInt(32)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32x32 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SARL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SARL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh32x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Rsh32x64 x (Const64 [c]))
	// cond: uint64(c) < 32
	// result: (SARLconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.Reset(ssaop.Op386SARLconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Rsh32x64 x (Const64 [c]))
	// cond: uint64(c) >= 32
	// result: (SARLconst x [31])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 32) {
			break
		}
		v.Reset(ssaop.Op386SARLconst)
		v.AuxInt = ssa.Int32ToAuxInt(31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_OpRsh32x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SARL <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPBconst y [32])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SARL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.Op386ORL, y.Type)
		v1 := b.NewValue0(v.Pos, ssaop.Op386NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, ssaop.Op386CMPBconst, types.TypeFlags)
		v3.AuxInt = ssa.Int8ToAuxInt(32)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32x8 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SARL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SARL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh8Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8Ux16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (ANDL (SHRB <t> x y) (SBBLcarrymask <t> (CMPWconst y [8])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386ANDL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHRB, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, ssaop.Op386CMPWconst, types.TypeFlags)
		v2.AuxInt = ssa.Int16ToAuxInt(8)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh8Ux16 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SHRB <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SHRB)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh8Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8Ux32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (ANDL (SHRB <t> x y) (SBBLcarrymask <t> (CMPLconst y [8])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386ANDL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHRB, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, ssaop.Op386CMPLconst, types.TypeFlags)
		v2.AuxInt = ssa.Int32ToAuxInt(8)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh8Ux32 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SHRB <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SHRB)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh8Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Rsh8Ux64 x (Const64 [c]))
	// cond: uint64(c) < 8
	// result: (SHRBconst x [int8(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.Reset(ssaop.Op386SHRBconst)
		v.AuxInt = ssa.Int8ToAuxInt(int8(c))
		v.AddArg(x)
		return true
	}
	// match: (Rsh8Ux64 _ (Const64 [c]))
	// cond: uint64(c) >= 8
	// result: (Const8 [0])
	for {
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 8) {
			break
		}
		v.Reset(ssaop.OpConst8)
		v.AuxInt = ssa.Int8ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue386_OpRsh8Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8Ux8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (ANDL (SHRB <t> x y) (SBBLcarrymask <t> (CMPBconst y [8])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386ANDL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SHRB, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, ssaop.Op386CMPBconst, types.TypeFlags)
		v2.AuxInt = ssa.Int8ToAuxInt(8)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh8Ux8 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SHRB <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SHRB)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SARB <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPWconst y [8])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SARB)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.Op386ORL, y.Type)
		v1 := b.NewValue0(v.Pos, ssaop.Op386NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, ssaop.Op386CMPWconst, types.TypeFlags)
		v3.AuxInt = ssa.Int16ToAuxInt(8)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8x16 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SARB x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SARB)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh8x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SARB <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPLconst y [8])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SARB)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.Op386ORL, y.Type)
		v1 := b.NewValue0(v.Pos, ssaop.Op386NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, ssaop.Op386CMPLconst, types.TypeFlags)
		v3.AuxInt = ssa.Int32ToAuxInt(8)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8x32 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SARB x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SARB)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpRsh8x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Rsh8x64 x (Const64 [c]))
	// cond: uint64(c) < 8
	// result: (SARBconst x [int8(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.Reset(ssaop.Op386SARBconst)
		v.AuxInt = ssa.Int8ToAuxInt(int8(c))
		v.AddArg(x)
		return true
	}
	// match: (Rsh8x64 x (Const64 [c]))
	// cond: uint64(c) >= 8
	// result: (SARBconst x [7])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 8) {
			break
		}
		v.Reset(ssaop.Op386SARBconst)
		v.AuxInt = ssa.Int8ToAuxInt(7)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue386_OpRsh8x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SARB <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPBconst y [8])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SARB)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.Op386ORL, y.Type)
		v1 := b.NewValue0(v.Pos, ssaop.Op386NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, ssaop.Op386CMPBconst, types.TypeFlags)
		v3.AuxInt = ssa.Int8ToAuxInt(8)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8x8 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SARB x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.Op386SARB)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue386_OpSelect0(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select0 (Mul32uover x y))
	// result: (Select0 <typ.UInt32> (MULLU x y))
	for {
		if v_0.Op != ssaop.OpMul32uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpSelect0)
		v.Type = typ.UInt32
		v0 := b.NewValue0(v.Pos, ssaop.Op386MULLU, types.NewTuple(typ.UInt32, types.TypeFlags))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue386_OpSelect1(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select1 (Mul32uover x y))
	// result: (SETO (Select1 <types.TypeFlags> (MULLU x y)))
	for {
		if v_0.Op != ssaop.OpMul32uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.Op386SETO)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.Op386MULLU, types.NewTuple(typ.UInt32, types.TypeFlags))
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue386_OpSignmask(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Signmask x)
	// result: (SARLconst x [31])
	for {
		x := v_0
		v.Reset(ssaop.Op386SARLconst)
		v.AuxInt = ssa.Int32ToAuxInt(31)
		v.AddArg(x)
		return true
	}
}
func rewriteValue386_OpSlicemask(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Slicemask <t> x)
	// result: (SARLconst (NEGL <t> x) [31])
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.Op386SARLconst)
		v.AuxInt = ssa.Int32ToAuxInt(31)
		v0 := b.NewValue0(v.Pos, ssaop.Op386NEGL, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue386_OpStore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && t.IsFloat()
	// result: (MOVSDstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && t.IsFloat()) {
			break
		}
		v.Reset(ssaop.Op386MOVSDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && t.IsFloat()
	// result: (MOVSSstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && t.IsFloat()) {
			break
		}
		v.Reset(ssaop.Op386MOVSSstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && !t.IsFloat()
	// result: (MOVLstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && !t.IsFloat()) {
			break
		}
		v.Reset(ssaop.Op386MOVLstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 2
	// result: (MOVWstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 2) {
			break
		}
		v.Reset(ssaop.Op386MOVWstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
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
		v.Reset(ssaop.Op386MOVBstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValue386_OpZero(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
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
	// match: (Zero [1] destptr mem)
	// result: (MOVBstoreconst [0] destptr mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.Op386MOVBstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(0)
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [2] destptr mem)
	// result: (MOVWstoreconst [0] destptr mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.Op386MOVWstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(0)
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [4] destptr mem)
	// result: (MOVLstoreconst [0] destptr mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.Op386MOVLstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(0)
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [3] destptr mem)
	// result: (MOVBstoreconst [ssa.MakeValAndOff(0,2)] destptr (MOVWstoreconst [ssa.MakeValAndOff(0,0)] destptr mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 3 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.Op386MOVBstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(0, 2))
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVWstoreconst, types.TypeMem)
		v0.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [5] destptr mem)
	// result: (MOVBstoreconst [ssa.MakeValAndOff(0,4)] destptr (MOVLstoreconst [ssa.MakeValAndOff(0,0)] destptr mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 5 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.Op386MOVBstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(0, 4))
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVLstoreconst, types.TypeMem)
		v0.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [6] destptr mem)
	// result: (MOVWstoreconst [ssa.MakeValAndOff(0,4)] destptr (MOVLstoreconst [ssa.MakeValAndOff(0,0)] destptr mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 6 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.Op386MOVWstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(0, 4))
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVLstoreconst, types.TypeMem)
		v0.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [7] destptr mem)
	// result: (MOVLstoreconst [ssa.MakeValAndOff(0,3)] destptr (MOVLstoreconst [ssa.MakeValAndOff(0,0)] destptr mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 7 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.Op386MOVLstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(0, 3))
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVLstoreconst, types.TypeMem)
		v0.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s%4 != 0 && s > 4
	// result: (Zero [s-s%4] (ADDLconst destptr [int32(s%4)]) (MOVLstoreconst [0] destptr mem))
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		if !(s%4 != 0 && s > 4) {
			break
		}
		v.Reset(ssaop.OpZero)
		v.AuxInt = ssa.Int64ToAuxInt(s - s%4)
		v0 := b.NewValue0(v.Pos, ssaop.Op386ADDLconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(s % 4))
		v0.AddArg(destptr)
		v1 := b.NewValue0(v.Pos, ssaop.Op386MOVLstoreconst, types.TypeMem)
		v1.AuxInt = ssa.ValAndOffToAuxInt(0)
		v1.AddArg2(destptr, mem)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Zero [8] destptr mem)
	// result: (MOVLstoreconst [ssa.MakeValAndOff(0,4)] destptr (MOVLstoreconst [ssa.MakeValAndOff(0,0)] destptr mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.Op386MOVLstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(0, 4))
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVLstoreconst, types.TypeMem)
		v0.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [12] destptr mem)
	// result: (MOVLstoreconst [ssa.MakeValAndOff(0,8)] destptr (MOVLstoreconst [ssa.MakeValAndOff(0,4)] destptr (MOVLstoreconst [ssa.MakeValAndOff(0,0)] destptr mem)))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 12 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.Op386MOVLstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(0, 8))
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVLstoreconst, types.TypeMem)
		v0.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(0, 4))
		v1 := b.NewValue0(v.Pos, ssaop.Op386MOVLstoreconst, types.TypeMem)
		v1.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(0, 0))
		v1.AddArg2(destptr, mem)
		v0.AddArg2(destptr, v1)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [16] destptr mem)
	// result: (MOVLstoreconst [ssa.MakeValAndOff(0,12)] destptr (MOVLstoreconst [ssa.MakeValAndOff(0,8)] destptr (MOVLstoreconst [ssa.MakeValAndOff(0,4)] destptr (MOVLstoreconst [ssa.MakeValAndOff(0,0)] destptr mem))))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 16 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.Op386MOVLstoreconst)
		v.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(0, 12))
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVLstoreconst, types.TypeMem)
		v0.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(0, 8))
		v1 := b.NewValue0(v.Pos, ssaop.Op386MOVLstoreconst, types.TypeMem)
		v1.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(0, 4))
		v2 := b.NewValue0(v.Pos, ssaop.Op386MOVLstoreconst, types.TypeMem)
		v2.AuxInt = ssa.ValAndOffToAuxInt(ssa.MakeValAndOff(0, 0))
		v2.AddArg2(destptr, mem)
		v1.AddArg2(destptr, v2)
		v0.AddArg2(destptr, v1)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s > 16 && s <= 4*128 && s%4 == 0
	// result: (DUFFZERO [1*(128-s/4)] destptr (MOVLconst [0]) mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		if !(s > 16 && s <= 4*128 && s%4 == 0) {
			break
		}
		v.Reset(ssaop.Op386DUFFZERO)
		v.AuxInt = ssa.Int64ToAuxInt(1 * (128 - s/4))
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVLconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(0)
		v.AddArg3(destptr, v0, mem)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s > 4*128 && s%4 == 0
	// result: (REPSTOSL destptr (MOVLconst [int32(s/4)]) (MOVLconst [0]) mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		if !(s > 4*128 && s%4 == 0) {
			break
		}
		v.Reset(ssaop.Op386REPSTOSL)
		v0 := b.NewValue0(v.Pos, ssaop.Op386MOVLconst, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(s / 4))
		v1 := b.NewValue0(v.Pos, ssaop.Op386MOVLconst, typ.UInt32)
		v1.AuxInt = ssa.Int32ToAuxInt(0)
		v.AddArg4(destptr, v0, v1, mem)
		return true
	}
	return false
}
func rewriteValue386_OpZeromask(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Zeromask <t> x)
	// result: (XORLconst [-1] (SBBLcarrymask <t> (CMPLconst x [1])))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.Op386XORLconst)
		v.AuxInt = ssa.Int32ToAuxInt(-1)
		v0 := b.NewValue0(v.Pos, ssaop.Op386SBBLcarrymask, t)
		v1 := b.NewValue0(v.Pos, ssaop.Op386CMPLconst, types.TypeFlags)
		v1.AuxInt = ssa.Int32ToAuxInt(1)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteBlock386(b *ssa.Block) bool {
	switch b.Kind {
	case block.Block386EQ:
		// match: (EQ (InvertFlags cmp) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == ssaop.Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386EQ, cmp)
			return true
		}
		// match: (EQ (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagEQ {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (EQ (FlagLT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagLT_ULT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (EQ (FlagLT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagLT_UGT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (EQ (FlagGT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagGT_ULT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (EQ (FlagGT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagGT_UGT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.Block386GE:
		// match: (GE (InvertFlags cmp) yes no)
		// result: (LE cmp yes no)
		for b.Controls[0].Op == ssaop.Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386LE, cmp)
			return true
		}
		// match: (GE (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagEQ {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GE (FlagLT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagLT_ULT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (GE (FlagLT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagLT_UGT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (GE (FlagGT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagGT_ULT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GE (FlagGT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagGT_UGT {
			b.Reset(block.BlockFirst)
			return true
		}
	case block.Block386GT:
		// match: (GT (InvertFlags cmp) yes no)
		// result: (LT cmp yes no)
		for b.Controls[0].Op == ssaop.Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386LT, cmp)
			return true
		}
		// match: (GT (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagEQ {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (GT (FlagLT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagLT_ULT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (GT (FlagLT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagLT_UGT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (GT (FlagGT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagGT_ULT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GT (FlagGT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagGT_UGT {
			b.Reset(block.BlockFirst)
			return true
		}
	case block.BlockIf:
		// match: (If (SETL cmp) yes no)
		// result: (LT cmp yes no)
		for b.Controls[0].Op == ssaop.Op386SETL {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386LT, cmp)
			return true
		}
		// match: (If (SETLE cmp) yes no)
		// result: (LE cmp yes no)
		for b.Controls[0].Op == ssaop.Op386SETLE {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386LE, cmp)
			return true
		}
		// match: (If (SETG cmp) yes no)
		// result: (GT cmp yes no)
		for b.Controls[0].Op == ssaop.Op386SETG {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386GT, cmp)
			return true
		}
		// match: (If (SETGE cmp) yes no)
		// result: (GE cmp yes no)
		for b.Controls[0].Op == ssaop.Op386SETGE {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386GE, cmp)
			return true
		}
		// match: (If (SETEQ cmp) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == ssaop.Op386SETEQ {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386EQ, cmp)
			return true
		}
		// match: (If (SETNE cmp) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == ssaop.Op386SETNE {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386NE, cmp)
			return true
		}
		// match: (If (SETB cmp) yes no)
		// result: (ULT cmp yes no)
		for b.Controls[0].Op == ssaop.Op386SETB {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386ULT, cmp)
			return true
		}
		// match: (If (SETBE cmp) yes no)
		// result: (ULE cmp yes no)
		for b.Controls[0].Op == ssaop.Op386SETBE {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386ULE, cmp)
			return true
		}
		// match: (If (SETA cmp) yes no)
		// result: (UGT cmp yes no)
		for b.Controls[0].Op == ssaop.Op386SETA {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386UGT, cmp)
			return true
		}
		// match: (If (SETAE cmp) yes no)
		// result: (UGE cmp yes no)
		for b.Controls[0].Op == ssaop.Op386SETAE {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386UGE, cmp)
			return true
		}
		// match: (If (SETO cmp) yes no)
		// result: (OS cmp yes no)
		for b.Controls[0].Op == ssaop.Op386SETO {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386OS, cmp)
			return true
		}
		// match: (If (SETGF cmp) yes no)
		// result: (UGT cmp yes no)
		for b.Controls[0].Op == ssaop.Op386SETGF {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386UGT, cmp)
			return true
		}
		// match: (If (SETGEF cmp) yes no)
		// result: (UGE cmp yes no)
		for b.Controls[0].Op == ssaop.Op386SETGEF {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386UGE, cmp)
			return true
		}
		// match: (If (SETEQF cmp) yes no)
		// result: (EQF cmp yes no)
		for b.Controls[0].Op == ssaop.Op386SETEQF {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386EQF, cmp)
			return true
		}
		// match: (If (SETNEF cmp) yes no)
		// result: (NEF cmp yes no)
		for b.Controls[0].Op == ssaop.Op386SETNEF {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386NEF, cmp)
			return true
		}
		// match: (If cond yes no)
		// result: (NE (TESTB cond cond) yes no)
		for {
			cond := b.Controls[0]
			v0 := b.NewValue0(cond.Pos, ssaop.Op386TESTB, types.TypeFlags)
			v0.AddArg2(cond, cond)
			b.ResetWithControl(block.Block386NE, v0)
			return true
		}
	case block.Block386LE:
		// match: (LE (InvertFlags cmp) yes no)
		// result: (GE cmp yes no)
		for b.Controls[0].Op == ssaop.Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386GE, cmp)
			return true
		}
		// match: (LE (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagEQ {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LE (FlagLT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagLT_ULT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LE (FlagLT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagLT_UGT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LE (FlagGT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagGT_ULT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (LE (FlagGT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagGT_UGT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.Block386LT:
		// match: (LT (InvertFlags cmp) yes no)
		// result: (GT cmp yes no)
		for b.Controls[0].Op == ssaop.Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386GT, cmp)
			return true
		}
		// match: (LT (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagEQ {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (LT (FlagLT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagLT_ULT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LT (FlagLT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagLT_UGT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LT (FlagGT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagGT_ULT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (LT (FlagGT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagGT_UGT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.Block386NE:
		// match: (NE (TESTB (SETL cmp) (SETL cmp)) yes no)
		// result: (LT cmp yes no)
		for b.Controls[0].Op == ssaop.Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.Op386SETL {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.Op386SETL || cmp != v_0_1.Args[0] {
				break
			}
			b.ResetWithControl(block.Block386LT, cmp)
			return true
		}
		// match: (NE (TESTB (SETLE cmp) (SETLE cmp)) yes no)
		// result: (LE cmp yes no)
		for b.Controls[0].Op == ssaop.Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.Op386SETLE {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.Op386SETLE || cmp != v_0_1.Args[0] {
				break
			}
			b.ResetWithControl(block.Block386LE, cmp)
			return true
		}
		// match: (NE (TESTB (SETG cmp) (SETG cmp)) yes no)
		// result: (GT cmp yes no)
		for b.Controls[0].Op == ssaop.Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.Op386SETG {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.Op386SETG || cmp != v_0_1.Args[0] {
				break
			}
			b.ResetWithControl(block.Block386GT, cmp)
			return true
		}
		// match: (NE (TESTB (SETGE cmp) (SETGE cmp)) yes no)
		// result: (GE cmp yes no)
		for b.Controls[0].Op == ssaop.Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.Op386SETGE {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.Op386SETGE || cmp != v_0_1.Args[0] {
				break
			}
			b.ResetWithControl(block.Block386GE, cmp)
			return true
		}
		// match: (NE (TESTB (SETEQ cmp) (SETEQ cmp)) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == ssaop.Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.Op386SETEQ {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.Op386SETEQ || cmp != v_0_1.Args[0] {
				break
			}
			b.ResetWithControl(block.Block386EQ, cmp)
			return true
		}
		// match: (NE (TESTB (SETNE cmp) (SETNE cmp)) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == ssaop.Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.Op386SETNE {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.Op386SETNE || cmp != v_0_1.Args[0] {
				break
			}
			b.ResetWithControl(block.Block386NE, cmp)
			return true
		}
		// match: (NE (TESTB (SETB cmp) (SETB cmp)) yes no)
		// result: (ULT cmp yes no)
		for b.Controls[0].Op == ssaop.Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.Op386SETB {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.Op386SETB || cmp != v_0_1.Args[0] {
				break
			}
			b.ResetWithControl(block.Block386ULT, cmp)
			return true
		}
		// match: (NE (TESTB (SETBE cmp) (SETBE cmp)) yes no)
		// result: (ULE cmp yes no)
		for b.Controls[0].Op == ssaop.Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.Op386SETBE {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.Op386SETBE || cmp != v_0_1.Args[0] {
				break
			}
			b.ResetWithControl(block.Block386ULE, cmp)
			return true
		}
		// match: (NE (TESTB (SETA cmp) (SETA cmp)) yes no)
		// result: (UGT cmp yes no)
		for b.Controls[0].Op == ssaop.Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.Op386SETA {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.Op386SETA || cmp != v_0_1.Args[0] {
				break
			}
			b.ResetWithControl(block.Block386UGT, cmp)
			return true
		}
		// match: (NE (TESTB (SETAE cmp) (SETAE cmp)) yes no)
		// result: (UGE cmp yes no)
		for b.Controls[0].Op == ssaop.Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.Op386SETAE {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.Op386SETAE || cmp != v_0_1.Args[0] {
				break
			}
			b.ResetWithControl(block.Block386UGE, cmp)
			return true
		}
		// match: (NE (TESTB (SETO cmp) (SETO cmp)) yes no)
		// result: (OS cmp yes no)
		for b.Controls[0].Op == ssaop.Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.Op386SETO {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.Op386SETO || cmp != v_0_1.Args[0] {
				break
			}
			b.ResetWithControl(block.Block386OS, cmp)
			return true
		}
		// match: (NE (TESTB (SETGF cmp) (SETGF cmp)) yes no)
		// result: (UGT cmp yes no)
		for b.Controls[0].Op == ssaop.Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.Op386SETGF {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.Op386SETGF || cmp != v_0_1.Args[0] {
				break
			}
			b.ResetWithControl(block.Block386UGT, cmp)
			return true
		}
		// match: (NE (TESTB (SETGEF cmp) (SETGEF cmp)) yes no)
		// result: (UGE cmp yes no)
		for b.Controls[0].Op == ssaop.Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.Op386SETGEF {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.Op386SETGEF || cmp != v_0_1.Args[0] {
				break
			}
			b.ResetWithControl(block.Block386UGE, cmp)
			return true
		}
		// match: (NE (TESTB (SETEQF cmp) (SETEQF cmp)) yes no)
		// result: (EQF cmp yes no)
		for b.Controls[0].Op == ssaop.Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.Op386SETEQF {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.Op386SETEQF || cmp != v_0_1.Args[0] {
				break
			}
			b.ResetWithControl(block.Block386EQF, cmp)
			return true
		}
		// match: (NE (TESTB (SETNEF cmp) (SETNEF cmp)) yes no)
		// result: (NEF cmp yes no)
		for b.Controls[0].Op == ssaop.Op386TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.Op386SETNEF {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.Op386SETNEF || cmp != v_0_1.Args[0] {
				break
			}
			b.ResetWithControl(block.Block386NEF, cmp)
			return true
		}
		// match: (NE (InvertFlags cmp) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == ssaop.Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386NE, cmp)
			return true
		}
		// match: (NE (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagEQ {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (NE (FlagLT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagLT_ULT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (NE (FlagLT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagLT_UGT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (NE (FlagGT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagGT_ULT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (NE (FlagGT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagGT_UGT {
			b.Reset(block.BlockFirst)
			return true
		}
	case block.Block386UGE:
		// match: (UGE (InvertFlags cmp) yes no)
		// result: (ULE cmp yes no)
		for b.Controls[0].Op == ssaop.Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386ULE, cmp)
			return true
		}
		// match: (UGE (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagEQ {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (UGE (FlagLT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagLT_ULT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (UGE (FlagLT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagLT_UGT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (UGE (FlagGT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagGT_ULT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (UGE (FlagGT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagGT_UGT {
			b.Reset(block.BlockFirst)
			return true
		}
	case block.Block386UGT:
		// match: (UGT (InvertFlags cmp) yes no)
		// result: (ULT cmp yes no)
		for b.Controls[0].Op == ssaop.Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386ULT, cmp)
			return true
		}
		// match: (UGT (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagEQ {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (UGT (FlagLT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagLT_ULT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (UGT (FlagLT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagLT_UGT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (UGT (FlagGT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagGT_ULT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (UGT (FlagGT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagGT_UGT {
			b.Reset(block.BlockFirst)
			return true
		}
	case block.Block386ULE:
		// match: (ULE (InvertFlags cmp) yes no)
		// result: (UGE cmp yes no)
		for b.Controls[0].Op == ssaop.Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386UGE, cmp)
			return true
		}
		// match: (ULE (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagEQ {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (ULE (FlagLT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagLT_ULT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (ULE (FlagLT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagLT_UGT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (ULE (FlagGT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagGT_ULT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (ULE (FlagGT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagGT_UGT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.Block386ULT:
		// match: (ULT (InvertFlags cmp) yes no)
		// result: (UGT cmp yes no)
		for b.Controls[0].Op == ssaop.Op386InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.Block386UGT, cmp)
			return true
		}
		// match: (ULT (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagEQ {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (ULT (FlagLT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagLT_ULT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (ULT (FlagLT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagLT_UGT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (ULT (FlagGT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.Op386FlagGT_ULT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (ULT (FlagGT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.Op386FlagGT_UGT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	}
	return false
}
