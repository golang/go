// Code generated from _gen/AMD64.rules using 'go generate'; DO NOT EDIT.

package ssa

import "internal/buildcfg"
import "math"
import "cmd/internal/obj"
import "cmd/compile/internal/types"

func rewriteValueAMD64(v *Value) bool {
	switch v.Op {
	case OpAMD64ADCQ:
		return rewriteValueAMD64_OpAMD64ADCQ(v)
	case OpAMD64ADCQconst:
		return rewriteValueAMD64_OpAMD64ADCQconst(v)
	case OpAMD64ADDL:
		return rewriteValueAMD64_OpAMD64ADDL(v)
	case OpAMD64ADDLconst:
		return rewriteValueAMD64_OpAMD64ADDLconst(v)
	case OpAMD64ADDLconstmodify:
		return rewriteValueAMD64_OpAMD64ADDLconstmodify(v)
	case OpAMD64ADDLload:
		return rewriteValueAMD64_OpAMD64ADDLload(v)
	case OpAMD64ADDLmodify:
		return rewriteValueAMD64_OpAMD64ADDLmodify(v)
	case OpAMD64ADDQ:
		return rewriteValueAMD64_OpAMD64ADDQ(v)
	case OpAMD64ADDQcarry:
		return rewriteValueAMD64_OpAMD64ADDQcarry(v)
	case OpAMD64ADDQconst:
		return rewriteValueAMD64_OpAMD64ADDQconst(v)
	case OpAMD64ADDQconstmodify:
		return rewriteValueAMD64_OpAMD64ADDQconstmodify(v)
	case OpAMD64ADDQload:
		return rewriteValueAMD64_OpAMD64ADDQload(v)
	case OpAMD64ADDQmodify:
		return rewriteValueAMD64_OpAMD64ADDQmodify(v)
	case OpAMD64ADDSD:
		return rewriteValueAMD64_OpAMD64ADDSD(v)
	case OpAMD64ADDSDload:
		return rewriteValueAMD64_OpAMD64ADDSDload(v)
	case OpAMD64ADDSS:
		return rewriteValueAMD64_OpAMD64ADDSS(v)
	case OpAMD64ADDSSload:
		return rewriteValueAMD64_OpAMD64ADDSSload(v)
	case OpAMD64ANDL:
		return rewriteValueAMD64_OpAMD64ANDL(v)
	case OpAMD64ANDLconst:
		return rewriteValueAMD64_OpAMD64ANDLconst(v)
	case OpAMD64ANDLconstmodify:
		return rewriteValueAMD64_OpAMD64ANDLconstmodify(v)
	case OpAMD64ANDLload:
		return rewriteValueAMD64_OpAMD64ANDLload(v)
	case OpAMD64ANDLmodify:
		return rewriteValueAMD64_OpAMD64ANDLmodify(v)
	case OpAMD64ANDNL:
		return rewriteValueAMD64_OpAMD64ANDNL(v)
	case OpAMD64ANDNQ:
		return rewriteValueAMD64_OpAMD64ANDNQ(v)
	case OpAMD64ANDQ:
		return rewriteValueAMD64_OpAMD64ANDQ(v)
	case OpAMD64ANDQconst:
		return rewriteValueAMD64_OpAMD64ANDQconst(v)
	case OpAMD64ANDQconstmodify:
		return rewriteValueAMD64_OpAMD64ANDQconstmodify(v)
	case OpAMD64ANDQload:
		return rewriteValueAMD64_OpAMD64ANDQload(v)
	case OpAMD64ANDQmodify:
		return rewriteValueAMD64_OpAMD64ANDQmodify(v)
	case OpAMD64BSFQ:
		return rewriteValueAMD64_OpAMD64BSFQ(v)
	case OpAMD64BSWAPL:
		return rewriteValueAMD64_OpAMD64BSWAPL(v)
	case OpAMD64BSWAPQ:
		return rewriteValueAMD64_OpAMD64BSWAPQ(v)
	case OpAMD64BTCQconst:
		return rewriteValueAMD64_OpAMD64BTCQconst(v)
	case OpAMD64BTLconst:
		return rewriteValueAMD64_OpAMD64BTLconst(v)
	case OpAMD64BTQconst:
		return rewriteValueAMD64_OpAMD64BTQconst(v)
	case OpAMD64BTRQconst:
		return rewriteValueAMD64_OpAMD64BTRQconst(v)
	case OpAMD64BTSQconst:
		return rewriteValueAMD64_OpAMD64BTSQconst(v)
	case OpAMD64CMOVLCC:
		return rewriteValueAMD64_OpAMD64CMOVLCC(v)
	case OpAMD64CMOVLCS:
		return rewriteValueAMD64_OpAMD64CMOVLCS(v)
	case OpAMD64CMOVLEQ:
		return rewriteValueAMD64_OpAMD64CMOVLEQ(v)
	case OpAMD64CMOVLGE:
		return rewriteValueAMD64_OpAMD64CMOVLGE(v)
	case OpAMD64CMOVLGT:
		return rewriteValueAMD64_OpAMD64CMOVLGT(v)
	case OpAMD64CMOVLHI:
		return rewriteValueAMD64_OpAMD64CMOVLHI(v)
	case OpAMD64CMOVLLE:
		return rewriteValueAMD64_OpAMD64CMOVLLE(v)
	case OpAMD64CMOVLLS:
		return rewriteValueAMD64_OpAMD64CMOVLLS(v)
	case OpAMD64CMOVLLT:
		return rewriteValueAMD64_OpAMD64CMOVLLT(v)
	case OpAMD64CMOVLNE:
		return rewriteValueAMD64_OpAMD64CMOVLNE(v)
	case OpAMD64CMOVQCC:
		return rewriteValueAMD64_OpAMD64CMOVQCC(v)
	case OpAMD64CMOVQCS:
		return rewriteValueAMD64_OpAMD64CMOVQCS(v)
	case OpAMD64CMOVQEQ:
		return rewriteValueAMD64_OpAMD64CMOVQEQ(v)
	case OpAMD64CMOVQGE:
		return rewriteValueAMD64_OpAMD64CMOVQGE(v)
	case OpAMD64CMOVQGT:
		return rewriteValueAMD64_OpAMD64CMOVQGT(v)
	case OpAMD64CMOVQHI:
		return rewriteValueAMD64_OpAMD64CMOVQHI(v)
	case OpAMD64CMOVQLE:
		return rewriteValueAMD64_OpAMD64CMOVQLE(v)
	case OpAMD64CMOVQLS:
		return rewriteValueAMD64_OpAMD64CMOVQLS(v)
	case OpAMD64CMOVQLT:
		return rewriteValueAMD64_OpAMD64CMOVQLT(v)
	case OpAMD64CMOVQNE:
		return rewriteValueAMD64_OpAMD64CMOVQNE(v)
	case OpAMD64CMOVWCC:
		return rewriteValueAMD64_OpAMD64CMOVWCC(v)
	case OpAMD64CMOVWCS:
		return rewriteValueAMD64_OpAMD64CMOVWCS(v)
	case OpAMD64CMOVWEQ:
		return rewriteValueAMD64_OpAMD64CMOVWEQ(v)
	case OpAMD64CMOVWGE:
		return rewriteValueAMD64_OpAMD64CMOVWGE(v)
	case OpAMD64CMOVWGT:
		return rewriteValueAMD64_OpAMD64CMOVWGT(v)
	case OpAMD64CMOVWHI:
		return rewriteValueAMD64_OpAMD64CMOVWHI(v)
	case OpAMD64CMOVWLE:
		return rewriteValueAMD64_OpAMD64CMOVWLE(v)
	case OpAMD64CMOVWLS:
		return rewriteValueAMD64_OpAMD64CMOVWLS(v)
	case OpAMD64CMOVWLT:
		return rewriteValueAMD64_OpAMD64CMOVWLT(v)
	case OpAMD64CMOVWNE:
		return rewriteValueAMD64_OpAMD64CMOVWNE(v)
	case OpAMD64CMPB:
		return rewriteValueAMD64_OpAMD64CMPB(v)
	case OpAMD64CMPBconst:
		return rewriteValueAMD64_OpAMD64CMPBconst(v)
	case OpAMD64CMPBconstload:
		return rewriteValueAMD64_OpAMD64CMPBconstload(v)
	case OpAMD64CMPBload:
		return rewriteValueAMD64_OpAMD64CMPBload(v)
	case OpAMD64CMPL:
		return rewriteValueAMD64_OpAMD64CMPL(v)
	case OpAMD64CMPLconst:
		return rewriteValueAMD64_OpAMD64CMPLconst(v)
	case OpAMD64CMPLconstload:
		return rewriteValueAMD64_OpAMD64CMPLconstload(v)
	case OpAMD64CMPLload:
		return rewriteValueAMD64_OpAMD64CMPLload(v)
	case OpAMD64CMPQ:
		return rewriteValueAMD64_OpAMD64CMPQ(v)
	case OpAMD64CMPQconst:
		return rewriteValueAMD64_OpAMD64CMPQconst(v)
	case OpAMD64CMPQconstload:
		return rewriteValueAMD64_OpAMD64CMPQconstload(v)
	case OpAMD64CMPQload:
		return rewriteValueAMD64_OpAMD64CMPQload(v)
	case OpAMD64CMPW:
		return rewriteValueAMD64_OpAMD64CMPW(v)
	case OpAMD64CMPWconst:
		return rewriteValueAMD64_OpAMD64CMPWconst(v)
	case OpAMD64CMPWconstload:
		return rewriteValueAMD64_OpAMD64CMPWconstload(v)
	case OpAMD64CMPWload:
		return rewriteValueAMD64_OpAMD64CMPWload(v)
	case OpAMD64CMPXCHGLlock:
		return rewriteValueAMD64_OpAMD64CMPXCHGLlock(v)
	case OpAMD64CMPXCHGQlock:
		return rewriteValueAMD64_OpAMD64CMPXCHGQlock(v)
	case OpAMD64DIVSD:
		return rewriteValueAMD64_OpAMD64DIVSD(v)
	case OpAMD64DIVSDload:
		return rewriteValueAMD64_OpAMD64DIVSDload(v)
	case OpAMD64DIVSS:
		return rewriteValueAMD64_OpAMD64DIVSS(v)
	case OpAMD64DIVSSload:
		return rewriteValueAMD64_OpAMD64DIVSSload(v)
	case OpAMD64HMULL:
		return rewriteValueAMD64_OpAMD64HMULL(v)
	case OpAMD64HMULLU:
		return rewriteValueAMD64_OpAMD64HMULLU(v)
	case OpAMD64HMULQ:
		return rewriteValueAMD64_OpAMD64HMULQ(v)
	case OpAMD64HMULQU:
		return rewriteValueAMD64_OpAMD64HMULQU(v)
	case OpAMD64LEAL:
		return rewriteValueAMD64_OpAMD64LEAL(v)
	case OpAMD64LEAL1:
		return rewriteValueAMD64_OpAMD64LEAL1(v)
	case OpAMD64LEAL2:
		return rewriteValueAMD64_OpAMD64LEAL2(v)
	case OpAMD64LEAL4:
		return rewriteValueAMD64_OpAMD64LEAL4(v)
	case OpAMD64LEAL8:
		return rewriteValueAMD64_OpAMD64LEAL8(v)
	case OpAMD64LEAQ:
		return rewriteValueAMD64_OpAMD64LEAQ(v)
	case OpAMD64LEAQ1:
		return rewriteValueAMD64_OpAMD64LEAQ1(v)
	case OpAMD64LEAQ2:
		return rewriteValueAMD64_OpAMD64LEAQ2(v)
	case OpAMD64LEAQ4:
		return rewriteValueAMD64_OpAMD64LEAQ4(v)
	case OpAMD64LEAQ8:
		return rewriteValueAMD64_OpAMD64LEAQ8(v)
	case OpAMD64LoweredPanicBoundsCR:
		return rewriteValueAMD64_OpAMD64LoweredPanicBoundsCR(v)
	case OpAMD64LoweredPanicBoundsRC:
		return rewriteValueAMD64_OpAMD64LoweredPanicBoundsRC(v)
	case OpAMD64LoweredPanicBoundsRR:
		return rewriteValueAMD64_OpAMD64LoweredPanicBoundsRR(v)
	case OpAMD64MOVBELstore:
		return rewriteValueAMD64_OpAMD64MOVBELstore(v)
	case OpAMD64MOVBEQstore:
		return rewriteValueAMD64_OpAMD64MOVBEQstore(v)
	case OpAMD64MOVBEWstore:
		return rewriteValueAMD64_OpAMD64MOVBEWstore(v)
	case OpAMD64MOVBQSX:
		return rewriteValueAMD64_OpAMD64MOVBQSX(v)
	case OpAMD64MOVBQSXload:
		return rewriteValueAMD64_OpAMD64MOVBQSXload(v)
	case OpAMD64MOVBQZX:
		return rewriteValueAMD64_OpAMD64MOVBQZX(v)
	case OpAMD64MOVBatomicload:
		return rewriteValueAMD64_OpAMD64MOVBatomicload(v)
	case OpAMD64MOVBload:
		return rewriteValueAMD64_OpAMD64MOVBload(v)
	case OpAMD64MOVBstore:
		return rewriteValueAMD64_OpAMD64MOVBstore(v)
	case OpAMD64MOVBstoreconst:
		return rewriteValueAMD64_OpAMD64MOVBstoreconst(v)
	case OpAMD64MOVLQSX:
		return rewriteValueAMD64_OpAMD64MOVLQSX(v)
	case OpAMD64MOVLQSXload:
		return rewriteValueAMD64_OpAMD64MOVLQSXload(v)
	case OpAMD64MOVLQZX:
		return rewriteValueAMD64_OpAMD64MOVLQZX(v)
	case OpAMD64MOVLatomicload:
		return rewriteValueAMD64_OpAMD64MOVLatomicload(v)
	case OpAMD64MOVLf2i:
		return rewriteValueAMD64_OpAMD64MOVLf2i(v)
	case OpAMD64MOVLi2f:
		return rewriteValueAMD64_OpAMD64MOVLi2f(v)
	case OpAMD64MOVLload:
		return rewriteValueAMD64_OpAMD64MOVLload(v)
	case OpAMD64MOVLstore:
		return rewriteValueAMD64_OpAMD64MOVLstore(v)
	case OpAMD64MOVLstoreconst:
		return rewriteValueAMD64_OpAMD64MOVLstoreconst(v)
	case OpAMD64MOVOload:
		return rewriteValueAMD64_OpAMD64MOVOload(v)
	case OpAMD64MOVOstore:
		return rewriteValueAMD64_OpAMD64MOVOstore(v)
	case OpAMD64MOVOstoreconst:
		return rewriteValueAMD64_OpAMD64MOVOstoreconst(v)
	case OpAMD64MOVQatomicload:
		return rewriteValueAMD64_OpAMD64MOVQatomicload(v)
	case OpAMD64MOVQf2i:
		return rewriteValueAMD64_OpAMD64MOVQf2i(v)
	case OpAMD64MOVQi2f:
		return rewriteValueAMD64_OpAMD64MOVQi2f(v)
	case OpAMD64MOVQload:
		return rewriteValueAMD64_OpAMD64MOVQload(v)
	case OpAMD64MOVQstore:
		return rewriteValueAMD64_OpAMD64MOVQstore(v)
	case OpAMD64MOVQstoreconst:
		return rewriteValueAMD64_OpAMD64MOVQstoreconst(v)
	case OpAMD64MOVSDload:
		return rewriteValueAMD64_OpAMD64MOVSDload(v)
	case OpAMD64MOVSDstore:
		return rewriteValueAMD64_OpAMD64MOVSDstore(v)
	case OpAMD64MOVSSload:
		return rewriteValueAMD64_OpAMD64MOVSSload(v)
	case OpAMD64MOVSSstore:
		return rewriteValueAMD64_OpAMD64MOVSSstore(v)
	case OpAMD64MOVWQSX:
		return rewriteValueAMD64_OpAMD64MOVWQSX(v)
	case OpAMD64MOVWQSXload:
		return rewriteValueAMD64_OpAMD64MOVWQSXload(v)
	case OpAMD64MOVWQZX:
		return rewriteValueAMD64_OpAMD64MOVWQZX(v)
	case OpAMD64MOVWload:
		return rewriteValueAMD64_OpAMD64MOVWload(v)
	case OpAMD64MOVWstore:
		return rewriteValueAMD64_OpAMD64MOVWstore(v)
	case OpAMD64MOVWstoreconst:
		return rewriteValueAMD64_OpAMD64MOVWstoreconst(v)
	case OpAMD64MULL:
		return rewriteValueAMD64_OpAMD64MULL(v)
	case OpAMD64MULLconst:
		return rewriteValueAMD64_OpAMD64MULLconst(v)
	case OpAMD64MULQ:
		return rewriteValueAMD64_OpAMD64MULQ(v)
	case OpAMD64MULQconst:
		return rewriteValueAMD64_OpAMD64MULQconst(v)
	case OpAMD64MULSD:
		return rewriteValueAMD64_OpAMD64MULSD(v)
	case OpAMD64MULSDload:
		return rewriteValueAMD64_OpAMD64MULSDload(v)
	case OpAMD64MULSS:
		return rewriteValueAMD64_OpAMD64MULSS(v)
	case OpAMD64MULSSload:
		return rewriteValueAMD64_OpAMD64MULSSload(v)
	case OpAMD64NEGL:
		return rewriteValueAMD64_OpAMD64NEGL(v)
	case OpAMD64NEGQ:
		return rewriteValueAMD64_OpAMD64NEGQ(v)
	case OpAMD64NOTL:
		return rewriteValueAMD64_OpAMD64NOTL(v)
	case OpAMD64NOTQ:
		return rewriteValueAMD64_OpAMD64NOTQ(v)
	case OpAMD64ORL:
		return rewriteValueAMD64_OpAMD64ORL(v)
	case OpAMD64ORLconst:
		return rewriteValueAMD64_OpAMD64ORLconst(v)
	case OpAMD64ORLconstmodify:
		return rewriteValueAMD64_OpAMD64ORLconstmodify(v)
	case OpAMD64ORLload:
		return rewriteValueAMD64_OpAMD64ORLload(v)
	case OpAMD64ORLmodify:
		return rewriteValueAMD64_OpAMD64ORLmodify(v)
	case OpAMD64ORQ:
		return rewriteValueAMD64_OpAMD64ORQ(v)
	case OpAMD64ORQconst:
		return rewriteValueAMD64_OpAMD64ORQconst(v)
	case OpAMD64ORQconstmodify:
		return rewriteValueAMD64_OpAMD64ORQconstmodify(v)
	case OpAMD64ORQload:
		return rewriteValueAMD64_OpAMD64ORQload(v)
	case OpAMD64ORQmodify:
		return rewriteValueAMD64_OpAMD64ORQmodify(v)
	case OpAMD64ROLB:
		return rewriteValueAMD64_OpAMD64ROLB(v)
	case OpAMD64ROLBconst:
		return rewriteValueAMD64_OpAMD64ROLBconst(v)
	case OpAMD64ROLL:
		return rewriteValueAMD64_OpAMD64ROLL(v)
	case OpAMD64ROLLconst:
		return rewriteValueAMD64_OpAMD64ROLLconst(v)
	case OpAMD64ROLQ:
		return rewriteValueAMD64_OpAMD64ROLQ(v)
	case OpAMD64ROLQconst:
		return rewriteValueAMD64_OpAMD64ROLQconst(v)
	case OpAMD64ROLW:
		return rewriteValueAMD64_OpAMD64ROLW(v)
	case OpAMD64ROLWconst:
		return rewriteValueAMD64_OpAMD64ROLWconst(v)
	case OpAMD64RORB:
		return rewriteValueAMD64_OpAMD64RORB(v)
	case OpAMD64RORL:
		return rewriteValueAMD64_OpAMD64RORL(v)
	case OpAMD64RORQ:
		return rewriteValueAMD64_OpAMD64RORQ(v)
	case OpAMD64RORW:
		return rewriteValueAMD64_OpAMD64RORW(v)
	case OpAMD64SARB:
		return rewriteValueAMD64_OpAMD64SARB(v)
	case OpAMD64SARBconst:
		return rewriteValueAMD64_OpAMD64SARBconst(v)
	case OpAMD64SARL:
		return rewriteValueAMD64_OpAMD64SARL(v)
	case OpAMD64SARLconst:
		return rewriteValueAMD64_OpAMD64SARLconst(v)
	case OpAMD64SARQ:
		return rewriteValueAMD64_OpAMD64SARQ(v)
	case OpAMD64SARQconst:
		return rewriteValueAMD64_OpAMD64SARQconst(v)
	case OpAMD64SARW:
		return rewriteValueAMD64_OpAMD64SARW(v)
	case OpAMD64SARWconst:
		return rewriteValueAMD64_OpAMD64SARWconst(v)
	case OpAMD64SARXLload:
		return rewriteValueAMD64_OpAMD64SARXLload(v)
	case OpAMD64SARXQload:
		return rewriteValueAMD64_OpAMD64SARXQload(v)
	case OpAMD64SBBLcarrymask:
		return rewriteValueAMD64_OpAMD64SBBLcarrymask(v)
	case OpAMD64SBBQ:
		return rewriteValueAMD64_OpAMD64SBBQ(v)
	case OpAMD64SBBQcarrymask:
		return rewriteValueAMD64_OpAMD64SBBQcarrymask(v)
	case OpAMD64SBBQconst:
		return rewriteValueAMD64_OpAMD64SBBQconst(v)
	case OpAMD64SETA:
		return rewriteValueAMD64_OpAMD64SETA(v)
	case OpAMD64SETAE:
		return rewriteValueAMD64_OpAMD64SETAE(v)
	case OpAMD64SETAEstore:
		return rewriteValueAMD64_OpAMD64SETAEstore(v)
	case OpAMD64SETAstore:
		return rewriteValueAMD64_OpAMD64SETAstore(v)
	case OpAMD64SETB:
		return rewriteValueAMD64_OpAMD64SETB(v)
	case OpAMD64SETBE:
		return rewriteValueAMD64_OpAMD64SETBE(v)
	case OpAMD64SETBEstore:
		return rewriteValueAMD64_OpAMD64SETBEstore(v)
	case OpAMD64SETBstore:
		return rewriteValueAMD64_OpAMD64SETBstore(v)
	case OpAMD64SETEQ:
		return rewriteValueAMD64_OpAMD64SETEQ(v)
	case OpAMD64SETEQstore:
		return rewriteValueAMD64_OpAMD64SETEQstore(v)
	case OpAMD64SETG:
		return rewriteValueAMD64_OpAMD64SETG(v)
	case OpAMD64SETGE:
		return rewriteValueAMD64_OpAMD64SETGE(v)
	case OpAMD64SETGEstore:
		return rewriteValueAMD64_OpAMD64SETGEstore(v)
	case OpAMD64SETGstore:
		return rewriteValueAMD64_OpAMD64SETGstore(v)
	case OpAMD64SETL:
		return rewriteValueAMD64_OpAMD64SETL(v)
	case OpAMD64SETLE:
		return rewriteValueAMD64_OpAMD64SETLE(v)
	case OpAMD64SETLEstore:
		return rewriteValueAMD64_OpAMD64SETLEstore(v)
	case OpAMD64SETLstore:
		return rewriteValueAMD64_OpAMD64SETLstore(v)
	case OpAMD64SETNE:
		return rewriteValueAMD64_OpAMD64SETNE(v)
	case OpAMD64SETNEstore:
		return rewriteValueAMD64_OpAMD64SETNEstore(v)
	case OpAMD64SHLL:
		return rewriteValueAMD64_OpAMD64SHLL(v)
	case OpAMD64SHLLconst:
		return rewriteValueAMD64_OpAMD64SHLLconst(v)
	case OpAMD64SHLQ:
		return rewriteValueAMD64_OpAMD64SHLQ(v)
	case OpAMD64SHLQconst:
		return rewriteValueAMD64_OpAMD64SHLQconst(v)
	case OpAMD64SHLXLload:
		return rewriteValueAMD64_OpAMD64SHLXLload(v)
	case OpAMD64SHLXQload:
		return rewriteValueAMD64_OpAMD64SHLXQload(v)
	case OpAMD64SHRB:
		return rewriteValueAMD64_OpAMD64SHRB(v)
	case OpAMD64SHRBconst:
		return rewriteValueAMD64_OpAMD64SHRBconst(v)
	case OpAMD64SHRL:
		return rewriteValueAMD64_OpAMD64SHRL(v)
	case OpAMD64SHRLconst:
		return rewriteValueAMD64_OpAMD64SHRLconst(v)
	case OpAMD64SHRQ:
		return rewriteValueAMD64_OpAMD64SHRQ(v)
	case OpAMD64SHRQconst:
		return rewriteValueAMD64_OpAMD64SHRQconst(v)
	case OpAMD64SHRW:
		return rewriteValueAMD64_OpAMD64SHRW(v)
	case OpAMD64SHRWconst:
		return rewriteValueAMD64_OpAMD64SHRWconst(v)
	case OpAMD64SHRXLload:
		return rewriteValueAMD64_OpAMD64SHRXLload(v)
	case OpAMD64SHRXQload:
		return rewriteValueAMD64_OpAMD64SHRXQload(v)
	case OpAMD64SUBL:
		return rewriteValueAMD64_OpAMD64SUBL(v)
	case OpAMD64SUBLconst:
		return rewriteValueAMD64_OpAMD64SUBLconst(v)
	case OpAMD64SUBLload:
		return rewriteValueAMD64_OpAMD64SUBLload(v)
	case OpAMD64SUBLmodify:
		return rewriteValueAMD64_OpAMD64SUBLmodify(v)
	case OpAMD64SUBQ:
		return rewriteValueAMD64_OpAMD64SUBQ(v)
	case OpAMD64SUBQborrow:
		return rewriteValueAMD64_OpAMD64SUBQborrow(v)
	case OpAMD64SUBQconst:
		return rewriteValueAMD64_OpAMD64SUBQconst(v)
	case OpAMD64SUBQload:
		return rewriteValueAMD64_OpAMD64SUBQload(v)
	case OpAMD64SUBQmodify:
		return rewriteValueAMD64_OpAMD64SUBQmodify(v)
	case OpAMD64SUBSD:
		return rewriteValueAMD64_OpAMD64SUBSD(v)
	case OpAMD64SUBSDload:
		return rewriteValueAMD64_OpAMD64SUBSDload(v)
	case OpAMD64SUBSS:
		return rewriteValueAMD64_OpAMD64SUBSS(v)
	case OpAMD64SUBSSload:
		return rewriteValueAMD64_OpAMD64SUBSSload(v)
	case OpAMD64TESTB:
		return rewriteValueAMD64_OpAMD64TESTB(v)
	case OpAMD64TESTBconst:
		return rewriteValueAMD64_OpAMD64TESTBconst(v)
	case OpAMD64TESTL:
		return rewriteValueAMD64_OpAMD64TESTL(v)
	case OpAMD64TESTLconst:
		return rewriteValueAMD64_OpAMD64TESTLconst(v)
	case OpAMD64TESTQ:
		return rewriteValueAMD64_OpAMD64TESTQ(v)
	case OpAMD64TESTQconst:
		return rewriteValueAMD64_OpAMD64TESTQconst(v)
	case OpAMD64TESTW:
		return rewriteValueAMD64_OpAMD64TESTW(v)
	case OpAMD64TESTWconst:
		return rewriteValueAMD64_OpAMD64TESTWconst(v)
	case OpAMD64VPANDQ512:
		return rewriteValueAMD64_OpAMD64VPANDQ512(v)
	case OpAMD64VPMOVVec16x16ToM:
		return rewriteValueAMD64_OpAMD64VPMOVVec16x16ToM(v)
	case OpAMD64VPMOVVec16x32ToM:
		return rewriteValueAMD64_OpAMD64VPMOVVec16x32ToM(v)
	case OpAMD64VPMOVVec16x8ToM:
		return rewriteValueAMD64_OpAMD64VPMOVVec16x8ToM(v)
	case OpAMD64VPMOVVec32x16ToM:
		return rewriteValueAMD64_OpAMD64VPMOVVec32x16ToM(v)
	case OpAMD64VPMOVVec32x4ToM:
		return rewriteValueAMD64_OpAMD64VPMOVVec32x4ToM(v)
	case OpAMD64VPMOVVec32x8ToM:
		return rewriteValueAMD64_OpAMD64VPMOVVec32x8ToM(v)
	case OpAMD64VPMOVVec64x2ToM:
		return rewriteValueAMD64_OpAMD64VPMOVVec64x2ToM(v)
	case OpAMD64VPMOVVec64x4ToM:
		return rewriteValueAMD64_OpAMD64VPMOVVec64x4ToM(v)
	case OpAMD64VPMOVVec64x8ToM:
		return rewriteValueAMD64_OpAMD64VPMOVVec64x8ToM(v)
	case OpAMD64VPMOVVec8x16ToM:
		return rewriteValueAMD64_OpAMD64VPMOVVec8x16ToM(v)
	case OpAMD64VPMOVVec8x32ToM:
		return rewriteValueAMD64_OpAMD64VPMOVVec8x32ToM(v)
	case OpAMD64VPMOVVec8x64ToM:
		return rewriteValueAMD64_OpAMD64VPMOVVec8x64ToM(v)
	case OpAMD64VPSLLD128:
		return rewriteValueAMD64_OpAMD64VPSLLD128(v)
	case OpAMD64VPSLLD256:
		return rewriteValueAMD64_OpAMD64VPSLLD256(v)
	case OpAMD64VPSLLD512:
		return rewriteValueAMD64_OpAMD64VPSLLD512(v)
	case OpAMD64VPSLLQ128:
		return rewriteValueAMD64_OpAMD64VPSLLQ128(v)
	case OpAMD64VPSLLQ256:
		return rewriteValueAMD64_OpAMD64VPSLLQ256(v)
	case OpAMD64VPSLLQ512:
		return rewriteValueAMD64_OpAMD64VPSLLQ512(v)
	case OpAMD64VPSLLW128:
		return rewriteValueAMD64_OpAMD64VPSLLW128(v)
	case OpAMD64VPSLLW256:
		return rewriteValueAMD64_OpAMD64VPSLLW256(v)
	case OpAMD64VPSLLW512:
		return rewriteValueAMD64_OpAMD64VPSLLW512(v)
	case OpAMD64VPSRAD128:
		return rewriteValueAMD64_OpAMD64VPSRAD128(v)
	case OpAMD64VPSRAD256:
		return rewriteValueAMD64_OpAMD64VPSRAD256(v)
	case OpAMD64VPSRAD512:
		return rewriteValueAMD64_OpAMD64VPSRAD512(v)
	case OpAMD64VPSRAQ128:
		return rewriteValueAMD64_OpAMD64VPSRAQ128(v)
	case OpAMD64VPSRAQ256:
		return rewriteValueAMD64_OpAMD64VPSRAQ256(v)
	case OpAMD64VPSRAQ512:
		return rewriteValueAMD64_OpAMD64VPSRAQ512(v)
	case OpAMD64VPSRAW128:
		return rewriteValueAMD64_OpAMD64VPSRAW128(v)
	case OpAMD64VPSRAW256:
		return rewriteValueAMD64_OpAMD64VPSRAW256(v)
	case OpAMD64VPSRAW512:
		return rewriteValueAMD64_OpAMD64VPSRAW512(v)
	case OpAMD64XADDLlock:
		return rewriteValueAMD64_OpAMD64XADDLlock(v)
	case OpAMD64XADDQlock:
		return rewriteValueAMD64_OpAMD64XADDQlock(v)
	case OpAMD64XCHGL:
		return rewriteValueAMD64_OpAMD64XCHGL(v)
	case OpAMD64XCHGQ:
		return rewriteValueAMD64_OpAMD64XCHGQ(v)
	case OpAMD64XORL:
		return rewriteValueAMD64_OpAMD64XORL(v)
	case OpAMD64XORLconst:
		return rewriteValueAMD64_OpAMD64XORLconst(v)
	case OpAMD64XORLconstmodify:
		return rewriteValueAMD64_OpAMD64XORLconstmodify(v)
	case OpAMD64XORLload:
		return rewriteValueAMD64_OpAMD64XORLload(v)
	case OpAMD64XORLmodify:
		return rewriteValueAMD64_OpAMD64XORLmodify(v)
	case OpAMD64XORQ:
		return rewriteValueAMD64_OpAMD64XORQ(v)
	case OpAMD64XORQconst:
		return rewriteValueAMD64_OpAMD64XORQconst(v)
	case OpAMD64XORQconstmodify:
		return rewriteValueAMD64_OpAMD64XORQconstmodify(v)
	case OpAMD64XORQload:
		return rewriteValueAMD64_OpAMD64XORQload(v)
	case OpAMD64XORQmodify:
		return rewriteValueAMD64_OpAMD64XORQmodify(v)
	case OpAbsInt16x16:
		v.Op = OpAMD64VPABSW256
		return true
	case OpAbsInt16x32:
		v.Op = OpAMD64VPABSW512
		return true
	case OpAbsInt16x8:
		v.Op = OpAMD64VPABSW128
		return true
	case OpAbsInt32x16:
		v.Op = OpAMD64VPABSD512
		return true
	case OpAbsInt32x4:
		v.Op = OpAMD64VPABSD128
		return true
	case OpAbsInt32x8:
		v.Op = OpAMD64VPABSD256
		return true
	case OpAbsInt64x2:
		v.Op = OpAMD64VPABSQ128
		return true
	case OpAbsInt64x4:
		v.Op = OpAMD64VPABSQ256
		return true
	case OpAbsInt64x8:
		v.Op = OpAMD64VPABSQ512
		return true
	case OpAbsInt8x16:
		v.Op = OpAMD64VPABSB128
		return true
	case OpAbsInt8x32:
		v.Op = OpAMD64VPABSB256
		return true
	case OpAbsInt8x64:
		v.Op = OpAMD64VPABSB512
		return true
	case OpAdd16:
		v.Op = OpAMD64ADDL
		return true
	case OpAdd32:
		v.Op = OpAMD64ADDL
		return true
	case OpAdd32F:
		v.Op = OpAMD64ADDSS
		return true
	case OpAdd64:
		v.Op = OpAMD64ADDQ
		return true
	case OpAdd64F:
		v.Op = OpAMD64ADDSD
		return true
	case OpAdd8:
		v.Op = OpAMD64ADDL
		return true
	case OpAddDotProdPairsSaturatedInt32x16:
		v.Op = OpAMD64VPDPWSSDS512
		return true
	case OpAddDotProdPairsSaturatedInt32x4:
		v.Op = OpAMD64VPDPWSSDS128
		return true
	case OpAddDotProdPairsSaturatedInt32x8:
		v.Op = OpAMD64VPDPWSSDS256
		return true
	case OpAddDotProdQuadrupleInt32x16:
		v.Op = OpAMD64VPDPBUSD512
		return true
	case OpAddDotProdQuadrupleInt32x4:
		v.Op = OpAMD64VPDPBUSD128
		return true
	case OpAddDotProdQuadrupleInt32x8:
		v.Op = OpAMD64VPDPBUSD256
		return true
	case OpAddDotProdQuadrupleSaturatedInt32x16:
		v.Op = OpAMD64VPDPBUSDS512
		return true
	case OpAddDotProdQuadrupleSaturatedInt32x4:
		v.Op = OpAMD64VPDPBUSDS128
		return true
	case OpAddDotProdQuadrupleSaturatedInt32x8:
		v.Op = OpAMD64VPDPBUSDS256
		return true
	case OpAddFloat32x16:
		v.Op = OpAMD64VADDPS512
		return true
	case OpAddFloat32x4:
		v.Op = OpAMD64VADDPS128
		return true
	case OpAddFloat32x8:
		v.Op = OpAMD64VADDPS256
		return true
	case OpAddFloat64x2:
		v.Op = OpAMD64VADDPD128
		return true
	case OpAddFloat64x4:
		v.Op = OpAMD64VADDPD256
		return true
	case OpAddFloat64x8:
		v.Op = OpAMD64VADDPD512
		return true
	case OpAddInt16x16:
		v.Op = OpAMD64VPADDW256
		return true
	case OpAddInt16x32:
		v.Op = OpAMD64VPADDW512
		return true
	case OpAddInt16x8:
		v.Op = OpAMD64VPADDW128
		return true
	case OpAddInt32x16:
		v.Op = OpAMD64VPADDD512
		return true
	case OpAddInt32x4:
		v.Op = OpAMD64VPADDD128
		return true
	case OpAddInt32x8:
		v.Op = OpAMD64VPADDD256
		return true
	case OpAddInt64x2:
		v.Op = OpAMD64VPADDQ128
		return true
	case OpAddInt64x4:
		v.Op = OpAMD64VPADDQ256
		return true
	case OpAddInt64x8:
		v.Op = OpAMD64VPADDQ512
		return true
	case OpAddInt8x16:
		v.Op = OpAMD64VPADDB128
		return true
	case OpAddInt8x32:
		v.Op = OpAMD64VPADDB256
		return true
	case OpAddInt8x64:
		v.Op = OpAMD64VPADDB512
		return true
	case OpAddPairsFloat32x4:
		v.Op = OpAMD64VHADDPS128
		return true
	case OpAddPairsFloat32x8:
		v.Op = OpAMD64VHADDPS256
		return true
	case OpAddPairsFloat64x2:
		v.Op = OpAMD64VHADDPD128
		return true
	case OpAddPairsFloat64x4:
		v.Op = OpAMD64VHADDPD256
		return true
	case OpAddPairsInt16x16:
		v.Op = OpAMD64VPHADDW256
		return true
	case OpAddPairsInt16x8:
		v.Op = OpAMD64VPHADDW128
		return true
	case OpAddPairsInt32x4:
		v.Op = OpAMD64VPHADDD128
		return true
	case OpAddPairsInt32x8:
		v.Op = OpAMD64VPHADDD256
		return true
	case OpAddPairsSaturatedInt16x16:
		v.Op = OpAMD64VPHADDSW256
		return true
	case OpAddPairsSaturatedInt16x8:
		v.Op = OpAMD64VPHADDSW128
		return true
	case OpAddPairsUint16x16:
		v.Op = OpAMD64VPHADDW256
		return true
	case OpAddPairsUint16x8:
		v.Op = OpAMD64VPHADDW128
		return true
	case OpAddPairsUint32x4:
		v.Op = OpAMD64VPHADDD128
		return true
	case OpAddPairsUint32x8:
		v.Op = OpAMD64VPHADDD256
		return true
	case OpAddPtr:
		v.Op = OpAMD64ADDQ
		return true
	case OpAddSaturatedInt16x16:
		v.Op = OpAMD64VPADDSW256
		return true
	case OpAddSaturatedInt16x32:
		v.Op = OpAMD64VPADDSW512
		return true
	case OpAddSaturatedInt16x8:
		v.Op = OpAMD64VPADDSW128
		return true
	case OpAddSaturatedInt8x16:
		v.Op = OpAMD64VPADDSB128
		return true
	case OpAddSaturatedInt8x32:
		v.Op = OpAMD64VPADDSB256
		return true
	case OpAddSaturatedInt8x64:
		v.Op = OpAMD64VPADDSB512
		return true
	case OpAddSaturatedUint16x16:
		v.Op = OpAMD64VPADDUSW256
		return true
	case OpAddSaturatedUint16x32:
		v.Op = OpAMD64VPADDUSW512
		return true
	case OpAddSaturatedUint16x8:
		v.Op = OpAMD64VPADDUSW128
		return true
	case OpAddSaturatedUint8x16:
		v.Op = OpAMD64VPADDUSB128
		return true
	case OpAddSaturatedUint8x32:
		v.Op = OpAMD64VPADDUSB256
		return true
	case OpAddSaturatedUint8x64:
		v.Op = OpAMD64VPADDUSB512
		return true
	case OpAddSubFloat32x4:
		v.Op = OpAMD64VADDSUBPS128
		return true
	case OpAddSubFloat32x8:
		v.Op = OpAMD64VADDSUBPS256
		return true
	case OpAddSubFloat64x2:
		v.Op = OpAMD64VADDSUBPD128
		return true
	case OpAddSubFloat64x4:
		v.Op = OpAMD64VADDSUBPD256
		return true
	case OpAddUint16x16:
		v.Op = OpAMD64VPADDW256
		return true
	case OpAddUint16x32:
		v.Op = OpAMD64VPADDW512
		return true
	case OpAddUint16x8:
		v.Op = OpAMD64VPADDW128
		return true
	case OpAddUint32x16:
		v.Op = OpAMD64VPADDD512
		return true
	case OpAddUint32x4:
		v.Op = OpAMD64VPADDD128
		return true
	case OpAddUint32x8:
		v.Op = OpAMD64VPADDD256
		return true
	case OpAddUint64x2:
		v.Op = OpAMD64VPADDQ128
		return true
	case OpAddUint64x4:
		v.Op = OpAMD64VPADDQ256
		return true
	case OpAddUint64x8:
		v.Op = OpAMD64VPADDQ512
		return true
	case OpAddUint8x16:
		v.Op = OpAMD64VPADDB128
		return true
	case OpAddUint8x32:
		v.Op = OpAMD64VPADDB256
		return true
	case OpAddUint8x64:
		v.Op = OpAMD64VPADDB512
		return true
	case OpAddr:
		return rewriteValueAMD64_OpAddr(v)
	case OpAnd16:
		v.Op = OpAMD64ANDL
		return true
	case OpAnd32:
		v.Op = OpAMD64ANDL
		return true
	case OpAnd64:
		v.Op = OpAMD64ANDQ
		return true
	case OpAnd8:
		v.Op = OpAMD64ANDL
		return true
	case OpAndB:
		v.Op = OpAMD64ANDL
		return true
	case OpAndInt16x16:
		v.Op = OpAMD64VPAND256
		return true
	case OpAndInt16x32:
		v.Op = OpAMD64VPANDD512
		return true
	case OpAndInt16x8:
		v.Op = OpAMD64VPAND128
		return true
	case OpAndInt32x16:
		v.Op = OpAMD64VPANDD512
		return true
	case OpAndInt32x4:
		v.Op = OpAMD64VPAND128
		return true
	case OpAndInt32x8:
		v.Op = OpAMD64VPAND256
		return true
	case OpAndInt64x2:
		v.Op = OpAMD64VPAND128
		return true
	case OpAndInt64x4:
		v.Op = OpAMD64VPAND256
		return true
	case OpAndInt64x8:
		v.Op = OpAMD64VPANDQ512
		return true
	case OpAndInt8x16:
		v.Op = OpAMD64VPAND128
		return true
	case OpAndInt8x32:
		v.Op = OpAMD64VPAND256
		return true
	case OpAndInt8x64:
		v.Op = OpAMD64VPANDD512
		return true
	case OpAndNotInt16x16:
		v.Op = OpAMD64VPANDN256
		return true
	case OpAndNotInt16x32:
		v.Op = OpAMD64VPANDND512
		return true
	case OpAndNotInt16x8:
		v.Op = OpAMD64VPANDN128
		return true
	case OpAndNotInt32x16:
		v.Op = OpAMD64VPANDND512
		return true
	case OpAndNotInt32x4:
		v.Op = OpAMD64VPANDN128
		return true
	case OpAndNotInt32x8:
		v.Op = OpAMD64VPANDN256
		return true
	case OpAndNotInt64x2:
		v.Op = OpAMD64VPANDN128
		return true
	case OpAndNotInt64x4:
		v.Op = OpAMD64VPANDN256
		return true
	case OpAndNotInt64x8:
		v.Op = OpAMD64VPANDNQ512
		return true
	case OpAndNotInt8x16:
		v.Op = OpAMD64VPANDN128
		return true
	case OpAndNotInt8x32:
		v.Op = OpAMD64VPANDN256
		return true
	case OpAndNotInt8x64:
		v.Op = OpAMD64VPANDND512
		return true
	case OpAndNotUint16x16:
		v.Op = OpAMD64VPANDN256
		return true
	case OpAndNotUint16x32:
		v.Op = OpAMD64VPANDND512
		return true
	case OpAndNotUint16x8:
		v.Op = OpAMD64VPANDN128
		return true
	case OpAndNotUint32x16:
		v.Op = OpAMD64VPANDND512
		return true
	case OpAndNotUint32x4:
		v.Op = OpAMD64VPANDN128
		return true
	case OpAndNotUint32x8:
		v.Op = OpAMD64VPANDN256
		return true
	case OpAndNotUint64x2:
		v.Op = OpAMD64VPANDN128
		return true
	case OpAndNotUint64x4:
		v.Op = OpAMD64VPANDN256
		return true
	case OpAndNotUint64x8:
		v.Op = OpAMD64VPANDNQ512
		return true
	case OpAndNotUint8x16:
		v.Op = OpAMD64VPANDN128
		return true
	case OpAndNotUint8x32:
		v.Op = OpAMD64VPANDN256
		return true
	case OpAndNotUint8x64:
		v.Op = OpAMD64VPANDND512
		return true
	case OpAndUint16x16:
		v.Op = OpAMD64VPAND256
		return true
	case OpAndUint16x32:
		v.Op = OpAMD64VPANDD512
		return true
	case OpAndUint16x8:
		v.Op = OpAMD64VPAND128
		return true
	case OpAndUint32x16:
		v.Op = OpAMD64VPANDD512
		return true
	case OpAndUint32x4:
		v.Op = OpAMD64VPAND128
		return true
	case OpAndUint32x8:
		v.Op = OpAMD64VPAND256
		return true
	case OpAndUint64x2:
		v.Op = OpAMD64VPAND128
		return true
	case OpAndUint64x4:
		v.Op = OpAMD64VPAND256
		return true
	case OpAndUint64x8:
		v.Op = OpAMD64VPANDQ512
		return true
	case OpAndUint8x16:
		v.Op = OpAMD64VPAND128
		return true
	case OpAndUint8x32:
		v.Op = OpAMD64VPAND256
		return true
	case OpAndUint8x64:
		v.Op = OpAMD64VPANDD512
		return true
	case OpAtomicAdd32:
		return rewriteValueAMD64_OpAtomicAdd32(v)
	case OpAtomicAdd64:
		return rewriteValueAMD64_OpAtomicAdd64(v)
	case OpAtomicAnd32:
		return rewriteValueAMD64_OpAtomicAnd32(v)
	case OpAtomicAnd32value:
		return rewriteValueAMD64_OpAtomicAnd32value(v)
	case OpAtomicAnd64value:
		return rewriteValueAMD64_OpAtomicAnd64value(v)
	case OpAtomicAnd8:
		return rewriteValueAMD64_OpAtomicAnd8(v)
	case OpAtomicCompareAndSwap32:
		return rewriteValueAMD64_OpAtomicCompareAndSwap32(v)
	case OpAtomicCompareAndSwap64:
		return rewriteValueAMD64_OpAtomicCompareAndSwap64(v)
	case OpAtomicExchange32:
		return rewriteValueAMD64_OpAtomicExchange32(v)
	case OpAtomicExchange64:
		return rewriteValueAMD64_OpAtomicExchange64(v)
	case OpAtomicExchange8:
		return rewriteValueAMD64_OpAtomicExchange8(v)
	case OpAtomicLoad32:
		return rewriteValueAMD64_OpAtomicLoad32(v)
	case OpAtomicLoad64:
		return rewriteValueAMD64_OpAtomicLoad64(v)
	case OpAtomicLoad8:
		return rewriteValueAMD64_OpAtomicLoad8(v)
	case OpAtomicLoadPtr:
		return rewriteValueAMD64_OpAtomicLoadPtr(v)
	case OpAtomicOr32:
		return rewriteValueAMD64_OpAtomicOr32(v)
	case OpAtomicOr32value:
		return rewriteValueAMD64_OpAtomicOr32value(v)
	case OpAtomicOr64value:
		return rewriteValueAMD64_OpAtomicOr64value(v)
	case OpAtomicOr8:
		return rewriteValueAMD64_OpAtomicOr8(v)
	case OpAtomicStore32:
		return rewriteValueAMD64_OpAtomicStore32(v)
	case OpAtomicStore64:
		return rewriteValueAMD64_OpAtomicStore64(v)
	case OpAtomicStore8:
		return rewriteValueAMD64_OpAtomicStore8(v)
	case OpAtomicStorePtrNoWB:
		return rewriteValueAMD64_OpAtomicStorePtrNoWB(v)
	case OpAverageUint16x16:
		v.Op = OpAMD64VPAVGW256
		return true
	case OpAverageUint16x32:
		v.Op = OpAMD64VPAVGW512
		return true
	case OpAverageUint16x8:
		v.Op = OpAMD64VPAVGW128
		return true
	case OpAverageUint8x16:
		v.Op = OpAMD64VPAVGB128
		return true
	case OpAverageUint8x32:
		v.Op = OpAMD64VPAVGB256
		return true
	case OpAverageUint8x64:
		v.Op = OpAMD64VPAVGB512
		return true
	case OpAvg64u:
		v.Op = OpAMD64AVGQU
		return true
	case OpBitLen16:
		return rewriteValueAMD64_OpBitLen16(v)
	case OpBitLen32:
		return rewriteValueAMD64_OpBitLen32(v)
	case OpBitLen64:
		return rewriteValueAMD64_OpBitLen64(v)
	case OpBitLen8:
		return rewriteValueAMD64_OpBitLen8(v)
	case OpBroadcast128Float32x4:
		v.Op = OpAMD64VBROADCASTSS128
		return true
	case OpBroadcast128Float64x2:
		v.Op = OpAMD64VPBROADCASTQ128
		return true
	case OpBroadcast128Int16x8:
		v.Op = OpAMD64VPBROADCASTW128
		return true
	case OpBroadcast128Int32x4:
		v.Op = OpAMD64VPBROADCASTD128
		return true
	case OpBroadcast128Int64x2:
		v.Op = OpAMD64VPBROADCASTQ128
		return true
	case OpBroadcast128Int8x16:
		v.Op = OpAMD64VPBROADCASTB128
		return true
	case OpBroadcast128Uint16x8:
		v.Op = OpAMD64VPBROADCASTW128
		return true
	case OpBroadcast128Uint32x4:
		v.Op = OpAMD64VPBROADCASTD128
		return true
	case OpBroadcast128Uint64x2:
		v.Op = OpAMD64VPBROADCASTQ128
		return true
	case OpBroadcast128Uint8x16:
		v.Op = OpAMD64VPBROADCASTB128
		return true
	case OpBroadcast256Float32x4:
		v.Op = OpAMD64VBROADCASTSS256
		return true
	case OpBroadcast256Float64x2:
		v.Op = OpAMD64VBROADCASTSD256
		return true
	case OpBroadcast256Int16x8:
		v.Op = OpAMD64VPBROADCASTW256
		return true
	case OpBroadcast256Int32x4:
		v.Op = OpAMD64VPBROADCASTD256
		return true
	case OpBroadcast256Int64x2:
		v.Op = OpAMD64VPBROADCASTQ256
		return true
	case OpBroadcast256Int8x16:
		v.Op = OpAMD64VPBROADCASTB256
		return true
	case OpBroadcast256Uint16x8:
		v.Op = OpAMD64VPBROADCASTW256
		return true
	case OpBroadcast256Uint32x4:
		v.Op = OpAMD64VPBROADCASTD256
		return true
	case OpBroadcast256Uint64x2:
		v.Op = OpAMD64VPBROADCASTQ256
		return true
	case OpBroadcast256Uint8x16:
		v.Op = OpAMD64VPBROADCASTB256
		return true
	case OpBroadcast512Float32x4:
		v.Op = OpAMD64VBROADCASTSS512
		return true
	case OpBroadcast512Float64x2:
		v.Op = OpAMD64VBROADCASTSD512
		return true
	case OpBroadcast512Int16x8:
		v.Op = OpAMD64VPBROADCASTW512
		return true
	case OpBroadcast512Int32x4:
		v.Op = OpAMD64VPBROADCASTD512
		return true
	case OpBroadcast512Int64x2:
		v.Op = OpAMD64VPBROADCASTQ512
		return true
	case OpBroadcast512Int8x16:
		v.Op = OpAMD64VPBROADCASTB512
		return true
	case OpBroadcast512Uint16x8:
		v.Op = OpAMD64VPBROADCASTW512
		return true
	case OpBroadcast512Uint32x4:
		v.Op = OpAMD64VPBROADCASTD512
		return true
	case OpBroadcast512Uint64x2:
		v.Op = OpAMD64VPBROADCASTQ512
		return true
	case OpBroadcast512Uint8x16:
		v.Op = OpAMD64VPBROADCASTB512
		return true
	case OpBswap16:
		return rewriteValueAMD64_OpBswap16(v)
	case OpBswap32:
		v.Op = OpAMD64BSWAPL
		return true
	case OpBswap64:
		v.Op = OpAMD64BSWAPQ
		return true
	case OpCeil:
		return rewriteValueAMD64_OpCeil(v)
	case OpCeilFloat32x4:
		return rewriteValueAMD64_OpCeilFloat32x4(v)
	case OpCeilFloat32x8:
		return rewriteValueAMD64_OpCeilFloat32x8(v)
	case OpCeilFloat64x2:
		return rewriteValueAMD64_OpCeilFloat64x2(v)
	case OpCeilFloat64x4:
		return rewriteValueAMD64_OpCeilFloat64x4(v)
	case OpCeilScaledFloat32x16:
		return rewriteValueAMD64_OpCeilScaledFloat32x16(v)
	case OpCeilScaledFloat32x4:
		return rewriteValueAMD64_OpCeilScaledFloat32x4(v)
	case OpCeilScaledFloat32x8:
		return rewriteValueAMD64_OpCeilScaledFloat32x8(v)
	case OpCeilScaledFloat64x2:
		return rewriteValueAMD64_OpCeilScaledFloat64x2(v)
	case OpCeilScaledFloat64x4:
		return rewriteValueAMD64_OpCeilScaledFloat64x4(v)
	case OpCeilScaledFloat64x8:
		return rewriteValueAMD64_OpCeilScaledFloat64x8(v)
	case OpCeilScaledResidueFloat32x16:
		return rewriteValueAMD64_OpCeilScaledResidueFloat32x16(v)
	case OpCeilScaledResidueFloat32x4:
		return rewriteValueAMD64_OpCeilScaledResidueFloat32x4(v)
	case OpCeilScaledResidueFloat32x8:
		return rewriteValueAMD64_OpCeilScaledResidueFloat32x8(v)
	case OpCeilScaledResidueFloat64x2:
		return rewriteValueAMD64_OpCeilScaledResidueFloat64x2(v)
	case OpCeilScaledResidueFloat64x4:
		return rewriteValueAMD64_OpCeilScaledResidueFloat64x4(v)
	case OpCeilScaledResidueFloat64x8:
		return rewriteValueAMD64_OpCeilScaledResidueFloat64x8(v)
	case OpClosureCall:
		v.Op = OpAMD64CALLclosure
		return true
	case OpCom16:
		v.Op = OpAMD64NOTL
		return true
	case OpCom32:
		v.Op = OpAMD64NOTL
		return true
	case OpCom64:
		v.Op = OpAMD64NOTQ
		return true
	case OpCom8:
		v.Op = OpAMD64NOTL
		return true
	case OpCompressFloat32x16:
		return rewriteValueAMD64_OpCompressFloat32x16(v)
	case OpCompressFloat32x4:
		return rewriteValueAMD64_OpCompressFloat32x4(v)
	case OpCompressFloat32x8:
		return rewriteValueAMD64_OpCompressFloat32x8(v)
	case OpCompressFloat64x2:
		return rewriteValueAMD64_OpCompressFloat64x2(v)
	case OpCompressFloat64x4:
		return rewriteValueAMD64_OpCompressFloat64x4(v)
	case OpCompressFloat64x8:
		return rewriteValueAMD64_OpCompressFloat64x8(v)
	case OpCompressInt16x16:
		return rewriteValueAMD64_OpCompressInt16x16(v)
	case OpCompressInt16x32:
		return rewriteValueAMD64_OpCompressInt16x32(v)
	case OpCompressInt16x8:
		return rewriteValueAMD64_OpCompressInt16x8(v)
	case OpCompressInt32x16:
		return rewriteValueAMD64_OpCompressInt32x16(v)
	case OpCompressInt32x4:
		return rewriteValueAMD64_OpCompressInt32x4(v)
	case OpCompressInt32x8:
		return rewriteValueAMD64_OpCompressInt32x8(v)
	case OpCompressInt64x2:
		return rewriteValueAMD64_OpCompressInt64x2(v)
	case OpCompressInt64x4:
		return rewriteValueAMD64_OpCompressInt64x4(v)
	case OpCompressInt64x8:
		return rewriteValueAMD64_OpCompressInt64x8(v)
	case OpCompressInt8x16:
		return rewriteValueAMD64_OpCompressInt8x16(v)
	case OpCompressInt8x32:
		return rewriteValueAMD64_OpCompressInt8x32(v)
	case OpCompressInt8x64:
		return rewriteValueAMD64_OpCompressInt8x64(v)
	case OpCompressUint16x16:
		return rewriteValueAMD64_OpCompressUint16x16(v)
	case OpCompressUint16x32:
		return rewriteValueAMD64_OpCompressUint16x32(v)
	case OpCompressUint16x8:
		return rewriteValueAMD64_OpCompressUint16x8(v)
	case OpCompressUint32x16:
		return rewriteValueAMD64_OpCompressUint32x16(v)
	case OpCompressUint32x4:
		return rewriteValueAMD64_OpCompressUint32x4(v)
	case OpCompressUint32x8:
		return rewriteValueAMD64_OpCompressUint32x8(v)
	case OpCompressUint64x2:
		return rewriteValueAMD64_OpCompressUint64x2(v)
	case OpCompressUint64x4:
		return rewriteValueAMD64_OpCompressUint64x4(v)
	case OpCompressUint64x8:
		return rewriteValueAMD64_OpCompressUint64x8(v)
	case OpCompressUint8x16:
		return rewriteValueAMD64_OpCompressUint8x16(v)
	case OpCompressUint8x32:
		return rewriteValueAMD64_OpCompressUint8x32(v)
	case OpCompressUint8x64:
		return rewriteValueAMD64_OpCompressUint8x64(v)
	case OpCondSelect:
		return rewriteValueAMD64_OpCondSelect(v)
	case OpConst16:
		return rewriteValueAMD64_OpConst16(v)
	case OpConst32:
		v.Op = OpAMD64MOVLconst
		return true
	case OpConst32F:
		v.Op = OpAMD64MOVSSconst
		return true
	case OpConst64:
		v.Op = OpAMD64MOVQconst
		return true
	case OpConst64F:
		v.Op = OpAMD64MOVSDconst
		return true
	case OpConst8:
		return rewriteValueAMD64_OpConst8(v)
	case OpConstBool:
		return rewriteValueAMD64_OpConstBool(v)
	case OpConstNil:
		return rewriteValueAMD64_OpConstNil(v)
	case OpConvertToInt32Float32x16:
		v.Op = OpAMD64VCVTTPS2DQ512
		return true
	case OpConvertToInt32Float32x4:
		v.Op = OpAMD64VCVTTPS2DQ128
		return true
	case OpConvertToInt32Float32x8:
		v.Op = OpAMD64VCVTTPS2DQ256
		return true
	case OpConvertToUint16Uint8x16:
		v.Op = OpAMD64VPMOVZXBW256
		return true
	case OpConvertToUint16Uint8x32:
		v.Op = OpAMD64VPMOVZXBW512
		return true
	case OpConvertToUint16x8Uint8x16:
		v.Op = OpAMD64VPMOVZXBW128
		return true
	case OpConvertToUint32Float32x16:
		v.Op = OpAMD64VCVTPS2UDQ512
		return true
	case OpConvertToUint32Float32x4:
		v.Op = OpAMD64VCVTPS2UDQ128
		return true
	case OpConvertToUint32Float32x8:
		v.Op = OpAMD64VCVTPS2UDQ256
		return true
	case OpConvertToUint32Uint16x16:
		v.Op = OpAMD64VPMOVZXWD512
		return true
	case OpConvertToUint32Uint16x8:
		v.Op = OpAMD64VPMOVZXWD256
		return true
	case OpConvertToUint32x4Uint16x8:
		v.Op = OpAMD64VPMOVZXWD128
		return true
	case OpCopySignInt16x16:
		v.Op = OpAMD64VPSIGNW256
		return true
	case OpCopySignInt16x8:
		v.Op = OpAMD64VPSIGNW128
		return true
	case OpCopySignInt32x4:
		v.Op = OpAMD64VPSIGND128
		return true
	case OpCopySignInt32x8:
		v.Op = OpAMD64VPSIGND256
		return true
	case OpCopySignInt8x16:
		v.Op = OpAMD64VPSIGNB128
		return true
	case OpCopySignInt8x32:
		v.Op = OpAMD64VPSIGNB256
		return true
	case OpCtz16:
		return rewriteValueAMD64_OpCtz16(v)
	case OpCtz16NonZero:
		return rewriteValueAMD64_OpCtz16NonZero(v)
	case OpCtz32:
		return rewriteValueAMD64_OpCtz32(v)
	case OpCtz32NonZero:
		return rewriteValueAMD64_OpCtz32NonZero(v)
	case OpCtz64:
		return rewriteValueAMD64_OpCtz64(v)
	case OpCtz64NonZero:
		return rewriteValueAMD64_OpCtz64NonZero(v)
	case OpCtz8:
		return rewriteValueAMD64_OpCtz8(v)
	case OpCtz8NonZero:
		return rewriteValueAMD64_OpCtz8NonZero(v)
	case OpCvt16toMask16x16:
		return rewriteValueAMD64_OpCvt16toMask16x16(v)
	case OpCvt16toMask32x16:
		return rewriteValueAMD64_OpCvt16toMask32x16(v)
	case OpCvt16toMask8x16:
		return rewriteValueAMD64_OpCvt16toMask8x16(v)
	case OpCvt32Fto32:
		v.Op = OpAMD64CVTTSS2SL
		return true
	case OpCvt32Fto64:
		v.Op = OpAMD64CVTTSS2SQ
		return true
	case OpCvt32Fto64F:
		v.Op = OpAMD64CVTSS2SD
		return true
	case OpCvt32to32F:
		v.Op = OpAMD64CVTSL2SS
		return true
	case OpCvt32to64F:
		v.Op = OpAMD64CVTSL2SD
		return true
	case OpCvt32toMask16x32:
		return rewriteValueAMD64_OpCvt32toMask16x32(v)
	case OpCvt32toMask8x32:
		return rewriteValueAMD64_OpCvt32toMask8x32(v)
	case OpCvt64Fto32:
		v.Op = OpAMD64CVTTSD2SL
		return true
	case OpCvt64Fto32F:
		v.Op = OpAMD64CVTSD2SS
		return true
	case OpCvt64Fto64:
		v.Op = OpAMD64CVTTSD2SQ
		return true
	case OpCvt64to32F:
		v.Op = OpAMD64CVTSQ2SS
		return true
	case OpCvt64to64F:
		v.Op = OpAMD64CVTSQ2SD
		return true
	case OpCvt64toMask8x64:
		return rewriteValueAMD64_OpCvt64toMask8x64(v)
	case OpCvt8toMask16x8:
		return rewriteValueAMD64_OpCvt8toMask16x8(v)
	case OpCvt8toMask32x4:
		return rewriteValueAMD64_OpCvt8toMask32x4(v)
	case OpCvt8toMask32x8:
		return rewriteValueAMD64_OpCvt8toMask32x8(v)
	case OpCvt8toMask64x2:
		return rewriteValueAMD64_OpCvt8toMask64x2(v)
	case OpCvt8toMask64x4:
		return rewriteValueAMD64_OpCvt8toMask64x4(v)
	case OpCvt8toMask64x8:
		return rewriteValueAMD64_OpCvt8toMask64x8(v)
	case OpCvtBoolToUint8:
		v.Op = OpCopy
		return true
	case OpCvtMask16x16to16:
		return rewriteValueAMD64_OpCvtMask16x16to16(v)
	case OpCvtMask16x32to32:
		return rewriteValueAMD64_OpCvtMask16x32to32(v)
	case OpCvtMask16x8to8:
		return rewriteValueAMD64_OpCvtMask16x8to8(v)
	case OpCvtMask32x16to16:
		return rewriteValueAMD64_OpCvtMask32x16to16(v)
	case OpCvtMask32x4to8:
		return rewriteValueAMD64_OpCvtMask32x4to8(v)
	case OpCvtMask32x8to8:
		return rewriteValueAMD64_OpCvtMask32x8to8(v)
	case OpCvtMask64x2to8:
		return rewriteValueAMD64_OpCvtMask64x2to8(v)
	case OpCvtMask64x4to8:
		return rewriteValueAMD64_OpCvtMask64x4to8(v)
	case OpCvtMask64x8to8:
		return rewriteValueAMD64_OpCvtMask64x8to8(v)
	case OpCvtMask8x16to16:
		return rewriteValueAMD64_OpCvtMask8x16to16(v)
	case OpCvtMask8x32to32:
		return rewriteValueAMD64_OpCvtMask8x32to32(v)
	case OpCvtMask8x64to64:
		return rewriteValueAMD64_OpCvtMask8x64to64(v)
	case OpDiv128u:
		v.Op = OpAMD64DIVQU2
		return true
	case OpDiv16:
		return rewriteValueAMD64_OpDiv16(v)
	case OpDiv16u:
		return rewriteValueAMD64_OpDiv16u(v)
	case OpDiv32:
		return rewriteValueAMD64_OpDiv32(v)
	case OpDiv32F:
		v.Op = OpAMD64DIVSS
		return true
	case OpDiv32u:
		return rewriteValueAMD64_OpDiv32u(v)
	case OpDiv64:
		return rewriteValueAMD64_OpDiv64(v)
	case OpDiv64F:
		v.Op = OpAMD64DIVSD
		return true
	case OpDiv64u:
		return rewriteValueAMD64_OpDiv64u(v)
	case OpDiv8:
		return rewriteValueAMD64_OpDiv8(v)
	case OpDiv8u:
		return rewriteValueAMD64_OpDiv8u(v)
	case OpDivFloat32x16:
		v.Op = OpAMD64VDIVPS512
		return true
	case OpDivFloat32x4:
		v.Op = OpAMD64VDIVPS128
		return true
	case OpDivFloat32x8:
		v.Op = OpAMD64VDIVPS256
		return true
	case OpDivFloat64x2:
		v.Op = OpAMD64VDIVPD128
		return true
	case OpDivFloat64x4:
		v.Op = OpAMD64VDIVPD256
		return true
	case OpDivFloat64x8:
		v.Op = OpAMD64VDIVPD512
		return true
	case OpDotProdPairsInt16x16:
		v.Op = OpAMD64VPMADDWD256
		return true
	case OpDotProdPairsInt16x32:
		v.Op = OpAMD64VPMADDWD512
		return true
	case OpDotProdPairsInt16x8:
		v.Op = OpAMD64VPMADDWD128
		return true
	case OpDotProdPairsSaturatedUint8x16:
		v.Op = OpAMD64VPMADDUBSW128
		return true
	case OpDotProdPairsSaturatedUint8x32:
		v.Op = OpAMD64VPMADDUBSW256
		return true
	case OpDotProdPairsSaturatedUint8x64:
		v.Op = OpAMD64VPMADDUBSW512
		return true
	case OpEq16:
		return rewriteValueAMD64_OpEq16(v)
	case OpEq32:
		return rewriteValueAMD64_OpEq32(v)
	case OpEq32F:
		return rewriteValueAMD64_OpEq32F(v)
	case OpEq64:
		return rewriteValueAMD64_OpEq64(v)
	case OpEq64F:
		return rewriteValueAMD64_OpEq64F(v)
	case OpEq8:
		return rewriteValueAMD64_OpEq8(v)
	case OpEqB:
		return rewriteValueAMD64_OpEqB(v)
	case OpEqPtr:
		return rewriteValueAMD64_OpEqPtr(v)
	case OpEqualFloat32x16:
		return rewriteValueAMD64_OpEqualFloat32x16(v)
	case OpEqualFloat32x4:
		return rewriteValueAMD64_OpEqualFloat32x4(v)
	case OpEqualFloat32x8:
		return rewriteValueAMD64_OpEqualFloat32x8(v)
	case OpEqualFloat64x2:
		return rewriteValueAMD64_OpEqualFloat64x2(v)
	case OpEqualFloat64x4:
		return rewriteValueAMD64_OpEqualFloat64x4(v)
	case OpEqualFloat64x8:
		return rewriteValueAMD64_OpEqualFloat64x8(v)
	case OpEqualInt16x16:
		v.Op = OpAMD64VPCMPEQW256
		return true
	case OpEqualInt16x32:
		return rewriteValueAMD64_OpEqualInt16x32(v)
	case OpEqualInt16x8:
		v.Op = OpAMD64VPCMPEQW128
		return true
	case OpEqualInt32x16:
		return rewriteValueAMD64_OpEqualInt32x16(v)
	case OpEqualInt32x4:
		v.Op = OpAMD64VPCMPEQD128
		return true
	case OpEqualInt32x8:
		v.Op = OpAMD64VPCMPEQD256
		return true
	case OpEqualInt64x2:
		v.Op = OpAMD64VPCMPEQQ128
		return true
	case OpEqualInt64x4:
		v.Op = OpAMD64VPCMPEQQ256
		return true
	case OpEqualInt64x8:
		return rewriteValueAMD64_OpEqualInt64x8(v)
	case OpEqualInt8x16:
		v.Op = OpAMD64VPCMPEQB128
		return true
	case OpEqualInt8x32:
		v.Op = OpAMD64VPCMPEQB256
		return true
	case OpEqualInt8x64:
		return rewriteValueAMD64_OpEqualInt8x64(v)
	case OpEqualUint16x16:
		v.Op = OpAMD64VPCMPEQW256
		return true
	case OpEqualUint16x32:
		return rewriteValueAMD64_OpEqualUint16x32(v)
	case OpEqualUint16x8:
		v.Op = OpAMD64VPCMPEQW128
		return true
	case OpEqualUint32x16:
		return rewriteValueAMD64_OpEqualUint32x16(v)
	case OpEqualUint32x4:
		v.Op = OpAMD64VPCMPEQD128
		return true
	case OpEqualUint32x8:
		v.Op = OpAMD64VPCMPEQD256
		return true
	case OpEqualUint64x2:
		v.Op = OpAMD64VPCMPEQQ128
		return true
	case OpEqualUint64x4:
		v.Op = OpAMD64VPCMPEQQ256
		return true
	case OpEqualUint64x8:
		return rewriteValueAMD64_OpEqualUint64x8(v)
	case OpEqualUint8x16:
		v.Op = OpAMD64VPCMPEQB128
		return true
	case OpEqualUint8x32:
		v.Op = OpAMD64VPCMPEQB256
		return true
	case OpEqualUint8x64:
		return rewriteValueAMD64_OpEqualUint8x64(v)
	case OpExpandFloat32x16:
		return rewriteValueAMD64_OpExpandFloat32x16(v)
	case OpExpandFloat32x4:
		return rewriteValueAMD64_OpExpandFloat32x4(v)
	case OpExpandFloat32x8:
		return rewriteValueAMD64_OpExpandFloat32x8(v)
	case OpExpandFloat64x2:
		return rewriteValueAMD64_OpExpandFloat64x2(v)
	case OpExpandFloat64x4:
		return rewriteValueAMD64_OpExpandFloat64x4(v)
	case OpExpandFloat64x8:
		return rewriteValueAMD64_OpExpandFloat64x8(v)
	case OpExpandInt16x16:
		return rewriteValueAMD64_OpExpandInt16x16(v)
	case OpExpandInt16x32:
		return rewriteValueAMD64_OpExpandInt16x32(v)
	case OpExpandInt16x8:
		return rewriteValueAMD64_OpExpandInt16x8(v)
	case OpExpandInt32x16:
		return rewriteValueAMD64_OpExpandInt32x16(v)
	case OpExpandInt32x4:
		return rewriteValueAMD64_OpExpandInt32x4(v)
	case OpExpandInt32x8:
		return rewriteValueAMD64_OpExpandInt32x8(v)
	case OpExpandInt64x2:
		return rewriteValueAMD64_OpExpandInt64x2(v)
	case OpExpandInt64x4:
		return rewriteValueAMD64_OpExpandInt64x4(v)
	case OpExpandInt64x8:
		return rewriteValueAMD64_OpExpandInt64x8(v)
	case OpExpandInt8x16:
		return rewriteValueAMD64_OpExpandInt8x16(v)
	case OpExpandInt8x32:
		return rewriteValueAMD64_OpExpandInt8x32(v)
	case OpExpandInt8x64:
		return rewriteValueAMD64_OpExpandInt8x64(v)
	case OpExpandUint16x16:
		return rewriteValueAMD64_OpExpandUint16x16(v)
	case OpExpandUint16x32:
		return rewriteValueAMD64_OpExpandUint16x32(v)
	case OpExpandUint16x8:
		return rewriteValueAMD64_OpExpandUint16x8(v)
	case OpExpandUint32x16:
		return rewriteValueAMD64_OpExpandUint32x16(v)
	case OpExpandUint32x4:
		return rewriteValueAMD64_OpExpandUint32x4(v)
	case OpExpandUint32x8:
		return rewriteValueAMD64_OpExpandUint32x8(v)
	case OpExpandUint64x2:
		return rewriteValueAMD64_OpExpandUint64x2(v)
	case OpExpandUint64x4:
		return rewriteValueAMD64_OpExpandUint64x4(v)
	case OpExpandUint64x8:
		return rewriteValueAMD64_OpExpandUint64x8(v)
	case OpExpandUint8x16:
		return rewriteValueAMD64_OpExpandUint8x16(v)
	case OpExpandUint8x32:
		return rewriteValueAMD64_OpExpandUint8x32(v)
	case OpExpandUint8x64:
		return rewriteValueAMD64_OpExpandUint8x64(v)
	case OpFMA:
		return rewriteValueAMD64_OpFMA(v)
	case OpFloor:
		return rewriteValueAMD64_OpFloor(v)
	case OpFloorFloat32x4:
		return rewriteValueAMD64_OpFloorFloat32x4(v)
	case OpFloorFloat32x8:
		return rewriteValueAMD64_OpFloorFloat32x8(v)
	case OpFloorFloat64x2:
		return rewriteValueAMD64_OpFloorFloat64x2(v)
	case OpFloorFloat64x4:
		return rewriteValueAMD64_OpFloorFloat64x4(v)
	case OpFloorScaledFloat32x16:
		return rewriteValueAMD64_OpFloorScaledFloat32x16(v)
	case OpFloorScaledFloat32x4:
		return rewriteValueAMD64_OpFloorScaledFloat32x4(v)
	case OpFloorScaledFloat32x8:
		return rewriteValueAMD64_OpFloorScaledFloat32x8(v)
	case OpFloorScaledFloat64x2:
		return rewriteValueAMD64_OpFloorScaledFloat64x2(v)
	case OpFloorScaledFloat64x4:
		return rewriteValueAMD64_OpFloorScaledFloat64x4(v)
	case OpFloorScaledFloat64x8:
		return rewriteValueAMD64_OpFloorScaledFloat64x8(v)
	case OpFloorScaledResidueFloat32x16:
		return rewriteValueAMD64_OpFloorScaledResidueFloat32x16(v)
	case OpFloorScaledResidueFloat32x4:
		return rewriteValueAMD64_OpFloorScaledResidueFloat32x4(v)
	case OpFloorScaledResidueFloat32x8:
		return rewriteValueAMD64_OpFloorScaledResidueFloat32x8(v)
	case OpFloorScaledResidueFloat64x2:
		return rewriteValueAMD64_OpFloorScaledResidueFloat64x2(v)
	case OpFloorScaledResidueFloat64x4:
		return rewriteValueAMD64_OpFloorScaledResidueFloat64x4(v)
	case OpFloorScaledResidueFloat64x8:
		return rewriteValueAMD64_OpFloorScaledResidueFloat64x8(v)
	case OpGaloisFieldAffineTransformInverseUint8x16:
		v.Op = OpAMD64VGF2P8AFFINEINVQB128
		return true
	case OpGaloisFieldAffineTransformInverseUint8x32:
		v.Op = OpAMD64VGF2P8AFFINEINVQB256
		return true
	case OpGaloisFieldAffineTransformInverseUint8x64:
		v.Op = OpAMD64VGF2P8AFFINEINVQB512
		return true
	case OpGaloisFieldAffineTransformUint8x16:
		v.Op = OpAMD64VGF2P8AFFINEQB128
		return true
	case OpGaloisFieldAffineTransformUint8x32:
		v.Op = OpAMD64VGF2P8AFFINEQB256
		return true
	case OpGaloisFieldAffineTransformUint8x64:
		v.Op = OpAMD64VGF2P8AFFINEQB512
		return true
	case OpGaloisFieldMulUint8x16:
		v.Op = OpAMD64VGF2P8MULB128
		return true
	case OpGaloisFieldMulUint8x32:
		v.Op = OpAMD64VGF2P8MULB256
		return true
	case OpGaloisFieldMulUint8x64:
		v.Op = OpAMD64VGF2P8MULB512
		return true
	case OpGetCallerPC:
		v.Op = OpAMD64LoweredGetCallerPC
		return true
	case OpGetCallerSP:
		v.Op = OpAMD64LoweredGetCallerSP
		return true
	case OpGetClosurePtr:
		v.Op = OpAMD64LoweredGetClosurePtr
		return true
	case OpGetElemFloat32x4:
		v.Op = OpAMD64VPEXTRD128
		return true
	case OpGetElemFloat64x2:
		v.Op = OpAMD64VPEXTRQ128
		return true
	case OpGetElemInt16x8:
		v.Op = OpAMD64VPEXTRW128
		return true
	case OpGetElemInt32x4:
		v.Op = OpAMD64VPEXTRD128
		return true
	case OpGetElemInt64x2:
		v.Op = OpAMD64VPEXTRQ128
		return true
	case OpGetElemInt8x16:
		v.Op = OpAMD64VPEXTRB128
		return true
	case OpGetElemUint16x8:
		v.Op = OpAMD64VPEXTRW128
		return true
	case OpGetElemUint32x4:
		v.Op = OpAMD64VPEXTRD128
		return true
	case OpGetElemUint64x2:
		v.Op = OpAMD64VPEXTRQ128
		return true
	case OpGetElemUint8x16:
		v.Op = OpAMD64VPEXTRB128
		return true
	case OpGetG:
		return rewriteValueAMD64_OpGetG(v)
	case OpGetHiFloat32x16:
		return rewriteValueAMD64_OpGetHiFloat32x16(v)
	case OpGetHiFloat32x8:
		return rewriteValueAMD64_OpGetHiFloat32x8(v)
	case OpGetHiFloat64x4:
		return rewriteValueAMD64_OpGetHiFloat64x4(v)
	case OpGetHiFloat64x8:
		return rewriteValueAMD64_OpGetHiFloat64x8(v)
	case OpGetHiInt16x16:
		return rewriteValueAMD64_OpGetHiInt16x16(v)
	case OpGetHiInt16x32:
		return rewriteValueAMD64_OpGetHiInt16x32(v)
	case OpGetHiInt32x16:
		return rewriteValueAMD64_OpGetHiInt32x16(v)
	case OpGetHiInt32x8:
		return rewriteValueAMD64_OpGetHiInt32x8(v)
	case OpGetHiInt64x4:
		return rewriteValueAMD64_OpGetHiInt64x4(v)
	case OpGetHiInt64x8:
		return rewriteValueAMD64_OpGetHiInt64x8(v)
	case OpGetHiInt8x32:
		return rewriteValueAMD64_OpGetHiInt8x32(v)
	case OpGetHiInt8x64:
		return rewriteValueAMD64_OpGetHiInt8x64(v)
	case OpGetHiUint16x16:
		return rewriteValueAMD64_OpGetHiUint16x16(v)
	case OpGetHiUint16x32:
		return rewriteValueAMD64_OpGetHiUint16x32(v)
	case OpGetHiUint32x16:
		return rewriteValueAMD64_OpGetHiUint32x16(v)
	case OpGetHiUint32x8:
		return rewriteValueAMD64_OpGetHiUint32x8(v)
	case OpGetHiUint64x4:
		return rewriteValueAMD64_OpGetHiUint64x4(v)
	case OpGetHiUint64x8:
		return rewriteValueAMD64_OpGetHiUint64x8(v)
	case OpGetHiUint8x32:
		return rewriteValueAMD64_OpGetHiUint8x32(v)
	case OpGetHiUint8x64:
		return rewriteValueAMD64_OpGetHiUint8x64(v)
	case OpGetLoFloat32x16:
		return rewriteValueAMD64_OpGetLoFloat32x16(v)
	case OpGetLoFloat32x8:
		return rewriteValueAMD64_OpGetLoFloat32x8(v)
	case OpGetLoFloat64x4:
		return rewriteValueAMD64_OpGetLoFloat64x4(v)
	case OpGetLoFloat64x8:
		return rewriteValueAMD64_OpGetLoFloat64x8(v)
	case OpGetLoInt16x16:
		return rewriteValueAMD64_OpGetLoInt16x16(v)
	case OpGetLoInt16x32:
		return rewriteValueAMD64_OpGetLoInt16x32(v)
	case OpGetLoInt32x16:
		return rewriteValueAMD64_OpGetLoInt32x16(v)
	case OpGetLoInt32x8:
		return rewriteValueAMD64_OpGetLoInt32x8(v)
	case OpGetLoInt64x4:
		return rewriteValueAMD64_OpGetLoInt64x4(v)
	case OpGetLoInt64x8:
		return rewriteValueAMD64_OpGetLoInt64x8(v)
	case OpGetLoInt8x32:
		return rewriteValueAMD64_OpGetLoInt8x32(v)
	case OpGetLoInt8x64:
		return rewriteValueAMD64_OpGetLoInt8x64(v)
	case OpGetLoUint16x16:
		return rewriteValueAMD64_OpGetLoUint16x16(v)
	case OpGetLoUint16x32:
		return rewriteValueAMD64_OpGetLoUint16x32(v)
	case OpGetLoUint32x16:
		return rewriteValueAMD64_OpGetLoUint32x16(v)
	case OpGetLoUint32x8:
		return rewriteValueAMD64_OpGetLoUint32x8(v)
	case OpGetLoUint64x4:
		return rewriteValueAMD64_OpGetLoUint64x4(v)
	case OpGetLoUint64x8:
		return rewriteValueAMD64_OpGetLoUint64x8(v)
	case OpGetLoUint8x32:
		return rewriteValueAMD64_OpGetLoUint8x32(v)
	case OpGetLoUint8x64:
		return rewriteValueAMD64_OpGetLoUint8x64(v)
	case OpGreaterEqualFloat32x16:
		return rewriteValueAMD64_OpGreaterEqualFloat32x16(v)
	case OpGreaterEqualFloat32x4:
		return rewriteValueAMD64_OpGreaterEqualFloat32x4(v)
	case OpGreaterEqualFloat32x8:
		return rewriteValueAMD64_OpGreaterEqualFloat32x8(v)
	case OpGreaterEqualFloat64x2:
		return rewriteValueAMD64_OpGreaterEqualFloat64x2(v)
	case OpGreaterEqualFloat64x4:
		return rewriteValueAMD64_OpGreaterEqualFloat64x4(v)
	case OpGreaterEqualFloat64x8:
		return rewriteValueAMD64_OpGreaterEqualFloat64x8(v)
	case OpGreaterEqualInt16x32:
		return rewriteValueAMD64_OpGreaterEqualInt16x32(v)
	case OpGreaterEqualInt32x16:
		return rewriteValueAMD64_OpGreaterEqualInt32x16(v)
	case OpGreaterEqualInt64x8:
		return rewriteValueAMD64_OpGreaterEqualInt64x8(v)
	case OpGreaterEqualInt8x64:
		return rewriteValueAMD64_OpGreaterEqualInt8x64(v)
	case OpGreaterEqualUint16x32:
		return rewriteValueAMD64_OpGreaterEqualUint16x32(v)
	case OpGreaterEqualUint32x16:
		return rewriteValueAMD64_OpGreaterEqualUint32x16(v)
	case OpGreaterEqualUint64x8:
		return rewriteValueAMD64_OpGreaterEqualUint64x8(v)
	case OpGreaterEqualUint8x64:
		return rewriteValueAMD64_OpGreaterEqualUint8x64(v)
	case OpGreaterFloat32x16:
		return rewriteValueAMD64_OpGreaterFloat32x16(v)
	case OpGreaterFloat32x4:
		return rewriteValueAMD64_OpGreaterFloat32x4(v)
	case OpGreaterFloat32x8:
		return rewriteValueAMD64_OpGreaterFloat32x8(v)
	case OpGreaterFloat64x2:
		return rewriteValueAMD64_OpGreaterFloat64x2(v)
	case OpGreaterFloat64x4:
		return rewriteValueAMD64_OpGreaterFloat64x4(v)
	case OpGreaterFloat64x8:
		return rewriteValueAMD64_OpGreaterFloat64x8(v)
	case OpGreaterInt16x16:
		v.Op = OpAMD64VPCMPGTW256
		return true
	case OpGreaterInt16x32:
		return rewriteValueAMD64_OpGreaterInt16x32(v)
	case OpGreaterInt16x8:
		v.Op = OpAMD64VPCMPGTW128
		return true
	case OpGreaterInt32x16:
		return rewriteValueAMD64_OpGreaterInt32x16(v)
	case OpGreaterInt32x4:
		v.Op = OpAMD64VPCMPGTD128
		return true
	case OpGreaterInt32x8:
		v.Op = OpAMD64VPCMPGTD256
		return true
	case OpGreaterInt64x2:
		v.Op = OpAMD64VPCMPGTQ128
		return true
	case OpGreaterInt64x4:
		v.Op = OpAMD64VPCMPGTQ256
		return true
	case OpGreaterInt64x8:
		return rewriteValueAMD64_OpGreaterInt64x8(v)
	case OpGreaterInt8x16:
		v.Op = OpAMD64VPCMPGTB128
		return true
	case OpGreaterInt8x32:
		v.Op = OpAMD64VPCMPGTB256
		return true
	case OpGreaterInt8x64:
		return rewriteValueAMD64_OpGreaterInt8x64(v)
	case OpGreaterUint16x32:
		return rewriteValueAMD64_OpGreaterUint16x32(v)
	case OpGreaterUint32x16:
		return rewriteValueAMD64_OpGreaterUint32x16(v)
	case OpGreaterUint64x8:
		return rewriteValueAMD64_OpGreaterUint64x8(v)
	case OpGreaterUint8x64:
		return rewriteValueAMD64_OpGreaterUint8x64(v)
	case OpHasCPUFeature:
		return rewriteValueAMD64_OpHasCPUFeature(v)
	case OpHmul32:
		v.Op = OpAMD64HMULL
		return true
	case OpHmul32u:
		v.Op = OpAMD64HMULLU
		return true
	case OpHmul64:
		v.Op = OpAMD64HMULQ
		return true
	case OpHmul64u:
		v.Op = OpAMD64HMULQU
		return true
	case OpInterCall:
		v.Op = OpAMD64CALLinter
		return true
	case OpIsInBounds:
		return rewriteValueAMD64_OpIsInBounds(v)
	case OpIsNanFloat32x16:
		return rewriteValueAMD64_OpIsNanFloat32x16(v)
	case OpIsNanFloat32x4:
		return rewriteValueAMD64_OpIsNanFloat32x4(v)
	case OpIsNanFloat32x8:
		return rewriteValueAMD64_OpIsNanFloat32x8(v)
	case OpIsNanFloat64x2:
		return rewriteValueAMD64_OpIsNanFloat64x2(v)
	case OpIsNanFloat64x4:
		return rewriteValueAMD64_OpIsNanFloat64x4(v)
	case OpIsNanFloat64x8:
		return rewriteValueAMD64_OpIsNanFloat64x8(v)
	case OpIsNonNil:
		return rewriteValueAMD64_OpIsNonNil(v)
	case OpIsSliceInBounds:
		return rewriteValueAMD64_OpIsSliceInBounds(v)
	case OpLeq16:
		return rewriteValueAMD64_OpLeq16(v)
	case OpLeq16U:
		return rewriteValueAMD64_OpLeq16U(v)
	case OpLeq32:
		return rewriteValueAMD64_OpLeq32(v)
	case OpLeq32F:
		return rewriteValueAMD64_OpLeq32F(v)
	case OpLeq32U:
		return rewriteValueAMD64_OpLeq32U(v)
	case OpLeq64:
		return rewriteValueAMD64_OpLeq64(v)
	case OpLeq64F:
		return rewriteValueAMD64_OpLeq64F(v)
	case OpLeq64U:
		return rewriteValueAMD64_OpLeq64U(v)
	case OpLeq8:
		return rewriteValueAMD64_OpLeq8(v)
	case OpLeq8U:
		return rewriteValueAMD64_OpLeq8U(v)
	case OpLess16:
		return rewriteValueAMD64_OpLess16(v)
	case OpLess16U:
		return rewriteValueAMD64_OpLess16U(v)
	case OpLess32:
		return rewriteValueAMD64_OpLess32(v)
	case OpLess32F:
		return rewriteValueAMD64_OpLess32F(v)
	case OpLess32U:
		return rewriteValueAMD64_OpLess32U(v)
	case OpLess64:
		return rewriteValueAMD64_OpLess64(v)
	case OpLess64F:
		return rewriteValueAMD64_OpLess64F(v)
	case OpLess64U:
		return rewriteValueAMD64_OpLess64U(v)
	case OpLess8:
		return rewriteValueAMD64_OpLess8(v)
	case OpLess8U:
		return rewriteValueAMD64_OpLess8U(v)
	case OpLessEqualFloat32x16:
		return rewriteValueAMD64_OpLessEqualFloat32x16(v)
	case OpLessEqualFloat32x4:
		return rewriteValueAMD64_OpLessEqualFloat32x4(v)
	case OpLessEqualFloat32x8:
		return rewriteValueAMD64_OpLessEqualFloat32x8(v)
	case OpLessEqualFloat64x2:
		return rewriteValueAMD64_OpLessEqualFloat64x2(v)
	case OpLessEqualFloat64x4:
		return rewriteValueAMD64_OpLessEqualFloat64x4(v)
	case OpLessEqualFloat64x8:
		return rewriteValueAMD64_OpLessEqualFloat64x8(v)
	case OpLessEqualInt16x32:
		return rewriteValueAMD64_OpLessEqualInt16x32(v)
	case OpLessEqualInt32x16:
		return rewriteValueAMD64_OpLessEqualInt32x16(v)
	case OpLessEqualInt64x8:
		return rewriteValueAMD64_OpLessEqualInt64x8(v)
	case OpLessEqualInt8x64:
		return rewriteValueAMD64_OpLessEqualInt8x64(v)
	case OpLessEqualUint16x32:
		return rewriteValueAMD64_OpLessEqualUint16x32(v)
	case OpLessEqualUint32x16:
		return rewriteValueAMD64_OpLessEqualUint32x16(v)
	case OpLessEqualUint64x8:
		return rewriteValueAMD64_OpLessEqualUint64x8(v)
	case OpLessEqualUint8x64:
		return rewriteValueAMD64_OpLessEqualUint8x64(v)
	case OpLessFloat32x16:
		return rewriteValueAMD64_OpLessFloat32x16(v)
	case OpLessFloat32x4:
		return rewriteValueAMD64_OpLessFloat32x4(v)
	case OpLessFloat32x8:
		return rewriteValueAMD64_OpLessFloat32x8(v)
	case OpLessFloat64x2:
		return rewriteValueAMD64_OpLessFloat64x2(v)
	case OpLessFloat64x4:
		return rewriteValueAMD64_OpLessFloat64x4(v)
	case OpLessFloat64x8:
		return rewriteValueAMD64_OpLessFloat64x8(v)
	case OpLessInt16x32:
		return rewriteValueAMD64_OpLessInt16x32(v)
	case OpLessInt32x16:
		return rewriteValueAMD64_OpLessInt32x16(v)
	case OpLessInt64x8:
		return rewriteValueAMD64_OpLessInt64x8(v)
	case OpLessInt8x64:
		return rewriteValueAMD64_OpLessInt8x64(v)
	case OpLessUint16x32:
		return rewriteValueAMD64_OpLessUint16x32(v)
	case OpLessUint32x16:
		return rewriteValueAMD64_OpLessUint32x16(v)
	case OpLessUint64x8:
		return rewriteValueAMD64_OpLessUint64x8(v)
	case OpLessUint8x64:
		return rewriteValueAMD64_OpLessUint8x64(v)
	case OpLoad:
		return rewriteValueAMD64_OpLoad(v)
	case OpLoadMask16x16:
		return rewriteValueAMD64_OpLoadMask16x16(v)
	case OpLoadMask16x32:
		return rewriteValueAMD64_OpLoadMask16x32(v)
	case OpLoadMask16x8:
		return rewriteValueAMD64_OpLoadMask16x8(v)
	case OpLoadMask32x16:
		return rewriteValueAMD64_OpLoadMask32x16(v)
	case OpLoadMask32x4:
		return rewriteValueAMD64_OpLoadMask32x4(v)
	case OpLoadMask32x8:
		return rewriteValueAMD64_OpLoadMask32x8(v)
	case OpLoadMask64x2:
		return rewriteValueAMD64_OpLoadMask64x2(v)
	case OpLoadMask64x4:
		return rewriteValueAMD64_OpLoadMask64x4(v)
	case OpLoadMask64x8:
		return rewriteValueAMD64_OpLoadMask64x8(v)
	case OpLoadMask8x16:
		return rewriteValueAMD64_OpLoadMask8x16(v)
	case OpLoadMask8x32:
		return rewriteValueAMD64_OpLoadMask8x32(v)
	case OpLoadMask8x64:
		return rewriteValueAMD64_OpLoadMask8x64(v)
	case OpLoadMasked16:
		return rewriteValueAMD64_OpLoadMasked16(v)
	case OpLoadMasked32:
		return rewriteValueAMD64_OpLoadMasked32(v)
	case OpLoadMasked64:
		return rewriteValueAMD64_OpLoadMasked64(v)
	case OpLoadMasked8:
		return rewriteValueAMD64_OpLoadMasked8(v)
	case OpLocalAddr:
		return rewriteValueAMD64_OpLocalAddr(v)
	case OpLsh16x16:
		return rewriteValueAMD64_OpLsh16x16(v)
	case OpLsh16x32:
		return rewriteValueAMD64_OpLsh16x32(v)
	case OpLsh16x64:
		return rewriteValueAMD64_OpLsh16x64(v)
	case OpLsh16x8:
		return rewriteValueAMD64_OpLsh16x8(v)
	case OpLsh32x16:
		return rewriteValueAMD64_OpLsh32x16(v)
	case OpLsh32x32:
		return rewriteValueAMD64_OpLsh32x32(v)
	case OpLsh32x64:
		return rewriteValueAMD64_OpLsh32x64(v)
	case OpLsh32x8:
		return rewriteValueAMD64_OpLsh32x8(v)
	case OpLsh64x16:
		return rewriteValueAMD64_OpLsh64x16(v)
	case OpLsh64x32:
		return rewriteValueAMD64_OpLsh64x32(v)
	case OpLsh64x64:
		return rewriteValueAMD64_OpLsh64x64(v)
	case OpLsh64x8:
		return rewriteValueAMD64_OpLsh64x8(v)
	case OpLsh8x16:
		return rewriteValueAMD64_OpLsh8x16(v)
	case OpLsh8x32:
		return rewriteValueAMD64_OpLsh8x32(v)
	case OpLsh8x64:
		return rewriteValueAMD64_OpLsh8x64(v)
	case OpLsh8x8:
		return rewriteValueAMD64_OpLsh8x8(v)
	case OpMax32F:
		return rewriteValueAMD64_OpMax32F(v)
	case OpMax64F:
		return rewriteValueAMD64_OpMax64F(v)
	case OpMaxFloat32x16:
		v.Op = OpAMD64VMAXPS512
		return true
	case OpMaxFloat32x4:
		v.Op = OpAMD64VMAXPS128
		return true
	case OpMaxFloat32x8:
		v.Op = OpAMD64VMAXPS256
		return true
	case OpMaxFloat64x2:
		v.Op = OpAMD64VMAXPD128
		return true
	case OpMaxFloat64x4:
		v.Op = OpAMD64VMAXPD256
		return true
	case OpMaxFloat64x8:
		v.Op = OpAMD64VMAXPD512
		return true
	case OpMaxInt16x16:
		v.Op = OpAMD64VPMAXSW256
		return true
	case OpMaxInt16x32:
		v.Op = OpAMD64VPMAXSW512
		return true
	case OpMaxInt16x8:
		v.Op = OpAMD64VPMAXSW128
		return true
	case OpMaxInt32x16:
		v.Op = OpAMD64VPMAXSD512
		return true
	case OpMaxInt32x4:
		v.Op = OpAMD64VPMAXSD128
		return true
	case OpMaxInt32x8:
		v.Op = OpAMD64VPMAXSD256
		return true
	case OpMaxInt64x2:
		v.Op = OpAMD64VPMAXSQ128
		return true
	case OpMaxInt64x4:
		v.Op = OpAMD64VPMAXSQ256
		return true
	case OpMaxInt64x8:
		v.Op = OpAMD64VPMAXSQ512
		return true
	case OpMaxInt8x16:
		v.Op = OpAMD64VPMAXSB128
		return true
	case OpMaxInt8x32:
		v.Op = OpAMD64VPMAXSB256
		return true
	case OpMaxInt8x64:
		v.Op = OpAMD64VPMAXSB512
		return true
	case OpMaxUint16x16:
		v.Op = OpAMD64VPMAXUW256
		return true
	case OpMaxUint16x32:
		v.Op = OpAMD64VPMAXUW512
		return true
	case OpMaxUint16x8:
		v.Op = OpAMD64VPMAXUW128
		return true
	case OpMaxUint32x16:
		v.Op = OpAMD64VPMAXUD512
		return true
	case OpMaxUint32x4:
		v.Op = OpAMD64VPMAXUD128
		return true
	case OpMaxUint32x8:
		v.Op = OpAMD64VPMAXUD256
		return true
	case OpMaxUint64x2:
		v.Op = OpAMD64VPMAXUQ128
		return true
	case OpMaxUint64x4:
		v.Op = OpAMD64VPMAXUQ256
		return true
	case OpMaxUint64x8:
		v.Op = OpAMD64VPMAXUQ512
		return true
	case OpMaxUint8x16:
		v.Op = OpAMD64VPMAXUB128
		return true
	case OpMaxUint8x32:
		v.Op = OpAMD64VPMAXUB256
		return true
	case OpMaxUint8x64:
		v.Op = OpAMD64VPMAXUB512
		return true
	case OpMin32F:
		return rewriteValueAMD64_OpMin32F(v)
	case OpMin64F:
		return rewriteValueAMD64_OpMin64F(v)
	case OpMinFloat32x16:
		v.Op = OpAMD64VMINPS512
		return true
	case OpMinFloat32x4:
		v.Op = OpAMD64VMINPS128
		return true
	case OpMinFloat32x8:
		v.Op = OpAMD64VMINPS256
		return true
	case OpMinFloat64x2:
		v.Op = OpAMD64VMINPD128
		return true
	case OpMinFloat64x4:
		v.Op = OpAMD64VMINPD256
		return true
	case OpMinFloat64x8:
		v.Op = OpAMD64VMINPD512
		return true
	case OpMinInt16x16:
		v.Op = OpAMD64VPMINSW256
		return true
	case OpMinInt16x32:
		v.Op = OpAMD64VPMINSW512
		return true
	case OpMinInt16x8:
		v.Op = OpAMD64VPMINSW128
		return true
	case OpMinInt32x16:
		v.Op = OpAMD64VPMINSD512
		return true
	case OpMinInt32x4:
		v.Op = OpAMD64VPMINSD128
		return true
	case OpMinInt32x8:
		v.Op = OpAMD64VPMINSD256
		return true
	case OpMinInt64x2:
		v.Op = OpAMD64VPMINSQ128
		return true
	case OpMinInt64x4:
		v.Op = OpAMD64VPMINSQ256
		return true
	case OpMinInt64x8:
		v.Op = OpAMD64VPMINSQ512
		return true
	case OpMinInt8x16:
		v.Op = OpAMD64VPMINSB128
		return true
	case OpMinInt8x32:
		v.Op = OpAMD64VPMINSB256
		return true
	case OpMinInt8x64:
		v.Op = OpAMD64VPMINSB512
		return true
	case OpMinUint16x16:
		v.Op = OpAMD64VPMINUW256
		return true
	case OpMinUint16x32:
		v.Op = OpAMD64VPMINUW512
		return true
	case OpMinUint16x8:
		v.Op = OpAMD64VPMINUW128
		return true
	case OpMinUint32x16:
		v.Op = OpAMD64VPMINUD512
		return true
	case OpMinUint32x4:
		v.Op = OpAMD64VPMINUD128
		return true
	case OpMinUint32x8:
		v.Op = OpAMD64VPMINUD256
		return true
	case OpMinUint64x2:
		v.Op = OpAMD64VPMINUQ128
		return true
	case OpMinUint64x4:
		v.Op = OpAMD64VPMINUQ256
		return true
	case OpMinUint64x8:
		v.Op = OpAMD64VPMINUQ512
		return true
	case OpMinUint8x16:
		v.Op = OpAMD64VPMINUB128
		return true
	case OpMinUint8x32:
		v.Op = OpAMD64VPMINUB256
		return true
	case OpMinUint8x64:
		v.Op = OpAMD64VPMINUB512
		return true
	case OpMod16:
		return rewriteValueAMD64_OpMod16(v)
	case OpMod16u:
		return rewriteValueAMD64_OpMod16u(v)
	case OpMod32:
		return rewriteValueAMD64_OpMod32(v)
	case OpMod32u:
		return rewriteValueAMD64_OpMod32u(v)
	case OpMod64:
		return rewriteValueAMD64_OpMod64(v)
	case OpMod64u:
		return rewriteValueAMD64_OpMod64u(v)
	case OpMod8:
		return rewriteValueAMD64_OpMod8(v)
	case OpMod8u:
		return rewriteValueAMD64_OpMod8u(v)
	case OpMove:
		return rewriteValueAMD64_OpMove(v)
	case OpMul16:
		v.Op = OpAMD64MULL
		return true
	case OpMul32:
		v.Op = OpAMD64MULL
		return true
	case OpMul32F:
		v.Op = OpAMD64MULSS
		return true
	case OpMul64:
		v.Op = OpAMD64MULQ
		return true
	case OpMul64F:
		v.Op = OpAMD64MULSD
		return true
	case OpMul64uhilo:
		v.Op = OpAMD64MULQU2
		return true
	case OpMul8:
		v.Op = OpAMD64MULL
		return true
	case OpMulAddFloat32x16:
		v.Op = OpAMD64VFMADD213PS512
		return true
	case OpMulAddFloat32x4:
		v.Op = OpAMD64VFMADD213PS128
		return true
	case OpMulAddFloat32x8:
		v.Op = OpAMD64VFMADD213PS256
		return true
	case OpMulAddFloat64x2:
		v.Op = OpAMD64VFMADD213PD128
		return true
	case OpMulAddFloat64x4:
		v.Op = OpAMD64VFMADD213PD256
		return true
	case OpMulAddFloat64x8:
		v.Op = OpAMD64VFMADD213PD512
		return true
	case OpMulAddSubFloat32x16:
		v.Op = OpAMD64VFMADDSUB213PS512
		return true
	case OpMulAddSubFloat32x4:
		v.Op = OpAMD64VFMADDSUB213PS128
		return true
	case OpMulAddSubFloat32x8:
		v.Op = OpAMD64VFMADDSUB213PS256
		return true
	case OpMulAddSubFloat64x2:
		v.Op = OpAMD64VFMADDSUB213PD128
		return true
	case OpMulAddSubFloat64x4:
		v.Op = OpAMD64VFMADDSUB213PD256
		return true
	case OpMulAddSubFloat64x8:
		v.Op = OpAMD64VFMADDSUB213PD512
		return true
	case OpMulEvenWidenInt32x4:
		v.Op = OpAMD64VPMULDQ128
		return true
	case OpMulEvenWidenInt32x8:
		v.Op = OpAMD64VPMULDQ256
		return true
	case OpMulEvenWidenUint32x4:
		v.Op = OpAMD64VPMULUDQ128
		return true
	case OpMulEvenWidenUint32x8:
		v.Op = OpAMD64VPMULUDQ256
		return true
	case OpMulFloat32x16:
		v.Op = OpAMD64VMULPS512
		return true
	case OpMulFloat32x4:
		v.Op = OpAMD64VMULPS128
		return true
	case OpMulFloat32x8:
		v.Op = OpAMD64VMULPS256
		return true
	case OpMulFloat64x2:
		v.Op = OpAMD64VMULPD128
		return true
	case OpMulFloat64x4:
		v.Op = OpAMD64VMULPD256
		return true
	case OpMulFloat64x8:
		v.Op = OpAMD64VMULPD512
		return true
	case OpMulHighInt16x16:
		v.Op = OpAMD64VPMULHW256
		return true
	case OpMulHighInt16x32:
		v.Op = OpAMD64VPMULHW512
		return true
	case OpMulHighInt16x8:
		v.Op = OpAMD64VPMULHW128
		return true
	case OpMulHighUint16x16:
		v.Op = OpAMD64VPMULHUW256
		return true
	case OpMulHighUint16x32:
		v.Op = OpAMD64VPMULHUW512
		return true
	case OpMulHighUint16x8:
		v.Op = OpAMD64VPMULHUW128
		return true
	case OpMulInt16x16:
		v.Op = OpAMD64VPMULLW256
		return true
	case OpMulInt16x32:
		v.Op = OpAMD64VPMULLW512
		return true
	case OpMulInt16x8:
		v.Op = OpAMD64VPMULLW128
		return true
	case OpMulInt32x16:
		v.Op = OpAMD64VPMULLD512
		return true
	case OpMulInt32x4:
		v.Op = OpAMD64VPMULLD128
		return true
	case OpMulInt32x8:
		v.Op = OpAMD64VPMULLD256
		return true
	case OpMulInt64x2:
		v.Op = OpAMD64VPMULLQ128
		return true
	case OpMulInt64x4:
		v.Op = OpAMD64VPMULLQ256
		return true
	case OpMulInt64x8:
		v.Op = OpAMD64VPMULLQ512
		return true
	case OpMulSubAddFloat32x16:
		v.Op = OpAMD64VFMSUBADD213PS512
		return true
	case OpMulSubAddFloat32x4:
		v.Op = OpAMD64VFMSUBADD213PS128
		return true
	case OpMulSubAddFloat32x8:
		v.Op = OpAMD64VFMSUBADD213PS256
		return true
	case OpMulSubAddFloat64x2:
		v.Op = OpAMD64VFMSUBADD213PD128
		return true
	case OpMulSubAddFloat64x4:
		v.Op = OpAMD64VFMSUBADD213PD256
		return true
	case OpMulSubAddFloat64x8:
		v.Op = OpAMD64VFMSUBADD213PD512
		return true
	case OpMulUint16x16:
		v.Op = OpAMD64VPMULLW256
		return true
	case OpMulUint16x32:
		v.Op = OpAMD64VPMULLW512
		return true
	case OpMulUint16x8:
		v.Op = OpAMD64VPMULLW128
		return true
	case OpMulUint32x16:
		v.Op = OpAMD64VPMULLD512
		return true
	case OpMulUint32x4:
		v.Op = OpAMD64VPMULLD128
		return true
	case OpMulUint32x8:
		v.Op = OpAMD64VPMULLD256
		return true
	case OpMulUint64x2:
		v.Op = OpAMD64VPMULLQ128
		return true
	case OpMulUint64x4:
		v.Op = OpAMD64VPMULLQ256
		return true
	case OpMulUint64x8:
		v.Op = OpAMD64VPMULLQ512
		return true
	case OpNeg16:
		v.Op = OpAMD64NEGL
		return true
	case OpNeg32:
		v.Op = OpAMD64NEGL
		return true
	case OpNeg32F:
		return rewriteValueAMD64_OpNeg32F(v)
	case OpNeg64:
		v.Op = OpAMD64NEGQ
		return true
	case OpNeg64F:
		return rewriteValueAMD64_OpNeg64F(v)
	case OpNeg8:
		v.Op = OpAMD64NEGL
		return true
	case OpNeq16:
		return rewriteValueAMD64_OpNeq16(v)
	case OpNeq32:
		return rewriteValueAMD64_OpNeq32(v)
	case OpNeq32F:
		return rewriteValueAMD64_OpNeq32F(v)
	case OpNeq64:
		return rewriteValueAMD64_OpNeq64(v)
	case OpNeq64F:
		return rewriteValueAMD64_OpNeq64F(v)
	case OpNeq8:
		return rewriteValueAMD64_OpNeq8(v)
	case OpNeqB:
		return rewriteValueAMD64_OpNeqB(v)
	case OpNeqPtr:
		return rewriteValueAMD64_OpNeqPtr(v)
	case OpNilCheck:
		v.Op = OpAMD64LoweredNilCheck
		return true
	case OpNot:
		return rewriteValueAMD64_OpNot(v)
	case OpNotEqualFloat32x16:
		return rewriteValueAMD64_OpNotEqualFloat32x16(v)
	case OpNotEqualFloat32x4:
		return rewriteValueAMD64_OpNotEqualFloat32x4(v)
	case OpNotEqualFloat32x8:
		return rewriteValueAMD64_OpNotEqualFloat32x8(v)
	case OpNotEqualFloat64x2:
		return rewriteValueAMD64_OpNotEqualFloat64x2(v)
	case OpNotEqualFloat64x4:
		return rewriteValueAMD64_OpNotEqualFloat64x4(v)
	case OpNotEqualFloat64x8:
		return rewriteValueAMD64_OpNotEqualFloat64x8(v)
	case OpNotEqualInt16x32:
		return rewriteValueAMD64_OpNotEqualInt16x32(v)
	case OpNotEqualInt32x16:
		return rewriteValueAMD64_OpNotEqualInt32x16(v)
	case OpNotEqualInt64x8:
		return rewriteValueAMD64_OpNotEqualInt64x8(v)
	case OpNotEqualInt8x64:
		return rewriteValueAMD64_OpNotEqualInt8x64(v)
	case OpNotEqualUint16x32:
		return rewriteValueAMD64_OpNotEqualUint16x32(v)
	case OpNotEqualUint32x16:
		return rewriteValueAMD64_OpNotEqualUint32x16(v)
	case OpNotEqualUint64x8:
		return rewriteValueAMD64_OpNotEqualUint64x8(v)
	case OpNotEqualUint8x64:
		return rewriteValueAMD64_OpNotEqualUint8x64(v)
	case OpOffPtr:
		return rewriteValueAMD64_OpOffPtr(v)
	case OpOnesCountInt16x16:
		v.Op = OpAMD64VPOPCNTW256
		return true
	case OpOnesCountInt16x32:
		v.Op = OpAMD64VPOPCNTW512
		return true
	case OpOnesCountInt16x8:
		v.Op = OpAMD64VPOPCNTW128
		return true
	case OpOnesCountInt32x16:
		v.Op = OpAMD64VPOPCNTD512
		return true
	case OpOnesCountInt32x4:
		v.Op = OpAMD64VPOPCNTD128
		return true
	case OpOnesCountInt32x8:
		v.Op = OpAMD64VPOPCNTD256
		return true
	case OpOnesCountInt64x2:
		v.Op = OpAMD64VPOPCNTQ128
		return true
	case OpOnesCountInt64x4:
		v.Op = OpAMD64VPOPCNTQ256
		return true
	case OpOnesCountInt64x8:
		v.Op = OpAMD64VPOPCNTQ512
		return true
	case OpOnesCountInt8x16:
		v.Op = OpAMD64VPOPCNTB128
		return true
	case OpOnesCountInt8x32:
		v.Op = OpAMD64VPOPCNTB256
		return true
	case OpOnesCountInt8x64:
		v.Op = OpAMD64VPOPCNTB512
		return true
	case OpOnesCountUint16x16:
		v.Op = OpAMD64VPOPCNTW256
		return true
	case OpOnesCountUint16x32:
		v.Op = OpAMD64VPOPCNTW512
		return true
	case OpOnesCountUint16x8:
		v.Op = OpAMD64VPOPCNTW128
		return true
	case OpOnesCountUint32x16:
		v.Op = OpAMD64VPOPCNTD512
		return true
	case OpOnesCountUint32x4:
		v.Op = OpAMD64VPOPCNTD128
		return true
	case OpOnesCountUint32x8:
		v.Op = OpAMD64VPOPCNTD256
		return true
	case OpOnesCountUint64x2:
		v.Op = OpAMD64VPOPCNTQ128
		return true
	case OpOnesCountUint64x4:
		v.Op = OpAMD64VPOPCNTQ256
		return true
	case OpOnesCountUint64x8:
		v.Op = OpAMD64VPOPCNTQ512
		return true
	case OpOnesCountUint8x16:
		v.Op = OpAMD64VPOPCNTB128
		return true
	case OpOnesCountUint8x32:
		v.Op = OpAMD64VPOPCNTB256
		return true
	case OpOnesCountUint8x64:
		v.Op = OpAMD64VPOPCNTB512
		return true
	case OpOr16:
		v.Op = OpAMD64ORL
		return true
	case OpOr32:
		v.Op = OpAMD64ORL
		return true
	case OpOr64:
		v.Op = OpAMD64ORQ
		return true
	case OpOr8:
		v.Op = OpAMD64ORL
		return true
	case OpOrB:
		v.Op = OpAMD64ORL
		return true
	case OpOrInt16x16:
		v.Op = OpAMD64VPOR256
		return true
	case OpOrInt16x32:
		v.Op = OpAMD64VPORD512
		return true
	case OpOrInt16x8:
		v.Op = OpAMD64VPOR128
		return true
	case OpOrInt32x16:
		v.Op = OpAMD64VPORD512
		return true
	case OpOrInt32x4:
		v.Op = OpAMD64VPOR128
		return true
	case OpOrInt32x8:
		v.Op = OpAMD64VPOR256
		return true
	case OpOrInt64x2:
		v.Op = OpAMD64VPOR128
		return true
	case OpOrInt64x4:
		v.Op = OpAMD64VPOR256
		return true
	case OpOrInt64x8:
		v.Op = OpAMD64VPORQ512
		return true
	case OpOrInt8x16:
		v.Op = OpAMD64VPOR128
		return true
	case OpOrInt8x32:
		v.Op = OpAMD64VPOR256
		return true
	case OpOrInt8x64:
		v.Op = OpAMD64VPORD512
		return true
	case OpOrUint16x16:
		v.Op = OpAMD64VPOR256
		return true
	case OpOrUint16x32:
		v.Op = OpAMD64VPORD512
		return true
	case OpOrUint16x8:
		v.Op = OpAMD64VPOR128
		return true
	case OpOrUint32x16:
		v.Op = OpAMD64VPORD512
		return true
	case OpOrUint32x4:
		v.Op = OpAMD64VPOR128
		return true
	case OpOrUint32x8:
		v.Op = OpAMD64VPOR256
		return true
	case OpOrUint64x2:
		v.Op = OpAMD64VPOR128
		return true
	case OpOrUint64x4:
		v.Op = OpAMD64VPOR256
		return true
	case OpOrUint64x8:
		v.Op = OpAMD64VPORQ512
		return true
	case OpOrUint8x16:
		v.Op = OpAMD64VPOR128
		return true
	case OpOrUint8x32:
		v.Op = OpAMD64VPOR256
		return true
	case OpOrUint8x64:
		v.Op = OpAMD64VPORD512
		return true
	case OpPanicBounds:
		v.Op = OpAMD64LoweredPanicBoundsRR
		return true
	case OpPermute2Float32x16:
		v.Op = OpAMD64VPERMI2PS512
		return true
	case OpPermute2Float32x4:
		v.Op = OpAMD64VPERMI2PS128
		return true
	case OpPermute2Float32x8:
		v.Op = OpAMD64VPERMI2PS256
		return true
	case OpPermute2Float64x2:
		v.Op = OpAMD64VPERMI2PD128
		return true
	case OpPermute2Float64x4:
		v.Op = OpAMD64VPERMI2PD256
		return true
	case OpPermute2Float64x8:
		v.Op = OpAMD64VPERMI2PD512
		return true
	case OpPermute2Int16x16:
		v.Op = OpAMD64VPERMI2W256
		return true
	case OpPermute2Int16x32:
		v.Op = OpAMD64VPERMI2W512
		return true
	case OpPermute2Int16x8:
		v.Op = OpAMD64VPERMI2W128
		return true
	case OpPermute2Int32x16:
		v.Op = OpAMD64VPERMI2D512
		return true
	case OpPermute2Int32x4:
		v.Op = OpAMD64VPERMI2D128
		return true
	case OpPermute2Int32x8:
		v.Op = OpAMD64VPERMI2D256
		return true
	case OpPermute2Int64x2:
		v.Op = OpAMD64VPERMI2Q128
		return true
	case OpPermute2Int64x4:
		v.Op = OpAMD64VPERMI2Q256
		return true
	case OpPermute2Int64x8:
		v.Op = OpAMD64VPERMI2Q512
		return true
	case OpPermute2Int8x16:
		v.Op = OpAMD64VPERMI2B128
		return true
	case OpPermute2Int8x32:
		v.Op = OpAMD64VPERMI2B256
		return true
	case OpPermute2Int8x64:
		v.Op = OpAMD64VPERMI2B512
		return true
	case OpPermute2Uint16x16:
		v.Op = OpAMD64VPERMI2W256
		return true
	case OpPermute2Uint16x32:
		v.Op = OpAMD64VPERMI2W512
		return true
	case OpPermute2Uint16x8:
		v.Op = OpAMD64VPERMI2W128
		return true
	case OpPermute2Uint32x16:
		v.Op = OpAMD64VPERMI2D512
		return true
	case OpPermute2Uint32x4:
		v.Op = OpAMD64VPERMI2D128
		return true
	case OpPermute2Uint32x8:
		v.Op = OpAMD64VPERMI2D256
		return true
	case OpPermute2Uint64x2:
		v.Op = OpAMD64VPERMI2Q128
		return true
	case OpPermute2Uint64x4:
		v.Op = OpAMD64VPERMI2Q256
		return true
	case OpPermute2Uint64x8:
		v.Op = OpAMD64VPERMI2Q512
		return true
	case OpPermute2Uint8x16:
		v.Op = OpAMD64VPERMI2B128
		return true
	case OpPermute2Uint8x32:
		v.Op = OpAMD64VPERMI2B256
		return true
	case OpPermute2Uint8x64:
		v.Op = OpAMD64VPERMI2B512
		return true
	case OpPermuteFloat32x16:
		v.Op = OpAMD64VPERMPS512
		return true
	case OpPermuteFloat32x8:
		v.Op = OpAMD64VPERMPS256
		return true
	case OpPermuteFloat64x4:
		v.Op = OpAMD64VPERMPD256
		return true
	case OpPermuteFloat64x8:
		v.Op = OpAMD64VPERMPD512
		return true
	case OpPermuteInt16x16:
		v.Op = OpAMD64VPERMW256
		return true
	case OpPermuteInt16x32:
		v.Op = OpAMD64VPERMW512
		return true
	case OpPermuteInt16x8:
		v.Op = OpAMD64VPERMW128
		return true
	case OpPermuteInt32x16:
		v.Op = OpAMD64VPERMD512
		return true
	case OpPermuteInt32x8:
		v.Op = OpAMD64VPERMD256
		return true
	case OpPermuteInt64x4:
		v.Op = OpAMD64VPERMQ256
		return true
	case OpPermuteInt64x8:
		v.Op = OpAMD64VPERMQ512
		return true
	case OpPermuteInt8x16:
		v.Op = OpAMD64VPERMB128
		return true
	case OpPermuteInt8x32:
		v.Op = OpAMD64VPERMB256
		return true
	case OpPermuteInt8x64:
		v.Op = OpAMD64VPERMB512
		return true
	case OpPermuteUint16x16:
		v.Op = OpAMD64VPERMW256
		return true
	case OpPermuteUint16x32:
		v.Op = OpAMD64VPERMW512
		return true
	case OpPermuteUint16x8:
		v.Op = OpAMD64VPERMW128
		return true
	case OpPermuteUint32x16:
		v.Op = OpAMD64VPERMD512
		return true
	case OpPermuteUint32x8:
		v.Op = OpAMD64VPERMD256
		return true
	case OpPermuteUint64x4:
		v.Op = OpAMD64VPERMQ256
		return true
	case OpPermuteUint64x8:
		v.Op = OpAMD64VPERMQ512
		return true
	case OpPermuteUint8x16:
		v.Op = OpAMD64VPERMB128
		return true
	case OpPermuteUint8x32:
		v.Op = OpAMD64VPERMB256
		return true
	case OpPermuteUint8x64:
		v.Op = OpAMD64VPERMB512
		return true
	case OpPopCount16:
		return rewriteValueAMD64_OpPopCount16(v)
	case OpPopCount32:
		v.Op = OpAMD64POPCNTL
		return true
	case OpPopCount64:
		v.Op = OpAMD64POPCNTQ
		return true
	case OpPopCount8:
		return rewriteValueAMD64_OpPopCount8(v)
	case OpPrefetchCache:
		v.Op = OpAMD64PrefetchT0
		return true
	case OpPrefetchCacheStreamed:
		v.Op = OpAMD64PrefetchNTA
		return true
	case OpReciprocalFloat32x16:
		v.Op = OpAMD64VRCP14PS512
		return true
	case OpReciprocalFloat32x4:
		v.Op = OpAMD64VRCPPS128
		return true
	case OpReciprocalFloat32x8:
		v.Op = OpAMD64VRCPPS256
		return true
	case OpReciprocalFloat64x2:
		v.Op = OpAMD64VRCP14PD128
		return true
	case OpReciprocalFloat64x4:
		v.Op = OpAMD64VRCP14PD256
		return true
	case OpReciprocalFloat64x8:
		v.Op = OpAMD64VRCP14PD512
		return true
	case OpReciprocalSqrtFloat32x16:
		v.Op = OpAMD64VRSQRT14PS512
		return true
	case OpReciprocalSqrtFloat32x4:
		v.Op = OpAMD64VRSQRTPS128
		return true
	case OpReciprocalSqrtFloat32x8:
		v.Op = OpAMD64VRSQRTPS256
		return true
	case OpReciprocalSqrtFloat64x2:
		v.Op = OpAMD64VRSQRT14PD128
		return true
	case OpReciprocalSqrtFloat64x4:
		v.Op = OpAMD64VRSQRT14PD256
		return true
	case OpReciprocalSqrtFloat64x8:
		v.Op = OpAMD64VRSQRT14PD512
		return true
	case OpRotateAllLeftInt32x16:
		v.Op = OpAMD64VPROLD512
		return true
	case OpRotateAllLeftInt32x4:
		v.Op = OpAMD64VPROLD128
		return true
	case OpRotateAllLeftInt32x8:
		v.Op = OpAMD64VPROLD256
		return true
	case OpRotateAllLeftInt64x2:
		v.Op = OpAMD64VPROLQ128
		return true
	case OpRotateAllLeftInt64x4:
		v.Op = OpAMD64VPROLQ256
		return true
	case OpRotateAllLeftInt64x8:
		v.Op = OpAMD64VPROLQ512
		return true
	case OpRotateAllLeftUint32x16:
		v.Op = OpAMD64VPROLD512
		return true
	case OpRotateAllLeftUint32x4:
		v.Op = OpAMD64VPROLD128
		return true
	case OpRotateAllLeftUint32x8:
		v.Op = OpAMD64VPROLD256
		return true
	case OpRotateAllLeftUint64x2:
		v.Op = OpAMD64VPROLQ128
		return true
	case OpRotateAllLeftUint64x4:
		v.Op = OpAMD64VPROLQ256
		return true
	case OpRotateAllLeftUint64x8:
		v.Op = OpAMD64VPROLQ512
		return true
	case OpRotateAllRightInt32x16:
		v.Op = OpAMD64VPRORD512
		return true
	case OpRotateAllRightInt32x4:
		v.Op = OpAMD64VPRORD128
		return true
	case OpRotateAllRightInt32x8:
		v.Op = OpAMD64VPRORD256
		return true
	case OpRotateAllRightInt64x2:
		v.Op = OpAMD64VPRORQ128
		return true
	case OpRotateAllRightInt64x4:
		v.Op = OpAMD64VPRORQ256
		return true
	case OpRotateAllRightInt64x8:
		v.Op = OpAMD64VPRORQ512
		return true
	case OpRotateAllRightUint32x16:
		v.Op = OpAMD64VPRORD512
		return true
	case OpRotateAllRightUint32x4:
		v.Op = OpAMD64VPRORD128
		return true
	case OpRotateAllRightUint32x8:
		v.Op = OpAMD64VPRORD256
		return true
	case OpRotateAllRightUint64x2:
		v.Op = OpAMD64VPRORQ128
		return true
	case OpRotateAllRightUint64x4:
		v.Op = OpAMD64VPRORQ256
		return true
	case OpRotateAllRightUint64x8:
		v.Op = OpAMD64VPRORQ512
		return true
	case OpRotateLeft16:
		v.Op = OpAMD64ROLW
		return true
	case OpRotateLeft32:
		v.Op = OpAMD64ROLL
		return true
	case OpRotateLeft64:
		v.Op = OpAMD64ROLQ
		return true
	case OpRotateLeft8:
		v.Op = OpAMD64ROLB
		return true
	case OpRotateLeftInt32x16:
		v.Op = OpAMD64VPROLVD512
		return true
	case OpRotateLeftInt32x4:
		v.Op = OpAMD64VPROLVD128
		return true
	case OpRotateLeftInt32x8:
		v.Op = OpAMD64VPROLVD256
		return true
	case OpRotateLeftInt64x2:
		v.Op = OpAMD64VPROLVQ128
		return true
	case OpRotateLeftInt64x4:
		v.Op = OpAMD64VPROLVQ256
		return true
	case OpRotateLeftInt64x8:
		v.Op = OpAMD64VPROLVQ512
		return true
	case OpRotateLeftUint32x16:
		v.Op = OpAMD64VPROLVD512
		return true
	case OpRotateLeftUint32x4:
		v.Op = OpAMD64VPROLVD128
		return true
	case OpRotateLeftUint32x8:
		v.Op = OpAMD64VPROLVD256
		return true
	case OpRotateLeftUint64x2:
		v.Op = OpAMD64VPROLVQ128
		return true
	case OpRotateLeftUint64x4:
		v.Op = OpAMD64VPROLVQ256
		return true
	case OpRotateLeftUint64x8:
		v.Op = OpAMD64VPROLVQ512
		return true
	case OpRotateRightInt32x16:
		v.Op = OpAMD64VPRORVD512
		return true
	case OpRotateRightInt32x4:
		v.Op = OpAMD64VPRORVD128
		return true
	case OpRotateRightInt32x8:
		v.Op = OpAMD64VPRORVD256
		return true
	case OpRotateRightInt64x2:
		v.Op = OpAMD64VPRORVQ128
		return true
	case OpRotateRightInt64x4:
		v.Op = OpAMD64VPRORVQ256
		return true
	case OpRotateRightInt64x8:
		v.Op = OpAMD64VPRORVQ512
		return true
	case OpRotateRightUint32x16:
		v.Op = OpAMD64VPRORVD512
		return true
	case OpRotateRightUint32x4:
		v.Op = OpAMD64VPRORVD128
		return true
	case OpRotateRightUint32x8:
		v.Op = OpAMD64VPRORVD256
		return true
	case OpRotateRightUint64x2:
		v.Op = OpAMD64VPRORVQ128
		return true
	case OpRotateRightUint64x4:
		v.Op = OpAMD64VPRORVQ256
		return true
	case OpRotateRightUint64x8:
		v.Op = OpAMD64VPRORVQ512
		return true
	case OpRound32F:
		v.Op = OpAMD64LoweredRound32F
		return true
	case OpRound64F:
		v.Op = OpAMD64LoweredRound64F
		return true
	case OpRoundToEven:
		return rewriteValueAMD64_OpRoundToEven(v)
	case OpRoundToEvenFloat32x4:
		return rewriteValueAMD64_OpRoundToEvenFloat32x4(v)
	case OpRoundToEvenFloat32x8:
		return rewriteValueAMD64_OpRoundToEvenFloat32x8(v)
	case OpRoundToEvenFloat64x2:
		return rewriteValueAMD64_OpRoundToEvenFloat64x2(v)
	case OpRoundToEvenFloat64x4:
		return rewriteValueAMD64_OpRoundToEvenFloat64x4(v)
	case OpRoundToEvenScaledFloat32x16:
		return rewriteValueAMD64_OpRoundToEvenScaledFloat32x16(v)
	case OpRoundToEvenScaledFloat32x4:
		return rewriteValueAMD64_OpRoundToEvenScaledFloat32x4(v)
	case OpRoundToEvenScaledFloat32x8:
		return rewriteValueAMD64_OpRoundToEvenScaledFloat32x8(v)
	case OpRoundToEvenScaledFloat64x2:
		return rewriteValueAMD64_OpRoundToEvenScaledFloat64x2(v)
	case OpRoundToEvenScaledFloat64x4:
		return rewriteValueAMD64_OpRoundToEvenScaledFloat64x4(v)
	case OpRoundToEvenScaledFloat64x8:
		return rewriteValueAMD64_OpRoundToEvenScaledFloat64x8(v)
	case OpRoundToEvenScaledResidueFloat32x16:
		return rewriteValueAMD64_OpRoundToEvenScaledResidueFloat32x16(v)
	case OpRoundToEvenScaledResidueFloat32x4:
		return rewriteValueAMD64_OpRoundToEvenScaledResidueFloat32x4(v)
	case OpRoundToEvenScaledResidueFloat32x8:
		return rewriteValueAMD64_OpRoundToEvenScaledResidueFloat32x8(v)
	case OpRoundToEvenScaledResidueFloat64x2:
		return rewriteValueAMD64_OpRoundToEvenScaledResidueFloat64x2(v)
	case OpRoundToEvenScaledResidueFloat64x4:
		return rewriteValueAMD64_OpRoundToEvenScaledResidueFloat64x4(v)
	case OpRoundToEvenScaledResidueFloat64x8:
		return rewriteValueAMD64_OpRoundToEvenScaledResidueFloat64x8(v)
	case OpRsh16Ux16:
		return rewriteValueAMD64_OpRsh16Ux16(v)
	case OpRsh16Ux32:
		return rewriteValueAMD64_OpRsh16Ux32(v)
	case OpRsh16Ux64:
		return rewriteValueAMD64_OpRsh16Ux64(v)
	case OpRsh16Ux8:
		return rewriteValueAMD64_OpRsh16Ux8(v)
	case OpRsh16x16:
		return rewriteValueAMD64_OpRsh16x16(v)
	case OpRsh16x32:
		return rewriteValueAMD64_OpRsh16x32(v)
	case OpRsh16x64:
		return rewriteValueAMD64_OpRsh16x64(v)
	case OpRsh16x8:
		return rewriteValueAMD64_OpRsh16x8(v)
	case OpRsh32Ux16:
		return rewriteValueAMD64_OpRsh32Ux16(v)
	case OpRsh32Ux32:
		return rewriteValueAMD64_OpRsh32Ux32(v)
	case OpRsh32Ux64:
		return rewriteValueAMD64_OpRsh32Ux64(v)
	case OpRsh32Ux8:
		return rewriteValueAMD64_OpRsh32Ux8(v)
	case OpRsh32x16:
		return rewriteValueAMD64_OpRsh32x16(v)
	case OpRsh32x32:
		return rewriteValueAMD64_OpRsh32x32(v)
	case OpRsh32x64:
		return rewriteValueAMD64_OpRsh32x64(v)
	case OpRsh32x8:
		return rewriteValueAMD64_OpRsh32x8(v)
	case OpRsh64Ux16:
		return rewriteValueAMD64_OpRsh64Ux16(v)
	case OpRsh64Ux32:
		return rewriteValueAMD64_OpRsh64Ux32(v)
	case OpRsh64Ux64:
		return rewriteValueAMD64_OpRsh64Ux64(v)
	case OpRsh64Ux8:
		return rewriteValueAMD64_OpRsh64Ux8(v)
	case OpRsh64x16:
		return rewriteValueAMD64_OpRsh64x16(v)
	case OpRsh64x32:
		return rewriteValueAMD64_OpRsh64x32(v)
	case OpRsh64x64:
		return rewriteValueAMD64_OpRsh64x64(v)
	case OpRsh64x8:
		return rewriteValueAMD64_OpRsh64x8(v)
	case OpRsh8Ux16:
		return rewriteValueAMD64_OpRsh8Ux16(v)
	case OpRsh8Ux32:
		return rewriteValueAMD64_OpRsh8Ux32(v)
	case OpRsh8Ux64:
		return rewriteValueAMD64_OpRsh8Ux64(v)
	case OpRsh8Ux8:
		return rewriteValueAMD64_OpRsh8Ux8(v)
	case OpRsh8x16:
		return rewriteValueAMD64_OpRsh8x16(v)
	case OpRsh8x32:
		return rewriteValueAMD64_OpRsh8x32(v)
	case OpRsh8x64:
		return rewriteValueAMD64_OpRsh8x64(v)
	case OpRsh8x8:
		return rewriteValueAMD64_OpRsh8x8(v)
	case OpScaleFloat32x16:
		v.Op = OpAMD64VSCALEFPS512
		return true
	case OpScaleFloat32x4:
		v.Op = OpAMD64VSCALEFPS128
		return true
	case OpScaleFloat32x8:
		v.Op = OpAMD64VSCALEFPS256
		return true
	case OpScaleFloat64x2:
		v.Op = OpAMD64VSCALEFPD128
		return true
	case OpScaleFloat64x4:
		v.Op = OpAMD64VSCALEFPD256
		return true
	case OpScaleFloat64x8:
		v.Op = OpAMD64VSCALEFPD512
		return true
	case OpSelect0:
		return rewriteValueAMD64_OpSelect0(v)
	case OpSelect1:
		return rewriteValueAMD64_OpSelect1(v)
	case OpSelectN:
		return rewriteValueAMD64_OpSelectN(v)
	case OpSetElemFloat32x4:
		v.Op = OpAMD64VPINSRD128
		return true
	case OpSetElemFloat64x2:
		v.Op = OpAMD64VPINSRQ128
		return true
	case OpSetElemInt16x8:
		v.Op = OpAMD64VPINSRW128
		return true
	case OpSetElemInt32x4:
		v.Op = OpAMD64VPINSRD128
		return true
	case OpSetElemInt64x2:
		v.Op = OpAMD64VPINSRQ128
		return true
	case OpSetElemInt8x16:
		v.Op = OpAMD64VPINSRB128
		return true
	case OpSetElemUint16x8:
		v.Op = OpAMD64VPINSRW128
		return true
	case OpSetElemUint32x4:
		v.Op = OpAMD64VPINSRD128
		return true
	case OpSetElemUint64x2:
		v.Op = OpAMD64VPINSRQ128
		return true
	case OpSetElemUint8x16:
		v.Op = OpAMD64VPINSRB128
		return true
	case OpSetHiFloat32x16:
		return rewriteValueAMD64_OpSetHiFloat32x16(v)
	case OpSetHiFloat32x8:
		return rewriteValueAMD64_OpSetHiFloat32x8(v)
	case OpSetHiFloat64x4:
		return rewriteValueAMD64_OpSetHiFloat64x4(v)
	case OpSetHiFloat64x8:
		return rewriteValueAMD64_OpSetHiFloat64x8(v)
	case OpSetHiInt16x16:
		return rewriteValueAMD64_OpSetHiInt16x16(v)
	case OpSetHiInt16x32:
		return rewriteValueAMD64_OpSetHiInt16x32(v)
	case OpSetHiInt32x16:
		return rewriteValueAMD64_OpSetHiInt32x16(v)
	case OpSetHiInt32x8:
		return rewriteValueAMD64_OpSetHiInt32x8(v)
	case OpSetHiInt64x4:
		return rewriteValueAMD64_OpSetHiInt64x4(v)
	case OpSetHiInt64x8:
		return rewriteValueAMD64_OpSetHiInt64x8(v)
	case OpSetHiInt8x32:
		return rewriteValueAMD64_OpSetHiInt8x32(v)
	case OpSetHiInt8x64:
		return rewriteValueAMD64_OpSetHiInt8x64(v)
	case OpSetHiUint16x16:
		return rewriteValueAMD64_OpSetHiUint16x16(v)
	case OpSetHiUint16x32:
		return rewriteValueAMD64_OpSetHiUint16x32(v)
	case OpSetHiUint32x16:
		return rewriteValueAMD64_OpSetHiUint32x16(v)
	case OpSetHiUint32x8:
		return rewriteValueAMD64_OpSetHiUint32x8(v)
	case OpSetHiUint64x4:
		return rewriteValueAMD64_OpSetHiUint64x4(v)
	case OpSetHiUint64x8:
		return rewriteValueAMD64_OpSetHiUint64x8(v)
	case OpSetHiUint8x32:
		return rewriteValueAMD64_OpSetHiUint8x32(v)
	case OpSetHiUint8x64:
		return rewriteValueAMD64_OpSetHiUint8x64(v)
	case OpSetLoFloat32x16:
		return rewriteValueAMD64_OpSetLoFloat32x16(v)
	case OpSetLoFloat32x8:
		return rewriteValueAMD64_OpSetLoFloat32x8(v)
	case OpSetLoFloat64x4:
		return rewriteValueAMD64_OpSetLoFloat64x4(v)
	case OpSetLoFloat64x8:
		return rewriteValueAMD64_OpSetLoFloat64x8(v)
	case OpSetLoInt16x16:
		return rewriteValueAMD64_OpSetLoInt16x16(v)
	case OpSetLoInt16x32:
		return rewriteValueAMD64_OpSetLoInt16x32(v)
	case OpSetLoInt32x16:
		return rewriteValueAMD64_OpSetLoInt32x16(v)
	case OpSetLoInt32x8:
		return rewriteValueAMD64_OpSetLoInt32x8(v)
	case OpSetLoInt64x4:
		return rewriteValueAMD64_OpSetLoInt64x4(v)
	case OpSetLoInt64x8:
		return rewriteValueAMD64_OpSetLoInt64x8(v)
	case OpSetLoInt8x32:
		return rewriteValueAMD64_OpSetLoInt8x32(v)
	case OpSetLoInt8x64:
		return rewriteValueAMD64_OpSetLoInt8x64(v)
	case OpSetLoUint16x16:
		return rewriteValueAMD64_OpSetLoUint16x16(v)
	case OpSetLoUint16x32:
		return rewriteValueAMD64_OpSetLoUint16x32(v)
	case OpSetLoUint32x16:
		return rewriteValueAMD64_OpSetLoUint32x16(v)
	case OpSetLoUint32x8:
		return rewriteValueAMD64_OpSetLoUint32x8(v)
	case OpSetLoUint64x4:
		return rewriteValueAMD64_OpSetLoUint64x4(v)
	case OpSetLoUint64x8:
		return rewriteValueAMD64_OpSetLoUint64x8(v)
	case OpSetLoUint8x32:
		return rewriteValueAMD64_OpSetLoUint8x32(v)
	case OpSetLoUint8x64:
		return rewriteValueAMD64_OpSetLoUint8x64(v)
	case OpShiftAllLeftConcatInt16x16:
		v.Op = OpAMD64VPSHLDW256
		return true
	case OpShiftAllLeftConcatInt16x32:
		v.Op = OpAMD64VPSHLDW512
		return true
	case OpShiftAllLeftConcatInt16x8:
		v.Op = OpAMD64VPSHLDW128
		return true
	case OpShiftAllLeftConcatInt32x16:
		v.Op = OpAMD64VPSHLDD512
		return true
	case OpShiftAllLeftConcatInt32x4:
		v.Op = OpAMD64VPSHLDD128
		return true
	case OpShiftAllLeftConcatInt32x8:
		v.Op = OpAMD64VPSHLDD256
		return true
	case OpShiftAllLeftConcatInt64x2:
		v.Op = OpAMD64VPSHLDQ128
		return true
	case OpShiftAllLeftConcatInt64x4:
		v.Op = OpAMD64VPSHLDQ256
		return true
	case OpShiftAllLeftConcatInt64x8:
		v.Op = OpAMD64VPSHLDQ512
		return true
	case OpShiftAllLeftConcatUint16x16:
		v.Op = OpAMD64VPSHLDW256
		return true
	case OpShiftAllLeftConcatUint16x32:
		v.Op = OpAMD64VPSHLDW512
		return true
	case OpShiftAllLeftConcatUint16x8:
		v.Op = OpAMD64VPSHLDW128
		return true
	case OpShiftAllLeftConcatUint32x16:
		v.Op = OpAMD64VPSHLDD512
		return true
	case OpShiftAllLeftConcatUint32x4:
		v.Op = OpAMD64VPSHLDD128
		return true
	case OpShiftAllLeftConcatUint32x8:
		v.Op = OpAMD64VPSHLDD256
		return true
	case OpShiftAllLeftConcatUint64x2:
		v.Op = OpAMD64VPSHLDQ128
		return true
	case OpShiftAllLeftConcatUint64x4:
		v.Op = OpAMD64VPSHLDQ256
		return true
	case OpShiftAllLeftConcatUint64x8:
		v.Op = OpAMD64VPSHLDQ512
		return true
	case OpShiftAllLeftInt16x16:
		v.Op = OpAMD64VPSLLW256
		return true
	case OpShiftAllLeftInt16x32:
		v.Op = OpAMD64VPSLLW512
		return true
	case OpShiftAllLeftInt16x8:
		v.Op = OpAMD64VPSLLW128
		return true
	case OpShiftAllLeftInt32x16:
		v.Op = OpAMD64VPSLLD512
		return true
	case OpShiftAllLeftInt32x4:
		v.Op = OpAMD64VPSLLD128
		return true
	case OpShiftAllLeftInt32x8:
		v.Op = OpAMD64VPSLLD256
		return true
	case OpShiftAllLeftInt64x2:
		v.Op = OpAMD64VPSLLQ128
		return true
	case OpShiftAllLeftInt64x4:
		v.Op = OpAMD64VPSLLQ256
		return true
	case OpShiftAllLeftInt64x8:
		v.Op = OpAMD64VPSLLQ512
		return true
	case OpShiftAllLeftUint16x16:
		v.Op = OpAMD64VPSLLW256
		return true
	case OpShiftAllLeftUint16x32:
		v.Op = OpAMD64VPSLLW512
		return true
	case OpShiftAllLeftUint16x8:
		v.Op = OpAMD64VPSLLW128
		return true
	case OpShiftAllLeftUint32x16:
		v.Op = OpAMD64VPSLLD512
		return true
	case OpShiftAllLeftUint32x4:
		v.Op = OpAMD64VPSLLD128
		return true
	case OpShiftAllLeftUint32x8:
		v.Op = OpAMD64VPSLLD256
		return true
	case OpShiftAllLeftUint64x2:
		v.Op = OpAMD64VPSLLQ128
		return true
	case OpShiftAllLeftUint64x4:
		v.Op = OpAMD64VPSLLQ256
		return true
	case OpShiftAllLeftUint64x8:
		v.Op = OpAMD64VPSLLQ512
		return true
	case OpShiftAllRightConcatInt16x16:
		v.Op = OpAMD64VPSHRDW256
		return true
	case OpShiftAllRightConcatInt16x32:
		v.Op = OpAMD64VPSHRDW512
		return true
	case OpShiftAllRightConcatInt16x8:
		v.Op = OpAMD64VPSHRDW128
		return true
	case OpShiftAllRightConcatInt32x16:
		v.Op = OpAMD64VPSHRDD512
		return true
	case OpShiftAllRightConcatInt32x4:
		v.Op = OpAMD64VPSHRDD128
		return true
	case OpShiftAllRightConcatInt32x8:
		v.Op = OpAMD64VPSHRDD256
		return true
	case OpShiftAllRightConcatInt64x2:
		v.Op = OpAMD64VPSHRDQ128
		return true
	case OpShiftAllRightConcatInt64x4:
		v.Op = OpAMD64VPSHRDQ256
		return true
	case OpShiftAllRightConcatInt64x8:
		v.Op = OpAMD64VPSHRDQ512
		return true
	case OpShiftAllRightConcatUint16x16:
		v.Op = OpAMD64VPSHRDW256
		return true
	case OpShiftAllRightConcatUint16x32:
		v.Op = OpAMD64VPSHRDW512
		return true
	case OpShiftAllRightConcatUint16x8:
		v.Op = OpAMD64VPSHRDW128
		return true
	case OpShiftAllRightConcatUint32x16:
		v.Op = OpAMD64VPSHRDD512
		return true
	case OpShiftAllRightConcatUint32x4:
		v.Op = OpAMD64VPSHRDD128
		return true
	case OpShiftAllRightConcatUint32x8:
		v.Op = OpAMD64VPSHRDD256
		return true
	case OpShiftAllRightConcatUint64x2:
		v.Op = OpAMD64VPSHRDQ128
		return true
	case OpShiftAllRightConcatUint64x4:
		v.Op = OpAMD64VPSHRDQ256
		return true
	case OpShiftAllRightConcatUint64x8:
		v.Op = OpAMD64VPSHRDQ512
		return true
	case OpShiftAllRightInt16x16:
		v.Op = OpAMD64VPSRAW256
		return true
	case OpShiftAllRightInt16x32:
		v.Op = OpAMD64VPSRAW512
		return true
	case OpShiftAllRightInt16x8:
		v.Op = OpAMD64VPSRAW128
		return true
	case OpShiftAllRightInt32x16:
		v.Op = OpAMD64VPSRAD512
		return true
	case OpShiftAllRightInt32x4:
		v.Op = OpAMD64VPSRAD128
		return true
	case OpShiftAllRightInt32x8:
		v.Op = OpAMD64VPSRAD256
		return true
	case OpShiftAllRightInt64x2:
		v.Op = OpAMD64VPSRAQ128
		return true
	case OpShiftAllRightInt64x4:
		v.Op = OpAMD64VPSRAQ256
		return true
	case OpShiftAllRightInt64x8:
		v.Op = OpAMD64VPSRAQ512
		return true
	case OpShiftAllRightUint16x16:
		v.Op = OpAMD64VPSRLW256
		return true
	case OpShiftAllRightUint16x32:
		v.Op = OpAMD64VPSRLW512
		return true
	case OpShiftAllRightUint16x8:
		v.Op = OpAMD64VPSRLW128
		return true
	case OpShiftAllRightUint32x16:
		v.Op = OpAMD64VPSRLD512
		return true
	case OpShiftAllRightUint32x4:
		v.Op = OpAMD64VPSRLD128
		return true
	case OpShiftAllRightUint32x8:
		v.Op = OpAMD64VPSRLD256
		return true
	case OpShiftAllRightUint64x2:
		v.Op = OpAMD64VPSRLQ128
		return true
	case OpShiftAllRightUint64x4:
		v.Op = OpAMD64VPSRLQ256
		return true
	case OpShiftAllRightUint64x8:
		v.Op = OpAMD64VPSRLQ512
		return true
	case OpShiftLeftConcatInt16x16:
		v.Op = OpAMD64VPSHLDVW256
		return true
	case OpShiftLeftConcatInt16x32:
		v.Op = OpAMD64VPSHLDVW512
		return true
	case OpShiftLeftConcatInt16x8:
		v.Op = OpAMD64VPSHLDVW128
		return true
	case OpShiftLeftConcatInt32x16:
		v.Op = OpAMD64VPSHLDVD512
		return true
	case OpShiftLeftConcatInt32x4:
		v.Op = OpAMD64VPSHLDVD128
		return true
	case OpShiftLeftConcatInt32x8:
		v.Op = OpAMD64VPSHLDVD256
		return true
	case OpShiftLeftConcatInt64x2:
		v.Op = OpAMD64VPSHLDVQ128
		return true
	case OpShiftLeftConcatInt64x4:
		v.Op = OpAMD64VPSHLDVQ256
		return true
	case OpShiftLeftConcatInt64x8:
		v.Op = OpAMD64VPSHLDVQ512
		return true
	case OpShiftLeftConcatUint16x16:
		v.Op = OpAMD64VPSHLDVW256
		return true
	case OpShiftLeftConcatUint16x32:
		v.Op = OpAMD64VPSHLDVW512
		return true
	case OpShiftLeftConcatUint16x8:
		v.Op = OpAMD64VPSHLDVW128
		return true
	case OpShiftLeftConcatUint32x16:
		v.Op = OpAMD64VPSHLDVD512
		return true
	case OpShiftLeftConcatUint32x4:
		v.Op = OpAMD64VPSHLDVD128
		return true
	case OpShiftLeftConcatUint32x8:
		v.Op = OpAMD64VPSHLDVD256
		return true
	case OpShiftLeftConcatUint64x2:
		v.Op = OpAMD64VPSHLDVQ128
		return true
	case OpShiftLeftConcatUint64x4:
		v.Op = OpAMD64VPSHLDVQ256
		return true
	case OpShiftLeftConcatUint64x8:
		v.Op = OpAMD64VPSHLDVQ512
		return true
	case OpShiftLeftInt16x16:
		v.Op = OpAMD64VPSLLVW256
		return true
	case OpShiftLeftInt16x32:
		v.Op = OpAMD64VPSLLVW512
		return true
	case OpShiftLeftInt16x8:
		v.Op = OpAMD64VPSLLVW128
		return true
	case OpShiftLeftInt32x16:
		v.Op = OpAMD64VPSLLVD512
		return true
	case OpShiftLeftInt32x4:
		v.Op = OpAMD64VPSLLVD128
		return true
	case OpShiftLeftInt32x8:
		v.Op = OpAMD64VPSLLVD256
		return true
	case OpShiftLeftInt64x2:
		v.Op = OpAMD64VPSLLVQ128
		return true
	case OpShiftLeftInt64x4:
		v.Op = OpAMD64VPSLLVQ256
		return true
	case OpShiftLeftInt64x8:
		v.Op = OpAMD64VPSLLVQ512
		return true
	case OpShiftLeftUint16x16:
		v.Op = OpAMD64VPSLLVW256
		return true
	case OpShiftLeftUint16x32:
		v.Op = OpAMD64VPSLLVW512
		return true
	case OpShiftLeftUint16x8:
		v.Op = OpAMD64VPSLLVW128
		return true
	case OpShiftLeftUint32x16:
		v.Op = OpAMD64VPSLLVD512
		return true
	case OpShiftLeftUint32x4:
		v.Op = OpAMD64VPSLLVD128
		return true
	case OpShiftLeftUint32x8:
		v.Op = OpAMD64VPSLLVD256
		return true
	case OpShiftLeftUint64x2:
		v.Op = OpAMD64VPSLLVQ128
		return true
	case OpShiftLeftUint64x4:
		v.Op = OpAMD64VPSLLVQ256
		return true
	case OpShiftLeftUint64x8:
		v.Op = OpAMD64VPSLLVQ512
		return true
	case OpShiftRightConcatInt16x16:
		v.Op = OpAMD64VPSHRDVW256
		return true
	case OpShiftRightConcatInt16x32:
		v.Op = OpAMD64VPSHRDVW512
		return true
	case OpShiftRightConcatInt16x8:
		v.Op = OpAMD64VPSHRDVW128
		return true
	case OpShiftRightConcatInt32x16:
		v.Op = OpAMD64VPSHRDVD512
		return true
	case OpShiftRightConcatInt32x4:
		v.Op = OpAMD64VPSHRDVD128
		return true
	case OpShiftRightConcatInt32x8:
		v.Op = OpAMD64VPSHRDVD256
		return true
	case OpShiftRightConcatInt64x2:
		v.Op = OpAMD64VPSHRDVQ128
		return true
	case OpShiftRightConcatInt64x4:
		v.Op = OpAMD64VPSHRDVQ256
		return true
	case OpShiftRightConcatInt64x8:
		v.Op = OpAMD64VPSHRDVQ512
		return true
	case OpShiftRightConcatUint16x16:
		v.Op = OpAMD64VPSHRDVW256
		return true
	case OpShiftRightConcatUint16x32:
		v.Op = OpAMD64VPSHRDVW512
		return true
	case OpShiftRightConcatUint16x8:
		v.Op = OpAMD64VPSHRDVW128
		return true
	case OpShiftRightConcatUint32x16:
		v.Op = OpAMD64VPSHRDVD512
		return true
	case OpShiftRightConcatUint32x4:
		v.Op = OpAMD64VPSHRDVD128
		return true
	case OpShiftRightConcatUint32x8:
		v.Op = OpAMD64VPSHRDVD256
		return true
	case OpShiftRightConcatUint64x2:
		v.Op = OpAMD64VPSHRDVQ128
		return true
	case OpShiftRightConcatUint64x4:
		v.Op = OpAMD64VPSHRDVQ256
		return true
	case OpShiftRightConcatUint64x8:
		v.Op = OpAMD64VPSHRDVQ512
		return true
	case OpShiftRightInt16x16:
		v.Op = OpAMD64VPSRAVW256
		return true
	case OpShiftRightInt16x32:
		v.Op = OpAMD64VPSRAVW512
		return true
	case OpShiftRightInt16x8:
		v.Op = OpAMD64VPSRAVW128
		return true
	case OpShiftRightInt32x16:
		v.Op = OpAMD64VPSRAVD512
		return true
	case OpShiftRightInt32x4:
		v.Op = OpAMD64VPSRAVD128
		return true
	case OpShiftRightInt32x8:
		v.Op = OpAMD64VPSRAVD256
		return true
	case OpShiftRightInt64x2:
		v.Op = OpAMD64VPSRAVQ128
		return true
	case OpShiftRightInt64x4:
		v.Op = OpAMD64VPSRAVQ256
		return true
	case OpShiftRightInt64x8:
		v.Op = OpAMD64VPSRAVQ512
		return true
	case OpShiftRightUint16x16:
		v.Op = OpAMD64VPSRLVW256
		return true
	case OpShiftRightUint16x32:
		v.Op = OpAMD64VPSRLVW512
		return true
	case OpShiftRightUint16x8:
		v.Op = OpAMD64VPSRLVW128
		return true
	case OpShiftRightUint32x16:
		v.Op = OpAMD64VPSRLVD512
		return true
	case OpShiftRightUint32x4:
		v.Op = OpAMD64VPSRLVD128
		return true
	case OpShiftRightUint32x8:
		v.Op = OpAMD64VPSRLVD256
		return true
	case OpShiftRightUint64x2:
		v.Op = OpAMD64VPSRLVQ128
		return true
	case OpShiftRightUint64x4:
		v.Op = OpAMD64VPSRLVQ256
		return true
	case OpShiftRightUint64x8:
		v.Op = OpAMD64VPSRLVQ512
		return true
	case OpSignExt16to32:
		v.Op = OpAMD64MOVWQSX
		return true
	case OpSignExt16to64:
		v.Op = OpAMD64MOVWQSX
		return true
	case OpSignExt32to64:
		v.Op = OpAMD64MOVLQSX
		return true
	case OpSignExt8to16:
		v.Op = OpAMD64MOVBQSX
		return true
	case OpSignExt8to32:
		v.Op = OpAMD64MOVBQSX
		return true
	case OpSignExt8to64:
		v.Op = OpAMD64MOVBQSX
		return true
	case OpSlicemask:
		return rewriteValueAMD64_OpSlicemask(v)
	case OpSpectreIndex:
		return rewriteValueAMD64_OpSpectreIndex(v)
	case OpSpectreSliceIndex:
		return rewriteValueAMD64_OpSpectreSliceIndex(v)
	case OpSqrt:
		v.Op = OpAMD64SQRTSD
		return true
	case OpSqrt32:
		v.Op = OpAMD64SQRTSS
		return true
	case OpSqrtFloat32x16:
		v.Op = OpAMD64VSQRTPS512
		return true
	case OpSqrtFloat32x4:
		v.Op = OpAMD64VSQRTPS128
		return true
	case OpSqrtFloat32x8:
		v.Op = OpAMD64VSQRTPS256
		return true
	case OpSqrtFloat64x2:
		v.Op = OpAMD64VSQRTPD128
		return true
	case OpSqrtFloat64x4:
		v.Op = OpAMD64VSQRTPD256
		return true
	case OpSqrtFloat64x8:
		v.Op = OpAMD64VSQRTPD512
		return true
	case OpStaticCall:
		v.Op = OpAMD64CALLstatic
		return true
	case OpStore:
		return rewriteValueAMD64_OpStore(v)
	case OpStoreMask16x16:
		return rewriteValueAMD64_OpStoreMask16x16(v)
	case OpStoreMask16x32:
		return rewriteValueAMD64_OpStoreMask16x32(v)
	case OpStoreMask16x8:
		return rewriteValueAMD64_OpStoreMask16x8(v)
	case OpStoreMask32x16:
		return rewriteValueAMD64_OpStoreMask32x16(v)
	case OpStoreMask32x4:
		return rewriteValueAMD64_OpStoreMask32x4(v)
	case OpStoreMask32x8:
		return rewriteValueAMD64_OpStoreMask32x8(v)
	case OpStoreMask64x2:
		return rewriteValueAMD64_OpStoreMask64x2(v)
	case OpStoreMask64x4:
		return rewriteValueAMD64_OpStoreMask64x4(v)
	case OpStoreMask64x8:
		return rewriteValueAMD64_OpStoreMask64x8(v)
	case OpStoreMask8x16:
		return rewriteValueAMD64_OpStoreMask8x16(v)
	case OpStoreMask8x32:
		return rewriteValueAMD64_OpStoreMask8x32(v)
	case OpStoreMask8x64:
		return rewriteValueAMD64_OpStoreMask8x64(v)
	case OpStoreMasked16:
		return rewriteValueAMD64_OpStoreMasked16(v)
	case OpStoreMasked32:
		return rewriteValueAMD64_OpStoreMasked32(v)
	case OpStoreMasked64:
		return rewriteValueAMD64_OpStoreMasked64(v)
	case OpStoreMasked8:
		return rewriteValueAMD64_OpStoreMasked8(v)
	case OpSub16:
		v.Op = OpAMD64SUBL
		return true
	case OpSub32:
		v.Op = OpAMD64SUBL
		return true
	case OpSub32F:
		v.Op = OpAMD64SUBSS
		return true
	case OpSub64:
		v.Op = OpAMD64SUBQ
		return true
	case OpSub64F:
		v.Op = OpAMD64SUBSD
		return true
	case OpSub8:
		v.Op = OpAMD64SUBL
		return true
	case OpSubFloat32x16:
		v.Op = OpAMD64VSUBPS512
		return true
	case OpSubFloat32x4:
		v.Op = OpAMD64VSUBPS128
		return true
	case OpSubFloat32x8:
		v.Op = OpAMD64VSUBPS256
		return true
	case OpSubFloat64x2:
		v.Op = OpAMD64VSUBPD128
		return true
	case OpSubFloat64x4:
		v.Op = OpAMD64VSUBPD256
		return true
	case OpSubFloat64x8:
		v.Op = OpAMD64VSUBPD512
		return true
	case OpSubInt16x16:
		v.Op = OpAMD64VPSUBW256
		return true
	case OpSubInt16x32:
		v.Op = OpAMD64VPSUBW512
		return true
	case OpSubInt16x8:
		v.Op = OpAMD64VPSUBW128
		return true
	case OpSubInt32x16:
		v.Op = OpAMD64VPSUBD512
		return true
	case OpSubInt32x4:
		v.Op = OpAMD64VPSUBD128
		return true
	case OpSubInt32x8:
		v.Op = OpAMD64VPSUBD256
		return true
	case OpSubInt64x2:
		v.Op = OpAMD64VPSUBQ128
		return true
	case OpSubInt64x4:
		v.Op = OpAMD64VPSUBQ256
		return true
	case OpSubInt64x8:
		v.Op = OpAMD64VPSUBQ512
		return true
	case OpSubInt8x16:
		v.Op = OpAMD64VPSUBB128
		return true
	case OpSubInt8x32:
		v.Op = OpAMD64VPSUBB256
		return true
	case OpSubInt8x64:
		v.Op = OpAMD64VPSUBB512
		return true
	case OpSubPairsFloat32x4:
		v.Op = OpAMD64VHSUBPS128
		return true
	case OpSubPairsFloat32x8:
		v.Op = OpAMD64VHSUBPS256
		return true
	case OpSubPairsFloat64x2:
		v.Op = OpAMD64VHSUBPD128
		return true
	case OpSubPairsFloat64x4:
		v.Op = OpAMD64VHSUBPD256
		return true
	case OpSubPairsInt16x16:
		v.Op = OpAMD64VPHSUBW256
		return true
	case OpSubPairsInt16x8:
		v.Op = OpAMD64VPHSUBW128
		return true
	case OpSubPairsInt32x4:
		v.Op = OpAMD64VPHSUBD128
		return true
	case OpSubPairsInt32x8:
		v.Op = OpAMD64VPHSUBD256
		return true
	case OpSubPairsSaturatedInt16x16:
		v.Op = OpAMD64VPHSUBSW256
		return true
	case OpSubPairsSaturatedInt16x8:
		v.Op = OpAMD64VPHSUBSW128
		return true
	case OpSubPairsUint16x16:
		v.Op = OpAMD64VPHSUBW256
		return true
	case OpSubPairsUint16x8:
		v.Op = OpAMD64VPHSUBW128
		return true
	case OpSubPairsUint32x4:
		v.Op = OpAMD64VPHSUBD128
		return true
	case OpSubPairsUint32x8:
		v.Op = OpAMD64VPHSUBD256
		return true
	case OpSubPtr:
		v.Op = OpAMD64SUBQ
		return true
	case OpSubSaturatedInt16x16:
		v.Op = OpAMD64VPSUBSW256
		return true
	case OpSubSaturatedInt16x32:
		v.Op = OpAMD64VPSUBSW512
		return true
	case OpSubSaturatedInt16x8:
		v.Op = OpAMD64VPSUBSW128
		return true
	case OpSubSaturatedInt8x16:
		v.Op = OpAMD64VPSUBSB128
		return true
	case OpSubSaturatedInt8x32:
		v.Op = OpAMD64VPSUBSB256
		return true
	case OpSubSaturatedInt8x64:
		v.Op = OpAMD64VPSUBSB512
		return true
	case OpSubSaturatedUint16x16:
		v.Op = OpAMD64VPSUBUSW256
		return true
	case OpSubSaturatedUint16x32:
		v.Op = OpAMD64VPSUBUSW512
		return true
	case OpSubSaturatedUint16x8:
		v.Op = OpAMD64VPSUBUSW128
		return true
	case OpSubSaturatedUint8x16:
		v.Op = OpAMD64VPSUBUSB128
		return true
	case OpSubSaturatedUint8x32:
		v.Op = OpAMD64VPSUBUSB256
		return true
	case OpSubSaturatedUint8x64:
		v.Op = OpAMD64VPSUBUSB512
		return true
	case OpSubUint16x16:
		v.Op = OpAMD64VPSUBW256
		return true
	case OpSubUint16x32:
		v.Op = OpAMD64VPSUBW512
		return true
	case OpSubUint16x8:
		v.Op = OpAMD64VPSUBW128
		return true
	case OpSubUint32x16:
		v.Op = OpAMD64VPSUBD512
		return true
	case OpSubUint32x4:
		v.Op = OpAMD64VPSUBD128
		return true
	case OpSubUint32x8:
		v.Op = OpAMD64VPSUBD256
		return true
	case OpSubUint64x2:
		v.Op = OpAMD64VPSUBQ128
		return true
	case OpSubUint64x4:
		v.Op = OpAMD64VPSUBQ256
		return true
	case OpSubUint64x8:
		v.Op = OpAMD64VPSUBQ512
		return true
	case OpSubUint8x16:
		v.Op = OpAMD64VPSUBB128
		return true
	case OpSubUint8x32:
		v.Op = OpAMD64VPSUBB256
		return true
	case OpSubUint8x64:
		v.Op = OpAMD64VPSUBB512
		return true
	case OpTailCall:
		v.Op = OpAMD64CALLtail
		return true
	case OpTrunc:
		return rewriteValueAMD64_OpTrunc(v)
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
	case OpTruncFloat32x4:
		return rewriteValueAMD64_OpTruncFloat32x4(v)
	case OpTruncFloat32x8:
		return rewriteValueAMD64_OpTruncFloat32x8(v)
	case OpTruncFloat64x2:
		return rewriteValueAMD64_OpTruncFloat64x2(v)
	case OpTruncFloat64x4:
		return rewriteValueAMD64_OpTruncFloat64x4(v)
	case OpTruncScaledFloat32x16:
		return rewriteValueAMD64_OpTruncScaledFloat32x16(v)
	case OpTruncScaledFloat32x4:
		return rewriteValueAMD64_OpTruncScaledFloat32x4(v)
	case OpTruncScaledFloat32x8:
		return rewriteValueAMD64_OpTruncScaledFloat32x8(v)
	case OpTruncScaledFloat64x2:
		return rewriteValueAMD64_OpTruncScaledFloat64x2(v)
	case OpTruncScaledFloat64x4:
		return rewriteValueAMD64_OpTruncScaledFloat64x4(v)
	case OpTruncScaledFloat64x8:
		return rewriteValueAMD64_OpTruncScaledFloat64x8(v)
	case OpTruncScaledResidueFloat32x16:
		return rewriteValueAMD64_OpTruncScaledResidueFloat32x16(v)
	case OpTruncScaledResidueFloat32x4:
		return rewriteValueAMD64_OpTruncScaledResidueFloat32x4(v)
	case OpTruncScaledResidueFloat32x8:
		return rewriteValueAMD64_OpTruncScaledResidueFloat32x8(v)
	case OpTruncScaledResidueFloat64x2:
		return rewriteValueAMD64_OpTruncScaledResidueFloat64x2(v)
	case OpTruncScaledResidueFloat64x4:
		return rewriteValueAMD64_OpTruncScaledResidueFloat64x4(v)
	case OpTruncScaledResidueFloat64x8:
		return rewriteValueAMD64_OpTruncScaledResidueFloat64x8(v)
	case OpWB:
		v.Op = OpAMD64LoweredWB
		return true
	case OpXor16:
		v.Op = OpAMD64XORL
		return true
	case OpXor32:
		v.Op = OpAMD64XORL
		return true
	case OpXor64:
		v.Op = OpAMD64XORQ
		return true
	case OpXor8:
		v.Op = OpAMD64XORL
		return true
	case OpXorInt16x16:
		v.Op = OpAMD64VPXOR256
		return true
	case OpXorInt16x32:
		v.Op = OpAMD64VPXORD512
		return true
	case OpXorInt16x8:
		v.Op = OpAMD64VPXOR128
		return true
	case OpXorInt32x16:
		v.Op = OpAMD64VPXORD512
		return true
	case OpXorInt32x4:
		v.Op = OpAMD64VPXOR128
		return true
	case OpXorInt32x8:
		v.Op = OpAMD64VPXOR256
		return true
	case OpXorInt64x2:
		v.Op = OpAMD64VPXOR128
		return true
	case OpXorInt64x4:
		v.Op = OpAMD64VPXOR256
		return true
	case OpXorInt64x8:
		v.Op = OpAMD64VPXORQ512
		return true
	case OpXorInt8x16:
		v.Op = OpAMD64VPXOR128
		return true
	case OpXorInt8x32:
		v.Op = OpAMD64VPXOR256
		return true
	case OpXorInt8x64:
		v.Op = OpAMD64VPXORD512
		return true
	case OpXorUint16x16:
		v.Op = OpAMD64VPXOR256
		return true
	case OpXorUint16x32:
		v.Op = OpAMD64VPXORD512
		return true
	case OpXorUint16x8:
		v.Op = OpAMD64VPXOR128
		return true
	case OpXorUint32x16:
		v.Op = OpAMD64VPXORD512
		return true
	case OpXorUint32x4:
		v.Op = OpAMD64VPXOR128
		return true
	case OpXorUint32x8:
		v.Op = OpAMD64VPXOR256
		return true
	case OpXorUint64x2:
		v.Op = OpAMD64VPXOR128
		return true
	case OpXorUint64x4:
		v.Op = OpAMD64VPXOR256
		return true
	case OpXorUint64x8:
		v.Op = OpAMD64VPXORQ512
		return true
	case OpXorUint8x16:
		v.Op = OpAMD64VPXOR128
		return true
	case OpXorUint8x32:
		v.Op = OpAMD64VPXOR256
		return true
	case OpXorUint8x64:
		v.Op = OpAMD64VPXORD512
		return true
	case OpZero:
		return rewriteValueAMD64_OpZero(v)
	case OpZeroExt16to32:
		v.Op = OpAMD64MOVWQZX
		return true
	case OpZeroExt16to64:
		v.Op = OpAMD64MOVWQZX
		return true
	case OpZeroExt32to64:
		v.Op = OpAMD64MOVLQZX
		return true
	case OpZeroExt8to16:
		v.Op = OpAMD64MOVBQZX
		return true
	case OpZeroExt8to32:
		v.Op = OpAMD64MOVBQZX
		return true
	case OpZeroExt8to64:
		v.Op = OpAMD64MOVBQZX
		return true
	case OpZeroSIMD:
		return rewriteValueAMD64_OpZeroSIMD(v)
	case OpblendInt8x16:
		v.Op = OpAMD64VPBLENDVB128
		return true
	case OpblendInt8x32:
		v.Op = OpAMD64VPBLENDVB256
		return true
	case OpblendMaskedInt16x32:
		return rewriteValueAMD64_OpblendMaskedInt16x32(v)
	case OpblendMaskedInt32x16:
		return rewriteValueAMD64_OpblendMaskedInt32x16(v)
	case OpblendMaskedInt64x8:
		return rewriteValueAMD64_OpblendMaskedInt64x8(v)
	case OpblendMaskedInt8x64:
		return rewriteValueAMD64_OpblendMaskedInt8x64(v)
	case OpmoveMaskedFloat32x16:
		return rewriteValueAMD64_OpmoveMaskedFloat32x16(v)
	case OpmoveMaskedFloat64x8:
		return rewriteValueAMD64_OpmoveMaskedFloat64x8(v)
	case OpmoveMaskedInt16x32:
		return rewriteValueAMD64_OpmoveMaskedInt16x32(v)
	case OpmoveMaskedInt32x16:
		return rewriteValueAMD64_OpmoveMaskedInt32x16(v)
	case OpmoveMaskedInt64x8:
		return rewriteValueAMD64_OpmoveMaskedInt64x8(v)
	case OpmoveMaskedInt8x64:
		return rewriteValueAMD64_OpmoveMaskedInt8x64(v)
	case OpmoveMaskedUint16x32:
		return rewriteValueAMD64_OpmoveMaskedUint16x32(v)
	case OpmoveMaskedUint32x16:
		return rewriteValueAMD64_OpmoveMaskedUint32x16(v)
	case OpmoveMaskedUint64x8:
		return rewriteValueAMD64_OpmoveMaskedUint64x8(v)
	case OpmoveMaskedUint8x64:
		return rewriteValueAMD64_OpmoveMaskedUint8x64(v)
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADCQ(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADCQ x (MOVQconst [c]) carry)
	// cond: is32Bit(c)
	// result: (ADCQconst x [int32(c)] carry)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64MOVQconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			carry := v_2
			if !(is32Bit(c)) {
				continue
			}
			v.reset(OpAMD64ADCQconst)
			v.AuxInt = int32ToAuxInt(int32(c))
			v.AddArg2(x, carry)
			return true
		}
		break
	}
	// match: (ADCQ x y (FlagEQ))
	// result: (ADDQcarry x y)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64ADDQcarry)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADCQconst(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADCQconst x [c] (FlagEQ))
	// result: (ADDQconstcarry x [c])
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64ADDQconstcarry)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDL (SHRLconst [1] x) (SHRLconst [1] x))
	// result: (ANDLconst [-2] x)
	for {
		if v_0.Op != OpAMD64SHRLconst || auxIntToInt8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		if v_1.Op != OpAMD64SHRLconst || auxIntToInt8(v_1.AuxInt) != 1 || x != v_1.Args[0] {
			break
		}
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = int32ToAuxInt(-2)
		v.AddArg(x)
		return true
	}
	// match: (ADDL x (MOVLconst [c]))
	// result: (ADDLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64MOVLconst {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			v.reset(OpAMD64ADDLconst)
			v.AuxInt = int32ToAuxInt(c)
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
			if v_1.Op != OpAMD64SHLLconst || auxIntToInt8(v_1.AuxInt) != 3 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpAMD64LEAL8)
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
			if v_1.Op != OpAMD64SHLLconst || auxIntToInt8(v_1.AuxInt) != 2 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpAMD64LEAL4)
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
			if v_1.Op != OpAMD64ADDL {
				continue
			}
			y := v_1.Args[1]
			if y != v_1.Args[0] {
				continue
			}
			v.reset(OpAMD64LEAL2)
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
			if v_1.Op != OpAMD64ADDL {
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
				v.reset(OpAMD64LEAL2)
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
			if v_0.Op != OpAMD64ADDLconst {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			y := v_1
			v.reset(OpAMD64LEAL1)
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
			if v_1.Op != OpAMD64LEAL {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			s := auxToSym(v_1.Aux)
			y := v_1.Args[0]
			if !(x.Op != OpSB && y.Op != OpSB) {
				continue
			}
			v.reset(OpAMD64LEAL1)
			v.AuxInt = int32ToAuxInt(c)
			v.Aux = symToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADDL x (NEGL y))
	// result: (SUBL x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64NEGL {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpAMD64SUBL)
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
			if l.Op != OpAMD64MOVLload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(OpAMD64ADDLload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ADDLconst [c] (ADDL x y))
	// result: (LEAL1 [c] x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64ADDL {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpAMD64LEAL1)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (ADDLconst [c] (ADDL x x))
	// result: (LEAL1 [c] x x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64ADDL {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpAMD64LEAL1)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg2(x, x)
		return true
	}
	// match: (ADDLconst [c] (LEAL [d] {s} x))
	// cond: is32Bit(int64(c)+int64(d))
	// result: (LEAL [c+d] {s} x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64LEAL {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		s := auxToSym(v_0.Aux)
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(OpAMD64LEAL)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg(x)
		return true
	}
	// match: (ADDLconst [c] (LEAL1 [d] {s} x y))
	// cond: is32Bit(int64(c)+int64(d))
	// result: (LEAL1 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64LEAL1 {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		s := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(OpAMD64LEAL1)
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
		if v_0.Op != OpAMD64LEAL2 {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		s := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(OpAMD64LEAL2)
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
		if v_0.Op != OpAMD64LEAL4 {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		s := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(OpAMD64LEAL4)
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
		if v_0.Op != OpAMD64LEAL8 {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		s := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(OpAMD64LEAL8)
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
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(c + d)
		return true
	}
	// match: (ADDLconst [c] (ADDLconst [d] x))
	// result: (ADDLconst [c+d] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64ADDLconst)
		v.AuxInt = int32ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	// match: (ADDLconst [off] x:(SP))
	// result: (LEAL [off] x)
	for {
		off := auxIntToInt32(v.AuxInt)
		x := v_0
		if x.Op != OpSP {
			break
		}
		v.reset(OpAMD64LEAL)
		v.AuxInt = int32ToAuxInt(off)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDLconstmodify(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDLconstmodify [valoff1] {sym} (ADDQconst [off2] base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2)
	// result: (ADDLconstmodify [ValAndOff(valoff1).addOffset32(off2)] {sym} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2)) {
			break
		}
		v.reset(OpAMD64ADDLconstmodify)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (ADDLconstmodify [valoff1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)
	// result: (ADDLconstmodify [ValAndOff(valoff1).addOffset32(off2)] {mergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64ADDLconstmodify)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDLload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADDLload [off1] {sym} val (ADDQconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ADDLload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64ADDLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ADDLload [off1] {sym1} val (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (ADDLload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64ADDLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ADDLload x [off] {sym} ptr (MOVSSstore [off] {sym} ptr y _))
	// result: (ADDL x (MOVLf2i y))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		ptr := v_1
		if v_2.Op != OpAMD64MOVSSstore || auxIntToInt32(v_2.AuxInt) != off || auxToSym(v_2.Aux) != sym {
			break
		}
		y := v_2.Args[1]
		if ptr != v_2.Args[0] {
			break
		}
		v.reset(OpAMD64ADDL)
		v0 := b.NewValue0(v_2.Pos, OpAMD64MOVLf2i, typ.UInt32)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDLmodify(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDLmodify [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ADDLmodify [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64ADDLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (ADDLmodify [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (ADDLmodify [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64ADDLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDQ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDQ (SHRQconst [1] x) (SHRQconst [1] x))
	// result: (ANDQconst [-2] x)
	for {
		if v_0.Op != OpAMD64SHRQconst || auxIntToInt8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		if v_1.Op != OpAMD64SHRQconst || auxIntToInt8(v_1.AuxInt) != 1 || x != v_1.Args[0] {
			break
		}
		v.reset(OpAMD64ANDQconst)
		v.AuxInt = int32ToAuxInt(-2)
		v.AddArg(x)
		return true
	}
	// match: (ADDQ x (MOVQconst <t> [c]))
	// cond: is32Bit(c) && !t.IsPtr()
	// result: (ADDQconst [int32(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64MOVQconst {
				continue
			}
			t := v_1.Type
			c := auxIntToInt64(v_1.AuxInt)
			if !(is32Bit(c) && !t.IsPtr()) {
				continue
			}
			v.reset(OpAMD64ADDQconst)
			v.AuxInt = int32ToAuxInt(int32(c))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ADDQ x (MOVLconst [c]))
	// result: (ADDQconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64MOVLconst {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			v.reset(OpAMD64ADDQconst)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ADDQ x (SHLQconst [3] y))
	// result: (LEAQ8 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64SHLQconst || auxIntToInt8(v_1.AuxInt) != 3 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpAMD64LEAQ8)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADDQ x (SHLQconst [2] y))
	// result: (LEAQ4 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64SHLQconst || auxIntToInt8(v_1.AuxInt) != 2 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpAMD64LEAQ4)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADDQ x (ADDQ y y))
	// result: (LEAQ2 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64ADDQ {
				continue
			}
			y := v_1.Args[1]
			if y != v_1.Args[0] {
				continue
			}
			v.reset(OpAMD64LEAQ2)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADDQ x (ADDQ x y))
	// result: (LEAQ2 y x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64ADDQ {
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
				v.reset(OpAMD64LEAQ2)
				v.AddArg2(y, x)
				return true
			}
		}
		break
	}
	// match: (ADDQ (ADDQconst [c] x) y)
	// result: (LEAQ1 [c] x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64ADDQconst {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			y := v_1
			v.reset(OpAMD64LEAQ1)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADDQ x (LEAQ [c] {s} y))
	// cond: x.Op != OpSB && y.Op != OpSB
	// result: (LEAQ1 [c] {s} x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64LEAQ {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			s := auxToSym(v_1.Aux)
			y := v_1.Args[0]
			if !(x.Op != OpSB && y.Op != OpSB) {
				continue
			}
			v.reset(OpAMD64LEAQ1)
			v.AuxInt = int32ToAuxInt(c)
			v.Aux = symToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADDQ x (NEGQ y))
	// result: (SUBQ x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64NEGQ {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpAMD64SUBQ)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADDQ x l:(MOVQload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (ADDQload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != OpAMD64MOVQload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(OpAMD64ADDQload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDQcarry(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDQcarry x (MOVQconst [c]))
	// cond: is32Bit(c)
	// result: (ADDQconstcarry x [int32(c)])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64MOVQconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(is32Bit(c)) {
				continue
			}
			v.reset(OpAMD64ADDQconstcarry)
			v.AuxInt = int32ToAuxInt(int32(c))
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ADDQconst [c] (ADDQ x y))
	// result: (LEAQ1 [c] x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpAMD64LEAQ1)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (ADDQconst [c] (ADDQ x x))
	// result: (LEAQ1 [c] x x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpAMD64LEAQ1)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg2(x, x)
		return true
	}
	// match: (ADDQconst [c] (LEAQ [d] {s} x))
	// cond: is32Bit(int64(c)+int64(d))
	// result: (LEAQ [c+d] {s} x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		s := auxToSym(v_0.Aux)
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(OpAMD64LEAQ)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg(x)
		return true
	}
	// match: (ADDQconst [c] (LEAQ1 [d] {s} x y))
	// cond: is32Bit(int64(c)+int64(d))
	// result: (LEAQ1 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		s := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(OpAMD64LEAQ1)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (ADDQconst [c] (LEAQ2 [d] {s} x y))
	// cond: is32Bit(int64(c)+int64(d))
	// result: (LEAQ2 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64LEAQ2 {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		s := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(OpAMD64LEAQ2)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (ADDQconst [c] (LEAQ4 [d] {s} x y))
	// cond: is32Bit(int64(c)+int64(d))
	// result: (LEAQ4 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64LEAQ4 {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		s := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(OpAMD64LEAQ4)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (ADDQconst [c] (LEAQ8 [d] {s} x y))
	// cond: is32Bit(int64(c)+int64(d))
	// result: (LEAQ8 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64LEAQ8 {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		s := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(OpAMD64LEAQ8)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (ADDQconst [0] x)
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (ADDQconst [c] (MOVQconst [d]))
	// result: (MOVQconst [int64(c)+d])
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(int64(c) + d)
		return true
	}
	// match: (ADDQconst [c] (ADDQconst [d] x))
	// cond: is32Bit(int64(c)+int64(d))
	// result: (ADDQconst [c+d] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(OpAMD64ADDQconst)
		v.AuxInt = int32ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	// match: (ADDQconst [off] x:(SP))
	// result: (LEAQ [off] x)
	for {
		off := auxIntToInt32(v.AuxInt)
		x := v_0
		if x.Op != OpSP {
			break
		}
		v.reset(OpAMD64LEAQ)
		v.AuxInt = int32ToAuxInt(off)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDQconstmodify(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDQconstmodify [valoff1] {sym} (ADDQconst [off2] base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2)
	// result: (ADDQconstmodify [ValAndOff(valoff1).addOffset32(off2)] {sym} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2)) {
			break
		}
		v.reset(OpAMD64ADDQconstmodify)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (ADDQconstmodify [valoff1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)
	// result: (ADDQconstmodify [ValAndOff(valoff1).addOffset32(off2)] {mergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64ADDQconstmodify)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDQload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADDQload [off1] {sym} val (ADDQconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ADDQload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64ADDQload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ADDQload [off1] {sym1} val (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (ADDQload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64ADDQload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ADDQload x [off] {sym} ptr (MOVSDstore [off] {sym} ptr y _))
	// result: (ADDQ x (MOVQf2i y))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		ptr := v_1
		if v_2.Op != OpAMD64MOVSDstore || auxIntToInt32(v_2.AuxInt) != off || auxToSym(v_2.Aux) != sym {
			break
		}
		y := v_2.Args[1]
		if ptr != v_2.Args[0] {
			break
		}
		v.reset(OpAMD64ADDQ)
		v0 := b.NewValue0(v_2.Pos, OpAMD64MOVQf2i, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDQmodify(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDQmodify [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ADDQmodify [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64ADDQmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (ADDQmodify [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (ADDQmodify [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64ADDQmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDSD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDSD x l:(MOVSDload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (ADDSDload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != OpAMD64MOVSDload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(OpAMD64ADDSDload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	// match: (ADDSD (MULSD x y) z)
	// cond: buildcfg.GOAMD64 >= 3 && z.Block.Func.useFMA(v)
	// result: (VFMADD231SD z x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64MULSD {
				continue
			}
			y := v_0.Args[1]
			x := v_0.Args[0]
			z := v_1
			if !(buildcfg.GOAMD64 >= 3 && z.Block.Func.useFMA(v)) {
				continue
			}
			v.reset(OpAMD64VFMADD231SD)
			v.AddArg3(z, x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDSDload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADDSDload [off1] {sym} val (ADDQconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ADDSDload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64ADDSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ADDSDload [off1] {sym1} val (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (ADDSDload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64ADDSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ADDSDload x [off] {sym} ptr (MOVQstore [off] {sym} ptr y _))
	// result: (ADDSD x (MOVQi2f y))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		ptr := v_1
		if v_2.Op != OpAMD64MOVQstore || auxIntToInt32(v_2.AuxInt) != off || auxToSym(v_2.Aux) != sym {
			break
		}
		y := v_2.Args[1]
		if ptr != v_2.Args[0] {
			break
		}
		v.reset(OpAMD64ADDSD)
		v0 := b.NewValue0(v_2.Pos, OpAMD64MOVQi2f, typ.Float64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDSS(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDSS x l:(MOVSSload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (ADDSSload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != OpAMD64MOVSSload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(OpAMD64ADDSSload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	// match: (ADDSS (MULSS x y) z)
	// cond: buildcfg.GOAMD64 >= 3 && z.Block.Func.useFMA(v)
	// result: (VFMADD231SS z x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64MULSS {
				continue
			}
			y := v_0.Args[1]
			x := v_0.Args[0]
			z := v_1
			if !(buildcfg.GOAMD64 >= 3 && z.Block.Func.useFMA(v)) {
				continue
			}
			v.reset(OpAMD64VFMADD231SS)
			v.AddArg3(z, x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDSSload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADDSSload [off1] {sym} val (ADDQconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ADDSSload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64ADDSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ADDSSload [off1] {sym1} val (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (ADDSSload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64ADDSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ADDSSload x [off] {sym} ptr (MOVLstore [off] {sym} ptr y _))
	// result: (ADDSS x (MOVLi2f y))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		ptr := v_1
		if v_2.Op != OpAMD64MOVLstore || auxIntToInt32(v_2.AuxInt) != off || auxToSym(v_2.Aux) != sym {
			break
		}
		y := v_2.Args[1]
		if ptr != v_2.Args[0] {
			break
		}
		v.reset(OpAMD64ADDSS)
		v0 := b.NewValue0(v_2.Pos, OpAMD64MOVLi2f, typ.Float32)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ANDL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ANDL (NOTL (SHLL (MOVLconst [1]) y)) x)
	// result: (BTRL x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64NOTL {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SHLL {
				continue
			}
			y := v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != OpAMD64MOVLconst || auxIntToInt32(v_0_0_0.AuxInt) != 1 {
				continue
			}
			x := v_1
			v.reset(OpAMD64BTRL)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ANDL x (MOVLconst [c]))
	// result: (ANDLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64MOVLconst {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			v.reset(OpAMD64ANDLconst)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg(x)
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
	// match: (ANDL x l:(MOVLload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (ANDLload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != OpAMD64MOVLload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(OpAMD64ANDLload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	// match: (ANDL x (NOTL y))
	// cond: buildcfg.GOAMD64 >= 3
	// result: (ANDNL x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64NOTL {
				continue
			}
			y := v_1.Args[0]
			if !(buildcfg.GOAMD64 >= 3) {
				continue
			}
			v.reset(OpAMD64ANDNL)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ANDL x (NEGL x))
	// cond: buildcfg.GOAMD64 >= 3
	// result: (BLSIL x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64NEGL || x != v_1.Args[0] || !(buildcfg.GOAMD64 >= 3) {
				continue
			}
			v.reset(OpAMD64BLSIL)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ANDL <t> x (ADDLconst [-1] x))
	// cond: buildcfg.GOAMD64 >= 3
	// result: (Select0 <t> (BLSRL x))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64ADDLconst || auxIntToInt32(v_1.AuxInt) != -1 || x != v_1.Args[0] || !(buildcfg.GOAMD64 >= 3) {
				continue
			}
			v.reset(OpSelect0)
			v.Type = t
			v0 := b.NewValue0(v.Pos, OpAMD64BLSRL, types.NewTuple(typ.UInt32, types.TypeFlags))
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64ANDLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ANDLconst [c] (ANDLconst [d] x))
	// result: (ANDLconst [c & d] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = int32ToAuxInt(c & d)
		v.AddArg(x)
		return true
	}
	// match: (ANDLconst [ 0xFF] x)
	// result: (MOVBQZX x)
	for {
		if auxIntToInt32(v.AuxInt) != 0xFF {
			break
		}
		x := v_0
		v.reset(OpAMD64MOVBQZX)
		v.AddArg(x)
		return true
	}
	// match: (ANDLconst [0xFFFF] x)
	// result: (MOVWQZX x)
	for {
		if auxIntToInt32(v.AuxInt) != 0xFFFF {
			break
		}
		x := v_0
		v.reset(OpAMD64MOVWQZX)
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
		v.reset(OpAMD64MOVLconst)
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
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(c & d)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ANDLconstmodify(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ANDLconstmodify [valoff1] {sym} (ADDQconst [off2] base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2)
	// result: (ANDLconstmodify [ValAndOff(valoff1).addOffset32(off2)] {sym} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2)) {
			break
		}
		v.reset(OpAMD64ANDLconstmodify)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (ANDLconstmodify [valoff1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)
	// result: (ANDLconstmodify [ValAndOff(valoff1).addOffset32(off2)] {mergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64ANDLconstmodify)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ANDLload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ANDLload [off1] {sym} val (ADDQconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ANDLload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64ANDLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ANDLload [off1] {sym1} val (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (ANDLload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64ANDLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ANDLload x [off] {sym} ptr (MOVSSstore [off] {sym} ptr y _))
	// result: (ANDL x (MOVLf2i y))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		ptr := v_1
		if v_2.Op != OpAMD64MOVSSstore || auxIntToInt32(v_2.AuxInt) != off || auxToSym(v_2.Aux) != sym {
			break
		}
		y := v_2.Args[1]
		if ptr != v_2.Args[0] {
			break
		}
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v_2.Pos, OpAMD64MOVLf2i, typ.UInt32)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ANDLmodify(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ANDLmodify [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ANDLmodify [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64ANDLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (ANDLmodify [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (ANDLmodify [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64ANDLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ANDNL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ANDNL x (SHLL (MOVLconst [1]) y))
	// result: (BTRL x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64SHLL {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64MOVLconst || auxIntToInt32(v_1_0.AuxInt) != 1 {
			break
		}
		v.reset(OpAMD64BTRL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ANDNQ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ANDNQ x (SHLQ (MOVQconst [1]) y))
	// result: (BTRQ x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64SHLQ {
			break
		}
		y := v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64MOVQconst || auxIntToInt64(v_1_0.AuxInt) != 1 {
			break
		}
		v.reset(OpAMD64BTRQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ANDQ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ANDQ (NOTQ (SHLQ (MOVQconst [1]) y)) x)
	// result: (BTRQ x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64NOTQ {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SHLQ {
				continue
			}
			y := v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != OpAMD64MOVQconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
				continue
			}
			x := v_1
			v.reset(OpAMD64BTRQ)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ANDQ (MOVQconst [c]) x)
	// cond: isUnsignedPowerOfTwo(uint64(^c)) && uint64(^c) >= 1<<31
	// result: (BTRQconst [int8(log64u(uint64(^c)))] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64MOVQconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			x := v_1
			if !(isUnsignedPowerOfTwo(uint64(^c)) && uint64(^c) >= 1<<31) {
				continue
			}
			v.reset(OpAMD64BTRQconst)
			v.AuxInt = int8ToAuxInt(int8(log64u(uint64(^c))))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ANDQ x (MOVQconst [c]))
	// cond: is32Bit(c)
	// result: (ANDQconst [int32(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64MOVQconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(is32Bit(c)) {
				continue
			}
			v.reset(OpAMD64ANDQconst)
			v.AuxInt = int32ToAuxInt(int32(c))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ANDQ x x)
	// result: x
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (ANDQ x l:(MOVQload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (ANDQload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != OpAMD64MOVQload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(OpAMD64ANDQload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	// match: (ANDQ x (NOTQ y))
	// cond: buildcfg.GOAMD64 >= 3
	// result: (ANDNQ x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64NOTQ {
				continue
			}
			y := v_1.Args[0]
			if !(buildcfg.GOAMD64 >= 3) {
				continue
			}
			v.reset(OpAMD64ANDNQ)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ANDQ x (NEGQ x))
	// cond: buildcfg.GOAMD64 >= 3
	// result: (BLSIQ x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64NEGQ || x != v_1.Args[0] || !(buildcfg.GOAMD64 >= 3) {
				continue
			}
			v.reset(OpAMD64BLSIQ)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ANDQ <t> x (ADDQconst [-1] x))
	// cond: buildcfg.GOAMD64 >= 3
	// result: (Select0 <t> (BLSRQ x))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64ADDQconst || auxIntToInt32(v_1.AuxInt) != -1 || x != v_1.Args[0] || !(buildcfg.GOAMD64 >= 3) {
				continue
			}
			v.reset(OpSelect0)
			v.Type = t
			v0 := b.NewValue0(v.Pos, OpAMD64BLSRQ, types.NewTuple(typ.UInt64, types.TypeFlags))
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64ANDQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ANDQconst [c] (ANDQconst [d] x))
	// result: (ANDQconst [c & d] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64ANDQconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64ANDQconst)
		v.AuxInt = int32ToAuxInt(c & d)
		v.AddArg(x)
		return true
	}
	// match: (ANDQconst [ 0xFF] x)
	// result: (MOVBQZX x)
	for {
		if auxIntToInt32(v.AuxInt) != 0xFF {
			break
		}
		x := v_0
		v.reset(OpAMD64MOVBQZX)
		v.AddArg(x)
		return true
	}
	// match: (ANDQconst [0xFFFF] x)
	// result: (MOVWQZX x)
	for {
		if auxIntToInt32(v.AuxInt) != 0xFFFF {
			break
		}
		x := v_0
		v.reset(OpAMD64MOVWQZX)
		v.AddArg(x)
		return true
	}
	// match: (ANDQconst [0] _)
	// result: (MOVQconst [0])
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (ANDQconst [-1] x)
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != -1 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (ANDQconst [c] (MOVQconst [d]))
	// result: (MOVQconst [int64(c)&d])
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(int64(c) & d)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ANDQconstmodify(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ANDQconstmodify [valoff1] {sym} (ADDQconst [off2] base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2)
	// result: (ANDQconstmodify [ValAndOff(valoff1).addOffset32(off2)] {sym} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2)) {
			break
		}
		v.reset(OpAMD64ANDQconstmodify)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (ANDQconstmodify [valoff1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)
	// result: (ANDQconstmodify [ValAndOff(valoff1).addOffset32(off2)] {mergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64ANDQconstmodify)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ANDQload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ANDQload [off1] {sym} val (ADDQconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ANDQload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64ANDQload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ANDQload [off1] {sym1} val (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (ANDQload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64ANDQload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ANDQload x [off] {sym} ptr (MOVSDstore [off] {sym} ptr y _))
	// result: (ANDQ x (MOVQf2i y))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		ptr := v_1
		if v_2.Op != OpAMD64MOVSDstore || auxIntToInt32(v_2.AuxInt) != off || auxToSym(v_2.Aux) != sym {
			break
		}
		y := v_2.Args[1]
		if ptr != v_2.Args[0] {
			break
		}
		v.reset(OpAMD64ANDQ)
		v0 := b.NewValue0(v_2.Pos, OpAMD64MOVQf2i, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ANDQmodify(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ANDQmodify [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ANDQmodify [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64ANDQmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (ANDQmodify [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (ANDQmodify [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64ANDQmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64BSFQ(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (BSFQ (ORQconst <t> [1<<8] (MOVBQZX x)))
	// result: (BSFQ (ORQconst <t> [1<<8] x))
	for {
		if v_0.Op != OpAMD64ORQconst {
			break
		}
		t := v_0.Type
		if auxIntToInt32(v_0.AuxInt) != 1<<8 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64MOVBQZX {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpAMD64BSFQ)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQconst, t)
		v0.AuxInt = int32ToAuxInt(1 << 8)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (BSFQ (ORQconst <t> [1<<16] (MOVWQZX x)))
	// result: (BSFQ (ORQconst <t> [1<<16] x))
	for {
		if v_0.Op != OpAMD64ORQconst {
			break
		}
		t := v_0.Type
		if auxIntToInt32(v_0.AuxInt) != 1<<16 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64MOVWQZX {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpAMD64BSFQ)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQconst, t)
		v0.AuxInt = int32ToAuxInt(1 << 16)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64BSWAPL(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BSWAPL (BSWAPL p))
	// result: p
	for {
		if v_0.Op != OpAMD64BSWAPL {
			break
		}
		p := v_0.Args[0]
		v.copyOf(p)
		return true
	}
	// match: (BSWAPL x:(MOVLload [i] {s} p mem))
	// cond: x.Uses == 1 && buildcfg.GOAMD64 >= 3
	// result: @x.Block (MOVBELload [i] {s} p mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVLload {
			break
		}
		i := auxIntToInt32(x.AuxInt)
		s := auxToSym(x.Aux)
		mem := x.Args[1]
		p := x.Args[0]
		if !(x.Uses == 1 && buildcfg.GOAMD64 >= 3) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, OpAMD64MOVBELload, typ.UInt32)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(i)
		v0.Aux = symToAux(s)
		v0.AddArg2(p, mem)
		return true
	}
	// match: (BSWAPL x:(MOVBELload [i] {s} p mem))
	// cond: x.Uses == 1
	// result: @x.Block (MOVLload [i] {s} p mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVBELload {
			break
		}
		i := auxIntToInt32(x.AuxInt)
		s := auxToSym(x.Aux)
		mem := x.Args[1]
		p := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, OpAMD64MOVLload, typ.UInt32)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(i)
		v0.Aux = symToAux(s)
		v0.AddArg2(p, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64BSWAPQ(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BSWAPQ (BSWAPQ p))
	// result: p
	for {
		if v_0.Op != OpAMD64BSWAPQ {
			break
		}
		p := v_0.Args[0]
		v.copyOf(p)
		return true
	}
	// match: (BSWAPQ x:(MOVQload [i] {s} p mem))
	// cond: x.Uses == 1 && buildcfg.GOAMD64 >= 3
	// result: @x.Block (MOVBEQload [i] {s} p mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVQload {
			break
		}
		i := auxIntToInt32(x.AuxInt)
		s := auxToSym(x.Aux)
		mem := x.Args[1]
		p := x.Args[0]
		if !(x.Uses == 1 && buildcfg.GOAMD64 >= 3) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, OpAMD64MOVBEQload, typ.UInt64)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(i)
		v0.Aux = symToAux(s)
		v0.AddArg2(p, mem)
		return true
	}
	// match: (BSWAPQ x:(MOVBEQload [i] {s} p mem))
	// cond: x.Uses == 1
	// result: @x.Block (MOVQload [i] {s} p mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVBEQload {
			break
		}
		i := auxIntToInt32(x.AuxInt)
		s := auxToSym(x.Aux)
		mem := x.Args[1]
		p := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, OpAMD64MOVQload, typ.UInt64)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(i)
		v0.Aux = symToAux(s)
		v0.AddArg2(p, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64BTCQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (BTCQconst [c] (MOVQconst [d]))
	// result: (MOVQconst [d^(1<<uint32(c))])
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(d ^ (1 << uint32(c)))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64BTLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (BTLconst [c] (SHRQconst [d] x))
	// cond: (c+d)<64
	// result: (BTQconst [c+d] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64SHRQconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		if !((c + d) < 64) {
			break
		}
		v.reset(OpAMD64BTQconst)
		v.AuxInt = int8ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	// match: (BTLconst [c] (ADDQ x x))
	// cond: c>1
	// result: (BTLconst [c-1] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] || !(c > 1) {
			break
		}
		v.reset(OpAMD64BTLconst)
		v.AuxInt = int8ToAuxInt(c - 1)
		v.AddArg(x)
		return true
	}
	// match: (BTLconst [c] (SHLQconst [d] x))
	// cond: c>d
	// result: (BTLconst [c-d] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64SHLQconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c > d) {
			break
		}
		v.reset(OpAMD64BTLconst)
		v.AuxInt = int8ToAuxInt(c - d)
		v.AddArg(x)
		return true
	}
	// match: (BTLconst [0] s:(SHRQ x y))
	// result: (BTQ y x)
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		s := v_0
		if s.Op != OpAMD64SHRQ {
			break
		}
		y := s.Args[1]
		x := s.Args[0]
		v.reset(OpAMD64BTQ)
		v.AddArg2(y, x)
		return true
	}
	// match: (BTLconst [c] (SHRLconst [d] x))
	// cond: (c+d)<32
	// result: (BTLconst [c+d] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64SHRLconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		if !((c + d) < 32) {
			break
		}
		v.reset(OpAMD64BTLconst)
		v.AuxInt = int8ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	// match: (BTLconst [c] (ADDL x x))
	// cond: c>1
	// result: (BTLconst [c-1] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64ADDL {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] || !(c > 1) {
			break
		}
		v.reset(OpAMD64BTLconst)
		v.AuxInt = int8ToAuxInt(c - 1)
		v.AddArg(x)
		return true
	}
	// match: (BTLconst [c] (SHLLconst [d] x))
	// cond: c>d
	// result: (BTLconst [c-d] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64SHLLconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c > d) {
			break
		}
		v.reset(OpAMD64BTLconst)
		v.AuxInt = int8ToAuxInt(c - d)
		v.AddArg(x)
		return true
	}
	// match: (BTLconst [0] s:(SHRL x y))
	// result: (BTL y x)
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		s := v_0
		if s.Op != OpAMD64SHRL {
			break
		}
		y := s.Args[1]
		x := s.Args[0]
		v.reset(OpAMD64BTL)
		v.AddArg2(y, x)
		return true
	}
	// match: (BTLconst [0] s:(SHRXL x y))
	// result: (BTL y x)
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		s := v_0
		if s.Op != OpAMD64SHRXL {
			break
		}
		y := s.Args[1]
		x := s.Args[0]
		v.reset(OpAMD64BTL)
		v.AddArg2(y, x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64BTQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (BTQconst [c] (SHRQconst [d] x))
	// cond: (c+d)<64
	// result: (BTQconst [c+d] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64SHRQconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		if !((c + d) < 64) {
			break
		}
		v.reset(OpAMD64BTQconst)
		v.AuxInt = int8ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	// match: (BTQconst [c] (ADDQ x x))
	// cond: c>1
	// result: (BTQconst [c-1] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] || !(c > 1) {
			break
		}
		v.reset(OpAMD64BTQconst)
		v.AuxInt = int8ToAuxInt(c - 1)
		v.AddArg(x)
		return true
	}
	// match: (BTQconst [c] (SHLQconst [d] x))
	// cond: c>d
	// result: (BTQconst [c-d] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64SHLQconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c > d) {
			break
		}
		v.reset(OpAMD64BTQconst)
		v.AuxInt = int8ToAuxInt(c - d)
		v.AddArg(x)
		return true
	}
	// match: (BTQconst [0] s:(SHRQ x y))
	// result: (BTQ y x)
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		s := v_0
		if s.Op != OpAMD64SHRQ {
			break
		}
		y := s.Args[1]
		x := s.Args[0]
		v.reset(OpAMD64BTQ)
		v.AddArg2(y, x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64BTRQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (BTRQconst [c] (BTSQconst [c] x))
	// result: (BTRQconst [c] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64BTSQconst || auxIntToInt8(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64BTRQconst)
		v.AuxInt = int8ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (BTRQconst [c] (BTCQconst [c] x))
	// result: (BTRQconst [c] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64BTCQconst || auxIntToInt8(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64BTRQconst)
		v.AuxInt = int8ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (BTRQconst [c] (MOVQconst [d]))
	// result: (MOVQconst [d&^(1<<uint32(c))])
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(d &^ (1 << uint32(c)))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64BTSQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (BTSQconst [c] (BTRQconst [c] x))
	// result: (BTSQconst [c] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64BTRQconst || auxIntToInt8(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64BTSQconst)
		v.AuxInt = int8ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (BTSQconst [c] (BTCQconst [c] x))
	// result: (BTSQconst [c] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64BTCQconst || auxIntToInt8(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64BTSQconst)
		v.AuxInt = int8ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (BTSQconst [c] (MOVQconst [d]))
	// result: (MOVQconst [d|(1<<uint32(c))])
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(d | (1 << uint32(c)))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVLCC(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVLCC x y (InvertFlags cond))
	// result: (CMOVLLS x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVLLS)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVLCC _ x (FlagEQ))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLCC _ x (FlagGT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLCC y _ (FlagGT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLCC y _ (FlagLT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLCC _ x (FlagLT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVLCS(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVLCS x y (InvertFlags cond))
	// result: (CMOVLHI x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVLHI)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVLCS y _ (FlagEQ))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLCS y _ (FlagGT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLCS _ x (FlagGT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLCS _ x (FlagLT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLCS y _ (FlagLT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVLEQ(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMOVLEQ x y (InvertFlags cond))
	// result: (CMOVLEQ x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVLEQ)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVLEQ _ x (FlagEQ))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLEQ y _ (FlagGT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLEQ y _ (FlagGT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLEQ y _ (FlagLT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLEQ y _ (FlagLT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLEQ x y (TESTQ s:(Select0 blsr:(BLSRQ _)) s))
	// result: (CMOVLEQ x y (Select1 <types.TypeFlags> blsr))
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64TESTQ {
			break
		}
		_ = v_2.Args[1]
		v_2_0 := v_2.Args[0]
		v_2_1 := v_2.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_2_0, v_2_1 = _i0+1, v_2_1, v_2_0 {
			s := v_2_0
			if s.Op != OpSelect0 {
				continue
			}
			blsr := s.Args[0]
			if blsr.Op != OpAMD64BLSRQ || s != v_2_1 {
				continue
			}
			v.reset(OpAMD64CMOVLEQ)
			v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(blsr)
			v.AddArg3(x, y, v0)
			return true
		}
		break
	}
	// match: (CMOVLEQ x y (TESTL s:(Select0 blsr:(BLSRL _)) s))
	// result: (CMOVLEQ x y (Select1 <types.TypeFlags> blsr))
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64TESTL {
			break
		}
		_ = v_2.Args[1]
		v_2_0 := v_2.Args[0]
		v_2_1 := v_2.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_2_0, v_2_1 = _i0+1, v_2_1, v_2_0 {
			s := v_2_0
			if s.Op != OpSelect0 {
				continue
			}
			blsr := s.Args[0]
			if blsr.Op != OpAMD64BLSRL || s != v_2_1 {
				continue
			}
			v.reset(OpAMD64CMOVLEQ)
			v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(blsr)
			v.AddArg3(x, y, v0)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVLGE(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMOVLGE x y (InvertFlags cond))
	// result: (CMOVLLE x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVLLE)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVLGE _ x (FlagEQ))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLGE _ x (FlagGT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLGE _ x (FlagGT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLGE y _ (FlagLT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLGE y _ (FlagLT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLGE x y c:(CMPQconst [128] z))
	// cond: c.Uses == 1
	// result: (CMOVLGT x y (CMPQconst [127] z))
	for {
		x := v_0
		y := v_1
		c := v_2
		if c.Op != OpAMD64CMPQconst || auxIntToInt32(c.AuxInt) != 128 {
			break
		}
		z := c.Args[0]
		if !(c.Uses == 1) {
			break
		}
		v.reset(OpAMD64CMOVLGT)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(127)
		v0.AddArg(z)
		v.AddArg3(x, y, v0)
		return true
	}
	// match: (CMOVLGE x y c:(CMPLconst [128] z))
	// cond: c.Uses == 1
	// result: (CMOVLGT x y (CMPLconst [127] z))
	for {
		x := v_0
		y := v_1
		c := v_2
		if c.Op != OpAMD64CMPLconst || auxIntToInt32(c.AuxInt) != 128 {
			break
		}
		z := c.Args[0]
		if !(c.Uses == 1) {
			break
		}
		v.reset(OpAMD64CMOVLGT)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(127)
		v0.AddArg(z)
		v.AddArg3(x, y, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVLGT(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVLGT x y (InvertFlags cond))
	// result: (CMOVLLT x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVLLT)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVLGT y _ (FlagEQ))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLGT _ x (FlagGT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLGT _ x (FlagGT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLGT y _ (FlagLT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLGT y _ (FlagLT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVLHI(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVLHI x y (InvertFlags cond))
	// result: (CMOVLCS x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVLCS)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVLHI y _ (FlagEQ))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLHI _ x (FlagGT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLHI y _ (FlagGT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLHI y _ (FlagLT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLHI _ x (FlagLT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVLLE(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVLLE x y (InvertFlags cond))
	// result: (CMOVLGE x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVLGE)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVLLE _ x (FlagEQ))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLLE y _ (FlagGT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLLE y _ (FlagGT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLLE _ x (FlagLT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLLE _ x (FlagLT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVLLS(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVLLS x y (InvertFlags cond))
	// result: (CMOVLCC x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVLCC)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVLLS _ x (FlagEQ))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLLS y _ (FlagGT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLLS _ x (FlagGT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLLS _ x (FlagLT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLLS y _ (FlagLT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVLLT(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMOVLLT x y (InvertFlags cond))
	// result: (CMOVLGT x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVLGT)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVLLT y _ (FlagEQ))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLLT y _ (FlagGT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLLT y _ (FlagGT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLLT _ x (FlagLT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLLT _ x (FlagLT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLLT x y c:(CMPQconst [128] z))
	// cond: c.Uses == 1
	// result: (CMOVLLE x y (CMPQconst [127] z))
	for {
		x := v_0
		y := v_1
		c := v_2
		if c.Op != OpAMD64CMPQconst || auxIntToInt32(c.AuxInt) != 128 {
			break
		}
		z := c.Args[0]
		if !(c.Uses == 1) {
			break
		}
		v.reset(OpAMD64CMOVLLE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(127)
		v0.AddArg(z)
		v.AddArg3(x, y, v0)
		return true
	}
	// match: (CMOVLLT x y c:(CMPLconst [128] z))
	// cond: c.Uses == 1
	// result: (CMOVLLE x y (CMPLconst [127] z))
	for {
		x := v_0
		y := v_1
		c := v_2
		if c.Op != OpAMD64CMPLconst || auxIntToInt32(c.AuxInt) != 128 {
			break
		}
		z := c.Args[0]
		if !(c.Uses == 1) {
			break
		}
		v.reset(OpAMD64CMOVLLE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(127)
		v0.AddArg(z)
		v.AddArg3(x, y, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVLNE(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMOVLNE x y (InvertFlags cond))
	// result: (CMOVLNE x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVLNE)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVLNE y _ (FlagEQ))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVLNE _ x (FlagGT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLNE _ x (FlagGT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLNE _ x (FlagLT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLNE _ x (FlagLT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVLNE x y (TESTQ s:(Select0 blsr:(BLSRQ _)) s))
	// result: (CMOVLNE x y (Select1 <types.TypeFlags> blsr))
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64TESTQ {
			break
		}
		_ = v_2.Args[1]
		v_2_0 := v_2.Args[0]
		v_2_1 := v_2.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_2_0, v_2_1 = _i0+1, v_2_1, v_2_0 {
			s := v_2_0
			if s.Op != OpSelect0 {
				continue
			}
			blsr := s.Args[0]
			if blsr.Op != OpAMD64BLSRQ || s != v_2_1 {
				continue
			}
			v.reset(OpAMD64CMOVLNE)
			v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(blsr)
			v.AddArg3(x, y, v0)
			return true
		}
		break
	}
	// match: (CMOVLNE x y (TESTL s:(Select0 blsr:(BLSRL _)) s))
	// result: (CMOVLNE x y (Select1 <types.TypeFlags> blsr))
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64TESTL {
			break
		}
		_ = v_2.Args[1]
		v_2_0 := v_2.Args[0]
		v_2_1 := v_2.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_2_0, v_2_1 = _i0+1, v_2_1, v_2_0 {
			s := v_2_0
			if s.Op != OpSelect0 {
				continue
			}
			blsr := s.Args[0]
			if blsr.Op != OpAMD64BLSRL || s != v_2_1 {
				continue
			}
			v.reset(OpAMD64CMOVLNE)
			v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(blsr)
			v.AddArg3(x, y, v0)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVQCC(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVQCC x y (InvertFlags cond))
	// result: (CMOVQLS x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVQLS)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVQCC _ x (FlagEQ))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQCC _ x (FlagGT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQCC y _ (FlagGT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQCC y _ (FlagLT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQCC _ x (FlagLT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVQCS(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVQCS x y (InvertFlags cond))
	// result: (CMOVQHI x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVQHI)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVQCS y _ (FlagEQ))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQCS y _ (FlagGT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQCS _ x (FlagGT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQCS _ x (FlagLT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQCS y _ (FlagLT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVQEQ(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMOVQEQ x y (InvertFlags cond))
	// result: (CMOVQEQ x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVQEQ)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVQEQ _ x (FlagEQ))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQEQ y _ (FlagGT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQEQ y _ (FlagGT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQEQ y _ (FlagLT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQEQ y _ (FlagLT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQEQ x _ (Select1 (BSFQ (ORQconst [c] _))))
	// cond: c != 0
	// result: x
	for {
		x := v_0
		if v_2.Op != OpSelect1 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpAMD64BSFQ {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != OpAMD64ORQconst {
			break
		}
		c := auxIntToInt32(v_2_0_0.AuxInt)
		if !(c != 0) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQEQ x _ (Select1 (BSRQ (ORQconst [c] _))))
	// cond: c != 0
	// result: x
	for {
		x := v_0
		if v_2.Op != OpSelect1 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpAMD64BSRQ {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != OpAMD64ORQconst {
			break
		}
		c := auxIntToInt32(v_2_0_0.AuxInt)
		if !(c != 0) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQEQ x y (TESTQ s:(Select0 blsr:(BLSRQ _)) s))
	// result: (CMOVQEQ x y (Select1 <types.TypeFlags> blsr))
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64TESTQ {
			break
		}
		_ = v_2.Args[1]
		v_2_0 := v_2.Args[0]
		v_2_1 := v_2.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_2_0, v_2_1 = _i0+1, v_2_1, v_2_0 {
			s := v_2_0
			if s.Op != OpSelect0 {
				continue
			}
			blsr := s.Args[0]
			if blsr.Op != OpAMD64BLSRQ || s != v_2_1 {
				continue
			}
			v.reset(OpAMD64CMOVQEQ)
			v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(blsr)
			v.AddArg3(x, y, v0)
			return true
		}
		break
	}
	// match: (CMOVQEQ x y (TESTL s:(Select0 blsr:(BLSRL _)) s))
	// result: (CMOVQEQ x y (Select1 <types.TypeFlags> blsr))
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64TESTL {
			break
		}
		_ = v_2.Args[1]
		v_2_0 := v_2.Args[0]
		v_2_1 := v_2.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_2_0, v_2_1 = _i0+1, v_2_1, v_2_0 {
			s := v_2_0
			if s.Op != OpSelect0 {
				continue
			}
			blsr := s.Args[0]
			if blsr.Op != OpAMD64BLSRL || s != v_2_1 {
				continue
			}
			v.reset(OpAMD64CMOVQEQ)
			v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(blsr)
			v.AddArg3(x, y, v0)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVQGE(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMOVQGE x y (InvertFlags cond))
	// result: (CMOVQLE x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVQLE)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVQGE _ x (FlagEQ))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQGE _ x (FlagGT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQGE _ x (FlagGT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQGE y _ (FlagLT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQGE y _ (FlagLT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQGE x y c:(CMPQconst [128] z))
	// cond: c.Uses == 1
	// result: (CMOVQGT x y (CMPQconst [127] z))
	for {
		x := v_0
		y := v_1
		c := v_2
		if c.Op != OpAMD64CMPQconst || auxIntToInt32(c.AuxInt) != 128 {
			break
		}
		z := c.Args[0]
		if !(c.Uses == 1) {
			break
		}
		v.reset(OpAMD64CMOVQGT)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(127)
		v0.AddArg(z)
		v.AddArg3(x, y, v0)
		return true
	}
	// match: (CMOVQGE x y c:(CMPLconst [128] z))
	// cond: c.Uses == 1
	// result: (CMOVQGT x y (CMPLconst [127] z))
	for {
		x := v_0
		y := v_1
		c := v_2
		if c.Op != OpAMD64CMPLconst || auxIntToInt32(c.AuxInt) != 128 {
			break
		}
		z := c.Args[0]
		if !(c.Uses == 1) {
			break
		}
		v.reset(OpAMD64CMOVQGT)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(127)
		v0.AddArg(z)
		v.AddArg3(x, y, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVQGT(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVQGT x y (InvertFlags cond))
	// result: (CMOVQLT x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVQLT)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVQGT y _ (FlagEQ))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQGT _ x (FlagGT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQGT _ x (FlagGT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQGT y _ (FlagLT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQGT y _ (FlagLT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVQHI(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVQHI x y (InvertFlags cond))
	// result: (CMOVQCS x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVQCS)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVQHI y _ (FlagEQ))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQHI _ x (FlagGT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQHI y _ (FlagGT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQHI y _ (FlagLT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQHI _ x (FlagLT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVQLE(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVQLE x y (InvertFlags cond))
	// result: (CMOVQGE x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVQGE)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVQLE _ x (FlagEQ))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQLE y _ (FlagGT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQLE y _ (FlagGT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQLE _ x (FlagLT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQLE _ x (FlagLT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVQLS(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVQLS x y (InvertFlags cond))
	// result: (CMOVQCC x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVQCC)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVQLS _ x (FlagEQ))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQLS y _ (FlagGT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQLS _ x (FlagGT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQLS _ x (FlagLT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQLS y _ (FlagLT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVQLT(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMOVQLT x y (InvertFlags cond))
	// result: (CMOVQGT x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVQGT)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVQLT y _ (FlagEQ))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQLT y _ (FlagGT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQLT y _ (FlagGT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQLT _ x (FlagLT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQLT _ x (FlagLT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQLT x y c:(CMPQconst [128] z))
	// cond: c.Uses == 1
	// result: (CMOVQLE x y (CMPQconst [127] z))
	for {
		x := v_0
		y := v_1
		c := v_2
		if c.Op != OpAMD64CMPQconst || auxIntToInt32(c.AuxInt) != 128 {
			break
		}
		z := c.Args[0]
		if !(c.Uses == 1) {
			break
		}
		v.reset(OpAMD64CMOVQLE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(127)
		v0.AddArg(z)
		v.AddArg3(x, y, v0)
		return true
	}
	// match: (CMOVQLT x y c:(CMPLconst [128] z))
	// cond: c.Uses == 1
	// result: (CMOVQLE x y (CMPLconst [127] z))
	for {
		x := v_0
		y := v_1
		c := v_2
		if c.Op != OpAMD64CMPLconst || auxIntToInt32(c.AuxInt) != 128 {
			break
		}
		z := c.Args[0]
		if !(c.Uses == 1) {
			break
		}
		v.reset(OpAMD64CMOVQLE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(127)
		v0.AddArg(z)
		v.AddArg3(x, y, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVQNE(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMOVQNE x y (InvertFlags cond))
	// result: (CMOVQNE x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVQNE)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVQNE y _ (FlagEQ))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVQNE _ x (FlagGT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQNE _ x (FlagGT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQNE _ x (FlagLT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQNE _ x (FlagLT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVQNE x y (TESTQ s:(Select0 blsr:(BLSRQ _)) s))
	// result: (CMOVQNE x y (Select1 <types.TypeFlags> blsr))
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64TESTQ {
			break
		}
		_ = v_2.Args[1]
		v_2_0 := v_2.Args[0]
		v_2_1 := v_2.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_2_0, v_2_1 = _i0+1, v_2_1, v_2_0 {
			s := v_2_0
			if s.Op != OpSelect0 {
				continue
			}
			blsr := s.Args[0]
			if blsr.Op != OpAMD64BLSRQ || s != v_2_1 {
				continue
			}
			v.reset(OpAMD64CMOVQNE)
			v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(blsr)
			v.AddArg3(x, y, v0)
			return true
		}
		break
	}
	// match: (CMOVQNE x y (TESTL s:(Select0 blsr:(BLSRL _)) s))
	// result: (CMOVQNE x y (Select1 <types.TypeFlags> blsr))
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64TESTL {
			break
		}
		_ = v_2.Args[1]
		v_2_0 := v_2.Args[0]
		v_2_1 := v_2.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_2_0, v_2_1 = _i0+1, v_2_1, v_2_0 {
			s := v_2_0
			if s.Op != OpSelect0 {
				continue
			}
			blsr := s.Args[0]
			if blsr.Op != OpAMD64BLSRL || s != v_2_1 {
				continue
			}
			v.reset(OpAMD64CMOVQNE)
			v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(blsr)
			v.AddArg3(x, y, v0)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVWCC(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVWCC x y (InvertFlags cond))
	// result: (CMOVWLS x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVWLS)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVWCC _ x (FlagEQ))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWCC _ x (FlagGT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWCC y _ (FlagGT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWCC y _ (FlagLT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWCC _ x (FlagLT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVWCS(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVWCS x y (InvertFlags cond))
	// result: (CMOVWHI x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVWHI)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVWCS y _ (FlagEQ))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWCS y _ (FlagGT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWCS _ x (FlagGT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWCS _ x (FlagLT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWCS y _ (FlagLT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVWEQ(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVWEQ x y (InvertFlags cond))
	// result: (CMOVWEQ x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVWEQ)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVWEQ _ x (FlagEQ))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWEQ y _ (FlagGT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWEQ y _ (FlagGT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWEQ y _ (FlagLT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWEQ y _ (FlagLT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVWGE(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVWGE x y (InvertFlags cond))
	// result: (CMOVWLE x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVWLE)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVWGE _ x (FlagEQ))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWGE _ x (FlagGT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWGE _ x (FlagGT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWGE y _ (FlagLT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWGE y _ (FlagLT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVWGT(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVWGT x y (InvertFlags cond))
	// result: (CMOVWLT x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVWLT)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVWGT y _ (FlagEQ))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWGT _ x (FlagGT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWGT _ x (FlagGT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWGT y _ (FlagLT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWGT y _ (FlagLT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVWHI(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVWHI x y (InvertFlags cond))
	// result: (CMOVWCS x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVWCS)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVWHI y _ (FlagEQ))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWHI _ x (FlagGT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWHI y _ (FlagGT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWHI y _ (FlagLT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWHI _ x (FlagLT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVWLE(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVWLE x y (InvertFlags cond))
	// result: (CMOVWGE x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVWGE)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVWLE _ x (FlagEQ))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWLE y _ (FlagGT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWLE y _ (FlagGT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWLE _ x (FlagLT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWLE _ x (FlagLT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVWLS(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVWLS x y (InvertFlags cond))
	// result: (CMOVWCC x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVWCC)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVWLS _ x (FlagEQ))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWLS y _ (FlagGT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWLS _ x (FlagGT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWLS _ x (FlagLT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWLS y _ (FlagLT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVWLT(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVWLT x y (InvertFlags cond))
	// result: (CMOVWGT x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVWGT)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVWLT y _ (FlagEQ))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWLT y _ (FlagGT_UGT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWLT y _ (FlagGT_ULT))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWLT _ x (FlagLT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWLT _ x (FlagLT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVWNE(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVWNE x y (InvertFlags cond))
	// result: (CMOVWNE x y cond)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64InvertFlags {
			break
		}
		cond := v_2.Args[0]
		v.reset(OpAMD64CMOVWNE)
		v.AddArg3(x, y, cond)
		return true
	}
	// match: (CMOVWNE y _ (FlagEQ))
	// result: y
	for {
		y := v_0
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CMOVWNE _ x (FlagGT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWNE _ x (FlagGT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWNE _ x (FlagLT_ULT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CMOVWNE _ x (FlagLT_UGT))
	// result: x
	for {
		x := v_1
		if v_2.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPB x (MOVLconst [c]))
	// result: (CMPBconst x [int8(c)])
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64CMPBconst)
		v.AuxInt = int8ToAuxInt(int8(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPB (MOVLconst [c]) x)
	// result: (InvertFlags (CMPBconst x [int8(c)]))
	for {
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_1
		v.reset(OpAMD64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v0.AuxInt = int8ToAuxInt(int8(c))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPB x y)
	// cond: canonLessThan(x,y)
	// result: (InvertFlags (CMPB y x))
	for {
		x := v_0
		y := v_1
		if !(canonLessThan(x, y)) {
			break
		}
		v.reset(OpAMD64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPB l:(MOVBload {sym} [off] ptr mem) x)
	// cond: canMergeLoad(v, l) && clobber(l)
	// result: (CMPBload {sym} [off] ptr x mem)
	for {
		l := v_0
		if l.Op != OpAMD64MOVBload {
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
		v.reset(OpAMD64CMPBload)
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
		if l.Op != OpAMD64MOVBload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(canMergeLoad(v, l) && clobber(l)) {
			break
		}
		v.reset(OpAMD64InvertFlags)
		v0 := b.NewValue0(l.Pos, OpAMD64CMPBload, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, x, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPBconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)==y
	// result: (FlagEQ)
	for {
		y := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int8(x) == y) {
			break
		}
		v.reset(OpAMD64FlagEQ)
		return true
	}
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)<y && uint8(x)<uint8(y)
	// result: (FlagLT_ULT)
	for {
		y := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int8(x) < y && uint8(x) < uint8(y)) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)<y && uint8(x)>uint8(y)
	// result: (FlagLT_UGT)
	for {
		y := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int8(x) < y && uint8(x) > uint8(y)) {
			break
		}
		v.reset(OpAMD64FlagLT_UGT)
		return true
	}
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)>y && uint8(x)<uint8(y)
	// result: (FlagGT_ULT)
	for {
		y := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int8(x) > y && uint8(x) < uint8(y)) {
			break
		}
		v.reset(OpAMD64FlagGT_ULT)
		return true
	}
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)>y && uint8(x)>uint8(y)
	// result: (FlagGT_UGT)
	for {
		y := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int8(x) > y && uint8(x) > uint8(y)) {
			break
		}
		v.reset(OpAMD64FlagGT_UGT)
		return true
	}
	// match: (CMPBconst (ANDLconst _ [m]) [n])
	// cond: 0 <= int8(m) && int8(m) < n
	// result: (FlagLT_ULT)
	for {
		n := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		m := auxIntToInt32(v_0.AuxInt)
		if !(0 <= int8(m) && int8(m) < n) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPBconst a:(ANDL x y) [0])
	// cond: a.Uses == 1
	// result: (TESTB x y)
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		a := v_0
		if a.Op != OpAMD64ANDL {
			break
		}
		y := a.Args[1]
		x := a.Args[0]
		if !(a.Uses == 1) {
			break
		}
		v.reset(OpAMD64TESTB)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMPBconst a:(ANDLconst [c] x) [0])
	// cond: a.Uses == 1
	// result: (TESTBconst [int8(c)] x)
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		a := v_0
		if a.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(a.AuxInt)
		x := a.Args[0]
		if !(a.Uses == 1) {
			break
		}
		v.reset(OpAMD64TESTBconst)
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
		v.reset(OpAMD64TESTB)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPBconst l:(MOVBload {sym} [off] ptr mem) [c])
	// cond: l.Uses == 1 && clobber(l)
	// result: @l.Block (CMPBconstload {sym} [makeValAndOff(int32(c),off)] ptr mem)
	for {
		c := auxIntToInt8(v.AuxInt)
		l := v_0
		if l.Op != OpAMD64MOVBload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(l.Uses == 1 && clobber(l)) {
			break
		}
		b = l.Block
		v0 := b.NewValue0(l.Pos, OpAMD64CMPBconstload, types.TypeFlags)
		v.copyOf(v0)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(c), off))
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPBconstload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMPBconstload [valoff1] {sym} (ADDQconst [off2] base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2)
	// result: (CMPBconstload [ValAndOff(valoff1).addOffset32(off2)] {sym} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2)) {
			break
		}
		v.reset(OpAMD64CMPBconstload)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (CMPBconstload [valoff1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)
	// result: (CMPBconstload [ValAndOff(valoff1).addOffset32(off2)] {mergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64CMPBconstload)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPBload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMPBload [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (CMPBload [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64CMPBload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (CMPBload [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (CMPBload [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64CMPBload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (CMPBload {sym} [off] ptr (MOVLconst [c]) mem)
	// result: (CMPBconstload {sym} [makeValAndOff(int32(int8(c)),off)] ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64CMPBconstload)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(int8(c)), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPL x (MOVLconst [c]))
	// result: (CMPLconst x [c])
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64CMPLconst)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (CMPL (MOVLconst [c]) x)
	// result: (InvertFlags (CMPLconst x [c]))
	for {
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_1
		v.reset(OpAMD64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPL x y)
	// cond: canonLessThan(x,y)
	// result: (InvertFlags (CMPL y x))
	for {
		x := v_0
		y := v_1
		if !(canonLessThan(x, y)) {
			break
		}
		v.reset(OpAMD64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPL l:(MOVLload {sym} [off] ptr mem) x)
	// cond: canMergeLoad(v, l) && clobber(l)
	// result: (CMPLload {sym} [off] ptr x mem)
	for {
		l := v_0
		if l.Op != OpAMD64MOVLload {
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
		v.reset(OpAMD64CMPLload)
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
		if l.Op != OpAMD64MOVLload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(canMergeLoad(v, l) && clobber(l)) {
			break
		}
		v.reset(OpAMD64InvertFlags)
		v0 := b.NewValue0(l.Pos, OpAMD64CMPLload, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, x, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPLconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: x==y
	// result: (FlagEQ)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(x == y) {
			break
		}
		v.reset(OpAMD64FlagEQ)
		return true
	}
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: x<y && uint32(x)<uint32(y)
	// result: (FlagLT_ULT)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(x < y && uint32(x) < uint32(y)) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: x<y && uint32(x)>uint32(y)
	// result: (FlagLT_UGT)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(x < y && uint32(x) > uint32(y)) {
			break
		}
		v.reset(OpAMD64FlagLT_UGT)
		return true
	}
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: x>y && uint32(x)<uint32(y)
	// result: (FlagGT_ULT)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(x > y && uint32(x) < uint32(y)) {
			break
		}
		v.reset(OpAMD64FlagGT_ULT)
		return true
	}
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: x>y && uint32(x)>uint32(y)
	// result: (FlagGT_UGT)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(x > y && uint32(x) > uint32(y)) {
			break
		}
		v.reset(OpAMD64FlagGT_UGT)
		return true
	}
	// match: (CMPLconst (SHRLconst _ [c]) [n])
	// cond: 0 <= n && 0 < c && c <= 32 && (1<<uint64(32-c)) <= uint64(n)
	// result: (FlagLT_ULT)
	for {
		n := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64SHRLconst {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if !(0 <= n && 0 < c && c <= 32 && (1<<uint64(32-c)) <= uint64(n)) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPLconst (ANDLconst _ [m]) [n])
	// cond: 0 <= m && m < n
	// result: (FlagLT_ULT)
	for {
		n := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		m := auxIntToInt32(v_0.AuxInt)
		if !(0 <= m && m < n) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPLconst a:(ANDL x y) [0])
	// cond: a.Uses == 1
	// result: (TESTL x y)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		a := v_0
		if a.Op != OpAMD64ANDL {
			break
		}
		y := a.Args[1]
		x := a.Args[0]
		if !(a.Uses == 1) {
			break
		}
		v.reset(OpAMD64TESTL)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMPLconst a:(ANDLconst [c] x) [0])
	// cond: a.Uses == 1
	// result: (TESTLconst [c] x)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		a := v_0
		if a.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(a.AuxInt)
		x := a.Args[0]
		if !(a.Uses == 1) {
			break
		}
		v.reset(OpAMD64TESTLconst)
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
		v.reset(OpAMD64TESTL)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPLconst l:(MOVLload {sym} [off] ptr mem) [c])
	// cond: l.Uses == 1 && clobber(l)
	// result: @l.Block (CMPLconstload {sym} [makeValAndOff(c,off)] ptr mem)
	for {
		c := auxIntToInt32(v.AuxInt)
		l := v_0
		if l.Op != OpAMD64MOVLload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(l.Uses == 1 && clobber(l)) {
			break
		}
		b = l.Block
		v0 := b.NewValue0(l.Pos, OpAMD64CMPLconstload, types.TypeFlags)
		v.copyOf(v0)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(c, off))
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPLconstload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMPLconstload [valoff1] {sym} (ADDQconst [off2] base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2)
	// result: (CMPLconstload [ValAndOff(valoff1).addOffset32(off2)] {sym} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2)) {
			break
		}
		v.reset(OpAMD64CMPLconstload)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (CMPLconstload [valoff1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)
	// result: (CMPLconstload [ValAndOff(valoff1).addOffset32(off2)] {mergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64CMPLconstload)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPLload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMPLload [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (CMPLload [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64CMPLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (CMPLload [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (CMPLload [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64CMPLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (CMPLload {sym} [off] ptr (MOVLconst [c]) mem)
	// result: (CMPLconstload {sym} [makeValAndOff(c,off)] ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64CMPLconstload)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(c, off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPQ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPQ x (MOVQconst [c]))
	// cond: is32Bit(c)
	// result: (CMPQconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64CMPQconst)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPQ (MOVQconst [c]) x)
	// cond: is32Bit(c)
	// result: (InvertFlags (CMPQconst x [int32(c)]))
	for {
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPQ x y)
	// cond: canonLessThan(x,y)
	// result: (InvertFlags (CMPQ y x))
	for {
		x := v_0
		y := v_1
		if !(canonLessThan(x, y)) {
			break
		}
		v.reset(OpAMD64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPQ (MOVQconst [x]) (MOVQconst [y]))
	// cond: x==y
	// result: (FlagEQ)
	for {
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		y := auxIntToInt64(v_1.AuxInt)
		if !(x == y) {
			break
		}
		v.reset(OpAMD64FlagEQ)
		return true
	}
	// match: (CMPQ (MOVQconst [x]) (MOVQconst [y]))
	// cond: x<y && uint64(x)<uint64(y)
	// result: (FlagLT_ULT)
	for {
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		y := auxIntToInt64(v_1.AuxInt)
		if !(x < y && uint64(x) < uint64(y)) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPQ (MOVQconst [x]) (MOVQconst [y]))
	// cond: x<y && uint64(x)>uint64(y)
	// result: (FlagLT_UGT)
	for {
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		y := auxIntToInt64(v_1.AuxInt)
		if !(x < y && uint64(x) > uint64(y)) {
			break
		}
		v.reset(OpAMD64FlagLT_UGT)
		return true
	}
	// match: (CMPQ (MOVQconst [x]) (MOVQconst [y]))
	// cond: x>y && uint64(x)<uint64(y)
	// result: (FlagGT_ULT)
	for {
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		y := auxIntToInt64(v_1.AuxInt)
		if !(x > y && uint64(x) < uint64(y)) {
			break
		}
		v.reset(OpAMD64FlagGT_ULT)
		return true
	}
	// match: (CMPQ (MOVQconst [x]) (MOVQconst [y]))
	// cond: x>y && uint64(x)>uint64(y)
	// result: (FlagGT_UGT)
	for {
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		y := auxIntToInt64(v_1.AuxInt)
		if !(x > y && uint64(x) > uint64(y)) {
			break
		}
		v.reset(OpAMD64FlagGT_UGT)
		return true
	}
	// match: (CMPQ l:(MOVQload {sym} [off] ptr mem) x)
	// cond: canMergeLoad(v, l) && clobber(l)
	// result: (CMPQload {sym} [off] ptr x mem)
	for {
		l := v_0
		if l.Op != OpAMD64MOVQload {
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
		v.reset(OpAMD64CMPQload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (CMPQ x l:(MOVQload {sym} [off] ptr mem))
	// cond: canMergeLoad(v, l) && clobber(l)
	// result: (InvertFlags (CMPQload {sym} [off] ptr x mem))
	for {
		x := v_0
		l := v_1
		if l.Op != OpAMD64MOVQload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(canMergeLoad(v, l) && clobber(l)) {
			break
		}
		v.reset(OpAMD64InvertFlags)
		v0 := b.NewValue0(l.Pos, OpAMD64CMPQload, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, x, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPQconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPQconst (MOVQconst [x]) [y])
	// cond: x==int64(y)
	// result: (FlagEQ)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if !(x == int64(y)) {
			break
		}
		v.reset(OpAMD64FlagEQ)
		return true
	}
	// match: (CMPQconst (MOVQconst [x]) [y])
	// cond: x<int64(y) && uint64(x)<uint64(int64(y))
	// result: (FlagLT_ULT)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if !(x < int64(y) && uint64(x) < uint64(int64(y))) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPQconst (MOVQconst [x]) [y])
	// cond: x<int64(y) && uint64(x)>uint64(int64(y))
	// result: (FlagLT_UGT)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if !(x < int64(y) && uint64(x) > uint64(int64(y))) {
			break
		}
		v.reset(OpAMD64FlagLT_UGT)
		return true
	}
	// match: (CMPQconst (MOVQconst [x]) [y])
	// cond: x>int64(y) && uint64(x)<uint64(int64(y))
	// result: (FlagGT_ULT)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if !(x > int64(y) && uint64(x) < uint64(int64(y))) {
			break
		}
		v.reset(OpAMD64FlagGT_ULT)
		return true
	}
	// match: (CMPQconst (MOVQconst [x]) [y])
	// cond: x>int64(y) && uint64(x)>uint64(int64(y))
	// result: (FlagGT_UGT)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if !(x > int64(y) && uint64(x) > uint64(int64(y))) {
			break
		}
		v.reset(OpAMD64FlagGT_UGT)
		return true
	}
	// match: (CMPQconst (MOVBQZX _) [c])
	// cond: 0xFF < c
	// result: (FlagLT_ULT)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVBQZX || !(0xFF < c) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPQconst (MOVWQZX _) [c])
	// cond: 0xFFFF < c
	// result: (FlagLT_ULT)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVWQZX || !(0xFFFF < c) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPQconst (SHRQconst _ [c]) [n])
	// cond: 0 <= n && 0 < c && c <= 64 && (1<<uint64(64-c)) <= uint64(n)
	// result: (FlagLT_ULT)
	for {
		n := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64SHRQconst {
			break
		}
		c := auxIntToInt8(v_0.AuxInt)
		if !(0 <= n && 0 < c && c <= 64 && (1<<uint64(64-c)) <= uint64(n)) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPQconst (ANDQconst _ [m]) [n])
	// cond: 0 <= m && m < n
	// result: (FlagLT_ULT)
	for {
		n := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64ANDQconst {
			break
		}
		m := auxIntToInt32(v_0.AuxInt)
		if !(0 <= m && m < n) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPQconst (ANDLconst _ [m]) [n])
	// cond: 0 <= m && m < n
	// result: (FlagLT_ULT)
	for {
		n := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		m := auxIntToInt32(v_0.AuxInt)
		if !(0 <= m && m < n) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPQconst a:(ANDQ x y) [0])
	// cond: a.Uses == 1
	// result: (TESTQ x y)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		a := v_0
		if a.Op != OpAMD64ANDQ {
			break
		}
		y := a.Args[1]
		x := a.Args[0]
		if !(a.Uses == 1) {
			break
		}
		v.reset(OpAMD64TESTQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMPQconst a:(ANDQconst [c] x) [0])
	// cond: a.Uses == 1
	// result: (TESTQconst [c] x)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		a := v_0
		if a.Op != OpAMD64ANDQconst {
			break
		}
		c := auxIntToInt32(a.AuxInt)
		x := a.Args[0]
		if !(a.Uses == 1) {
			break
		}
		v.reset(OpAMD64TESTQconst)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (CMPQconst x [0])
	// result: (TESTQ x x)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.reset(OpAMD64TESTQ)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPQconst l:(MOVQload {sym} [off] ptr mem) [c])
	// cond: l.Uses == 1 && clobber(l)
	// result: @l.Block (CMPQconstload {sym} [makeValAndOff(c,off)] ptr mem)
	for {
		c := auxIntToInt32(v.AuxInt)
		l := v_0
		if l.Op != OpAMD64MOVQload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(l.Uses == 1 && clobber(l)) {
			break
		}
		b = l.Block
		v0 := b.NewValue0(l.Pos, OpAMD64CMPQconstload, types.TypeFlags)
		v.copyOf(v0)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(c, off))
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPQconstload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMPQconstload [valoff1] {sym} (ADDQconst [off2] base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2)
	// result: (CMPQconstload [ValAndOff(valoff1).addOffset32(off2)] {sym} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2)) {
			break
		}
		v.reset(OpAMD64CMPQconstload)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (CMPQconstload [valoff1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)
	// result: (CMPQconstload [ValAndOff(valoff1).addOffset32(off2)] {mergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64CMPQconstload)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPQload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMPQload [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (CMPQload [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64CMPQload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (CMPQload [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (CMPQload [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64CMPQload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (CMPQload {sym} [off] ptr (MOVQconst [c]) mem)
	// cond: validVal(c)
	// result: (CMPQconstload {sym} [makeValAndOff(int32(c),off)] ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(validVal(c)) {
			break
		}
		v.reset(OpAMD64CMPQconstload)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(c), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPW x (MOVLconst [c]))
	// result: (CMPWconst x [int16(c)])
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64CMPWconst)
		v.AuxInt = int16ToAuxInt(int16(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPW (MOVLconst [c]) x)
	// result: (InvertFlags (CMPWconst x [int16(c)]))
	for {
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_1
		v.reset(OpAMD64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v0.AuxInt = int16ToAuxInt(int16(c))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPW x y)
	// cond: canonLessThan(x,y)
	// result: (InvertFlags (CMPW y x))
	for {
		x := v_0
		y := v_1
		if !(canonLessThan(x, y)) {
			break
		}
		v.reset(OpAMD64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPW, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPW l:(MOVWload {sym} [off] ptr mem) x)
	// cond: canMergeLoad(v, l) && clobber(l)
	// result: (CMPWload {sym} [off] ptr x mem)
	for {
		l := v_0
		if l.Op != OpAMD64MOVWload {
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
		v.reset(OpAMD64CMPWload)
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
		if l.Op != OpAMD64MOVWload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(canMergeLoad(v, l) && clobber(l)) {
			break
		}
		v.reset(OpAMD64InvertFlags)
		v0 := b.NewValue0(l.Pos, OpAMD64CMPWload, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg3(ptr, x, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPWconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)==y
	// result: (FlagEQ)
	for {
		y := auxIntToInt16(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int16(x) == y) {
			break
		}
		v.reset(OpAMD64FlagEQ)
		return true
	}
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)<y && uint16(x)<uint16(y)
	// result: (FlagLT_ULT)
	for {
		y := auxIntToInt16(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int16(x) < y && uint16(x) < uint16(y)) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)<y && uint16(x)>uint16(y)
	// result: (FlagLT_UGT)
	for {
		y := auxIntToInt16(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int16(x) < y && uint16(x) > uint16(y)) {
			break
		}
		v.reset(OpAMD64FlagLT_UGT)
		return true
	}
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)>y && uint16(x)<uint16(y)
	// result: (FlagGT_ULT)
	for {
		y := auxIntToInt16(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int16(x) > y && uint16(x) < uint16(y)) {
			break
		}
		v.reset(OpAMD64FlagGT_ULT)
		return true
	}
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)>y && uint16(x)>uint16(y)
	// result: (FlagGT_UGT)
	for {
		y := auxIntToInt16(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := auxIntToInt32(v_0.AuxInt)
		if !(int16(x) > y && uint16(x) > uint16(y)) {
			break
		}
		v.reset(OpAMD64FlagGT_UGT)
		return true
	}
	// match: (CMPWconst (ANDLconst _ [m]) [n])
	// cond: 0 <= int16(m) && int16(m) < n
	// result: (FlagLT_ULT)
	for {
		n := auxIntToInt16(v.AuxInt)
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		m := auxIntToInt32(v_0.AuxInt)
		if !(0 <= int16(m) && int16(m) < n) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPWconst a:(ANDL x y) [0])
	// cond: a.Uses == 1
	// result: (TESTW x y)
	for {
		if auxIntToInt16(v.AuxInt) != 0 {
			break
		}
		a := v_0
		if a.Op != OpAMD64ANDL {
			break
		}
		y := a.Args[1]
		x := a.Args[0]
		if !(a.Uses == 1) {
			break
		}
		v.reset(OpAMD64TESTW)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMPWconst a:(ANDLconst [c] x) [0])
	// cond: a.Uses == 1
	// result: (TESTWconst [int16(c)] x)
	for {
		if auxIntToInt16(v.AuxInt) != 0 {
			break
		}
		a := v_0
		if a.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(a.AuxInt)
		x := a.Args[0]
		if !(a.Uses == 1) {
			break
		}
		v.reset(OpAMD64TESTWconst)
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
		v.reset(OpAMD64TESTW)
		v.AddArg2(x, x)
		return true
	}
	// match: (CMPWconst l:(MOVWload {sym} [off] ptr mem) [c])
	// cond: l.Uses == 1 && clobber(l)
	// result: @l.Block (CMPWconstload {sym} [makeValAndOff(int32(c),off)] ptr mem)
	for {
		c := auxIntToInt16(v.AuxInt)
		l := v_0
		if l.Op != OpAMD64MOVWload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(l.Uses == 1 && clobber(l)) {
			break
		}
		b = l.Block
		v0 := b.NewValue0(l.Pos, OpAMD64CMPWconstload, types.TypeFlags)
		v.copyOf(v0)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(c), off))
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPWconstload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMPWconstload [valoff1] {sym} (ADDQconst [off2] base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2)
	// result: (CMPWconstload [ValAndOff(valoff1).addOffset32(off2)] {sym} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2)) {
			break
		}
		v.reset(OpAMD64CMPWconstload)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (CMPWconstload [valoff1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)
	// result: (CMPWconstload [ValAndOff(valoff1).addOffset32(off2)] {mergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64CMPWconstload)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPWload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMPWload [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (CMPWload [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64CMPWload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (CMPWload [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (CMPWload [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64CMPWload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (CMPWload {sym} [off] ptr (MOVLconst [c]) mem)
	// result: (CMPWconstload {sym} [makeValAndOff(int32(int16(c)),off)] ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64CMPWconstload)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(int16(c)), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPXCHGLlock(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMPXCHGLlock [off1] {sym} (ADDQconst [off2] ptr) old new_ mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (CMPXCHGLlock [off1+off2] {sym} ptr old new_ mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		old := v_1
		new_ := v_2
		mem := v_3
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64CMPXCHGLlock)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg4(ptr, old, new_, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPXCHGQlock(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMPXCHGQlock [off1] {sym} (ADDQconst [off2] ptr) old new_ mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (CMPXCHGQlock [off1+off2] {sym} ptr old new_ mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		old := v_1
		new_ := v_2
		mem := v_3
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64CMPXCHGQlock)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg4(ptr, old, new_, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64DIVSD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (DIVSD x l:(MOVSDload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (DIVSDload x [off] {sym} ptr mem)
	for {
		x := v_0
		l := v_1
		if l.Op != OpAMD64MOVSDload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64DIVSDload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(x, ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64DIVSDload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (DIVSDload [off1] {sym} val (ADDQconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (DIVSDload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64DIVSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (DIVSDload [off1] {sym1} val (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (DIVSDload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64DIVSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64DIVSS(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (DIVSS x l:(MOVSSload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (DIVSSload x [off] {sym} ptr mem)
	for {
		x := v_0
		l := v_1
		if l.Op != OpAMD64MOVSSload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64DIVSSload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(x, ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64DIVSSload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (DIVSSload [off1] {sym} val (ADDQconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (DIVSSload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64DIVSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (DIVSSload [off1] {sym1} val (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (DIVSSload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64DIVSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64HMULL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (HMULL x y)
	// cond: !x.rematerializeable() && y.rematerializeable()
	// result: (HMULL y x)
	for {
		x := v_0
		y := v_1
		if !(!x.rematerializeable() && y.rematerializeable()) {
			break
		}
		v.reset(OpAMD64HMULL)
		v.AddArg2(y, x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64HMULLU(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (HMULLU x y)
	// cond: !x.rematerializeable() && y.rematerializeable()
	// result: (HMULLU y x)
	for {
		x := v_0
		y := v_1
		if !(!x.rematerializeable() && y.rematerializeable()) {
			break
		}
		v.reset(OpAMD64HMULLU)
		v.AddArg2(y, x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64HMULQ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (HMULQ x y)
	// cond: !x.rematerializeable() && y.rematerializeable()
	// result: (HMULQ y x)
	for {
		x := v_0
		y := v_1
		if !(!x.rematerializeable() && y.rematerializeable()) {
			break
		}
		v.reset(OpAMD64HMULQ)
		v.AddArg2(y, x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64HMULQU(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (HMULQU x y)
	// cond: !x.rematerializeable() && y.rematerializeable()
	// result: (HMULQU y x)
	for {
		x := v_0
		y := v_1
		if !(!x.rematerializeable() && y.rematerializeable()) {
			break
		}
		v.reset(OpAMD64HMULQU)
		v.AddArg2(y, x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64LEAL(v *Value) bool {
	v_0 := v.Args[0]
	// match: (LEAL [c] {s} (ADDLconst [d] x))
	// cond: is32Bit(int64(c)+int64(d))
	// result: (LEAL [c+d] {s} x)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(OpAMD64LEAL)
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
		if v_0.Op != OpAMD64ADDL {
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
			v.reset(OpAMD64LEAL1)
			v.AuxInt = int32ToAuxInt(c)
			v.Aux = symToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64LEAL1(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LEAL1 [c] {s} (ADDLconst [d] x) y)
	// cond: is32Bit(int64(c)+int64(d)) && x.Op != OpSB
	// result: (LEAL1 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64ADDLconst {
				continue
			}
			d := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			y := v_1
			if !(is32Bit(int64(c)+int64(d)) && x.Op != OpSB) {
				continue
			}
			v.reset(OpAMD64LEAL1)
			v.AuxInt = int32ToAuxInt(c + d)
			v.Aux = symToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAL1 [c] {s} x z:(ADDL y y))
	// cond: x != z
	// result: (LEAL2 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			z := v_1
			if z.Op != OpAMD64ADDL {
				continue
			}
			y := z.Args[1]
			if y != z.Args[0] || !(x != z) {
				continue
			}
			v.reset(OpAMD64LEAL2)
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
			if v_1.Op != OpAMD64SHLLconst || auxIntToInt8(v_1.AuxInt) != 2 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpAMD64LEAL4)
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
			if v_1.Op != OpAMD64SHLLconst || auxIntToInt8(v_1.AuxInt) != 3 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpAMD64LEAL8)
			v.AuxInt = int32ToAuxInt(c)
			v.Aux = symToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64LEAL2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LEAL2 [c] {s} (ADDLconst [d] x) y)
	// cond: is32Bit(int64(c)+int64(d)) && x.Op != OpSB
	// result: (LEAL2 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		y := v_1
		if !(is32Bit(int64(c)+int64(d)) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAL2)
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
		if v_1.Op != OpAMD64ADDLconst {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(is32Bit(int64(c)+2*int64(d)) && y.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAL2)
		v.AuxInt = int32ToAuxInt(c + 2*d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL2 [c] {s} x z:(ADDL y y))
	// cond: x != z
	// result: (LEAL4 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		z := v_1
		if z.Op != OpAMD64ADDL {
			break
		}
		y := z.Args[1]
		if y != z.Args[0] || !(x != z) {
			break
		}
		v.reset(OpAMD64LEAL4)
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
		if v_1.Op != OpAMD64SHLLconst || auxIntToInt8(v_1.AuxInt) != 2 {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64LEAL8)
		v.AuxInt = int32ToAuxInt(c)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL2 [0] {s} (ADDL x x) x)
	// cond: s == nil
	// result: (SHLLconst [2] x)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDL {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] || x != v_1 || !(s == nil) {
			break
		}
		v.reset(OpAMD64SHLLconst)
		v.AuxInt = int8ToAuxInt(2)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64LEAL4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LEAL4 [c] {s} (ADDLconst [d] x) y)
	// cond: is32Bit(int64(c)+int64(d)) && x.Op != OpSB
	// result: (LEAL4 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		y := v_1
		if !(is32Bit(int64(c)+int64(d)) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAL4)
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
		if v_1.Op != OpAMD64ADDLconst {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(is32Bit(int64(c)+4*int64(d)) && y.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAL4)
		v.AuxInt = int32ToAuxInt(c + 4*d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAL4 [c] {s} x z:(ADDL y y))
	// cond: x != z
	// result: (LEAL8 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		z := v_1
		if z.Op != OpAMD64ADDL {
			break
		}
		y := z.Args[1]
		if y != z.Args[0] || !(x != z) {
			break
		}
		v.reset(OpAMD64LEAL8)
		v.AuxInt = int32ToAuxInt(c)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64LEAL8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LEAL8 [c] {s} (ADDLconst [d] x) y)
	// cond: is32Bit(int64(c)+int64(d)) && x.Op != OpSB
	// result: (LEAL8 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		y := v_1
		if !(is32Bit(int64(c)+int64(d)) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAL8)
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
		if v_1.Op != OpAMD64ADDLconst {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(is32Bit(int64(c)+8*int64(d)) && y.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAL8)
		v.AuxInt = int32ToAuxInt(c + 8*d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64LEAQ(v *Value) bool {
	v_0 := v.Args[0]
	// match: (LEAQ [c] {s} (ADDQconst [d] x))
	// cond: is32Bit(int64(c)+int64(d))
	// result: (LEAQ [c+d] {s} x)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(int64(c) + int64(d))) {
			break
		}
		v.reset(OpAMD64LEAQ)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg(x)
		return true
	}
	// match: (LEAQ [c] {s} (ADDQ x y))
	// cond: x.Op != OpSB && y.Op != OpSB
	// result: (LEAQ1 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQ {
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
			v.reset(OpAMD64LEAQ1)
			v.AuxInt = int32ToAuxInt(c)
			v.Aux = symToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAQ [off1] {sym1} (LEAQ [off2] {sym2} x))
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (LEAQ [off1+off2] {mergeSym(sym1,sym2)} x)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		x := v_0.Args[0]
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64LEAQ)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg(x)
		return true
	}
	// match: (LEAQ [off1] {sym1} (LEAQ1 [off2] {sym2} x y))
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (LEAQ1 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64LEAQ1)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAQ [off1] {sym1} (LEAQ2 [off2] {sym2} x y))
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (LEAQ2 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ2 {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64LEAQ2)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAQ [off1] {sym1} (LEAQ4 [off2] {sym2} x y))
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (LEAQ4 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ4 {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64LEAQ4)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAQ [off1] {sym1} (LEAQ8 [off2] {sym2} x y))
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (LEAQ8 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ8 {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64LEAQ8)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64LEAQ1(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LEAQ1 [c] {s} (ADDQconst [d] x) y)
	// cond: is32Bit(int64(c)+int64(d)) && x.Op != OpSB
	// result: (LEAQ1 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64ADDQconst {
				continue
			}
			d := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			y := v_1
			if !(is32Bit(int64(c)+int64(d)) && x.Op != OpSB) {
				continue
			}
			v.reset(OpAMD64LEAQ1)
			v.AuxInt = int32ToAuxInt(c + d)
			v.Aux = symToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAQ1 [c] {s} x z:(ADDQ y y))
	// cond: x != z
	// result: (LEAQ2 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			z := v_1
			if z.Op != OpAMD64ADDQ {
				continue
			}
			y := z.Args[1]
			if y != z.Args[0] || !(x != z) {
				continue
			}
			v.reset(OpAMD64LEAQ2)
			v.AuxInt = int32ToAuxInt(c)
			v.Aux = symToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAQ1 [c] {s} x (SHLQconst [2] y))
	// result: (LEAQ4 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64SHLQconst || auxIntToInt8(v_1.AuxInt) != 2 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpAMD64LEAQ4)
			v.AuxInt = int32ToAuxInt(c)
			v.Aux = symToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAQ1 [c] {s} x (SHLQconst [3] y))
	// result: (LEAQ8 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64SHLQconst || auxIntToInt8(v_1.AuxInt) != 3 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpAMD64LEAQ8)
			v.AuxInt = int32ToAuxInt(c)
			v.Aux = symToAux(s)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAQ1 [off1] {sym1} (LEAQ [off2] {sym2} x) y)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && x.Op != OpSB
	// result: (LEAQ1 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64LEAQ {
				continue
			}
			off2 := auxIntToInt32(v_0.AuxInt)
			sym2 := auxToSym(v_0.Aux)
			x := v_0.Args[0]
			y := v_1
			if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && x.Op != OpSB) {
				continue
			}
			v.reset(OpAMD64LEAQ1)
			v.AuxInt = int32ToAuxInt(off1 + off2)
			v.Aux = symToAux(mergeSym(sym1, sym2))
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAQ1 [off1] {sym1} x (LEAQ1 [off2] {sym2} y y))
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (LEAQ2 [off1+off2] {mergeSym(sym1, sym2)} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64LEAQ1 {
				continue
			}
			off2 := auxIntToInt32(v_1.AuxInt)
			sym2 := auxToSym(v_1.Aux)
			y := v_1.Args[1]
			if y != v_1.Args[0] || !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
				continue
			}
			v.reset(OpAMD64LEAQ2)
			v.AuxInt = int32ToAuxInt(off1 + off2)
			v.Aux = symToAux(mergeSym(sym1, sym2))
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (LEAQ1 [off1] {sym1} x (LEAQ1 [off2] {sym2} x y))
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (LEAQ2 [off1+off2] {mergeSym(sym1, sym2)} y x)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64LEAQ1 {
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
				v.reset(OpAMD64LEAQ2)
				v.AuxInt = int32ToAuxInt(off1 + off2)
				v.Aux = symToAux(mergeSym(sym1, sym2))
				v.AddArg2(y, x)
				return true
			}
		}
		break
	}
	// match: (LEAQ1 [0] x y)
	// cond: v.Aux == nil
	// result: (ADDQ x y)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		y := v_1
		if !(v.Aux == nil) {
			break
		}
		v.reset(OpAMD64ADDQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64LEAQ2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LEAQ2 [c] {s} (ADDQconst [d] x) y)
	// cond: is32Bit(int64(c)+int64(d)) && x.Op != OpSB
	// result: (LEAQ2 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		y := v_1
		if !(is32Bit(int64(c)+int64(d)) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ2)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAQ2 [c] {s} x (ADDQconst [d] y))
	// cond: is32Bit(int64(c)+2*int64(d)) && y.Op != OpSB
	// result: (LEAQ2 [c+2*d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(is32Bit(int64(c)+2*int64(d)) && y.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ2)
		v.AuxInt = int32ToAuxInt(c + 2*d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAQ2 [c] {s} x z:(ADDQ y y))
	// cond: x != z
	// result: (LEAQ4 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		z := v_1
		if z.Op != OpAMD64ADDQ {
			break
		}
		y := z.Args[1]
		if y != z.Args[0] || !(x != z) {
			break
		}
		v.reset(OpAMD64LEAQ4)
		v.AuxInt = int32ToAuxInt(c)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAQ2 [c] {s} x (SHLQconst [2] y))
	// result: (LEAQ8 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != OpAMD64SHLQconst || auxIntToInt8(v_1.AuxInt) != 2 {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64LEAQ8)
		v.AuxInt = int32ToAuxInt(c)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAQ2 [0] {s} (ADDQ x x) x)
	// cond: s == nil
	// result: (SHLQconst [2] x)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] || x != v_1 || !(s == nil) {
			break
		}
		v.reset(OpAMD64SHLQconst)
		v.AuxInt = int8ToAuxInt(2)
		v.AddArg(x)
		return true
	}
	// match: (LEAQ2 [off1] {sym1} (LEAQ [off2] {sym2} x) y)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && x.Op != OpSB
	// result: (LEAQ2 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		x := v_0.Args[0]
		y := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ2)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAQ2 [off1] {sym1} x (LEAQ1 [off2] {sym2} y y))
	// cond: is32Bit(int64(off1)+2*int64(off2)) && sym2 == nil
	// result: (LEAQ4 [off1+2*off2] {sym1} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != OpAMD64LEAQ1 {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		y := v_1.Args[1]
		if y != v_1.Args[0] || !(is32Bit(int64(off1)+2*int64(off2)) && sym2 == nil) {
			break
		}
		v.reset(OpAMD64LEAQ4)
		v.AuxInt = int32ToAuxInt(off1 + 2*off2)
		v.Aux = symToAux(sym1)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAQ2 [off] {sym} x (MOVQconst [scale]))
	// cond: is32Bit(int64(off)+int64(scale)*2)
	// result: (LEAQ [off+int32(scale)*2] {sym} x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		scale := auxIntToInt64(v_1.AuxInt)
		if !(is32Bit(int64(off) + int64(scale)*2)) {
			break
		}
		v.reset(OpAMD64LEAQ)
		v.AuxInt = int32ToAuxInt(off + int32(scale)*2)
		v.Aux = symToAux(sym)
		v.AddArg(x)
		return true
	}
	// match: (LEAQ2 [off] {sym} x (MOVLconst [scale]))
	// cond: is32Bit(int64(off)+int64(scale)*2)
	// result: (LEAQ [off+int32(scale)*2] {sym} x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		scale := auxIntToInt32(v_1.AuxInt)
		if !(is32Bit(int64(off) + int64(scale)*2)) {
			break
		}
		v.reset(OpAMD64LEAQ)
		v.AuxInt = int32ToAuxInt(off + int32(scale)*2)
		v.Aux = symToAux(sym)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64LEAQ4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LEAQ4 [c] {s} (ADDQconst [d] x) y)
	// cond: is32Bit(int64(c)+int64(d)) && x.Op != OpSB
	// result: (LEAQ4 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		y := v_1
		if !(is32Bit(int64(c)+int64(d)) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ4)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAQ4 [c] {s} x (ADDQconst [d] y))
	// cond: is32Bit(int64(c)+4*int64(d)) && y.Op != OpSB
	// result: (LEAQ4 [c+4*d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(is32Bit(int64(c)+4*int64(d)) && y.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ4)
		v.AuxInt = int32ToAuxInt(c + 4*d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAQ4 [c] {s} x z:(ADDQ y y))
	// cond: x != z
	// result: (LEAQ8 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		z := v_1
		if z.Op != OpAMD64ADDQ {
			break
		}
		y := z.Args[1]
		if y != z.Args[0] || !(x != z) {
			break
		}
		v.reset(OpAMD64LEAQ8)
		v.AuxInt = int32ToAuxInt(c)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAQ4 [off1] {sym1} (LEAQ [off2] {sym2} x) y)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && x.Op != OpSB
	// result: (LEAQ4 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		x := v_0.Args[0]
		y := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ4)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAQ4 [off1] {sym1} x (LEAQ1 [off2] {sym2} y y))
	// cond: is32Bit(int64(off1)+4*int64(off2)) && sym2 == nil
	// result: (LEAQ8 [off1+4*off2] {sym1} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != OpAMD64LEAQ1 {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		y := v_1.Args[1]
		if y != v_1.Args[0] || !(is32Bit(int64(off1)+4*int64(off2)) && sym2 == nil) {
			break
		}
		v.reset(OpAMD64LEAQ8)
		v.AuxInt = int32ToAuxInt(off1 + 4*off2)
		v.Aux = symToAux(sym1)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAQ4 [off] {sym} x (MOVQconst [scale]))
	// cond: is32Bit(int64(off)+int64(scale)*4)
	// result: (LEAQ [off+int32(scale)*4] {sym} x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		scale := auxIntToInt64(v_1.AuxInt)
		if !(is32Bit(int64(off) + int64(scale)*4)) {
			break
		}
		v.reset(OpAMD64LEAQ)
		v.AuxInt = int32ToAuxInt(off + int32(scale)*4)
		v.Aux = symToAux(sym)
		v.AddArg(x)
		return true
	}
	// match: (LEAQ4 [off] {sym} x (MOVLconst [scale]))
	// cond: is32Bit(int64(off)+int64(scale)*4)
	// result: (LEAQ [off+int32(scale)*4] {sym} x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		scale := auxIntToInt32(v_1.AuxInt)
		if !(is32Bit(int64(off) + int64(scale)*4)) {
			break
		}
		v.reset(OpAMD64LEAQ)
		v.AuxInt = int32ToAuxInt(off + int32(scale)*4)
		v.Aux = symToAux(sym)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64LEAQ8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LEAQ8 [c] {s} (ADDQconst [d] x) y)
	// cond: is32Bit(int64(c)+int64(d)) && x.Op != OpSB
	// result: (LEAQ8 [c+d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		y := v_1
		if !(is32Bit(int64(c)+int64(d)) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ8)
		v.AuxInt = int32ToAuxInt(c + d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAQ8 [c] {s} x (ADDQconst [d] y))
	// cond: is32Bit(int64(c)+8*int64(d)) && y.Op != OpSB
	// result: (LEAQ8 [c+8*d] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(is32Bit(int64(c)+8*int64(d)) && y.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ8)
		v.AuxInt = int32ToAuxInt(c + 8*d)
		v.Aux = symToAux(s)
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAQ8 [off1] {sym1} (LEAQ [off2] {sym2} x) y)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && x.Op != OpSB
	// result: (LEAQ8 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		x := v_0.Args[0]
		y := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ8)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(x, y)
		return true
	}
	// match: (LEAQ8 [off] {sym} x (MOVQconst [scale]))
	// cond: is32Bit(int64(off)+int64(scale)*8)
	// result: (LEAQ [off+int32(scale)*8] {sym} x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		scale := auxIntToInt64(v_1.AuxInt)
		if !(is32Bit(int64(off) + int64(scale)*8)) {
			break
		}
		v.reset(OpAMD64LEAQ)
		v.AuxInt = int32ToAuxInt(off + int32(scale)*8)
		v.Aux = symToAux(sym)
		v.AddArg(x)
		return true
	}
	// match: (LEAQ8 [off] {sym} x (MOVLconst [scale]))
	// cond: is32Bit(int64(off)+int64(scale)*8)
	// result: (LEAQ [off+int32(scale)*8] {sym} x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		scale := auxIntToInt32(v_1.AuxInt)
		if !(is32Bit(int64(off) + int64(scale)*8)) {
			break
		}
		v.reset(OpAMD64LEAQ)
		v.AuxInt = int32ToAuxInt(off + int32(scale)*8)
		v.Aux = symToAux(sym)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64LoweredPanicBoundsCR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsCR [kind] {p} (MOVQconst [c]) mem)
	// result: (LoweredPanicBoundsCC [kind] {PanicBoundsCC{Cx:p.C, Cy:c}} mem)
	for {
		kind := auxIntToInt64(v.AuxInt)
		p := auxToPanicBoundsC(v.Aux)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		mem := v_1
		v.reset(OpAMD64LoweredPanicBoundsCC)
		v.AuxInt = int64ToAuxInt(kind)
		v.Aux = panicBoundsCCToAux(PanicBoundsCC{Cx: p.C, Cy: c})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64LoweredPanicBoundsRC(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRC [kind] {p} (MOVQconst [c]) mem)
	// result: (LoweredPanicBoundsCC [kind] {PanicBoundsCC{Cx:c, Cy:p.C}} mem)
	for {
		kind := auxIntToInt64(v.AuxInt)
		p := auxToPanicBoundsC(v.Aux)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		mem := v_1
		v.reset(OpAMD64LoweredPanicBoundsCC)
		v.AuxInt = int64ToAuxInt(kind)
		v.Aux = panicBoundsCCToAux(PanicBoundsCC{Cx: c, Cy: p.C})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64LoweredPanicBoundsRR(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRR [kind] x (MOVQconst [c]) mem)
	// result: (LoweredPanicBoundsRC [kind] x {PanicBoundsC{C:c}} mem)
	for {
		kind := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64LoweredPanicBoundsRC)
		v.AuxInt = int64ToAuxInt(kind)
		v.Aux = panicBoundsCToAux(PanicBoundsC{C: c})
		v.AddArg2(x, mem)
		return true
	}
	// match: (LoweredPanicBoundsRR [kind] (MOVQconst [c]) y mem)
	// result: (LoweredPanicBoundsCR [kind] {PanicBoundsC{C:c}} y mem)
	for {
		kind := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		y := v_1
		mem := v_2
		v.reset(OpAMD64LoweredPanicBoundsCR)
		v.AuxInt = int64ToAuxInt(kind)
		v.Aux = panicBoundsCToAux(PanicBoundsC{C: c})
		v.AddArg2(y, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBELstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBELstore [i] {s} p x:(BSWAPL w) mem)
	// cond: x.Uses == 1
	// result: (MOVLstore [i] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		x := v_1
		if x.Op != OpAMD64BSWAPL {
			break
		}
		w := x.Args[0]
		mem := v_2
		if !(x.Uses == 1) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p, w, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBEQstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBEQstore [i] {s} p x:(BSWAPQ w) mem)
	// cond: x.Uses == 1
	// result: (MOVQstore [i] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		x := v_1
		if x.Op != OpAMD64BSWAPQ {
			break
		}
		w := x.Args[0]
		mem := v_2
		if !(x.Uses == 1) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p, w, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBEWstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBEWstore [i] {s} p x:(ROLWconst [8] w) mem)
	// cond: x.Uses == 1
	// result: (MOVWstore [i] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		x := v_1
		if x.Op != OpAMD64ROLWconst || auxIntToInt8(x.AuxInt) != 8 {
			break
		}
		w := x.Args[0]
		mem := v_2
		if !(x.Uses == 1) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p, w, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBQSX(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVBQSX x:(MOVBload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBQSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVBload {
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
		v0 := b.NewValue0(x.Pos, OpAMD64MOVBQSXload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBQSX x:(MOVWload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBQSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVWload {
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
		v0 := b.NewValue0(x.Pos, OpAMD64MOVBQSXload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBQSX x:(MOVLload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBQSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVLload {
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
		v0 := b.NewValue0(x.Pos, OpAMD64MOVBQSXload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBQSX x:(MOVQload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBQSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVQload {
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
		v0 := b.NewValue0(x.Pos, OpAMD64MOVBQSXload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBQSX (ANDLconst [c] x))
	// cond: c & 0x80 == 0
	// result: (ANDLconst [c & 0x7f] x)
	for {
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c&0x80 == 0) {
			break
		}
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = int32ToAuxInt(c & 0x7f)
		v.AddArg(x)
		return true
	}
	// match: (MOVBQSX (MOVBQSX x))
	// result: (MOVBQSX x)
	for {
		if v_0.Op != OpAMD64MOVBQSX {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64MOVBQSX)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBQSXload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBQSXload [off] {sym} ptr (MOVBstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVBQSX x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVBstore {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpAMD64MOVBQSX)
		v.AddArg(x)
		return true
	}
	// match: (MOVBQSXload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVBQSXload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVBQSXload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVBQSXload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVQconst [int64(int8(read8(sym, int64(off))))])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpSB || !(symIsRO(sym)) {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(int64(int8(read8(sym, int64(off)))))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBQZX(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVBQZX x:(MOVBload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVBload {
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
		v0 := b.NewValue0(x.Pos, OpAMD64MOVBload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBQZX x:(MOVWload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVWload {
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
		v0 := b.NewValue0(x.Pos, OpAMD64MOVBload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBQZX x:(MOVLload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVLload {
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
		v0 := b.NewValue0(x.Pos, OpAMD64MOVBload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBQZX x:(MOVQload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVQload {
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
		v0 := b.NewValue0(x.Pos, OpAMD64MOVBload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBQZX (ANDLconst [c] x))
	// result: (ANDLconst [c & 0xff] x)
	for {
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = int32ToAuxInt(c & 0xff)
		v.AddArg(x)
		return true
	}
	// match: (MOVBQZX (MOVBQZX x))
	// result: (MOVBQZX x)
	for {
		if v_0.Op != OpAMD64MOVBQZX {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64MOVBQZX)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBatomicload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBatomicload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVBatomicload [off1+off2] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MOVBatomicload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBatomicload [off1] {sym1} (LEAQ [off2] {sym2} ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVBatomicload [off1+off2] {mergeSym(sym1, sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVBatomicload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBload [off] {sym} ptr (MOVBstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVBQZX x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVBstore {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpAMD64MOVBQZX)
		v.AddArg(x)
		return true
	}
	// match: (MOVBload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVBload [off1+off2] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MOVBload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVBload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVBload)
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
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(int32(read8(sym, int64(off))))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBstore [off] {sym} ptr y:(SETL x) mem)
	// cond: y.Uses == 1
	// result: (SETLstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64SETL {
			break
		}
		x := y.Args[0]
		mem := v_2
		if !(y.Uses == 1) {
			break
		}
		v.reset(OpAMD64SETLstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr y:(SETLE x) mem)
	// cond: y.Uses == 1
	// result: (SETLEstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64SETLE {
			break
		}
		x := y.Args[0]
		mem := v_2
		if !(y.Uses == 1) {
			break
		}
		v.reset(OpAMD64SETLEstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr y:(SETG x) mem)
	// cond: y.Uses == 1
	// result: (SETGstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64SETG {
			break
		}
		x := y.Args[0]
		mem := v_2
		if !(y.Uses == 1) {
			break
		}
		v.reset(OpAMD64SETGstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr y:(SETGE x) mem)
	// cond: y.Uses == 1
	// result: (SETGEstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64SETGE {
			break
		}
		x := y.Args[0]
		mem := v_2
		if !(y.Uses == 1) {
			break
		}
		v.reset(OpAMD64SETGEstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr y:(SETEQ x) mem)
	// cond: y.Uses == 1
	// result: (SETEQstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64SETEQ {
			break
		}
		x := y.Args[0]
		mem := v_2
		if !(y.Uses == 1) {
			break
		}
		v.reset(OpAMD64SETEQstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr y:(SETNE x) mem)
	// cond: y.Uses == 1
	// result: (SETNEstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64SETNE {
			break
		}
		x := y.Args[0]
		mem := v_2
		if !(y.Uses == 1) {
			break
		}
		v.reset(OpAMD64SETNEstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr y:(SETB x) mem)
	// cond: y.Uses == 1
	// result: (SETBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64SETB {
			break
		}
		x := y.Args[0]
		mem := v_2
		if !(y.Uses == 1) {
			break
		}
		v.reset(OpAMD64SETBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr y:(SETBE x) mem)
	// cond: y.Uses == 1
	// result: (SETBEstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64SETBE {
			break
		}
		x := y.Args[0]
		mem := v_2
		if !(y.Uses == 1) {
			break
		}
		v.reset(OpAMD64SETBEstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr y:(SETA x) mem)
	// cond: y.Uses == 1
	// result: (SETAstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64SETA {
			break
		}
		x := y.Args[0]
		mem := v_2
		if !(y.Uses == 1) {
			break
		}
		v.reset(OpAMD64SETAstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr y:(SETAE x) mem)
	// cond: y.Uses == 1
	// result: (SETAEstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64SETAE {
			break
		}
		x := y.Args[0]
		mem := v_2
		if !(y.Uses == 1) {
			break
		}
		v.reset(OpAMD64SETAEstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBQSX x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVBQSX {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBQZX x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVBQZX {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off1] {sym} (ADDQconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVBstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVLconst [c]) mem)
	// result: (MOVBstoreconst [makeValAndOff(int32(int8(c)),off)] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64MOVBstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(int8(c)), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVQconst [c]) mem)
	// result: (MOVBstoreconst [makeValAndOff(int32(int8(c)),off)] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64MOVBstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(int8(c)), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVBstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBstoreconst(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBstoreconst [sc] {s} (ADDQconst [off] ptr) mem)
	// cond: ValAndOff(sc).canAdd32(off)
	// result: (MOVBstoreconst [ValAndOff(sc).addOffset32(off)] {s} ptr mem)
	for {
		sc := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(sc).canAdd32(off)) {
			break
		}
		v.reset(OpAMD64MOVBstoreconst)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(sc).addOffset32(off))
		v.Aux = symToAux(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstoreconst [sc] {sym1} (LEAQ [off] {sym2} ptr) mem)
	// cond: canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd32(off)
	// result: (MOVBstoreconst [ValAndOff(sc).addOffset32(off)] {mergeSym(sym1, sym2)} ptr mem)
	for {
		sc := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd32(off)) {
			break
		}
		v.reset(OpAMD64MOVBstoreconst)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(sc).addOffset32(off))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLQSX(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVLQSX x:(MOVLload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVLQSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVLload {
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
		v0 := b.NewValue0(x.Pos, OpAMD64MOVLQSXload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLQSX x:(MOVQload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVLQSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVQload {
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
		v0 := b.NewValue0(x.Pos, OpAMD64MOVLQSXload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLQSX (ANDLconst [c] x))
	// cond: uint32(c) & 0x80000000 == 0
	// result: (ANDLconst [c & 0x7fffffff] x)
	for {
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(uint32(c)&0x80000000 == 0) {
			break
		}
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = int32ToAuxInt(c & 0x7fffffff)
		v.AddArg(x)
		return true
	}
	// match: (MOVLQSX (MOVLQSX x))
	// result: (MOVLQSX x)
	for {
		if v_0.Op != OpAMD64MOVLQSX {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64MOVLQSX)
		v.AddArg(x)
		return true
	}
	// match: (MOVLQSX (MOVWQSX x))
	// result: (MOVWQSX x)
	for {
		if v_0.Op != OpAMD64MOVWQSX {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64MOVWQSX)
		v.AddArg(x)
		return true
	}
	// match: (MOVLQSX (MOVBQSX x))
	// result: (MOVBQSX x)
	for {
		if v_0.Op != OpAMD64MOVBQSX {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64MOVBQSX)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLQSXload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVLQSXload [off] {sym} ptr (MOVLstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVLQSX x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLstore {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpAMD64MOVLQSX)
		v.AddArg(x)
		return true
	}
	// match: (MOVLQSXload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVLQSXload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVLQSXload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVLQSXload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVQconst [int64(int32(read32(sym, int64(off), config.ctxt.Arch.ByteOrder)))])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpSB || !(symIsRO(sym)) {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(int64(int32(read32(sym, int64(off), config.ctxt.Arch.ByteOrder))))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLQZX(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVLQZX x:(MOVLload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVLload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVLload {
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
		v0 := b.NewValue0(x.Pos, OpAMD64MOVLload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLQZX x:(MOVQload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVLload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVQload {
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
		v0 := b.NewValue0(x.Pos, OpAMD64MOVLload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLQZX (ANDLconst [c] x))
	// result: (ANDLconst [c] x)
	for {
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVLQZX (MOVLQZX x))
	// result: (MOVLQZX x)
	for {
		if v_0.Op != OpAMD64MOVLQZX {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64MOVLQZX)
		v.AddArg(x)
		return true
	}
	// match: (MOVLQZX (MOVWQZX x))
	// result: (MOVWQZX x)
	for {
		if v_0.Op != OpAMD64MOVWQZX {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64MOVWQZX)
		v.AddArg(x)
		return true
	}
	// match: (MOVLQZX (MOVBQZX x))
	// result: (MOVBQZX x)
	for {
		if v_0.Op != OpAMD64MOVBQZX {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64MOVBQZX)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLatomicload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVLatomicload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVLatomicload [off1+off2] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MOVLatomicload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLatomicload [off1] {sym1} (LEAQ [off2] {sym2} ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVLatomicload [off1+off2] {mergeSym(sym1, sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVLatomicload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLf2i(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVLf2i <t> (Arg <u> [off] {sym}))
	// cond: t.Size() == u.Size()
	// result: @b.Func.Entry (Arg <t> [off] {sym})
	for {
		t := v.Type
		if v_0.Op != OpArg {
			break
		}
		u := v_0.Type
		off := auxIntToInt32(v_0.AuxInt)
		sym := auxToSym(v_0.Aux)
		if !(t.Size() == u.Size()) {
			break
		}
		b = b.Func.Entry
		v0 := b.NewValue0(v.Pos, OpArg, t)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLi2f(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVLi2f <t> (Arg <u> [off] {sym}))
	// cond: t.Size() == u.Size()
	// result: @b.Func.Entry (Arg <t> [off] {sym})
	for {
		t := v.Type
		if v_0.Op != OpArg {
			break
		}
		u := v_0.Type
		off := auxIntToInt32(v_0.AuxInt)
		sym := auxToSym(v_0.Aux)
		if !(t.Size() == u.Size()) {
			break
		}
		b = b.Func.Entry
		v0 := b.NewValue0(v.Pos, OpArg, t)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVLload [off] {sym} ptr (MOVLstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVLQZX x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLstore {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpAMD64MOVLQZX)
		v.AddArg(x)
		return true
	}
	// match: (MOVLload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVLload [off1+off2] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MOVLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVLload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVLload [off] {sym} ptr (MOVSSstore [off] {sym} ptr val _))
	// result: (MOVLf2i val)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVSSstore || auxIntToInt32(v_1.AuxInt) != off || auxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.reset(OpAMD64MOVLf2i)
		v.AddArg(val)
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
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(int32(read32(sym, int64(off), config.ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVLstore [off] {sym} ptr (MOVLQSX x) mem)
	// result: (MOVLstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLQSX {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVLstore [off] {sym} ptr (MOVLQZX x) mem)
	// result: (MOVLstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLQZX {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVLstore [off1] {sym} (ADDQconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVLstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVLstore [off] {sym} ptr (MOVLconst [c]) mem)
	// result: (MOVLstoreconst [makeValAndOff(int32(c),off)] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(c), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLstore [off] {sym} ptr (MOVQconst [c]) mem)
	// result: (MOVLstoreconst [makeValAndOff(int32(c),off)] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(c), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVLstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64MOVLstore)
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
		if y.Op != OpAMD64ADDLload || auxIntToInt32(y.AuxInt) != off || auxToSym(y.Aux) != sym {
			break
		}
		mem := y.Args[2]
		x := y.Args[0]
		if ptr != y.Args[1] || mem != v_2 || !(y.Uses == 1 && clobber(y)) {
			break
		}
		v.reset(OpAMD64ADDLmodify)
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
		if y.Op != OpAMD64ANDLload || auxIntToInt32(y.AuxInt) != off || auxToSym(y.Aux) != sym {
			break
		}
		mem := y.Args[2]
		x := y.Args[0]
		if ptr != y.Args[1] || mem != v_2 || !(y.Uses == 1 && clobber(y)) {
			break
		}
		v.reset(OpAMD64ANDLmodify)
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
		if y.Op != OpAMD64ORLload || auxIntToInt32(y.AuxInt) != off || auxToSym(y.Aux) != sym {
			break
		}
		mem := y.Args[2]
		x := y.Args[0]
		if ptr != y.Args[1] || mem != v_2 || !(y.Uses == 1 && clobber(y)) {
			break
		}
		v.reset(OpAMD64ORLmodify)
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
		if y.Op != OpAMD64XORLload || auxIntToInt32(y.AuxInt) != off || auxToSym(y.Aux) != sym {
			break
		}
		mem := y.Args[2]
		x := y.Args[0]
		if ptr != y.Args[1] || mem != v_2 || !(y.Uses == 1 && clobber(y)) {
			break
		}
		v.reset(OpAMD64XORLmodify)
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
		if y.Op != OpAMD64ADDL {
			break
		}
		_ = y.Args[1]
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			l := y_0
			if l.Op != OpAMD64MOVLload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
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
			v.reset(OpAMD64ADDLmodify)
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
		if y.Op != OpAMD64SUBL {
			break
		}
		x := y.Args[1]
		l := y.Args[0]
		if l.Op != OpAMD64MOVLload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		if ptr != l.Args[0] || mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && clobber(y, l)) {
			break
		}
		v.reset(OpAMD64SUBLmodify)
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
		if y.Op != OpAMD64ANDL {
			break
		}
		_ = y.Args[1]
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			l := y_0
			if l.Op != OpAMD64MOVLload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
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
			v.reset(OpAMD64ANDLmodify)
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
		if y.Op != OpAMD64ORL {
			break
		}
		_ = y.Args[1]
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			l := y_0
			if l.Op != OpAMD64MOVLload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
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
			v.reset(OpAMD64ORLmodify)
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
		if y.Op != OpAMD64XORL {
			break
		}
		_ = y.Args[1]
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			l := y_0
			if l.Op != OpAMD64MOVLload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
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
			v.reset(OpAMD64XORLmodify)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(ptr, x, mem)
			return true
		}
		break
	}
	// match: (MOVLstore [off] {sym} ptr a:(ADDLconst [c] l:(MOVLload [off] {sym} ptr2 mem)) mem)
	// cond: isSamePtr(ptr, ptr2) && a.Uses == 1 && l.Uses == 1 && clobber(l, a)
	// result: (ADDLconstmodify {sym} [makeValAndOff(int32(c),off)] ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		a := v_1
		if a.Op != OpAMD64ADDLconst {
			break
		}
		c := auxIntToInt32(a.AuxInt)
		l := a.Args[0]
		if l.Op != OpAMD64MOVLload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		ptr2 := l.Args[0]
		if mem != v_2 || !(isSamePtr(ptr, ptr2) && a.Uses == 1 && l.Uses == 1 && clobber(l, a)) {
			break
		}
		v.reset(OpAMD64ADDLconstmodify)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(c), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLstore [off] {sym} ptr a:(ANDLconst [c] l:(MOVLload [off] {sym} ptr2 mem)) mem)
	// cond: isSamePtr(ptr, ptr2) && a.Uses == 1 && l.Uses == 1 && clobber(l, a)
	// result: (ANDLconstmodify {sym} [makeValAndOff(int32(c),off)] ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		a := v_1
		if a.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(a.AuxInt)
		l := a.Args[0]
		if l.Op != OpAMD64MOVLload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		ptr2 := l.Args[0]
		if mem != v_2 || !(isSamePtr(ptr, ptr2) && a.Uses == 1 && l.Uses == 1 && clobber(l, a)) {
			break
		}
		v.reset(OpAMD64ANDLconstmodify)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(c), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLstore [off] {sym} ptr a:(ORLconst [c] l:(MOVLload [off] {sym} ptr2 mem)) mem)
	// cond: isSamePtr(ptr, ptr2) && a.Uses == 1 && l.Uses == 1 && clobber(l, a)
	// result: (ORLconstmodify {sym} [makeValAndOff(int32(c),off)] ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		a := v_1
		if a.Op != OpAMD64ORLconst {
			break
		}
		c := auxIntToInt32(a.AuxInt)
		l := a.Args[0]
		if l.Op != OpAMD64MOVLload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		ptr2 := l.Args[0]
		if mem != v_2 || !(isSamePtr(ptr, ptr2) && a.Uses == 1 && l.Uses == 1 && clobber(l, a)) {
			break
		}
		v.reset(OpAMD64ORLconstmodify)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(c), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLstore [off] {sym} ptr a:(XORLconst [c] l:(MOVLload [off] {sym} ptr2 mem)) mem)
	// cond: isSamePtr(ptr, ptr2) && a.Uses == 1 && l.Uses == 1 && clobber(l, a)
	// result: (XORLconstmodify {sym} [makeValAndOff(int32(c),off)] ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		a := v_1
		if a.Op != OpAMD64XORLconst {
			break
		}
		c := auxIntToInt32(a.AuxInt)
		l := a.Args[0]
		if l.Op != OpAMD64MOVLload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		ptr2 := l.Args[0]
		if mem != v_2 || !(isSamePtr(ptr, ptr2) && a.Uses == 1 && l.Uses == 1 && clobber(l, a)) {
			break
		}
		v.reset(OpAMD64XORLconstmodify)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(c), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLstore [off] {sym} ptr (MOVLf2i val) mem)
	// result: (MOVSSstore [off] {sym} ptr val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLf2i {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64MOVSSstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVLstore [i] {s} p x:(BSWAPL w) mem)
	// cond: x.Uses == 1 && buildcfg.GOAMD64 >= 3
	// result: (MOVBELstore [i] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		x := v_1
		if x.Op != OpAMD64BSWAPL {
			break
		}
		w := x.Args[0]
		mem := v_2
		if !(x.Uses == 1 && buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64MOVBELstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p, w, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLstoreconst(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVLstoreconst [sc] {s} (ADDQconst [off] ptr) mem)
	// cond: ValAndOff(sc).canAdd32(off)
	// result: (MOVLstoreconst [ValAndOff(sc).addOffset32(off)] {s} ptr mem)
	for {
		sc := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(sc).canAdd32(off)) {
			break
		}
		v.reset(OpAMD64MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(sc).addOffset32(off))
		v.Aux = symToAux(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVLstoreconst [sc] {sym1} (LEAQ [off] {sym2} ptr) mem)
	// cond: canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd32(off)
	// result: (MOVLstoreconst [ValAndOff(sc).addOffset32(off)] {mergeSym(sym1, sym2)} ptr mem)
	for {
		sc := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd32(off)) {
			break
		}
		v.reset(OpAMD64MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(sc).addOffset32(off))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVOload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVOload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVOload [off1+off2] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MOVOload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVOload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVOload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVOload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVOstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVOstore [off1] {sym} (ADDQconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVOstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MOVOstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVOstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVOstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64MOVOstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVOstore [dstOff] {dstSym} ptr (MOVOload [srcOff] {srcSym} (SB) _) mem)
	// cond: symIsRO(srcSym)
	// result: (MOVQstore [dstOff+8] {dstSym} ptr (MOVQconst [int64(read64(srcSym, int64(srcOff)+8, config.ctxt.Arch.ByteOrder))]) (MOVQstore [dstOff] {dstSym} ptr (MOVQconst [int64(read64(srcSym, int64(srcOff), config.ctxt.Arch.ByteOrder))]) mem))
	for {
		dstOff := auxIntToInt32(v.AuxInt)
		dstSym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVOload {
			break
		}
		srcOff := auxIntToInt32(v_1.AuxInt)
		srcSym := auxToSym(v_1.Aux)
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpSB {
			break
		}
		mem := v_2
		if !(symIsRO(srcSym)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(dstOff + 8)
		v.Aux = symToAux(dstSym)
		v0 := b.NewValue0(v_1.Pos, OpAMD64MOVQconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(int64(read64(srcSym, int64(srcOff)+8, config.ctxt.Arch.ByteOrder)))
		v1 := b.NewValue0(v_1.Pos, OpAMD64MOVQstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(dstOff)
		v1.Aux = symToAux(dstSym)
		v2 := b.NewValue0(v_1.Pos, OpAMD64MOVQconst, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(int64(read64(srcSym, int64(srcOff), config.ctxt.Arch.ByteOrder)))
		v1.AddArg3(ptr, v2, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVOstoreconst(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVOstoreconst [sc] {s} (ADDQconst [off] ptr) mem)
	// cond: ValAndOff(sc).canAdd32(off)
	// result: (MOVOstoreconst [ValAndOff(sc).addOffset32(off)] {s} ptr mem)
	for {
		sc := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(sc).canAdd32(off)) {
			break
		}
		v.reset(OpAMD64MOVOstoreconst)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(sc).addOffset32(off))
		v.Aux = symToAux(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVOstoreconst [sc] {sym1} (LEAQ [off] {sym2} ptr) mem)
	// cond: canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd32(off)
	// result: (MOVOstoreconst [ValAndOff(sc).addOffset32(off)] {mergeSym(sym1, sym2)} ptr mem)
	for {
		sc := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd32(off)) {
			break
		}
		v.reset(OpAMD64MOVOstoreconst)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(sc).addOffset32(off))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVQatomicload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVQatomicload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVQatomicload [off1+off2] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MOVQatomicload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVQatomicload [off1] {sym1} (LEAQ [off2] {sym2} ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVQatomicload [off1+off2] {mergeSym(sym1, sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVQatomicload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVQf2i(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVQf2i <t> (Arg <u> [off] {sym}))
	// cond: t.Size() == u.Size()
	// result: @b.Func.Entry (Arg <t> [off] {sym})
	for {
		t := v.Type
		if v_0.Op != OpArg {
			break
		}
		u := v_0.Type
		off := auxIntToInt32(v_0.AuxInt)
		sym := auxToSym(v_0.Aux)
		if !(t.Size() == u.Size()) {
			break
		}
		b = b.Func.Entry
		v0 := b.NewValue0(v.Pos, OpArg, t)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVQi2f(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVQi2f <t> (Arg <u> [off] {sym}))
	// cond: t.Size() == u.Size()
	// result: @b.Func.Entry (Arg <t> [off] {sym})
	for {
		t := v.Type
		if v_0.Op != OpArg {
			break
		}
		u := v_0.Type
		off := auxIntToInt32(v_0.AuxInt)
		sym := auxToSym(v_0.Aux)
		if !(t.Size() == u.Size()) {
			break
		}
		b = b.Func.Entry
		v0 := b.NewValue0(v.Pos, OpArg, t)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVQload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVQload [off] {sym} ptr (MOVQstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: x
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVQstore {
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
	// match: (MOVQload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVQload [off1+off2] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MOVQload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVQload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVQload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVQload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVQload [off] {sym} ptr (MOVSDstore [off] {sym} ptr val _))
	// result: (MOVQf2i val)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVSDstore || auxIntToInt32(v_1.AuxInt) != off || auxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.reset(OpAMD64MOVQf2i)
		v.AddArg(val)
		return true
	}
	// match: (MOVQload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVQconst [int64(read64(sym, int64(off), config.ctxt.Arch.ByteOrder))])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpSB || !(symIsRO(sym)) {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(int64(read64(sym, int64(off), config.ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVQstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVQstore [off1] {sym} (ADDQconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVQstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVQstore [off] {sym} ptr (MOVQconst [c]) mem)
	// cond: validVal(c)
	// result: (MOVQstoreconst [makeValAndOff(int32(c),off)] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(validVal(c)) {
			break
		}
		v.reset(OpAMD64MOVQstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(c), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVQstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVQstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVQstore {sym} [off] ptr y:(ADDQload x [off] {sym} ptr mem) mem)
	// cond: y.Uses==1 && clobber(y)
	// result: (ADDQmodify [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64ADDQload || auxIntToInt32(y.AuxInt) != off || auxToSym(y.Aux) != sym {
			break
		}
		mem := y.Args[2]
		x := y.Args[0]
		if ptr != y.Args[1] || mem != v_2 || !(y.Uses == 1 && clobber(y)) {
			break
		}
		v.reset(OpAMD64ADDQmodify)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVQstore {sym} [off] ptr y:(ANDQload x [off] {sym} ptr mem) mem)
	// cond: y.Uses==1 && clobber(y)
	// result: (ANDQmodify [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64ANDQload || auxIntToInt32(y.AuxInt) != off || auxToSym(y.Aux) != sym {
			break
		}
		mem := y.Args[2]
		x := y.Args[0]
		if ptr != y.Args[1] || mem != v_2 || !(y.Uses == 1 && clobber(y)) {
			break
		}
		v.reset(OpAMD64ANDQmodify)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVQstore {sym} [off] ptr y:(ORQload x [off] {sym} ptr mem) mem)
	// cond: y.Uses==1 && clobber(y)
	// result: (ORQmodify [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64ORQload || auxIntToInt32(y.AuxInt) != off || auxToSym(y.Aux) != sym {
			break
		}
		mem := y.Args[2]
		x := y.Args[0]
		if ptr != y.Args[1] || mem != v_2 || !(y.Uses == 1 && clobber(y)) {
			break
		}
		v.reset(OpAMD64ORQmodify)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVQstore {sym} [off] ptr y:(XORQload x [off] {sym} ptr mem) mem)
	// cond: y.Uses==1 && clobber(y)
	// result: (XORQmodify [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64XORQload || auxIntToInt32(y.AuxInt) != off || auxToSym(y.Aux) != sym {
			break
		}
		mem := y.Args[2]
		x := y.Args[0]
		if ptr != y.Args[1] || mem != v_2 || !(y.Uses == 1 && clobber(y)) {
			break
		}
		v.reset(OpAMD64XORQmodify)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVQstore {sym} [off] ptr y:(ADDQ l:(MOVQload [off] {sym} ptr mem) x) mem)
	// cond: y.Uses==1 && l.Uses==1 && clobber(y, l)
	// result: (ADDQmodify [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64ADDQ {
			break
		}
		_ = y.Args[1]
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			l := y_0
			if l.Op != OpAMD64MOVQload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
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
			v.reset(OpAMD64ADDQmodify)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(ptr, x, mem)
			return true
		}
		break
	}
	// match: (MOVQstore {sym} [off] ptr y:(SUBQ l:(MOVQload [off] {sym} ptr mem) x) mem)
	// cond: y.Uses==1 && l.Uses==1 && clobber(y, l)
	// result: (SUBQmodify [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64SUBQ {
			break
		}
		x := y.Args[1]
		l := y.Args[0]
		if l.Op != OpAMD64MOVQload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		if ptr != l.Args[0] || mem != v_2 || !(y.Uses == 1 && l.Uses == 1 && clobber(y, l)) {
			break
		}
		v.reset(OpAMD64SUBQmodify)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVQstore {sym} [off] ptr y:(ANDQ l:(MOVQload [off] {sym} ptr mem) x) mem)
	// cond: y.Uses==1 && l.Uses==1 && clobber(y, l)
	// result: (ANDQmodify [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64ANDQ {
			break
		}
		_ = y.Args[1]
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			l := y_0
			if l.Op != OpAMD64MOVQload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
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
			v.reset(OpAMD64ANDQmodify)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(ptr, x, mem)
			return true
		}
		break
	}
	// match: (MOVQstore {sym} [off] ptr y:(ORQ l:(MOVQload [off] {sym} ptr mem) x) mem)
	// cond: y.Uses==1 && l.Uses==1 && clobber(y, l)
	// result: (ORQmodify [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64ORQ {
			break
		}
		_ = y.Args[1]
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			l := y_0
			if l.Op != OpAMD64MOVQload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
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
			v.reset(OpAMD64ORQmodify)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(ptr, x, mem)
			return true
		}
		break
	}
	// match: (MOVQstore {sym} [off] ptr y:(XORQ l:(MOVQload [off] {sym} ptr mem) x) mem)
	// cond: y.Uses==1 && l.Uses==1 && clobber(y, l)
	// result: (XORQmodify [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		y := v_1
		if y.Op != OpAMD64XORQ {
			break
		}
		_ = y.Args[1]
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			l := y_0
			if l.Op != OpAMD64MOVQload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
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
			v.reset(OpAMD64XORQmodify)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(ptr, x, mem)
			return true
		}
		break
	}
	// match: (MOVQstore {sym} [off] ptr x:(BTSQconst [c] l:(MOVQload {sym} [off] ptr mem)) mem)
	// cond: x.Uses == 1 && l.Uses == 1 && clobber(x, l)
	// result: (BTSQconstmodify {sym} [makeValAndOff(int32(c),off)] ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		x := v_1
		if x.Op != OpAMD64BTSQconst {
			break
		}
		c := auxIntToInt8(x.AuxInt)
		l := x.Args[0]
		if l.Op != OpAMD64MOVQload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		if ptr != l.Args[0] || mem != v_2 || !(x.Uses == 1 && l.Uses == 1 && clobber(x, l)) {
			break
		}
		v.reset(OpAMD64BTSQconstmodify)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(c), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVQstore {sym} [off] ptr x:(BTRQconst [c] l:(MOVQload {sym} [off] ptr mem)) mem)
	// cond: x.Uses == 1 && l.Uses == 1 && clobber(x, l)
	// result: (BTRQconstmodify {sym} [makeValAndOff(int32(c),off)] ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		x := v_1
		if x.Op != OpAMD64BTRQconst {
			break
		}
		c := auxIntToInt8(x.AuxInt)
		l := x.Args[0]
		if l.Op != OpAMD64MOVQload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		if ptr != l.Args[0] || mem != v_2 || !(x.Uses == 1 && l.Uses == 1 && clobber(x, l)) {
			break
		}
		v.reset(OpAMD64BTRQconstmodify)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(c), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVQstore {sym} [off] ptr x:(BTCQconst [c] l:(MOVQload {sym} [off] ptr mem)) mem)
	// cond: x.Uses == 1 && l.Uses == 1 && clobber(x, l)
	// result: (BTCQconstmodify {sym} [makeValAndOff(int32(c),off)] ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		x := v_1
		if x.Op != OpAMD64BTCQconst {
			break
		}
		c := auxIntToInt8(x.AuxInt)
		l := x.Args[0]
		if l.Op != OpAMD64MOVQload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		if ptr != l.Args[0] || mem != v_2 || !(x.Uses == 1 && l.Uses == 1 && clobber(x, l)) {
			break
		}
		v.reset(OpAMD64BTCQconstmodify)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(c), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVQstore [off] {sym} ptr a:(ADDQconst [c] l:(MOVQload [off] {sym} ptr2 mem)) mem)
	// cond: isSamePtr(ptr, ptr2) && a.Uses == 1 && l.Uses == 1 && clobber(l, a)
	// result: (ADDQconstmodify {sym} [makeValAndOff(int32(c),off)] ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		a := v_1
		if a.Op != OpAMD64ADDQconst {
			break
		}
		c := auxIntToInt32(a.AuxInt)
		l := a.Args[0]
		if l.Op != OpAMD64MOVQload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		ptr2 := l.Args[0]
		if mem != v_2 || !(isSamePtr(ptr, ptr2) && a.Uses == 1 && l.Uses == 1 && clobber(l, a)) {
			break
		}
		v.reset(OpAMD64ADDQconstmodify)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(c), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVQstore [off] {sym} ptr a:(ANDQconst [c] l:(MOVQload [off] {sym} ptr2 mem)) mem)
	// cond: isSamePtr(ptr, ptr2) && a.Uses == 1 && l.Uses == 1 && clobber(l, a)
	// result: (ANDQconstmodify {sym} [makeValAndOff(int32(c),off)] ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		a := v_1
		if a.Op != OpAMD64ANDQconst {
			break
		}
		c := auxIntToInt32(a.AuxInt)
		l := a.Args[0]
		if l.Op != OpAMD64MOVQload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		ptr2 := l.Args[0]
		if mem != v_2 || !(isSamePtr(ptr, ptr2) && a.Uses == 1 && l.Uses == 1 && clobber(l, a)) {
			break
		}
		v.reset(OpAMD64ANDQconstmodify)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(c), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVQstore [off] {sym} ptr a:(ORQconst [c] l:(MOVQload [off] {sym} ptr2 mem)) mem)
	// cond: isSamePtr(ptr, ptr2) && a.Uses == 1 && l.Uses == 1 && clobber(l, a)
	// result: (ORQconstmodify {sym} [makeValAndOff(int32(c),off)] ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		a := v_1
		if a.Op != OpAMD64ORQconst {
			break
		}
		c := auxIntToInt32(a.AuxInt)
		l := a.Args[0]
		if l.Op != OpAMD64MOVQload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		ptr2 := l.Args[0]
		if mem != v_2 || !(isSamePtr(ptr, ptr2) && a.Uses == 1 && l.Uses == 1 && clobber(l, a)) {
			break
		}
		v.reset(OpAMD64ORQconstmodify)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(c), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVQstore [off] {sym} ptr a:(XORQconst [c] l:(MOVQload [off] {sym} ptr2 mem)) mem)
	// cond: isSamePtr(ptr, ptr2) && a.Uses == 1 && l.Uses == 1 && clobber(l, a)
	// result: (XORQconstmodify {sym} [makeValAndOff(int32(c),off)] ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		a := v_1
		if a.Op != OpAMD64XORQconst {
			break
		}
		c := auxIntToInt32(a.AuxInt)
		l := a.Args[0]
		if l.Op != OpAMD64MOVQload || auxIntToInt32(l.AuxInt) != off || auxToSym(l.Aux) != sym {
			break
		}
		mem := l.Args[1]
		ptr2 := l.Args[0]
		if mem != v_2 || !(isSamePtr(ptr, ptr2) && a.Uses == 1 && l.Uses == 1 && clobber(l, a)) {
			break
		}
		v.reset(OpAMD64XORQconstmodify)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(c), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVQstore [off] {sym} ptr (MOVQf2i val) mem)
	// result: (MOVSDstore [off] {sym} ptr val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVQf2i {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64MOVSDstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVQstore [i] {s} p x:(BSWAPQ w) mem)
	// cond: x.Uses == 1 && buildcfg.GOAMD64 >= 3
	// result: (MOVBEQstore [i] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		x := v_1
		if x.Op != OpAMD64BSWAPQ {
			break
		}
		w := x.Args[0]
		mem := v_2
		if !(x.Uses == 1 && buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64MOVBEQstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p, w, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVQstoreconst(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVQstoreconst [sc] {s} (ADDQconst [off] ptr) mem)
	// cond: ValAndOff(sc).canAdd32(off)
	// result: (MOVQstoreconst [ValAndOff(sc).addOffset32(off)] {s} ptr mem)
	for {
		sc := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(sc).canAdd32(off)) {
			break
		}
		v.reset(OpAMD64MOVQstoreconst)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(sc).addOffset32(off))
		v.Aux = symToAux(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVQstoreconst [sc] {sym1} (LEAQ [off] {sym2} ptr) mem)
	// cond: canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd32(off)
	// result: (MOVQstoreconst [ValAndOff(sc).addOffset32(off)] {mergeSym(sym1, sym2)} ptr mem)
	for {
		sc := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd32(off)) {
			break
		}
		v.reset(OpAMD64MOVQstoreconst)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(sc).addOffset32(off))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVQstoreconst [c] {s} p1 x:(MOVQstoreconst [a] {s} p0 mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+8-c.Off())) && a.Val() == 0 && c.Val() == 0 && setPos(v, x.Pos) && clobber(x)
	// result: (MOVOstoreconst [makeValAndOff(0,a.Off())] {s} p0 mem)
	for {
		c := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		p1 := v_0
		x := v_1
		if x.Op != OpAMD64MOVQstoreconst {
			break
		}
		a := auxIntToValAndOff(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		p0 := x.Args[0]
		if !(x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+8-c.Off())) && a.Val() == 0 && c.Val() == 0 && setPos(v, x.Pos) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVOstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, a.Off()))
		v.Aux = symToAux(s)
		v.AddArg2(p0, mem)
		return true
	}
	// match: (MOVQstoreconst [a] {s} p0 x:(MOVQstoreconst [c] {s} p1 mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+8-c.Off())) && a.Val() == 0 && c.Val() == 0 && setPos(v, x.Pos) && clobber(x)
	// result: (MOVOstoreconst [makeValAndOff(0,a.Off())] {s} p0 mem)
	for {
		a := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		p0 := v_0
		x := v_1
		if x.Op != OpAMD64MOVQstoreconst {
			break
		}
		c := auxIntToValAndOff(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		p1 := x.Args[0]
		if !(x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+8-c.Off())) && a.Val() == 0 && c.Val() == 0 && setPos(v, x.Pos) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVOstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, a.Off()))
		v.Aux = symToAux(s)
		v.AddArg2(p0, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVSDload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVSDload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVSDload [off1+off2] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MOVSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVSDload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVSDload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVSDload [off] {sym} ptr (MOVQstore [off] {sym} ptr val _))
	// result: (MOVQi2f val)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVQstore || auxIntToInt32(v_1.AuxInt) != off || auxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.reset(OpAMD64MOVQi2f)
		v.AddArg(val)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVSDstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVSDstore [off1] {sym} (ADDQconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVSDstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MOVSDstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVSDstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVSDstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64MOVSDstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVSDstore [off] {sym} ptr (MOVQi2f val) mem)
	// result: (MOVQstore [off] {sym} ptr val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVQi2f {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVSDstore [off] {sym} ptr (MOVSDconst [f]) mem)
	// cond: f == f
	// result: (MOVQstore [off] {sym} ptr (MOVQconst [int64(math.Float64bits(f))]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVSDconst {
			break
		}
		f := auxIntToFloat64(v_1.AuxInt)
		mem := v_2
		if !(f == f) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(int64(math.Float64bits(f)))
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVSSload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVSSload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVSSload [off1+off2] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MOVSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVSSload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVSSload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVSSload [off] {sym} ptr (MOVLstore [off] {sym} ptr val _))
	// result: (MOVLi2f val)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLstore || auxIntToInt32(v_1.AuxInt) != off || auxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.reset(OpAMD64MOVLi2f)
		v.AddArg(val)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVSSstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVSSstore [off1] {sym} (ADDQconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVSSstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MOVSSstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVSSstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVSSstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64MOVSSstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVSSstore [off] {sym} ptr (MOVLi2f val) mem)
	// result: (MOVLstore [off] {sym} ptr val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLi2f {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVSSstore [off] {sym} ptr (MOVSSconst [f]) mem)
	// cond: f == f
	// result: (MOVLstore [off] {sym} ptr (MOVLconst [int32(math.Float32bits(f))]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVSSconst {
			break
		}
		f := auxIntToFloat32(v_1.AuxInt)
		mem := v_2
		if !(f == f) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(int32(math.Float32bits(f)))
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWQSX(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVWQSX x:(MOVWload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVWQSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVWload {
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
		v0 := b.NewValue0(x.Pos, OpAMD64MOVWQSXload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWQSX x:(MOVLload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVWQSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVLload {
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
		v0 := b.NewValue0(x.Pos, OpAMD64MOVWQSXload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWQSX x:(MOVQload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVWQSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVQload {
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
		v0 := b.NewValue0(x.Pos, OpAMD64MOVWQSXload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWQSX (ANDLconst [c] x))
	// cond: c & 0x8000 == 0
	// result: (ANDLconst [c & 0x7fff] x)
	for {
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c&0x8000 == 0) {
			break
		}
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = int32ToAuxInt(c & 0x7fff)
		v.AddArg(x)
		return true
	}
	// match: (MOVWQSX (MOVWQSX x))
	// result: (MOVWQSX x)
	for {
		if v_0.Op != OpAMD64MOVWQSX {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64MOVWQSX)
		v.AddArg(x)
		return true
	}
	// match: (MOVWQSX (MOVBQSX x))
	// result: (MOVBQSX x)
	for {
		if v_0.Op != OpAMD64MOVBQSX {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64MOVBQSX)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWQSXload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWQSXload [off] {sym} ptr (MOVWstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVWQSX x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVWstore {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpAMD64MOVWQSX)
		v.AddArg(x)
		return true
	}
	// match: (MOVWQSXload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVWQSXload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVWQSXload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	// match: (MOVWQSXload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVQconst [int64(int16(read16(sym, int64(off), config.ctxt.Arch.ByteOrder)))])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpSB || !(symIsRO(sym)) {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(int64(int16(read16(sym, int64(off), config.ctxt.Arch.ByteOrder))))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWQZX(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVWQZX x:(MOVWload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVWload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVWload {
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
		v0 := b.NewValue0(x.Pos, OpAMD64MOVWload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWQZX x:(MOVLload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVWload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVLload {
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
		v0 := b.NewValue0(x.Pos, OpAMD64MOVWload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWQZX x:(MOVQload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVWload <v.Type> [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpAMD64MOVQload {
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
		v0 := b.NewValue0(x.Pos, OpAMD64MOVWload, v.Type)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWQZX (ANDLconst [c] x))
	// result: (ANDLconst [c & 0xffff] x)
	for {
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = int32ToAuxInt(c & 0xffff)
		v.AddArg(x)
		return true
	}
	// match: (MOVWQZX (MOVWQZX x))
	// result: (MOVWQZX x)
	for {
		if v_0.Op != OpAMD64MOVWQZX {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64MOVWQZX)
		v.AddArg(x)
		return true
	}
	// match: (MOVWQZX (MOVBQZX x))
	// result: (MOVBQZX x)
	for {
		if v_0.Op != OpAMD64MOVBQZX {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64MOVBQZX)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWload [off] {sym} ptr (MOVWstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVWQZX x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVWstore {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpAMD64MOVWQZX)
		v.AddArg(x)
		return true
	}
	// match: (MOVWload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVWload [off1+off2] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MOVWload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVWload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVWload)
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
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(int32(read16(sym, int64(off), config.ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstore [off] {sym} ptr (MOVWQSX x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVWQSX {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWQZX x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVWQZX {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym} (ADDQconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MOVWstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVLconst [c]) mem)
	// result: (MOVWstoreconst [makeValAndOff(int32(int16(c)),off)] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64MOVWstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(int16(c)), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVQconst [c]) mem)
	// result: (MOVWstoreconst [makeValAndOff(int32(int16(c)),off)] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64MOVWstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(int32(int16(c)), off))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MOVWstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (MOVWstore [i] {s} p x:(ROLWconst [8] w) mem)
	// cond: x.Uses == 1 && buildcfg.GOAMD64 >= 3
	// result: (MOVBEWstore [i] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		x := v_1
		if x.Op != OpAMD64ROLWconst || auxIntToInt8(x.AuxInt) != 8 {
			break
		}
		w := x.Args[0]
		mem := v_2
		if !(x.Uses == 1 && buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64MOVBEWstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p, w, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWstoreconst(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstoreconst [sc] {s} (ADDQconst [off] ptr) mem)
	// cond: ValAndOff(sc).canAdd32(off)
	// result: (MOVWstoreconst [ValAndOff(sc).addOffset32(off)] {s} ptr mem)
	for {
		sc := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off := auxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(sc).canAdd32(off)) {
			break
		}
		v.reset(OpAMD64MOVWstoreconst)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(sc).addOffset32(off))
		v.Aux = symToAux(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstoreconst [sc] {sym1} (LEAQ [off] {sym2} ptr) mem)
	// cond: canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd32(off)
	// result: (MOVWstoreconst [ValAndOff(sc).addOffset32(off)] {mergeSym(sym1, sym2)} ptr mem)
	for {
		sc := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd32(off)) {
			break
		}
		v.reset(OpAMD64MOVWstoreconst)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(sc).addOffset32(off))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MULL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MULL x (MOVLconst [c]))
	// result: (MULLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64MOVLconst {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			v.reset(OpAMD64MULLconst)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64MULLconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MULLconst [c] (MULLconst [d] x))
	// result: (MULLconst [c * d] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MULLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64MULLconst)
		v.AuxInt = int32ToAuxInt(c * d)
		v.AddArg(x)
		return true
	}
	// match: (MULLconst [ 0] _)
	// result: (MOVLconst [0])
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (MULLconst [ 1] x)
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: v.Type.Size() <= 4 && canMulStrengthReduce32(config, c)
	// result: {mulStrengthReduce32(v, x, c)}
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(v.Type.Size() <= 4 && canMulStrengthReduce32(config, c)) {
			break
		}
		v.copyOf(mulStrengthReduce32(v, x, c))
		return true
	}
	// match: (MULLconst [c] (MOVLconst [d]))
	// result: (MOVLconst [c*d])
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(c * d)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MULQ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MULQ x (MOVQconst [c]))
	// cond: is32Bit(c)
	// result: (MULQconst [int32(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64MOVQconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(is32Bit(c)) {
				continue
			}
			v.reset(OpAMD64MULQconst)
			v.AuxInt = int32ToAuxInt(int32(c))
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64MULQconst(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MULQconst [c] (MULQconst [d] x))
	// cond: is32Bit(int64(c)*int64(d))
	// result: (MULQconst [c * d] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MULQconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(int64(c) * int64(d))) {
			break
		}
		v.reset(OpAMD64MULQconst)
		v.AuxInt = int32ToAuxInt(c * d)
		v.AddArg(x)
		return true
	}
	// match: (MULQconst [ 0] _)
	// result: (MOVQconst [0])
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (MULQconst [ 1] x)
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (MULQconst [c] x)
	// cond: canMulStrengthReduce(config, int64(c))
	// result: {mulStrengthReduce(v, x, int64(c))}
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(canMulStrengthReduce(config, int64(c))) {
			break
		}
		v.copyOf(mulStrengthReduce(v, x, int64(c)))
		return true
	}
	// match: (MULQconst [c] (MOVQconst [d]))
	// result: (MOVQconst [int64(c)*d])
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(int64(c) * d)
		return true
	}
	// match: (MULQconst [c] (NEGQ x))
	// cond: c != -(1<<31)
	// result: (MULQconst [-c] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64NEGQ {
			break
		}
		x := v_0.Args[0]
		if !(c != -(1 << 31)) {
			break
		}
		v.reset(OpAMD64MULQconst)
		v.AuxInt = int32ToAuxInt(-c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MULSD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MULSD x l:(MOVSDload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (MULSDload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != OpAMD64MOVSDload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(OpAMD64MULSDload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64MULSDload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MULSDload [off1] {sym} val (ADDQconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MULSDload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MULSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (MULSDload [off1] {sym1} val (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MULSDload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MULSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (MULSDload x [off] {sym} ptr (MOVQstore [off] {sym} ptr y _))
	// result: (MULSD x (MOVQi2f y))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		ptr := v_1
		if v_2.Op != OpAMD64MOVQstore || auxIntToInt32(v_2.AuxInt) != off || auxToSym(v_2.Aux) != sym {
			break
		}
		y := v_2.Args[1]
		if ptr != v_2.Args[0] {
			break
		}
		v.reset(OpAMD64MULSD)
		v0 := b.NewValue0(v_2.Pos, OpAMD64MOVQi2f, typ.Float64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MULSS(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MULSS x l:(MOVSSload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (MULSSload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != OpAMD64MOVSSload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(OpAMD64MULSSload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64MULSSload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MULSSload [off1] {sym} val (ADDQconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (MULSSload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64MULSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (MULSSload [off1] {sym1} val (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (MULSSload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MULSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (MULSSload x [off] {sym} ptr (MOVLstore [off] {sym} ptr y _))
	// result: (MULSS x (MOVLi2f y))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		ptr := v_1
		if v_2.Op != OpAMD64MOVLstore || auxIntToInt32(v_2.AuxInt) != off || auxToSym(v_2.Aux) != sym {
			break
		}
		y := v_2.Args[1]
		if ptr != v_2.Args[0] {
			break
		}
		v.reset(OpAMD64MULSS)
		v0 := b.NewValue0(v_2.Pos, OpAMD64MOVLi2f, typ.Float32)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64NEGL(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NEGL (NEGL x))
	// result: x
	for {
		if v_0.Op != OpAMD64NEGL {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (NEGL s:(SUBL x y))
	// cond: s.Uses == 1
	// result: (SUBL y x)
	for {
		s := v_0
		if s.Op != OpAMD64SUBL {
			break
		}
		y := s.Args[1]
		x := s.Args[0]
		if !(s.Uses == 1) {
			break
		}
		v.reset(OpAMD64SUBL)
		v.AddArg2(y, x)
		return true
	}
	// match: (NEGL (MOVLconst [c]))
	// result: (MOVLconst [-c])
	for {
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(-c)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64NEGQ(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NEGQ (NEGQ x))
	// result: x
	for {
		if v_0.Op != OpAMD64NEGQ {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (NEGQ s:(SUBQ x y))
	// cond: s.Uses == 1
	// result: (SUBQ y x)
	for {
		s := v_0
		if s.Op != OpAMD64SUBQ {
			break
		}
		y := s.Args[1]
		x := s.Args[0]
		if !(s.Uses == 1) {
			break
		}
		v.reset(OpAMD64SUBQ)
		v.AddArg2(y, x)
		return true
	}
	// match: (NEGQ (MOVQconst [c]))
	// result: (MOVQconst [-c])
	for {
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(-c)
		return true
	}
	// match: (NEGQ (ADDQconst [c] (NEGQ x)))
	// cond: c != -(1<<31)
	// result: (ADDQconst [-c] x)
	for {
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64NEGQ {
			break
		}
		x := v_0_0.Args[0]
		if !(c != -(1 << 31)) {
			break
		}
		v.reset(OpAMD64ADDQconst)
		v.AuxInt = int32ToAuxInt(-c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64NOTL(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NOTL (MOVLconst [c]))
	// result: (MOVLconst [^c])
	for {
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(^c)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64NOTQ(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NOTQ (MOVQconst [c]))
	// result: (MOVQconst [^c])
	for {
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(^c)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORL (SHLL (MOVLconst [1]) y) x)
	// result: (BTSL x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64SHLL {
				continue
			}
			y := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64MOVLconst || auxIntToInt32(v_0_0.AuxInt) != 1 {
				continue
			}
			x := v_1
			v.reset(OpAMD64BTSL)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ORL x (MOVLconst [c]))
	// result: (ORLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64MOVLconst {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			v.reset(OpAMD64ORLconst)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg(x)
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
	// match: (ORL x l:(MOVLload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (ORLload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != OpAMD64MOVLload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(OpAMD64ORLload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ORLconst [c] (ORLconst [d] x))
	// result: (ORLconst [c | d] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64ORLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64ORLconst)
		v.AuxInt = int32ToAuxInt(c | d)
		v.AddArg(x)
		return true
	}
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
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(-1)
		return true
	}
	// match: (ORLconst [c] (MOVLconst [d]))
	// result: (MOVLconst [c|d])
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(c | d)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORLconstmodify(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORLconstmodify [valoff1] {sym} (ADDQconst [off2] base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2)
	// result: (ORLconstmodify [ValAndOff(valoff1).addOffset32(off2)] {sym} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2)) {
			break
		}
		v.reset(OpAMD64ORLconstmodify)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (ORLconstmodify [valoff1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)
	// result: (ORLconstmodify [ValAndOff(valoff1).addOffset32(off2)] {mergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64ORLconstmodify)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORLload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ORLload [off1] {sym} val (ADDQconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ORLload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64ORLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ORLload [off1] {sym1} val (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (ORLload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64ORLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	// match: ( ORLload x [off] {sym} ptr (MOVSSstore [off] {sym} ptr y _))
	// result: ( ORL x (MOVLf2i y))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		ptr := v_1
		if v_2.Op != OpAMD64MOVSSstore || auxIntToInt32(v_2.AuxInt) != off || auxToSym(v_2.Aux) != sym {
			break
		}
		y := v_2.Args[1]
		if ptr != v_2.Args[0] {
			break
		}
		v.reset(OpAMD64ORL)
		v0 := b.NewValue0(v_2.Pos, OpAMD64MOVLf2i, typ.UInt32)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORLmodify(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORLmodify [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ORLmodify [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64ORLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (ORLmodify [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (ORLmodify [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64ORLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORQ (SHLQ (MOVQconst [1]) y) x)
	// result: (BTSQ x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64SHLQ {
				continue
			}
			y := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64MOVQconst || auxIntToInt64(v_0_0.AuxInt) != 1 {
				continue
			}
			x := v_1
			v.reset(OpAMD64BTSQ)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ORQ (MOVQconst [c]) x)
	// cond: isUnsignedPowerOfTwo(uint64(c)) && uint64(c) >= 1<<31
	// result: (BTSQconst [int8(log64u(uint64(c)))] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64MOVQconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			x := v_1
			if !(isUnsignedPowerOfTwo(uint64(c)) && uint64(c) >= 1<<31) {
				continue
			}
			v.reset(OpAMD64BTSQconst)
			v.AuxInt = int8ToAuxInt(int8(log64u(uint64(c))))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ORQ x (MOVQconst [c]))
	// cond: is32Bit(c)
	// result: (ORQconst [int32(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64MOVQconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(is32Bit(c)) {
				continue
			}
			v.reset(OpAMD64ORQconst)
			v.AuxInt = int32ToAuxInt(int32(c))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ORQ x (MOVLconst [c]))
	// result: (ORQconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64MOVLconst {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			v.reset(OpAMD64ORQconst)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ORQ (SHRQ lo bits) (SHLQ hi (NEGQ bits)))
	// result: (SHRDQ lo hi bits)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64SHRQ {
				continue
			}
			bits := v_0.Args[1]
			lo := v_0.Args[0]
			if v_1.Op != OpAMD64SHLQ {
				continue
			}
			_ = v_1.Args[1]
			hi := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpAMD64NEGQ || bits != v_1_1.Args[0] {
				continue
			}
			v.reset(OpAMD64SHRDQ)
			v.AddArg3(lo, hi, bits)
			return true
		}
		break
	}
	// match: (ORQ (SHLQ lo bits) (SHRQ hi (NEGQ bits)))
	// result: (SHLDQ lo hi bits)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64SHLQ {
				continue
			}
			bits := v_0.Args[1]
			lo := v_0.Args[0]
			if v_1.Op != OpAMD64SHRQ {
				continue
			}
			_ = v_1.Args[1]
			hi := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpAMD64NEGQ || bits != v_1_1.Args[0] {
				continue
			}
			v.reset(OpAMD64SHLDQ)
			v.AddArg3(lo, hi, bits)
			return true
		}
		break
	}
	// match: (ORQ (SHRXQ lo bits) (SHLXQ hi (NEGQ bits)))
	// result: (SHRDQ lo hi bits)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64SHRXQ {
				continue
			}
			bits := v_0.Args[1]
			lo := v_0.Args[0]
			if v_1.Op != OpAMD64SHLXQ {
				continue
			}
			_ = v_1.Args[1]
			hi := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpAMD64NEGQ || bits != v_1_1.Args[0] {
				continue
			}
			v.reset(OpAMD64SHRDQ)
			v.AddArg3(lo, hi, bits)
			return true
		}
		break
	}
	// match: (ORQ (SHLXQ lo bits) (SHRXQ hi (NEGQ bits)))
	// result: (SHLDQ lo hi bits)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64SHLXQ {
				continue
			}
			bits := v_0.Args[1]
			lo := v_0.Args[0]
			if v_1.Op != OpAMD64SHRXQ {
				continue
			}
			_ = v_1.Args[1]
			hi := v_1.Args[0]
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpAMD64NEGQ || bits != v_1_1.Args[0] {
				continue
			}
			v.reset(OpAMD64SHLDQ)
			v.AddArg3(lo, hi, bits)
			return true
		}
		break
	}
	// match: (ORQ (MOVQconst [c]) (MOVQconst [d]))
	// result: (MOVQconst [c|d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64MOVQconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpAMD64MOVQconst {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpAMD64MOVQconst)
			v.AuxInt = int64ToAuxInt(c | d)
			return true
		}
		break
	}
	// match: (ORQ x x)
	// result: x
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (ORQ x l:(MOVQload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (ORQload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != OpAMD64MOVQload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(OpAMD64ORQload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ORQconst [c] (ORQconst [d] x))
	// result: (ORQconst [c | d] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64ORQconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64ORQconst)
		v.AuxInt = int32ToAuxInt(c | d)
		v.AddArg(x)
		return true
	}
	// match: (ORQconst [0] x)
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (ORQconst [-1] _)
	// result: (MOVQconst [-1])
	for {
		if auxIntToInt32(v.AuxInt) != -1 {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	// match: (ORQconst [c] (MOVQconst [d]))
	// result: (MOVQconst [int64(c)|d])
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(int64(c) | d)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQconstmodify(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORQconstmodify [valoff1] {sym} (ADDQconst [off2] base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2)
	// result: (ORQconstmodify [ValAndOff(valoff1).addOffset32(off2)] {sym} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2)) {
			break
		}
		v.reset(OpAMD64ORQconstmodify)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (ORQconstmodify [valoff1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)
	// result: (ORQconstmodify [ValAndOff(valoff1).addOffset32(off2)] {mergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64ORQconstmodify)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ORQload [off1] {sym} val (ADDQconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ORQload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64ORQload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (ORQload [off1] {sym1} val (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (ORQload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64ORQload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	// match: ( ORQload x [off] {sym} ptr (MOVSDstore [off] {sym} ptr y _))
	// result: ( ORQ x (MOVQf2i y))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		ptr := v_1
		if v_2.Op != OpAMD64MOVSDstore || auxIntToInt32(v_2.AuxInt) != off || auxToSym(v_2.Aux) != sym {
			break
		}
		y := v_2.Args[1]
		if ptr != v_2.Args[0] {
			break
		}
		v.reset(OpAMD64ORQ)
		v0 := b.NewValue0(v_2.Pos, OpAMD64MOVQf2i, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQmodify(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORQmodify [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (ORQmodify [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64ORQmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (ORQmodify [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (ORQmodify [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64ORQmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ROLB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROLB x (NEGQ y))
	// result: (RORB x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64RORB)
		v.AddArg2(x, y)
		return true
	}
	// match: (ROLB x (NEGL y))
	// result: (RORB x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64RORB)
		v.AddArg2(x, y)
		return true
	}
	// match: (ROLB x (MOVQconst [c]))
	// result: (ROLBconst [int8(c&7) ] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64ROLBconst)
		v.AuxInt = int8ToAuxInt(int8(c & 7))
		v.AddArg(x)
		return true
	}
	// match: (ROLB x (MOVLconst [c]))
	// result: (ROLBconst [int8(c&7) ] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64ROLBconst)
		v.AuxInt = int8ToAuxInt(int8(c & 7))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ROLBconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ROLBconst x [0])
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
func rewriteValueAMD64_OpAMD64ROLL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROLL x (NEGQ y))
	// result: (RORL x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64RORL)
		v.AddArg2(x, y)
		return true
	}
	// match: (ROLL x (NEGL y))
	// result: (RORL x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64RORL)
		v.AddArg2(x, y)
		return true
	}
	// match: (ROLL x (MOVQconst [c]))
	// result: (ROLLconst [int8(c&31)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64ROLLconst)
		v.AuxInt = int8ToAuxInt(int8(c & 31))
		v.AddArg(x)
		return true
	}
	// match: (ROLL x (MOVLconst [c]))
	// result: (ROLLconst [int8(c&31)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64ROLLconst)
		v.AuxInt = int8ToAuxInt(int8(c & 31))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ROLLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ROLLconst x [0])
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
func rewriteValueAMD64_OpAMD64ROLQ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROLQ x (NEGQ y))
	// result: (RORQ x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64RORQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (ROLQ x (NEGL y))
	// result: (RORQ x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64RORQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (ROLQ x (MOVQconst [c]))
	// result: (ROLQconst [int8(c&63)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64ROLQconst)
		v.AuxInt = int8ToAuxInt(int8(c & 63))
		v.AddArg(x)
		return true
	}
	// match: (ROLQ x (MOVLconst [c]))
	// result: (ROLQconst [int8(c&63)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64ROLQconst)
		v.AuxInt = int8ToAuxInt(int8(c & 63))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ROLQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ROLQconst x [0])
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
func rewriteValueAMD64_OpAMD64ROLW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROLW x (NEGQ y))
	// result: (RORW x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64RORW)
		v.AddArg2(x, y)
		return true
	}
	// match: (ROLW x (NEGL y))
	// result: (RORW x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64RORW)
		v.AddArg2(x, y)
		return true
	}
	// match: (ROLW x (MOVQconst [c]))
	// result: (ROLWconst [int8(c&15)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64ROLWconst)
		v.AuxInt = int8ToAuxInt(int8(c & 15))
		v.AddArg(x)
		return true
	}
	// match: (ROLW x (MOVLconst [c]))
	// result: (ROLWconst [int8(c&15)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64ROLWconst)
		v.AuxInt = int8ToAuxInt(int8(c & 15))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ROLWconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ROLWconst x [0])
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
func rewriteValueAMD64_OpAMD64RORB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (RORB x (NEGQ y))
	// result: (ROLB x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64ROLB)
		v.AddArg2(x, y)
		return true
	}
	// match: (RORB x (NEGL y))
	// result: (ROLB x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64ROLB)
		v.AddArg2(x, y)
		return true
	}
	// match: (RORB x (MOVQconst [c]))
	// result: (ROLBconst [int8((-c)&7) ] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64ROLBconst)
		v.AuxInt = int8ToAuxInt(int8((-c) & 7))
		v.AddArg(x)
		return true
	}
	// match: (RORB x (MOVLconst [c]))
	// result: (ROLBconst [int8((-c)&7) ] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64ROLBconst)
		v.AuxInt = int8ToAuxInt(int8((-c) & 7))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64RORL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (RORL x (NEGQ y))
	// result: (ROLL x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64ROLL)
		v.AddArg2(x, y)
		return true
	}
	// match: (RORL x (NEGL y))
	// result: (ROLL x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64ROLL)
		v.AddArg2(x, y)
		return true
	}
	// match: (RORL x (MOVQconst [c]))
	// result: (ROLLconst [int8((-c)&31)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64ROLLconst)
		v.AuxInt = int8ToAuxInt(int8((-c) & 31))
		v.AddArg(x)
		return true
	}
	// match: (RORL x (MOVLconst [c]))
	// result: (ROLLconst [int8((-c)&31)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64ROLLconst)
		v.AuxInt = int8ToAuxInt(int8((-c) & 31))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64RORQ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (RORQ x (NEGQ y))
	// result: (ROLQ x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64ROLQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (RORQ x (NEGL y))
	// result: (ROLQ x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64ROLQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (RORQ x (MOVQconst [c]))
	// result: (ROLQconst [int8((-c)&63)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64ROLQconst)
		v.AuxInt = int8ToAuxInt(int8((-c) & 63))
		v.AddArg(x)
		return true
	}
	// match: (RORQ x (MOVLconst [c]))
	// result: (ROLQconst [int8((-c)&63)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64ROLQconst)
		v.AuxInt = int8ToAuxInt(int8((-c) & 63))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64RORW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (RORW x (NEGQ y))
	// result: (ROLW x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64ROLW)
		v.AddArg2(x, y)
		return true
	}
	// match: (RORW x (NEGL y))
	// result: (ROLW x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64ROLW)
		v.AddArg2(x, y)
		return true
	}
	// match: (RORW x (MOVQconst [c]))
	// result: (ROLWconst [int8((-c)&15)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64ROLWconst)
		v.AuxInt = int8ToAuxInt(int8((-c) & 15))
		v.AddArg(x)
		return true
	}
	// match: (RORW x (MOVLconst [c]))
	// result: (ROLWconst [int8((-c)&15)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64ROLWconst)
		v.AuxInt = int8ToAuxInt(int8((-c) & 15))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SARB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SARB x (MOVQconst [c]))
	// result: (SARBconst [int8(min(int64(c)&31,7))] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64SARBconst)
		v.AuxInt = int8ToAuxInt(int8(min(int64(c)&31, 7)))
		v.AddArg(x)
		return true
	}
	// match: (SARB x (MOVLconst [c]))
	// result: (SARBconst [int8(min(int64(c)&31,7))] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64SARBconst)
		v.AuxInt = int8ToAuxInt(int8(min(int64(c)&31, 7)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SARBconst(v *Value) bool {
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
	// match: (SARBconst [c] (MOVQconst [d]))
	// result: (MOVQconst [int64(int8(d))>>uint64(c)])
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(int64(int8(d)) >> uint64(c))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SARL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SARL x (MOVQconst [c]))
	// result: (SARLconst [int8(c&31)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64SARLconst)
		v.AuxInt = int8ToAuxInt(int8(c & 31))
		v.AddArg(x)
		return true
	}
	// match: (SARL x (MOVLconst [c]))
	// result: (SARLconst [int8(c&31)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64SARLconst)
		v.AuxInt = int8ToAuxInt(int8(c & 31))
		v.AddArg(x)
		return true
	}
	// match: (SARL x (ADDQconst [c] y))
	// cond: c & 31 == 0
	// result: (SARL x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SARL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SARL x (NEGQ <t> (ADDQconst [c] y)))
	// cond: c & 31 == 0
	// result: (SARL x (NEGQ <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDQconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SARL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SARL x (ANDQconst [c] y))
	// cond: c & 31 == 31
	// result: (SARL x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ANDQconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SARL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SARL x (NEGQ <t> (ANDQconst [c] y)))
	// cond: c & 31 == 31
	// result: (SARL x (NEGQ <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDQconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SARL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SARL x (ADDLconst [c] y))
	// cond: c & 31 == 0
	// result: (SARL x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ADDLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SARL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SARL x (NEGL <t> (ADDLconst [c] y)))
	// cond: c & 31 == 0
	// result: (SARL x (NEGL <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDLconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SARL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SARL x (ANDLconst [c] y))
	// cond: c & 31 == 31
	// result: (SARL x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SARL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SARL x (NEGL <t> (ANDLconst [c] y)))
	// cond: c & 31 == 31
	// result: (SARL x (NEGL <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SARL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SARL l:(MOVLload [off] {sym} ptr mem) x)
	// cond: buildcfg.GOAMD64 >= 3 && canMergeLoad(v, l) && clobber(l)
	// result: (SARXLload [off] {sym} ptr x mem)
	for {
		l := v_0
		if l.Op != OpAMD64MOVLload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		x := v_1
		if !(buildcfg.GOAMD64 >= 3 && canMergeLoad(v, l) && clobber(l)) {
			break
		}
		v.reset(OpAMD64SARXLload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SARLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SARLconst x [0])
	// result: x
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (SARLconst [c] (MOVQconst [d]))
	// result: (MOVQconst [int64(int32(d))>>uint64(c)])
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(int64(int32(d)) >> uint64(c))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SARQ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SARQ x (MOVQconst [c]))
	// result: (SARQconst [int8(c&63)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64SARQconst)
		v.AuxInt = int8ToAuxInt(int8(c & 63))
		v.AddArg(x)
		return true
	}
	// match: (SARQ x (MOVLconst [c]))
	// result: (SARQconst [int8(c&63)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64SARQconst)
		v.AuxInt = int8ToAuxInt(int8(c & 63))
		v.AddArg(x)
		return true
	}
	// match: (SARQ x (ADDQconst [c] y))
	// cond: c & 63 == 0
	// result: (SARQ x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SARQ x (NEGQ <t> (ADDQconst [c] y)))
	// cond: c & 63 == 0
	// result: (SARQ x (NEGQ <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDQconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SARQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SARQ x (ANDQconst [c] y))
	// cond: c & 63 == 63
	// result: (SARQ x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ANDQconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SARQ x (NEGQ <t> (ANDQconst [c] y)))
	// cond: c & 63 == 63
	// result: (SARQ x (NEGQ <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDQconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SARQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SARQ x (ADDLconst [c] y))
	// cond: c & 63 == 0
	// result: (SARQ x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ADDLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SARQ x (NEGL <t> (ADDLconst [c] y)))
	// cond: c & 63 == 0
	// result: (SARQ x (NEGL <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDLconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SARQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SARQ x (ANDLconst [c] y))
	// cond: c & 63 == 63
	// result: (SARQ x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SARQ x (NEGL <t> (ANDLconst [c] y)))
	// cond: c & 63 == 63
	// result: (SARQ x (NEGL <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SARQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SARQ l:(MOVQload [off] {sym} ptr mem) x)
	// cond: buildcfg.GOAMD64 >= 3 && canMergeLoad(v, l) && clobber(l)
	// result: (SARXQload [off] {sym} ptr x mem)
	for {
		l := v_0
		if l.Op != OpAMD64MOVQload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		x := v_1
		if !(buildcfg.GOAMD64 >= 3 && canMergeLoad(v, l) && clobber(l)) {
			break
		}
		v.reset(OpAMD64SARXQload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SARQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SARQconst x [0])
	// result: x
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (SARQconst [c] (MOVQconst [d]))
	// result: (MOVQconst [d>>uint64(c)])
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(d >> uint64(c))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SARW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SARW x (MOVQconst [c]))
	// result: (SARWconst [int8(min(int64(c)&31,15))] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64SARWconst)
		v.AuxInt = int8ToAuxInt(int8(min(int64(c)&31, 15)))
		v.AddArg(x)
		return true
	}
	// match: (SARW x (MOVLconst [c]))
	// result: (SARWconst [int8(min(int64(c)&31,15))] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64SARWconst)
		v.AuxInt = int8ToAuxInt(int8(min(int64(c)&31, 15)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SARWconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SARWconst x [0])
	// result: x
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (SARWconst [c] (MOVQconst [d]))
	// result: (MOVQconst [int64(int16(d))>>uint64(c)])
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(int64(int16(d)) >> uint64(c))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SARXLload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SARXLload [off] {sym} ptr (MOVLconst [c]) mem)
	// result: (SARLconst [int8(c&31)] (MOVLload [off] {sym} ptr mem))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64SARLconst)
		v.AuxInt = int8ToAuxInt(int8(c & 31))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SARXQload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SARXQload [off] {sym} ptr (MOVQconst [c]) mem)
	// result: (SARQconst [int8(c&63)] (MOVQload [off] {sym} ptr mem))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64SARQconst)
		v.AuxInt = int8ToAuxInt(int8(c & 63))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	// match: (SARXQload [off] {sym} ptr (MOVLconst [c]) mem)
	// result: (SARQconst [int8(c&63)] (MOVQload [off] {sym} ptr mem))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64SARQconst)
		v.AuxInt = int8ToAuxInt(int8(c & 63))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SBBLcarrymask(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SBBLcarrymask (FlagEQ))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SBBLcarrymask (FlagLT_ULT))
	// result: (MOVLconst [-1])
	for {
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(-1)
		return true
	}
	// match: (SBBLcarrymask (FlagLT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SBBLcarrymask (FlagGT_ULT))
	// result: (MOVLconst [-1])
	for {
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(-1)
		return true
	}
	// match: (SBBLcarrymask (FlagGT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SBBQ(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SBBQ x (MOVQconst [c]) borrow)
	// cond: is32Bit(c)
	// result: (SBBQconst x [int32(c)] borrow)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		borrow := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64SBBQconst)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(x, borrow)
		return true
	}
	// match: (SBBQ x y (FlagEQ))
	// result: (SUBQborrow x y)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64SUBQborrow)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SBBQcarrymask(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SBBQcarrymask (FlagEQ))
	// result: (MOVQconst [0])
	for {
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SBBQcarrymask (FlagLT_ULT))
	// result: (MOVQconst [-1])
	for {
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	// match: (SBBQcarrymask (FlagLT_UGT))
	// result: (MOVQconst [0])
	for {
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SBBQcarrymask (FlagGT_ULT))
	// result: (MOVQconst [-1])
	for {
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	// match: (SBBQcarrymask (FlagGT_UGT))
	// result: (MOVQconst [0])
	for {
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SBBQconst(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SBBQconst x [c] (FlagEQ))
	// result: (SUBQconstborrow x [c])
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64SUBQconstborrow)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETA(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SETA (InvertFlags x))
	// result: (SETB x)
	for {
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETB)
		v.AddArg(x)
		return true
	}
	// match: (SETA (FlagEQ))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETA (FlagLT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETA (FlagLT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETA (FlagGT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETA (FlagGT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETAE(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETAE (TESTQ x x))
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (SETAE (TESTL x x))
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpAMD64TESTL {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (SETAE (TESTW x x))
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpAMD64TESTW {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (SETAE (TESTB x x))
	// result: (ConstBool [true])
	for {
		if v_0.Op != OpAMD64TESTB {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(true)
		return true
	}
	// match: (SETAE (BTLconst [0] x))
	// result: (XORLconst [1] (ANDLconst <typ.Bool> [1] x))
	for {
		if v_0.Op != OpAMD64BTLconst || auxIntToInt8(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64XORLconst)
		v.AuxInt = int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, OpAMD64ANDLconst, typ.Bool)
		v0.AuxInt = int32ToAuxInt(1)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETAE (BTQconst [0] x))
	// result: (XORLconst [1] (ANDLconst <typ.Bool> [1] x))
	for {
		if v_0.Op != OpAMD64BTQconst || auxIntToInt8(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64XORLconst)
		v.AuxInt = int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, OpAMD64ANDLconst, typ.Bool)
		v0.AuxInt = int32ToAuxInt(1)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETAE c:(CMPQconst [128] x))
	// cond: c.Uses == 1
	// result: (SETA (CMPQconst [127] x))
	for {
		c := v_0
		if c.Op != OpAMD64CMPQconst || auxIntToInt32(c.AuxInt) != 128 {
			break
		}
		x := c.Args[0]
		if !(c.Uses == 1) {
			break
		}
		v.reset(OpAMD64SETA)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(127)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETAE c:(CMPLconst [128] x))
	// cond: c.Uses == 1
	// result: (SETA (CMPLconst [127] x))
	for {
		c := v_0
		if c.Op != OpAMD64CMPLconst || auxIntToInt32(c.AuxInt) != 128 {
			break
		}
		x := c.Args[0]
		if !(c.Uses == 1) {
			break
		}
		v.reset(OpAMD64SETA)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(127)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETAE (InvertFlags x))
	// result: (SETBE x)
	for {
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETBE)
		v.AddArg(x)
		return true
	}
	// match: (SETAE (FlagEQ))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETAE (FlagLT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETAE (FlagLT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETAE (FlagGT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETAE (FlagGT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETAEstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETAEstore [off] {sym} ptr (InvertFlags x) mem)
	// result: (SETBEstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64InvertFlags {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64SETBEstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (SETAEstore [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SETAEstore [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64SETAEstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETAEstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (SETAEstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64SETAEstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETAEstore [off] {sym} ptr (FlagEQ) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagEQ {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETAEstore [off] {sym} ptr (FlagLT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETAEstore [off] {sym} ptr (FlagLT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETAEstore [off] {sym} ptr (FlagGT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETAEstore [off] {sym} ptr (FlagGT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETAstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETAstore [off] {sym} ptr (InvertFlags x) mem)
	// result: (SETBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64InvertFlags {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64SETBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (SETAstore [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SETAstore [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64SETAstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETAstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (SETAstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64SETAstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETAstore [off] {sym} ptr (FlagEQ) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagEQ {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETAstore [off] {sym} ptr (FlagLT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETAstore [off] {sym} ptr (FlagLT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETAstore [off] {sym} ptr (FlagGT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETAstore [off] {sym} ptr (FlagGT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETB(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (SETB (TESTQ x x))
	// result: (ConstBool [false])
	for {
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (SETB (TESTL x x))
	// result: (ConstBool [false])
	for {
		if v_0.Op != OpAMD64TESTL {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (SETB (TESTW x x))
	// result: (ConstBool [false])
	for {
		if v_0.Op != OpAMD64TESTW {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (SETB (TESTB x x))
	// result: (ConstBool [false])
	for {
		if v_0.Op != OpAMD64TESTB {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = boolToAuxInt(false)
		return true
	}
	// match: (SETB (BTLconst [0] x))
	// result: (ANDLconst [1] x)
	for {
		if v_0.Op != OpAMD64BTLconst || auxIntToInt8(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = int32ToAuxInt(1)
		v.AddArg(x)
		return true
	}
	// match: (SETB (BTQconst [0] x))
	// result: (ANDQconst [1] x)
	for {
		if v_0.Op != OpAMD64BTQconst || auxIntToInt8(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64ANDQconst)
		v.AuxInt = int32ToAuxInt(1)
		v.AddArg(x)
		return true
	}
	// match: (SETB c:(CMPQconst [128] x))
	// cond: c.Uses == 1
	// result: (SETBE (CMPQconst [127] x))
	for {
		c := v_0
		if c.Op != OpAMD64CMPQconst || auxIntToInt32(c.AuxInt) != 128 {
			break
		}
		x := c.Args[0]
		if !(c.Uses == 1) {
			break
		}
		v.reset(OpAMD64SETBE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(127)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETB c:(CMPLconst [128] x))
	// cond: c.Uses == 1
	// result: (SETBE (CMPLconst [127] x))
	for {
		c := v_0
		if c.Op != OpAMD64CMPLconst || auxIntToInt32(c.AuxInt) != 128 {
			break
		}
		x := c.Args[0]
		if !(c.Uses == 1) {
			break
		}
		v.reset(OpAMD64SETBE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(127)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETB (InvertFlags x))
	// result: (SETA x)
	for {
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETA)
		v.AddArg(x)
		return true
	}
	// match: (SETB (FlagEQ))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETB (FlagLT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETB (FlagLT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETB (FlagGT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETB (FlagGT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETBE(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SETBE (InvertFlags x))
	// result: (SETAE x)
	for {
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETAE)
		v.AddArg(x)
		return true
	}
	// match: (SETBE (FlagEQ))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETBE (FlagLT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETBE (FlagLT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETBE (FlagGT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETBE (FlagGT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETBEstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETBEstore [off] {sym} ptr (InvertFlags x) mem)
	// result: (SETAEstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64InvertFlags {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64SETAEstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (SETBEstore [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SETBEstore [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64SETBEstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETBEstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (SETBEstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64SETBEstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETBEstore [off] {sym} ptr (FlagEQ) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagEQ {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETBEstore [off] {sym} ptr (FlagLT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETBEstore [off] {sym} ptr (FlagLT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETBEstore [off] {sym} ptr (FlagGT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETBEstore [off] {sym} ptr (FlagGT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETBstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETBstore [off] {sym} ptr (InvertFlags x) mem)
	// result: (SETAstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64InvertFlags {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64SETAstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (SETBstore [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SETBstore [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64SETBstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETBstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (SETBstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64SETBstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETBstore [off] {sym} ptr (FlagEQ) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagEQ {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETBstore [off] {sym} ptr (FlagLT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETBstore [off] {sym} ptr (FlagLT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETBstore [off] {sym} ptr (FlagGT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETBstore [off] {sym} ptr (FlagGT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETEQ(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (SETEQ (TESTL (SHLL (MOVLconst [1]) x) y))
	// result: (SETAE (BTL x y))
	for {
		if v_0.Op != OpAMD64TESTL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpAMD64SHLL {
				continue
			}
			x := v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != OpAMD64MOVLconst || auxIntToInt32(v_0_0_0.AuxInt) != 1 {
				continue
			}
			y := v_0_1
			v.reset(OpAMD64SETAE)
			v0 := b.NewValue0(v.Pos, OpAMD64BTL, types.TypeFlags)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETEQ (TESTQ (SHLQ (MOVQconst [1]) x) y))
	// result: (SETAE (BTQ x y))
	for {
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpAMD64SHLQ {
				continue
			}
			x := v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != OpAMD64MOVQconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
				continue
			}
			y := v_0_1
			v.reset(OpAMD64SETAE)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQ, types.TypeFlags)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETEQ (TESTLconst [c] x))
	// cond: isUnsignedPowerOfTwo(uint32(c))
	// result: (SETAE (BTLconst [int8(log32u(uint32(c)))] x))
	for {
		if v_0.Op != OpAMD64TESTLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isUnsignedPowerOfTwo(uint32(c))) {
			break
		}
		v.reset(OpAMD64SETAE)
		v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
		v0.AuxInt = int8ToAuxInt(int8(log32u(uint32(c))))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETEQ (TESTQconst [c] x))
	// cond: isUnsignedPowerOfTwo(uint64(c))
	// result: (SETAE (BTQconst [int8(log32u(uint32(c)))] x))
	for {
		if v_0.Op != OpAMD64TESTQconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isUnsignedPowerOfTwo(uint64(c))) {
			break
		}
		v.reset(OpAMD64SETAE)
		v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
		v0.AuxInt = int8ToAuxInt(int8(log32u(uint32(c))))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETEQ (TESTQ (MOVQconst [c]) x))
	// cond: isUnsignedPowerOfTwo(uint64(c))
	// result: (SETAE (BTQconst [int8(log64u(uint64(c)))] x))
	for {
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpAMD64MOVQconst {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			x := v_0_1
			if !(isUnsignedPowerOfTwo(uint64(c))) {
				continue
			}
			v.reset(OpAMD64SETAE)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(int8(log64u(uint64(c))))
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETEQ (CMPLconst [1] s:(ANDLconst [1] _)))
	// result: (SETNE (CMPLconst [0] s))
	for {
		if v_0.Op != OpAMD64CMPLconst || auxIntToInt32(v_0.AuxInt) != 1 {
			break
		}
		s := v_0.Args[0]
		if s.Op != OpAMD64ANDLconst || auxIntToInt32(s.AuxInt) != 1 {
			break
		}
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(0)
		v0.AddArg(s)
		v.AddArg(v0)
		return true
	}
	// match: (SETEQ (CMPQconst [1] s:(ANDQconst [1] _)))
	// result: (SETNE (CMPQconst [0] s))
	for {
		if v_0.Op != OpAMD64CMPQconst || auxIntToInt32(v_0.AuxInt) != 1 {
			break
		}
		s := v_0.Args[0]
		if s.Op != OpAMD64ANDQconst || auxIntToInt32(s.AuxInt) != 1 {
			break
		}
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(0)
		v0.AddArg(s)
		v.AddArg(v0)
		return true
	}
	// match: (SETEQ (TESTQ z1:(SHLQconst [63] (SHRQconst [63] x)) z2))
	// cond: z1==z2
	// result: (SETAE (BTQconst [63] x))
	for {
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			z1 := v_0_0
			if z1.Op != OpAMD64SHLQconst || auxIntToInt8(z1.AuxInt) != 63 {
				continue
			}
			z1_0 := z1.Args[0]
			if z1_0.Op != OpAMD64SHRQconst || auxIntToInt8(z1_0.AuxInt) != 63 {
				continue
			}
			x := z1_0.Args[0]
			z2 := v_0_1
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETAE)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(63)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETEQ (TESTL z1:(SHLLconst [31] (SHRQconst [31] x)) z2))
	// cond: z1==z2
	// result: (SETAE (BTQconst [31] x))
	for {
		if v_0.Op != OpAMD64TESTL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			z1 := v_0_0
			if z1.Op != OpAMD64SHLLconst || auxIntToInt8(z1.AuxInt) != 31 {
				continue
			}
			z1_0 := z1.Args[0]
			if z1_0.Op != OpAMD64SHRQconst || auxIntToInt8(z1_0.AuxInt) != 31 {
				continue
			}
			x := z1_0.Args[0]
			z2 := v_0_1
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETAE)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(31)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETEQ (TESTQ z1:(SHRQconst [63] (SHLQconst [63] x)) z2))
	// cond: z1==z2
	// result: (SETAE (BTQconst [0] x))
	for {
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			z1 := v_0_0
			if z1.Op != OpAMD64SHRQconst || auxIntToInt8(z1.AuxInt) != 63 {
				continue
			}
			z1_0 := z1.Args[0]
			if z1_0.Op != OpAMD64SHLQconst || auxIntToInt8(z1_0.AuxInt) != 63 {
				continue
			}
			x := z1_0.Args[0]
			z2 := v_0_1
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETAE)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(0)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETEQ (TESTL z1:(SHRLconst [31] (SHLLconst [31] x)) z2))
	// cond: z1==z2
	// result: (SETAE (BTLconst [0] x))
	for {
		if v_0.Op != OpAMD64TESTL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			z1 := v_0_0
			if z1.Op != OpAMD64SHRLconst || auxIntToInt8(z1.AuxInt) != 31 {
				continue
			}
			z1_0 := z1.Args[0]
			if z1_0.Op != OpAMD64SHLLconst || auxIntToInt8(z1_0.AuxInt) != 31 {
				continue
			}
			x := z1_0.Args[0]
			z2 := v_0_1
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETAE)
			v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(0)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETEQ (TESTQ z1:(SHRQconst [63] x) z2))
	// cond: z1==z2
	// result: (SETAE (BTQconst [63] x))
	for {
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			z1 := v_0_0
			if z1.Op != OpAMD64SHRQconst || auxIntToInt8(z1.AuxInt) != 63 {
				continue
			}
			x := z1.Args[0]
			z2 := v_0_1
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETAE)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(63)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETEQ (TESTL z1:(SHRLconst [31] x) z2))
	// cond: z1==z2
	// result: (SETAE (BTLconst [31] x))
	for {
		if v_0.Op != OpAMD64TESTL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			z1 := v_0_0
			if z1.Op != OpAMD64SHRLconst || auxIntToInt8(z1.AuxInt) != 31 {
				continue
			}
			x := z1.Args[0]
			z2 := v_0_1
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETAE)
			v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(31)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETEQ (InvertFlags x))
	// result: (SETEQ x)
	for {
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETEQ)
		v.AddArg(x)
		return true
	}
	// match: (SETEQ (FlagEQ))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETEQ (FlagLT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETEQ (FlagLT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETEQ (FlagGT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETEQ (FlagGT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETEQ (TESTQ s:(Select0 blsr:(BLSRQ _)) s))
	// result: (SETEQ (Select1 <types.TypeFlags> blsr))
	for {
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			s := v_0_0
			if s.Op != OpSelect0 {
				continue
			}
			blsr := s.Args[0]
			if blsr.Op != OpAMD64BLSRQ || s != v_0_1 {
				continue
			}
			v.reset(OpAMD64SETEQ)
			v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(blsr)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETEQ (TESTL s:(Select0 blsr:(BLSRL _)) s))
	// result: (SETEQ (Select1 <types.TypeFlags> blsr))
	for {
		if v_0.Op != OpAMD64TESTL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			s := v_0_0
			if s.Op != OpSelect0 {
				continue
			}
			blsr := s.Args[0]
			if blsr.Op != OpAMD64BLSRL || s != v_0_1 {
				continue
			}
			v.reset(OpAMD64SETEQ)
			v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(blsr)
			v.AddArg(v0)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETEQstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETEQstore [off] {sym} ptr (TESTL (SHLL (MOVLconst [1]) x) y) mem)
	// result: (SETAEstore [off] {sym} ptr (BTL x y) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			if v_1_0.Op != OpAMD64SHLL {
				continue
			}
			x := v_1_0.Args[1]
			v_1_0_0 := v_1_0.Args[0]
			if v_1_0_0.Op != OpAMD64MOVLconst || auxIntToInt32(v_1_0_0.AuxInt) != 1 {
				continue
			}
			y := v_1_1
			mem := v_2
			v.reset(OpAMD64SETAEstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTL, types.TypeFlags)
			v0.AddArg2(x, y)
			v.AddArg3(ptr, v0, mem)
			return true
		}
		break
	}
	// match: (SETEQstore [off] {sym} ptr (TESTQ (SHLQ (MOVQconst [1]) x) y) mem)
	// result: (SETAEstore [off] {sym} ptr (BTQ x y) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTQ {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			if v_1_0.Op != OpAMD64SHLQ {
				continue
			}
			x := v_1_0.Args[1]
			v_1_0_0 := v_1_0.Args[0]
			if v_1_0_0.Op != OpAMD64MOVQconst || auxIntToInt64(v_1_0_0.AuxInt) != 1 {
				continue
			}
			y := v_1_1
			mem := v_2
			v.reset(OpAMD64SETAEstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQ, types.TypeFlags)
			v0.AddArg2(x, y)
			v.AddArg3(ptr, v0, mem)
			return true
		}
		break
	}
	// match: (SETEQstore [off] {sym} ptr (TESTLconst [c] x) mem)
	// cond: isUnsignedPowerOfTwo(uint32(c))
	// result: (SETAEstore [off] {sym} ptr (BTLconst [int8(log32u(uint32(c)))] x) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		x := v_1.Args[0]
		mem := v_2
		if !(isUnsignedPowerOfTwo(uint32(c))) {
			break
		}
		v.reset(OpAMD64SETAEstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
		v0.AuxInt = int8ToAuxInt(int8(log32u(uint32(c))))
		v0.AddArg(x)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETEQstore [off] {sym} ptr (TESTQconst [c] x) mem)
	// cond: isUnsignedPowerOfTwo(uint64(c))
	// result: (SETAEstore [off] {sym} ptr (BTQconst [int8(log32u(uint32(c)))] x) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTQconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		x := v_1.Args[0]
		mem := v_2
		if !(isUnsignedPowerOfTwo(uint64(c))) {
			break
		}
		v.reset(OpAMD64SETAEstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
		v0.AuxInt = int8ToAuxInt(int8(log32u(uint32(c))))
		v0.AddArg(x)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETEQstore [off] {sym} ptr (TESTQ (MOVQconst [c]) x) mem)
	// cond: isUnsignedPowerOfTwo(uint64(c))
	// result: (SETAEstore [off] {sym} ptr (BTQconst [int8(log64u(uint64(c)))] x) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTQ {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			if v_1_0.Op != OpAMD64MOVQconst {
				continue
			}
			c := auxIntToInt64(v_1_0.AuxInt)
			x := v_1_1
			mem := v_2
			if !(isUnsignedPowerOfTwo(uint64(c))) {
				continue
			}
			v.reset(OpAMD64SETAEstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(int8(log64u(uint64(c))))
			v0.AddArg(x)
			v.AddArg3(ptr, v0, mem)
			return true
		}
		break
	}
	// match: (SETEQstore [off] {sym} ptr (CMPLconst [1] s:(ANDLconst [1] _)) mem)
	// result: (SETNEstore [off] {sym} ptr (CMPLconst [0] s) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64CMPLconst || auxIntToInt32(v_1.AuxInt) != 1 {
			break
		}
		s := v_1.Args[0]
		if s.Op != OpAMD64ANDLconst || auxIntToInt32(s.AuxInt) != 1 {
			break
		}
		mem := v_2
		v.reset(OpAMD64SETNEstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(0)
		v0.AddArg(s)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETEQstore [off] {sym} ptr (CMPQconst [1] s:(ANDQconst [1] _)) mem)
	// result: (SETNEstore [off] {sym} ptr (CMPQconst [0] s) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64CMPQconst || auxIntToInt32(v_1.AuxInt) != 1 {
			break
		}
		s := v_1.Args[0]
		if s.Op != OpAMD64ANDQconst || auxIntToInt32(s.AuxInt) != 1 {
			break
		}
		mem := v_2
		v.reset(OpAMD64SETNEstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(0)
		v0.AddArg(s)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETEQstore [off] {sym} ptr (TESTQ z1:(SHLQconst [63] (SHRQconst [63] x)) z2) mem)
	// cond: z1==z2
	// result: (SETAEstore [off] {sym} ptr (BTQconst [63] x) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTQ {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			z1 := v_1_0
			if z1.Op != OpAMD64SHLQconst || auxIntToInt8(z1.AuxInt) != 63 {
				continue
			}
			z1_0 := z1.Args[0]
			if z1_0.Op != OpAMD64SHRQconst || auxIntToInt8(z1_0.AuxInt) != 63 {
				continue
			}
			x := z1_0.Args[0]
			z2 := v_1_1
			mem := v_2
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETAEstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(63)
			v0.AddArg(x)
			v.AddArg3(ptr, v0, mem)
			return true
		}
		break
	}
	// match: (SETEQstore [off] {sym} ptr (TESTL z1:(SHLLconst [31] (SHRLconst [31] x)) z2) mem)
	// cond: z1==z2
	// result: (SETAEstore [off] {sym} ptr (BTLconst [31] x) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			z1 := v_1_0
			if z1.Op != OpAMD64SHLLconst || auxIntToInt8(z1.AuxInt) != 31 {
				continue
			}
			z1_0 := z1.Args[0]
			if z1_0.Op != OpAMD64SHRLconst || auxIntToInt8(z1_0.AuxInt) != 31 {
				continue
			}
			x := z1_0.Args[0]
			z2 := v_1_1
			mem := v_2
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETAEstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(31)
			v0.AddArg(x)
			v.AddArg3(ptr, v0, mem)
			return true
		}
		break
	}
	// match: (SETEQstore [off] {sym} ptr (TESTQ z1:(SHRQconst [63] (SHLQconst [63] x)) z2) mem)
	// cond: z1==z2
	// result: (SETAEstore [off] {sym} ptr (BTQconst [0] x) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTQ {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			z1 := v_1_0
			if z1.Op != OpAMD64SHRQconst || auxIntToInt8(z1.AuxInt) != 63 {
				continue
			}
			z1_0 := z1.Args[0]
			if z1_0.Op != OpAMD64SHLQconst || auxIntToInt8(z1_0.AuxInt) != 63 {
				continue
			}
			x := z1_0.Args[0]
			z2 := v_1_1
			mem := v_2
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETAEstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(0)
			v0.AddArg(x)
			v.AddArg3(ptr, v0, mem)
			return true
		}
		break
	}
	// match: (SETEQstore [off] {sym} ptr (TESTL z1:(SHRLconst [31] (SHLLconst [31] x)) z2) mem)
	// cond: z1==z2
	// result: (SETAEstore [off] {sym} ptr (BTLconst [0] x) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			z1 := v_1_0
			if z1.Op != OpAMD64SHRLconst || auxIntToInt8(z1.AuxInt) != 31 {
				continue
			}
			z1_0 := z1.Args[0]
			if z1_0.Op != OpAMD64SHLLconst || auxIntToInt8(z1_0.AuxInt) != 31 {
				continue
			}
			x := z1_0.Args[0]
			z2 := v_1_1
			mem := v_2
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETAEstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(0)
			v0.AddArg(x)
			v.AddArg3(ptr, v0, mem)
			return true
		}
		break
	}
	// match: (SETEQstore [off] {sym} ptr (TESTQ z1:(SHRQconst [63] x) z2) mem)
	// cond: z1==z2
	// result: (SETAEstore [off] {sym} ptr (BTQconst [63] x) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTQ {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			z1 := v_1_0
			if z1.Op != OpAMD64SHRQconst || auxIntToInt8(z1.AuxInt) != 63 {
				continue
			}
			x := z1.Args[0]
			z2 := v_1_1
			mem := v_2
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETAEstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(63)
			v0.AddArg(x)
			v.AddArg3(ptr, v0, mem)
			return true
		}
		break
	}
	// match: (SETEQstore [off] {sym} ptr (TESTL z1:(SHRLconst [31] x) z2) mem)
	// cond: z1==z2
	// result: (SETAEstore [off] {sym} ptr (BTLconst [31] x) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			z1 := v_1_0
			if z1.Op != OpAMD64SHRLconst || auxIntToInt8(z1.AuxInt) != 31 {
				continue
			}
			x := z1.Args[0]
			z2 := v_1_1
			mem := v_2
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETAEstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(31)
			v0.AddArg(x)
			v.AddArg3(ptr, v0, mem)
			return true
		}
		break
	}
	// match: (SETEQstore [off] {sym} ptr (InvertFlags x) mem)
	// result: (SETEQstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64InvertFlags {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64SETEQstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (SETEQstore [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SETEQstore [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64SETEQstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETEQstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (SETEQstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64SETEQstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETEQstore [off] {sym} ptr (FlagEQ) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagEQ {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETEQstore [off] {sym} ptr (FlagLT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETEQstore [off] {sym} ptr (FlagLT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETEQstore [off] {sym} ptr (FlagGT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETEQstore [off] {sym} ptr (FlagGT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETG(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SETG (InvertFlags x))
	// result: (SETL x)
	for {
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETL)
		v.AddArg(x)
		return true
	}
	// match: (SETG (FlagEQ))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETG (FlagLT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETG (FlagLT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETG (FlagGT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETG (FlagGT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETGE(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (SETGE c:(CMPQconst [128] x))
	// cond: c.Uses == 1
	// result: (SETG (CMPQconst [127] x))
	for {
		c := v_0
		if c.Op != OpAMD64CMPQconst || auxIntToInt32(c.AuxInt) != 128 {
			break
		}
		x := c.Args[0]
		if !(c.Uses == 1) {
			break
		}
		v.reset(OpAMD64SETG)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(127)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETGE c:(CMPLconst [128] x))
	// cond: c.Uses == 1
	// result: (SETG (CMPLconst [127] x))
	for {
		c := v_0
		if c.Op != OpAMD64CMPLconst || auxIntToInt32(c.AuxInt) != 128 {
			break
		}
		x := c.Args[0]
		if !(c.Uses == 1) {
			break
		}
		v.reset(OpAMD64SETG)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(127)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETGE (InvertFlags x))
	// result: (SETLE x)
	for {
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETLE)
		v.AddArg(x)
		return true
	}
	// match: (SETGE (FlagEQ))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETGE (FlagLT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETGE (FlagLT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETGE (FlagGT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETGE (FlagGT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETGEstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETGEstore [off] {sym} ptr (InvertFlags x) mem)
	// result: (SETLEstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64InvertFlags {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64SETLEstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (SETGEstore [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SETGEstore [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64SETGEstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETGEstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (SETGEstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64SETGEstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETGEstore [off] {sym} ptr (FlagEQ) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagEQ {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETGEstore [off] {sym} ptr (FlagLT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETGEstore [off] {sym} ptr (FlagLT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETGEstore [off] {sym} ptr (FlagGT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETGEstore [off] {sym} ptr (FlagGT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETGstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETGstore [off] {sym} ptr (InvertFlags x) mem)
	// result: (SETLstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64InvertFlags {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64SETLstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (SETGstore [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SETGstore [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64SETGstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETGstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (SETGstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64SETGstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETGstore [off] {sym} ptr (FlagEQ) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagEQ {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETGstore [off] {sym} ptr (FlagLT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETGstore [off] {sym} ptr (FlagLT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETGstore [off] {sym} ptr (FlagGT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETGstore [off] {sym} ptr (FlagGT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETL(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (SETL c:(CMPQconst [128] x))
	// cond: c.Uses == 1
	// result: (SETLE (CMPQconst [127] x))
	for {
		c := v_0
		if c.Op != OpAMD64CMPQconst || auxIntToInt32(c.AuxInt) != 128 {
			break
		}
		x := c.Args[0]
		if !(c.Uses == 1) {
			break
		}
		v.reset(OpAMD64SETLE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(127)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETL c:(CMPLconst [128] x))
	// cond: c.Uses == 1
	// result: (SETLE (CMPLconst [127] x))
	for {
		c := v_0
		if c.Op != OpAMD64CMPLconst || auxIntToInt32(c.AuxInt) != 128 {
			break
		}
		x := c.Args[0]
		if !(c.Uses == 1) {
			break
		}
		v.reset(OpAMD64SETLE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(127)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETL (InvertFlags x))
	// result: (SETG x)
	for {
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETG)
		v.AddArg(x)
		return true
	}
	// match: (SETL (FlagEQ))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETL (FlagLT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETL (FlagLT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETL (FlagGT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETL (FlagGT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETLE(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SETLE (InvertFlags x))
	// result: (SETGE x)
	for {
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETGE)
		v.AddArg(x)
		return true
	}
	// match: (SETLE (FlagEQ))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETLE (FlagLT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETLE (FlagLT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETLE (FlagGT_ULT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETLE (FlagGT_UGT))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETLEstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETLEstore [off] {sym} ptr (InvertFlags x) mem)
	// result: (SETGEstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64InvertFlags {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64SETGEstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (SETLEstore [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SETLEstore [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64SETLEstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETLEstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (SETLEstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64SETLEstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETLEstore [off] {sym} ptr (FlagEQ) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagEQ {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETLEstore [off] {sym} ptr (FlagLT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETLEstore [off] {sym} ptr (FlagLT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETLEstore [off] {sym} ptr (FlagGT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETLEstore [off] {sym} ptr (FlagGT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETLstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETLstore [off] {sym} ptr (InvertFlags x) mem)
	// result: (SETGstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64InvertFlags {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64SETGstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (SETLstore [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SETLstore [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64SETLstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETLstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (SETLstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64SETLstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETLstore [off] {sym} ptr (FlagEQ) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagEQ {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETLstore [off] {sym} ptr (FlagLT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETLstore [off] {sym} ptr (FlagLT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETLstore [off] {sym} ptr (FlagGT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETLstore [off] {sym} ptr (FlagGT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETNE(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (SETNE (TESTBconst [1] x))
	// result: (ANDLconst [1] x)
	for {
		if v_0.Op != OpAMD64TESTBconst || auxIntToInt8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = int32ToAuxInt(1)
		v.AddArg(x)
		return true
	}
	// match: (SETNE (TESTWconst [1] x))
	// result: (ANDLconst [1] x)
	for {
		if v_0.Op != OpAMD64TESTWconst || auxIntToInt16(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = int32ToAuxInt(1)
		v.AddArg(x)
		return true
	}
	// match: (SETNE (TESTL (SHLL (MOVLconst [1]) x) y))
	// result: (SETB (BTL x y))
	for {
		if v_0.Op != OpAMD64TESTL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpAMD64SHLL {
				continue
			}
			x := v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != OpAMD64MOVLconst || auxIntToInt32(v_0_0_0.AuxInt) != 1 {
				continue
			}
			y := v_0_1
			v.reset(OpAMD64SETB)
			v0 := b.NewValue0(v.Pos, OpAMD64BTL, types.TypeFlags)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETNE (TESTQ (SHLQ (MOVQconst [1]) x) y))
	// result: (SETB (BTQ x y))
	for {
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpAMD64SHLQ {
				continue
			}
			x := v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != OpAMD64MOVQconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
				continue
			}
			y := v_0_1
			v.reset(OpAMD64SETB)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQ, types.TypeFlags)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETNE (TESTLconst [c] x))
	// cond: isUnsignedPowerOfTwo(uint32(c))
	// result: (SETB (BTLconst [int8(log32u(uint32(c)))] x))
	for {
		if v_0.Op != OpAMD64TESTLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isUnsignedPowerOfTwo(uint32(c))) {
			break
		}
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
		v0.AuxInt = int8ToAuxInt(int8(log32u(uint32(c))))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETNE (TESTQconst [c] x))
	// cond: isUnsignedPowerOfTwo(uint64(c))
	// result: (SETB (BTQconst [int8(log32u(uint32(c)))] x))
	for {
		if v_0.Op != OpAMD64TESTQconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isUnsignedPowerOfTwo(uint64(c))) {
			break
		}
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
		v0.AuxInt = int8ToAuxInt(int8(log32u(uint32(c))))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETNE (TESTQ (MOVQconst [c]) x))
	// cond: isUnsignedPowerOfTwo(uint64(c))
	// result: (SETB (BTQconst [int8(log64u(uint64(c)))] x))
	for {
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpAMD64MOVQconst {
				continue
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			x := v_0_1
			if !(isUnsignedPowerOfTwo(uint64(c))) {
				continue
			}
			v.reset(OpAMD64SETB)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(int8(log64u(uint64(c))))
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETNE (CMPLconst [1] s:(ANDLconst [1] _)))
	// result: (SETEQ (CMPLconst [0] s))
	for {
		if v_0.Op != OpAMD64CMPLconst || auxIntToInt32(v_0.AuxInt) != 1 {
			break
		}
		s := v_0.Args[0]
		if s.Op != OpAMD64ANDLconst || auxIntToInt32(s.AuxInt) != 1 {
			break
		}
		v.reset(OpAMD64SETEQ)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(0)
		v0.AddArg(s)
		v.AddArg(v0)
		return true
	}
	// match: (SETNE (CMPQconst [1] s:(ANDQconst [1] _)))
	// result: (SETEQ (CMPQconst [0] s))
	for {
		if v_0.Op != OpAMD64CMPQconst || auxIntToInt32(v_0.AuxInt) != 1 {
			break
		}
		s := v_0.Args[0]
		if s.Op != OpAMD64ANDQconst || auxIntToInt32(s.AuxInt) != 1 {
			break
		}
		v.reset(OpAMD64SETEQ)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(0)
		v0.AddArg(s)
		v.AddArg(v0)
		return true
	}
	// match: (SETNE (TESTQ z1:(SHLQconst [63] (SHRQconst [63] x)) z2))
	// cond: z1==z2
	// result: (SETB (BTQconst [63] x))
	for {
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			z1 := v_0_0
			if z1.Op != OpAMD64SHLQconst || auxIntToInt8(z1.AuxInt) != 63 {
				continue
			}
			z1_0 := z1.Args[0]
			if z1_0.Op != OpAMD64SHRQconst || auxIntToInt8(z1_0.AuxInt) != 63 {
				continue
			}
			x := z1_0.Args[0]
			z2 := v_0_1
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETB)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(63)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETNE (TESTL z1:(SHLLconst [31] (SHRQconst [31] x)) z2))
	// cond: z1==z2
	// result: (SETB (BTQconst [31] x))
	for {
		if v_0.Op != OpAMD64TESTL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			z1 := v_0_0
			if z1.Op != OpAMD64SHLLconst || auxIntToInt8(z1.AuxInt) != 31 {
				continue
			}
			z1_0 := z1.Args[0]
			if z1_0.Op != OpAMD64SHRQconst || auxIntToInt8(z1_0.AuxInt) != 31 {
				continue
			}
			x := z1_0.Args[0]
			z2 := v_0_1
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETB)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(31)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETNE (TESTQ z1:(SHRQconst [63] (SHLQconst [63] x)) z2))
	// cond: z1==z2
	// result: (SETB (BTQconst [0] x))
	for {
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			z1 := v_0_0
			if z1.Op != OpAMD64SHRQconst || auxIntToInt8(z1.AuxInt) != 63 {
				continue
			}
			z1_0 := z1.Args[0]
			if z1_0.Op != OpAMD64SHLQconst || auxIntToInt8(z1_0.AuxInt) != 63 {
				continue
			}
			x := z1_0.Args[0]
			z2 := v_0_1
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETB)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(0)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETNE (TESTL z1:(SHRLconst [31] (SHLLconst [31] x)) z2))
	// cond: z1==z2
	// result: (SETB (BTLconst [0] x))
	for {
		if v_0.Op != OpAMD64TESTL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			z1 := v_0_0
			if z1.Op != OpAMD64SHRLconst || auxIntToInt8(z1.AuxInt) != 31 {
				continue
			}
			z1_0 := z1.Args[0]
			if z1_0.Op != OpAMD64SHLLconst || auxIntToInt8(z1_0.AuxInt) != 31 {
				continue
			}
			x := z1_0.Args[0]
			z2 := v_0_1
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETB)
			v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(0)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETNE (TESTQ z1:(SHRQconst [63] x) z2))
	// cond: z1==z2
	// result: (SETB (BTQconst [63] x))
	for {
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			z1 := v_0_0
			if z1.Op != OpAMD64SHRQconst || auxIntToInt8(z1.AuxInt) != 63 {
				continue
			}
			x := z1.Args[0]
			z2 := v_0_1
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETB)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(63)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETNE (TESTL z1:(SHRLconst [31] x) z2))
	// cond: z1==z2
	// result: (SETB (BTLconst [31] x))
	for {
		if v_0.Op != OpAMD64TESTL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			z1 := v_0_0
			if z1.Op != OpAMD64SHRLconst || auxIntToInt8(z1.AuxInt) != 31 {
				continue
			}
			x := z1.Args[0]
			z2 := v_0_1
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETB)
			v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(31)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETNE (InvertFlags x))
	// result: (SETNE x)
	for {
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETNE)
		v.AddArg(x)
		return true
	}
	// match: (SETNE (FlagEQ))
	// result: (MOVLconst [0])
	for {
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SETNE (FlagLT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETNE (FlagLT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETNE (FlagGT_ULT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETNE (FlagGT_UGT))
	// result: (MOVLconst [1])
	for {
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(1)
		return true
	}
	// match: (SETNE (TESTQ s:(Select0 blsr:(BLSRQ _)) s))
	// result: (SETNE (Select1 <types.TypeFlags> blsr))
	for {
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			s := v_0_0
			if s.Op != OpSelect0 {
				continue
			}
			blsr := s.Args[0]
			if blsr.Op != OpAMD64BLSRQ || s != v_0_1 {
				continue
			}
			v.reset(OpAMD64SETNE)
			v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(blsr)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (SETNE (TESTL s:(Select0 blsr:(BLSRL _)) s))
	// result: (SETNE (Select1 <types.TypeFlags> blsr))
	for {
		if v_0.Op != OpAMD64TESTL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			s := v_0_0
			if s.Op != OpSelect0 {
				continue
			}
			blsr := s.Args[0]
			if blsr.Op != OpAMD64BLSRL || s != v_0_1 {
				continue
			}
			v.reset(OpAMD64SETNE)
			v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(blsr)
			v.AddArg(v0)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETNEstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETNEstore [off] {sym} ptr (TESTL (SHLL (MOVLconst [1]) x) y) mem)
	// result: (SETBstore [off] {sym} ptr (BTL x y) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			if v_1_0.Op != OpAMD64SHLL {
				continue
			}
			x := v_1_0.Args[1]
			v_1_0_0 := v_1_0.Args[0]
			if v_1_0_0.Op != OpAMD64MOVLconst || auxIntToInt32(v_1_0_0.AuxInt) != 1 {
				continue
			}
			y := v_1_1
			mem := v_2
			v.reset(OpAMD64SETBstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTL, types.TypeFlags)
			v0.AddArg2(x, y)
			v.AddArg3(ptr, v0, mem)
			return true
		}
		break
	}
	// match: (SETNEstore [off] {sym} ptr (TESTQ (SHLQ (MOVQconst [1]) x) y) mem)
	// result: (SETBstore [off] {sym} ptr (BTQ x y) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTQ {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			if v_1_0.Op != OpAMD64SHLQ {
				continue
			}
			x := v_1_0.Args[1]
			v_1_0_0 := v_1_0.Args[0]
			if v_1_0_0.Op != OpAMD64MOVQconst || auxIntToInt64(v_1_0_0.AuxInt) != 1 {
				continue
			}
			y := v_1_1
			mem := v_2
			v.reset(OpAMD64SETBstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQ, types.TypeFlags)
			v0.AddArg2(x, y)
			v.AddArg3(ptr, v0, mem)
			return true
		}
		break
	}
	// match: (SETNEstore [off] {sym} ptr (TESTLconst [c] x) mem)
	// cond: isUnsignedPowerOfTwo(uint32(c))
	// result: (SETBstore [off] {sym} ptr (BTLconst [int8(log32u(uint32(c)))] x) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		x := v_1.Args[0]
		mem := v_2
		if !(isUnsignedPowerOfTwo(uint32(c))) {
			break
		}
		v.reset(OpAMD64SETBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
		v0.AuxInt = int8ToAuxInt(int8(log32u(uint32(c))))
		v0.AddArg(x)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETNEstore [off] {sym} ptr (TESTQconst [c] x) mem)
	// cond: isUnsignedPowerOfTwo(uint64(c))
	// result: (SETBstore [off] {sym} ptr (BTQconst [int8(log32u(uint32(c)))] x) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTQconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		x := v_1.Args[0]
		mem := v_2
		if !(isUnsignedPowerOfTwo(uint64(c))) {
			break
		}
		v.reset(OpAMD64SETBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
		v0.AuxInt = int8ToAuxInt(int8(log32u(uint32(c))))
		v0.AddArg(x)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETNEstore [off] {sym} ptr (TESTQ (MOVQconst [c]) x) mem)
	// cond: isUnsignedPowerOfTwo(uint64(c))
	// result: (SETBstore [off] {sym} ptr (BTQconst [int8(log64u(uint64(c)))] x) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTQ {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			if v_1_0.Op != OpAMD64MOVQconst {
				continue
			}
			c := auxIntToInt64(v_1_0.AuxInt)
			x := v_1_1
			mem := v_2
			if !(isUnsignedPowerOfTwo(uint64(c))) {
				continue
			}
			v.reset(OpAMD64SETBstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(int8(log64u(uint64(c))))
			v0.AddArg(x)
			v.AddArg3(ptr, v0, mem)
			return true
		}
		break
	}
	// match: (SETNEstore [off] {sym} ptr (CMPLconst [1] s:(ANDLconst [1] _)) mem)
	// result: (SETEQstore [off] {sym} ptr (CMPLconst [0] s) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64CMPLconst || auxIntToInt32(v_1.AuxInt) != 1 {
			break
		}
		s := v_1.Args[0]
		if s.Op != OpAMD64ANDLconst || auxIntToInt32(s.AuxInt) != 1 {
			break
		}
		mem := v_2
		v.reset(OpAMD64SETEQstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(0)
		v0.AddArg(s)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETNEstore [off] {sym} ptr (CMPQconst [1] s:(ANDQconst [1] _)) mem)
	// result: (SETEQstore [off] {sym} ptr (CMPQconst [0] s) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64CMPQconst || auxIntToInt32(v_1.AuxInt) != 1 {
			break
		}
		s := v_1.Args[0]
		if s.Op != OpAMD64ANDQconst || auxIntToInt32(s.AuxInt) != 1 {
			break
		}
		mem := v_2
		v.reset(OpAMD64SETEQstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(0)
		v0.AddArg(s)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETNEstore [off] {sym} ptr (TESTQ z1:(SHLQconst [63] (SHRQconst [63] x)) z2) mem)
	// cond: z1==z2
	// result: (SETBstore [off] {sym} ptr (BTQconst [63] x) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTQ {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			z1 := v_1_0
			if z1.Op != OpAMD64SHLQconst || auxIntToInt8(z1.AuxInt) != 63 {
				continue
			}
			z1_0 := z1.Args[0]
			if z1_0.Op != OpAMD64SHRQconst || auxIntToInt8(z1_0.AuxInt) != 63 {
				continue
			}
			x := z1_0.Args[0]
			z2 := v_1_1
			mem := v_2
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETBstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(63)
			v0.AddArg(x)
			v.AddArg3(ptr, v0, mem)
			return true
		}
		break
	}
	// match: (SETNEstore [off] {sym} ptr (TESTL z1:(SHLLconst [31] (SHRLconst [31] x)) z2) mem)
	// cond: z1==z2
	// result: (SETBstore [off] {sym} ptr (BTLconst [31] x) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			z1 := v_1_0
			if z1.Op != OpAMD64SHLLconst || auxIntToInt8(z1.AuxInt) != 31 {
				continue
			}
			z1_0 := z1.Args[0]
			if z1_0.Op != OpAMD64SHRLconst || auxIntToInt8(z1_0.AuxInt) != 31 {
				continue
			}
			x := z1_0.Args[0]
			z2 := v_1_1
			mem := v_2
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETBstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(31)
			v0.AddArg(x)
			v.AddArg3(ptr, v0, mem)
			return true
		}
		break
	}
	// match: (SETNEstore [off] {sym} ptr (TESTQ z1:(SHRQconst [63] (SHLQconst [63] x)) z2) mem)
	// cond: z1==z2
	// result: (SETBstore [off] {sym} ptr (BTQconst [0] x) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTQ {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			z1 := v_1_0
			if z1.Op != OpAMD64SHRQconst || auxIntToInt8(z1.AuxInt) != 63 {
				continue
			}
			z1_0 := z1.Args[0]
			if z1_0.Op != OpAMD64SHLQconst || auxIntToInt8(z1_0.AuxInt) != 63 {
				continue
			}
			x := z1_0.Args[0]
			z2 := v_1_1
			mem := v_2
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETBstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(0)
			v0.AddArg(x)
			v.AddArg3(ptr, v0, mem)
			return true
		}
		break
	}
	// match: (SETNEstore [off] {sym} ptr (TESTL z1:(SHRLconst [31] (SHLLconst [31] x)) z2) mem)
	// cond: z1==z2
	// result: (SETBstore [off] {sym} ptr (BTLconst [0] x) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			z1 := v_1_0
			if z1.Op != OpAMD64SHRLconst || auxIntToInt8(z1.AuxInt) != 31 {
				continue
			}
			z1_0 := z1.Args[0]
			if z1_0.Op != OpAMD64SHLLconst || auxIntToInt8(z1_0.AuxInt) != 31 {
				continue
			}
			x := z1_0.Args[0]
			z2 := v_1_1
			mem := v_2
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETBstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(0)
			v0.AddArg(x)
			v.AddArg3(ptr, v0, mem)
			return true
		}
		break
	}
	// match: (SETNEstore [off] {sym} ptr (TESTQ z1:(SHRQconst [63] x) z2) mem)
	// cond: z1==z2
	// result: (SETBstore [off] {sym} ptr (BTQconst [63] x) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTQ {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			z1 := v_1_0
			if z1.Op != OpAMD64SHRQconst || auxIntToInt8(z1.AuxInt) != 63 {
				continue
			}
			x := z1.Args[0]
			z2 := v_1_1
			mem := v_2
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETBstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(63)
			v0.AddArg(x)
			v.AddArg3(ptr, v0, mem)
			return true
		}
		break
	}
	// match: (SETNEstore [off] {sym} ptr (TESTL z1:(SHRLconst [31] x) z2) mem)
	// cond: z1==z2
	// result: (SETBstore [off] {sym} ptr (BTLconst [31] x) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64TESTL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_1_0, v_1_1 = _i0+1, v_1_1, v_1_0 {
			z1 := v_1_0
			if z1.Op != OpAMD64SHRLconst || auxIntToInt8(z1.AuxInt) != 31 {
				continue
			}
			x := z1.Args[0]
			z2 := v_1_1
			mem := v_2
			if !(z1 == z2) {
				continue
			}
			v.reset(OpAMD64SETBstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(31)
			v0.AddArg(x)
			v.AddArg3(ptr, v0, mem)
			return true
		}
		break
	}
	// match: (SETNEstore [off] {sym} ptr (InvertFlags x) mem)
	// result: (SETNEstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64InvertFlags {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpAMD64SETNEstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (SETNEstore [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SETNEstore [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64SETNEstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETNEstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (SETNEstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64SETNEstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SETNEstore [off] {sym} ptr (FlagEQ) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [0]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagEQ {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETNEstore [off] {sym} ptr (FlagLT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETNEstore [off] {sym} ptr (FlagLT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagLT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETNEstore [off] {sym} ptr (FlagGT_ULT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_ULT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETNEstore [off] {sym} ptr (FlagGT_UGT) mem)
	// result: (MOVBstore [off] {sym} ptr (MOVLconst <typ.UInt8> [1]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64FlagGT_UGT {
			break
		}
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLconst, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(1)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHLL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SHLL x (MOVQconst [c]))
	// result: (SHLLconst [int8(c&31)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64SHLLconst)
		v.AuxInt = int8ToAuxInt(int8(c & 31))
		v.AddArg(x)
		return true
	}
	// match: (SHLL x (MOVLconst [c]))
	// result: (SHLLconst [int8(c&31)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64SHLLconst)
		v.AuxInt = int8ToAuxInt(int8(c & 31))
		v.AddArg(x)
		return true
	}
	// match: (SHLL x (ADDQconst [c] y))
	// cond: c & 31 == 0
	// result: (SHLL x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHLL x (NEGQ <t> (ADDQconst [c] y)))
	// cond: c & 31 == 0
	// result: (SHLL x (NEGQ <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDQconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SHLL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHLL x (ANDQconst [c] y))
	// cond: c & 31 == 31
	// result: (SHLL x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ANDQconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHLL x (NEGQ <t> (ANDQconst [c] y)))
	// cond: c & 31 == 31
	// result: (SHLL x (NEGQ <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDQconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SHLL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHLL x (ADDLconst [c] y))
	// cond: c & 31 == 0
	// result: (SHLL x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ADDLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHLL x (NEGL <t> (ADDLconst [c] y)))
	// cond: c & 31 == 0
	// result: (SHLL x (NEGL <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDLconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SHLL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHLL x (ANDLconst [c] y))
	// cond: c & 31 == 31
	// result: (SHLL x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHLL x (NEGL <t> (ANDLconst [c] y)))
	// cond: c & 31 == 31
	// result: (SHLL x (NEGL <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SHLL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHLL l:(MOVLload [off] {sym} ptr mem) x)
	// cond: buildcfg.GOAMD64 >= 3 && canMergeLoad(v, l) && clobber(l)
	// result: (SHLXLload [off] {sym} ptr x mem)
	for {
		l := v_0
		if l.Op != OpAMD64MOVLload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		x := v_1
		if !(buildcfg.GOAMD64 >= 3 && canMergeLoad(v, l) && clobber(l)) {
			break
		}
		v.reset(OpAMD64SHLXLload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHLLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SHLLconst x [0])
	// result: x
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (SHLLconst [1] x)
	// result: (ADDL x x)
	for {
		if auxIntToInt8(v.AuxInt) != 1 {
			break
		}
		x := v_0
		v.reset(OpAMD64ADDL)
		v.AddArg2(x, x)
		return true
	}
	// match: (SHLLconst [c] (ADDL x x))
	// result: (SHLLconst [c+1] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64ADDL {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpAMD64SHLLconst)
		v.AuxInt = int8ToAuxInt(c + 1)
		v.AddArg(x)
		return true
	}
	// match: (SHLLconst [d] (MOVLconst [c]))
	// result: (MOVLconst [c << uint64(d)])
	for {
		d := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(c << uint64(d))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHLQ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SHLQ x (MOVQconst [c]))
	// result: (SHLQconst [int8(c&63)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64SHLQconst)
		v.AuxInt = int8ToAuxInt(int8(c & 63))
		v.AddArg(x)
		return true
	}
	// match: (SHLQ x (MOVLconst [c]))
	// result: (SHLQconst [int8(c&63)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64SHLQconst)
		v.AuxInt = int8ToAuxInt(int8(c & 63))
		v.AddArg(x)
		return true
	}
	// match: (SHLQ x (ADDQconst [c] y))
	// cond: c & 63 == 0
	// result: (SHLQ x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHLQ x (NEGQ <t> (ADDQconst [c] y)))
	// cond: c & 63 == 0
	// result: (SHLQ x (NEGQ <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDQconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHLQ x (ANDQconst [c] y))
	// cond: c & 63 == 63
	// result: (SHLQ x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ANDQconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHLQ x (NEGQ <t> (ANDQconst [c] y)))
	// cond: c & 63 == 63
	// result: (SHLQ x (NEGQ <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDQconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHLQ x (ADDLconst [c] y))
	// cond: c & 63 == 0
	// result: (SHLQ x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ADDLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHLQ x (NEGL <t> (ADDLconst [c] y)))
	// cond: c & 63 == 0
	// result: (SHLQ x (NEGL <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDLconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHLQ x (ANDLconst [c] y))
	// cond: c & 63 == 63
	// result: (SHLQ x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHLQ x (NEGL <t> (ANDLconst [c] y)))
	// cond: c & 63 == 63
	// result: (SHLQ x (NEGL <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHLQ l:(MOVQload [off] {sym} ptr mem) x)
	// cond: buildcfg.GOAMD64 >= 3 && canMergeLoad(v, l) && clobber(l)
	// result: (SHLXQload [off] {sym} ptr x mem)
	for {
		l := v_0
		if l.Op != OpAMD64MOVQload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		x := v_1
		if !(buildcfg.GOAMD64 >= 3 && canMergeLoad(v, l) && clobber(l)) {
			break
		}
		v.reset(OpAMD64SHLXQload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHLQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SHLQconst x [0])
	// result: x
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (SHLQconst [1] x)
	// result: (ADDQ x x)
	for {
		if auxIntToInt8(v.AuxInt) != 1 {
			break
		}
		x := v_0
		v.reset(OpAMD64ADDQ)
		v.AddArg2(x, x)
		return true
	}
	// match: (SHLQconst [c] (ADDQ x x))
	// result: (SHLQconst [c+1] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpAMD64SHLQconst)
		v.AuxInt = int8ToAuxInt(c + 1)
		v.AddArg(x)
		return true
	}
	// match: (SHLQconst [d] (MOVQconst [c]))
	// result: (MOVQconst [c << uint64(d)])
	for {
		d := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(c << uint64(d))
		return true
	}
	// match: (SHLQconst [d] (MOVLconst [c]))
	// result: (MOVQconst [int64(c) << uint64(d)])
	for {
		d := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(int64(c) << uint64(d))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHLXLload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SHLXLload [off] {sym} ptr (MOVLconst [c]) mem)
	// result: (SHLLconst [int8(c&31)] (MOVLload [off] {sym} ptr mem))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64SHLLconst)
		v.AuxInt = int8ToAuxInt(int8(c & 31))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHLXQload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SHLXQload [off] {sym} ptr (MOVQconst [c]) mem)
	// result: (SHLQconst [int8(c&63)] (MOVQload [off] {sym} ptr mem))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64SHLQconst)
		v.AuxInt = int8ToAuxInt(int8(c & 63))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	// match: (SHLXQload [off] {sym} ptr (MOVLconst [c]) mem)
	// result: (SHLQconst [int8(c&63)] (MOVQload [off] {sym} ptr mem))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64SHLQconst)
		v.AuxInt = int8ToAuxInt(int8(c & 63))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHRB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SHRB x (MOVQconst [c]))
	// cond: c&31 < 8
	// result: (SHRBconst [int8(c&31)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(c&31 < 8) {
			break
		}
		v.reset(OpAMD64SHRBconst)
		v.AuxInt = int8ToAuxInt(int8(c & 31))
		v.AddArg(x)
		return true
	}
	// match: (SHRB x (MOVLconst [c]))
	// cond: c&31 < 8
	// result: (SHRBconst [int8(c&31)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(c&31 < 8) {
			break
		}
		v.reset(OpAMD64SHRBconst)
		v.AuxInt = int8ToAuxInt(int8(c & 31))
		v.AddArg(x)
		return true
	}
	// match: (SHRB _ (MOVQconst [c]))
	// cond: c&31 >= 8
	// result: (MOVLconst [0])
	for {
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(c&31 >= 8) {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SHRB _ (MOVLconst [c]))
	// cond: c&31 >= 8
	// result: (MOVLconst [0])
	for {
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(c&31 >= 8) {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHRBconst(v *Value) bool {
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
func rewriteValueAMD64_OpAMD64SHRL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SHRL x (MOVQconst [c]))
	// result: (SHRLconst [int8(c&31)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64SHRLconst)
		v.AuxInt = int8ToAuxInt(int8(c & 31))
		v.AddArg(x)
		return true
	}
	// match: (SHRL x (MOVLconst [c]))
	// result: (SHRLconst [int8(c&31)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64SHRLconst)
		v.AuxInt = int8ToAuxInt(int8(c & 31))
		v.AddArg(x)
		return true
	}
	// match: (SHRL x (ADDQconst [c] y))
	// cond: c & 31 == 0
	// result: (SHRL x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SHRL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHRL x (NEGQ <t> (ADDQconst [c] y)))
	// cond: c & 31 == 0
	// result: (SHRL x (NEGQ <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDQconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SHRL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHRL x (ANDQconst [c] y))
	// cond: c & 31 == 31
	// result: (SHRL x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ANDQconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SHRL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHRL x (NEGQ <t> (ANDQconst [c] y)))
	// cond: c & 31 == 31
	// result: (SHRL x (NEGQ <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDQconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SHRL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHRL x (ADDLconst [c] y))
	// cond: c & 31 == 0
	// result: (SHRL x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ADDLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SHRL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHRL x (NEGL <t> (ADDLconst [c] y)))
	// cond: c & 31 == 0
	// result: (SHRL x (NEGL <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDLconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SHRL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHRL x (ANDLconst [c] y))
	// cond: c & 31 == 31
	// result: (SHRL x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SHRL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHRL x (NEGL <t> (ANDLconst [c] y)))
	// cond: c & 31 == 31
	// result: (SHRL x (NEGL <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SHRL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHRL l:(MOVLload [off] {sym} ptr mem) x)
	// cond: buildcfg.GOAMD64 >= 3 && canMergeLoad(v, l) && clobber(l)
	// result: (SHRXLload [off] {sym} ptr x mem)
	for {
		l := v_0
		if l.Op != OpAMD64MOVLload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		x := v_1
		if !(buildcfg.GOAMD64 >= 3 && canMergeLoad(v, l) && clobber(l)) {
			break
		}
		v.reset(OpAMD64SHRXLload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHRLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SHRLconst [1] (ADDL x x))
	// result: (ANDLconst [0x7fffffff] x)
	for {
		if auxIntToInt8(v.AuxInt) != 1 || v_0.Op != OpAMD64ADDL {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = int32ToAuxInt(0x7fffffff)
		v.AddArg(x)
		return true
	}
	// match: (SHRLconst x [0])
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
func rewriteValueAMD64_OpAMD64SHRQ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SHRQ x (MOVQconst [c]))
	// result: (SHRQconst [int8(c&63)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64SHRQconst)
		v.AuxInt = int8ToAuxInt(int8(c & 63))
		v.AddArg(x)
		return true
	}
	// match: (SHRQ x (MOVLconst [c]))
	// result: (SHRQconst [int8(c&63)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64SHRQconst)
		v.AuxInt = int8ToAuxInt(int8(c & 63))
		v.AddArg(x)
		return true
	}
	// match: (SHRQ x (ADDQconst [c] y))
	// cond: c & 63 == 0
	// result: (SHRQ x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHRQ x (NEGQ <t> (ADDQconst [c] y)))
	// cond: c & 63 == 0
	// result: (SHRQ x (NEGQ <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDQconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHRQ x (ANDQconst [c] y))
	// cond: c & 63 == 63
	// result: (SHRQ x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ANDQconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHRQ x (NEGQ <t> (ANDQconst [c] y)))
	// cond: c & 63 == 63
	// result: (SHRQ x (NEGQ <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDQconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHRQ x (ADDLconst [c] y))
	// cond: c & 63 == 0
	// result: (SHRQ x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ADDLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHRQ x (NEGL <t> (ADDLconst [c] y)))
	// cond: c & 63 == 0
	// result: (SHRQ x (NEGL <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDLconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHRQ x (ANDLconst [c] y))
	// cond: c & 63 == 63
	// result: (SHRQ x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHRQ x (NEGL <t> (ANDLconst [c] y)))
	// cond: c & 63 == 63
	// result: (SHRQ x (NEGL <t> y))
	for {
		x := v_0
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDLconst {
			break
		}
		c := auxIntToInt32(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHRQ l:(MOVQload [off] {sym} ptr mem) x)
	// cond: buildcfg.GOAMD64 >= 3 && canMergeLoad(v, l) && clobber(l)
	// result: (SHRXQload [off] {sym} ptr x mem)
	for {
		l := v_0
		if l.Op != OpAMD64MOVQload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		x := v_1
		if !(buildcfg.GOAMD64 >= 3 && canMergeLoad(v, l) && clobber(l)) {
			break
		}
		v.reset(OpAMD64SHRXQload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHRQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SHRQconst [1] (ADDQ x x))
	// result: (BTRQconst [63] x)
	for {
		if auxIntToInt8(v.AuxInt) != 1 || v_0.Op != OpAMD64ADDQ {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpAMD64BTRQconst)
		v.AuxInt = int8ToAuxInt(63)
		v.AddArg(x)
		return true
	}
	// match: (SHRQconst x [0])
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
func rewriteValueAMD64_OpAMD64SHRW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SHRW x (MOVQconst [c]))
	// cond: c&31 < 16
	// result: (SHRWconst [int8(c&31)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(c&31 < 16) {
			break
		}
		v.reset(OpAMD64SHRWconst)
		v.AuxInt = int8ToAuxInt(int8(c & 31))
		v.AddArg(x)
		return true
	}
	// match: (SHRW x (MOVLconst [c]))
	// cond: c&31 < 16
	// result: (SHRWconst [int8(c&31)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(c&31 < 16) {
			break
		}
		v.reset(OpAMD64SHRWconst)
		v.AuxInt = int8ToAuxInt(int8(c & 31))
		v.AddArg(x)
		return true
	}
	// match: (SHRW _ (MOVQconst [c]))
	// cond: c&31 >= 16
	// result: (MOVLconst [0])
	for {
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(c&31 >= 16) {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SHRW _ (MOVLconst [c]))
	// cond: c&31 >= 16
	// result: (MOVLconst [0])
	for {
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		if !(c&31 >= 16) {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHRWconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SHRWconst x [0])
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
func rewriteValueAMD64_OpAMD64SHRXLload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SHRXLload [off] {sym} ptr (MOVLconst [c]) mem)
	// result: (SHRLconst [int8(c&31)] (MOVLload [off] {sym} ptr mem))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64SHRLconst)
		v.AuxInt = int8ToAuxInt(int8(c & 31))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHRXQload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SHRXQload [off] {sym} ptr (MOVQconst [c]) mem)
	// result: (SHRQconst [int8(c&63)] (MOVQload [off] {sym} ptr mem))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64SHRQconst)
		v.AuxInt = int8ToAuxInt(int8(c & 63))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	// match: (SHRXQload [off] {sym} ptr (MOVLconst [c]) mem)
	// result: (SHRQconst [int8(c&63)] (MOVQload [off] {sym} ptr mem))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.reset(OpAMD64SHRQconst)
		v.AuxInt = int8ToAuxInt(int8(c & 63))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SUBL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUBL x (MOVLconst [c]))
	// result: (SUBLconst x [c])
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_1.AuxInt)
		v.reset(OpAMD64SUBLconst)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SUBL (MOVLconst [c]) x)
	// result: (NEGL (SUBLconst <v.Type> x [c]))
	for {
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_1
		v.reset(OpAMD64NEGL)
		v0 := b.NewValue0(v.Pos, OpAMD64SUBLconst, v.Type)
		v0.AuxInt = int32ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SUBL x x)
	// result: (MOVLconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (SUBL x l:(MOVLload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (SUBLload x [off] {sym} ptr mem)
	for {
		x := v_0
		l := v_1
		if l.Op != OpAMD64MOVLload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64SUBLload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(x, ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SUBLconst(v *Value) bool {
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
		v.reset(OpAMD64ADDLconst)
		v.AuxInt = int32ToAuxInt(-c)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpAMD64SUBLload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SUBLload [off1] {sym} val (ADDQconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SUBLload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64SUBLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (SUBLload [off1] {sym1} val (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (SUBLload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64SUBLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (SUBLload x [off] {sym} ptr (MOVSSstore [off] {sym} ptr y _))
	// result: (SUBL x (MOVLf2i y))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		ptr := v_1
		if v_2.Op != OpAMD64MOVSSstore || auxIntToInt32(v_2.AuxInt) != off || auxToSym(v_2.Aux) != sym {
			break
		}
		y := v_2.Args[1]
		if ptr != v_2.Args[0] {
			break
		}
		v.reset(OpAMD64SUBL)
		v0 := b.NewValue0(v_2.Pos, OpAMD64MOVLf2i, typ.UInt32)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SUBLmodify(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBLmodify [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SUBLmodify [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64SUBLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SUBLmodify [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (SUBLmodify [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64SUBLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SUBQ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUBQ x (MOVQconst [c]))
	// cond: is32Bit(c)
	// result: (SUBQconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64SUBQconst)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (SUBQ (MOVQconst [c]) x)
	// cond: is32Bit(c)
	// result: (NEGQ (SUBQconst <v.Type> x [int32(c)]))
	for {
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64NEGQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SUBQconst, v.Type)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SUBQ x x)
	// result: (MOVQconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SUBQ x l:(MOVQload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (SUBQload x [off] {sym} ptr mem)
	for {
		x := v_0
		l := v_1
		if l.Op != OpAMD64MOVQload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64SUBQload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(x, ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SUBQborrow(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBQborrow x (MOVQconst [c]))
	// cond: is32Bit(c)
	// result: (SUBQconstborrow x [int32(c)])
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64SUBQconstborrow)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SUBQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SUBQconst [0] x)
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (SUBQconst [c] x)
	// cond: c != -(1<<31)
	// result: (ADDQconst [-c] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(c != -(1 << 31)) {
			break
		}
		v.reset(OpAMD64ADDQconst)
		v.AuxInt = int32ToAuxInt(-c)
		v.AddArg(x)
		return true
	}
	// match: (SUBQconst (MOVQconst [d]) [c])
	// result: (MOVQconst [d-int64(c)])
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(d - int64(c))
		return true
	}
	// match: (SUBQconst (SUBQconst x [d]) [c])
	// cond: is32Bit(int64(-c)-int64(d))
	// result: (ADDQconst [-c-d] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64SUBQconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(int64(-c) - int64(d))) {
			break
		}
		v.reset(OpAMD64ADDQconst)
		v.AuxInt = int32ToAuxInt(-c - d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SUBQload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SUBQload [off1] {sym} val (ADDQconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SUBQload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64SUBQload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (SUBQload [off1] {sym1} val (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (SUBQload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64SUBQload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (SUBQload x [off] {sym} ptr (MOVSDstore [off] {sym} ptr y _))
	// result: (SUBQ x (MOVQf2i y))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		ptr := v_1
		if v_2.Op != OpAMD64MOVSDstore || auxIntToInt32(v_2.AuxInt) != off || auxToSym(v_2.Aux) != sym {
			break
		}
		y := v_2.Args[1]
		if ptr != v_2.Args[0] {
			break
		}
		v.reset(OpAMD64SUBQ)
		v0 := b.NewValue0(v_2.Pos, OpAMD64MOVQf2i, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SUBQmodify(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBQmodify [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SUBQmodify [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64SUBQmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (SUBQmodify [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (SUBQmodify [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64SUBQmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SUBSD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBSD x l:(MOVSDload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (SUBSDload x [off] {sym} ptr mem)
	for {
		x := v_0
		l := v_1
		if l.Op != OpAMD64MOVSDload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64SUBSDload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(x, ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SUBSDload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SUBSDload [off1] {sym} val (ADDQconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SUBSDload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64SUBSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (SUBSDload [off1] {sym1} val (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (SUBSDload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64SUBSDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (SUBSDload x [off] {sym} ptr (MOVQstore [off] {sym} ptr y _))
	// result: (SUBSD x (MOVQi2f y))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		ptr := v_1
		if v_2.Op != OpAMD64MOVQstore || auxIntToInt32(v_2.AuxInt) != off || auxToSym(v_2.Aux) != sym {
			break
		}
		y := v_2.Args[1]
		if ptr != v_2.Args[0] {
			break
		}
		v.reset(OpAMD64SUBSD)
		v0 := b.NewValue0(v_2.Pos, OpAMD64MOVQi2f, typ.Float64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SUBSS(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBSS x l:(MOVSSload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (SUBSSload x [off] {sym} ptr mem)
	for {
		x := v_0
		l := v_1
		if l.Op != OpAMD64MOVSSload {
			break
		}
		off := auxIntToInt32(l.AuxInt)
		sym := auxToSym(l.Aux)
		mem := l.Args[1]
		ptr := l.Args[0]
		if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64SUBSSload)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(x, ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SUBSSload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SUBSSload [off1] {sym} val (ADDQconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (SUBSSload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64SUBSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (SUBSSload [off1] {sym1} val (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (SUBSSload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64SUBSSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (SUBSSload x [off] {sym} ptr (MOVLstore [off] {sym} ptr y _))
	// result: (SUBSS x (MOVLi2f y))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		ptr := v_1
		if v_2.Op != OpAMD64MOVLstore || auxIntToInt32(v_2.AuxInt) != off || auxToSym(v_2.Aux) != sym {
			break
		}
		y := v_2.Args[1]
		if ptr != v_2.Args[0] {
			break
		}
		v.reset(OpAMD64SUBSS)
		v0 := b.NewValue0(v_2.Pos, OpAMD64MOVLi2f, typ.Float32)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64TESTB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TESTB (MOVLconst [c]) x)
	// result: (TESTBconst [int8(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64MOVLconst {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			x := v_1
			v.reset(OpAMD64TESTBconst)
			v.AuxInt = int8ToAuxInt(int8(c))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (TESTB l:(MOVBload {sym} [off] ptr mem) l2)
	// cond: l == l2 && l.Uses == 2 && clobber(l)
	// result: @l.Block (CMPBconstload {sym} [makeValAndOff(0, off)] ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			l := v_0
			if l.Op != OpAMD64MOVBload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			l2 := v_1
			if !(l == l2 && l.Uses == 2 && clobber(l)) {
				continue
			}
			b = l.Block
			v0 := b.NewValue0(l.Pos, OpAMD64CMPBconstload, types.TypeFlags)
			v.copyOf(v0)
			v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, off))
			v0.Aux = symToAux(sym)
			v0.AddArg2(ptr, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64TESTBconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TESTBconst [-1] x)
	// cond: x.Op != OpAMD64MOVLconst
	// result: (TESTB x x)
	for {
		if auxIntToInt8(v.AuxInt) != -1 {
			break
		}
		x := v_0
		if !(x.Op != OpAMD64MOVLconst) {
			break
		}
		v.reset(OpAMD64TESTB)
		v.AddArg2(x, x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64TESTL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TESTL (MOVLconst [c]) x)
	// result: (TESTLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64MOVLconst {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			x := v_1
			v.reset(OpAMD64TESTLconst)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (TESTL l:(MOVLload {sym} [off] ptr mem) l2)
	// cond: l == l2 && l.Uses == 2 && clobber(l)
	// result: @l.Block (CMPLconstload {sym} [makeValAndOff(0, off)] ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			l := v_0
			if l.Op != OpAMD64MOVLload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			l2 := v_1
			if !(l == l2 && l.Uses == 2 && clobber(l)) {
				continue
			}
			b = l.Block
			v0 := b.NewValue0(l.Pos, OpAMD64CMPLconstload, types.TypeFlags)
			v.copyOf(v0)
			v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, off))
			v0.Aux = symToAux(sym)
			v0.AddArg2(ptr, mem)
			return true
		}
		break
	}
	// match: (TESTL a:(ANDLload [off] {sym} x ptr mem) a)
	// cond: a.Uses == 2 && a.Block == v.Block && clobber(a)
	// result: (TESTL (MOVLload <a.Type> [off] {sym} ptr mem) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if a.Op != OpAMD64ANDLload {
				continue
			}
			off := auxIntToInt32(a.AuxInt)
			sym := auxToSym(a.Aux)
			mem := a.Args[2]
			x := a.Args[0]
			ptr := a.Args[1]
			if a != v_1 || !(a.Uses == 2 && a.Block == v.Block && clobber(a)) {
				continue
			}
			v.reset(OpAMD64TESTL)
			v0 := b.NewValue0(a.Pos, OpAMD64MOVLload, a.Type)
			v0.AuxInt = int32ToAuxInt(off)
			v0.Aux = symToAux(sym)
			v0.AddArg2(ptr, mem)
			v.AddArg2(v0, x)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64TESTLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TESTLconst [c] (MOVLconst [c]))
	// cond: c == 0
	// result: (FlagEQ)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst || auxIntToInt32(v_0.AuxInt) != c || !(c == 0) {
			break
		}
		v.reset(OpAMD64FlagEQ)
		return true
	}
	// match: (TESTLconst [c] (MOVLconst [c]))
	// cond: c < 0
	// result: (FlagLT_UGT)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst || auxIntToInt32(v_0.AuxInt) != c || !(c < 0) {
			break
		}
		v.reset(OpAMD64FlagLT_UGT)
		return true
	}
	// match: (TESTLconst [c] (MOVLconst [c]))
	// cond: c > 0
	// result: (FlagGT_UGT)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst || auxIntToInt32(v_0.AuxInt) != c || !(c > 0) {
			break
		}
		v.reset(OpAMD64FlagGT_UGT)
		return true
	}
	// match: (TESTLconst [-1] x)
	// cond: x.Op != OpAMD64MOVLconst
	// result: (TESTL x x)
	for {
		if auxIntToInt32(v.AuxInt) != -1 {
			break
		}
		x := v_0
		if !(x.Op != OpAMD64MOVLconst) {
			break
		}
		v.reset(OpAMD64TESTL)
		v.AddArg2(x, x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64TESTQ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TESTQ (MOVQconst [c]) x)
	// cond: is32Bit(c)
	// result: (TESTQconst [int32(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64MOVQconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			x := v_1
			if !(is32Bit(c)) {
				continue
			}
			v.reset(OpAMD64TESTQconst)
			v.AuxInt = int32ToAuxInt(int32(c))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (TESTQ l:(MOVQload {sym} [off] ptr mem) l2)
	// cond: l == l2 && l.Uses == 2 && clobber(l)
	// result: @l.Block (CMPQconstload {sym} [makeValAndOff(0, off)] ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			l := v_0
			if l.Op != OpAMD64MOVQload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			l2 := v_1
			if !(l == l2 && l.Uses == 2 && clobber(l)) {
				continue
			}
			b = l.Block
			v0 := b.NewValue0(l.Pos, OpAMD64CMPQconstload, types.TypeFlags)
			v.copyOf(v0)
			v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, off))
			v0.Aux = symToAux(sym)
			v0.AddArg2(ptr, mem)
			return true
		}
		break
	}
	// match: (TESTQ a:(ANDQload [off] {sym} x ptr mem) a)
	// cond: a.Uses == 2 && a.Block == v.Block && clobber(a)
	// result: (TESTQ (MOVQload <a.Type> [off] {sym} ptr mem) x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if a.Op != OpAMD64ANDQload {
				continue
			}
			off := auxIntToInt32(a.AuxInt)
			sym := auxToSym(a.Aux)
			mem := a.Args[2]
			x := a.Args[0]
			ptr := a.Args[1]
			if a != v_1 || !(a.Uses == 2 && a.Block == v.Block && clobber(a)) {
				continue
			}
			v.reset(OpAMD64TESTQ)
			v0 := b.NewValue0(a.Pos, OpAMD64MOVQload, a.Type)
			v0.AuxInt = int32ToAuxInt(off)
			v0.Aux = symToAux(sym)
			v0.AddArg2(ptr, mem)
			v.AddArg2(v0, x)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64TESTQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TESTQconst [c] (MOVQconst [d]))
	// cond: int64(c) == d && c == 0
	// result: (FlagEQ)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		if !(int64(c) == d && c == 0) {
			break
		}
		v.reset(OpAMD64FlagEQ)
		return true
	}
	// match: (TESTQconst [c] (MOVQconst [d]))
	// cond: int64(c) == d && c < 0
	// result: (FlagLT_UGT)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		if !(int64(c) == d && c < 0) {
			break
		}
		v.reset(OpAMD64FlagLT_UGT)
		return true
	}
	// match: (TESTQconst [c] (MOVQconst [d]))
	// cond: int64(c) == d && c > 0
	// result: (FlagGT_UGT)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		if !(int64(c) == d && c > 0) {
			break
		}
		v.reset(OpAMD64FlagGT_UGT)
		return true
	}
	// match: (TESTQconst [-1] x)
	// cond: x.Op != OpAMD64MOVQconst
	// result: (TESTQ x x)
	for {
		if auxIntToInt32(v.AuxInt) != -1 {
			break
		}
		x := v_0
		if !(x.Op != OpAMD64MOVQconst) {
			break
		}
		v.reset(OpAMD64TESTQ)
		v.AddArg2(x, x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64TESTW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TESTW (MOVLconst [c]) x)
	// result: (TESTWconst [int16(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64MOVLconst {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			x := v_1
			v.reset(OpAMD64TESTWconst)
			v.AuxInt = int16ToAuxInt(int16(c))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (TESTW l:(MOVWload {sym} [off] ptr mem) l2)
	// cond: l == l2 && l.Uses == 2 && clobber(l)
	// result: @l.Block (CMPWconstload {sym} [makeValAndOff(0, off)] ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			l := v_0
			if l.Op != OpAMD64MOVWload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			l2 := v_1
			if !(l == l2 && l.Uses == 2 && clobber(l)) {
				continue
			}
			b = l.Block
			v0 := b.NewValue0(l.Pos, OpAMD64CMPWconstload, types.TypeFlags)
			v.copyOf(v0)
			v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, off))
			v0.Aux = symToAux(sym)
			v0.AddArg2(ptr, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64TESTWconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TESTWconst [-1] x)
	// cond: x.Op != OpAMD64MOVLconst
	// result: (TESTW x x)
	for {
		if auxIntToInt16(v.AuxInt) != -1 {
			break
		}
		x := v_0
		if !(x.Op != OpAMD64MOVLconst) {
			break
		}
		v.reset(OpAMD64TESTW)
		v.AddArg2(x, x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPANDQ512(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPANDQ512 x (VPMOVMToVec64x8 k))
	// result: (VMOVDQU64Masked512 x k)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64VPMOVMToVec64x8 {
				continue
			}
			k := v_1.Args[0]
			v.reset(OpAMD64VMOVDQU64Masked512)
			v.AddArg2(x, k)
			return true
		}
		break
	}
	// match: (VPANDQ512 x (VPMOVMToVec32x16 k))
	// result: (VMOVDQU32Masked512 x k)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64VPMOVMToVec32x16 {
				continue
			}
			k := v_1.Args[0]
			v.reset(OpAMD64VMOVDQU32Masked512)
			v.AddArg2(x, k)
			return true
		}
		break
	}
	// match: (VPANDQ512 x (VPMOVMToVec16x32 k))
	// result: (VMOVDQU16Masked512 x k)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64VPMOVMToVec16x32 {
				continue
			}
			k := v_1.Args[0]
			v.reset(OpAMD64VMOVDQU16Masked512)
			v.AddArg2(x, k)
			return true
		}
		break
	}
	// match: (VPANDQ512 x (VPMOVMToVec8x64 k))
	// result: (VMOVDQU8Masked512 x k)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64VPMOVMToVec8x64 {
				continue
			}
			k := v_1.Args[0]
			v.reset(OpAMD64VMOVDQU8Masked512)
			v.AddArg2(x, k)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPMOVVec16x16ToM(v *Value) bool {
	v_0 := v.Args[0]
	// match: (VPMOVVec16x16ToM (VPMOVMToVec16x16 x))
	// result: x
	for {
		if v_0.Op != OpAMD64VPMOVMToVec16x16 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPMOVVec16x32ToM(v *Value) bool {
	v_0 := v.Args[0]
	// match: (VPMOVVec16x32ToM (VPMOVMToVec16x32 x))
	// result: x
	for {
		if v_0.Op != OpAMD64VPMOVMToVec16x32 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPMOVVec16x8ToM(v *Value) bool {
	v_0 := v.Args[0]
	// match: (VPMOVVec16x8ToM (VPMOVMToVec16x8 x))
	// result: x
	for {
		if v_0.Op != OpAMD64VPMOVMToVec16x8 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPMOVVec32x16ToM(v *Value) bool {
	v_0 := v.Args[0]
	// match: (VPMOVVec32x16ToM (VPMOVMToVec32x16 x))
	// result: x
	for {
		if v_0.Op != OpAMD64VPMOVMToVec32x16 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPMOVVec32x4ToM(v *Value) bool {
	v_0 := v.Args[0]
	// match: (VPMOVVec32x4ToM (VPMOVMToVec32x4 x))
	// result: x
	for {
		if v_0.Op != OpAMD64VPMOVMToVec32x4 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPMOVVec32x8ToM(v *Value) bool {
	v_0 := v.Args[0]
	// match: (VPMOVVec32x8ToM (VPMOVMToVec32x8 x))
	// result: x
	for {
		if v_0.Op != OpAMD64VPMOVMToVec32x8 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPMOVVec64x2ToM(v *Value) bool {
	v_0 := v.Args[0]
	// match: (VPMOVVec64x2ToM (VPMOVMToVec64x2 x))
	// result: x
	for {
		if v_0.Op != OpAMD64VPMOVMToVec64x2 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPMOVVec64x4ToM(v *Value) bool {
	v_0 := v.Args[0]
	// match: (VPMOVVec64x4ToM (VPMOVMToVec64x4 x))
	// result: x
	for {
		if v_0.Op != OpAMD64VPMOVMToVec64x4 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPMOVVec64x8ToM(v *Value) bool {
	v_0 := v.Args[0]
	// match: (VPMOVVec64x8ToM (VPMOVMToVec64x8 x))
	// result: x
	for {
		if v_0.Op != OpAMD64VPMOVMToVec64x8 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPMOVVec8x16ToM(v *Value) bool {
	v_0 := v.Args[0]
	// match: (VPMOVVec8x16ToM (VPMOVMToVec8x16 x))
	// result: x
	for {
		if v_0.Op != OpAMD64VPMOVMToVec8x16 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPMOVVec8x32ToM(v *Value) bool {
	v_0 := v.Args[0]
	// match: (VPMOVVec8x32ToM (VPMOVMToVec8x32 x))
	// result: x
	for {
		if v_0.Op != OpAMD64VPMOVMToVec8x32 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPMOVVec8x64ToM(v *Value) bool {
	v_0 := v.Args[0]
	// match: (VPMOVVec8x64ToM (VPMOVMToVec8x64 x))
	// result: x
	for {
		if v_0.Op != OpAMD64VPMOVMToVec8x64 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPSLLD128(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPSLLD128 x (MOVQconst [c]))
	// result: (VPSLLD128const [uint8(c)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64VPSLLD128const)
		v.AuxInt = uint8ToAuxInt(uint8(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPSLLD256(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPSLLD256 x (MOVQconst [c]))
	// result: (VPSLLD256const [uint8(c)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64VPSLLD256const)
		v.AuxInt = uint8ToAuxInt(uint8(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPSLLD512(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPSLLD512 x (MOVQconst [c]))
	// result: (VPSLLD512const [uint8(c)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64VPSLLD512const)
		v.AuxInt = uint8ToAuxInt(uint8(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPSLLQ128(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPSLLQ128 x (MOVQconst [c]))
	// result: (VPSLLQ128const [uint8(c)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64VPSLLQ128const)
		v.AuxInt = uint8ToAuxInt(uint8(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPSLLQ256(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPSLLQ256 x (MOVQconst [c]))
	// result: (VPSLLQ256const [uint8(c)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64VPSLLQ256const)
		v.AuxInt = uint8ToAuxInt(uint8(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPSLLQ512(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPSLLQ512 x (MOVQconst [c]))
	// result: (VPSLLQ512const [uint8(c)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64VPSLLQ512const)
		v.AuxInt = uint8ToAuxInt(uint8(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPSLLW128(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPSLLW128 x (MOVQconst [c]))
	// result: (VPSLLW128const [uint8(c)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64VPSLLW128const)
		v.AuxInt = uint8ToAuxInt(uint8(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPSLLW256(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPSLLW256 x (MOVQconst [c]))
	// result: (VPSLLW256const [uint8(c)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64VPSLLW256const)
		v.AuxInt = uint8ToAuxInt(uint8(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPSLLW512(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPSLLW512 x (MOVQconst [c]))
	// result: (VPSLLW512const [uint8(c)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64VPSLLW512const)
		v.AuxInt = uint8ToAuxInt(uint8(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPSRAD128(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPSRAD128 x (MOVQconst [c]))
	// result: (VPSRAD128const [uint8(c)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64VPSRAD128const)
		v.AuxInt = uint8ToAuxInt(uint8(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPSRAD256(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPSRAD256 x (MOVQconst [c]))
	// result: (VPSRAD256const [uint8(c)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64VPSRAD256const)
		v.AuxInt = uint8ToAuxInt(uint8(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPSRAD512(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPSRAD512 x (MOVQconst [c]))
	// result: (VPSRAD512const [uint8(c)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64VPSRAD512const)
		v.AuxInt = uint8ToAuxInt(uint8(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPSRAQ128(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPSRAQ128 x (MOVQconst [c]))
	// result: (VPSRAQ128const [uint8(c)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64VPSRAQ128const)
		v.AuxInt = uint8ToAuxInt(uint8(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPSRAQ256(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPSRAQ256 x (MOVQconst [c]))
	// result: (VPSRAQ256const [uint8(c)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64VPSRAQ256const)
		v.AuxInt = uint8ToAuxInt(uint8(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPSRAQ512(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPSRAQ512 x (MOVQconst [c]))
	// result: (VPSRAQ512const [uint8(c)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64VPSRAQ512const)
		v.AuxInt = uint8ToAuxInt(uint8(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPSRAW128(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPSRAW128 x (MOVQconst [c]))
	// result: (VPSRAW128const [uint8(c)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64VPSRAW128const)
		v.AuxInt = uint8ToAuxInt(uint8(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPSRAW256(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPSRAW256 x (MOVQconst [c]))
	// result: (VPSRAW256const [uint8(c)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64VPSRAW256const)
		v.AuxInt = uint8ToAuxInt(uint8(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64VPSRAW512(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPSRAW512 x (MOVQconst [c]))
	// result: (VPSRAW512const [uint8(c)] x)
	for {
		x := v_0
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpAMD64VPSRAW512const)
		v.AuxInt = uint8ToAuxInt(uint8(c))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XADDLlock(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XADDLlock [off1] {sym} val (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (XADDLlock [off1+off2] {sym} val ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		ptr := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64XADDLlock)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XADDQlock(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XADDQlock [off1] {sym} val (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (XADDQlock [off1+off2] {sym} val ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		ptr := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64XADDQlock)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XCHGL(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XCHGL [off1] {sym} val (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (XCHGL [off1+off2] {sym} val ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		ptr := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64XCHGL)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, ptr, mem)
		return true
	}
	// match: (XCHGL [off1] {sym1} val (LEAQ [off2] {sym2} ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && ptr.Op != OpSB
	// result: (XCHGL [off1+off2] {mergeSym(sym1,sym2)} val ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		ptr := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && ptr.Op != OpSB) {
			break
		}
		v.reset(OpAMD64XCHGL)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XCHGQ(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XCHGQ [off1] {sym} val (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (XCHGQ [off1+off2] {sym} val ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		ptr := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64XCHGQ)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, ptr, mem)
		return true
	}
	// match: (XCHGQ [off1] {sym1} val (LEAQ [off2] {sym2} ptr) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && ptr.Op != OpSB
	// result: (XCHGQ [off1+off2] {mergeSym(sym1,sym2)} val ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		ptr := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2) && ptr.Op != OpSB) {
			break
		}
		v.reset(OpAMD64XCHGQ)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XORL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XORL (SHLL (MOVLconst [1]) y) x)
	// result: (BTCL x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64SHLL {
				continue
			}
			y := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64MOVLconst || auxIntToInt32(v_0_0.AuxInt) != 1 {
				continue
			}
			x := v_1
			v.reset(OpAMD64BTCL)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (XORL x (MOVLconst [c]))
	// result: (XORLconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64MOVLconst {
				continue
			}
			c := auxIntToInt32(v_1.AuxInt)
			v.reset(OpAMD64XORLconst)
			v.AuxInt = int32ToAuxInt(c)
			v.AddArg(x)
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
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(0)
		return true
	}
	// match: (XORL x l:(MOVLload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (XORLload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != OpAMD64MOVLload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(OpAMD64XORLload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	// match: (XORL x (ADDLconst [-1] x))
	// cond: buildcfg.GOAMD64 >= 3
	// result: (BLSMSKL x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64ADDLconst || auxIntToInt32(v_1.AuxInt) != -1 || x != v_1.Args[0] || !(buildcfg.GOAMD64 >= 3) {
				continue
			}
			v.reset(OpAMD64BLSMSKL)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64XORLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (XORLconst [1] (SETNE x))
	// result: (SETEQ x)
	for {
		if auxIntToInt32(v.AuxInt) != 1 || v_0.Op != OpAMD64SETNE {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETEQ)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [1] (SETEQ x))
	// result: (SETNE x)
	for {
		if auxIntToInt32(v.AuxInt) != 1 || v_0.Op != OpAMD64SETEQ {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETNE)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [1] (SETL x))
	// result: (SETGE x)
	for {
		if auxIntToInt32(v.AuxInt) != 1 || v_0.Op != OpAMD64SETL {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETGE)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [1] (SETGE x))
	// result: (SETL x)
	for {
		if auxIntToInt32(v.AuxInt) != 1 || v_0.Op != OpAMD64SETGE {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETL)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [1] (SETLE x))
	// result: (SETG x)
	for {
		if auxIntToInt32(v.AuxInt) != 1 || v_0.Op != OpAMD64SETLE {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETG)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [1] (SETG x))
	// result: (SETLE x)
	for {
		if auxIntToInt32(v.AuxInt) != 1 || v_0.Op != OpAMD64SETG {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETLE)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [1] (SETB x))
	// result: (SETAE x)
	for {
		if auxIntToInt32(v.AuxInt) != 1 || v_0.Op != OpAMD64SETB {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETAE)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [1] (SETAE x))
	// result: (SETB x)
	for {
		if auxIntToInt32(v.AuxInt) != 1 || v_0.Op != OpAMD64SETAE {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETB)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [1] (SETBE x))
	// result: (SETA x)
	for {
		if auxIntToInt32(v.AuxInt) != 1 || v_0.Op != OpAMD64SETBE {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETA)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [1] (SETA x))
	// result: (SETBE x)
	for {
		if auxIntToInt32(v.AuxInt) != 1 || v_0.Op != OpAMD64SETA {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETBE)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [c] (XORLconst [d] x))
	// result: (XORLconst [c ^ d] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64XORLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64XORLconst)
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
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(c ^ d)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XORLconstmodify(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XORLconstmodify [valoff1] {sym} (ADDQconst [off2] base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2)
	// result: (XORLconstmodify [ValAndOff(valoff1).addOffset32(off2)] {sym} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2)) {
			break
		}
		v.reset(OpAMD64XORLconstmodify)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (XORLconstmodify [valoff1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)
	// result: (XORLconstmodify [ValAndOff(valoff1).addOffset32(off2)] {mergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64XORLconstmodify)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XORLload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (XORLload [off1] {sym} val (ADDQconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (XORLload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64XORLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (XORLload [off1] {sym1} val (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (XORLload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64XORLload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (XORLload x [off] {sym} ptr (MOVSSstore [off] {sym} ptr y _))
	// result: (XORL x (MOVLf2i y))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		ptr := v_1
		if v_2.Op != OpAMD64MOVSSstore || auxIntToInt32(v_2.AuxInt) != off || auxToSym(v_2.Aux) != sym {
			break
		}
		y := v_2.Args[1]
		if ptr != v_2.Args[0] {
			break
		}
		v.reset(OpAMD64XORL)
		v0 := b.NewValue0(v_2.Pos, OpAMD64MOVLf2i, typ.UInt32)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XORLmodify(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XORLmodify [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (XORLmodify [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64XORLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (XORLmodify [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (XORLmodify [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64XORLmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XORQ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XORQ (SHLQ (MOVQconst [1]) y) x)
	// result: (BTCQ x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64SHLQ {
				continue
			}
			y := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64MOVQconst || auxIntToInt64(v_0_0.AuxInt) != 1 {
				continue
			}
			x := v_1
			v.reset(OpAMD64BTCQ)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (XORQ (MOVQconst [c]) x)
	// cond: isUnsignedPowerOfTwo(uint64(c)) && uint64(c) >= 1<<31
	// result: (BTCQconst [int8(log64u(uint64(c)))] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64MOVQconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			x := v_1
			if !(isUnsignedPowerOfTwo(uint64(c)) && uint64(c) >= 1<<31) {
				continue
			}
			v.reset(OpAMD64BTCQconst)
			v.AuxInt = int8ToAuxInt(int8(log64u(uint64(c))))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (XORQ x (MOVQconst [c]))
	// cond: is32Bit(c)
	// result: (XORQconst [int32(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64MOVQconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(is32Bit(c)) {
				continue
			}
			v.reset(OpAMD64XORQconst)
			v.AuxInt = int32ToAuxInt(int32(c))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (XORQ x x)
	// result: (MOVQconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (XORQ x l:(MOVQload [off] {sym} ptr mem))
	// cond: canMergeLoadClobber(v, l, x) && clobber(l)
	// result: (XORQload x [off] {sym} ptr mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			l := v_1
			if l.Op != OpAMD64MOVQload {
				continue
			}
			off := auxIntToInt32(l.AuxInt)
			sym := auxToSym(l.Aux)
			mem := l.Args[1]
			ptr := l.Args[0]
			if !(canMergeLoadClobber(v, l, x) && clobber(l)) {
				continue
			}
			v.reset(OpAMD64XORQload)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v.AddArg3(x, ptr, mem)
			return true
		}
		break
	}
	// match: (XORQ x (ADDQconst [-1] x))
	// cond: buildcfg.GOAMD64 >= 3
	// result: (BLSMSKQ x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64ADDQconst || auxIntToInt32(v_1.AuxInt) != -1 || x != v_1.Args[0] || !(buildcfg.GOAMD64 >= 3) {
				continue
			}
			v.reset(OpAMD64BLSMSKQ)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64XORQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (XORQconst [c] (XORQconst [d] x))
	// result: (XORQconst [c ^ d] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64XORQconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64XORQconst)
		v.AuxInt = int32ToAuxInt(c ^ d)
		v.AddArg(x)
		return true
	}
	// match: (XORQconst [0] x)
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (XORQconst [c] (MOVQconst [d]))
	// result: (MOVQconst [int64(c)^d])
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(int64(c) ^ d)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XORQconstmodify(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XORQconstmodify [valoff1] {sym} (ADDQconst [off2] base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2)
	// result: (XORQconstmodify [ValAndOff(valoff1).addOffset32(off2)] {sym} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2)) {
			break
		}
		v.reset(OpAMD64XORQconstmodify)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(base, mem)
		return true
	}
	// match: (XORQconstmodify [valoff1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)
	// result: (XORQconstmodify [ValAndOff(valoff1).addOffset32(off2)] {mergeSym(sym1,sym2)} base mem)
	for {
		valoff1 := auxIntToValAndOff(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		base := v_0.Args[0]
		mem := v_1
		if !(ValAndOff(valoff1).canAdd32(off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64XORQconstmodify)
		v.AuxInt = valAndOffToAuxInt(ValAndOff(valoff1).addOffset32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(base, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XORQload(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (XORQload [off1] {sym} val (ADDQconst [off2] base) mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (XORQload [off1+off2] {sym} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64XORQload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (XORQload [off1] {sym1} val (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (XORQload [off1+off2] {mergeSym(sym1,sym2)} val base mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		val := v_0
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		base := v_1.Args[0]
		mem := v_2
		if !(is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64XORQload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(val, base, mem)
		return true
	}
	// match: (XORQload x [off] {sym} ptr (MOVSDstore [off] {sym} ptr y _))
	// result: (XORQ x (MOVQf2i y))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		x := v_0
		ptr := v_1
		if v_2.Op != OpAMD64MOVSDstore || auxIntToInt32(v_2.AuxInt) != off || auxToSym(v_2.Aux) != sym {
			break
		}
		y := v_2.Args[1]
		if ptr != v_2.Args[0] {
			break
		}
		v.reset(OpAMD64XORQ)
		v0 := b.NewValue0(v_2.Pos, OpAMD64MOVQf2i, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XORQmodify(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XORQmodify [off1] {sym} (ADDQconst [off2] base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2))
	// result: (XORQmodify [off1+off2] {sym} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		base := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1) + int64(off2))) {
			break
		}
		v.reset(OpAMD64XORQmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(sym)
		v.AddArg3(base, val, mem)
		return true
	}
	// match: (XORQmodify [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(int64(off1)+int64(off2)) && canMergeSym(sym1, sym2)
	// result: (XORQmodify [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpAMD64LEAQ {
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
		v.reset(OpAMD64XORQmodify)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(base, val, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAddr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Addr {sym} base)
	// result: (LEAQ {sym} base)
	for {
		sym := auxToSym(v.Aux)
		base := v_0
		v.reset(OpAMD64LEAQ)
		v.Aux = symToAux(sym)
		v.AddArg(base)
		return true
	}
}
func rewriteValueAMD64_OpAtomicAdd32(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicAdd32 ptr val mem)
	// result: (AddTupleFirst32 val (XADDLlock val ptr mem))
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64AddTupleFirst32)
		v0 := b.NewValue0(v.Pos, OpAMD64XADDLlock, types.NewTuple(typ.UInt32, types.TypeMem))
		v0.AddArg3(val, ptr, mem)
		v.AddArg2(val, v0)
		return true
	}
}
func rewriteValueAMD64_OpAtomicAdd64(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicAdd64 ptr val mem)
	// result: (AddTupleFirst64 val (XADDQlock val ptr mem))
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64AddTupleFirst64)
		v0 := b.NewValue0(v.Pos, OpAMD64XADDQlock, types.NewTuple(typ.UInt64, types.TypeMem))
		v0.AddArg3(val, ptr, mem)
		v.AddArg2(val, v0)
		return true
	}
}
func rewriteValueAMD64_OpAtomicAnd32(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicAnd32 ptr val mem)
	// result: (ANDLlock ptr val mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64ANDLlock)
		v.AddArg3(ptr, val, mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicAnd32value(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicAnd32value ptr val mem)
	// result: (LoweredAtomicAnd32 ptr val mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64LoweredAtomicAnd32)
		v.AddArg3(ptr, val, mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicAnd64value(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicAnd64value ptr val mem)
	// result: (LoweredAtomicAnd64 ptr val mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64LoweredAtomicAnd64)
		v.AddArg3(ptr, val, mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicAnd8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicAnd8 ptr val mem)
	// result: (ANDBlock ptr val mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64ANDBlock)
		v.AddArg3(ptr, val, mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicCompareAndSwap32(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicCompareAndSwap32 ptr old new_ mem)
	// result: (CMPXCHGLlock ptr old new_ mem)
	for {
		ptr := v_0
		old := v_1
		new_ := v_2
		mem := v_3
		v.reset(OpAMD64CMPXCHGLlock)
		v.AddArg4(ptr, old, new_, mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicCompareAndSwap64(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicCompareAndSwap64 ptr old new_ mem)
	// result: (CMPXCHGQlock ptr old new_ mem)
	for {
		ptr := v_0
		old := v_1
		new_ := v_2
		mem := v_3
		v.reset(OpAMD64CMPXCHGQlock)
		v.AddArg4(ptr, old, new_, mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicExchange32(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicExchange32 ptr val mem)
	// result: (XCHGL val ptr mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64XCHGL)
		v.AddArg3(val, ptr, mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicExchange64(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicExchange64 ptr val mem)
	// result: (XCHGQ val ptr mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64XCHGQ)
		v.AddArg3(val, ptr, mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicExchange8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicExchange8 ptr val mem)
	// result: (XCHGB val ptr mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64XCHGB)
		v.AddArg3(val, ptr, mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicLoad32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicLoad32 ptr mem)
	// result: (MOVLatomicload ptr mem)
	for {
		ptr := v_0
		mem := v_1
		v.reset(OpAMD64MOVLatomicload)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicLoad64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicLoad64 ptr mem)
	// result: (MOVQatomicload ptr mem)
	for {
		ptr := v_0
		mem := v_1
		v.reset(OpAMD64MOVQatomicload)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicLoad8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicLoad8 ptr mem)
	// result: (MOVBatomicload ptr mem)
	for {
		ptr := v_0
		mem := v_1
		v.reset(OpAMD64MOVBatomicload)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicLoadPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicLoadPtr ptr mem)
	// result: (MOVQatomicload ptr mem)
	for {
		ptr := v_0
		mem := v_1
		v.reset(OpAMD64MOVQatomicload)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicOr32(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicOr32 ptr val mem)
	// result: (ORLlock ptr val mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64ORLlock)
		v.AddArg3(ptr, val, mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicOr32value(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicOr32value ptr val mem)
	// result: (LoweredAtomicOr32 ptr val mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64LoweredAtomicOr32)
		v.AddArg3(ptr, val, mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicOr64value(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicOr64value ptr val mem)
	// result: (LoweredAtomicOr64 ptr val mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64LoweredAtomicOr64)
		v.AddArg3(ptr, val, mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicOr8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicOr8 ptr val mem)
	// result: (ORBlock ptr val mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64ORBlock)
		v.AddArg3(ptr, val, mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicStore32(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicStore32 ptr val mem)
	// result: (Select1 (XCHGL <types.NewTuple(typ.UInt32,types.TypeMem)> val ptr mem))
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64XCHGL, types.NewTuple(typ.UInt32, types.TypeMem))
		v0.AddArg3(val, ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpAtomicStore64(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicStore64 ptr val mem)
	// result: (Select1 (XCHGQ <types.NewTuple(typ.UInt64,types.TypeMem)> val ptr mem))
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64XCHGQ, types.NewTuple(typ.UInt64, types.TypeMem))
		v0.AddArg3(val, ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpAtomicStore8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicStore8 ptr val mem)
	// result: (Select1 (XCHGB <types.NewTuple(typ.UInt8,types.TypeMem)> val ptr mem))
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64XCHGB, types.NewTuple(typ.UInt8, types.TypeMem))
		v0.AddArg3(val, ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpAtomicStorePtrNoWB(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicStorePtrNoWB ptr val mem)
	// result: (Select1 (XCHGQ <types.NewTuple(typ.BytePtr,types.TypeMem)> val ptr mem))
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64XCHGQ, types.NewTuple(typ.BytePtr, types.TypeMem))
		v0.AddArg3(val, ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpBitLen16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen16 x)
	// cond: buildcfg.GOAMD64 < 3
	// result: (BSRL (LEAL1 <typ.UInt32> [1] (MOVWQZX <typ.UInt32> x) (MOVWQZX <typ.UInt32> x)))
	for {
		x := v_0
		if !(buildcfg.GOAMD64 < 3) {
			break
		}
		v.reset(OpAMD64BSRL)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL1, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWQZX, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg2(v1, v1)
		v.AddArg(v0)
		return true
	}
	// match: (BitLen16 <t> x)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (NEGQ (ADDQconst <t> [-32] (LZCNTL (MOVWQZX <x.Type> x))))
	for {
		t := v.Type
		x := v_0
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64NEGQ)
		v0 := b.NewValue0(v.Pos, OpAMD64ADDQconst, t)
		v0.AuxInt = int32ToAuxInt(-32)
		v1 := b.NewValue0(v.Pos, OpAMD64LZCNTL, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWQZX, x.Type)
		v2.AddArg(x)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpBitLen32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen32 x)
	// cond: buildcfg.GOAMD64 < 3
	// result: (Select0 (BSRQ (LEAQ1 <typ.UInt64> [1] (MOVLQZX <typ.UInt64> x) (MOVLQZX <typ.UInt64> x))))
	for {
		x := v_0
		if !(buildcfg.GOAMD64 < 3) {
			break
		}
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64BSRQ, types.NewTuple(typ.UInt64, types.TypeFlags))
		v1 := b.NewValue0(v.Pos, OpAMD64LEAQ1, typ.UInt64)
		v1.AuxInt = int32ToAuxInt(1)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLQZX, typ.UInt64)
		v2.AddArg(x)
		v1.AddArg2(v2, v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (BitLen32 <t> x)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (NEGQ (ADDQconst <t> [-32] (LZCNTL x)))
	for {
		t := v.Type
		x := v_0
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64NEGQ)
		v0 := b.NewValue0(v.Pos, OpAMD64ADDQconst, t)
		v0.AuxInt = int32ToAuxInt(-32)
		v1 := b.NewValue0(v.Pos, OpAMD64LZCNTL, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpBitLen64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen64 <t> x)
	// cond: buildcfg.GOAMD64 < 3
	// result: (ADDQconst [1] (CMOVQEQ <t> (Select0 <t> (BSRQ x)) (MOVQconst <t> [-1]) (Select1 <types.TypeFlags> (BSRQ x))))
	for {
		t := v.Type
		x := v_0
		if !(buildcfg.GOAMD64 < 3) {
			break
		}
		v.reset(OpAMD64ADDQconst)
		v.AuxInt = int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, OpAMD64CMOVQEQ, t)
		v1 := b.NewValue0(v.Pos, OpSelect0, t)
		v2 := b.NewValue0(v.Pos, OpAMD64BSRQ, types.NewTuple(typ.UInt64, types.TypeFlags))
		v2.AddArg(x)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVQconst, t)
		v3.AuxInt = int64ToAuxInt(-1)
		v4 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v4.AddArg(v2)
		v0.AddArg3(v1, v3, v4)
		v.AddArg(v0)
		return true
	}
	// match: (BitLen64 <t> x)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (NEGQ (ADDQconst <t> [-64] (LZCNTQ x)))
	for {
		t := v.Type
		x := v_0
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64NEGQ)
		v0 := b.NewValue0(v.Pos, OpAMD64ADDQconst, t)
		v0.AuxInt = int32ToAuxInt(-64)
		v1 := b.NewValue0(v.Pos, OpAMD64LZCNTQ, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpBitLen8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen8 x)
	// cond: buildcfg.GOAMD64 < 3
	// result: (BSRL (LEAL1 <typ.UInt32> [1] (MOVBQZX <typ.UInt32> x) (MOVBQZX <typ.UInt32> x)))
	for {
		x := v_0
		if !(buildcfg.GOAMD64 < 3) {
			break
		}
		v.reset(OpAMD64BSRL)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL1, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVBQZX, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg2(v1, v1)
		v.AddArg(v0)
		return true
	}
	// match: (BitLen8 <t> x)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (NEGQ (ADDQconst <t> [-32] (LZCNTL (MOVBQZX <x.Type> x))))
	for {
		t := v.Type
		x := v_0
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64NEGQ)
		v0 := b.NewValue0(v.Pos, OpAMD64ADDQconst, t)
		v0.AuxInt = int32ToAuxInt(-32)
		v1 := b.NewValue0(v.Pos, OpAMD64LZCNTL, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVBQZX, x.Type)
		v2.AddArg(x)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpBswap16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Bswap16 x)
	// result: (ROLWconst [8] x)
	for {
		x := v_0
		v.reset(OpAMD64ROLWconst)
		v.AuxInt = int8ToAuxInt(8)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCeil(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Ceil x)
	// result: (ROUNDSD [2] x)
	for {
		x := v_0
		v.reset(OpAMD64ROUNDSD)
		v.AuxInt = int8ToAuxInt(2)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCeilFloat32x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CeilFloat32x4 x)
	// result: (VROUNDPS128 [2] x)
	for {
		x := v_0
		v.reset(OpAMD64VROUNDPS128)
		v.AuxInt = uint8ToAuxInt(2)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCeilFloat32x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CeilFloat32x8 x)
	// result: (VROUNDPS256 [2] x)
	for {
		x := v_0
		v.reset(OpAMD64VROUNDPS256)
		v.AuxInt = uint8ToAuxInt(2)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCeilFloat64x2(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CeilFloat64x2 x)
	// result: (VROUNDPD128 [2] x)
	for {
		x := v_0
		v.reset(OpAMD64VROUNDPD128)
		v.AuxInt = uint8ToAuxInt(2)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCeilFloat64x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CeilFloat64x4 x)
	// result: (VROUNDPD256 [2] x)
	for {
		x := v_0
		v.reset(OpAMD64VROUNDPD256)
		v.AuxInt = uint8ToAuxInt(2)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCeilScaledFloat32x16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CeilScaledFloat32x16 [a] x)
	// result: (VRNDSCALEPS512 [a+2] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPS512)
		v.AuxInt = uint8ToAuxInt(a + 2)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCeilScaledFloat32x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CeilScaledFloat32x4 [a] x)
	// result: (VRNDSCALEPS128 [a+2] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPS128)
		v.AuxInt = uint8ToAuxInt(a + 2)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCeilScaledFloat32x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CeilScaledFloat32x8 [a] x)
	// result: (VRNDSCALEPS256 [a+2] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPS256)
		v.AuxInt = uint8ToAuxInt(a + 2)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCeilScaledFloat64x2(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CeilScaledFloat64x2 [a] x)
	// result: (VRNDSCALEPD128 [a+2] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPD128)
		v.AuxInt = uint8ToAuxInt(a + 2)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCeilScaledFloat64x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CeilScaledFloat64x4 [a] x)
	// result: (VRNDSCALEPD256 [a+2] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPD256)
		v.AuxInt = uint8ToAuxInt(a + 2)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCeilScaledFloat64x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CeilScaledFloat64x8 [a] x)
	// result: (VRNDSCALEPD512 [a+2] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPD512)
		v.AuxInt = uint8ToAuxInt(a + 2)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCeilScaledResidueFloat32x16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CeilScaledResidueFloat32x16 [a] x)
	// result: (VREDUCEPS512 [a+2] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPS512)
		v.AuxInt = uint8ToAuxInt(a + 2)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCeilScaledResidueFloat32x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CeilScaledResidueFloat32x4 [a] x)
	// result: (VREDUCEPS128 [a+2] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPS128)
		v.AuxInt = uint8ToAuxInt(a + 2)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCeilScaledResidueFloat32x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CeilScaledResidueFloat32x8 [a] x)
	// result: (VREDUCEPS256 [a+2] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPS256)
		v.AuxInt = uint8ToAuxInt(a + 2)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCeilScaledResidueFloat64x2(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CeilScaledResidueFloat64x2 [a] x)
	// result: (VREDUCEPD128 [a+2] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPD128)
		v.AuxInt = uint8ToAuxInt(a + 2)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCeilScaledResidueFloat64x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CeilScaledResidueFloat64x4 [a] x)
	// result: (VREDUCEPD256 [a+2] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPD256)
		v.AuxInt = uint8ToAuxInt(a + 2)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCeilScaledResidueFloat64x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CeilScaledResidueFloat64x8 [a] x)
	// result: (VREDUCEPD512 [a+2] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPD512)
		v.AuxInt = uint8ToAuxInt(a + 2)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCompressFloat32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressFloat32x16 x mask)
	// result: (VCOMPRESSPSMasked512 x (VPMOVVec32x16ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VCOMPRESSPSMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressFloat32x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressFloat32x4 x mask)
	// result: (VCOMPRESSPSMasked128 x (VPMOVVec32x4ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VCOMPRESSPSMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x4ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressFloat32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressFloat32x8 x mask)
	// result: (VCOMPRESSPSMasked256 x (VPMOVVec32x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VCOMPRESSPSMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressFloat64x2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressFloat64x2 x mask)
	// result: (VCOMPRESSPDMasked128 x (VPMOVVec64x2ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VCOMPRESSPDMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x2ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressFloat64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressFloat64x4 x mask)
	// result: (VCOMPRESSPDMasked256 x (VPMOVVec64x4ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VCOMPRESSPDMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x4ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressFloat64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressFloat64x8 x mask)
	// result: (VCOMPRESSPDMasked512 x (VPMOVVec64x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VCOMPRESSPDMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressInt16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressInt16x16 x mask)
	// result: (VPCOMPRESSWMasked256 x (VPMOVVec16x16ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSWMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressInt16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressInt16x32 x mask)
	// result: (VPCOMPRESSWMasked512 x (VPMOVVec16x32ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSWMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x32ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressInt16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressInt16x8 x mask)
	// result: (VPCOMPRESSWMasked128 x (VPMOVVec16x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSWMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressInt32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressInt32x16 x mask)
	// result: (VPCOMPRESSDMasked512 x (VPMOVVec32x16ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSDMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressInt32x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressInt32x4 x mask)
	// result: (VPCOMPRESSDMasked128 x (VPMOVVec32x4ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSDMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x4ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressInt32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressInt32x8 x mask)
	// result: (VPCOMPRESSDMasked256 x (VPMOVVec32x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSDMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressInt64x2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressInt64x2 x mask)
	// result: (VPCOMPRESSQMasked128 x (VPMOVVec64x2ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSQMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x2ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressInt64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressInt64x4 x mask)
	// result: (VPCOMPRESSQMasked256 x (VPMOVVec64x4ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSQMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x4ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressInt64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressInt64x8 x mask)
	// result: (VPCOMPRESSQMasked512 x (VPMOVVec64x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSQMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressInt8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressInt8x16 x mask)
	// result: (VPCOMPRESSBMasked128 x (VPMOVVec8x16ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSBMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressInt8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressInt8x32 x mask)
	// result: (VPCOMPRESSBMasked256 x (VPMOVVec8x32ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSBMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x32ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressInt8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressInt8x64 x mask)
	// result: (VPCOMPRESSBMasked512 x (VPMOVVec8x64ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSBMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x64ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressUint16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressUint16x16 x mask)
	// result: (VPCOMPRESSWMasked256 x (VPMOVVec16x16ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSWMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressUint16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressUint16x32 x mask)
	// result: (VPCOMPRESSWMasked512 x (VPMOVVec16x32ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSWMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x32ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressUint16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressUint16x8 x mask)
	// result: (VPCOMPRESSWMasked128 x (VPMOVVec16x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSWMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressUint32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressUint32x16 x mask)
	// result: (VPCOMPRESSDMasked512 x (VPMOVVec32x16ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSDMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressUint32x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressUint32x4 x mask)
	// result: (VPCOMPRESSDMasked128 x (VPMOVVec32x4ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSDMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x4ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressUint32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressUint32x8 x mask)
	// result: (VPCOMPRESSDMasked256 x (VPMOVVec32x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSDMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressUint64x2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressUint64x2 x mask)
	// result: (VPCOMPRESSQMasked128 x (VPMOVVec64x2ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSQMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x2ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressUint64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressUint64x4 x mask)
	// result: (VPCOMPRESSQMasked256 x (VPMOVVec64x4ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSQMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x4ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressUint64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressUint64x8 x mask)
	// result: (VPCOMPRESSQMasked512 x (VPMOVVec64x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSQMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressUint8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressUint8x16 x mask)
	// result: (VPCOMPRESSBMasked128 x (VPMOVVec8x16ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSBMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressUint8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressUint8x32 x mask)
	// result: (VPCOMPRESSBMasked256 x (VPMOVVec8x32ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSBMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x32ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCompressUint8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CompressUint8x64 x mask)
	// result: (VPCOMPRESSBMasked512 x (VPMOVVec8x64ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPCOMPRESSBMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x64ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpCondSelect(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CondSelect <t> x y (SETEQ cond))
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (CMOVQEQ y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETEQ {
			break
		}
		cond := v_2.Args[0]
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpAMD64CMOVQEQ)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETNE cond))
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (CMOVQNE y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETNE {
			break
		}
		cond := v_2.Args[0]
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpAMD64CMOVQNE)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETL cond))
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (CMOVQLT y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETL {
			break
		}
		cond := v_2.Args[0]
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpAMD64CMOVQLT)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETG cond))
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (CMOVQGT y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETG {
			break
		}
		cond := v_2.Args[0]
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpAMD64CMOVQGT)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETLE cond))
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (CMOVQLE y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETLE {
			break
		}
		cond := v_2.Args[0]
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpAMD64CMOVQLE)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETGE cond))
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (CMOVQGE y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETGE {
			break
		}
		cond := v_2.Args[0]
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpAMD64CMOVQGE)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETA cond))
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (CMOVQHI y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETA {
			break
		}
		cond := v_2.Args[0]
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpAMD64CMOVQHI)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETB cond))
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (CMOVQCS y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETB {
			break
		}
		cond := v_2.Args[0]
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpAMD64CMOVQCS)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETAE cond))
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (CMOVQCC y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETAE {
			break
		}
		cond := v_2.Args[0]
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpAMD64CMOVQCC)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETBE cond))
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (CMOVQLS y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETBE {
			break
		}
		cond := v_2.Args[0]
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpAMD64CMOVQLS)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETEQF cond))
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (CMOVQEQF y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETEQF {
			break
		}
		cond := v_2.Args[0]
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpAMD64CMOVQEQF)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETNEF cond))
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (CMOVQNEF y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETNEF {
			break
		}
		cond := v_2.Args[0]
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpAMD64CMOVQNEF)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETGF cond))
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (CMOVQGTF y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETGF {
			break
		}
		cond := v_2.Args[0]
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpAMD64CMOVQGTF)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETGEF cond))
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (CMOVQGEF y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETGEF {
			break
		}
		cond := v_2.Args[0]
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpAMD64CMOVQGEF)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETEQ cond))
	// cond: is32BitInt(t)
	// result: (CMOVLEQ y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETEQ {
			break
		}
		cond := v_2.Args[0]
		if !(is32BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVLEQ)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETNE cond))
	// cond: is32BitInt(t)
	// result: (CMOVLNE y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETNE {
			break
		}
		cond := v_2.Args[0]
		if !(is32BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVLNE)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETL cond))
	// cond: is32BitInt(t)
	// result: (CMOVLLT y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETL {
			break
		}
		cond := v_2.Args[0]
		if !(is32BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVLLT)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETG cond))
	// cond: is32BitInt(t)
	// result: (CMOVLGT y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETG {
			break
		}
		cond := v_2.Args[0]
		if !(is32BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVLGT)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETLE cond))
	// cond: is32BitInt(t)
	// result: (CMOVLLE y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETLE {
			break
		}
		cond := v_2.Args[0]
		if !(is32BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVLLE)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETGE cond))
	// cond: is32BitInt(t)
	// result: (CMOVLGE y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETGE {
			break
		}
		cond := v_2.Args[0]
		if !(is32BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVLGE)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETA cond))
	// cond: is32BitInt(t)
	// result: (CMOVLHI y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETA {
			break
		}
		cond := v_2.Args[0]
		if !(is32BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVLHI)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETB cond))
	// cond: is32BitInt(t)
	// result: (CMOVLCS y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETB {
			break
		}
		cond := v_2.Args[0]
		if !(is32BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVLCS)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETAE cond))
	// cond: is32BitInt(t)
	// result: (CMOVLCC y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETAE {
			break
		}
		cond := v_2.Args[0]
		if !(is32BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVLCC)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETBE cond))
	// cond: is32BitInt(t)
	// result: (CMOVLLS y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETBE {
			break
		}
		cond := v_2.Args[0]
		if !(is32BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVLLS)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETEQF cond))
	// cond: is32BitInt(t)
	// result: (CMOVLEQF y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETEQF {
			break
		}
		cond := v_2.Args[0]
		if !(is32BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVLEQF)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETNEF cond))
	// cond: is32BitInt(t)
	// result: (CMOVLNEF y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETNEF {
			break
		}
		cond := v_2.Args[0]
		if !(is32BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVLNEF)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETGF cond))
	// cond: is32BitInt(t)
	// result: (CMOVLGTF y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETGF {
			break
		}
		cond := v_2.Args[0]
		if !(is32BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVLGTF)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETGEF cond))
	// cond: is32BitInt(t)
	// result: (CMOVLGEF y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETGEF {
			break
		}
		cond := v_2.Args[0]
		if !(is32BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVLGEF)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETEQ cond))
	// cond: is16BitInt(t)
	// result: (CMOVWEQ y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETEQ {
			break
		}
		cond := v_2.Args[0]
		if !(is16BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVWEQ)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETNE cond))
	// cond: is16BitInt(t)
	// result: (CMOVWNE y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETNE {
			break
		}
		cond := v_2.Args[0]
		if !(is16BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVWNE)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETL cond))
	// cond: is16BitInt(t)
	// result: (CMOVWLT y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETL {
			break
		}
		cond := v_2.Args[0]
		if !(is16BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVWLT)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETG cond))
	// cond: is16BitInt(t)
	// result: (CMOVWGT y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETG {
			break
		}
		cond := v_2.Args[0]
		if !(is16BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVWGT)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETLE cond))
	// cond: is16BitInt(t)
	// result: (CMOVWLE y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETLE {
			break
		}
		cond := v_2.Args[0]
		if !(is16BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVWLE)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETGE cond))
	// cond: is16BitInt(t)
	// result: (CMOVWGE y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETGE {
			break
		}
		cond := v_2.Args[0]
		if !(is16BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVWGE)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETA cond))
	// cond: is16BitInt(t)
	// result: (CMOVWHI y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETA {
			break
		}
		cond := v_2.Args[0]
		if !(is16BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVWHI)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETB cond))
	// cond: is16BitInt(t)
	// result: (CMOVWCS y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETB {
			break
		}
		cond := v_2.Args[0]
		if !(is16BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVWCS)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETAE cond))
	// cond: is16BitInt(t)
	// result: (CMOVWCC y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETAE {
			break
		}
		cond := v_2.Args[0]
		if !(is16BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVWCC)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETBE cond))
	// cond: is16BitInt(t)
	// result: (CMOVWLS y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETBE {
			break
		}
		cond := v_2.Args[0]
		if !(is16BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVWLS)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETEQF cond))
	// cond: is16BitInt(t)
	// result: (CMOVWEQF y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETEQF {
			break
		}
		cond := v_2.Args[0]
		if !(is16BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVWEQF)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETNEF cond))
	// cond: is16BitInt(t)
	// result: (CMOVWNEF y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETNEF {
			break
		}
		cond := v_2.Args[0]
		if !(is16BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVWNEF)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETGF cond))
	// cond: is16BitInt(t)
	// result: (CMOVWGTF y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETGF {
			break
		}
		cond := v_2.Args[0]
		if !(is16BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVWGTF)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y (SETGEF cond))
	// cond: is16BitInt(t)
	// result: (CMOVWGEF y x cond)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if v_2.Op != OpAMD64SETGEF {
			break
		}
		cond := v_2.Args[0]
		if !(is16BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVWGEF)
		v.AddArg3(y, x, cond)
		return true
	}
	// match: (CondSelect <t> x y check)
	// cond: !check.Type.IsFlags() && check.Type.Size() == 1
	// result: (CondSelect <t> x y (MOVBQZX <typ.UInt64> check))
	for {
		t := v.Type
		x := v_0
		y := v_1
		check := v_2
		if !(!check.Type.IsFlags() && check.Type.Size() == 1) {
			break
		}
		v.reset(OpCondSelect)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBQZX, typ.UInt64)
		v0.AddArg(check)
		v.AddArg3(x, y, v0)
		return true
	}
	// match: (CondSelect <t> x y check)
	// cond: !check.Type.IsFlags() && check.Type.Size() == 2
	// result: (CondSelect <t> x y (MOVWQZX <typ.UInt64> check))
	for {
		t := v.Type
		x := v_0
		y := v_1
		check := v_2
		if !(!check.Type.IsFlags() && check.Type.Size() == 2) {
			break
		}
		v.reset(OpCondSelect)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWQZX, typ.UInt64)
		v0.AddArg(check)
		v.AddArg3(x, y, v0)
		return true
	}
	// match: (CondSelect <t> x y check)
	// cond: !check.Type.IsFlags() && check.Type.Size() == 4
	// result: (CondSelect <t> x y (MOVLQZX <typ.UInt64> check))
	for {
		t := v.Type
		x := v_0
		y := v_1
		check := v_2
		if !(!check.Type.IsFlags() && check.Type.Size() == 4) {
			break
		}
		v.reset(OpCondSelect)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLQZX, typ.UInt64)
		v0.AddArg(check)
		v.AddArg3(x, y, v0)
		return true
	}
	// match: (CondSelect <t> x y check)
	// cond: !check.Type.IsFlags() && check.Type.Size() == 8 && (is64BitInt(t) || isPtr(t))
	// result: (CMOVQNE y x (CMPQconst [0] check))
	for {
		t := v.Type
		x := v_0
		y := v_1
		check := v_2
		if !(!check.Type.IsFlags() && check.Type.Size() == 8 && (is64BitInt(t) || isPtr(t))) {
			break
		}
		v.reset(OpAMD64CMOVQNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(0)
		v0.AddArg(check)
		v.AddArg3(y, x, v0)
		return true
	}
	// match: (CondSelect <t> x y check)
	// cond: !check.Type.IsFlags() && check.Type.Size() == 8 && is32BitInt(t)
	// result: (CMOVLNE y x (CMPQconst [0] check))
	for {
		t := v.Type
		x := v_0
		y := v_1
		check := v_2
		if !(!check.Type.IsFlags() && check.Type.Size() == 8 && is32BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVLNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(0)
		v0.AddArg(check)
		v.AddArg3(y, x, v0)
		return true
	}
	// match: (CondSelect <t> x y check)
	// cond: !check.Type.IsFlags() && check.Type.Size() == 8 && is16BitInt(t)
	// result: (CMOVWNE y x (CMPQconst [0] check))
	for {
		t := v.Type
		x := v_0
		y := v_1
		check := v_2
		if !(!check.Type.IsFlags() && check.Type.Size() == 8 && is16BitInt(t)) {
			break
		}
		v.reset(OpAMD64CMOVWNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(0)
		v0.AddArg(check)
		v.AddArg3(y, x, v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpConst16(v *Value) bool {
	// match: (Const16 [c])
	// result: (MOVLconst [int32(c)])
	for {
		c := auxIntToInt16(v.AuxInt)
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(int32(c))
		return true
	}
}
func rewriteValueAMD64_OpConst8(v *Value) bool {
	// match: (Const8 [c])
	// result: (MOVLconst [int32(c)])
	for {
		c := auxIntToInt8(v.AuxInt)
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(int32(c))
		return true
	}
}
func rewriteValueAMD64_OpConstBool(v *Value) bool {
	// match: (ConstBool [c])
	// result: (MOVLconst [b2i32(c)])
	for {
		c := auxIntToBool(v.AuxInt)
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(b2i32(c))
		return true
	}
}
func rewriteValueAMD64_OpConstNil(v *Value) bool {
	// match: (ConstNil )
	// result: (MOVQconst [0])
	for {
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
}
func rewriteValueAMD64_OpCtz16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz16 x)
	// result: (BSFL (ORLconst <typ.UInt32> [1<<16] x))
	for {
		x := v_0
		v.reset(OpAMD64BSFL)
		v0 := b.NewValue0(v.Pos, OpAMD64ORLconst, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(1 << 16)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCtz16NonZero(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Ctz16NonZero x)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (TZCNTL x)
	for {
		x := v_0
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64TZCNTL)
		v.AddArg(x)
		return true
	}
	// match: (Ctz16NonZero x)
	// cond: buildcfg.GOAMD64 < 3
	// result: (BSFL x)
	for {
		x := v_0
		if !(buildcfg.GOAMD64 < 3) {
			break
		}
		v.reset(OpAMD64BSFL)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpCtz32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz32 x)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (TZCNTL x)
	for {
		x := v_0
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64TZCNTL)
		v.AddArg(x)
		return true
	}
	// match: (Ctz32 x)
	// cond: buildcfg.GOAMD64 < 3
	// result: (Select0 (BSFQ (BTSQconst <typ.UInt64> [32] x)))
	for {
		x := v_0
		if !(buildcfg.GOAMD64 < 3) {
			break
		}
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64BSFQ, types.NewTuple(typ.UInt64, types.TypeFlags))
		v1 := b.NewValue0(v.Pos, OpAMD64BTSQconst, typ.UInt64)
		v1.AuxInt = int8ToAuxInt(32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpCtz32NonZero(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Ctz32NonZero x)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (TZCNTL x)
	for {
		x := v_0
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64TZCNTL)
		v.AddArg(x)
		return true
	}
	// match: (Ctz32NonZero x)
	// cond: buildcfg.GOAMD64 < 3
	// result: (BSFL x)
	for {
		x := v_0
		if !(buildcfg.GOAMD64 < 3) {
			break
		}
		v.reset(OpAMD64BSFL)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpCtz64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz64 x)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (TZCNTQ x)
	for {
		x := v_0
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64TZCNTQ)
		v.AddArg(x)
		return true
	}
	// match: (Ctz64 <t> x)
	// cond: buildcfg.GOAMD64 < 3
	// result: (CMOVQEQ (Select0 <t> (BSFQ x)) (MOVQconst <t> [64]) (Select1 <types.TypeFlags> (BSFQ x)))
	for {
		t := v.Type
		x := v_0
		if !(buildcfg.GOAMD64 < 3) {
			break
		}
		v.reset(OpAMD64CMOVQEQ)
		v0 := b.NewValue0(v.Pos, OpSelect0, t)
		v1 := b.NewValue0(v.Pos, OpAMD64BSFQ, types.NewTuple(typ.UInt64, types.TypeFlags))
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVQconst, t)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
	return false
}
func rewriteValueAMD64_OpCtz64NonZero(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz64NonZero x)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (TZCNTQ x)
	for {
		x := v_0
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64TZCNTQ)
		v.AddArg(x)
		return true
	}
	// match: (Ctz64NonZero x)
	// cond: buildcfg.GOAMD64 < 3
	// result: (Select0 (BSFQ x))
	for {
		x := v_0
		if !(buildcfg.GOAMD64 < 3) {
			break
		}
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64BSFQ, types.NewTuple(typ.UInt64, types.TypeFlags))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpCtz8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz8 x)
	// result: (BSFL (ORLconst <typ.UInt32> [1<<8 ] x))
	for {
		x := v_0
		v.reset(OpAMD64BSFL)
		v0 := b.NewValue0(v.Pos, OpAMD64ORLconst, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(1 << 8)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCtz8NonZero(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Ctz8NonZero x)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (TZCNTL x)
	for {
		x := v_0
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64TZCNTL)
		v.AddArg(x)
		return true
	}
	// match: (Ctz8NonZero x)
	// cond: buildcfg.GOAMD64 < 3
	// result: (BSFL x)
	for {
		x := v_0
		if !(buildcfg.GOAMD64 < 3) {
			break
		}
		v.reset(OpAMD64BSFL)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpCvt16toMask16x16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Cvt16toMask16x16 <t> x)
	// result: (VPMOVMToVec16x16 <types.TypeVec256> (KMOVWk <t> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64VPMOVMToVec16x16)
		v.Type = types.TypeVec256
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVWk, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvt16toMask32x16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Cvt16toMask32x16 <t> x)
	// result: (VPMOVMToVec32x16 <types.TypeVec512> (KMOVWk <t> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64VPMOVMToVec32x16)
		v.Type = types.TypeVec512
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVWk, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvt16toMask8x16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Cvt16toMask8x16 <t> x)
	// result: (VPMOVMToVec8x16 <types.TypeVec128> (KMOVWk <t> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64VPMOVMToVec8x16)
		v.Type = types.TypeVec128
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVWk, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvt32toMask16x32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Cvt32toMask16x32 <t> x)
	// result: (VPMOVMToVec16x32 <types.TypeVec512> (KMOVDk <t> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64VPMOVMToVec16x32)
		v.Type = types.TypeVec512
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVDk, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvt32toMask8x32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Cvt32toMask8x32 <t> x)
	// result: (VPMOVMToVec8x32 <types.TypeVec256> (KMOVDk <t> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64VPMOVMToVec8x32)
		v.Type = types.TypeVec256
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVDk, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvt64toMask8x64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Cvt64toMask8x64 <t> x)
	// result: (VPMOVMToVec8x64 <types.TypeVec512> (KMOVQk <t> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64VPMOVMToVec8x64)
		v.Type = types.TypeVec512
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVQk, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvt8toMask16x8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Cvt8toMask16x8 <t> x)
	// result: (VPMOVMToVec16x8 <types.TypeVec128> (KMOVBk <t> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64VPMOVMToVec16x8)
		v.Type = types.TypeVec128
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVBk, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvt8toMask32x4(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Cvt8toMask32x4 <t> x)
	// result: (VPMOVMToVec32x4 <types.TypeVec128> (KMOVBk <t> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64VPMOVMToVec32x4)
		v.Type = types.TypeVec128
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVBk, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvt8toMask32x8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Cvt8toMask32x8 <t> x)
	// result: (VPMOVMToVec32x8 <types.TypeVec256> (KMOVBk <t> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64VPMOVMToVec32x8)
		v.Type = types.TypeVec256
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVBk, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvt8toMask64x2(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Cvt8toMask64x2 <t> x)
	// result: (VPMOVMToVec64x2 <types.TypeVec128> (KMOVBk <t> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64VPMOVMToVec64x2)
		v.Type = types.TypeVec128
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVBk, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvt8toMask64x4(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Cvt8toMask64x4 <t> x)
	// result: (VPMOVMToVec64x4 <types.TypeVec256> (KMOVBk <t> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64VPMOVMToVec64x4)
		v.Type = types.TypeVec256
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVBk, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvt8toMask64x8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Cvt8toMask64x8 <t> x)
	// result: (VPMOVMToVec64x8 <types.TypeVec512> (KMOVBk <t> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64VPMOVMToVec64x8)
		v.Type = types.TypeVec512
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVBk, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvtMask16x16to16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CvtMask16x16to16 <t> x)
	// result: (KMOVWi <t> (VPMOVVec16x16ToM <types.TypeMask> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64KMOVWi)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x16ToM, types.TypeMask)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvtMask16x32to32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CvtMask16x32to32 <t> x)
	// result: (KMOVDi <t> (VPMOVVec16x32ToM <types.TypeMask> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64KMOVDi)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x32ToM, types.TypeMask)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvtMask16x8to8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CvtMask16x8to8 <t> x)
	// result: (KMOVBi <t> (VPMOVVec16x8ToM <types.TypeMask> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64KMOVBi)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x8ToM, types.TypeMask)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvtMask32x16to16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CvtMask32x16to16 <t> x)
	// result: (KMOVWi <t> (VPMOVVec32x16ToM <types.TypeMask> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64KMOVWi)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x16ToM, types.TypeMask)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvtMask32x4to8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CvtMask32x4to8 <t> x)
	// result: (KMOVBi <t> (VPMOVVec32x4ToM <types.TypeMask> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64KMOVBi)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x4ToM, types.TypeMask)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvtMask32x8to8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CvtMask32x8to8 <t> x)
	// result: (KMOVBi <t> (VPMOVVec32x8ToM <types.TypeMask> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64KMOVBi)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x8ToM, types.TypeMask)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvtMask64x2to8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CvtMask64x2to8 <t> x)
	// result: (KMOVBi <t> (VPMOVVec64x2ToM <types.TypeMask> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64KMOVBi)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x2ToM, types.TypeMask)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvtMask64x4to8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CvtMask64x4to8 <t> x)
	// result: (KMOVBi <t> (VPMOVVec64x4ToM <types.TypeMask> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64KMOVBi)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x4ToM, types.TypeMask)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvtMask64x8to8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CvtMask64x8to8 <t> x)
	// result: (KMOVBi <t> (VPMOVVec64x8ToM <types.TypeMask> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64KMOVBi)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x8ToM, types.TypeMask)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvtMask8x16to16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CvtMask8x16to16 <t> x)
	// result: (KMOVWi <t> (VPMOVVec8x16ToM <types.TypeMask> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64KMOVWi)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x16ToM, types.TypeMask)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvtMask8x32to32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CvtMask8x32to32 <t> x)
	// result: (KMOVDi <t> (VPMOVVec8x32ToM <types.TypeMask> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64KMOVDi)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x32ToM, types.TypeMask)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCvtMask8x64to64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (CvtMask8x64to64 <t> x)
	// result: (KMOVQi <t> (VPMOVVec8x64ToM <types.TypeMask> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64KMOVQi)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x64ToM, types.TypeMask)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpDiv16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 [a] x y)
	// result: (Select0 (DIVW [a] x y))
	for {
		a := auxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVW, types.NewTuple(typ.Int16, typ.Int16))
		v0.AuxInt = boolToAuxInt(a)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpDiv16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16u x y)
	// result: (Select0 (DIVWU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVWU, types.NewTuple(typ.UInt16, typ.UInt16))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpDiv32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div32 [a] x y)
	// result: (Select0 (DIVL [a] x y))
	for {
		a := auxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVL, types.NewTuple(typ.Int32, typ.Int32))
		v0.AuxInt = boolToAuxInt(a)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpDiv32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div32u x y)
	// result: (Select0 (DIVLU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVLU, types.NewTuple(typ.UInt32, typ.UInt32))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpDiv64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div64 [a] x y)
	// result: (Select0 (DIVQ [a] x y))
	for {
		a := auxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVQ, types.NewTuple(typ.Int64, typ.Int64))
		v0.AuxInt = boolToAuxInt(a)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpDiv64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div64u x y)
	// result: (Select0 (DIVQU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVQU, types.NewTuple(typ.UInt64, typ.UInt64))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpDiv8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// result: (Select0 (DIVW (SignExt8to16 x) (SignExt8to16 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVW, types.NewTuple(typ.Int16, typ.Int16))
		v1 := b.NewValue0(v.Pos, OpSignExt8to16, typ.Int16)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpSignExt8to16, typ.Int16)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpDiv8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// result: (Select0 (DIVWU (ZeroExt8to16 x) (ZeroExt8to16 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVWU, types.NewTuple(typ.UInt16, typ.UInt16))
		v1 := b.NewValue0(v.Pos, OpZeroExt8to16, typ.UInt16)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to16, typ.UInt16)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq16 x y)
	// result: (SETEQ (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETEQ)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32 x y)
	// result: (SETEQ (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETEQ)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32F x y)
	// result: (SETEQF (UCOMISS x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETEQF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISS, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64 x y)
	// result: (SETEQ (CMPQ x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETEQ)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64F x y)
	// result: (SETEQF (UCOMISD x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETEQF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq8 x y)
	// result: (SETEQ (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETEQ)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEqB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (EqB x y)
	// result: (SETEQ (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETEQ)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (EqPtr x y)
	// result: (SETEQ (CMPQ x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETEQ)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEqualFloat32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqualFloat32x16 x y)
	// result: (VPMOVMToVec32x16 (VCMPPS512 [0] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VCMPPS512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(0)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEqualFloat32x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (EqualFloat32x4 x y)
	// result: (VCMPPS128 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPS128)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpEqualFloat32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (EqualFloat32x8 x y)
	// result: (VCMPPS256 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPS256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpEqualFloat64x2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (EqualFloat64x2 x y)
	// result: (VCMPPD128 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPD128)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpEqualFloat64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (EqualFloat64x4 x y)
	// result: (VCMPPD256 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPD256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpEqualFloat64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqualFloat64x8 x y)
	// result: (VPMOVMToVec64x8 (VCMPPD512 [0] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VCMPPD512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(0)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEqualInt16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqualInt16x32 x y)
	// result: (VPMOVMToVec16x32 (VPCMPEQW512 x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec16x32)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPEQW512, typ.Mask)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEqualInt32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqualInt32x16 x y)
	// result: (VPMOVMToVec32x16 (VPCMPEQD512 x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPEQD512, typ.Mask)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEqualInt64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqualInt64x8 x y)
	// result: (VPMOVMToVec64x8 (VPCMPEQQ512 x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPEQQ512, typ.Mask)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEqualInt8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqualInt8x64 x y)
	// result: (VPMOVMToVec8x64 (VPCMPEQB512 x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec8x64)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPEQB512, typ.Mask)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEqualUint16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqualUint16x32 x y)
	// result: (VPMOVMToVec16x32 (VPCMPEQW512 x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec16x32)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPEQW512, typ.Mask)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEqualUint32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqualUint32x16 x y)
	// result: (VPMOVMToVec32x16 (VPCMPEQD512 x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPEQD512, typ.Mask)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEqualUint64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqualUint64x8 x y)
	// result: (VPMOVMToVec64x8 (VPCMPEQQ512 x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPEQQ512, typ.Mask)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEqualUint8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqualUint8x64 x y)
	// result: (VPMOVMToVec8x64 (VPCMPEQB512 x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec8x64)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPEQB512, typ.Mask)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandFloat32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandFloat32x16 x mask)
	// result: (VEXPANDPSMasked512 x (VPMOVVec32x16ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VEXPANDPSMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandFloat32x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandFloat32x4 x mask)
	// result: (VEXPANDPSMasked128 x (VPMOVVec32x4ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VEXPANDPSMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x4ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandFloat32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandFloat32x8 x mask)
	// result: (VEXPANDPSMasked256 x (VPMOVVec32x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VEXPANDPSMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandFloat64x2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandFloat64x2 x mask)
	// result: (VEXPANDPDMasked128 x (VPMOVVec64x2ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VEXPANDPDMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x2ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandFloat64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandFloat64x4 x mask)
	// result: (VEXPANDPDMasked256 x (VPMOVVec64x4ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VEXPANDPDMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x4ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandFloat64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandFloat64x8 x mask)
	// result: (VEXPANDPDMasked512 x (VPMOVVec64x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VEXPANDPDMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandInt16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandInt16x16 x mask)
	// result: (VPEXPANDWMasked256 x (VPMOVVec16x16ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDWMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandInt16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandInt16x32 x mask)
	// result: (VPEXPANDWMasked512 x (VPMOVVec16x32ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDWMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x32ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandInt16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandInt16x8 x mask)
	// result: (VPEXPANDWMasked128 x (VPMOVVec16x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDWMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandInt32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandInt32x16 x mask)
	// result: (VPEXPANDDMasked512 x (VPMOVVec32x16ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDDMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandInt32x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandInt32x4 x mask)
	// result: (VPEXPANDDMasked128 x (VPMOVVec32x4ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDDMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x4ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandInt32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandInt32x8 x mask)
	// result: (VPEXPANDDMasked256 x (VPMOVVec32x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDDMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandInt64x2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandInt64x2 x mask)
	// result: (VPEXPANDQMasked128 x (VPMOVVec64x2ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDQMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x2ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandInt64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandInt64x4 x mask)
	// result: (VPEXPANDQMasked256 x (VPMOVVec64x4ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDQMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x4ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandInt64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandInt64x8 x mask)
	// result: (VPEXPANDQMasked512 x (VPMOVVec64x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDQMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandInt8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandInt8x16 x mask)
	// result: (VPEXPANDBMasked128 x (VPMOVVec8x16ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDBMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandInt8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandInt8x32 x mask)
	// result: (VPEXPANDBMasked256 x (VPMOVVec8x32ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDBMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x32ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandInt8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandInt8x64 x mask)
	// result: (VPEXPANDBMasked512 x (VPMOVVec8x64ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDBMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x64ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandUint16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandUint16x16 x mask)
	// result: (VPEXPANDWMasked256 x (VPMOVVec16x16ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDWMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandUint16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandUint16x32 x mask)
	// result: (VPEXPANDWMasked512 x (VPMOVVec16x32ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDWMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x32ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandUint16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandUint16x8 x mask)
	// result: (VPEXPANDWMasked128 x (VPMOVVec16x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDWMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandUint32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandUint32x16 x mask)
	// result: (VPEXPANDDMasked512 x (VPMOVVec32x16ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDDMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandUint32x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandUint32x4 x mask)
	// result: (VPEXPANDDMasked128 x (VPMOVVec32x4ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDDMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x4ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandUint32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandUint32x8 x mask)
	// result: (VPEXPANDDMasked256 x (VPMOVVec32x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDDMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandUint64x2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandUint64x2 x mask)
	// result: (VPEXPANDQMasked128 x (VPMOVVec64x2ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDQMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x2ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandUint64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandUint64x4 x mask)
	// result: (VPEXPANDQMasked256 x (VPMOVVec64x4ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDQMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x4ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandUint64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandUint64x8 x mask)
	// result: (VPEXPANDQMasked512 x (VPMOVVec64x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDQMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandUint8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandUint8x16 x mask)
	// result: (VPEXPANDBMasked128 x (VPMOVVec8x16ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDBMasked128)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandUint8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandUint8x32 x mask)
	// result: (VPEXPANDBMasked256 x (VPMOVVec8x32ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDBMasked256)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x32ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpExpandUint8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ExpandUint8x64 x mask)
	// result: (VPEXPANDBMasked512 x (VPMOVVec8x64ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VPEXPANDBMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x64ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpFMA(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMA x y z)
	// result: (VFMADD231SD z x y)
	for {
		x := v_0
		y := v_1
		z := v_2
		v.reset(OpAMD64VFMADD231SD)
		v.AddArg3(z, x, y)
		return true
	}
}
func rewriteValueAMD64_OpFloor(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Floor x)
	// result: (ROUNDSD [1] x)
	for {
		x := v_0
		v.reset(OpAMD64ROUNDSD)
		v.AuxInt = int8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpFloorFloat32x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FloorFloat32x4 x)
	// result: (VROUNDPS128 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VROUNDPS128)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpFloorFloat32x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FloorFloat32x8 x)
	// result: (VROUNDPS256 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VROUNDPS256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpFloorFloat64x2(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FloorFloat64x2 x)
	// result: (VROUNDPD128 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VROUNDPD128)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpFloorFloat64x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FloorFloat64x4 x)
	// result: (VROUNDPD256 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VROUNDPD256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpFloorScaledFloat32x16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FloorScaledFloat32x16 [a] x)
	// result: (VRNDSCALEPS512 [a+1] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPS512)
		v.AuxInt = uint8ToAuxInt(a + 1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpFloorScaledFloat32x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FloorScaledFloat32x4 [a] x)
	// result: (VRNDSCALEPS128 [a+1] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPS128)
		v.AuxInt = uint8ToAuxInt(a + 1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpFloorScaledFloat32x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FloorScaledFloat32x8 [a] x)
	// result: (VRNDSCALEPS256 [a+1] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPS256)
		v.AuxInt = uint8ToAuxInt(a + 1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpFloorScaledFloat64x2(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FloorScaledFloat64x2 [a] x)
	// result: (VRNDSCALEPD128 [a+1] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPD128)
		v.AuxInt = uint8ToAuxInt(a + 1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpFloorScaledFloat64x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FloorScaledFloat64x4 [a] x)
	// result: (VRNDSCALEPD256 [a+1] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPD256)
		v.AuxInt = uint8ToAuxInt(a + 1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpFloorScaledFloat64x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FloorScaledFloat64x8 [a] x)
	// result: (VRNDSCALEPD512 [a+1] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPD512)
		v.AuxInt = uint8ToAuxInt(a + 1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpFloorScaledResidueFloat32x16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FloorScaledResidueFloat32x16 [a] x)
	// result: (VREDUCEPS512 [a+1] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPS512)
		v.AuxInt = uint8ToAuxInt(a + 1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpFloorScaledResidueFloat32x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FloorScaledResidueFloat32x4 [a] x)
	// result: (VREDUCEPS128 [a+1] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPS128)
		v.AuxInt = uint8ToAuxInt(a + 1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpFloorScaledResidueFloat32x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FloorScaledResidueFloat32x8 [a] x)
	// result: (VREDUCEPS256 [a+1] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPS256)
		v.AuxInt = uint8ToAuxInt(a + 1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpFloorScaledResidueFloat64x2(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FloorScaledResidueFloat64x2 [a] x)
	// result: (VREDUCEPD128 [a+1] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPD128)
		v.AuxInt = uint8ToAuxInt(a + 1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpFloorScaledResidueFloat64x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FloorScaledResidueFloat64x4 [a] x)
	// result: (VREDUCEPD256 [a+1] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPD256)
		v.AuxInt = uint8ToAuxInt(a + 1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpFloorScaledResidueFloat64x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FloorScaledResidueFloat64x8 [a] x)
	// result: (VREDUCEPD512 [a+1] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPD512)
		v.AuxInt = uint8ToAuxInt(a + 1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetG(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetG mem)
	// cond: v.Block.Func.OwnAux.Fn.ABI() != obj.ABIInternal
	// result: (LoweredGetG mem)
	for {
		mem := v_0
		if !(v.Block.Func.OwnAux.Fn.ABI() != obj.ABIInternal) {
			break
		}
		v.reset(OpAMD64LoweredGetG)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpGetHiFloat32x16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiFloat32x16 x)
	// result: (VEXTRACTF64X4256 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTF64X4256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiFloat32x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiFloat32x8 x)
	// result: (VEXTRACTF128128 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTF128128)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiFloat64x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiFloat64x4 x)
	// result: (VEXTRACTF128128 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTF128128)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiFloat64x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiFloat64x8 x)
	// result: (VEXTRACTF64X4256 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTF64X4256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiInt16x16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiInt16x16 x)
	// result: (VEXTRACTI128128 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI128128)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiInt16x32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiInt16x32 x)
	// result: (VEXTRACTI64X4256 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI64X4256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiInt32x16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiInt32x16 x)
	// result: (VEXTRACTI64X4256 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI64X4256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiInt32x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiInt32x8 x)
	// result: (VEXTRACTI128128 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI128128)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiInt64x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiInt64x4 x)
	// result: (VEXTRACTI128128 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI128128)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiInt64x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiInt64x8 x)
	// result: (VEXTRACTI64X4256 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI64X4256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiInt8x32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiInt8x32 x)
	// result: (VEXTRACTI128128 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI128128)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiInt8x64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiInt8x64 x)
	// result: (VEXTRACTI64X4256 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI64X4256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiUint16x16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiUint16x16 x)
	// result: (VEXTRACTI128128 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI128128)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiUint16x32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiUint16x32 x)
	// result: (VEXTRACTI64X4256 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI64X4256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiUint32x16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiUint32x16 x)
	// result: (VEXTRACTI64X4256 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI64X4256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiUint32x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiUint32x8 x)
	// result: (VEXTRACTI128128 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI128128)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiUint64x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiUint64x4 x)
	// result: (VEXTRACTI128128 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI128128)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiUint64x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiUint64x8 x)
	// result: (VEXTRACTI64X4256 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI64X4256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiUint8x32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiUint8x32 x)
	// result: (VEXTRACTI128128 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI128128)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetHiUint8x64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetHiUint8x64 x)
	// result: (VEXTRACTI64X4256 [1] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI64X4256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoFloat32x16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoFloat32x16 x)
	// result: (VEXTRACTF64X4256 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTF64X4256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoFloat32x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoFloat32x8 x)
	// result: (VEXTRACTF128128 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTF128128)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoFloat64x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoFloat64x4 x)
	// result: (VEXTRACTF128128 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTF128128)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoFloat64x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoFloat64x8 x)
	// result: (VEXTRACTF64X4256 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTF64X4256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoInt16x16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoInt16x16 x)
	// result: (VEXTRACTI128128 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI128128)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoInt16x32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoInt16x32 x)
	// result: (VEXTRACTI64X4256 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI64X4256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoInt32x16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoInt32x16 x)
	// result: (VEXTRACTI64X4256 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI64X4256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoInt32x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoInt32x8 x)
	// result: (VEXTRACTI128128 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI128128)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoInt64x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoInt64x4 x)
	// result: (VEXTRACTI128128 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI128128)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoInt64x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoInt64x8 x)
	// result: (VEXTRACTI64X4256 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI64X4256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoInt8x32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoInt8x32 x)
	// result: (VEXTRACTI128128 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI128128)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoInt8x64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoInt8x64 x)
	// result: (VEXTRACTI64X4256 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI64X4256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoUint16x16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoUint16x16 x)
	// result: (VEXTRACTI128128 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI128128)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoUint16x32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoUint16x32 x)
	// result: (VEXTRACTI64X4256 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI64X4256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoUint32x16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoUint32x16 x)
	// result: (VEXTRACTI64X4256 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI64X4256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoUint32x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoUint32x8 x)
	// result: (VEXTRACTI128128 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI128128)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoUint64x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoUint64x4 x)
	// result: (VEXTRACTI128128 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI128128)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoUint64x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoUint64x8 x)
	// result: (VEXTRACTI64X4256 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI64X4256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoUint8x32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoUint8x32 x)
	// result: (VEXTRACTI128128 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI128128)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGetLoUint8x64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GetLoUint8x64 x)
	// result: (VEXTRACTI64X4256 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VEXTRACTI64X4256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpGreaterEqualFloat32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterEqualFloat32x16 x y)
	// result: (VPMOVMToVec32x16 (VCMPPS512 [13] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VCMPPS512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(13)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterEqualFloat32x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (GreaterEqualFloat32x4 x y)
	// result: (VCMPPS128 [13] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPS128)
		v.AuxInt = uint8ToAuxInt(13)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpGreaterEqualFloat32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (GreaterEqualFloat32x8 x y)
	// result: (VCMPPS256 [13] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPS256)
		v.AuxInt = uint8ToAuxInt(13)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpGreaterEqualFloat64x2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (GreaterEqualFloat64x2 x y)
	// result: (VCMPPD128 [13] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPD128)
		v.AuxInt = uint8ToAuxInt(13)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpGreaterEqualFloat64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (GreaterEqualFloat64x4 x y)
	// result: (VCMPPD256 [13] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPD256)
		v.AuxInt = uint8ToAuxInt(13)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpGreaterEqualFloat64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterEqualFloat64x8 x y)
	// result: (VPMOVMToVec64x8 (VCMPPD512 [13] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VCMPPD512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(13)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterEqualInt16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterEqualInt16x32 x y)
	// result: (VPMOVMToVec16x32 (VPCMPW512 [13] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec16x32)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPW512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(13)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterEqualInt32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterEqualInt32x16 x y)
	// result: (VPMOVMToVec32x16 (VPCMPD512 [13] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPD512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(13)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterEqualInt64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterEqualInt64x8 x y)
	// result: (VPMOVMToVec64x8 (VPCMPQ512 [13] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPQ512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(13)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterEqualInt8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterEqualInt8x64 x y)
	// result: (VPMOVMToVec8x64 (VPCMPB512 [13] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec8x64)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPB512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(13)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterEqualUint16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterEqualUint16x32 x y)
	// result: (VPMOVMToVec16x32 (VPCMPUW512 [13] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec16x32)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUW512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(13)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterEqualUint32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterEqualUint32x16 x y)
	// result: (VPMOVMToVec32x16 (VPCMPUD512 [13] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUD512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(13)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterEqualUint64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterEqualUint64x8 x y)
	// result: (VPMOVMToVec64x8 (VPCMPUQ512 [13] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUQ512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(13)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterEqualUint8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterEqualUint8x64 x y)
	// result: (VPMOVMToVec8x64 (VPCMPUB512 [13] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec8x64)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUB512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(13)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterFloat32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterFloat32x16 x y)
	// result: (VPMOVMToVec32x16 (VCMPPS512 [14] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VCMPPS512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(14)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterFloat32x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (GreaterFloat32x4 x y)
	// result: (VCMPPS128 [14] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPS128)
		v.AuxInt = uint8ToAuxInt(14)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpGreaterFloat32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (GreaterFloat32x8 x y)
	// result: (VCMPPS256 [14] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPS256)
		v.AuxInt = uint8ToAuxInt(14)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpGreaterFloat64x2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (GreaterFloat64x2 x y)
	// result: (VCMPPD128 [14] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPD128)
		v.AuxInt = uint8ToAuxInt(14)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpGreaterFloat64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (GreaterFloat64x4 x y)
	// result: (VCMPPD256 [14] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPD256)
		v.AuxInt = uint8ToAuxInt(14)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpGreaterFloat64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterFloat64x8 x y)
	// result: (VPMOVMToVec64x8 (VCMPPD512 [14] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VCMPPD512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(14)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterInt16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterInt16x32 x y)
	// result: (VPMOVMToVec16x32 (VPCMPGTW512 x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec16x32)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPGTW512, typ.Mask)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterInt32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterInt32x16 x y)
	// result: (VPMOVMToVec32x16 (VPCMPGTD512 x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPGTD512, typ.Mask)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterInt64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterInt64x8 x y)
	// result: (VPMOVMToVec64x8 (VPCMPGTQ512 x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPGTQ512, typ.Mask)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterInt8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterInt8x64 x y)
	// result: (VPMOVMToVec8x64 (VPCMPGTB512 x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec8x64)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPGTB512, typ.Mask)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterUint16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterUint16x32 x y)
	// result: (VPMOVMToVec16x32 (VPCMPUW512 [14] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec16x32)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUW512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(14)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterUint32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterUint32x16 x y)
	// result: (VPMOVMToVec32x16 (VPCMPUD512 [14] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUD512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(14)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterUint64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterUint64x8 x y)
	// result: (VPMOVMToVec64x8 (VPCMPUQ512 [14] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUQ512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(14)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreaterUint8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterUint8x64 x y)
	// result: (VPMOVMToVec8x64 (VPCMPUB512 [14] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec8x64)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUB512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(14)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpHasCPUFeature(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (HasCPUFeature {s})
	// result: (SETNE (CMPLconst [0] (LoweredHasCPUFeature {s})))
	for {
		s := auxToSym(v.Aux)
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpAMD64LoweredHasCPUFeature, typ.UInt64)
		v1.Aux = symToAux(s)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpIsInBounds(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsInBounds idx len)
	// result: (SETB (CMPQ idx len))
	for {
		idx := v_0
		len := v_1
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg2(idx, len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpIsNanFloat32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (IsNanFloat32x16 x y)
	// result: (VPMOVMToVec32x16 (VCMPPS512 [3] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VCMPPS512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(3)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpIsNanFloat32x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IsNanFloat32x4 x y)
	// result: (VCMPPS128 [3] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPS128)
		v.AuxInt = uint8ToAuxInt(3)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpIsNanFloat32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IsNanFloat32x8 x y)
	// result: (VCMPPS256 [3] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPS256)
		v.AuxInt = uint8ToAuxInt(3)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpIsNanFloat64x2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IsNanFloat64x2 x y)
	// result: (VCMPPD128 [3] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPD128)
		v.AuxInt = uint8ToAuxInt(3)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpIsNanFloat64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IsNanFloat64x4 x y)
	// result: (VCMPPD256 [3] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPD256)
		v.AuxInt = uint8ToAuxInt(3)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpIsNanFloat64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (IsNanFloat64x8 x y)
	// result: (VPMOVMToVec64x8 (VCMPPD512 [3] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VCMPPD512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(3)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpIsNonNil(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsNonNil p)
	// result: (SETNE (TESTQ p p))
	for {
		p := v_0
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64TESTQ, types.TypeFlags)
		v0.AddArg2(p, p)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpIsSliceInBounds(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsSliceInBounds idx len)
	// result: (SETBE (CMPQ idx len))
	for {
		idx := v_0
		len := v_1
		v.reset(OpAMD64SETBE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg2(idx, len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq16 x y)
	// result: (SETLE (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETLE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq16U x y)
	// result: (SETBE (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETBE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32 x y)
	// result: (SETLE (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETLE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32F x y)
	// result: (SETGEF (UCOMISS y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETGEF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISS, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32U x y)
	// result: (SETBE (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETBE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64 x y)
	// result: (SETLE (CMPQ x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETLE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64F x y)
	// result: (SETGEF (UCOMISD y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETGEF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISD, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64U x y)
	// result: (SETBE (CMPQ x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETBE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq8 x y)
	// result: (SETLE (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETLE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq8U x y)
	// result: (SETBE (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETBE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less16 x y)
	// result: (SETL (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETL)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less16U x y)
	// result: (SETB (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32 x y)
	// result: (SETL (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETL)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32F x y)
	// result: (SETGF (UCOMISS y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETGF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISS, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32U x y)
	// result: (SETB (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64 x y)
	// result: (SETL (CMPQ x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETL)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64F x y)
	// result: (SETGF (UCOMISD y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETGF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISD, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64U x y)
	// result: (SETB (CMPQ x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less8 x y)
	// result: (SETL (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETL)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less8U x y)
	// result: (SETB (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessEqualFloat32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessEqualFloat32x16 x y)
	// result: (VPMOVMToVec32x16 (VCMPPS512 [2] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VCMPPS512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(2)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessEqualFloat32x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LessEqualFloat32x4 x y)
	// result: (VCMPPS128 [2] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPS128)
		v.AuxInt = uint8ToAuxInt(2)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpLessEqualFloat32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LessEqualFloat32x8 x y)
	// result: (VCMPPS256 [2] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPS256)
		v.AuxInt = uint8ToAuxInt(2)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpLessEqualFloat64x2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LessEqualFloat64x2 x y)
	// result: (VCMPPD128 [2] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPD128)
		v.AuxInt = uint8ToAuxInt(2)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpLessEqualFloat64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LessEqualFloat64x4 x y)
	// result: (VCMPPD256 [2] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPD256)
		v.AuxInt = uint8ToAuxInt(2)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpLessEqualFloat64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessEqualFloat64x8 x y)
	// result: (VPMOVMToVec64x8 (VCMPPD512 [2] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VCMPPD512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(2)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessEqualInt16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessEqualInt16x32 x y)
	// result: (VPMOVMToVec16x32 (VPCMPW512 [2] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec16x32)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPW512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(2)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessEqualInt32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessEqualInt32x16 x y)
	// result: (VPMOVMToVec32x16 (VPCMPD512 [2] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPD512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(2)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessEqualInt64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessEqualInt64x8 x y)
	// result: (VPMOVMToVec64x8 (VPCMPQ512 [2] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPQ512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(2)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessEqualInt8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessEqualInt8x64 x y)
	// result: (VPMOVMToVec8x64 (VPCMPB512 [2] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec8x64)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPB512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(2)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessEqualUint16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessEqualUint16x32 x y)
	// result: (VPMOVMToVec16x32 (VPCMPUW512 [2] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec16x32)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUW512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(2)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessEqualUint32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessEqualUint32x16 x y)
	// result: (VPMOVMToVec32x16 (VPCMPUD512 [2] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUD512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(2)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessEqualUint64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessEqualUint64x8 x y)
	// result: (VPMOVMToVec64x8 (VPCMPUQ512 [2] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUQ512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(2)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessEqualUint8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessEqualUint8x64 x y)
	// result: (VPMOVMToVec8x64 (VPCMPUB512 [2] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec8x64)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUB512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(2)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessFloat32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessFloat32x16 x y)
	// result: (VPMOVMToVec32x16 (VCMPPS512 [1] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VCMPPS512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(1)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessFloat32x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LessFloat32x4 x y)
	// result: (VCMPPS128 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPS128)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpLessFloat32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LessFloat32x8 x y)
	// result: (VCMPPS256 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPS256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpLessFloat64x2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LessFloat64x2 x y)
	// result: (VCMPPD128 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPD128)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpLessFloat64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LessFloat64x4 x y)
	// result: (VCMPPD256 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPD256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpLessFloat64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessFloat64x8 x y)
	// result: (VPMOVMToVec64x8 (VCMPPD512 [1] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VCMPPD512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(1)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessInt16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessInt16x32 x y)
	// result: (VPMOVMToVec16x32 (VPCMPW512 [1] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec16x32)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPW512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(1)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessInt32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessInt32x16 x y)
	// result: (VPMOVMToVec32x16 (VPCMPD512 [1] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPD512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(1)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessInt64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessInt64x8 x y)
	// result: (VPMOVMToVec64x8 (VPCMPQ512 [1] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPQ512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(1)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessInt8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessInt8x64 x y)
	// result: (VPMOVMToVec8x64 (VPCMPB512 [1] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec8x64)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPB512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(1)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessUint16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessUint16x32 x y)
	// result: (VPMOVMToVec16x32 (VPCMPUW512 [1] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec16x32)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUW512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(1)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessUint32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessUint32x16 x y)
	// result: (VPMOVMToVec32x16 (VPCMPUD512 [1] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUD512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(1)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessUint64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessUint64x8 x y)
	// result: (VPMOVMToVec64x8 (VPCMPUQ512 [1] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUQ512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(1)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLessUint8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessUint8x64 x y)
	// result: (VPMOVMToVec8x64 (VPCMPUB512 [1] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec8x64)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUB512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(1)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLoad(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Load <t> ptr mem)
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (MOVQload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpAMD64MOVQload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is32BitInt(t)
	// result: (MOVLload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is32BitInt(t)) {
			break
		}
		v.reset(OpAMD64MOVLload)
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
		v.reset(OpAMD64MOVWload)
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
		v.reset(OpAMD64MOVBload)
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
		v.reset(OpAMD64MOVSSload)
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
		v.reset(OpAMD64MOVSDload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 16
	// result: (VMOVDQUload128 ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 16) {
			break
		}
		v.reset(OpAMD64VMOVDQUload128)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 32
	// result: (VMOVDQUload256 ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 32) {
			break
		}
		v.reset(OpAMD64VMOVDQUload256)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 64
	// result: (VMOVDQUload512 ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 64) {
			break
		}
		v.reset(OpAMD64VMOVDQUload512)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLoadMask16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LoadMask16x16 <t> ptr mem)
	// result: (VPMOVMToVec16x16 <types.TypeVec256> (KMOVQload <t> ptr mem))
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		v.reset(OpAMD64VPMOVMToVec16x16)
		v.Type = types.TypeVec256
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVQload, t)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLoadMask16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LoadMask16x32 <t> ptr mem)
	// result: (VPMOVMToVec16x32 <types.TypeVec512> (KMOVQload <t> ptr mem))
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		v.reset(OpAMD64VPMOVMToVec16x32)
		v.Type = types.TypeVec512
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVQload, t)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLoadMask16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LoadMask16x8 <t> ptr mem)
	// result: (VPMOVMToVec16x8 <types.TypeVec128> (KMOVQload <t> ptr mem))
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		v.reset(OpAMD64VPMOVMToVec16x8)
		v.Type = types.TypeVec128
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVQload, t)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLoadMask32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LoadMask32x16 <t> ptr mem)
	// result: (VPMOVMToVec32x16 <types.TypeVec512> (KMOVQload <t> ptr mem))
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v.Type = types.TypeVec512
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVQload, t)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLoadMask32x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LoadMask32x4 <t> ptr mem)
	// result: (VPMOVMToVec32x4 <types.TypeVec128> (KMOVQload <t> ptr mem))
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		v.reset(OpAMD64VPMOVMToVec32x4)
		v.Type = types.TypeVec128
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVQload, t)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLoadMask32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LoadMask32x8 <t> ptr mem)
	// result: (VPMOVMToVec32x8 <types.TypeVec256> (KMOVQload <t> ptr mem))
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		v.reset(OpAMD64VPMOVMToVec32x8)
		v.Type = types.TypeVec256
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVQload, t)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLoadMask64x2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LoadMask64x2 <t> ptr mem)
	// result: (VPMOVMToVec64x2 <types.TypeVec128> (KMOVQload <t> ptr mem))
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		v.reset(OpAMD64VPMOVMToVec64x2)
		v.Type = types.TypeVec128
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVQload, t)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLoadMask64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LoadMask64x4 <t> ptr mem)
	// result: (VPMOVMToVec64x4 <types.TypeVec256> (KMOVQload <t> ptr mem))
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		v.reset(OpAMD64VPMOVMToVec64x4)
		v.Type = types.TypeVec256
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVQload, t)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLoadMask64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LoadMask64x8 <t> ptr mem)
	// result: (VPMOVMToVec64x8 <types.TypeVec512> (KMOVQload <t> ptr mem))
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v.Type = types.TypeVec512
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVQload, t)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLoadMask8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LoadMask8x16 <t> ptr mem)
	// result: (VPMOVMToVec8x16 <types.TypeVec128> (KMOVQload <t> ptr mem))
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		v.reset(OpAMD64VPMOVMToVec8x16)
		v.Type = types.TypeVec128
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVQload, t)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLoadMask8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LoadMask8x32 <t> ptr mem)
	// result: (VPMOVMToVec8x32 <types.TypeVec256> (KMOVQload <t> ptr mem))
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		v.reset(OpAMD64VPMOVMToVec8x32)
		v.Type = types.TypeVec256
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVQload, t)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLoadMask8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LoadMask8x64 <t> ptr mem)
	// result: (VPMOVMToVec8x64 <types.TypeVec512> (KMOVQload <t> ptr mem))
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		v.reset(OpAMD64VPMOVMToVec8x64)
		v.Type = types.TypeVec512
		v0 := b.NewValue0(v.Pos, OpAMD64KMOVQload, t)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLoadMasked16(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LoadMasked16 <t> ptr mask mem)
	// cond: t.Size() == 64
	// result: (VPMASK16load512 ptr (VPMOVVec16x32ToM <types.TypeMask> mask) mem)
	for {
		t := v.Type
		ptr := v_0
		mask := v_1
		mem := v_2
		if !(t.Size() == 64) {
			break
		}
		v.reset(OpAMD64VPMASK16load512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x32ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLoadMasked32(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LoadMasked32 <t> ptr mask mem)
	// cond: t.Size() == 16
	// result: (VPMASK32load128 ptr mask mem)
	for {
		t := v.Type
		ptr := v_0
		mask := v_1
		mem := v_2
		if !(t.Size() == 16) {
			break
		}
		v.reset(OpAMD64VPMASK32load128)
		v.AddArg3(ptr, mask, mem)
		return true
	}
	// match: (LoadMasked32 <t> ptr mask mem)
	// cond: t.Size() == 32
	// result: (VPMASK32load256 ptr mask mem)
	for {
		t := v.Type
		ptr := v_0
		mask := v_1
		mem := v_2
		if !(t.Size() == 32) {
			break
		}
		v.reset(OpAMD64VPMASK32load256)
		v.AddArg3(ptr, mask, mem)
		return true
	}
	// match: (LoadMasked32 <t> ptr mask mem)
	// cond: t.Size() == 64
	// result: (VPMASK32load512 ptr (VPMOVVec32x16ToM <types.TypeMask> mask) mem)
	for {
		t := v.Type
		ptr := v_0
		mask := v_1
		mem := v_2
		if !(t.Size() == 64) {
			break
		}
		v.reset(OpAMD64VPMASK32load512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLoadMasked64(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LoadMasked64 <t> ptr mask mem)
	// cond: t.Size() == 16
	// result: (VPMASK64load128 ptr mask mem)
	for {
		t := v.Type
		ptr := v_0
		mask := v_1
		mem := v_2
		if !(t.Size() == 16) {
			break
		}
		v.reset(OpAMD64VPMASK64load128)
		v.AddArg3(ptr, mask, mem)
		return true
	}
	// match: (LoadMasked64 <t> ptr mask mem)
	// cond: t.Size() == 32
	// result: (VPMASK64load256 ptr mask mem)
	for {
		t := v.Type
		ptr := v_0
		mask := v_1
		mem := v_2
		if !(t.Size() == 32) {
			break
		}
		v.reset(OpAMD64VPMASK64load256)
		v.AddArg3(ptr, mask, mem)
		return true
	}
	// match: (LoadMasked64 <t> ptr mask mem)
	// cond: t.Size() == 64
	// result: (VPMASK64load512 ptr (VPMOVVec64x8ToM <types.TypeMask> mask) mem)
	for {
		t := v.Type
		ptr := v_0
		mask := v_1
		mem := v_2
		if !(t.Size() == 64) {
			break
		}
		v.reset(OpAMD64VPMASK64load512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLoadMasked8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (LoadMasked8 <t> ptr mask mem)
	// cond: t.Size() == 64
	// result: (VPMASK8load512 ptr (VPMOVVec8x64ToM <types.TypeMask> mask) mem)
	for {
		t := v.Type
		ptr := v_0
		mask := v_1
		mem := v_2
		if !(t.Size() == 64) {
			break
		}
		v.reset(OpAMD64VPMASK8load512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x64ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLocalAddr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LocalAddr <t> {sym} base mem)
	// cond: t.Elem().HasPointers()
	// result: (LEAQ {sym} (SPanchored base mem))
	for {
		t := v.Type
		sym := auxToSym(v.Aux)
		base := v_0
		mem := v_1
		if !(t.Elem().HasPointers()) {
			break
		}
		v.reset(OpAMD64LEAQ)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpSPanchored, typ.Uintptr)
		v0.AddArg2(base, mem)
		v.AddArg(v0)
		return true
	}
	// match: (LocalAddr <t> {sym} base _)
	// cond: !t.Elem().HasPointers()
	// result: (LEAQ {sym} base)
	for {
		t := v.Type
		sym := auxToSym(v.Aux)
		base := v_0
		if !(!t.Elem().HasPointers()) {
			break
		}
		v.reset(OpAMD64LEAQ)
		v.Aux = symToAux(sym)
		v.AddArg(base)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLsh16x16(v *Value) bool {
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
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v2.AuxInt = int16ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh16x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLsh16x32(v *Value) bool {
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
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh16x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh16x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPQconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh16x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLsh16x8(v *Value) bool {
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
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v2.AuxInt = int8ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh16x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLsh32x16(v *Value) bool {
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
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v2.AuxInt = int16ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh32x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLsh32x32(v *Value) bool {
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
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh32x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh32x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPQconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh32x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLsh32x8(v *Value) bool {
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
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v2.AuxInt = int8ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh32x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh64x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDQ (SHLQ <t> x y) (SBBQcarrymask <t> (CMPWconst y [64])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64ANDQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQ, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v2.AuxInt = int16ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh64x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SHLQ x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh64x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDQ (SHLQ <t> x y) (SBBQcarrymask <t> (CMPLconst y [64])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64ANDQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQ, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh64x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SHLQ x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh64x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDQ (SHLQ <t> x y) (SBBQcarrymask <t> (CMPQconst y [64])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64ANDQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQ, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh64x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SHLQ x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh64x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDQ (SHLQ <t> x y) (SBBQcarrymask <t> (CMPBconst y [64])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64ANDQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQ, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v2.AuxInt = int8ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh64x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SHLQ x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLsh8x16(v *Value) bool {
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
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v2.AuxInt = int16ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh8x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLsh8x32(v *Value) bool {
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
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh8x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh8x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPQconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh8x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLsh8x8(v *Value) bool {
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
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v2.AuxInt = int8ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Lsh8x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SHLL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpMax32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Max32F <t> x y)
	// result: (Neg32F <t> (Min32F <t> (Neg32F <t> x) (Neg32F <t> y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpNeg32F)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpMin32F, t)
		v1 := b.NewValue0(v.Pos, OpNeg32F, t)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpNeg32F, t)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpMax64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Max64F <t> x y)
	// result: (Neg64F <t> (Min64F <t> (Neg64F <t> x) (Neg64F <t> y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpNeg64F)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpMin64F, t)
		v1 := b.NewValue0(v.Pos, OpNeg64F, t)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpNeg64F, t)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpMin32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Min32F <t> x y)
	// result: (POR (MINSS <t> (MINSS <t> x y) x) (MINSS <t> x y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpAMD64POR)
		v0 := b.NewValue0(v.Pos, OpAMD64MINSS, t)
		v1 := b.NewValue0(v.Pos, OpAMD64MINSS, t)
		v1.AddArg2(x, y)
		v0.AddArg2(v1, x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueAMD64_OpMin64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Min64F <t> x y)
	// result: (POR (MINSD <t> (MINSD <t> x y) x) (MINSD <t> x y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpAMD64POR)
		v0 := b.NewValue0(v.Pos, OpAMD64MINSD, t)
		v1 := b.NewValue0(v.Pos, OpAMD64MINSD, t)
		v1.AddArg2(x, y)
		v0.AddArg2(v1, x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueAMD64_OpMod16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16 [a] x y)
	// result: (Select1 (DIVW [a] x y))
	for {
		a := auxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVW, types.NewTuple(typ.Int16, typ.Int16))
		v0.AuxInt = boolToAuxInt(a)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpMod16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16u x y)
	// result: (Select1 (DIVWU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVWU, types.NewTuple(typ.UInt16, typ.UInt16))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpMod32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32 [a] x y)
	// result: (Select1 (DIVL [a] x y))
	for {
		a := auxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVL, types.NewTuple(typ.Int32, typ.Int32))
		v0.AuxInt = boolToAuxInt(a)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpMod32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32u x y)
	// result: (Select1 (DIVLU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVLU, types.NewTuple(typ.UInt32, typ.UInt32))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpMod64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod64 [a] x y)
	// result: (Select1 (DIVQ [a] x y))
	for {
		a := auxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVQ, types.NewTuple(typ.Int64, typ.Int64))
		v0.AuxInt = boolToAuxInt(a)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpMod64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod64u x y)
	// result: (Select1 (DIVQU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVQU, types.NewTuple(typ.UInt64, typ.UInt64))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpMod8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8 x y)
	// result: (Select1 (DIVW (SignExt8to16 x) (SignExt8to16 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVW, types.NewTuple(typ.Int16, typ.Int16))
		v1 := b.NewValue0(v.Pos, OpSignExt8to16, typ.Int16)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpSignExt8to16, typ.Int16)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpMod8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8u x y)
	// result: (Select1 (DIVWU (ZeroExt8to16 x) (ZeroExt8to16 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVWU, types.NewTuple(typ.UInt16, typ.UInt16))
		v1 := b.NewValue0(v.Pos, OpZeroExt8to16, typ.UInt16)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to16, typ.UInt16)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpMove(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
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
		v.reset(OpAMD64MOVBstore)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBload, typ.UInt8)
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
		v.reset(OpAMD64MOVWstore)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
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
		v.reset(OpAMD64MOVLstore)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [8] dst src mem)
	// result: (MOVQstore dst (MOVQload src mem) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpAMD64MOVQstore)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [16] dst src mem)
	// result: (MOVOstore dst (MOVOload src mem) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 16 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpAMD64MOVOstore)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVOload, types.TypeInt128)
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
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
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
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
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
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v0.AuxInt = int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
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
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(3)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [9] dst src mem)
	// result: (MOVBstore [8] dst (MOVBload [8] src mem) (MOVQstore dst (MOVQload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 9 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [10] dst src mem)
	// result: (MOVWstore [8] dst (MOVWload [8] src mem) (MOVQstore dst (MOVQload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 10 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v0.AuxInt = int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [11] dst src mem)
	// result: (MOVLstore [7] dst (MOVLload [7] src mem) (MOVQstore dst (MOVQload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 11 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = int32ToAuxInt(7)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(7)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [12] dst src mem)
	// result: (MOVLstore [8] dst (MOVLload [8] src mem) (MOVQstore dst (MOVQload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 12 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s >= 13 && s <= 15
	// result: (MOVQstore [int32(s-8)] dst (MOVQload [int32(s-8)] src mem) (MOVQstore dst (MOVQload src mem) mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s >= 13 && s <= 15) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(int32(s - 8))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(int32(s - 8))
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 16 && s < 192 && logLargeCopy(v, s)
	// result: (LoweredMove [s] dst src mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 16 && s < 192 && logLargeCopy(v, s)) {
			break
		}
		v.reset(OpAMD64LoweredMove)
		v.AuxInt = int64ToAuxInt(s)
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s >= 192 && s <= repMoveThreshold && logLargeCopy(v, s)
	// result: (LoweredMoveLoop [s] dst src mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s >= 192 && s <= repMoveThreshold && logLargeCopy(v, s)) {
			break
		}
		v.reset(OpAMD64LoweredMoveLoop)
		v.AuxInt = int64ToAuxInt(s)
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > repMoveThreshold && s%8 != 0
	// result: (Move [s-s%8] (OffPtr <dst.Type> dst [s%8]) (OffPtr <src.Type> src [s%8]) (MOVQstore dst (MOVQload src mem) mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > repMoveThreshold && s%8 != 0) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(s - s%8)
		v0 := b.NewValue0(v.Pos, OpOffPtr, dst.Type)
		v0.AuxInt = int64ToAuxInt(s % 8)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpOffPtr, src.Type)
		v1.AuxInt = int64ToAuxInt(s % 8)
		v1.AddArg(src)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVQstore, types.TypeMem)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v3.AddArg2(src, mem)
		v2.AddArg3(dst, v3, mem)
		v.AddArg3(v0, v1, v2)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > repMoveThreshold && s%8 == 0 && logLargeCopy(v, s)
	// result: (REPMOVSQ dst src (MOVQconst [s/8]) mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > repMoveThreshold && s%8 == 0 && logLargeCopy(v, s)) {
			break
		}
		v.reset(OpAMD64REPMOVSQ)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(s / 8)
		v.AddArg4(dst, src, v0, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpNeg32F(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg32F x)
	// result: (PXOR x (MOVSSconst <typ.Float32> [float32(math.Copysign(0, -1))]))
	for {
		x := v_0
		v.reset(OpAMD64PXOR)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVSSconst, typ.Float32)
		v0.AuxInt = float32ToAuxInt(float32(math.Copysign(0, -1)))
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpNeg64F(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg64F x)
	// result: (PXOR x (MOVSDconst <typ.Float64> [math.Copysign(0, -1)]))
	for {
		x := v_0
		v.reset(OpAMD64PXOR)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVSDconst, typ.Float64)
		v0.AuxInt = float64ToAuxInt(math.Copysign(0, -1))
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpNeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq16 x y)
	// result: (SETNE (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32 x y)
	// result: (SETNE (CMPL x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32F x y)
	// result: (SETNEF (UCOMISS x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETNEF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISS, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64 x y)
	// result: (SETNE (CMPQ x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64F x y)
	// result: (SETNEF (UCOMISD x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETNEF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq8 x y)
	// result: (SETNE (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNeqB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (NeqB x y)
	// result: (SETNE (CMPB x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNeqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (NeqPtr x y)
	// result: (SETNE (CMPQ x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNot(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Not x)
	// result: (XORLconst [1] x)
	for {
		x := v_0
		v.reset(OpAMD64XORLconst)
		v.AuxInt = int32ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpNotEqualFloat32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NotEqualFloat32x16 x y)
	// result: (VPMOVMToVec32x16 (VCMPPS512 [4] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VCMPPS512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(4)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNotEqualFloat32x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (NotEqualFloat32x4 x y)
	// result: (VCMPPS128 [4] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPS128)
		v.AuxInt = uint8ToAuxInt(4)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpNotEqualFloat32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (NotEqualFloat32x8 x y)
	// result: (VCMPPS256 [4] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPS256)
		v.AuxInt = uint8ToAuxInt(4)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpNotEqualFloat64x2(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (NotEqualFloat64x2 x y)
	// result: (VCMPPD128 [4] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPD128)
		v.AuxInt = uint8ToAuxInt(4)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpNotEqualFloat64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (NotEqualFloat64x4 x y)
	// result: (VCMPPD256 [4] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VCMPPD256)
		v.AuxInt = uint8ToAuxInt(4)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpNotEqualFloat64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NotEqualFloat64x8 x y)
	// result: (VPMOVMToVec64x8 (VCMPPD512 [4] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VCMPPD512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(4)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNotEqualInt16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NotEqualInt16x32 x y)
	// result: (VPMOVMToVec16x32 (VPCMPW512 [4] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec16x32)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPW512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(4)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNotEqualInt32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NotEqualInt32x16 x y)
	// result: (VPMOVMToVec32x16 (VPCMPD512 [4] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPD512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(4)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNotEqualInt64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NotEqualInt64x8 x y)
	// result: (VPMOVMToVec64x8 (VPCMPQ512 [4] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPQ512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(4)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNotEqualInt8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NotEqualInt8x64 x y)
	// result: (VPMOVMToVec8x64 (VPCMPB512 [4] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec8x64)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPB512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(4)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNotEqualUint16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NotEqualUint16x32 x y)
	// result: (VPMOVMToVec16x32 (VPCMPUW512 [4] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec16x32)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUW512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(4)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNotEqualUint32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NotEqualUint32x16 x y)
	// result: (VPMOVMToVec32x16 (VPCMPUD512 [4] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec32x16)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUD512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(4)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNotEqualUint64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NotEqualUint64x8 x y)
	// result: (VPMOVMToVec64x8 (VPCMPUQ512 [4] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec64x8)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUQ512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(4)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNotEqualUint8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NotEqualUint8x64 x y)
	// result: (VPMOVMToVec8x64 (VPCMPUB512 [4] x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VPMOVMToVec8x64)
		v0 := b.NewValue0(v.Pos, OpAMD64VPCMPUB512, typ.Mask)
		v0.AuxInt = uint8ToAuxInt(4)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpOffPtr(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (OffPtr [off] ptr)
	// cond: is32Bit(off)
	// result: (ADDQconst [int32(off)] ptr)
	for {
		off := auxIntToInt64(v.AuxInt)
		ptr := v_0
		if !(is32Bit(off)) {
			break
		}
		v.reset(OpAMD64ADDQconst)
		v.AuxInt = int32ToAuxInt(int32(off))
		v.AddArg(ptr)
		return true
	}
	// match: (OffPtr [off] ptr)
	// result: (ADDQ (MOVQconst [off]) ptr)
	for {
		off := auxIntToInt64(v.AuxInt)
		ptr := v_0
		v.reset(OpAMD64ADDQ)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(off)
		v.AddArg2(v0, ptr)
		return true
	}
}
func rewriteValueAMD64_OpPopCount16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount16 x)
	// result: (POPCNTL (MOVWQZX <typ.UInt32> x))
	for {
		x := v_0
		v.reset(OpAMD64POPCNTL)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWQZX, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpPopCount8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount8 x)
	// result: (POPCNTL (MOVBQZX <typ.UInt32> x))
	for {
		x := v_0
		v.reset(OpAMD64POPCNTL)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBQZX, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpRoundToEven(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEven x)
	// result: (ROUNDSD [0] x)
	for {
		x := v_0
		v.reset(OpAMD64ROUNDSD)
		v.AuxInt = int8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpRoundToEvenFloat32x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEvenFloat32x4 x)
	// result: (VROUNDPS128 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VROUNDPS128)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpRoundToEvenFloat32x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEvenFloat32x8 x)
	// result: (VROUNDPS256 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VROUNDPS256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpRoundToEvenFloat64x2(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEvenFloat64x2 x)
	// result: (VROUNDPD128 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VROUNDPD128)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpRoundToEvenFloat64x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEvenFloat64x4 x)
	// result: (VROUNDPD256 [0] x)
	for {
		x := v_0
		v.reset(OpAMD64VROUNDPD256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpRoundToEvenScaledFloat32x16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEvenScaledFloat32x16 [a] x)
	// result: (VRNDSCALEPS512 [a+0] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPS512)
		v.AuxInt = uint8ToAuxInt(a + 0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpRoundToEvenScaledFloat32x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEvenScaledFloat32x4 [a] x)
	// result: (VRNDSCALEPS128 [a+0] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPS128)
		v.AuxInt = uint8ToAuxInt(a + 0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpRoundToEvenScaledFloat32x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEvenScaledFloat32x8 [a] x)
	// result: (VRNDSCALEPS256 [a+0] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPS256)
		v.AuxInt = uint8ToAuxInt(a + 0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpRoundToEvenScaledFloat64x2(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEvenScaledFloat64x2 [a] x)
	// result: (VRNDSCALEPD128 [a+0] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPD128)
		v.AuxInt = uint8ToAuxInt(a + 0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpRoundToEvenScaledFloat64x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEvenScaledFloat64x4 [a] x)
	// result: (VRNDSCALEPD256 [a+0] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPD256)
		v.AuxInt = uint8ToAuxInt(a + 0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpRoundToEvenScaledFloat64x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEvenScaledFloat64x8 [a] x)
	// result: (VRNDSCALEPD512 [a+0] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPD512)
		v.AuxInt = uint8ToAuxInt(a + 0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpRoundToEvenScaledResidueFloat32x16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEvenScaledResidueFloat32x16 [a] x)
	// result: (VREDUCEPS512 [a+0] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPS512)
		v.AuxInt = uint8ToAuxInt(a + 0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpRoundToEvenScaledResidueFloat32x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEvenScaledResidueFloat32x4 [a] x)
	// result: (VREDUCEPS128 [a+0] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPS128)
		v.AuxInt = uint8ToAuxInt(a + 0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpRoundToEvenScaledResidueFloat32x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEvenScaledResidueFloat32x8 [a] x)
	// result: (VREDUCEPS256 [a+0] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPS256)
		v.AuxInt = uint8ToAuxInt(a + 0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpRoundToEvenScaledResidueFloat64x2(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEvenScaledResidueFloat64x2 [a] x)
	// result: (VREDUCEPD128 [a+0] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPD128)
		v.AuxInt = uint8ToAuxInt(a + 0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpRoundToEvenScaledResidueFloat64x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEvenScaledResidueFloat64x4 [a] x)
	// result: (VREDUCEPD256 [a+0] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPD256)
		v.AuxInt = uint8ToAuxInt(a + 0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpRoundToEvenScaledResidueFloat64x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RoundToEvenScaledResidueFloat64x8 [a] x)
	// result: (VREDUCEPD512 [a+0] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPD512)
		v.AuxInt = uint8ToAuxInt(a + 0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpRsh16Ux16(v *Value) bool {
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
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v2.AuxInt = int16ToAuxInt(16)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh16Ux16 x y)
	// cond: shiftIsBounded(v)
	// result: (SHRW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHRW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh16Ux32(v *Value) bool {
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
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(16)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh16Ux32 x y)
	// cond: shiftIsBounded(v)
	// result: (SHRW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHRW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh16Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16Ux64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHRW <t> x y) (SBBLcarrymask <t> (CMPQconst y [16])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(16)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh16Ux64 x y)
	// cond: shiftIsBounded(v)
	// result: (SHRW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHRW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh16Ux8(v *Value) bool {
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
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v2.AuxInt = int8ToAuxInt(16)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh16Ux8 x y)
	// cond: shiftIsBounded(v)
	// result: (SHRW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHRW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh16x16(v *Value) bool {
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
		v.reset(OpAMD64SARW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v3.AuxInt = int16ToAuxInt(16)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SARW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh16x32(v *Value) bool {
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
		v.reset(OpAMD64SARW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v3.AuxInt = int32ToAuxInt(16)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SARW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh16x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SARW <t> x (ORQ <y.Type> y (NOTQ <y.Type> (SBBQcarrymask <y.Type> (CMPQconst y [16])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, y.Type)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTQ, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v3.AuxInt = int32ToAuxInt(16)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SARW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh16x8(v *Value) bool {
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
		v.reset(OpAMD64SARW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v3.AuxInt = int8ToAuxInt(16)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh16x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SARW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh32Ux16(v *Value) bool {
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
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v2.AuxInt = int16ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh32Ux16 x y)
	// cond: shiftIsBounded(v)
	// result: (SHRL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHRL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh32Ux32(v *Value) bool {
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
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh32Ux32 x y)
	// cond: shiftIsBounded(v)
	// result: (SHRL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHRL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh32Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32Ux64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHRL <t> x y) (SBBLcarrymask <t> (CMPQconst y [32])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh32Ux64 x y)
	// cond: shiftIsBounded(v)
	// result: (SHRL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHRL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh32Ux8(v *Value) bool {
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
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v2.AuxInt = int8ToAuxInt(32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh32Ux8 x y)
	// cond: shiftIsBounded(v)
	// result: (SHRL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHRL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh32x16(v *Value) bool {
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
		v.reset(OpAMD64SARL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v3.AuxInt = int16ToAuxInt(32)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SARL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh32x32(v *Value) bool {
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
		v.reset(OpAMD64SARL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v3.AuxInt = int32ToAuxInt(32)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SARL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SARL <t> x (ORQ <y.Type> y (NOTQ <y.Type> (SBBQcarrymask <y.Type> (CMPQconst y [32])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, y.Type)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTQ, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v3.AuxInt = int32ToAuxInt(32)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SARL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh32x8(v *Value) bool {
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
		v.reset(OpAMD64SARL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v3.AuxInt = int8ToAuxInt(32)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh32x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SARL x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh64Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64Ux16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDQ (SHRQ <t> x y) (SBBQcarrymask <t> (CMPWconst y [64])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64ANDQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRQ, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v2.AuxInt = int16ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh64Ux16 x y)
	// cond: shiftIsBounded(v)
	// result: (SHRQ x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh64Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64Ux32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDQ (SHRQ <t> x y) (SBBQcarrymask <t> (CMPLconst y [64])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64ANDQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRQ, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh64Ux32 x y)
	// cond: shiftIsBounded(v)
	// result: (SHRQ x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh64Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64Ux64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDQ (SHRQ <t> x y) (SBBQcarrymask <t> (CMPQconst y [64])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64ANDQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRQ, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh64Ux64 x y)
	// cond: shiftIsBounded(v)
	// result: (SHRQ x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh64Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64Ux8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDQ (SHRQ <t> x y) (SBBQcarrymask <t> (CMPBconst y [64])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64ANDQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRQ, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v2.AuxInt = int8ToAuxInt(64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh64Ux8 x y)
	// cond: shiftIsBounded(v)
	// result: (SHRQ x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SARQ <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPWconst y [64])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v3.AuxInt = int16ToAuxInt(64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SARQ x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SARQ <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPLconst y [64])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v3.AuxInt = int32ToAuxInt(64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SARQ x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SARQ <t> x (ORQ <y.Type> y (NOTQ <y.Type> (SBBQcarrymask <y.Type> (CMPQconst y [64])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, y.Type)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTQ, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v3.AuxInt = int32ToAuxInt(64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SARQ x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SARQ <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPBconst y [64])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v3.AuxInt = int8ToAuxInt(64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SARQ x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh8Ux16(v *Value) bool {
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
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRB, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v2.AuxInt = int16ToAuxInt(8)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh8Ux16 x y)
	// cond: shiftIsBounded(v)
	// result: (SHRB x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHRB)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh8Ux32(v *Value) bool {
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
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRB, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(8)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh8Ux32 x y)
	// cond: shiftIsBounded(v)
	// result: (SHRB x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHRB)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh8Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8Ux64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (ANDL (SHRB <t> x y) (SBBLcarrymask <t> (CMPQconst y [8])))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRB, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(8)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh8Ux64 x y)
	// cond: shiftIsBounded(v)
	// result: (SHRB x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHRB)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh8Ux8(v *Value) bool {
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
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRB, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v2.AuxInt = int8ToAuxInt(8)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Rsh8Ux8 x y)
	// cond: shiftIsBounded(v)
	// result: (SHRB x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SHRB)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh8x16(v *Value) bool {
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
		v.reset(OpAMD64SARB)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v3.AuxInt = int16ToAuxInt(8)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SARB x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARB)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh8x32(v *Value) bool {
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
		v.reset(OpAMD64SARB)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v3.AuxInt = int32ToAuxInt(8)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SARB x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARB)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh8x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (SARB <t> x (ORQ <y.Type> y (NOTQ <y.Type> (SBBQcarrymask <y.Type> (CMPQconst y [8])))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARB)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, y.Type)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTQ, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v3.AuxInt = int32ToAuxInt(8)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SARB x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARB)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpRsh8x8(v *Value) bool {
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
		v.reset(OpAMD64SARB)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v3.AuxInt = int8ToAuxInt(8)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh8x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SARB x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpAMD64SARB)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpSelect0(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select0 (Mul64uover x y))
	// result: (Select0 <typ.UInt64> (MULQU x y))
	for {
		if v_0.Op != OpMul64uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpSelect0)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpAMD64MULQU, types.NewTuple(typ.UInt64, types.TypeFlags))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
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
		v0 := b.NewValue0(v.Pos, OpAMD64MULLU, types.NewTuple(typ.UInt32, types.TypeFlags))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (Select0 (Add64carry x y c))
	// result: (Select0 <typ.UInt64> (ADCQ x y (Select1 <types.TypeFlags> (NEGLflags c))))
	for {
		if v_0.Op != OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpSelect0)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpAMD64ADCQ, types.NewTuple(typ.UInt64, types.TypeFlags))
		v1 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v2 := b.NewValue0(v.Pos, OpAMD64NEGLflags, types.NewTuple(typ.UInt32, types.TypeFlags))
		v2.AddArg(c)
		v1.AddArg(v2)
		v0.AddArg3(x, y, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Select0 (Sub64borrow x y c))
	// result: (Select0 <typ.UInt64> (SBBQ x y (Select1 <types.TypeFlags> (NEGLflags c))))
	for {
		if v_0.Op != OpSub64borrow {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpSelect0)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpAMD64SBBQ, types.NewTuple(typ.UInt64, types.TypeFlags))
		v1 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v2 := b.NewValue0(v.Pos, OpAMD64NEGLflags, types.NewTuple(typ.UInt32, types.TypeFlags))
		v2.AddArg(c)
		v1.AddArg(v2)
		v0.AddArg3(x, y, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Select0 <t> (AddTupleFirst32 val tuple))
	// result: (ADDL val (Select0 <t> tuple))
	for {
		t := v.Type
		if v_0.Op != OpAMD64AddTupleFirst32 {
			break
		}
		tuple := v_0.Args[1]
		val := v_0.Args[0]
		v.reset(OpAMD64ADDL)
		v0 := b.NewValue0(v.Pos, OpSelect0, t)
		v0.AddArg(tuple)
		v.AddArg2(val, v0)
		return true
	}
	// match: (Select0 <t> (AddTupleFirst64 val tuple))
	// result: (ADDQ val (Select0 <t> tuple))
	for {
		t := v.Type
		if v_0.Op != OpAMD64AddTupleFirst64 {
			break
		}
		tuple := v_0.Args[1]
		val := v_0.Args[0]
		v.reset(OpAMD64ADDQ)
		v0 := b.NewValue0(v.Pos, OpSelect0, t)
		v0.AddArg(tuple)
		v.AddArg2(val, v0)
		return true
	}
	// match: (Select0 a:(ADDQconstflags [c] x))
	// cond: a.Uses == 1
	// result: (ADDQconst [c] x)
	for {
		a := v_0
		if a.Op != OpAMD64ADDQconstflags {
			break
		}
		c := auxIntToInt32(a.AuxInt)
		x := a.Args[0]
		if !(a.Uses == 1) {
			break
		}
		v.reset(OpAMD64ADDQconst)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (Select0 a:(ADDLconstflags [c] x))
	// cond: a.Uses == 1
	// result: (ADDLconst [c] x)
	for {
		a := v_0
		if a.Op != OpAMD64ADDLconstflags {
			break
		}
		c := auxIntToInt32(a.AuxInt)
		x := a.Args[0]
		if !(a.Uses == 1) {
			break
		}
		v.reset(OpAMD64ADDLconst)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpSelect1(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select1 (Mul64uover x y))
	// result: (SETO (Select1 <types.TypeFlags> (MULQU x y)))
	for {
		if v_0.Op != OpMul64uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpAMD64SETO)
		v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpAMD64MULQU, types.NewTuple(typ.UInt64, types.TypeFlags))
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Select1 (Mul32uover x y))
	// result: (SETO (Select1 <types.TypeFlags> (MULLU x y)))
	for {
		if v_0.Op != OpMul32uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpAMD64SETO)
		v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpAMD64MULLU, types.NewTuple(typ.UInt32, types.TypeFlags))
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Select1 (Add64carry x y c))
	// result: (NEGQ <typ.UInt64> (SBBQcarrymask <typ.UInt64> (Select1 <types.TypeFlags> (ADCQ x y (Select1 <types.TypeFlags> (NEGLflags c))))))
	for {
		if v_0.Op != OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpAMD64NEGQ)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v2 := b.NewValue0(v.Pos, OpAMD64ADCQ, types.NewTuple(typ.UInt64, types.TypeFlags))
		v3 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v4 := b.NewValue0(v.Pos, OpAMD64NEGLflags, types.NewTuple(typ.UInt32, types.TypeFlags))
		v4.AddArg(c)
		v3.AddArg(v4)
		v2.AddArg3(x, y, v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Select1 (Sub64borrow x y c))
	// result: (NEGQ <typ.UInt64> (SBBQcarrymask <typ.UInt64> (Select1 <types.TypeFlags> (SBBQ x y (Select1 <types.TypeFlags> (NEGLflags c))))))
	for {
		if v_0.Op != OpSub64borrow {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpAMD64NEGQ)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBQ, types.NewTuple(typ.UInt64, types.TypeFlags))
		v3 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v4 := b.NewValue0(v.Pos, OpAMD64NEGLflags, types.NewTuple(typ.UInt32, types.TypeFlags))
		v4.AddArg(c)
		v3.AddArg(v4)
		v2.AddArg3(x, y, v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Select1 (NEGLflags (MOVQconst [0])))
	// result: (FlagEQ)
	for {
		if v_0.Op != OpAMD64NEGLflags {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64MOVQconst || auxIntToInt64(v_0_0.AuxInt) != 0 {
			break
		}
		v.reset(OpAMD64FlagEQ)
		return true
	}
	// match: (Select1 (NEGLflags (NEGQ (SBBQcarrymask x))))
	// result: x
	for {
		if v_0.Op != OpAMD64NEGLflags {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64SBBQcarrymask {
			break
		}
		x := v_0_0_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Select1 (AddTupleFirst32 _ tuple))
	// result: (Select1 tuple)
	for {
		if v_0.Op != OpAMD64AddTupleFirst32 {
			break
		}
		tuple := v_0.Args[1]
		v.reset(OpSelect1)
		v.AddArg(tuple)
		return true
	}
	// match: (Select1 (AddTupleFirst64 _ tuple))
	// result: (Select1 tuple)
	for {
		if v_0.Op != OpAMD64AddTupleFirst64 {
			break
		}
		tuple := v_0.Args[1]
		v.reset(OpSelect1)
		v.AddArg(tuple)
		return true
	}
	// match: (Select1 a:(LoweredAtomicAnd64 ptr val mem))
	// cond: a.Uses == 1 && clobber(a)
	// result: (ANDQlock ptr val mem)
	for {
		a := v_0
		if a.Op != OpAMD64LoweredAtomicAnd64 {
			break
		}
		mem := a.Args[2]
		ptr := a.Args[0]
		val := a.Args[1]
		if !(a.Uses == 1 && clobber(a)) {
			break
		}
		v.reset(OpAMD64ANDQlock)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Select1 a:(LoweredAtomicAnd32 ptr val mem))
	// cond: a.Uses == 1 && clobber(a)
	// result: (ANDLlock ptr val mem)
	for {
		a := v_0
		if a.Op != OpAMD64LoweredAtomicAnd32 {
			break
		}
		mem := a.Args[2]
		ptr := a.Args[0]
		val := a.Args[1]
		if !(a.Uses == 1 && clobber(a)) {
			break
		}
		v.reset(OpAMD64ANDLlock)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Select1 a:(LoweredAtomicOr64 ptr val mem))
	// cond: a.Uses == 1 && clobber(a)
	// result: (ORQlock ptr val mem)
	for {
		a := v_0
		if a.Op != OpAMD64LoweredAtomicOr64 {
			break
		}
		mem := a.Args[2]
		ptr := a.Args[0]
		val := a.Args[1]
		if !(a.Uses == 1 && clobber(a)) {
			break
		}
		v.reset(OpAMD64ORQlock)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Select1 a:(LoweredAtomicOr32 ptr val mem))
	// cond: a.Uses == 1 && clobber(a)
	// result: (ORLlock ptr val mem)
	for {
		a := v_0
		if a.Op != OpAMD64LoweredAtomicOr32 {
			break
		}
		mem := a.Args[2]
		ptr := a.Args[0]
		val := a.Args[1]
		if !(a.Uses == 1 && clobber(a)) {
			break
		}
		v.reset(OpAMD64ORLlock)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpSelectN(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (SelectN [0] call:(CALLstatic {sym} s1:(MOVQstoreconst _ [sc] s2:(MOVQstore _ src s3:(MOVQstore _ dst mem)))))
	// cond: sc.Val64() >= 0 && isSameCall(sym, "runtime.memmove") && s1.Uses == 1 && s2.Uses == 1 && s3.Uses == 1 && isInlinableMemmove(dst, src, sc.Val64(), config) && clobber(s1, s2, s3, call)
	// result: (Move [sc.Val64()] dst src mem)
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		call := v_0
		if call.Op != OpAMD64CALLstatic || len(call.Args) != 1 {
			break
		}
		sym := auxToCall(call.Aux)
		s1 := call.Args[0]
		if s1.Op != OpAMD64MOVQstoreconst {
			break
		}
		sc := auxIntToValAndOff(s1.AuxInt)
		_ = s1.Args[1]
		s2 := s1.Args[1]
		if s2.Op != OpAMD64MOVQstore {
			break
		}
		_ = s2.Args[2]
		src := s2.Args[1]
		s3 := s2.Args[2]
		if s3.Op != OpAMD64MOVQstore {
			break
		}
		mem := s3.Args[2]
		dst := s3.Args[1]
		if !(sc.Val64() >= 0 && isSameCall(sym, "runtime.memmove") && s1.Uses == 1 && s2.Uses == 1 && s3.Uses == 1 && isInlinableMemmove(dst, src, sc.Val64(), config) && clobber(s1, s2, s3, call)) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(sc.Val64())
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (SelectN [0] call:(CALLstatic {sym} dst src (MOVQconst [sz]) mem))
	// cond: sz >= 0 && isSameCall(sym, "runtime.memmove") && call.Uses == 1 && isInlinableMemmove(dst, src, sz, config) && clobber(call)
	// result: (Move [sz] dst src mem)
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		call := v_0
		if call.Op != OpAMD64CALLstatic || len(call.Args) != 4 {
			break
		}
		sym := auxToCall(call.Aux)
		mem := call.Args[3]
		dst := call.Args[0]
		src := call.Args[1]
		call_2 := call.Args[2]
		if call_2.Op != OpAMD64MOVQconst {
			break
		}
		sz := auxIntToInt64(call_2.AuxInt)
		if !(sz >= 0 && isSameCall(sym, "runtime.memmove") && call.Uses == 1 && isInlinableMemmove(dst, src, sz, config) && clobber(call)) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(sz)
		v.AddArg3(dst, src, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpSetHiFloat32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiFloat32x16 x y)
	// result: (VINSERTF64X4512 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTF64X4512)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiFloat32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiFloat32x8 x y)
	// result: (VINSERTF128256 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTF128256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiFloat64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiFloat64x4 x y)
	// result: (VINSERTF128256 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTF128256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiFloat64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiFloat64x8 x y)
	// result: (VINSERTF64X4512 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTF64X4512)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiInt16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiInt16x16 x y)
	// result: (VINSERTI128256 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI128256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiInt16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiInt16x32 x y)
	// result: (VINSERTI64X4512 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI64X4512)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiInt32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiInt32x16 x y)
	// result: (VINSERTI64X4512 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI64X4512)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiInt32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiInt32x8 x y)
	// result: (VINSERTI128256 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI128256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiInt64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiInt64x4 x y)
	// result: (VINSERTI128256 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI128256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiInt64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiInt64x8 x y)
	// result: (VINSERTI64X4512 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI64X4512)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiInt8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiInt8x32 x y)
	// result: (VINSERTI128256 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI128256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiInt8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiInt8x64 x y)
	// result: (VINSERTI64X4512 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI64X4512)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiUint16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiUint16x16 x y)
	// result: (VINSERTI128256 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI128256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiUint16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiUint16x32 x y)
	// result: (VINSERTI64X4512 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI64X4512)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiUint32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiUint32x16 x y)
	// result: (VINSERTI64X4512 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI64X4512)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiUint32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiUint32x8 x y)
	// result: (VINSERTI128256 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI128256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiUint64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiUint64x4 x y)
	// result: (VINSERTI128256 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI128256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiUint64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiUint64x8 x y)
	// result: (VINSERTI64X4512 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI64X4512)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiUint8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiUint8x32 x y)
	// result: (VINSERTI128256 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI128256)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetHiUint8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetHiUint8x64 x y)
	// result: (VINSERTI64X4512 [1] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI64X4512)
		v.AuxInt = uint8ToAuxInt(1)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoFloat32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoFloat32x16 x y)
	// result: (VINSERTF64X4512 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTF64X4512)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoFloat32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoFloat32x8 x y)
	// result: (VINSERTF128256 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTF128256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoFloat64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoFloat64x4 x y)
	// result: (VINSERTF128256 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTF128256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoFloat64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoFloat64x8 x y)
	// result: (VINSERTF64X4512 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTF64X4512)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoInt16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoInt16x16 x y)
	// result: (VINSERTI128256 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI128256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoInt16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoInt16x32 x y)
	// result: (VINSERTI64X4512 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI64X4512)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoInt32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoInt32x16 x y)
	// result: (VINSERTI64X4512 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI64X4512)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoInt32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoInt32x8 x y)
	// result: (VINSERTI128256 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI128256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoInt64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoInt64x4 x y)
	// result: (VINSERTI128256 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI128256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoInt64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoInt64x8 x y)
	// result: (VINSERTI64X4512 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI64X4512)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoInt8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoInt8x32 x y)
	// result: (VINSERTI128256 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI128256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoInt8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoInt8x64 x y)
	// result: (VINSERTI64X4512 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI64X4512)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoUint16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoUint16x16 x y)
	// result: (VINSERTI128256 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI128256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoUint16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoUint16x32 x y)
	// result: (VINSERTI64X4512 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI64X4512)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoUint32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoUint32x16 x y)
	// result: (VINSERTI64X4512 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI64X4512)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoUint32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoUint32x8 x y)
	// result: (VINSERTI128256 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI128256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoUint64x4(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoUint64x4 x y)
	// result: (VINSERTI128256 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI128256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoUint64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoUint64x8 x y)
	// result: (VINSERTI64X4512 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI64X4512)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoUint8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoUint8x32 x y)
	// result: (VINSERTI128256 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI128256)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSetLoUint8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SetLoUint8x64 x y)
	// result: (VINSERTI64X4512 [0] x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64VINSERTI64X4512)
		v.AuxInt = uint8ToAuxInt(0)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueAMD64_OpSlicemask(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Slicemask <t> x)
	// result: (SARQconst (NEGQ <t> x) [63])
	for {
		t := v.Type
		x := v_0
		v.reset(OpAMD64SARQconst)
		v.AuxInt = int8ToAuxInt(63)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpSpectreIndex(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SpectreIndex <t> x y)
	// result: (CMOVQCC x (MOVQconst [0]) (CMPQ x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64CMOVQCC)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v1.AddArg2(x, y)
		v.AddArg3(x, v0, v1)
		return true
	}
}
func rewriteValueAMD64_OpSpectreSliceIndex(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SpectreSliceIndex <t> x y)
	// result: (CMOVQHI x (MOVQconst [0]) (CMPQ x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpAMD64CMOVQHI)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v1.AddArg2(x, y)
		v.AddArg3(x, v0, v1)
		return true
	}
}
func rewriteValueAMD64_OpStore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && t.IsFloat()
	// result: (MOVSDstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && t.IsFloat()) {
			break
		}
		v.reset(OpAMD64MOVSDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && t.IsFloat()
	// result: (MOVSSstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && t.IsFloat()) {
			break
		}
		v.reset(OpAMD64MOVSSstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && !t.IsFloat()
	// result: (MOVQstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && !t.IsFloat()) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && !t.IsFloat()
	// result: (MOVLstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && !t.IsFloat()) {
			break
		}
		v.reset(OpAMD64MOVLstore)
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
		v.reset(OpAMD64MOVWstore)
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
		v.reset(OpAMD64MOVBstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 16
	// result: (VMOVDQUstore128 ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 16) {
			break
		}
		v.reset(OpAMD64VMOVDQUstore128)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 32
	// result: (VMOVDQUstore256 ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 32) {
			break
		}
		v.reset(OpAMD64VMOVDQUstore256)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 64
	// result: (VMOVDQUstore512 ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 64) {
			break
		}
		v.reset(OpAMD64VMOVDQUstore512)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpStoreMask16x16(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (StoreMask16x16 {t} ptr val mem)
	// result: (KMOVQstore ptr (VPMOVVec16x16ToM <t> val) mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64KMOVQstore)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x16ToM, t)
		v0.AddArg(val)
		v.AddArg3(ptr, v0, mem)
		return true
	}
}
func rewriteValueAMD64_OpStoreMask16x32(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (StoreMask16x32 {t} ptr val mem)
	// result: (KMOVQstore ptr (VPMOVVec16x32ToM <t> val) mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64KMOVQstore)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x32ToM, t)
		v0.AddArg(val)
		v.AddArg3(ptr, v0, mem)
		return true
	}
}
func rewriteValueAMD64_OpStoreMask16x8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (StoreMask16x8 {t} ptr val mem)
	// result: (KMOVQstore ptr (VPMOVVec16x8ToM <t> val) mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64KMOVQstore)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x8ToM, t)
		v0.AddArg(val)
		v.AddArg3(ptr, v0, mem)
		return true
	}
}
func rewriteValueAMD64_OpStoreMask32x16(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (StoreMask32x16 {t} ptr val mem)
	// result: (KMOVQstore ptr (VPMOVVec32x16ToM <t> val) mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64KMOVQstore)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x16ToM, t)
		v0.AddArg(val)
		v.AddArg3(ptr, v0, mem)
		return true
	}
}
func rewriteValueAMD64_OpStoreMask32x4(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (StoreMask32x4 {t} ptr val mem)
	// result: (KMOVQstore ptr (VPMOVVec32x4ToM <t> val) mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64KMOVQstore)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x4ToM, t)
		v0.AddArg(val)
		v.AddArg3(ptr, v0, mem)
		return true
	}
}
func rewriteValueAMD64_OpStoreMask32x8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (StoreMask32x8 {t} ptr val mem)
	// result: (KMOVQstore ptr (VPMOVVec32x8ToM <t> val) mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64KMOVQstore)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x8ToM, t)
		v0.AddArg(val)
		v.AddArg3(ptr, v0, mem)
		return true
	}
}
func rewriteValueAMD64_OpStoreMask64x2(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (StoreMask64x2 {t} ptr val mem)
	// result: (KMOVQstore ptr (VPMOVVec64x2ToM <t> val) mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64KMOVQstore)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x2ToM, t)
		v0.AddArg(val)
		v.AddArg3(ptr, v0, mem)
		return true
	}
}
func rewriteValueAMD64_OpStoreMask64x4(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (StoreMask64x4 {t} ptr val mem)
	// result: (KMOVQstore ptr (VPMOVVec64x4ToM <t> val) mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64KMOVQstore)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x4ToM, t)
		v0.AddArg(val)
		v.AddArg3(ptr, v0, mem)
		return true
	}
}
func rewriteValueAMD64_OpStoreMask64x8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (StoreMask64x8 {t} ptr val mem)
	// result: (KMOVQstore ptr (VPMOVVec64x8ToM <t> val) mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64KMOVQstore)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x8ToM, t)
		v0.AddArg(val)
		v.AddArg3(ptr, v0, mem)
		return true
	}
}
func rewriteValueAMD64_OpStoreMask8x16(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (StoreMask8x16 {t} ptr val mem)
	// result: (KMOVQstore ptr (VPMOVVec8x16ToM <t> val) mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64KMOVQstore)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x16ToM, t)
		v0.AddArg(val)
		v.AddArg3(ptr, v0, mem)
		return true
	}
}
func rewriteValueAMD64_OpStoreMask8x32(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (StoreMask8x32 {t} ptr val mem)
	// result: (KMOVQstore ptr (VPMOVVec8x32ToM <t> val) mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64KMOVQstore)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x32ToM, t)
		v0.AddArg(val)
		v.AddArg3(ptr, v0, mem)
		return true
	}
}
func rewriteValueAMD64_OpStoreMask8x64(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (StoreMask8x64 {t} ptr val mem)
	// result: (KMOVQstore ptr (VPMOVVec8x64ToM <t> val) mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpAMD64KMOVQstore)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x64ToM, t)
		v0.AddArg(val)
		v.AddArg3(ptr, v0, mem)
		return true
	}
}
func rewriteValueAMD64_OpStoreMasked16(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (StoreMasked16 {t} ptr mask val mem)
	// cond: t.Size() == 64
	// result: (VPMASK16store512 ptr (VPMOVVec16x32ToM <types.TypeMask> mask) val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		mask := v_1
		val := v_2
		mem := v_3
		if !(t.Size() == 64) {
			break
		}
		v.reset(OpAMD64VPMASK16store512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x32ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg4(ptr, v0, val, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpStoreMasked32(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (StoreMasked32 {t} ptr mask val mem)
	// cond: t.Size() == 16
	// result: (VPMASK32store128 ptr mask val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		mask := v_1
		val := v_2
		mem := v_3
		if !(t.Size() == 16) {
			break
		}
		v.reset(OpAMD64VPMASK32store128)
		v.AddArg4(ptr, mask, val, mem)
		return true
	}
	// match: (StoreMasked32 {t} ptr mask val mem)
	// cond: t.Size() == 32
	// result: (VPMASK32store256 ptr mask val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		mask := v_1
		val := v_2
		mem := v_3
		if !(t.Size() == 32) {
			break
		}
		v.reset(OpAMD64VPMASK32store256)
		v.AddArg4(ptr, mask, val, mem)
		return true
	}
	// match: (StoreMasked32 {t} ptr mask val mem)
	// cond: t.Size() == 64
	// result: (VPMASK32store512 ptr (VPMOVVec32x16ToM <types.TypeMask> mask) val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		mask := v_1
		val := v_2
		mem := v_3
		if !(t.Size() == 64) {
			break
		}
		v.reset(OpAMD64VPMASK32store512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg4(ptr, v0, val, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpStoreMasked64(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (StoreMasked64 {t} ptr mask val mem)
	// cond: t.Size() == 16
	// result: (VPMASK64store128 ptr mask val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		mask := v_1
		val := v_2
		mem := v_3
		if !(t.Size() == 16) {
			break
		}
		v.reset(OpAMD64VPMASK64store128)
		v.AddArg4(ptr, mask, val, mem)
		return true
	}
	// match: (StoreMasked64 {t} ptr mask val mem)
	// cond: t.Size() == 32
	// result: (VPMASK64store256 ptr mask val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		mask := v_1
		val := v_2
		mem := v_3
		if !(t.Size() == 32) {
			break
		}
		v.reset(OpAMD64VPMASK64store256)
		v.AddArg4(ptr, mask, val, mem)
		return true
	}
	// match: (StoreMasked64 {t} ptr mask val mem)
	// cond: t.Size() == 64
	// result: (VPMASK64store512 ptr (VPMOVVec64x8ToM <types.TypeMask> mask) val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		mask := v_1
		val := v_2
		mem := v_3
		if !(t.Size() == 64) {
			break
		}
		v.reset(OpAMD64VPMASK64store512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg4(ptr, v0, val, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpStoreMasked8(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (StoreMasked8 {t} ptr mask val mem)
	// cond: t.Size() == 64
	// result: (VPMASK8store512 ptr (VPMOVVec8x64ToM <types.TypeMask> mask) val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		mask := v_1
		val := v_2
		mem := v_3
		if !(t.Size() == 64) {
			break
		}
		v.reset(OpAMD64VPMASK8store512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x64ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg4(ptr, v0, val, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpTrunc(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc x)
	// result: (ROUNDSD [3] x)
	for {
		x := v_0
		v.reset(OpAMD64ROUNDSD)
		v.AuxInt = int8ToAuxInt(3)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpTruncFloat32x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TruncFloat32x4 x)
	// result: (VROUNDPS128 [3] x)
	for {
		x := v_0
		v.reset(OpAMD64VROUNDPS128)
		v.AuxInt = uint8ToAuxInt(3)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpTruncFloat32x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TruncFloat32x8 x)
	// result: (VROUNDPS256 [3] x)
	for {
		x := v_0
		v.reset(OpAMD64VROUNDPS256)
		v.AuxInt = uint8ToAuxInt(3)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpTruncFloat64x2(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TruncFloat64x2 x)
	// result: (VROUNDPD128 [3] x)
	for {
		x := v_0
		v.reset(OpAMD64VROUNDPD128)
		v.AuxInt = uint8ToAuxInt(3)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpTruncFloat64x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TruncFloat64x4 x)
	// result: (VROUNDPD256 [3] x)
	for {
		x := v_0
		v.reset(OpAMD64VROUNDPD256)
		v.AuxInt = uint8ToAuxInt(3)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpTruncScaledFloat32x16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TruncScaledFloat32x16 [a] x)
	// result: (VRNDSCALEPS512 [a+3] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPS512)
		v.AuxInt = uint8ToAuxInt(a + 3)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpTruncScaledFloat32x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TruncScaledFloat32x4 [a] x)
	// result: (VRNDSCALEPS128 [a+3] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPS128)
		v.AuxInt = uint8ToAuxInt(a + 3)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpTruncScaledFloat32x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TruncScaledFloat32x8 [a] x)
	// result: (VRNDSCALEPS256 [a+3] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPS256)
		v.AuxInt = uint8ToAuxInt(a + 3)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpTruncScaledFloat64x2(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TruncScaledFloat64x2 [a] x)
	// result: (VRNDSCALEPD128 [a+3] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPD128)
		v.AuxInt = uint8ToAuxInt(a + 3)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpTruncScaledFloat64x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TruncScaledFloat64x4 [a] x)
	// result: (VRNDSCALEPD256 [a+3] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPD256)
		v.AuxInt = uint8ToAuxInt(a + 3)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpTruncScaledFloat64x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TruncScaledFloat64x8 [a] x)
	// result: (VRNDSCALEPD512 [a+3] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VRNDSCALEPD512)
		v.AuxInt = uint8ToAuxInt(a + 3)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpTruncScaledResidueFloat32x16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TruncScaledResidueFloat32x16 [a] x)
	// result: (VREDUCEPS512 [a+3] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPS512)
		v.AuxInt = uint8ToAuxInt(a + 3)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpTruncScaledResidueFloat32x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TruncScaledResidueFloat32x4 [a] x)
	// result: (VREDUCEPS128 [a+3] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPS128)
		v.AuxInt = uint8ToAuxInt(a + 3)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpTruncScaledResidueFloat32x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TruncScaledResidueFloat32x8 [a] x)
	// result: (VREDUCEPS256 [a+3] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPS256)
		v.AuxInt = uint8ToAuxInt(a + 3)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpTruncScaledResidueFloat64x2(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TruncScaledResidueFloat64x2 [a] x)
	// result: (VREDUCEPD128 [a+3] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPD128)
		v.AuxInt = uint8ToAuxInt(a + 3)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpTruncScaledResidueFloat64x4(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TruncScaledResidueFloat64x4 [a] x)
	// result: (VREDUCEPD256 [a+3] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPD256)
		v.AuxInt = uint8ToAuxInt(a + 3)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpTruncScaledResidueFloat64x8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TruncScaledResidueFloat64x8 [a] x)
	// result: (VREDUCEPD512 [a+3] x)
	for {
		a := auxIntToUint8(v.AuxInt)
		x := v_0
		v.reset(OpAMD64VREDUCEPD512)
		v.AuxInt = uint8ToAuxInt(a + 3)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpZero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
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
	// result: (MOVBstoreconst [makeValAndOff(0,0)] destptr mem)
	for {
		if auxIntToInt64(v.AuxInt) != 1 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpAMD64MOVBstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [2] destptr mem)
	// result: (MOVWstoreconst [makeValAndOff(0,0)] destptr mem)
	for {
		if auxIntToInt64(v.AuxInt) != 2 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpAMD64MOVWstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [4] destptr mem)
	// result: (MOVLstoreconst [makeValAndOff(0,0)] destptr mem)
	for {
		if auxIntToInt64(v.AuxInt) != 4 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpAMD64MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [8] destptr mem)
	// result: (MOVQstoreconst [makeValAndOff(0,0)] destptr mem)
	for {
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpAMD64MOVQstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [3] destptr mem)
	// result: (MOVBstoreconst [makeValAndOff(0,2)] destptr (MOVWstoreconst [makeValAndOff(0,0)] destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 3 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpAMD64MOVBstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 2))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [5] destptr mem)
	// result: (MOVBstoreconst [makeValAndOff(0,4)] destptr (MOVLstoreconst [makeValAndOff(0,0)] destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 5 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpAMD64MOVBstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 4))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [6] destptr mem)
	// result: (MOVWstoreconst [makeValAndOff(0,4)] destptr (MOVLstoreconst [makeValAndOff(0,0)] destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 6 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpAMD64MOVWstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 4))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [7] destptr mem)
	// result: (MOVLstoreconst [makeValAndOff(0,3)] destptr (MOVLstoreconst [makeValAndOff(0,0)] destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 7 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpAMD64MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 3))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [9] destptr mem)
	// result: (MOVBstoreconst [makeValAndOff(0,8)] destptr (MOVQstoreconst [makeValAndOff(0,0)] destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 9 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpAMD64MOVBstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 8))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [10] destptr mem)
	// result: (MOVWstoreconst [makeValAndOff(0,8)] destptr (MOVQstoreconst [makeValAndOff(0,0)] destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 10 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpAMD64MOVWstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 8))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [11] destptr mem)
	// result: (MOVLstoreconst [makeValAndOff(0,7)] destptr (MOVQstoreconst [makeValAndOff(0,0)] destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 11 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpAMD64MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 7))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [12] destptr mem)
	// result: (MOVLstoreconst [makeValAndOff(0,8)] destptr (MOVQstoreconst [makeValAndOff(0,0)] destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 12 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpAMD64MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 8))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s > 12 && s < 16
	// result: (MOVQstoreconst [makeValAndOff(0,int32(s-8))] destptr (MOVQstoreconst [makeValAndOff(0,0)] destptr mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		if !(s > 12 && s < 16) {
			break
		}
		v.reset(OpAMD64MOVQstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, int32(s-8)))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s >= 16 && s < 192
	// result: (LoweredZero [s] destptr mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		if !(s >= 16 && s < 192) {
			break
		}
		v.reset(OpAMD64LoweredZero)
		v.AuxInt = int64ToAuxInt(s)
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s >= 192 && s <= repZeroThreshold
	// result: (LoweredZeroLoop [s] destptr mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		if !(s >= 192 && s <= repZeroThreshold) {
			break
		}
		v.reset(OpAMD64LoweredZeroLoop)
		v.AuxInt = int64ToAuxInt(s)
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s > repZeroThreshold && s%8 != 0
	// result: (Zero [s-s%8] (OffPtr <destptr.Type> destptr [s%8]) (MOVOstoreconst [makeValAndOff(0,0)] destptr mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		if !(s > repZeroThreshold && s%8 != 0) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = int64ToAuxInt(s - s%8)
		v0 := b.NewValue0(v.Pos, OpOffPtr, destptr.Type)
		v0.AuxInt = int64ToAuxInt(s % 8)
		v0.AddArg(destptr)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVOstoreconst, types.TypeMem)
		v1.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v1.AddArg2(destptr, mem)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s > repZeroThreshold && s%8 == 0
	// result: (REPSTOSQ destptr (MOVQconst [s/8]) (MOVQconst [0]) mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		if !(s > repZeroThreshold && s%8 == 0) {
			break
		}
		v.reset(OpAMD64REPSTOSQ)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(s / 8)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQconst, typ.UInt64)
		v1.AuxInt = int64ToAuxInt(0)
		v.AddArg4(destptr, v0, v1, mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpZeroSIMD(v *Value) bool {
	// match: (ZeroSIMD <t>)
	// cond: t.Size() == 16
	// result: (Zero128 <t>)
	for {
		t := v.Type
		if !(t.Size() == 16) {
			break
		}
		v.reset(OpAMD64Zero128)
		v.Type = t
		return true
	}
	// match: (ZeroSIMD <t>)
	// cond: t.Size() == 32
	// result: (Zero256 <t>)
	for {
		t := v.Type
		if !(t.Size() == 32) {
			break
		}
		v.reset(OpAMD64Zero256)
		v.Type = t
		return true
	}
	// match: (ZeroSIMD <t>)
	// cond: t.Size() == 64
	// result: (Zero512 <t>)
	for {
		t := v.Type
		if !(t.Size() == 64) {
			break
		}
		v.reset(OpAMD64Zero512)
		v.Type = t
		return true
	}
	return false
}
func rewriteValueAMD64_OpblendMaskedInt16x32(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (blendMaskedInt16x32 x y mask)
	// result: (VPBLENDMWMasked512 x y (VPMOVVec16x32ToM <types.TypeMask> mask))
	for {
		x := v_0
		y := v_1
		mask := v_2
		v.reset(OpAMD64VPBLENDMWMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x32ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg3(x, y, v0)
		return true
	}
}
func rewriteValueAMD64_OpblendMaskedInt32x16(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (blendMaskedInt32x16 x y mask)
	// result: (VPBLENDMDMasked512 x y (VPMOVVec32x16ToM <types.TypeMask> mask))
	for {
		x := v_0
		y := v_1
		mask := v_2
		v.reset(OpAMD64VPBLENDMDMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg3(x, y, v0)
		return true
	}
}
func rewriteValueAMD64_OpblendMaskedInt64x8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (blendMaskedInt64x8 x y mask)
	// result: (VPBLENDMQMasked512 x y (VPMOVVec64x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		y := v_1
		mask := v_2
		v.reset(OpAMD64VPBLENDMQMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg3(x, y, v0)
		return true
	}
}
func rewriteValueAMD64_OpblendMaskedInt8x64(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (blendMaskedInt8x64 x y mask)
	// result: (VPBLENDMBMasked512 x y (VPMOVVec8x64ToM <types.TypeMask> mask))
	for {
		x := v_0
		y := v_1
		mask := v_2
		v.reset(OpAMD64VPBLENDMBMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x64ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg3(x, y, v0)
		return true
	}
}
func rewriteValueAMD64_OpmoveMaskedFloat32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (moveMaskedFloat32x16 x mask)
	// result: (VMOVUPSMasked512 x (VPMOVVec32x16ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VMOVUPSMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpmoveMaskedFloat64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (moveMaskedFloat64x8 x mask)
	// result: (VMOVUPDMasked512 x (VPMOVVec64x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VMOVUPDMasked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpmoveMaskedInt16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (moveMaskedInt16x32 x mask)
	// result: (VMOVDQU16Masked512 x (VPMOVVec16x32ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VMOVDQU16Masked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x32ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpmoveMaskedInt32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (moveMaskedInt32x16 x mask)
	// result: (VMOVDQU32Masked512 x (VPMOVVec32x16ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VMOVDQU32Masked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpmoveMaskedInt64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (moveMaskedInt64x8 x mask)
	// result: (VMOVDQU64Masked512 x (VPMOVVec64x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VMOVDQU64Masked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpmoveMaskedInt8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (moveMaskedInt8x64 x mask)
	// result: (VMOVDQU8Masked512 x (VPMOVVec8x64ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VMOVDQU8Masked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x64ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpmoveMaskedUint16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (moveMaskedUint16x32 x mask)
	// result: (VMOVDQU16Masked512 x (VPMOVVec16x32ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VMOVDQU16Masked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec16x32ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpmoveMaskedUint32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (moveMaskedUint32x16 x mask)
	// result: (VMOVDQU32Masked512 x (VPMOVVec32x16ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VMOVDQU32Masked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec32x16ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpmoveMaskedUint64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (moveMaskedUint64x8 x mask)
	// result: (VMOVDQU64Masked512 x (VPMOVVec64x8ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VMOVDQU64Masked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec64x8ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueAMD64_OpmoveMaskedUint8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (moveMaskedUint8x64 x mask)
	// result: (VMOVDQU8Masked512 x (VPMOVVec8x64ToM <types.TypeMask> mask))
	for {
		x := v_0
		mask := v_1
		v.reset(OpAMD64VMOVDQU8Masked512)
		v0 := b.NewValue0(v.Pos, OpAMD64VPMOVVec8x64ToM, types.TypeMask)
		v0.AddArg(mask)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteBlockAMD64(b *Block) bool {
	typ := &b.Func.Config.Types
	switch b.Kind {
	case BlockAMD64EQ:
		// match: (EQ (TESTL (SHLL (MOVLconst [1]) x) y))
		// result: (UGE (BTL x y))
		for b.Controls[0].Op == OpAMD64TESTL {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				if v_0_0.Op != OpAMD64SHLL {
					continue
				}
				x := v_0_0.Args[1]
				v_0_0_0 := v_0_0.Args[0]
				if v_0_0_0.Op != OpAMD64MOVLconst || auxIntToInt32(v_0_0_0.AuxInt) != 1 {
					continue
				}
				y := v_0_1
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTL, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockAMD64UGE, v0)
				return true
			}
			break
		}
		// match: (EQ (TESTQ (SHLQ (MOVQconst [1]) x) y))
		// result: (UGE (BTQ x y))
		for b.Controls[0].Op == OpAMD64TESTQ {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				if v_0_0.Op != OpAMD64SHLQ {
					continue
				}
				x := v_0_0.Args[1]
				v_0_0_0 := v_0_0.Args[0]
				if v_0_0_0.Op != OpAMD64MOVQconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
					continue
				}
				y := v_0_1
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTQ, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockAMD64UGE, v0)
				return true
			}
			break
		}
		// match: (EQ (TESTLconst [c] x))
		// cond: isUnsignedPowerOfTwo(uint32(c))
		// result: (UGE (BTLconst [int8(log32u(uint32(c)))] x))
		for b.Controls[0].Op == OpAMD64TESTLconst {
			v_0 := b.Controls[0]
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if !(isUnsignedPowerOfTwo(uint32(c))) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpAMD64BTLconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(int8(log32u(uint32(c))))
			v0.AddArg(x)
			b.resetWithControl(BlockAMD64UGE, v0)
			return true
		}
		// match: (EQ (TESTQconst [c] x))
		// cond: isUnsignedPowerOfTwo(uint64(c))
		// result: (UGE (BTQconst [int8(log32u(uint32(c)))] x))
		for b.Controls[0].Op == OpAMD64TESTQconst {
			v_0 := b.Controls[0]
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if !(isUnsignedPowerOfTwo(uint64(c))) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(int8(log32u(uint32(c))))
			v0.AddArg(x)
			b.resetWithControl(BlockAMD64UGE, v0)
			return true
		}
		// match: (EQ (TESTQ (MOVQconst [c]) x))
		// cond: isUnsignedPowerOfTwo(uint64(c))
		// result: (UGE (BTQconst [int8(log64u(uint64(c)))] x))
		for b.Controls[0].Op == OpAMD64TESTQ {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				if v_0_0.Op != OpAMD64MOVQconst {
					continue
				}
				c := auxIntToInt64(v_0_0.AuxInt)
				x := v_0_1
				if !(isUnsignedPowerOfTwo(uint64(c))) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTQconst, types.TypeFlags)
				v0.AuxInt = int8ToAuxInt(int8(log64u(uint64(c))))
				v0.AddArg(x)
				b.resetWithControl(BlockAMD64UGE, v0)
				return true
			}
			break
		}
		// match: (EQ (TESTQ z1:(SHLQconst [63] (SHRQconst [63] x)) z2))
		// cond: z1==z2
		// result: (UGE (BTQconst [63] x))
		for b.Controls[0].Op == OpAMD64TESTQ {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				z1 := v_0_0
				if z1.Op != OpAMD64SHLQconst || auxIntToInt8(z1.AuxInt) != 63 {
					continue
				}
				z1_0 := z1.Args[0]
				if z1_0.Op != OpAMD64SHRQconst || auxIntToInt8(z1_0.AuxInt) != 63 {
					continue
				}
				x := z1_0.Args[0]
				z2 := v_0_1
				if !(z1 == z2) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTQconst, types.TypeFlags)
				v0.AuxInt = int8ToAuxInt(63)
				v0.AddArg(x)
				b.resetWithControl(BlockAMD64UGE, v0)
				return true
			}
			break
		}
		// match: (EQ (TESTL z1:(SHLLconst [31] (SHRQconst [31] x)) z2))
		// cond: z1==z2
		// result: (UGE (BTQconst [31] x))
		for b.Controls[0].Op == OpAMD64TESTL {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				z1 := v_0_0
				if z1.Op != OpAMD64SHLLconst || auxIntToInt8(z1.AuxInt) != 31 {
					continue
				}
				z1_0 := z1.Args[0]
				if z1_0.Op != OpAMD64SHRQconst || auxIntToInt8(z1_0.AuxInt) != 31 {
					continue
				}
				x := z1_0.Args[0]
				z2 := v_0_1
				if !(z1 == z2) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTQconst, types.TypeFlags)
				v0.AuxInt = int8ToAuxInt(31)
				v0.AddArg(x)
				b.resetWithControl(BlockAMD64UGE, v0)
				return true
			}
			break
		}
		// match: (EQ (TESTQ z1:(SHRQconst [63] (SHLQconst [63] x)) z2))
		// cond: z1==z2
		// result: (UGE (BTQconst [0] x))
		for b.Controls[0].Op == OpAMD64TESTQ {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				z1 := v_0_0
				if z1.Op != OpAMD64SHRQconst || auxIntToInt8(z1.AuxInt) != 63 {
					continue
				}
				z1_0 := z1.Args[0]
				if z1_0.Op != OpAMD64SHLQconst || auxIntToInt8(z1_0.AuxInt) != 63 {
					continue
				}
				x := z1_0.Args[0]
				z2 := v_0_1
				if !(z1 == z2) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTQconst, types.TypeFlags)
				v0.AuxInt = int8ToAuxInt(0)
				v0.AddArg(x)
				b.resetWithControl(BlockAMD64UGE, v0)
				return true
			}
			break
		}
		// match: (EQ (TESTL z1:(SHRLconst [31] (SHLLconst [31] x)) z2))
		// cond: z1==z2
		// result: (UGE (BTLconst [0] x))
		for b.Controls[0].Op == OpAMD64TESTL {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				z1 := v_0_0
				if z1.Op != OpAMD64SHRLconst || auxIntToInt8(z1.AuxInt) != 31 {
					continue
				}
				z1_0 := z1.Args[0]
				if z1_0.Op != OpAMD64SHLLconst || auxIntToInt8(z1_0.AuxInt) != 31 {
					continue
				}
				x := z1_0.Args[0]
				z2 := v_0_1
				if !(z1 == z2) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTLconst, types.TypeFlags)
				v0.AuxInt = int8ToAuxInt(0)
				v0.AddArg(x)
				b.resetWithControl(BlockAMD64UGE, v0)
				return true
			}
			break
		}
		// match: (EQ (TESTQ z1:(SHRQconst [63] x) z2))
		// cond: z1==z2
		// result: (UGE (BTQconst [63] x))
		for b.Controls[0].Op == OpAMD64TESTQ {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				z1 := v_0_0
				if z1.Op != OpAMD64SHRQconst || auxIntToInt8(z1.AuxInt) != 63 {
					continue
				}
				x := z1.Args[0]
				z2 := v_0_1
				if !(z1 == z2) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTQconst, types.TypeFlags)
				v0.AuxInt = int8ToAuxInt(63)
				v0.AddArg(x)
				b.resetWithControl(BlockAMD64UGE, v0)
				return true
			}
			break
		}
		// match: (EQ (TESTL z1:(SHRLconst [31] x) z2))
		// cond: z1==z2
		// result: (UGE (BTLconst [31] x))
		for b.Controls[0].Op == OpAMD64TESTL {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				z1 := v_0_0
				if z1.Op != OpAMD64SHRLconst || auxIntToInt8(z1.AuxInt) != 31 {
					continue
				}
				x := z1.Args[0]
				z2 := v_0_1
				if !(z1 == z2) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTLconst, types.TypeFlags)
				v0.AuxInt = int8ToAuxInt(31)
				v0.AddArg(x)
				b.resetWithControl(BlockAMD64UGE, v0)
				return true
			}
			break
		}
		// match: (EQ (InvertFlags cmp) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == OpAMD64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64EQ, cmp)
			return true
		}
		// match: (EQ (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagEQ {
			b.Reset(BlockFirst)
			return true
		}
		// match: (EQ (FlagLT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagLT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (EQ (FlagLT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagLT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (EQ (FlagGT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagGT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (EQ (FlagGT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagGT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (EQ (TESTQ s:(Select0 blsr:(BLSRQ _)) s) yes no)
		// result: (EQ (Select1 <types.TypeFlags> blsr) yes no)
		for b.Controls[0].Op == OpAMD64TESTQ {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				s := v_0_0
				if s.Op != OpSelect0 {
					continue
				}
				blsr := s.Args[0]
				if blsr.Op != OpAMD64BLSRQ || s != v_0_1 {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v0.AddArg(blsr)
				b.resetWithControl(BlockAMD64EQ, v0)
				return true
			}
			break
		}
		// match: (EQ (TESTL s:(Select0 blsr:(BLSRL _)) s) yes no)
		// result: (EQ (Select1 <types.TypeFlags> blsr) yes no)
		for b.Controls[0].Op == OpAMD64TESTL {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				s := v_0_0
				if s.Op != OpSelect0 {
					continue
				}
				blsr := s.Args[0]
				if blsr.Op != OpAMD64BLSRL || s != v_0_1 {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v0.AddArg(blsr)
				b.resetWithControl(BlockAMD64EQ, v0)
				return true
			}
			break
		}
		// match: (EQ t:(TESTQ a:(ADDQconst [c] x) a))
		// cond: t.Uses == 1 && flagify(a)
		// result: (EQ (Select1 <types.TypeFlags> a.Args[0]))
		for b.Controls[0].Op == OpAMD64TESTQ {
			t := b.Controls[0]
			_ = t.Args[1]
			t_0 := t.Args[0]
			t_1 := t.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, t_0, t_1 = _i0+1, t_1, t_0 {
				a := t_0
				if a.Op != OpAMD64ADDQconst {
					continue
				}
				if a != t_1 || !(t.Uses == 1 && flagify(a)) {
					continue
				}
				v0 := b.NewValue0(t.Pos, OpSelect1, types.TypeFlags)
				v0.AddArg(a.Args[0])
				b.resetWithControl(BlockAMD64EQ, v0)
				return true
			}
			break
		}
		// match: (EQ t:(TESTL a:(ADDLconst [c] x) a))
		// cond: t.Uses == 1 && flagify(a)
		// result: (EQ (Select1 <types.TypeFlags> a.Args[0]))
		for b.Controls[0].Op == OpAMD64TESTL {
			t := b.Controls[0]
			_ = t.Args[1]
			t_0 := t.Args[0]
			t_1 := t.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, t_0, t_1 = _i0+1, t_1, t_0 {
				a := t_0
				if a.Op != OpAMD64ADDLconst {
					continue
				}
				if a != t_1 || !(t.Uses == 1 && flagify(a)) {
					continue
				}
				v0 := b.NewValue0(t.Pos, OpSelect1, types.TypeFlags)
				v0.AddArg(a.Args[0])
				b.resetWithControl(BlockAMD64EQ, v0)
				return true
			}
			break
		}
	case BlockAMD64GE:
		// match: (GE c:(CMPQconst [128] z) yes no)
		// cond: c.Uses == 1
		// result: (GT (CMPQconst [127] z) yes no)
		for b.Controls[0].Op == OpAMD64CMPQconst {
			c := b.Controls[0]
			if auxIntToInt32(c.AuxInt) != 128 {
				break
			}
			z := c.Args[0]
			if !(c.Uses == 1) {
				break
			}
			v0 := b.NewValue0(c.Pos, OpAMD64CMPQconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(127)
			v0.AddArg(z)
			b.resetWithControl(BlockAMD64GT, v0)
			return true
		}
		// match: (GE c:(CMPLconst [128] z) yes no)
		// cond: c.Uses == 1
		// result: (GT (CMPLconst [127] z) yes no)
		for b.Controls[0].Op == OpAMD64CMPLconst {
			c := b.Controls[0]
			if auxIntToInt32(c.AuxInt) != 128 {
				break
			}
			z := c.Args[0]
			if !(c.Uses == 1) {
				break
			}
			v0 := b.NewValue0(c.Pos, OpAMD64CMPLconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(127)
			v0.AddArg(z)
			b.resetWithControl(BlockAMD64GT, v0)
			return true
		}
		// match: (GE (InvertFlags cmp) yes no)
		// result: (LE cmp yes no)
		for b.Controls[0].Op == OpAMD64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64LE, cmp)
			return true
		}
		// match: (GE (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagEQ {
			b.Reset(BlockFirst)
			return true
		}
		// match: (GE (FlagLT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagLT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (GE (FlagLT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagLT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (GE (FlagGT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagGT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (GE (FlagGT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagGT_UGT {
			b.Reset(BlockFirst)
			return true
		}
	case BlockAMD64GT:
		// match: (GT (InvertFlags cmp) yes no)
		// result: (LT cmp yes no)
		for b.Controls[0].Op == OpAMD64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64LT, cmp)
			return true
		}
		// match: (GT (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagEQ {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (GT (FlagLT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagLT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (GT (FlagLT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagLT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (GT (FlagGT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagGT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (GT (FlagGT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagGT_UGT {
			b.Reset(BlockFirst)
			return true
		}
	case BlockIf:
		// match: (If (SETL cmp) yes no)
		// result: (LT cmp yes no)
		for b.Controls[0].Op == OpAMD64SETL {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64LT, cmp)
			return true
		}
		// match: (If (SETLE cmp) yes no)
		// result: (LE cmp yes no)
		for b.Controls[0].Op == OpAMD64SETLE {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64LE, cmp)
			return true
		}
		// match: (If (SETG cmp) yes no)
		// result: (GT cmp yes no)
		for b.Controls[0].Op == OpAMD64SETG {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64GT, cmp)
			return true
		}
		// match: (If (SETGE cmp) yes no)
		// result: (GE cmp yes no)
		for b.Controls[0].Op == OpAMD64SETGE {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64GE, cmp)
			return true
		}
		// match: (If (SETEQ cmp) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == OpAMD64SETEQ {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64EQ, cmp)
			return true
		}
		// match: (If (SETNE cmp) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == OpAMD64SETNE {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64NE, cmp)
			return true
		}
		// match: (If (SETB cmp) yes no)
		// result: (ULT cmp yes no)
		for b.Controls[0].Op == OpAMD64SETB {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64ULT, cmp)
			return true
		}
		// match: (If (SETBE cmp) yes no)
		// result: (ULE cmp yes no)
		for b.Controls[0].Op == OpAMD64SETBE {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64ULE, cmp)
			return true
		}
		// match: (If (SETA cmp) yes no)
		// result: (UGT cmp yes no)
		for b.Controls[0].Op == OpAMD64SETA {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64UGT, cmp)
			return true
		}
		// match: (If (SETAE cmp) yes no)
		// result: (UGE cmp yes no)
		for b.Controls[0].Op == OpAMD64SETAE {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64UGE, cmp)
			return true
		}
		// match: (If (SETO cmp) yes no)
		// result: (OS cmp yes no)
		for b.Controls[0].Op == OpAMD64SETO {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64OS, cmp)
			return true
		}
		// match: (If (SETGF cmp) yes no)
		// result: (UGT cmp yes no)
		for b.Controls[0].Op == OpAMD64SETGF {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64UGT, cmp)
			return true
		}
		// match: (If (SETGEF cmp) yes no)
		// result: (UGE cmp yes no)
		for b.Controls[0].Op == OpAMD64SETGEF {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64UGE, cmp)
			return true
		}
		// match: (If (SETEQF cmp) yes no)
		// result: (EQF cmp yes no)
		for b.Controls[0].Op == OpAMD64SETEQF {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64EQF, cmp)
			return true
		}
		// match: (If (SETNEF cmp) yes no)
		// result: (NEF cmp yes no)
		for b.Controls[0].Op == OpAMD64SETNEF {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64NEF, cmp)
			return true
		}
		// match: (If cond yes no)
		// result: (NE (TESTB cond cond) yes no)
		for {
			cond := b.Controls[0]
			v0 := b.NewValue0(cond.Pos, OpAMD64TESTB, types.TypeFlags)
			v0.AddArg2(cond, cond)
			b.resetWithControl(BlockAMD64NE, v0)
			return true
		}
	case BlockJumpTable:
		// match: (JumpTable idx)
		// result: (JUMPTABLE {makeJumpTableSym(b)} idx (LEAQ <typ.Uintptr> {makeJumpTableSym(b)} (SB)))
		for {
			idx := b.Controls[0]
			v0 := b.NewValue0(b.Pos, OpAMD64LEAQ, typ.Uintptr)
			v0.Aux = symToAux(makeJumpTableSym(b))
			v1 := b.NewValue0(b.Pos, OpSB, typ.Uintptr)
			v0.AddArg(v1)
			b.resetWithControl2(BlockAMD64JUMPTABLE, idx, v0)
			b.Aux = symToAux(makeJumpTableSym(b))
			return true
		}
	case BlockAMD64LE:
		// match: (LE (InvertFlags cmp) yes no)
		// result: (GE cmp yes no)
		for b.Controls[0].Op == OpAMD64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64GE, cmp)
			return true
		}
		// match: (LE (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagEQ {
			b.Reset(BlockFirst)
			return true
		}
		// match: (LE (FlagLT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagLT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (LE (FlagLT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagLT_UGT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (LE (FlagGT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagGT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (LE (FlagGT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagGT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
	case BlockAMD64LT:
		// match: (LT c:(CMPQconst [128] z) yes no)
		// cond: c.Uses == 1
		// result: (LE (CMPQconst [127] z) yes no)
		for b.Controls[0].Op == OpAMD64CMPQconst {
			c := b.Controls[0]
			if auxIntToInt32(c.AuxInt) != 128 {
				break
			}
			z := c.Args[0]
			if !(c.Uses == 1) {
				break
			}
			v0 := b.NewValue0(c.Pos, OpAMD64CMPQconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(127)
			v0.AddArg(z)
			b.resetWithControl(BlockAMD64LE, v0)
			return true
		}
		// match: (LT c:(CMPLconst [128] z) yes no)
		// cond: c.Uses == 1
		// result: (LE (CMPLconst [127] z) yes no)
		for b.Controls[0].Op == OpAMD64CMPLconst {
			c := b.Controls[0]
			if auxIntToInt32(c.AuxInt) != 128 {
				break
			}
			z := c.Args[0]
			if !(c.Uses == 1) {
				break
			}
			v0 := b.NewValue0(c.Pos, OpAMD64CMPLconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(127)
			v0.AddArg(z)
			b.resetWithControl(BlockAMD64LE, v0)
			return true
		}
		// match: (LT (InvertFlags cmp) yes no)
		// result: (GT cmp yes no)
		for b.Controls[0].Op == OpAMD64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64GT, cmp)
			return true
		}
		// match: (LT (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagEQ {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (LT (FlagLT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagLT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (LT (FlagLT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagLT_UGT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (LT (FlagGT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagGT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (LT (FlagGT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagGT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
	case BlockAMD64NE:
		// match: (NE (TESTB (SETL cmp) (SETL cmp)) yes no)
		// result: (LT cmp yes no)
		for b.Controls[0].Op == OpAMD64TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SETL {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpAMD64SETL || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(BlockAMD64LT, cmp)
			return true
		}
		// match: (NE (TESTB (SETLE cmp) (SETLE cmp)) yes no)
		// result: (LE cmp yes no)
		for b.Controls[0].Op == OpAMD64TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SETLE {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpAMD64SETLE || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(BlockAMD64LE, cmp)
			return true
		}
		// match: (NE (TESTB (SETG cmp) (SETG cmp)) yes no)
		// result: (GT cmp yes no)
		for b.Controls[0].Op == OpAMD64TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SETG {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpAMD64SETG || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(BlockAMD64GT, cmp)
			return true
		}
		// match: (NE (TESTB (SETGE cmp) (SETGE cmp)) yes no)
		// result: (GE cmp yes no)
		for b.Controls[0].Op == OpAMD64TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SETGE {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpAMD64SETGE || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(BlockAMD64GE, cmp)
			return true
		}
		// match: (NE (TESTB (SETEQ cmp) (SETEQ cmp)) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == OpAMD64TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SETEQ {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpAMD64SETEQ || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(BlockAMD64EQ, cmp)
			return true
		}
		// match: (NE (TESTB (SETNE cmp) (SETNE cmp)) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == OpAMD64TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SETNE {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpAMD64SETNE || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(BlockAMD64NE, cmp)
			return true
		}
		// match: (NE (TESTB (SETB cmp) (SETB cmp)) yes no)
		// result: (ULT cmp yes no)
		for b.Controls[0].Op == OpAMD64TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SETB {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpAMD64SETB || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(BlockAMD64ULT, cmp)
			return true
		}
		// match: (NE (TESTB (SETBE cmp) (SETBE cmp)) yes no)
		// result: (ULE cmp yes no)
		for b.Controls[0].Op == OpAMD64TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SETBE {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpAMD64SETBE || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(BlockAMD64ULE, cmp)
			return true
		}
		// match: (NE (TESTB (SETA cmp) (SETA cmp)) yes no)
		// result: (UGT cmp yes no)
		for b.Controls[0].Op == OpAMD64TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SETA {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpAMD64SETA || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(BlockAMD64UGT, cmp)
			return true
		}
		// match: (NE (TESTB (SETAE cmp) (SETAE cmp)) yes no)
		// result: (UGE cmp yes no)
		for b.Controls[0].Op == OpAMD64TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SETAE {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpAMD64SETAE || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(BlockAMD64UGE, cmp)
			return true
		}
		// match: (NE (TESTB (SETO cmp) (SETO cmp)) yes no)
		// result: (OS cmp yes no)
		for b.Controls[0].Op == OpAMD64TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SETO {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpAMD64SETO || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(BlockAMD64OS, cmp)
			return true
		}
		// match: (NE (TESTL (SHLL (MOVLconst [1]) x) y))
		// result: (ULT (BTL x y))
		for b.Controls[0].Op == OpAMD64TESTL {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				if v_0_0.Op != OpAMD64SHLL {
					continue
				}
				x := v_0_0.Args[1]
				v_0_0_0 := v_0_0.Args[0]
				if v_0_0_0.Op != OpAMD64MOVLconst || auxIntToInt32(v_0_0_0.AuxInt) != 1 {
					continue
				}
				y := v_0_1
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTL, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockAMD64ULT, v0)
				return true
			}
			break
		}
		// match: (NE (TESTQ (SHLQ (MOVQconst [1]) x) y))
		// result: (ULT (BTQ x y))
		for b.Controls[0].Op == OpAMD64TESTQ {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				if v_0_0.Op != OpAMD64SHLQ {
					continue
				}
				x := v_0_0.Args[1]
				v_0_0_0 := v_0_0.Args[0]
				if v_0_0_0.Op != OpAMD64MOVQconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
					continue
				}
				y := v_0_1
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTQ, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockAMD64ULT, v0)
				return true
			}
			break
		}
		// match: (NE (TESTLconst [c] x))
		// cond: isUnsignedPowerOfTwo(uint32(c))
		// result: (ULT (BTLconst [int8(log32u(uint32(c)))] x))
		for b.Controls[0].Op == OpAMD64TESTLconst {
			v_0 := b.Controls[0]
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if !(isUnsignedPowerOfTwo(uint32(c))) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpAMD64BTLconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(int8(log32u(uint32(c))))
			v0.AddArg(x)
			b.resetWithControl(BlockAMD64ULT, v0)
			return true
		}
		// match: (NE (TESTQconst [c] x))
		// cond: isUnsignedPowerOfTwo(uint64(c))
		// result: (ULT (BTQconst [int8(log32u(uint32(c)))] x))
		for b.Controls[0].Op == OpAMD64TESTQconst {
			v_0 := b.Controls[0]
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if !(isUnsignedPowerOfTwo(uint64(c))) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(int8(log32u(uint32(c))))
			v0.AddArg(x)
			b.resetWithControl(BlockAMD64ULT, v0)
			return true
		}
		// match: (NE (TESTQ (MOVQconst [c]) x))
		// cond: isUnsignedPowerOfTwo(uint64(c))
		// result: (ULT (BTQconst [int8(log64u(uint64(c)))] x))
		for b.Controls[0].Op == OpAMD64TESTQ {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				if v_0_0.Op != OpAMD64MOVQconst {
					continue
				}
				c := auxIntToInt64(v_0_0.AuxInt)
				x := v_0_1
				if !(isUnsignedPowerOfTwo(uint64(c))) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTQconst, types.TypeFlags)
				v0.AuxInt = int8ToAuxInt(int8(log64u(uint64(c))))
				v0.AddArg(x)
				b.resetWithControl(BlockAMD64ULT, v0)
				return true
			}
			break
		}
		// match: (NE (TESTQ z1:(SHLQconst [63] (SHRQconst [63] x)) z2))
		// cond: z1==z2
		// result: (ULT (BTQconst [63] x))
		for b.Controls[0].Op == OpAMD64TESTQ {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				z1 := v_0_0
				if z1.Op != OpAMD64SHLQconst || auxIntToInt8(z1.AuxInt) != 63 {
					continue
				}
				z1_0 := z1.Args[0]
				if z1_0.Op != OpAMD64SHRQconst || auxIntToInt8(z1_0.AuxInt) != 63 {
					continue
				}
				x := z1_0.Args[0]
				z2 := v_0_1
				if !(z1 == z2) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTQconst, types.TypeFlags)
				v0.AuxInt = int8ToAuxInt(63)
				v0.AddArg(x)
				b.resetWithControl(BlockAMD64ULT, v0)
				return true
			}
			break
		}
		// match: (NE (TESTL z1:(SHLLconst [31] (SHRQconst [31] x)) z2))
		// cond: z1==z2
		// result: (ULT (BTQconst [31] x))
		for b.Controls[0].Op == OpAMD64TESTL {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				z1 := v_0_0
				if z1.Op != OpAMD64SHLLconst || auxIntToInt8(z1.AuxInt) != 31 {
					continue
				}
				z1_0 := z1.Args[0]
				if z1_0.Op != OpAMD64SHRQconst || auxIntToInt8(z1_0.AuxInt) != 31 {
					continue
				}
				x := z1_0.Args[0]
				z2 := v_0_1
				if !(z1 == z2) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTQconst, types.TypeFlags)
				v0.AuxInt = int8ToAuxInt(31)
				v0.AddArg(x)
				b.resetWithControl(BlockAMD64ULT, v0)
				return true
			}
			break
		}
		// match: (NE (TESTQ z1:(SHRQconst [63] (SHLQconst [63] x)) z2))
		// cond: z1==z2
		// result: (ULT (BTQconst [0] x))
		for b.Controls[0].Op == OpAMD64TESTQ {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				z1 := v_0_0
				if z1.Op != OpAMD64SHRQconst || auxIntToInt8(z1.AuxInt) != 63 {
					continue
				}
				z1_0 := z1.Args[0]
				if z1_0.Op != OpAMD64SHLQconst || auxIntToInt8(z1_0.AuxInt) != 63 {
					continue
				}
				x := z1_0.Args[0]
				z2 := v_0_1
				if !(z1 == z2) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTQconst, types.TypeFlags)
				v0.AuxInt = int8ToAuxInt(0)
				v0.AddArg(x)
				b.resetWithControl(BlockAMD64ULT, v0)
				return true
			}
			break
		}
		// match: (NE (TESTL z1:(SHRLconst [31] (SHLLconst [31] x)) z2))
		// cond: z1==z2
		// result: (ULT (BTLconst [0] x))
		for b.Controls[0].Op == OpAMD64TESTL {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				z1 := v_0_0
				if z1.Op != OpAMD64SHRLconst || auxIntToInt8(z1.AuxInt) != 31 {
					continue
				}
				z1_0 := z1.Args[0]
				if z1_0.Op != OpAMD64SHLLconst || auxIntToInt8(z1_0.AuxInt) != 31 {
					continue
				}
				x := z1_0.Args[0]
				z2 := v_0_1
				if !(z1 == z2) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTLconst, types.TypeFlags)
				v0.AuxInt = int8ToAuxInt(0)
				v0.AddArg(x)
				b.resetWithControl(BlockAMD64ULT, v0)
				return true
			}
			break
		}
		// match: (NE (TESTQ z1:(SHRQconst [63] x) z2))
		// cond: z1==z2
		// result: (ULT (BTQconst [63] x))
		for b.Controls[0].Op == OpAMD64TESTQ {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				z1 := v_0_0
				if z1.Op != OpAMD64SHRQconst || auxIntToInt8(z1.AuxInt) != 63 {
					continue
				}
				x := z1.Args[0]
				z2 := v_0_1
				if !(z1 == z2) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTQconst, types.TypeFlags)
				v0.AuxInt = int8ToAuxInt(63)
				v0.AddArg(x)
				b.resetWithControl(BlockAMD64ULT, v0)
				return true
			}
			break
		}
		// match: (NE (TESTL z1:(SHRLconst [31] x) z2))
		// cond: z1==z2
		// result: (ULT (BTLconst [31] x))
		for b.Controls[0].Op == OpAMD64TESTL {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				z1 := v_0_0
				if z1.Op != OpAMD64SHRLconst || auxIntToInt8(z1.AuxInt) != 31 {
					continue
				}
				x := z1.Args[0]
				z2 := v_0_1
				if !(z1 == z2) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTLconst, types.TypeFlags)
				v0.AuxInt = int8ToAuxInt(31)
				v0.AddArg(x)
				b.resetWithControl(BlockAMD64ULT, v0)
				return true
			}
			break
		}
		// match: (NE (TESTB (SETGF cmp) (SETGF cmp)) yes no)
		// result: (UGT cmp yes no)
		for b.Controls[0].Op == OpAMD64TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SETGF {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpAMD64SETGF || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(BlockAMD64UGT, cmp)
			return true
		}
		// match: (NE (TESTB (SETGEF cmp) (SETGEF cmp)) yes no)
		// result: (UGE cmp yes no)
		for b.Controls[0].Op == OpAMD64TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SETGEF {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpAMD64SETGEF || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(BlockAMD64UGE, cmp)
			return true
		}
		// match: (NE (TESTB (SETEQF cmp) (SETEQF cmp)) yes no)
		// result: (EQF cmp yes no)
		for b.Controls[0].Op == OpAMD64TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SETEQF {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpAMD64SETEQF || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(BlockAMD64EQF, cmp)
			return true
		}
		// match: (NE (TESTB (SETNEF cmp) (SETNEF cmp)) yes no)
		// result: (NEF cmp yes no)
		for b.Controls[0].Op == OpAMD64TESTB {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SETNEF {
				break
			}
			cmp := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpAMD64SETNEF || cmp != v_0_1.Args[0] {
				break
			}
			b.resetWithControl(BlockAMD64NEF, cmp)
			return true
		}
		// match: (NE (InvertFlags cmp) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == OpAMD64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64NE, cmp)
			return true
		}
		// match: (NE (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagEQ {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (NE (FlagLT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagLT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (NE (FlagLT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagLT_UGT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (NE (FlagGT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagGT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (NE (FlagGT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagGT_UGT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (NE (TESTQ s:(Select0 blsr:(BLSRQ _)) s) yes no)
		// result: (NE (Select1 <types.TypeFlags> blsr) yes no)
		for b.Controls[0].Op == OpAMD64TESTQ {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				s := v_0_0
				if s.Op != OpSelect0 {
					continue
				}
				blsr := s.Args[0]
				if blsr.Op != OpAMD64BLSRQ || s != v_0_1 {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v0.AddArg(blsr)
				b.resetWithControl(BlockAMD64NE, v0)
				return true
			}
			break
		}
		// match: (NE (TESTL s:(Select0 blsr:(BLSRL _)) s) yes no)
		// result: (NE (Select1 <types.TypeFlags> blsr) yes no)
		for b.Controls[0].Op == OpAMD64TESTL {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				s := v_0_0
				if s.Op != OpSelect0 {
					continue
				}
				blsr := s.Args[0]
				if blsr.Op != OpAMD64BLSRL || s != v_0_1 {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v0.AddArg(blsr)
				b.resetWithControl(BlockAMD64NE, v0)
				return true
			}
			break
		}
		// match: (NE t:(TESTQ a:(ADDQconst [c] x) a))
		// cond: t.Uses == 1 && flagify(a)
		// result: (NE (Select1 <types.TypeFlags> a.Args[0]))
		for b.Controls[0].Op == OpAMD64TESTQ {
			t := b.Controls[0]
			_ = t.Args[1]
			t_0 := t.Args[0]
			t_1 := t.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, t_0, t_1 = _i0+1, t_1, t_0 {
				a := t_0
				if a.Op != OpAMD64ADDQconst {
					continue
				}
				if a != t_1 || !(t.Uses == 1 && flagify(a)) {
					continue
				}
				v0 := b.NewValue0(t.Pos, OpSelect1, types.TypeFlags)
				v0.AddArg(a.Args[0])
				b.resetWithControl(BlockAMD64NE, v0)
				return true
			}
			break
		}
		// match: (NE t:(TESTL a:(ADDLconst [c] x) a))
		// cond: t.Uses == 1 && flagify(a)
		// result: (NE (Select1 <types.TypeFlags> a.Args[0]))
		for b.Controls[0].Op == OpAMD64TESTL {
			t := b.Controls[0]
			_ = t.Args[1]
			t_0 := t.Args[0]
			t_1 := t.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, t_0, t_1 = _i0+1, t_1, t_0 {
				a := t_0
				if a.Op != OpAMD64ADDLconst {
					continue
				}
				if a != t_1 || !(t.Uses == 1 && flagify(a)) {
					continue
				}
				v0 := b.NewValue0(t.Pos, OpSelect1, types.TypeFlags)
				v0.AddArg(a.Args[0])
				b.resetWithControl(BlockAMD64NE, v0)
				return true
			}
			break
		}
	case BlockAMD64UGE:
		// match: (UGE (TESTQ x x) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64TESTQ {
			v_0 := b.Controls[0]
			x := v_0.Args[1]
			if x != v_0.Args[0] {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (UGE (TESTL x x) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64TESTL {
			v_0 := b.Controls[0]
			x := v_0.Args[1]
			if x != v_0.Args[0] {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (UGE (TESTW x x) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64TESTW {
			v_0 := b.Controls[0]
			x := v_0.Args[1]
			if x != v_0.Args[0] {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (UGE (TESTB x x) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64TESTB {
			v_0 := b.Controls[0]
			x := v_0.Args[1]
			if x != v_0.Args[0] {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (UGE (InvertFlags cmp) yes no)
		// result: (ULE cmp yes no)
		for b.Controls[0].Op == OpAMD64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64ULE, cmp)
			return true
		}
		// match: (UGE (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagEQ {
			b.Reset(BlockFirst)
			return true
		}
		// match: (UGE (FlagLT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagLT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (UGE (FlagLT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagLT_UGT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (UGE (FlagGT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagGT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (UGE (FlagGT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagGT_UGT {
			b.Reset(BlockFirst)
			return true
		}
	case BlockAMD64UGT:
		// match: (UGT (InvertFlags cmp) yes no)
		// result: (ULT cmp yes no)
		for b.Controls[0].Op == OpAMD64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64ULT, cmp)
			return true
		}
		// match: (UGT (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagEQ {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (UGT (FlagLT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagLT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (UGT (FlagLT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagLT_UGT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (UGT (FlagGT_ULT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagGT_ULT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (UGT (FlagGT_UGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagGT_UGT {
			b.Reset(BlockFirst)
			return true
		}
	case BlockAMD64ULE:
		// match: (ULE (InvertFlags cmp) yes no)
		// result: (UGE cmp yes no)
		for b.Controls[0].Op == OpAMD64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64UGE, cmp)
			return true
		}
		// match: (ULE (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagEQ {
			b.Reset(BlockFirst)
			return true
		}
		// match: (ULE (FlagLT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagLT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (ULE (FlagLT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagLT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (ULE (FlagGT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagGT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (ULE (FlagGT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagGT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
	case BlockAMD64ULT:
		// match: (ULT (TESTQ x x) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64TESTQ {
			v_0 := b.Controls[0]
			x := v_0.Args[1]
			if x != v_0.Args[0] {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (ULT (TESTL x x) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64TESTL {
			v_0 := b.Controls[0]
			x := v_0.Args[1]
			if x != v_0.Args[0] {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (ULT (TESTW x x) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64TESTW {
			v_0 := b.Controls[0]
			x := v_0.Args[1]
			if x != v_0.Args[0] {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (ULT (TESTB x x) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64TESTB {
			v_0 := b.Controls[0]
			x := v_0.Args[1]
			if x != v_0.Args[0] {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (ULT (InvertFlags cmp) yes no)
		// result: (UGT cmp yes no)
		for b.Controls[0].Op == OpAMD64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockAMD64UGT, cmp)
			return true
		}
		// match: (ULT (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagEQ {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (ULT (FlagLT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagLT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (ULT (FlagLT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagLT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (ULT (FlagGT_ULT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpAMD64FlagGT_ULT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (ULT (FlagGT_UGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpAMD64FlagGT_UGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
	}
	return false
}
