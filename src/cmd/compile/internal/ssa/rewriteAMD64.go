// Code generated from gen/AMD64.rules; DO NOT EDIT.
// generated with: cd gen; go run *.go

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
	case OpAMD64BTCLconst:
		return rewriteValueAMD64_OpAMD64BTCLconst(v)
	case OpAMD64BTCQconst:
		return rewriteValueAMD64_OpAMD64BTCQconst(v)
	case OpAMD64BTLconst:
		return rewriteValueAMD64_OpAMD64BTLconst(v)
	case OpAMD64BTQconst:
		return rewriteValueAMD64_OpAMD64BTQconst(v)
	case OpAMD64BTRLconst:
		return rewriteValueAMD64_OpAMD64BTRLconst(v)
	case OpAMD64BTRQconst:
		return rewriteValueAMD64_OpAMD64BTRQconst(v)
	case OpAMD64BTSLconst:
		return rewriteValueAMD64_OpAMD64BTSLconst(v)
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
	case OpAMD64SARXL:
		return rewriteValueAMD64_OpAMD64SARXL(v)
	case OpAMD64SARXLload:
		return rewriteValueAMD64_OpAMD64SARXLload(v)
	case OpAMD64SARXQ:
		return rewriteValueAMD64_OpAMD64SARXQ(v)
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
	case OpAMD64SHLXL:
		return rewriteValueAMD64_OpAMD64SHLXL(v)
	case OpAMD64SHLXLload:
		return rewriteValueAMD64_OpAMD64SHLXLload(v)
	case OpAMD64SHLXQ:
		return rewriteValueAMD64_OpAMD64SHLXQ(v)
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
	case OpAMD64SHRXL:
		return rewriteValueAMD64_OpAMD64SHRXL(v)
	case OpAMD64SHRXLload:
		return rewriteValueAMD64_OpAMD64SHRXLload(v)
	case OpAMD64SHRXQ:
		return rewriteValueAMD64_OpAMD64SHRXQ(v)
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
	case OpAddPtr:
		v.Op = OpAMD64ADDQ
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
	case OpAtomicAdd32:
		return rewriteValueAMD64_OpAtomicAdd32(v)
	case OpAtomicAdd64:
		return rewriteValueAMD64_OpAtomicAdd64(v)
	case OpAtomicAnd32:
		return rewriteValueAMD64_OpAtomicAnd32(v)
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
	case OpBswap32:
		v.Op = OpAMD64BSWAPL
		return true
	case OpBswap64:
		v.Op = OpAMD64BSWAPQ
		return true
	case OpCeil:
		return rewriteValueAMD64_OpCeil(v)
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
	case OpCvtBoolToUint8:
		v.Op = OpCopy
		return true
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
	case OpFMA:
		return rewriteValueAMD64_OpFMA(v)
	case OpFloor:
		return rewriteValueAMD64_OpFloor(v)
	case OpGetCallerPC:
		v.Op = OpAMD64LoweredGetCallerPC
		return true
	case OpGetCallerSP:
		v.Op = OpAMD64LoweredGetCallerSP
		return true
	case OpGetClosurePtr:
		v.Op = OpAMD64LoweredGetClosurePtr
		return true
	case OpGetG:
		return rewriteValueAMD64_OpGetG(v)
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
	case OpLoad:
		return rewriteValueAMD64_OpLoad(v)
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
	case OpOffPtr:
		return rewriteValueAMD64_OpOffPtr(v)
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
	case OpPanicBounds:
		return rewriteValueAMD64_OpPanicBounds(v)
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
	case OpRound32F:
		v.Op = OpCopy
		return true
	case OpRound64F:
		v.Op = OpCopy
		return true
	case OpRoundToEven:
		return rewriteValueAMD64_OpRoundToEven(v)
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
	case OpSelect0:
		return rewriteValueAMD64_OpSelect0(v)
	case OpSelect1:
		return rewriteValueAMD64_OpSelect1(v)
	case OpSelectN:
		return rewriteValueAMD64_OpSelectN(v)
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
	case OpStaticCall:
		v.Op = OpAMD64CALLstatic
		return true
	case OpStore:
		return rewriteValueAMD64_OpStore(v)
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
	case OpSubPtr:
		v.Op = OpAMD64SUBQ
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
	// match: (ADDL x (SHLLconst [1] y))
	// result: (LEAL2 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64SHLLconst || auxIntToInt8(v_1.AuxInt) != 1 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpAMD64LEAL2)
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
	// match: (ADDLconst [c] (SHLLconst [1] x))
	// result: (LEAL1 [c] x x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64SHLLconst || auxIntToInt8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
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
	// match: (ADDQ x (MOVQconst [c]))
	// cond: is32Bit(c)
	// result: (ADDQconst [int32(c)] x)
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
	// match: (ADDQ x (SHLQconst [1] y))
	// result: (LEAQ2 x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64SHLQconst || auxIntToInt8(v_1.AuxInt) != 1 {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpAMD64LEAQ2)
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
	// match: (ADDQconst [c] (SHLQconst [1] x))
	// result: (LEAQ1 [c] x x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64SHLQconst || auxIntToInt8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
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
	// match: (ANDL (NOTL (SHLXL (MOVLconst [1]) y)) x)
	// result: (BTRL x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64NOTL {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SHLXL {
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
	// match: (ANDL (MOVLconst [c]) x)
	// cond: isUint32PowerOfTwo(int64(^c)) && uint64(^c) >= 128
	// result: (BTRLconst [int8(log32(^c))] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64MOVLconst {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			x := v_1
			if !(isUint32PowerOfTwo(int64(^c)) && uint64(^c) >= 128) {
				continue
			}
			v.reset(OpAMD64BTRLconst)
			v.AuxInt = int8ToAuxInt(int8(log32(^c)))
			v.AddArg(x)
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
	// match: (ANDL x (ADDLconst [-1] x))
	// cond: buildcfg.GOAMD64 >= 3
	// result: (BLSRL x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64ADDLconst || auxIntToInt32(v_1.AuxInt) != -1 || x != v_1.Args[0] || !(buildcfg.GOAMD64 >= 3) {
				continue
			}
			v.reset(OpAMD64BLSRL)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64ANDLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ANDLconst [c] x)
	// cond: isUint32PowerOfTwo(int64(^c)) && uint64(^c) >= 128
	// result: (BTRLconst [int8(log32(^c))] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isUint32PowerOfTwo(int64(^c)) && uint64(^c) >= 128) {
			break
		}
		v.reset(OpAMD64BTRLconst)
		v.AuxInt = int8ToAuxInt(int8(log32(^c)))
		v.AddArg(x)
		return true
	}
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
	// match: (ANDLconst [c] (BTRLconst [d] x))
	// result: (ANDLconst [c &^ (1<<uint32(d))] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64BTRLconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = int32ToAuxInt(c &^ (1 << uint32(d)))
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
	// match: (ANDNL x (SHLXL (MOVLconst [1]) y))
	// result: (BTRL x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64SHLXL {
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
	// match: (ANDNQ x (SHLXQ (MOVQconst [1]) y))
	// result: (BTRQ x y)
	for {
		x := v_0
		if v_1.Op != OpAMD64SHLXQ {
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
	// match: (ANDQ (NOTQ (SHLXQ (MOVQconst [1]) y)) x)
	// result: (BTRQ x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64NOTQ {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64SHLXQ {
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
	// cond: isUint64PowerOfTwo(^c) && uint64(^c) >= 128
	// result: (BTRQconst [int8(log64(^c))] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64MOVQconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			x := v_1
			if !(isUint64PowerOfTwo(^c) && uint64(^c) >= 128) {
				continue
			}
			v.reset(OpAMD64BTRQconst)
			v.AuxInt = int8ToAuxInt(int8(log64(^c)))
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
	// match: (ANDQ x (ADDQconst [-1] x))
	// cond: buildcfg.GOAMD64 >= 3
	// result: (BLSRQ x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64ADDQconst || auxIntToInt32(v_1.AuxInt) != -1 || x != v_1.Args[0] || !(buildcfg.GOAMD64 >= 3) {
				continue
			}
			v.reset(OpAMD64BLSRQ)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64ANDQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ANDQconst [c] x)
	// cond: isUint64PowerOfTwo(int64(^c)) && uint64(^c) >= 128
	// result: (BTRQconst [int8(log32(^c))] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isUint64PowerOfTwo(int64(^c)) && uint64(^c) >= 128) {
			break
		}
		v.reset(OpAMD64BTRQconst)
		v.AuxInt = int8ToAuxInt(int8(log32(^c)))
		v.AddArg(x)
		return true
	}
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
	// match: (ANDQconst [c] (BTRQconst [d] x))
	// cond: is32Bit(int64(c) &^ (1<<uint32(d)))
	// result: (ANDQconst [c &^ (1<<uint32(d))] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64BTRQconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(int64(c) &^ (1 << uint32(d)))) {
			break
		}
		v.reset(OpAMD64ANDQconst)
		v.AuxInt = int32ToAuxInt(c &^ (1 << uint32(d)))
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
	// result: (MOVBELload [i] {s} p mem)
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
		v.reset(OpAMD64MOVBELload)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg2(p, mem)
		return true
	}
	// match: (BSWAPL (MOVBELload [i] {s} p m))
	// result: (MOVLload [i] {s} p m)
	for {
		if v_0.Op != OpAMD64MOVBELload {
			break
		}
		i := auxIntToInt32(v_0.AuxInt)
		s := auxToSym(v_0.Aux)
		m := v_0.Args[1]
		p := v_0.Args[0]
		v.reset(OpAMD64MOVLload)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg2(p, m)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64BSWAPQ(v *Value) bool {
	v_0 := v.Args[0]
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
	// result: (MOVBEQload [i] {s} p mem)
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
		v.reset(OpAMD64MOVBEQload)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg2(p, mem)
		return true
	}
	// match: (BSWAPQ (MOVBEQload [i] {s} p m))
	// result: (MOVQload [i] {s} p m)
	for {
		if v_0.Op != OpAMD64MOVBEQload {
			break
		}
		i := auxIntToInt32(v_0.AuxInt)
		s := auxToSym(v_0.Aux)
		m := v_0.Args[1]
		p := v_0.Args[0]
		v.reset(OpAMD64MOVQload)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg2(p, m)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64BTCLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (BTCLconst [c] (XORLconst [d] x))
	// result: (XORLconst [d ^ 1<<uint32(c)] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64XORLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64XORLconst)
		v.AuxInt = int32ToAuxInt(d ^ 1<<uint32(c))
		v.AddArg(x)
		return true
	}
	// match: (BTCLconst [c] (BTCLconst [d] x))
	// result: (XORLconst [1<<uint32(c) | 1<<uint32(d)] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64BTCLconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64XORLconst)
		v.AuxInt = int32ToAuxInt(1<<uint32(c) | 1<<uint32(d))
		v.AddArg(x)
		return true
	}
	// match: (BTCLconst [c] (MOVLconst [d]))
	// result: (MOVLconst [d^(1<<uint32(c))])
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(d ^ (1 << uint32(c)))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64BTCQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (BTCQconst [c] (XORQconst [d] x))
	// cond: is32Bit(int64(d) ^ 1<<uint32(c))
	// result: (XORQconst [d ^ 1<<uint32(c)] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64XORQconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(int64(d) ^ 1<<uint32(c))) {
			break
		}
		v.reset(OpAMD64XORQconst)
		v.AuxInt = int32ToAuxInt(d ^ 1<<uint32(c))
		v.AddArg(x)
		return true
	}
	// match: (BTCQconst [c] (BTCQconst [d] x))
	// cond: is32Bit(1<<uint32(c) ^ 1<<uint32(d))
	// result: (XORQconst [1<<uint32(c) ^ 1<<uint32(d)] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64BTCQconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(1<<uint32(c) ^ 1<<uint32(d))) {
			break
		}
		v.reset(OpAMD64XORQconst)
		v.AuxInt = int32ToAuxInt(1<<uint32(c) ^ 1<<uint32(d))
		v.AddArg(x)
		return true
	}
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
	// match: (BTLconst [0] s:(SHRXQ x y))
	// result: (BTQ y x)
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		s := v_0
		if s.Op != OpAMD64SHRXQ {
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
	// match: (BTQconst [0] s:(SHRXQ x y))
	// result: (BTQ y x)
	for {
		if auxIntToInt8(v.AuxInt) != 0 {
			break
		}
		s := v_0
		if s.Op != OpAMD64SHRXQ {
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
func rewriteValueAMD64_OpAMD64BTRLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (BTRLconst [c] (BTSLconst [c] x))
	// result: (BTRLconst [c] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64BTSLconst || auxIntToInt8(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64BTRLconst)
		v.AuxInt = int8ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (BTRLconst [c] (BTCLconst [c] x))
	// result: (BTRLconst [c] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64BTCLconst || auxIntToInt8(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64BTRLconst)
		v.AuxInt = int8ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (BTRLconst [c] (ANDLconst [d] x))
	// result: (ANDLconst [d &^ (1<<uint32(c))] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = int32ToAuxInt(d &^ (1 << uint32(c)))
		v.AddArg(x)
		return true
	}
	// match: (BTRLconst [c] (BTRLconst [d] x))
	// result: (ANDLconst [^(1<<uint32(c) | 1<<uint32(d))] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64BTRLconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = int32ToAuxInt(^(1<<uint32(c) | 1<<uint32(d)))
		v.AddArg(x)
		return true
	}
	// match: (BTRLconst [c] (MOVLconst [d]))
	// result: (MOVLconst [d&^(1<<uint32(c))])
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(d &^ (1 << uint32(c)))
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
	// match: (BTRQconst [c] (ANDQconst [d] x))
	// cond: is32Bit(int64(d) &^ (1<<uint32(c)))
	// result: (ANDQconst [d &^ (1<<uint32(c))] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64ANDQconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(int64(d) &^ (1 << uint32(c)))) {
			break
		}
		v.reset(OpAMD64ANDQconst)
		v.AuxInt = int32ToAuxInt(d &^ (1 << uint32(c)))
		v.AddArg(x)
		return true
	}
	// match: (BTRQconst [c] (BTRQconst [d] x))
	// cond: is32Bit(^(1<<uint32(c) | 1<<uint32(d)))
	// result: (ANDQconst [^(1<<uint32(c) | 1<<uint32(d))] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64BTRQconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(^(1<<uint32(c) | 1<<uint32(d)))) {
			break
		}
		v.reset(OpAMD64ANDQconst)
		v.AuxInt = int32ToAuxInt(^(1<<uint32(c) | 1<<uint32(d)))
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
func rewriteValueAMD64_OpAMD64BTSLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (BTSLconst [c] (BTRLconst [c] x))
	// result: (BTSLconst [c] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64BTRLconst || auxIntToInt8(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64BTSLconst)
		v.AuxInt = int8ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (BTSLconst [c] (BTCLconst [c] x))
	// result: (BTSLconst [c] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64BTCLconst || auxIntToInt8(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64BTSLconst)
		v.AuxInt = int8ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (BTSLconst [c] (ORLconst [d] x))
	// result: (ORLconst [d | 1<<uint32(c)] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64ORLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64ORLconst)
		v.AuxInt = int32ToAuxInt(d | 1<<uint32(c))
		v.AddArg(x)
		return true
	}
	// match: (BTSLconst [c] (BTSLconst [d] x))
	// result: (ORLconst [1<<uint32(c) | 1<<uint32(d)] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64BTSLconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64ORLconst)
		v.AuxInt = int32ToAuxInt(1<<uint32(c) | 1<<uint32(d))
		v.AddArg(x)
		return true
	}
	// match: (BTSLconst [c] (MOVLconst [d]))
	// result: (MOVLconst [d|(1<<uint32(c))])
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int32ToAuxInt(d | (1 << uint32(c)))
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
	// match: (BTSQconst [c] (ORQconst [d] x))
	// cond: is32Bit(int64(d) | 1<<uint32(c))
	// result: (ORQconst [d | 1<<uint32(c)] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64ORQconst {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(int64(d) | 1<<uint32(c))) {
			break
		}
		v.reset(OpAMD64ORQconst)
		v.AuxInt = int32ToAuxInt(d | 1<<uint32(c))
		v.AddArg(x)
		return true
	}
	// match: (BTSQconst [c] (BTSQconst [d] x))
	// cond: is32Bit(1<<uint32(c) | 1<<uint32(d))
	// result: (ORQconst [1<<uint32(c) | 1<<uint32(d)] x)
	for {
		c := auxIntToInt8(v.AuxInt)
		if v_0.Op != OpAMD64BTSQconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(1<<uint32(c) | 1<<uint32(d))) {
			break
		}
		v.reset(OpAMD64ORQconst)
		v.AuxInt = int32ToAuxInt(1<<uint32(c) | 1<<uint32(d))
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
	return false
}
func rewriteValueAMD64_OpAMD64CMOVLGE(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
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
	return false
}
func rewriteValueAMD64_OpAMD64CMOVLNE(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
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
	return false
}
func rewriteValueAMD64_OpAMD64CMOVQGE(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
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
	return false
}
func rewriteValueAMD64_OpAMD64CMOVQNE(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
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
	// match: (LEAL1 [c] {s} x (SHLLconst [1] y))
	// result: (LEAL2 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64SHLLconst || auxIntToInt8(v_1.AuxInt) != 1 {
				continue
			}
			y := v_1.Args[0]
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
	// match: (LEAL2 [c] {s} x (SHLLconst [1] y))
	// result: (LEAL4 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != OpAMD64SHLLconst || auxIntToInt8(v_1.AuxInt) != 1 {
			break
		}
		y := v_1.Args[0]
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
	// match: (LEAL4 [c] {s} x (SHLLconst [1] y))
	// result: (LEAL8 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != OpAMD64SHLLconst || auxIntToInt8(v_1.AuxInt) != 1 {
			break
		}
		y := v_1.Args[0]
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
	// match: (LEAQ1 [c] {s} x (SHLQconst [1] y))
	// result: (LEAQ2 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpAMD64SHLQconst || auxIntToInt8(v_1.AuxInt) != 1 {
				continue
			}
			y := v_1.Args[0]
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
	// match: (LEAQ2 [c] {s} x (SHLQconst [1] y))
	// result: (LEAQ4 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != OpAMD64SHLQconst || auxIntToInt8(v_1.AuxInt) != 1 {
			break
		}
		y := v_1.Args[0]
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
	// match: (LEAQ4 [c] {s} x (SHLQconst [1] y))
	// result: (LEAQ8 [c] {s} x y)
	for {
		c := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		x := v_0
		if v_1.Op != OpAMD64SHLQconst || auxIntToInt8(v_1.AuxInt) != 1 {
			break
		}
		y := v_1.Args[0]
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
func rewriteValueAMD64_OpAMD64MOVBELstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBELstore [i] {s} p (BSWAPL x) m)
	// result: (MOVLstore [i] {s} p x m)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		if v_1.Op != OpAMD64BSWAPL {
			break
		}
		x := v_1.Args[0]
		m := v_2
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p, x, m)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBEQstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBEQstore [i] {s} p (BSWAPQ x) m)
	// result: (MOVQstore [i] {s} p x m)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		if v_1.Op != OpAMD64BSWAPQ {
			break
		}
		x := v_1.Args[0]
		m := v_2
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p, x, m)
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
	// match: (MOVBQZX x)
	// cond: zeroUpper56Bits(x,3)
	// result: x
	for {
		x := v_0
		if !(zeroUpper56Bits(x, 3)) {
			break
		}
		v.copyOf(x)
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
	b := v.Block
	typ := &b.Func.Config.Types
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
	// match: (MOVBstore [i] {s} p w x0:(MOVBstore [i-1] {s} p (SHRWconst [8] w) mem))
	// cond: x0.Uses == 1 && clobber(x0)
	// result: (MOVWstore [i-1] {s} p (ROLWconst <w.Type> [8] w) mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		w := v_1
		x0 := v_2
		if x0.Op != OpAMD64MOVBstore || auxIntToInt32(x0.AuxInt) != i-1 || auxToSym(x0.Aux) != s {
			break
		}
		mem := x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpAMD64SHRWconst || auxIntToInt8(x0_1.AuxInt) != 8 || w != x0_1.Args[0] || !(x0.Uses == 1 && clobber(x0)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x0.Pos, OpAMD64ROLWconst, w.Type)
		v0.AuxInt = int8ToAuxInt(8)
		v0.AddArg(w)
		v.AddArg3(p, v0, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p1 w x0:(MOVBstore [i] {s} p0 (SHRWconst [8] w) mem))
	// cond: x0.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x0)
	// result: (MOVWstore [i] {s} p0 (ROLWconst <w.Type> [8] w) mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p1 := v_0
		w := v_1
		x0 := v_2
		if x0.Op != OpAMD64MOVBstore || auxIntToInt32(x0.AuxInt) != i || auxToSym(x0.Aux) != s {
			break
		}
		mem := x0.Args[2]
		p0 := x0.Args[0]
		x0_1 := x0.Args[1]
		if x0_1.Op != OpAMD64SHRWconst || auxIntToInt8(x0_1.AuxInt) != 8 || w != x0_1.Args[0] || !(x0.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x0)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x0.Pos, OpAMD64ROLWconst, w.Type)
		v0.AuxInt = int8ToAuxInt(8)
		v0.AddArg(w)
		v.AddArg3(p0, v0, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p w x2:(MOVBstore [i-1] {s} p (SHRLconst [8] w) x1:(MOVBstore [i-2] {s} p (SHRLconst [16] w) x0:(MOVBstore [i-3] {s} p (SHRLconst [24] w) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0, x1, x2)
	// result: (MOVLstore [i-3] {s} p (BSWAPL <w.Type> w) mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		w := v_1
		x2 := v_2
		if x2.Op != OpAMD64MOVBstore || auxIntToInt32(x2.AuxInt) != i-1 || auxToSym(x2.Aux) != s {
			break
		}
		_ = x2.Args[2]
		if p != x2.Args[0] {
			break
		}
		x2_1 := x2.Args[1]
		if x2_1.Op != OpAMD64SHRLconst || auxIntToInt8(x2_1.AuxInt) != 8 || w != x2_1.Args[0] {
			break
		}
		x1 := x2.Args[2]
		if x1.Op != OpAMD64MOVBstore || auxIntToInt32(x1.AuxInt) != i-2 || auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpAMD64SHRLconst || auxIntToInt8(x1_1.AuxInt) != 16 || w != x1_1.Args[0] {
			break
		}
		x0 := x1.Args[2]
		if x0.Op != OpAMD64MOVBstore || auxIntToInt32(x0.AuxInt) != i-3 || auxToSym(x0.Aux) != s {
			break
		}
		mem := x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpAMD64SHRLconst || auxIntToInt8(x0_1.AuxInt) != 24 || w != x0_1.Args[0] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0, x1, x2)) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = int32ToAuxInt(i - 3)
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x0.Pos, OpAMD64BSWAPL, w.Type)
		v0.AddArg(w)
		v.AddArg3(p, v0, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p3 w x2:(MOVBstore [i] {s} p2 (SHRLconst [8] w) x1:(MOVBstore [i] {s} p1 (SHRLconst [16] w) x0:(MOVBstore [i] {s} p0 (SHRLconst [24] w) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && sequentialAddresses(p0, p1, 1) && sequentialAddresses(p1, p2, 1) && sequentialAddresses(p2, p3, 1) && clobber(x0, x1, x2)
	// result: (MOVLstore [i] {s} p0 (BSWAPL <w.Type> w) mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p3 := v_0
		w := v_1
		x2 := v_2
		if x2.Op != OpAMD64MOVBstore || auxIntToInt32(x2.AuxInt) != i || auxToSym(x2.Aux) != s {
			break
		}
		_ = x2.Args[2]
		p2 := x2.Args[0]
		x2_1 := x2.Args[1]
		if x2_1.Op != OpAMD64SHRLconst || auxIntToInt8(x2_1.AuxInt) != 8 || w != x2_1.Args[0] {
			break
		}
		x1 := x2.Args[2]
		if x1.Op != OpAMD64MOVBstore || auxIntToInt32(x1.AuxInt) != i || auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[2]
		p1 := x1.Args[0]
		x1_1 := x1.Args[1]
		if x1_1.Op != OpAMD64SHRLconst || auxIntToInt8(x1_1.AuxInt) != 16 || w != x1_1.Args[0] {
			break
		}
		x0 := x1.Args[2]
		if x0.Op != OpAMD64MOVBstore || auxIntToInt32(x0.AuxInt) != i || auxToSym(x0.Aux) != s {
			break
		}
		mem := x0.Args[2]
		p0 := x0.Args[0]
		x0_1 := x0.Args[1]
		if x0_1.Op != OpAMD64SHRLconst || auxIntToInt8(x0_1.AuxInt) != 24 || w != x0_1.Args[0] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && sequentialAddresses(p0, p1, 1) && sequentialAddresses(p1, p2, 1) && sequentialAddresses(p2, p3, 1) && clobber(x0, x1, x2)) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x0.Pos, OpAMD64BSWAPL, w.Type)
		v0.AddArg(w)
		v.AddArg3(p0, v0, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p w x6:(MOVBstore [i-1] {s} p (SHRQconst [8] w) x5:(MOVBstore [i-2] {s} p (SHRQconst [16] w) x4:(MOVBstore [i-3] {s} p (SHRQconst [24] w) x3:(MOVBstore [i-4] {s} p (SHRQconst [32] w) x2:(MOVBstore [i-5] {s} p (SHRQconst [40] w) x1:(MOVBstore [i-6] {s} p (SHRQconst [48] w) x0:(MOVBstore [i-7] {s} p (SHRQconst [56] w) mem))))))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && clobber(x0, x1, x2, x3, x4, x5, x6)
	// result: (MOVQstore [i-7] {s} p (BSWAPQ <w.Type> w) mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		w := v_1
		x6 := v_2
		if x6.Op != OpAMD64MOVBstore || auxIntToInt32(x6.AuxInt) != i-1 || auxToSym(x6.Aux) != s {
			break
		}
		_ = x6.Args[2]
		if p != x6.Args[0] {
			break
		}
		x6_1 := x6.Args[1]
		if x6_1.Op != OpAMD64SHRQconst || auxIntToInt8(x6_1.AuxInt) != 8 || w != x6_1.Args[0] {
			break
		}
		x5 := x6.Args[2]
		if x5.Op != OpAMD64MOVBstore || auxIntToInt32(x5.AuxInt) != i-2 || auxToSym(x5.Aux) != s {
			break
		}
		_ = x5.Args[2]
		if p != x5.Args[0] {
			break
		}
		x5_1 := x5.Args[1]
		if x5_1.Op != OpAMD64SHRQconst || auxIntToInt8(x5_1.AuxInt) != 16 || w != x5_1.Args[0] {
			break
		}
		x4 := x5.Args[2]
		if x4.Op != OpAMD64MOVBstore || auxIntToInt32(x4.AuxInt) != i-3 || auxToSym(x4.Aux) != s {
			break
		}
		_ = x4.Args[2]
		if p != x4.Args[0] {
			break
		}
		x4_1 := x4.Args[1]
		if x4_1.Op != OpAMD64SHRQconst || auxIntToInt8(x4_1.AuxInt) != 24 || w != x4_1.Args[0] {
			break
		}
		x3 := x4.Args[2]
		if x3.Op != OpAMD64MOVBstore || auxIntToInt32(x3.AuxInt) != i-4 || auxToSym(x3.Aux) != s {
			break
		}
		_ = x3.Args[2]
		if p != x3.Args[0] {
			break
		}
		x3_1 := x3.Args[1]
		if x3_1.Op != OpAMD64SHRQconst || auxIntToInt8(x3_1.AuxInt) != 32 || w != x3_1.Args[0] {
			break
		}
		x2 := x3.Args[2]
		if x2.Op != OpAMD64MOVBstore || auxIntToInt32(x2.AuxInt) != i-5 || auxToSym(x2.Aux) != s {
			break
		}
		_ = x2.Args[2]
		if p != x2.Args[0] {
			break
		}
		x2_1 := x2.Args[1]
		if x2_1.Op != OpAMD64SHRQconst || auxIntToInt8(x2_1.AuxInt) != 40 || w != x2_1.Args[0] {
			break
		}
		x1 := x2.Args[2]
		if x1.Op != OpAMD64MOVBstore || auxIntToInt32(x1.AuxInt) != i-6 || auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpAMD64SHRQconst || auxIntToInt8(x1_1.AuxInt) != 48 || w != x1_1.Args[0] {
			break
		}
		x0 := x1.Args[2]
		if x0.Op != OpAMD64MOVBstore || auxIntToInt32(x0.AuxInt) != i-7 || auxToSym(x0.Aux) != s {
			break
		}
		mem := x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpAMD64SHRQconst || auxIntToInt8(x0_1.AuxInt) != 56 || w != x0_1.Args[0] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && clobber(x0, x1, x2, x3, x4, x5, x6)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(i - 7)
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x0.Pos, OpAMD64BSWAPQ, w.Type)
		v0.AddArg(w)
		v.AddArg3(p, v0, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p7 w x6:(MOVBstore [i] {s} p6 (SHRQconst [8] w) x5:(MOVBstore [i] {s} p5 (SHRQconst [16] w) x4:(MOVBstore [i] {s} p4 (SHRQconst [24] w) x3:(MOVBstore [i] {s} p3 (SHRQconst [32] w) x2:(MOVBstore [i] {s} p2 (SHRQconst [40] w) x1:(MOVBstore [i] {s} p1 (SHRQconst [48] w) x0:(MOVBstore [i] {s} p0 (SHRQconst [56] w) mem))))))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && sequentialAddresses(p0, p1, 1) && sequentialAddresses(p1, p2, 1) && sequentialAddresses(p2, p3, 1) && sequentialAddresses(p3, p4, 1) && sequentialAddresses(p4, p5, 1) && sequentialAddresses(p5, p6, 1) && sequentialAddresses(p6, p7, 1) && clobber(x0, x1, x2, x3, x4, x5, x6)
	// result: (MOVQstore [i] {s} p0 (BSWAPQ <w.Type> w) mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p7 := v_0
		w := v_1
		x6 := v_2
		if x6.Op != OpAMD64MOVBstore || auxIntToInt32(x6.AuxInt) != i || auxToSym(x6.Aux) != s {
			break
		}
		_ = x6.Args[2]
		p6 := x6.Args[0]
		x6_1 := x6.Args[1]
		if x6_1.Op != OpAMD64SHRQconst || auxIntToInt8(x6_1.AuxInt) != 8 || w != x6_1.Args[0] {
			break
		}
		x5 := x6.Args[2]
		if x5.Op != OpAMD64MOVBstore || auxIntToInt32(x5.AuxInt) != i || auxToSym(x5.Aux) != s {
			break
		}
		_ = x5.Args[2]
		p5 := x5.Args[0]
		x5_1 := x5.Args[1]
		if x5_1.Op != OpAMD64SHRQconst || auxIntToInt8(x5_1.AuxInt) != 16 || w != x5_1.Args[0] {
			break
		}
		x4 := x5.Args[2]
		if x4.Op != OpAMD64MOVBstore || auxIntToInt32(x4.AuxInt) != i || auxToSym(x4.Aux) != s {
			break
		}
		_ = x4.Args[2]
		p4 := x4.Args[0]
		x4_1 := x4.Args[1]
		if x4_1.Op != OpAMD64SHRQconst || auxIntToInt8(x4_1.AuxInt) != 24 || w != x4_1.Args[0] {
			break
		}
		x3 := x4.Args[2]
		if x3.Op != OpAMD64MOVBstore || auxIntToInt32(x3.AuxInt) != i || auxToSym(x3.Aux) != s {
			break
		}
		_ = x3.Args[2]
		p3 := x3.Args[0]
		x3_1 := x3.Args[1]
		if x3_1.Op != OpAMD64SHRQconst || auxIntToInt8(x3_1.AuxInt) != 32 || w != x3_1.Args[0] {
			break
		}
		x2 := x3.Args[2]
		if x2.Op != OpAMD64MOVBstore || auxIntToInt32(x2.AuxInt) != i || auxToSym(x2.Aux) != s {
			break
		}
		_ = x2.Args[2]
		p2 := x2.Args[0]
		x2_1 := x2.Args[1]
		if x2_1.Op != OpAMD64SHRQconst || auxIntToInt8(x2_1.AuxInt) != 40 || w != x2_1.Args[0] {
			break
		}
		x1 := x2.Args[2]
		if x1.Op != OpAMD64MOVBstore || auxIntToInt32(x1.AuxInt) != i || auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[2]
		p1 := x1.Args[0]
		x1_1 := x1.Args[1]
		if x1_1.Op != OpAMD64SHRQconst || auxIntToInt8(x1_1.AuxInt) != 48 || w != x1_1.Args[0] {
			break
		}
		x0 := x1.Args[2]
		if x0.Op != OpAMD64MOVBstore || auxIntToInt32(x0.AuxInt) != i || auxToSym(x0.Aux) != s {
			break
		}
		mem := x0.Args[2]
		p0 := x0.Args[0]
		x0_1 := x0.Args[1]
		if x0_1.Op != OpAMD64SHRQconst || auxIntToInt8(x0_1.AuxInt) != 56 || w != x0_1.Args[0] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && sequentialAddresses(p0, p1, 1) && sequentialAddresses(p1, p2, 1) && sequentialAddresses(p2, p3, 1) && sequentialAddresses(p3, p4, 1) && sequentialAddresses(p4, p5, 1) && sequentialAddresses(p5, p6, 1) && sequentialAddresses(p6, p7, 1) && clobber(x0, x1, x2, x3, x4, x5, x6)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x0.Pos, OpAMD64BSWAPQ, w.Type)
		v0.AddArg(w)
		v.AddArg3(p0, v0, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p (SHRWconst [8] w) x:(MOVBstore [i-1] {s} p w mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVWstore [i-1] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		if v_1.Op != OpAMD64SHRWconst || auxIntToInt8(v_1.AuxInt) != 8 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] || w != x.Args[1] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
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
		if v_1.Op != OpAMD64SHRLconst || auxIntToInt8(v_1.AuxInt) != 8 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] || w != x.Args[1] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v.AddArg3(p, w, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p (SHRQconst [8] w) x:(MOVBstore [i-1] {s} p w mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVWstore [i-1] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		if v_1.Op != OpAMD64SHRQconst || auxIntToInt8(v_1.AuxInt) != 8 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] || w != x.Args[1] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v.AddArg3(p, w, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p w x:(MOVBstore [i+1] {s} p (SHRWconst [8] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVWstore [i] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		w := v_1
		x := v_2
		if x.Op != OpAMD64MOVBstore || auxIntToInt32(x.AuxInt) != i+1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] {
			break
		}
		x_1 := x.Args[1]
		if x_1.Op != OpAMD64SHRWconst || auxIntToInt8(x_1.AuxInt) != 8 || w != x_1.Args[0] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p, w, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p w x:(MOVBstore [i+1] {s} p (SHRLconst [8] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVWstore [i] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		w := v_1
		x := v_2
		if x.Op != OpAMD64MOVBstore || auxIntToInt32(x.AuxInt) != i+1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] {
			break
		}
		x_1 := x.Args[1]
		if x_1.Op != OpAMD64SHRLconst || auxIntToInt8(x_1.AuxInt) != 8 || w != x_1.Args[0] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p, w, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p w x:(MOVBstore [i+1] {s} p (SHRQconst [8] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVWstore [i] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		w := v_1
		x := v_2
		if x.Op != OpAMD64MOVBstore || auxIntToInt32(x.AuxInt) != i+1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] {
			break
		}
		x_1 := x.Args[1]
		if x_1.Op != OpAMD64SHRQconst || auxIntToInt8(x_1.AuxInt) != 8 || w != x_1.Args[0] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
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
		if v_1.Op != OpAMD64SHRLconst {
			break
		}
		j := auxIntToInt8(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] {
			break
		}
		w0 := x.Args[1]
		if w0.Op != OpAMD64SHRLconst || auxIntToInt8(w0.AuxInt) != j-8 || w != w0.Args[0] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v.AddArg3(p, w0, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p (SHRQconst [j] w) x:(MOVBstore [i-1] {s} p w0:(SHRQconst [j-8] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVWstore [i-1] {s} p w0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		if v_1.Op != OpAMD64SHRQconst {
			break
		}
		j := auxIntToInt8(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] {
			break
		}
		w0 := x.Args[1]
		if w0.Op != OpAMD64SHRQconst || auxIntToInt8(w0.AuxInt) != j-8 || w != w0.Args[0] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
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
		if v_1.Op != OpAMD64SHRWconst || auxIntToInt8(v_1.AuxInt) != 8 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVBstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p0 := x.Args[0]
		if w != x.Args[1] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
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
		if v_1.Op != OpAMD64SHRLconst || auxIntToInt8(v_1.AuxInt) != 8 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVBstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p0 := x.Args[0]
		if w != x.Args[1] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p1 (SHRQconst [8] w) x:(MOVBstore [i] {s} p0 w mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)
	// result: (MOVWstore [i] {s} p0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p1 := v_0
		if v_1.Op != OpAMD64SHRQconst || auxIntToInt8(v_1.AuxInt) != 8 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVBstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p0 := x.Args[0]
		if w != x.Args[1] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p0 w x:(MOVBstore [i] {s} p1 (SHRWconst [8] w) mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)
	// result: (MOVWstore [i] {s} p0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p0 := v_0
		w := v_1
		x := v_2
		if x.Op != OpAMD64MOVBstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p1 := x.Args[0]
		x_1 := x.Args[1]
		if x_1.Op != OpAMD64SHRWconst || auxIntToInt8(x_1.AuxInt) != 8 || w != x_1.Args[0] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p0 w x:(MOVBstore [i] {s} p1 (SHRLconst [8] w) mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)
	// result: (MOVWstore [i] {s} p0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p0 := v_0
		w := v_1
		x := v_2
		if x.Op != OpAMD64MOVBstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p1 := x.Args[0]
		x_1 := x.Args[1]
		if x_1.Op != OpAMD64SHRLconst || auxIntToInt8(x_1.AuxInt) != 8 || w != x_1.Args[0] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p0 w x:(MOVBstore [i] {s} p1 (SHRQconst [8] w) mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)
	// result: (MOVWstore [i] {s} p0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p0 := v_0
		w := v_1
		x := v_2
		if x.Op != OpAMD64MOVBstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p1 := x.Args[0]
		x_1 := x.Args[1]
		if x_1.Op != OpAMD64SHRQconst || auxIntToInt8(x_1.AuxInt) != 8 || w != x_1.Args[0] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
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
		if v_1.Op != OpAMD64SHRLconst {
			break
		}
		j := auxIntToInt8(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVBstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p0 := x.Args[0]
		w0 := x.Args[1]
		if w0.Op != OpAMD64SHRLconst || auxIntToInt8(w0.AuxInt) != j-8 || w != w0.Args[0] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w0, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p1 (SHRQconst [j] w) x:(MOVBstore [i] {s} p0 w0:(SHRQconst [j-8] w) mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)
	// result: (MOVWstore [i] {s} p0 w0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p1 := v_0
		if v_1.Op != OpAMD64SHRQconst {
			break
		}
		j := auxIntToInt8(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVBstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p0 := x.Args[0]
		w0 := x.Args[1]
		if w0.Op != OpAMD64SHRQconst || auxIntToInt8(w0.AuxInt) != j-8 || w != w0.Args[0] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 1) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w0, mem)
		return true
	}
	// match: (MOVBstore [c3] {s} p3 (SHRQconst [56] w) x1:(MOVWstore [c2] {s} p2 (SHRQconst [40] w) x2:(MOVLstore [c1] {s} p1 (SHRQconst [8] w) x3:(MOVBstore [c0] {s} p0 w mem))))
	// cond: x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && sequentialAddresses(p0, p1, int64(1 + c0 - c1)) && sequentialAddresses(p0, p2, int64(5 + c0 - c2)) && sequentialAddresses(p0, p3, int64(7 + c0 - c3)) && clobber(x1, x2, x3)
	// result: (MOVQstore [c0] {s} p0 w mem)
	for {
		c3 := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p3 := v_0
		if v_1.Op != OpAMD64SHRQconst || auxIntToInt8(v_1.AuxInt) != 56 {
			break
		}
		w := v_1.Args[0]
		x1 := v_2
		if x1.Op != OpAMD64MOVWstore {
			break
		}
		c2 := auxIntToInt32(x1.AuxInt)
		if auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[2]
		p2 := x1.Args[0]
		x1_1 := x1.Args[1]
		if x1_1.Op != OpAMD64SHRQconst || auxIntToInt8(x1_1.AuxInt) != 40 || w != x1_1.Args[0] {
			break
		}
		x2 := x1.Args[2]
		if x2.Op != OpAMD64MOVLstore {
			break
		}
		c1 := auxIntToInt32(x2.AuxInt)
		if auxToSym(x2.Aux) != s {
			break
		}
		_ = x2.Args[2]
		p1 := x2.Args[0]
		x2_1 := x2.Args[1]
		if x2_1.Op != OpAMD64SHRQconst || auxIntToInt8(x2_1.AuxInt) != 8 || w != x2_1.Args[0] {
			break
		}
		x3 := x2.Args[2]
		if x3.Op != OpAMD64MOVBstore {
			break
		}
		c0 := auxIntToInt32(x3.AuxInt)
		if auxToSym(x3.Aux) != s {
			break
		}
		mem := x3.Args[2]
		p0 := x3.Args[0]
		if w != x3.Args[1] || !(x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && sequentialAddresses(p0, p1, int64(1+c0-c1)) && sequentialAddresses(p0, p2, int64(5+c0-c2)) && sequentialAddresses(p0, p3, int64(7+c0-c3)) && clobber(x1, x2, x3)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(c0)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w, mem)
		return true
	}
	// match: (MOVBstore [i] {s} p x1:(MOVBload [j] {s2} p2 mem) mem2:(MOVBstore [i-1] {s} p x2:(MOVBload [j-1] {s2} p2 mem) mem))
	// cond: x1.Uses == 1 && x2.Uses == 1 && mem2.Uses == 1 && clobber(x1, x2, mem2)
	// result: (MOVWstore [i-1] {s} p (MOVWload [j-1] {s2} p2 mem) mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		x1 := v_1
		if x1.Op != OpAMD64MOVBload {
			break
		}
		j := auxIntToInt32(x1.AuxInt)
		s2 := auxToSym(x1.Aux)
		mem := x1.Args[1]
		p2 := x1.Args[0]
		mem2 := v_2
		if mem2.Op != OpAMD64MOVBstore || auxIntToInt32(mem2.AuxInt) != i-1 || auxToSym(mem2.Aux) != s {
			break
		}
		_ = mem2.Args[2]
		if p != mem2.Args[0] {
			break
		}
		x2 := mem2.Args[1]
		if x2.Op != OpAMD64MOVBload || auxIntToInt32(x2.AuxInt) != j-1 || auxToSym(x2.Aux) != s2 {
			break
		}
		_ = x2.Args[1]
		if p2 != x2.Args[0] || mem != x2.Args[1] || mem != mem2.Args[2] || !(x1.Uses == 1 && x2.Uses == 1 && mem2.Uses == 1 && clobber(x1, x2, mem2)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x2.Pos, OpAMD64MOVWload, typ.UInt16)
		v0.AuxInt = int32ToAuxInt(j - 1)
		v0.Aux = symToAux(s2)
		v0.AddArg2(p2, mem)
		v.AddArg3(p, v0, mem)
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
	// match: (MOVBstoreconst [c] {s} p1 x:(MOVBstoreconst [a] {s} p0 mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+1-c.Off())) && clobber(x)
	// result: (MOVWstoreconst [makeValAndOff(a.Val()&0xff | c.Val()<<8, a.Off())] {s} p0 mem)
	for {
		c := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		p1 := v_0
		x := v_1
		if x.Op != OpAMD64MOVBstoreconst {
			break
		}
		a := auxIntToValAndOff(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		p0 := x.Args[0]
		if !(x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+1-c.Off())) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(a.Val()&0xff|c.Val()<<8, a.Off()))
		v.Aux = symToAux(s)
		v.AddArg2(p0, mem)
		return true
	}
	// match: (MOVBstoreconst [a] {s} p0 x:(MOVBstoreconst [c] {s} p1 mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+1-c.Off())) && clobber(x)
	// result: (MOVWstoreconst [makeValAndOff(a.Val()&0xff | c.Val()<<8, a.Off())] {s} p0 mem)
	for {
		a := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		p0 := v_0
		x := v_1
		if x.Op != OpAMD64MOVBstoreconst {
			break
		}
		c := auxIntToValAndOff(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		p1 := x.Args[0]
		if !(x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+1-c.Off())) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(a.Val()&0xff|c.Val()<<8, a.Off()))
		v.Aux = symToAux(s)
		v.AddArg2(p0, mem)
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
	// match: (MOVLQZX x)
	// cond: zeroUpper32Bits(x,3)
	// result: x
	for {
		x := v_0
		if !(zeroUpper32Bits(x, 3)) {
			break
		}
		v.copyOf(x)
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
	// result: (MOVQconst [int64(read32(sym, int64(off), config.ctxt.Arch.ByteOrder))])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpSB || !(symIsRO(sym)) {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = int64ToAuxInt(int64(read32(sym, int64(off), config.ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
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
	// match: (MOVLstore [i] {s} p (SHRQconst [32] w) x:(MOVLstore [i-4] {s} p w mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVQstore [i-4] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		if v_1.Op != OpAMD64SHRQconst || auxIntToInt8(v_1.AuxInt) != 32 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVLstore || auxIntToInt32(x.AuxInt) != i-4 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] || w != x.Args[1] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(i - 4)
		v.Aux = symToAux(s)
		v.AddArg3(p, w, mem)
		return true
	}
	// match: (MOVLstore [i] {s} p (SHRQconst [j] w) x:(MOVLstore [i-4] {s} p w0:(SHRQconst [j-32] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVQstore [i-4] {s} p w0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		if v_1.Op != OpAMD64SHRQconst {
			break
		}
		j := auxIntToInt8(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVLstore || auxIntToInt32(x.AuxInt) != i-4 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] {
			break
		}
		w0 := x.Args[1]
		if w0.Op != OpAMD64SHRQconst || auxIntToInt8(w0.AuxInt) != j-32 || w != w0.Args[0] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(i - 4)
		v.Aux = symToAux(s)
		v.AddArg3(p, w0, mem)
		return true
	}
	// match: (MOVLstore [i] {s} p1 (SHRQconst [32] w) x:(MOVLstore [i] {s} p0 w mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, 4) && clobber(x)
	// result: (MOVQstore [i] {s} p0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p1 := v_0
		if v_1.Op != OpAMD64SHRQconst || auxIntToInt8(v_1.AuxInt) != 32 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVLstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p0 := x.Args[0]
		if w != x.Args[1] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 4) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w, mem)
		return true
	}
	// match: (MOVLstore [i] {s} p1 (SHRQconst [j] w) x:(MOVLstore [i] {s} p0 w0:(SHRQconst [j-32] w) mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, 4) && clobber(x)
	// result: (MOVQstore [i] {s} p0 w0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p1 := v_0
		if v_1.Op != OpAMD64SHRQconst {
			break
		}
		j := auxIntToInt8(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVLstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p0 := x.Args[0]
		w0 := x.Args[1]
		if w0.Op != OpAMD64SHRQconst || auxIntToInt8(w0.AuxInt) != j-32 || w != w0.Args[0] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 4) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w0, mem)
		return true
	}
	// match: (MOVLstore [i] {s} p x1:(MOVLload [j] {s2} p2 mem) mem2:(MOVLstore [i-4] {s} p x2:(MOVLload [j-4] {s2} p2 mem) mem))
	// cond: x1.Uses == 1 && x2.Uses == 1 && mem2.Uses == 1 && clobber(x1, x2, mem2)
	// result: (MOVQstore [i-4] {s} p (MOVQload [j-4] {s2} p2 mem) mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		x1 := v_1
		if x1.Op != OpAMD64MOVLload {
			break
		}
		j := auxIntToInt32(x1.AuxInt)
		s2 := auxToSym(x1.Aux)
		mem := x1.Args[1]
		p2 := x1.Args[0]
		mem2 := v_2
		if mem2.Op != OpAMD64MOVLstore || auxIntToInt32(mem2.AuxInt) != i-4 || auxToSym(mem2.Aux) != s {
			break
		}
		_ = mem2.Args[2]
		if p != mem2.Args[0] {
			break
		}
		x2 := mem2.Args[1]
		if x2.Op != OpAMD64MOVLload || auxIntToInt32(x2.AuxInt) != j-4 || auxToSym(x2.Aux) != s2 {
			break
		}
		_ = x2.Args[1]
		if p2 != x2.Args[0] || mem != x2.Args[1] || mem != mem2.Args[2] || !(x1.Uses == 1 && x2.Uses == 1 && mem2.Uses == 1 && clobber(x1, x2, mem2)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(i - 4)
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x2.Pos, OpAMD64MOVQload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(j - 4)
		v0.Aux = symToAux(s2)
		v0.AddArg2(p2, mem)
		v.AddArg3(p, v0, mem)
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
	b := v.Block
	typ := &b.Func.Config.Types
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
	// match: (MOVLstoreconst [c] {s} p1 x:(MOVLstoreconst [a] {s} p0 mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+4-c.Off())) && clobber(x)
	// result: (MOVQstore [a.Off()] {s} p0 (MOVQconst [a.Val64()&0xffffffff | c.Val64()<<32]) mem)
	for {
		c := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		p1 := v_0
		x := v_1
		if x.Op != OpAMD64MOVLstoreconst {
			break
		}
		a := auxIntToValAndOff(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		p0 := x.Args[0]
		if !(x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+4-c.Off())) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(a.Off())
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x.Pos, OpAMD64MOVQconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(a.Val64()&0xffffffff | c.Val64()<<32)
		v.AddArg3(p0, v0, mem)
		return true
	}
	// match: (MOVLstoreconst [a] {s} p0 x:(MOVLstoreconst [c] {s} p1 mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+4-c.Off())) && clobber(x)
	// result: (MOVQstore [a.Off()] {s} p0 (MOVQconst [a.Val64()&0xffffffff | c.Val64()<<32]) mem)
	for {
		a := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		p0 := v_0
		x := v_1
		if x.Op != OpAMD64MOVLstoreconst {
			break
		}
		c := auxIntToValAndOff(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		p1 := x.Args[0]
		if !(x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+4-c.Off())) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(a.Off())
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x.Pos, OpAMD64MOVQconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(a.Val64()&0xffffffff | c.Val64()<<32)
		v.AddArg3(p0, v0, mem)
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
	b := v.Block
	config := b.Func.Config
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
	// cond: config.useSSE && x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+8-c.Off())) && a.Val() == 0 && c.Val() == 0 && clobber(x)
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
		if !(config.useSSE && x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+8-c.Off())) && a.Val() == 0 && c.Val() == 0 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVOstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, a.Off()))
		v.Aux = symToAux(s)
		v.AddArg2(p0, mem)
		return true
	}
	// match: (MOVQstoreconst [a] {s} p0 x:(MOVQstoreconst [c] {s} p1 mem))
	// cond: config.useSSE && x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+8-c.Off())) && a.Val() == 0 && c.Val() == 0 && clobber(x)
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
		if !(config.useSSE && x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+8-c.Off())) && a.Val() == 0 && c.Val() == 0 && clobber(x)) {
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
	// match: (MOVWQZX x)
	// cond: zeroUpper48Bits(x,3)
	// result: x
	for {
		x := v_0
		if !(zeroUpper48Bits(x, 3)) {
			break
		}
		v.copyOf(x)
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
	b := v.Block
	typ := &b.Func.Config.Types
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
	// match: (MOVWstore [i] {s} p (SHRLconst [16] w) x:(MOVWstore [i-2] {s} p w mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVLstore [i-2] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		if v_1.Op != OpAMD64SHRLconst || auxIntToInt8(v_1.AuxInt) != 16 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVWstore || auxIntToInt32(x.AuxInt) != i-2 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] || w != x.Args[1] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = int32ToAuxInt(i - 2)
		v.Aux = symToAux(s)
		v.AddArg3(p, w, mem)
		return true
	}
	// match: (MOVWstore [i] {s} p (SHRQconst [16] w) x:(MOVWstore [i-2] {s} p w mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVLstore [i-2] {s} p w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		if v_1.Op != OpAMD64SHRQconst || auxIntToInt8(v_1.AuxInt) != 16 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVWstore || auxIntToInt32(x.AuxInt) != i-2 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] || w != x.Args[1] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstore)
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
		if v_1.Op != OpAMD64SHRLconst {
			break
		}
		j := auxIntToInt8(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVWstore || auxIntToInt32(x.AuxInt) != i-2 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] {
			break
		}
		w0 := x.Args[1]
		if w0.Op != OpAMD64SHRLconst || auxIntToInt8(w0.AuxInt) != j-16 || w != w0.Args[0] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = int32ToAuxInt(i - 2)
		v.Aux = symToAux(s)
		v.AddArg3(p, w0, mem)
		return true
	}
	// match: (MOVWstore [i] {s} p (SHRQconst [j] w) x:(MOVWstore [i-2] {s} p w0:(SHRQconst [j-16] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVLstore [i-2] {s} p w0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		if v_1.Op != OpAMD64SHRQconst {
			break
		}
		j := auxIntToInt8(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVWstore || auxIntToInt32(x.AuxInt) != i-2 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if p != x.Args[0] {
			break
		}
		w0 := x.Args[1]
		if w0.Op != OpAMD64SHRQconst || auxIntToInt8(w0.AuxInt) != j-16 || w != w0.Args[0] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstore)
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
		if v_1.Op != OpAMD64SHRLconst || auxIntToInt8(v_1.AuxInt) != 16 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVWstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p0 := x.Args[0]
		if w != x.Args[1] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 2) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w, mem)
		return true
	}
	// match: (MOVWstore [i] {s} p1 (SHRQconst [16] w) x:(MOVWstore [i] {s} p0 w mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, 2) && clobber(x)
	// result: (MOVLstore [i] {s} p0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p1 := v_0
		if v_1.Op != OpAMD64SHRQconst || auxIntToInt8(v_1.AuxInt) != 16 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVWstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p0 := x.Args[0]
		if w != x.Args[1] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 2) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstore)
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
		if v_1.Op != OpAMD64SHRLconst {
			break
		}
		j := auxIntToInt8(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVWstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p0 := x.Args[0]
		w0 := x.Args[1]
		if w0.Op != OpAMD64SHRLconst || auxIntToInt8(w0.AuxInt) != j-16 || w != w0.Args[0] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 2) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w0, mem)
		return true
	}
	// match: (MOVWstore [i] {s} p1 (SHRQconst [j] w) x:(MOVWstore [i] {s} p0 w0:(SHRQconst [j-16] w) mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, 2) && clobber(x)
	// result: (MOVLstore [i] {s} p0 w0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p1 := v_0
		if v_1.Op != OpAMD64SHRQconst {
			break
		}
		j := auxIntToInt8(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpAMD64MOVWstore || auxIntToInt32(x.AuxInt) != i || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		p0 := x.Args[0]
		w0 := x.Args[1]
		if w0.Op != OpAMD64SHRQconst || auxIntToInt8(w0.AuxInt) != j-16 || w != w0.Args[0] || !(x.Uses == 1 && sequentialAddresses(p0, p1, 2) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = int32ToAuxInt(i)
		v.Aux = symToAux(s)
		v.AddArg3(p0, w0, mem)
		return true
	}
	// match: (MOVWstore [i] {s} p x1:(MOVWload [j] {s2} p2 mem) mem2:(MOVWstore [i-2] {s} p x2:(MOVWload [j-2] {s2} p2 mem) mem))
	// cond: x1.Uses == 1 && x2.Uses == 1 && mem2.Uses == 1 && clobber(x1, x2, mem2)
	// result: (MOVLstore [i-2] {s} p (MOVLload [j-2] {s2} p2 mem) mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		p := v_0
		x1 := v_1
		if x1.Op != OpAMD64MOVWload {
			break
		}
		j := auxIntToInt32(x1.AuxInt)
		s2 := auxToSym(x1.Aux)
		mem := x1.Args[1]
		p2 := x1.Args[0]
		mem2 := v_2
		if mem2.Op != OpAMD64MOVWstore || auxIntToInt32(mem2.AuxInt) != i-2 || auxToSym(mem2.Aux) != s {
			break
		}
		_ = mem2.Args[2]
		if p != mem2.Args[0] {
			break
		}
		x2 := mem2.Args[1]
		if x2.Op != OpAMD64MOVWload || auxIntToInt32(x2.AuxInt) != j-2 || auxToSym(x2.Aux) != s2 {
			break
		}
		_ = x2.Args[1]
		if p2 != x2.Args[0] || mem != x2.Args[1] || mem != mem2.Args[2] || !(x1.Uses == 1 && x2.Uses == 1 && mem2.Uses == 1 && clobber(x1, x2, mem2)) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = int32ToAuxInt(i - 2)
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x2.Pos, OpAMD64MOVLload, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(j - 2)
		v0.Aux = symToAux(s2)
		v0.AddArg2(p2, mem)
		v.AddArg3(p, v0, mem)
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
	// match: (MOVWstoreconst [c] {s} p1 x:(MOVWstoreconst [a] {s} p0 mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+2-c.Off())) && clobber(x)
	// result: (MOVLstoreconst [makeValAndOff(a.Val()&0xffff | c.Val()<<16, a.Off())] {s} p0 mem)
	for {
		c := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		p1 := v_0
		x := v_1
		if x.Op != OpAMD64MOVWstoreconst {
			break
		}
		a := auxIntToValAndOff(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		p0 := x.Args[0]
		if !(x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+2-c.Off())) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(a.Val()&0xffff|c.Val()<<16, a.Off()))
		v.Aux = symToAux(s)
		v.AddArg2(p0, mem)
		return true
	}
	// match: (MOVWstoreconst [a] {s} p0 x:(MOVWstoreconst [c] {s} p1 mem))
	// cond: x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+2-c.Off())) && clobber(x)
	// result: (MOVLstoreconst [makeValAndOff(a.Val()&0xffff | c.Val()<<16, a.Off())] {s} p0 mem)
	for {
		a := auxIntToValAndOff(v.AuxInt)
		s := auxToSym(v.Aux)
		p0 := v_0
		x := v_1
		if x.Op != OpAMD64MOVWstoreconst {
			break
		}
		c := auxIntToValAndOff(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		p1 := x.Args[0]
		if !(x.Uses == 1 && sequentialAddresses(p0, p1, int64(a.Off()+2-c.Off())) && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(a.Val()&0xffff|c.Val()<<16, a.Off()))
		v.Aux = symToAux(s)
		v.AddArg2(p0, mem)
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
	// match: (MULLconst [-9] x)
	// result: (NEGL (LEAL8 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != -9 {
			break
		}
		x := v_0
		v.reset(OpAMD64NEGL)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL8, v.Type)
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
		v.reset(OpAMD64NEGL)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL4, v.Type)
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
		v.reset(OpAMD64NEGL)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL2, v.Type)
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
		v.reset(OpAMD64NEGL)
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
	// match: (MULLconst [ 3] x)
	// result: (LEAL2 x x)
	for {
		if auxIntToInt32(v.AuxInt) != 3 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAL2)
		v.AddArg2(x, x)
		return true
	}
	// match: (MULLconst [ 5] x)
	// result: (LEAL4 x x)
	for {
		if auxIntToInt32(v.AuxInt) != 5 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAL4)
		v.AddArg2(x, x)
		return true
	}
	// match: (MULLconst [ 7] x)
	// result: (LEAL2 x (LEAL2 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 7 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAL2)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULLconst [ 9] x)
	// result: (LEAL8 x x)
	for {
		if auxIntToInt32(v.AuxInt) != 9 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAL8)
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
		v.reset(OpAMD64LEAL2)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL4, v.Type)
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
		v.reset(OpAMD64LEAL4)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL2, v.Type)
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
		v.reset(OpAMD64LEAL2)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL8, v.Type)
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
		v.reset(OpAMD64LEAL4)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL4, v.Type)
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
		v.reset(OpAMD64LEAL8)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL2, v.Type)
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
		v.reset(OpAMD64LEAL8)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL2, v.Type)
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
		v.reset(OpAMD64LEAL4)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL8, v.Type)
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
		v.reset(OpAMD64LEAL8)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL4, v.Type)
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
		v.reset(OpAMD64LEAL8)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL4, v.Type)
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
		v.reset(OpAMD64LEAL8)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL8, v.Type)
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
		v.reset(OpAMD64LEAL8)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(v0, v0)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: isPowerOfTwo64(int64(c)+1) && c >= 15
	// result: (SUBL (SHLLconst <v.Type> [int8(log64(int64(c)+1))] x) x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isPowerOfTwo64(int64(c)+1) && c >= 15) {
			break
		}
		v.reset(OpAMD64SUBL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v0.AuxInt = int8ToAuxInt(int8(log64(int64(c) + 1)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: isPowerOfTwo32(c-1) && c >= 17
	// result: (LEAL1 (SHLLconst <v.Type> [int8(log32(c-1))] x) x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isPowerOfTwo32(c-1) && c >= 17) {
			break
		}
		v.reset(OpAMD64LEAL1)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v0.AuxInt = int8ToAuxInt(int8(log32(c - 1)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: isPowerOfTwo32(c-2) && c >= 34
	// result: (LEAL2 (SHLLconst <v.Type> [int8(log32(c-2))] x) x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isPowerOfTwo32(c-2) && c >= 34) {
			break
		}
		v.reset(OpAMD64LEAL2)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v0.AuxInt = int8ToAuxInt(int8(log32(c - 2)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: isPowerOfTwo32(c-4) && c >= 68
	// result: (LEAL4 (SHLLconst <v.Type> [int8(log32(c-4))] x) x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isPowerOfTwo32(c-4) && c >= 68) {
			break
		}
		v.reset(OpAMD64LEAL4)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v0.AuxInt = int8ToAuxInt(int8(log32(c - 4)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: isPowerOfTwo32(c-8) && c >= 136
	// result: (LEAL8 (SHLLconst <v.Type> [int8(log32(c-8))] x) x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isPowerOfTwo32(c-8) && c >= 136) {
			break
		}
		v.reset(OpAMD64LEAL8)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v0.AuxInt = int8ToAuxInt(int8(log32(c - 8)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: c%3 == 0 && isPowerOfTwo32(c/3)
	// result: (SHLLconst [int8(log32(c/3))] (LEAL2 <v.Type> x x))
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(c%3 == 0 && isPowerOfTwo32(c/3)) {
			break
		}
		v.reset(OpAMD64SHLLconst)
		v.AuxInt = int8ToAuxInt(int8(log32(c / 3)))
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: c%5 == 0 && isPowerOfTwo32(c/5)
	// result: (SHLLconst [int8(log32(c/5))] (LEAL4 <v.Type> x x))
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(c%5 == 0 && isPowerOfTwo32(c/5)) {
			break
		}
		v.reset(OpAMD64SHLLconst)
		v.AuxInt = int8ToAuxInt(int8(log32(c / 5)))
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULLconst [c] x)
	// cond: c%9 == 0 && isPowerOfTwo32(c/9)
	// result: (SHLLconst [int8(log32(c/9))] (LEAL8 <v.Type> x x))
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(c%9 == 0 && isPowerOfTwo32(c/9)) {
			break
		}
		v.reset(OpAMD64SHLLconst)
		v.AuxInt = int8ToAuxInt(int8(log32(c / 9)))
		v0 := b.NewValue0(v.Pos, OpAMD64LEAL8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
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
	// match: (MULQconst [-9] x)
	// result: (NEGQ (LEAQ8 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != -9 {
			break
		}
		x := v_0
		v.reset(OpAMD64NEGQ)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULQconst [-5] x)
	// result: (NEGQ (LEAQ4 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != -5 {
			break
		}
		x := v_0
		v.reset(OpAMD64NEGQ)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULQconst [-3] x)
	// result: (NEGQ (LEAQ2 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != -3 {
			break
		}
		x := v_0
		v.reset(OpAMD64NEGQ)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULQconst [-1] x)
	// result: (NEGQ x)
	for {
		if auxIntToInt32(v.AuxInt) != -1 {
			break
		}
		x := v_0
		v.reset(OpAMD64NEGQ)
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
	// match: (MULQconst [ 3] x)
	// result: (LEAQ2 x x)
	for {
		if auxIntToInt32(v.AuxInt) != 3 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAQ2)
		v.AddArg2(x, x)
		return true
	}
	// match: (MULQconst [ 5] x)
	// result: (LEAQ4 x x)
	for {
		if auxIntToInt32(v.AuxInt) != 5 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAQ4)
		v.AddArg2(x, x)
		return true
	}
	// match: (MULQconst [ 7] x)
	// result: (LEAQ2 x (LEAQ2 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 7 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAQ2)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULQconst [ 9] x)
	// result: (LEAQ8 x x)
	for {
		if auxIntToInt32(v.AuxInt) != 9 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAQ8)
		v.AddArg2(x, x)
		return true
	}
	// match: (MULQconst [11] x)
	// result: (LEAQ2 x (LEAQ4 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 11 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAQ2)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULQconst [13] x)
	// result: (LEAQ4 x (LEAQ2 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 13 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAQ4)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULQconst [19] x)
	// result: (LEAQ2 x (LEAQ8 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 19 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAQ2)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULQconst [21] x)
	// result: (LEAQ4 x (LEAQ4 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 21 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAQ4)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULQconst [25] x)
	// result: (LEAQ8 x (LEAQ2 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 25 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAQ8)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULQconst [27] x)
	// result: (LEAQ8 (LEAQ2 <v.Type> x x) (LEAQ2 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 27 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAQ8)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(v0, v0)
		return true
	}
	// match: (MULQconst [37] x)
	// result: (LEAQ4 x (LEAQ8 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 37 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAQ4)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULQconst [41] x)
	// result: (LEAQ8 x (LEAQ4 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 41 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAQ8)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULQconst [45] x)
	// result: (LEAQ8 (LEAQ4 <v.Type> x x) (LEAQ4 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 45 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAQ8)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(v0, v0)
		return true
	}
	// match: (MULQconst [73] x)
	// result: (LEAQ8 x (LEAQ8 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 73 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAQ8)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(x, v0)
		return true
	}
	// match: (MULQconst [81] x)
	// result: (LEAQ8 (LEAQ8 <v.Type> x x) (LEAQ8 <v.Type> x x))
	for {
		if auxIntToInt32(v.AuxInt) != 81 {
			break
		}
		x := v_0
		v.reset(OpAMD64LEAQ8)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg2(v0, v0)
		return true
	}
	// match: (MULQconst [c] x)
	// cond: isPowerOfTwo64(int64(c)+1) && c >= 15
	// result: (SUBQ (SHLQconst <v.Type> [int8(log64(int64(c)+1))] x) x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isPowerOfTwo64(int64(c)+1) && c >= 15) {
			break
		}
		v.reset(OpAMD64SUBQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v0.AuxInt = int8ToAuxInt(int8(log64(int64(c) + 1)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULQconst [c] x)
	// cond: isPowerOfTwo32(c-1) && c >= 17
	// result: (LEAQ1 (SHLQconst <v.Type> [int8(log32(c-1))] x) x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isPowerOfTwo32(c-1) && c >= 17) {
			break
		}
		v.reset(OpAMD64LEAQ1)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v0.AuxInt = int8ToAuxInt(int8(log32(c - 1)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULQconst [c] x)
	// cond: isPowerOfTwo32(c-2) && c >= 34
	// result: (LEAQ2 (SHLQconst <v.Type> [int8(log32(c-2))] x) x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isPowerOfTwo32(c-2) && c >= 34) {
			break
		}
		v.reset(OpAMD64LEAQ2)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v0.AuxInt = int8ToAuxInt(int8(log32(c - 2)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULQconst [c] x)
	// cond: isPowerOfTwo32(c-4) && c >= 68
	// result: (LEAQ4 (SHLQconst <v.Type> [int8(log32(c-4))] x) x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isPowerOfTwo32(c-4) && c >= 68) {
			break
		}
		v.reset(OpAMD64LEAQ4)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v0.AuxInt = int8ToAuxInt(int8(log32(c - 4)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULQconst [c] x)
	// cond: isPowerOfTwo32(c-8) && c >= 136
	// result: (LEAQ8 (SHLQconst <v.Type> [int8(log32(c-8))] x) x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isPowerOfTwo32(c-8) && c >= 136) {
			break
		}
		v.reset(OpAMD64LEAQ8)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v0.AuxInt = int8ToAuxInt(int8(log32(c - 8)))
		v0.AddArg(x)
		v.AddArg2(v0, x)
		return true
	}
	// match: (MULQconst [c] x)
	// cond: c%3 == 0 && isPowerOfTwo32(c/3)
	// result: (SHLQconst [int8(log32(c/3))] (LEAQ2 <v.Type> x x))
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(c%3 == 0 && isPowerOfTwo32(c/3)) {
			break
		}
		v.reset(OpAMD64SHLQconst)
		v.AuxInt = int8ToAuxInt(int8(log32(c / 3)))
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ2, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULQconst [c] x)
	// cond: c%5 == 0 && isPowerOfTwo32(c/5)
	// result: (SHLQconst [int8(log32(c/5))] (LEAQ4 <v.Type> x x))
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(c%5 == 0 && isPowerOfTwo32(c/5)) {
			break
		}
		v.reset(OpAMD64SHLQconst)
		v.AuxInt = int8ToAuxInt(int8(log32(c / 5)))
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ4, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
	// match: (MULQconst [c] x)
	// cond: c%9 == 0 && isPowerOfTwo32(c/9)
	// result: (SHLQconst [int8(log32(c/9))] (LEAQ8 <v.Type> x x))
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(c%9 == 0 && isPowerOfTwo32(c/9)) {
			break
		}
		v.reset(OpAMD64SHLQconst)
		v.AuxInt = int8ToAuxInt(int8(log32(c / 9)))
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ8, v.Type)
		v0.AddArg2(x, x)
		v.AddArg(v0)
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
	b := v.Block
	typ := &b.Func.Config.Types
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
	// match: (ORL (SHLXL (MOVLconst [1]) y) x)
	// result: (BTSL x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64SHLXL {
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
	// match: (ORL (MOVLconst [c]) x)
	// cond: isUint32PowerOfTwo(int64(c)) && uint64(c) >= 128
	// result: (BTSLconst [int8(log32(c))] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64MOVLconst {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			x := v_1
			if !(isUint32PowerOfTwo(int64(c)) && uint64(c) >= 128) {
				continue
			}
			v.reset(OpAMD64BTSLconst)
			v.AuxInt = int8ToAuxInt(int8(log32(c)))
			v.AddArg(x)
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
	// match: (ORL x0:(MOVBload [i0] {s} p mem) sh:(SHLLconst [8] x1:(MOVBload [i1] {s} p mem)))
	// cond: i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0, x1, sh)
	// result: @mergePoint(b,x0,x1) (MOVWload [i0] {s} p mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			if x0.Op != OpAMD64MOVBload {
				continue
			}
			i0 := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p := x0.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLLconst || auxIntToInt8(sh.AuxInt) != 8 {
				continue
			}
			x1 := sh.Args[0]
			if x1.Op != OpAMD64MOVBload {
				continue
			}
			i1 := auxIntToInt32(x1.AuxInt)
			if auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			if p != x1.Args[0] || mem != x1.Args[1] || !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0, x1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x1.Pos, OpAMD64MOVWload, typ.UInt16)
			v.copyOf(v0)
			v0.AuxInt = int32ToAuxInt(i0)
			v0.Aux = symToAux(s)
			v0.AddArg2(p, mem)
			return true
		}
		break
	}
	// match: (ORL x0:(MOVBload [i] {s} p0 mem) sh:(SHLLconst [8] x1:(MOVBload [i] {s} p1 mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 1) && mergePoint(b,x0,x1) != nil && clobber(x0, x1, sh)
	// result: @mergePoint(b,x0,x1) (MOVWload [i] {s} p0 mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			if x0.Op != OpAMD64MOVBload {
				continue
			}
			i := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p0 := x0.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLLconst || auxIntToInt8(sh.AuxInt) != 8 {
				continue
			}
			x1 := sh.Args[0]
			if x1.Op != OpAMD64MOVBload || auxIntToInt32(x1.AuxInt) != i || auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			p1 := x1.Args[0]
			if mem != x1.Args[1] || !(x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 1) && mergePoint(b, x0, x1) != nil && clobber(x0, x1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x1.Pos, OpAMD64MOVWload, typ.UInt16)
			v.copyOf(v0)
			v0.AuxInt = int32ToAuxInt(i)
			v0.Aux = symToAux(s)
			v0.AddArg2(p0, mem)
			return true
		}
		break
	}
	// match: (ORL x0:(MOVWload [i0] {s} p mem) sh:(SHLLconst [16] x1:(MOVWload [i1] {s} p mem)))
	// cond: i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0, x1, sh)
	// result: @mergePoint(b,x0,x1) (MOVLload [i0] {s} p mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			if x0.Op != OpAMD64MOVWload {
				continue
			}
			i0 := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p := x0.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLLconst || auxIntToInt8(sh.AuxInt) != 16 {
				continue
			}
			x1 := sh.Args[0]
			if x1.Op != OpAMD64MOVWload {
				continue
			}
			i1 := auxIntToInt32(x1.AuxInt)
			if auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			if p != x1.Args[0] || mem != x1.Args[1] || !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0, x1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x1.Pos, OpAMD64MOVLload, typ.UInt32)
			v.copyOf(v0)
			v0.AuxInt = int32ToAuxInt(i0)
			v0.Aux = symToAux(s)
			v0.AddArg2(p, mem)
			return true
		}
		break
	}
	// match: (ORL x0:(MOVWload [i] {s} p0 mem) sh:(SHLLconst [16] x1:(MOVWload [i] {s} p1 mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 2) && mergePoint(b,x0,x1) != nil && clobber(x0, x1, sh)
	// result: @mergePoint(b,x0,x1) (MOVLload [i] {s} p0 mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			if x0.Op != OpAMD64MOVWload {
				continue
			}
			i := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p0 := x0.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLLconst || auxIntToInt8(sh.AuxInt) != 16 {
				continue
			}
			x1 := sh.Args[0]
			if x1.Op != OpAMD64MOVWload || auxIntToInt32(x1.AuxInt) != i || auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			p1 := x1.Args[0]
			if mem != x1.Args[1] || !(x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 2) && mergePoint(b, x0, x1) != nil && clobber(x0, x1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x1.Pos, OpAMD64MOVLload, typ.UInt32)
			v.copyOf(v0)
			v0.AuxInt = int32ToAuxInt(i)
			v0.Aux = symToAux(s)
			v0.AddArg2(p0, mem)
			return true
		}
		break
	}
	// match: (ORL s1:(SHLLconst [j1] x1:(MOVBload [i1] {s} p mem)) or:(ORL s0:(SHLLconst [j0] x0:(MOVBload [i0] {s} p mem)) y))
	// cond: i1 == i0+1 && j1 == j0+8 && j0 % 16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b,x0,x1,y) != nil && clobber(x0, x1, s0, s1, or)
	// result: @mergePoint(b,x0,x1,y) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWload [i0] {s} p mem)) y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s1 := v_0
			if s1.Op != OpAMD64SHLLconst {
				continue
			}
			j1 := auxIntToInt8(s1.AuxInt)
			x1 := s1.Args[0]
			if x1.Op != OpAMD64MOVBload {
				continue
			}
			i1 := auxIntToInt32(x1.AuxInt)
			s := auxToSym(x1.Aux)
			mem := x1.Args[1]
			p := x1.Args[0]
			or := v_1
			if or.Op != OpAMD64ORL {
				continue
			}
			_ = or.Args[1]
			or_0 := or.Args[0]
			or_1 := or.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, or_0, or_1 = _i1+1, or_1, or_0 {
				s0 := or_0
				if s0.Op != OpAMD64SHLLconst {
					continue
				}
				j0 := auxIntToInt8(s0.AuxInt)
				x0 := s0.Args[0]
				if x0.Op != OpAMD64MOVBload {
					continue
				}
				i0 := auxIntToInt32(x0.AuxInt)
				if auxToSym(x0.Aux) != s {
					continue
				}
				_ = x0.Args[1]
				if p != x0.Args[0] || mem != x0.Args[1] {
					continue
				}
				y := or_1
				if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1, y) != nil && clobber(x0, x1, s0, s1, or)) {
					continue
				}
				b = mergePoint(b, x0, x1, y)
				v0 := b.NewValue0(x0.Pos, OpAMD64ORL, v.Type)
				v.copyOf(v0)
				v1 := b.NewValue0(x0.Pos, OpAMD64SHLLconst, v.Type)
				v1.AuxInt = int8ToAuxInt(j0)
				v2 := b.NewValue0(x0.Pos, OpAMD64MOVWload, typ.UInt16)
				v2.AuxInt = int32ToAuxInt(i0)
				v2.Aux = symToAux(s)
				v2.AddArg2(p, mem)
				v1.AddArg(v2)
				v0.AddArg2(v1, y)
				return true
			}
		}
		break
	}
	// match: (ORL s1:(SHLLconst [j1] x1:(MOVBload [i] {s} p1 mem)) or:(ORL s0:(SHLLconst [j0] x0:(MOVBload [i] {s} p0 mem)) y))
	// cond: j1 == j0+8 && j0 % 16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && sequentialAddresses(p0, p1, 1) && mergePoint(b,x0,x1,y) != nil && clobber(x0, x1, s0, s1, or)
	// result: @mergePoint(b,x0,x1,y) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWload [i] {s} p0 mem)) y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s1 := v_0
			if s1.Op != OpAMD64SHLLconst {
				continue
			}
			j1 := auxIntToInt8(s1.AuxInt)
			x1 := s1.Args[0]
			if x1.Op != OpAMD64MOVBload {
				continue
			}
			i := auxIntToInt32(x1.AuxInt)
			s := auxToSym(x1.Aux)
			mem := x1.Args[1]
			p1 := x1.Args[0]
			or := v_1
			if or.Op != OpAMD64ORL {
				continue
			}
			_ = or.Args[1]
			or_0 := or.Args[0]
			or_1 := or.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, or_0, or_1 = _i1+1, or_1, or_0 {
				s0 := or_0
				if s0.Op != OpAMD64SHLLconst {
					continue
				}
				j0 := auxIntToInt8(s0.AuxInt)
				x0 := s0.Args[0]
				if x0.Op != OpAMD64MOVBload || auxIntToInt32(x0.AuxInt) != i || auxToSym(x0.Aux) != s {
					continue
				}
				_ = x0.Args[1]
				p0 := x0.Args[0]
				if mem != x0.Args[1] {
					continue
				}
				y := or_1
				if !(j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && sequentialAddresses(p0, p1, 1) && mergePoint(b, x0, x1, y) != nil && clobber(x0, x1, s0, s1, or)) {
					continue
				}
				b = mergePoint(b, x0, x1, y)
				v0 := b.NewValue0(x0.Pos, OpAMD64ORL, v.Type)
				v.copyOf(v0)
				v1 := b.NewValue0(x0.Pos, OpAMD64SHLLconst, v.Type)
				v1.AuxInt = int8ToAuxInt(j0)
				v2 := b.NewValue0(x0.Pos, OpAMD64MOVWload, typ.UInt16)
				v2.AuxInt = int32ToAuxInt(i)
				v2.Aux = symToAux(s)
				v2.AddArg2(p0, mem)
				v1.AddArg(v2)
				v0.AddArg2(v1, y)
				return true
			}
		}
		break
	}
	// match: (ORL x1:(MOVBload [i1] {s} p mem) sh:(SHLLconst [8] x0:(MOVBload [i0] {s} p mem)))
	// cond: i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0, x1, sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWload [i0] {s} p mem))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x1 := v_0
			if x1.Op != OpAMD64MOVBload {
				continue
			}
			i1 := auxIntToInt32(x1.AuxInt)
			s := auxToSym(x1.Aux)
			mem := x1.Args[1]
			p := x1.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLLconst || auxIntToInt8(sh.AuxInt) != 8 {
				continue
			}
			x0 := sh.Args[0]
			if x0.Op != OpAMD64MOVBload {
				continue
			}
			i0 := auxIntToInt32(x0.AuxInt)
			if auxToSym(x0.Aux) != s {
				continue
			}
			_ = x0.Args[1]
			if p != x0.Args[0] || mem != x0.Args[1] || !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0, x1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x0.Pos, OpAMD64ROLWconst, v.Type)
			v.copyOf(v0)
			v0.AuxInt = int8ToAuxInt(8)
			v1 := b.NewValue0(x0.Pos, OpAMD64MOVWload, typ.UInt16)
			v1.AuxInt = int32ToAuxInt(i0)
			v1.Aux = symToAux(s)
			v1.AddArg2(p, mem)
			v0.AddArg(v1)
			return true
		}
		break
	}
	// match: (ORL x1:(MOVBload [i] {s} p1 mem) sh:(SHLLconst [8] x0:(MOVBload [i] {s} p0 mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 1) && mergePoint(b,x0,x1) != nil && clobber(x0, x1, sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWload [i] {s} p0 mem))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x1 := v_0
			if x1.Op != OpAMD64MOVBload {
				continue
			}
			i := auxIntToInt32(x1.AuxInt)
			s := auxToSym(x1.Aux)
			mem := x1.Args[1]
			p1 := x1.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLLconst || auxIntToInt8(sh.AuxInt) != 8 {
				continue
			}
			x0 := sh.Args[0]
			if x0.Op != OpAMD64MOVBload || auxIntToInt32(x0.AuxInt) != i || auxToSym(x0.Aux) != s {
				continue
			}
			_ = x0.Args[1]
			p0 := x0.Args[0]
			if mem != x0.Args[1] || !(x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 1) && mergePoint(b, x0, x1) != nil && clobber(x0, x1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x0.Pos, OpAMD64ROLWconst, v.Type)
			v.copyOf(v0)
			v0.AuxInt = int8ToAuxInt(8)
			v1 := b.NewValue0(x0.Pos, OpAMD64MOVWload, typ.UInt16)
			v1.AuxInt = int32ToAuxInt(i)
			v1.Aux = symToAux(s)
			v1.AddArg2(p0, mem)
			v0.AddArg(v1)
			return true
		}
		break
	}
	// match: (ORL r1:(ROLWconst [8] x1:(MOVWload [i1] {s} p mem)) sh:(SHLLconst [16] r0:(ROLWconst [8] x0:(MOVWload [i0] {s} p mem))))
	// cond: i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0, x1, r0, r1, sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLload [i0] {s} p mem))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			r1 := v_0
			if r1.Op != OpAMD64ROLWconst || auxIntToInt8(r1.AuxInt) != 8 {
				continue
			}
			x1 := r1.Args[0]
			if x1.Op != OpAMD64MOVWload {
				continue
			}
			i1 := auxIntToInt32(x1.AuxInt)
			s := auxToSym(x1.Aux)
			mem := x1.Args[1]
			p := x1.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLLconst || auxIntToInt8(sh.AuxInt) != 16 {
				continue
			}
			r0 := sh.Args[0]
			if r0.Op != OpAMD64ROLWconst || auxIntToInt8(r0.AuxInt) != 8 {
				continue
			}
			x0 := r0.Args[0]
			if x0.Op != OpAMD64MOVWload {
				continue
			}
			i0 := auxIntToInt32(x0.AuxInt)
			if auxToSym(x0.Aux) != s {
				continue
			}
			_ = x0.Args[1]
			if p != x0.Args[0] || mem != x0.Args[1] || !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0, x1, r0, r1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x0.Pos, OpAMD64BSWAPL, v.Type)
			v.copyOf(v0)
			v1 := b.NewValue0(x0.Pos, OpAMD64MOVLload, typ.UInt32)
			v1.AuxInt = int32ToAuxInt(i0)
			v1.Aux = symToAux(s)
			v1.AddArg2(p, mem)
			v0.AddArg(v1)
			return true
		}
		break
	}
	// match: (ORL r1:(ROLWconst [8] x1:(MOVWload [i] {s} p1 mem)) sh:(SHLLconst [16] r0:(ROLWconst [8] x0:(MOVWload [i] {s} p0 mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 2) && mergePoint(b,x0,x1) != nil && clobber(x0, x1, r0, r1, sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLload [i] {s} p0 mem))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			r1 := v_0
			if r1.Op != OpAMD64ROLWconst || auxIntToInt8(r1.AuxInt) != 8 {
				continue
			}
			x1 := r1.Args[0]
			if x1.Op != OpAMD64MOVWload {
				continue
			}
			i := auxIntToInt32(x1.AuxInt)
			s := auxToSym(x1.Aux)
			mem := x1.Args[1]
			p1 := x1.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLLconst || auxIntToInt8(sh.AuxInt) != 16 {
				continue
			}
			r0 := sh.Args[0]
			if r0.Op != OpAMD64ROLWconst || auxIntToInt8(r0.AuxInt) != 8 {
				continue
			}
			x0 := r0.Args[0]
			if x0.Op != OpAMD64MOVWload || auxIntToInt32(x0.AuxInt) != i || auxToSym(x0.Aux) != s {
				continue
			}
			_ = x0.Args[1]
			p0 := x0.Args[0]
			if mem != x0.Args[1] || !(x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 2) && mergePoint(b, x0, x1) != nil && clobber(x0, x1, r0, r1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x0.Pos, OpAMD64BSWAPL, v.Type)
			v.copyOf(v0)
			v1 := b.NewValue0(x0.Pos, OpAMD64MOVLload, typ.UInt32)
			v1.AuxInt = int32ToAuxInt(i)
			v1.Aux = symToAux(s)
			v1.AddArg2(p0, mem)
			v0.AddArg(v1)
			return true
		}
		break
	}
	// match: (ORL s0:(SHLLconst [j0] x0:(MOVBload [i0] {s} p mem)) or:(ORL s1:(SHLLconst [j1] x1:(MOVBload [i1] {s} p mem)) y))
	// cond: i1 == i0+1 && j1 == j0-8 && j1 % 16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b,x0,x1,y) != nil && clobber(x0, x1, s0, s1, or)
	// result: @mergePoint(b,x0,x1,y) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWload [i0] {s} p mem))) y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s0 := v_0
			if s0.Op != OpAMD64SHLLconst {
				continue
			}
			j0 := auxIntToInt8(s0.AuxInt)
			x0 := s0.Args[0]
			if x0.Op != OpAMD64MOVBload {
				continue
			}
			i0 := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p := x0.Args[0]
			or := v_1
			if or.Op != OpAMD64ORL {
				continue
			}
			_ = or.Args[1]
			or_0 := or.Args[0]
			or_1 := or.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, or_0, or_1 = _i1+1, or_1, or_0 {
				s1 := or_0
				if s1.Op != OpAMD64SHLLconst {
					continue
				}
				j1 := auxIntToInt8(s1.AuxInt)
				x1 := s1.Args[0]
				if x1.Op != OpAMD64MOVBload {
					continue
				}
				i1 := auxIntToInt32(x1.AuxInt)
				if auxToSym(x1.Aux) != s {
					continue
				}
				_ = x1.Args[1]
				if p != x1.Args[0] || mem != x1.Args[1] {
					continue
				}
				y := or_1
				if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1, y) != nil && clobber(x0, x1, s0, s1, or)) {
					continue
				}
				b = mergePoint(b, x0, x1, y)
				v0 := b.NewValue0(x1.Pos, OpAMD64ORL, v.Type)
				v.copyOf(v0)
				v1 := b.NewValue0(x1.Pos, OpAMD64SHLLconst, v.Type)
				v1.AuxInt = int8ToAuxInt(j1)
				v2 := b.NewValue0(x1.Pos, OpAMD64ROLWconst, typ.UInt16)
				v2.AuxInt = int8ToAuxInt(8)
				v3 := b.NewValue0(x1.Pos, OpAMD64MOVWload, typ.UInt16)
				v3.AuxInt = int32ToAuxInt(i0)
				v3.Aux = symToAux(s)
				v3.AddArg2(p, mem)
				v2.AddArg(v3)
				v1.AddArg(v2)
				v0.AddArg2(v1, y)
				return true
			}
		}
		break
	}
	// match: (ORL s0:(SHLLconst [j0] x0:(MOVBload [i] {s} p0 mem)) or:(ORL s1:(SHLLconst [j1] x1:(MOVBload [i] {s} p1 mem)) y))
	// cond: j1 == j0-8 && j1 % 16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && sequentialAddresses(p0, p1, 1) && mergePoint(b,x0,x1,y) != nil && clobber(x0, x1, s0, s1, or)
	// result: @mergePoint(b,x0,x1,y) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWload [i] {s} p0 mem))) y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s0 := v_0
			if s0.Op != OpAMD64SHLLconst {
				continue
			}
			j0 := auxIntToInt8(s0.AuxInt)
			x0 := s0.Args[0]
			if x0.Op != OpAMD64MOVBload {
				continue
			}
			i := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p0 := x0.Args[0]
			or := v_1
			if or.Op != OpAMD64ORL {
				continue
			}
			_ = or.Args[1]
			or_0 := or.Args[0]
			or_1 := or.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, or_0, or_1 = _i1+1, or_1, or_0 {
				s1 := or_0
				if s1.Op != OpAMD64SHLLconst {
					continue
				}
				j1 := auxIntToInt8(s1.AuxInt)
				x1 := s1.Args[0]
				if x1.Op != OpAMD64MOVBload || auxIntToInt32(x1.AuxInt) != i || auxToSym(x1.Aux) != s {
					continue
				}
				_ = x1.Args[1]
				p1 := x1.Args[0]
				if mem != x1.Args[1] {
					continue
				}
				y := or_1
				if !(j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && sequentialAddresses(p0, p1, 1) && mergePoint(b, x0, x1, y) != nil && clobber(x0, x1, s0, s1, or)) {
					continue
				}
				b = mergePoint(b, x0, x1, y)
				v0 := b.NewValue0(x1.Pos, OpAMD64ORL, v.Type)
				v.copyOf(v0)
				v1 := b.NewValue0(x1.Pos, OpAMD64SHLLconst, v.Type)
				v1.AuxInt = int8ToAuxInt(j1)
				v2 := b.NewValue0(x1.Pos, OpAMD64ROLWconst, typ.UInt16)
				v2.AuxInt = int8ToAuxInt(8)
				v3 := b.NewValue0(x1.Pos, OpAMD64MOVWload, typ.UInt16)
				v3.AuxInt = int32ToAuxInt(i)
				v3.Aux = symToAux(s)
				v3.AddArg2(p0, mem)
				v2.AddArg(v3)
				v1.AddArg(v2)
				v0.AddArg2(v1, y)
				return true
			}
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
	// match: (ORLconst [c] x)
	// cond: isUint32PowerOfTwo(int64(c)) && uint64(c) >= 128
	// result: (BTSLconst [int8(log32(c))] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isUint32PowerOfTwo(int64(c)) && uint64(c) >= 128) {
			break
		}
		v.reset(OpAMD64BTSLconst)
		v.AuxInt = int8ToAuxInt(int8(log32(c)))
		v.AddArg(x)
		return true
	}
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
	// match: (ORLconst [c] (BTSLconst [d] x))
	// result: (ORLconst [c | 1<<uint32(d)] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64BTSLconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64ORLconst)
		v.AuxInt = int32ToAuxInt(c | 1<<uint32(d))
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
	b := v.Block
	typ := &b.Func.Config.Types
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
	// match: (ORQ (SHLXQ (MOVQconst [1]) y) x)
	// result: (BTSQ x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64SHLXQ {
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
	// cond: isUint64PowerOfTwo(c) && uint64(c) >= 128
	// result: (BTSQconst [int8(log64(c))] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64MOVQconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			x := v_1
			if !(isUint64PowerOfTwo(c) && uint64(c) >= 128) {
				continue
			}
			v.reset(OpAMD64BTSQconst)
			v.AuxInt = int8ToAuxInt(int8(log64(c)))
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
	// match: (ORQ x0:(MOVBload [i0] {s} p mem) sh:(SHLQconst [8] x1:(MOVBload [i1] {s} p mem)))
	// cond: i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0, x1, sh)
	// result: @mergePoint(b,x0,x1) (MOVWload [i0] {s} p mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			if x0.Op != OpAMD64MOVBload {
				continue
			}
			i0 := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p := x0.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLQconst || auxIntToInt8(sh.AuxInt) != 8 {
				continue
			}
			x1 := sh.Args[0]
			if x1.Op != OpAMD64MOVBload {
				continue
			}
			i1 := auxIntToInt32(x1.AuxInt)
			if auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			if p != x1.Args[0] || mem != x1.Args[1] || !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0, x1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x1.Pos, OpAMD64MOVWload, typ.UInt16)
			v.copyOf(v0)
			v0.AuxInt = int32ToAuxInt(i0)
			v0.Aux = symToAux(s)
			v0.AddArg2(p, mem)
			return true
		}
		break
	}
	// match: (ORQ x0:(MOVBload [i] {s} p0 mem) sh:(SHLQconst [8] x1:(MOVBload [i] {s} p1 mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 1) && mergePoint(b,x0,x1) != nil && clobber(x0, x1, sh)
	// result: @mergePoint(b,x0,x1) (MOVWload [i] {s} p0 mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			if x0.Op != OpAMD64MOVBload {
				continue
			}
			i := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p0 := x0.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLQconst || auxIntToInt8(sh.AuxInt) != 8 {
				continue
			}
			x1 := sh.Args[0]
			if x1.Op != OpAMD64MOVBload || auxIntToInt32(x1.AuxInt) != i || auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			p1 := x1.Args[0]
			if mem != x1.Args[1] || !(x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 1) && mergePoint(b, x0, x1) != nil && clobber(x0, x1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x1.Pos, OpAMD64MOVWload, typ.UInt16)
			v.copyOf(v0)
			v0.AuxInt = int32ToAuxInt(i)
			v0.Aux = symToAux(s)
			v0.AddArg2(p0, mem)
			return true
		}
		break
	}
	// match: (ORQ x0:(MOVWload [i0] {s} p mem) sh:(SHLQconst [16] x1:(MOVWload [i1] {s} p mem)))
	// cond: i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0, x1, sh)
	// result: @mergePoint(b,x0,x1) (MOVLload [i0] {s} p mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			if x0.Op != OpAMD64MOVWload {
				continue
			}
			i0 := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p := x0.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLQconst || auxIntToInt8(sh.AuxInt) != 16 {
				continue
			}
			x1 := sh.Args[0]
			if x1.Op != OpAMD64MOVWload {
				continue
			}
			i1 := auxIntToInt32(x1.AuxInt)
			if auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			if p != x1.Args[0] || mem != x1.Args[1] || !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0, x1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x1.Pos, OpAMD64MOVLload, typ.UInt32)
			v.copyOf(v0)
			v0.AuxInt = int32ToAuxInt(i0)
			v0.Aux = symToAux(s)
			v0.AddArg2(p, mem)
			return true
		}
		break
	}
	// match: (ORQ x0:(MOVWload [i] {s} p0 mem) sh:(SHLQconst [16] x1:(MOVWload [i] {s} p1 mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 2) && mergePoint(b,x0,x1) != nil && clobber(x0, x1, sh)
	// result: @mergePoint(b,x0,x1) (MOVLload [i] {s} p0 mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			if x0.Op != OpAMD64MOVWload {
				continue
			}
			i := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p0 := x0.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLQconst || auxIntToInt8(sh.AuxInt) != 16 {
				continue
			}
			x1 := sh.Args[0]
			if x1.Op != OpAMD64MOVWload || auxIntToInt32(x1.AuxInt) != i || auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			p1 := x1.Args[0]
			if mem != x1.Args[1] || !(x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 2) && mergePoint(b, x0, x1) != nil && clobber(x0, x1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x1.Pos, OpAMD64MOVLload, typ.UInt32)
			v.copyOf(v0)
			v0.AuxInt = int32ToAuxInt(i)
			v0.Aux = symToAux(s)
			v0.AddArg2(p0, mem)
			return true
		}
		break
	}
	// match: (ORQ x0:(MOVLload [i0] {s} p mem) sh:(SHLQconst [32] x1:(MOVLload [i1] {s} p mem)))
	// cond: i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0, x1, sh)
	// result: @mergePoint(b,x0,x1) (MOVQload [i0] {s} p mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			if x0.Op != OpAMD64MOVLload {
				continue
			}
			i0 := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p := x0.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLQconst || auxIntToInt8(sh.AuxInt) != 32 {
				continue
			}
			x1 := sh.Args[0]
			if x1.Op != OpAMD64MOVLload {
				continue
			}
			i1 := auxIntToInt32(x1.AuxInt)
			if auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			if p != x1.Args[0] || mem != x1.Args[1] || !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0, x1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x1.Pos, OpAMD64MOVQload, typ.UInt64)
			v.copyOf(v0)
			v0.AuxInt = int32ToAuxInt(i0)
			v0.Aux = symToAux(s)
			v0.AddArg2(p, mem)
			return true
		}
		break
	}
	// match: (ORQ x0:(MOVLload [i] {s} p0 mem) sh:(SHLQconst [32] x1:(MOVLload [i] {s} p1 mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 4) && mergePoint(b,x0,x1) != nil && clobber(x0, x1, sh)
	// result: @mergePoint(b,x0,x1) (MOVQload [i] {s} p0 mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			if x0.Op != OpAMD64MOVLload {
				continue
			}
			i := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p0 := x0.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLQconst || auxIntToInt8(sh.AuxInt) != 32 {
				continue
			}
			x1 := sh.Args[0]
			if x1.Op != OpAMD64MOVLload || auxIntToInt32(x1.AuxInt) != i || auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			p1 := x1.Args[0]
			if mem != x1.Args[1] || !(x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 4) && mergePoint(b, x0, x1) != nil && clobber(x0, x1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x1.Pos, OpAMD64MOVQload, typ.UInt64)
			v.copyOf(v0)
			v0.AuxInt = int32ToAuxInt(i)
			v0.Aux = symToAux(s)
			v0.AddArg2(p0, mem)
			return true
		}
		break
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVBload [i1] {s} p mem)) or:(ORQ s0:(SHLQconst [j0] x0:(MOVBload [i0] {s} p mem)) y))
	// cond: i1 == i0+1 && j1 == j0+8 && j0 % 16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b,x0,x1,y) != nil && clobber(x0, x1, s0, s1, or)
	// result: @mergePoint(b,x0,x1,y) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWload [i0] {s} p mem)) y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s1 := v_0
			if s1.Op != OpAMD64SHLQconst {
				continue
			}
			j1 := auxIntToInt8(s1.AuxInt)
			x1 := s1.Args[0]
			if x1.Op != OpAMD64MOVBload {
				continue
			}
			i1 := auxIntToInt32(x1.AuxInt)
			s := auxToSym(x1.Aux)
			mem := x1.Args[1]
			p := x1.Args[0]
			or := v_1
			if or.Op != OpAMD64ORQ {
				continue
			}
			_ = or.Args[1]
			or_0 := or.Args[0]
			or_1 := or.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, or_0, or_1 = _i1+1, or_1, or_0 {
				s0 := or_0
				if s0.Op != OpAMD64SHLQconst {
					continue
				}
				j0 := auxIntToInt8(s0.AuxInt)
				x0 := s0.Args[0]
				if x0.Op != OpAMD64MOVBload {
					continue
				}
				i0 := auxIntToInt32(x0.AuxInt)
				if auxToSym(x0.Aux) != s {
					continue
				}
				_ = x0.Args[1]
				if p != x0.Args[0] || mem != x0.Args[1] {
					continue
				}
				y := or_1
				if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1, y) != nil && clobber(x0, x1, s0, s1, or)) {
					continue
				}
				b = mergePoint(b, x0, x1, y)
				v0 := b.NewValue0(x0.Pos, OpAMD64ORQ, v.Type)
				v.copyOf(v0)
				v1 := b.NewValue0(x0.Pos, OpAMD64SHLQconst, v.Type)
				v1.AuxInt = int8ToAuxInt(j0)
				v2 := b.NewValue0(x0.Pos, OpAMD64MOVWload, typ.UInt16)
				v2.AuxInt = int32ToAuxInt(i0)
				v2.Aux = symToAux(s)
				v2.AddArg2(p, mem)
				v1.AddArg(v2)
				v0.AddArg2(v1, y)
				return true
			}
		}
		break
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVBload [i] {s} p1 mem)) or:(ORQ s0:(SHLQconst [j0] x0:(MOVBload [i] {s} p0 mem)) y))
	// cond: j1 == j0+8 && j0 % 16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && sequentialAddresses(p0, p1, 1) && mergePoint(b,x0,x1,y) != nil && clobber(x0, x1, s0, s1, or)
	// result: @mergePoint(b,x0,x1,y) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWload [i] {s} p0 mem)) y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s1 := v_0
			if s1.Op != OpAMD64SHLQconst {
				continue
			}
			j1 := auxIntToInt8(s1.AuxInt)
			x1 := s1.Args[0]
			if x1.Op != OpAMD64MOVBload {
				continue
			}
			i := auxIntToInt32(x1.AuxInt)
			s := auxToSym(x1.Aux)
			mem := x1.Args[1]
			p1 := x1.Args[0]
			or := v_1
			if or.Op != OpAMD64ORQ {
				continue
			}
			_ = or.Args[1]
			or_0 := or.Args[0]
			or_1 := or.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, or_0, or_1 = _i1+1, or_1, or_0 {
				s0 := or_0
				if s0.Op != OpAMD64SHLQconst {
					continue
				}
				j0 := auxIntToInt8(s0.AuxInt)
				x0 := s0.Args[0]
				if x0.Op != OpAMD64MOVBload || auxIntToInt32(x0.AuxInt) != i || auxToSym(x0.Aux) != s {
					continue
				}
				_ = x0.Args[1]
				p0 := x0.Args[0]
				if mem != x0.Args[1] {
					continue
				}
				y := or_1
				if !(j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && sequentialAddresses(p0, p1, 1) && mergePoint(b, x0, x1, y) != nil && clobber(x0, x1, s0, s1, or)) {
					continue
				}
				b = mergePoint(b, x0, x1, y)
				v0 := b.NewValue0(x0.Pos, OpAMD64ORQ, v.Type)
				v.copyOf(v0)
				v1 := b.NewValue0(x0.Pos, OpAMD64SHLQconst, v.Type)
				v1.AuxInt = int8ToAuxInt(j0)
				v2 := b.NewValue0(x0.Pos, OpAMD64MOVWload, typ.UInt16)
				v2.AuxInt = int32ToAuxInt(i)
				v2.Aux = symToAux(s)
				v2.AddArg2(p0, mem)
				v1.AddArg(v2)
				v0.AddArg2(v1, y)
				return true
			}
		}
		break
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVWload [i1] {s} p mem)) or:(ORQ s0:(SHLQconst [j0] x0:(MOVWload [i0] {s} p mem)) y))
	// cond: i1 == i0+2 && j1 == j0+16 && j0 % 32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b,x0,x1,y) != nil && clobber(x0, x1, s0, s1, or)
	// result: @mergePoint(b,x0,x1,y) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLload [i0] {s} p mem)) y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s1 := v_0
			if s1.Op != OpAMD64SHLQconst {
				continue
			}
			j1 := auxIntToInt8(s1.AuxInt)
			x1 := s1.Args[0]
			if x1.Op != OpAMD64MOVWload {
				continue
			}
			i1 := auxIntToInt32(x1.AuxInt)
			s := auxToSym(x1.Aux)
			mem := x1.Args[1]
			p := x1.Args[0]
			or := v_1
			if or.Op != OpAMD64ORQ {
				continue
			}
			_ = or.Args[1]
			or_0 := or.Args[0]
			or_1 := or.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, or_0, or_1 = _i1+1, or_1, or_0 {
				s0 := or_0
				if s0.Op != OpAMD64SHLQconst {
					continue
				}
				j0 := auxIntToInt8(s0.AuxInt)
				x0 := s0.Args[0]
				if x0.Op != OpAMD64MOVWload {
					continue
				}
				i0 := auxIntToInt32(x0.AuxInt)
				if auxToSym(x0.Aux) != s {
					continue
				}
				_ = x0.Args[1]
				if p != x0.Args[0] || mem != x0.Args[1] {
					continue
				}
				y := or_1
				if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1, y) != nil && clobber(x0, x1, s0, s1, or)) {
					continue
				}
				b = mergePoint(b, x0, x1, y)
				v0 := b.NewValue0(x0.Pos, OpAMD64ORQ, v.Type)
				v.copyOf(v0)
				v1 := b.NewValue0(x0.Pos, OpAMD64SHLQconst, v.Type)
				v1.AuxInt = int8ToAuxInt(j0)
				v2 := b.NewValue0(x0.Pos, OpAMD64MOVLload, typ.UInt32)
				v2.AuxInt = int32ToAuxInt(i0)
				v2.Aux = symToAux(s)
				v2.AddArg2(p, mem)
				v1.AddArg(v2)
				v0.AddArg2(v1, y)
				return true
			}
		}
		break
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVWload [i] {s} p1 mem)) or:(ORQ s0:(SHLQconst [j0] x0:(MOVWload [i] {s} p0 mem)) y))
	// cond: j1 == j0+16 && j0 % 32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && sequentialAddresses(p0, p1, 2) && mergePoint(b,x0,x1,y) != nil && clobber(x0, x1, s0, s1, or)
	// result: @mergePoint(b,x0,x1,y) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLload [i] {s} p0 mem)) y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s1 := v_0
			if s1.Op != OpAMD64SHLQconst {
				continue
			}
			j1 := auxIntToInt8(s1.AuxInt)
			x1 := s1.Args[0]
			if x1.Op != OpAMD64MOVWload {
				continue
			}
			i := auxIntToInt32(x1.AuxInt)
			s := auxToSym(x1.Aux)
			mem := x1.Args[1]
			p1 := x1.Args[0]
			or := v_1
			if or.Op != OpAMD64ORQ {
				continue
			}
			_ = or.Args[1]
			or_0 := or.Args[0]
			or_1 := or.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, or_0, or_1 = _i1+1, or_1, or_0 {
				s0 := or_0
				if s0.Op != OpAMD64SHLQconst {
					continue
				}
				j0 := auxIntToInt8(s0.AuxInt)
				x0 := s0.Args[0]
				if x0.Op != OpAMD64MOVWload || auxIntToInt32(x0.AuxInt) != i || auxToSym(x0.Aux) != s {
					continue
				}
				_ = x0.Args[1]
				p0 := x0.Args[0]
				if mem != x0.Args[1] {
					continue
				}
				y := or_1
				if !(j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && sequentialAddresses(p0, p1, 2) && mergePoint(b, x0, x1, y) != nil && clobber(x0, x1, s0, s1, or)) {
					continue
				}
				b = mergePoint(b, x0, x1, y)
				v0 := b.NewValue0(x0.Pos, OpAMD64ORQ, v.Type)
				v.copyOf(v0)
				v1 := b.NewValue0(x0.Pos, OpAMD64SHLQconst, v.Type)
				v1.AuxInt = int8ToAuxInt(j0)
				v2 := b.NewValue0(x0.Pos, OpAMD64MOVLload, typ.UInt32)
				v2.AuxInt = int32ToAuxInt(i)
				v2.Aux = symToAux(s)
				v2.AddArg2(p0, mem)
				v1.AddArg(v2)
				v0.AddArg2(v1, y)
				return true
			}
		}
		break
	}
	// match: (ORQ x1:(MOVBload [i1] {s} p mem) sh:(SHLQconst [8] x0:(MOVBload [i0] {s} p mem)))
	// cond: i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0, x1, sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWload [i0] {s} p mem))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x1 := v_0
			if x1.Op != OpAMD64MOVBload {
				continue
			}
			i1 := auxIntToInt32(x1.AuxInt)
			s := auxToSym(x1.Aux)
			mem := x1.Args[1]
			p := x1.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLQconst || auxIntToInt8(sh.AuxInt) != 8 {
				continue
			}
			x0 := sh.Args[0]
			if x0.Op != OpAMD64MOVBload {
				continue
			}
			i0 := auxIntToInt32(x0.AuxInt)
			if auxToSym(x0.Aux) != s {
				continue
			}
			_ = x0.Args[1]
			if p != x0.Args[0] || mem != x0.Args[1] || !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0, x1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x0.Pos, OpAMD64ROLWconst, v.Type)
			v.copyOf(v0)
			v0.AuxInt = int8ToAuxInt(8)
			v1 := b.NewValue0(x0.Pos, OpAMD64MOVWload, typ.UInt16)
			v1.AuxInt = int32ToAuxInt(i0)
			v1.Aux = symToAux(s)
			v1.AddArg2(p, mem)
			v0.AddArg(v1)
			return true
		}
		break
	}
	// match: (ORQ x1:(MOVBload [i] {s} p1 mem) sh:(SHLQconst [8] x0:(MOVBload [i] {s} p0 mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 1) && mergePoint(b,x0,x1) != nil && clobber(x0, x1, sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWload [i] {s} p0 mem))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x1 := v_0
			if x1.Op != OpAMD64MOVBload {
				continue
			}
			i := auxIntToInt32(x1.AuxInt)
			s := auxToSym(x1.Aux)
			mem := x1.Args[1]
			p1 := x1.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLQconst || auxIntToInt8(sh.AuxInt) != 8 {
				continue
			}
			x0 := sh.Args[0]
			if x0.Op != OpAMD64MOVBload || auxIntToInt32(x0.AuxInt) != i || auxToSym(x0.Aux) != s {
				continue
			}
			_ = x0.Args[1]
			p0 := x0.Args[0]
			if mem != x0.Args[1] || !(x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 1) && mergePoint(b, x0, x1) != nil && clobber(x0, x1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x0.Pos, OpAMD64ROLWconst, v.Type)
			v.copyOf(v0)
			v0.AuxInt = int8ToAuxInt(8)
			v1 := b.NewValue0(x0.Pos, OpAMD64MOVWload, typ.UInt16)
			v1.AuxInt = int32ToAuxInt(i)
			v1.Aux = symToAux(s)
			v1.AddArg2(p0, mem)
			v0.AddArg(v1)
			return true
		}
		break
	}
	// match: (ORQ r1:(ROLWconst [8] x1:(MOVWload [i1] {s} p mem)) sh:(SHLQconst [16] r0:(ROLWconst [8] x0:(MOVWload [i0] {s} p mem))))
	// cond: i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0, x1, r0, r1, sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLload [i0] {s} p mem))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			r1 := v_0
			if r1.Op != OpAMD64ROLWconst || auxIntToInt8(r1.AuxInt) != 8 {
				continue
			}
			x1 := r1.Args[0]
			if x1.Op != OpAMD64MOVWload {
				continue
			}
			i1 := auxIntToInt32(x1.AuxInt)
			s := auxToSym(x1.Aux)
			mem := x1.Args[1]
			p := x1.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLQconst || auxIntToInt8(sh.AuxInt) != 16 {
				continue
			}
			r0 := sh.Args[0]
			if r0.Op != OpAMD64ROLWconst || auxIntToInt8(r0.AuxInt) != 8 {
				continue
			}
			x0 := r0.Args[0]
			if x0.Op != OpAMD64MOVWload {
				continue
			}
			i0 := auxIntToInt32(x0.AuxInt)
			if auxToSym(x0.Aux) != s {
				continue
			}
			_ = x0.Args[1]
			if p != x0.Args[0] || mem != x0.Args[1] || !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0, x1, r0, r1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x0.Pos, OpAMD64BSWAPL, v.Type)
			v.copyOf(v0)
			v1 := b.NewValue0(x0.Pos, OpAMD64MOVLload, typ.UInt32)
			v1.AuxInt = int32ToAuxInt(i0)
			v1.Aux = symToAux(s)
			v1.AddArg2(p, mem)
			v0.AddArg(v1)
			return true
		}
		break
	}
	// match: (ORQ r1:(ROLWconst [8] x1:(MOVWload [i] {s} p1 mem)) sh:(SHLQconst [16] r0:(ROLWconst [8] x0:(MOVWload [i] {s} p0 mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 2) && mergePoint(b,x0,x1) != nil && clobber(x0, x1, r0, r1, sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLload [i] {s} p0 mem))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			r1 := v_0
			if r1.Op != OpAMD64ROLWconst || auxIntToInt8(r1.AuxInt) != 8 {
				continue
			}
			x1 := r1.Args[0]
			if x1.Op != OpAMD64MOVWload {
				continue
			}
			i := auxIntToInt32(x1.AuxInt)
			s := auxToSym(x1.Aux)
			mem := x1.Args[1]
			p1 := x1.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLQconst || auxIntToInt8(sh.AuxInt) != 16 {
				continue
			}
			r0 := sh.Args[0]
			if r0.Op != OpAMD64ROLWconst || auxIntToInt8(r0.AuxInt) != 8 {
				continue
			}
			x0 := r0.Args[0]
			if x0.Op != OpAMD64MOVWload || auxIntToInt32(x0.AuxInt) != i || auxToSym(x0.Aux) != s {
				continue
			}
			_ = x0.Args[1]
			p0 := x0.Args[0]
			if mem != x0.Args[1] || !(x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 2) && mergePoint(b, x0, x1) != nil && clobber(x0, x1, r0, r1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x0.Pos, OpAMD64BSWAPL, v.Type)
			v.copyOf(v0)
			v1 := b.NewValue0(x0.Pos, OpAMD64MOVLload, typ.UInt32)
			v1.AuxInt = int32ToAuxInt(i)
			v1.Aux = symToAux(s)
			v1.AddArg2(p0, mem)
			v0.AddArg(v1)
			return true
		}
		break
	}
	// match: (ORQ r1:(BSWAPL x1:(MOVLload [i1] {s} p mem)) sh:(SHLQconst [32] r0:(BSWAPL x0:(MOVLload [i0] {s} p mem))))
	// cond: i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0, x1, r0, r1, sh)
	// result: @mergePoint(b,x0,x1) (BSWAPQ <v.Type> (MOVQload [i0] {s} p mem))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			r1 := v_0
			if r1.Op != OpAMD64BSWAPL {
				continue
			}
			x1 := r1.Args[0]
			if x1.Op != OpAMD64MOVLload {
				continue
			}
			i1 := auxIntToInt32(x1.AuxInt)
			s := auxToSym(x1.Aux)
			mem := x1.Args[1]
			p := x1.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLQconst || auxIntToInt8(sh.AuxInt) != 32 {
				continue
			}
			r0 := sh.Args[0]
			if r0.Op != OpAMD64BSWAPL {
				continue
			}
			x0 := r0.Args[0]
			if x0.Op != OpAMD64MOVLload {
				continue
			}
			i0 := auxIntToInt32(x0.AuxInt)
			if auxToSym(x0.Aux) != s {
				continue
			}
			_ = x0.Args[1]
			if p != x0.Args[0] || mem != x0.Args[1] || !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0, x1, r0, r1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x0.Pos, OpAMD64BSWAPQ, v.Type)
			v.copyOf(v0)
			v1 := b.NewValue0(x0.Pos, OpAMD64MOVQload, typ.UInt64)
			v1.AuxInt = int32ToAuxInt(i0)
			v1.Aux = symToAux(s)
			v1.AddArg2(p, mem)
			v0.AddArg(v1)
			return true
		}
		break
	}
	// match: (ORQ r1:(BSWAPL x1:(MOVLload [i] {s} p1 mem)) sh:(SHLQconst [32] r0:(BSWAPL x0:(MOVLload [i] {s} p0 mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 4) && mergePoint(b,x0,x1) != nil && clobber(x0, x1, r0, r1, sh)
	// result: @mergePoint(b,x0,x1) (BSWAPQ <v.Type> (MOVQload [i] {s} p0 mem))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			r1 := v_0
			if r1.Op != OpAMD64BSWAPL {
				continue
			}
			x1 := r1.Args[0]
			if x1.Op != OpAMD64MOVLload {
				continue
			}
			i := auxIntToInt32(x1.AuxInt)
			s := auxToSym(x1.Aux)
			mem := x1.Args[1]
			p1 := x1.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLQconst || auxIntToInt8(sh.AuxInt) != 32 {
				continue
			}
			r0 := sh.Args[0]
			if r0.Op != OpAMD64BSWAPL {
				continue
			}
			x0 := r0.Args[0]
			if x0.Op != OpAMD64MOVLload || auxIntToInt32(x0.AuxInt) != i || auxToSym(x0.Aux) != s {
				continue
			}
			_ = x0.Args[1]
			p0 := x0.Args[0]
			if mem != x0.Args[1] || !(x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p0, p1, 4) && mergePoint(b, x0, x1) != nil && clobber(x0, x1, r0, r1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x0.Pos, OpAMD64BSWAPQ, v.Type)
			v.copyOf(v0)
			v1 := b.NewValue0(x0.Pos, OpAMD64MOVQload, typ.UInt64)
			v1.AuxInt = int32ToAuxInt(i)
			v1.Aux = symToAux(s)
			v1.AddArg2(p0, mem)
			v0.AddArg(v1)
			return true
		}
		break
	}
	// match: (ORQ s0:(SHLQconst [j0] x0:(MOVBload [i0] {s} p mem)) or:(ORQ s1:(SHLQconst [j1] x1:(MOVBload [i1] {s} p mem)) y))
	// cond: i1 == i0+1 && j1 == j0-8 && j1 % 16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b,x0,x1,y) != nil && clobber(x0, x1, s0, s1, or)
	// result: @mergePoint(b,x0,x1,y) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWload [i0] {s} p mem))) y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s0 := v_0
			if s0.Op != OpAMD64SHLQconst {
				continue
			}
			j0 := auxIntToInt8(s0.AuxInt)
			x0 := s0.Args[0]
			if x0.Op != OpAMD64MOVBload {
				continue
			}
			i0 := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p := x0.Args[0]
			or := v_1
			if or.Op != OpAMD64ORQ {
				continue
			}
			_ = or.Args[1]
			or_0 := or.Args[0]
			or_1 := or.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, or_0, or_1 = _i1+1, or_1, or_0 {
				s1 := or_0
				if s1.Op != OpAMD64SHLQconst {
					continue
				}
				j1 := auxIntToInt8(s1.AuxInt)
				x1 := s1.Args[0]
				if x1.Op != OpAMD64MOVBload {
					continue
				}
				i1 := auxIntToInt32(x1.AuxInt)
				if auxToSym(x1.Aux) != s {
					continue
				}
				_ = x1.Args[1]
				if p != x1.Args[0] || mem != x1.Args[1] {
					continue
				}
				y := or_1
				if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1, y) != nil && clobber(x0, x1, s0, s1, or)) {
					continue
				}
				b = mergePoint(b, x0, x1, y)
				v0 := b.NewValue0(x1.Pos, OpAMD64ORQ, v.Type)
				v.copyOf(v0)
				v1 := b.NewValue0(x1.Pos, OpAMD64SHLQconst, v.Type)
				v1.AuxInt = int8ToAuxInt(j1)
				v2 := b.NewValue0(x1.Pos, OpAMD64ROLWconst, typ.UInt16)
				v2.AuxInt = int8ToAuxInt(8)
				v3 := b.NewValue0(x1.Pos, OpAMD64MOVWload, typ.UInt16)
				v3.AuxInt = int32ToAuxInt(i0)
				v3.Aux = symToAux(s)
				v3.AddArg2(p, mem)
				v2.AddArg(v3)
				v1.AddArg(v2)
				v0.AddArg2(v1, y)
				return true
			}
		}
		break
	}
	// match: (ORQ s0:(SHLQconst [j0] x0:(MOVBload [i] {s} p0 mem)) or:(ORQ s1:(SHLQconst [j1] x1:(MOVBload [i] {s} p1 mem)) y))
	// cond: j1 == j0-8 && j1 % 16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && sequentialAddresses(p0, p1, 1) && mergePoint(b,x0,x1,y) != nil && clobber(x0, x1, s0, s1, or)
	// result: @mergePoint(b,x0,x1,y) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWload [i] {s} p0 mem))) y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s0 := v_0
			if s0.Op != OpAMD64SHLQconst {
				continue
			}
			j0 := auxIntToInt8(s0.AuxInt)
			x0 := s0.Args[0]
			if x0.Op != OpAMD64MOVBload {
				continue
			}
			i := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p0 := x0.Args[0]
			or := v_1
			if or.Op != OpAMD64ORQ {
				continue
			}
			_ = or.Args[1]
			or_0 := or.Args[0]
			or_1 := or.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, or_0, or_1 = _i1+1, or_1, or_0 {
				s1 := or_0
				if s1.Op != OpAMD64SHLQconst {
					continue
				}
				j1 := auxIntToInt8(s1.AuxInt)
				x1 := s1.Args[0]
				if x1.Op != OpAMD64MOVBload || auxIntToInt32(x1.AuxInt) != i || auxToSym(x1.Aux) != s {
					continue
				}
				_ = x1.Args[1]
				p1 := x1.Args[0]
				if mem != x1.Args[1] {
					continue
				}
				y := or_1
				if !(j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && sequentialAddresses(p0, p1, 1) && mergePoint(b, x0, x1, y) != nil && clobber(x0, x1, s0, s1, or)) {
					continue
				}
				b = mergePoint(b, x0, x1, y)
				v0 := b.NewValue0(x1.Pos, OpAMD64ORQ, v.Type)
				v.copyOf(v0)
				v1 := b.NewValue0(x1.Pos, OpAMD64SHLQconst, v.Type)
				v1.AuxInt = int8ToAuxInt(j1)
				v2 := b.NewValue0(x1.Pos, OpAMD64ROLWconst, typ.UInt16)
				v2.AuxInt = int8ToAuxInt(8)
				v3 := b.NewValue0(x1.Pos, OpAMD64MOVWload, typ.UInt16)
				v3.AuxInt = int32ToAuxInt(i)
				v3.Aux = symToAux(s)
				v3.AddArg2(p0, mem)
				v2.AddArg(v3)
				v1.AddArg(v2)
				v0.AddArg2(v1, y)
				return true
			}
		}
		break
	}
	// match: (ORQ s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWload [i0] {s} p mem))) or:(ORQ s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWload [i1] {s} p mem))) y))
	// cond: i1 == i0+2 && j1 == j0-16 && j1 % 32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b,x0,x1,y) != nil && clobber(x0, x1, r0, r1, s0, s1, or)
	// result: @mergePoint(b,x0,x1,y) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLload [i0] {s} p mem))) y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s0 := v_0
			if s0.Op != OpAMD64SHLQconst {
				continue
			}
			j0 := auxIntToInt8(s0.AuxInt)
			r0 := s0.Args[0]
			if r0.Op != OpAMD64ROLWconst || auxIntToInt8(r0.AuxInt) != 8 {
				continue
			}
			x0 := r0.Args[0]
			if x0.Op != OpAMD64MOVWload {
				continue
			}
			i0 := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p := x0.Args[0]
			or := v_1
			if or.Op != OpAMD64ORQ {
				continue
			}
			_ = or.Args[1]
			or_0 := or.Args[0]
			or_1 := or.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, or_0, or_1 = _i1+1, or_1, or_0 {
				s1 := or_0
				if s1.Op != OpAMD64SHLQconst {
					continue
				}
				j1 := auxIntToInt8(s1.AuxInt)
				r1 := s1.Args[0]
				if r1.Op != OpAMD64ROLWconst || auxIntToInt8(r1.AuxInt) != 8 {
					continue
				}
				x1 := r1.Args[0]
				if x1.Op != OpAMD64MOVWload {
					continue
				}
				i1 := auxIntToInt32(x1.AuxInt)
				if auxToSym(x1.Aux) != s {
					continue
				}
				_ = x1.Args[1]
				if p != x1.Args[0] || mem != x1.Args[1] {
					continue
				}
				y := or_1
				if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1, y) != nil && clobber(x0, x1, r0, r1, s0, s1, or)) {
					continue
				}
				b = mergePoint(b, x0, x1, y)
				v0 := b.NewValue0(x1.Pos, OpAMD64ORQ, v.Type)
				v.copyOf(v0)
				v1 := b.NewValue0(x1.Pos, OpAMD64SHLQconst, v.Type)
				v1.AuxInt = int8ToAuxInt(j1)
				v2 := b.NewValue0(x1.Pos, OpAMD64BSWAPL, typ.UInt32)
				v3 := b.NewValue0(x1.Pos, OpAMD64MOVLload, typ.UInt32)
				v3.AuxInt = int32ToAuxInt(i0)
				v3.Aux = symToAux(s)
				v3.AddArg2(p, mem)
				v2.AddArg(v3)
				v1.AddArg(v2)
				v0.AddArg2(v1, y)
				return true
			}
		}
		break
	}
	// match: (ORQ s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWload [i] {s} p0 mem))) or:(ORQ s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWload [i] {s} p1 mem))) y))
	// cond: j1 == j0-16 && j1 % 32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && sequentialAddresses(p0, p1, 2) && mergePoint(b,x0,x1,y) != nil && clobber(x0, x1, r0, r1, s0, s1, or)
	// result: @mergePoint(b,x0,x1,y) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLload [i] {s} p0 mem))) y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			s0 := v_0
			if s0.Op != OpAMD64SHLQconst {
				continue
			}
			j0 := auxIntToInt8(s0.AuxInt)
			r0 := s0.Args[0]
			if r0.Op != OpAMD64ROLWconst || auxIntToInt8(r0.AuxInt) != 8 {
				continue
			}
			x0 := r0.Args[0]
			if x0.Op != OpAMD64MOVWload {
				continue
			}
			i := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p0 := x0.Args[0]
			or := v_1
			if or.Op != OpAMD64ORQ {
				continue
			}
			_ = or.Args[1]
			or_0 := or.Args[0]
			or_1 := or.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, or_0, or_1 = _i1+1, or_1, or_0 {
				s1 := or_0
				if s1.Op != OpAMD64SHLQconst {
					continue
				}
				j1 := auxIntToInt8(s1.AuxInt)
				r1 := s1.Args[0]
				if r1.Op != OpAMD64ROLWconst || auxIntToInt8(r1.AuxInt) != 8 {
					continue
				}
				x1 := r1.Args[0]
				if x1.Op != OpAMD64MOVWload || auxIntToInt32(x1.AuxInt) != i || auxToSym(x1.Aux) != s {
					continue
				}
				_ = x1.Args[1]
				p1 := x1.Args[0]
				if mem != x1.Args[1] {
					continue
				}
				y := or_1
				if !(j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && sequentialAddresses(p0, p1, 2) && mergePoint(b, x0, x1, y) != nil && clobber(x0, x1, r0, r1, s0, s1, or)) {
					continue
				}
				b = mergePoint(b, x0, x1, y)
				v0 := b.NewValue0(x1.Pos, OpAMD64ORQ, v.Type)
				v.copyOf(v0)
				v1 := b.NewValue0(x1.Pos, OpAMD64SHLQconst, v.Type)
				v1.AuxInt = int8ToAuxInt(j1)
				v2 := b.NewValue0(x1.Pos, OpAMD64BSWAPL, typ.UInt32)
				v3 := b.NewValue0(x1.Pos, OpAMD64MOVLload, typ.UInt32)
				v3.AuxInt = int32ToAuxInt(i)
				v3.Aux = symToAux(s)
				v3.AddArg2(p0, mem)
				v2.AddArg(v3)
				v1.AddArg(v2)
				v0.AddArg2(v1, y)
				return true
			}
		}
		break
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
	// match: (ORQ x0:(MOVBELload [i0] {s} p mem) sh:(SHLQconst [32] x1:(MOVBELload [i1] {s} p mem)))
	// cond: i0 == i1+4 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0, x1, sh)
	// result: @mergePoint(b,x0,x1) (MOVBEQload [i1] {s} p mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			if x0.Op != OpAMD64MOVBELload {
				continue
			}
			i0 := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p := x0.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLQconst || auxIntToInt8(sh.AuxInt) != 32 {
				continue
			}
			x1 := sh.Args[0]
			if x1.Op != OpAMD64MOVBELload {
				continue
			}
			i1 := auxIntToInt32(x1.AuxInt)
			if auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			if p != x1.Args[0] || mem != x1.Args[1] || !(i0 == i1+4 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0, x1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x1.Pos, OpAMD64MOVBEQload, typ.UInt64)
			v.copyOf(v0)
			v0.AuxInt = int32ToAuxInt(i1)
			v0.Aux = symToAux(s)
			v0.AddArg2(p, mem)
			return true
		}
		break
	}
	// match: (ORQ x0:(MOVBELload [i] {s} p0 mem) sh:(SHLQconst [32] x1:(MOVBELload [i] {s} p1 mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p1, p0, 4) && mergePoint(b,x0,x1) != nil && clobber(x0, x1, sh)
	// result: @mergePoint(b,x0,x1) (MOVBEQload [i] {s} p1 mem)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			if x0.Op != OpAMD64MOVBELload {
				continue
			}
			i := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p0 := x0.Args[0]
			sh := v_1
			if sh.Op != OpAMD64SHLQconst || auxIntToInt8(sh.AuxInt) != 32 {
				continue
			}
			x1 := sh.Args[0]
			if x1.Op != OpAMD64MOVBELload || auxIntToInt32(x1.AuxInt) != i || auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			p1 := x1.Args[0]
			if mem != x1.Args[1] || !(x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && sequentialAddresses(p1, p0, 4) && mergePoint(b, x0, x1) != nil && clobber(x0, x1, sh)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x1.Pos, OpAMD64MOVBEQload, typ.UInt64)
			v.copyOf(v0)
			v0.AuxInt = int32ToAuxInt(i)
			v0.Aux = symToAux(s)
			v0.AddArg2(p1, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ORQconst [c] x)
	// cond: isUint64PowerOfTwo(int64(c)) && uint64(c) >= 128
	// result: (BTSQconst [int8(log32(c))] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isUint64PowerOfTwo(int64(c)) && uint64(c) >= 128) {
			break
		}
		v.reset(OpAMD64BTSQconst)
		v.AuxInt = int8ToAuxInt(int8(log32(c)))
		v.AddArg(x)
		return true
	}
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
	// match: (ORQconst [c] (BTSQconst [d] x))
	// cond: is32Bit(int64(c) | 1<<uint32(d))
	// result: (ORQconst [c | 1<<uint32(d)] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64BTSQconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(int64(c) | 1<<uint32(d))) {
			break
		}
		v.reset(OpAMD64ORQconst)
		v.AuxInt = int32ToAuxInt(c | 1<<uint32(d))
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
	// match: (SARL x y)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (SARXL x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64SARXL)
		v.AddArg2(x, y)
		return true
	}
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
	// match: (SARQ x y)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (SARXQ x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64SARXQ)
		v.AddArg2(x, y)
		return true
	}
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
func rewriteValueAMD64_OpAMD64SARXL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SARXL x (MOVQconst [c]))
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
	// match: (SARXL x (MOVLconst [c]))
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
	// match: (SARXL x (ADDQconst [c] y))
	// cond: c & 31 == 0
	// result: (SARXL x y)
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
		v.reset(OpAMD64SARXL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SARXL x (NEGQ <t> (ADDQconst [c] y)))
	// cond: c & 31 == 0
	// result: (SARXL x (NEGQ <t> y))
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
		v.reset(OpAMD64SARXL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SARXL x (ANDQconst [c] y))
	// cond: c & 31 == 31
	// result: (SARXL x y)
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
		v.reset(OpAMD64SARXL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SARXL x (NEGQ <t> (ANDQconst [c] y)))
	// cond: c & 31 == 31
	// result: (SARXL x (NEGQ <t> y))
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
		v.reset(OpAMD64SARXL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SARXL x (ADDLconst [c] y))
	// cond: c & 31 == 0
	// result: (SARXL x y)
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
		v.reset(OpAMD64SARXL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SARXL x (NEGL <t> (ADDLconst [c] y)))
	// cond: c & 31 == 0
	// result: (SARXL x (NEGL <t> y))
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
		v.reset(OpAMD64SARXL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SARXL x (ANDLconst [c] y))
	// cond: c & 31 == 31
	// result: (SARXL x y)
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
		v.reset(OpAMD64SARXL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SARXL x (NEGL <t> (ANDLconst [c] y)))
	// cond: c & 31 == 31
	// result: (SARXL x (NEGL <t> y))
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
		v.reset(OpAMD64SARXL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SARXL l:(MOVLload [off] {sym} ptr mem) x)
	// cond: canMergeLoad(v, l) && clobber(l)
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
		if !(canMergeLoad(v, l) && clobber(l)) {
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
func rewriteValueAMD64_OpAMD64SARXQ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SARXQ x (MOVQconst [c]))
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
	// match: (SARXQ x (MOVLconst [c]))
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
	// match: (SARXQ x (ADDQconst [c] y))
	// cond: c & 63 == 0
	// result: (SARXQ x y)
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
		v.reset(OpAMD64SARXQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SARXQ x (NEGQ <t> (ADDQconst [c] y)))
	// cond: c & 63 == 0
	// result: (SARXQ x (NEGQ <t> y))
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
		v.reset(OpAMD64SARXQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SARXQ x (ANDQconst [c] y))
	// cond: c & 63 == 63
	// result: (SARXQ x y)
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
		v.reset(OpAMD64SARXQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SARXQ x (NEGQ <t> (ANDQconst [c] y)))
	// cond: c & 63 == 63
	// result: (SARXQ x (NEGQ <t> y))
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
		v.reset(OpAMD64SARXQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SARXQ x (ADDLconst [c] y))
	// cond: c & 63 == 0
	// result: (SARXQ x y)
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
		v.reset(OpAMD64SARXQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SARXQ x (NEGL <t> (ADDLconst [c] y)))
	// cond: c & 63 == 0
	// result: (SARXQ x (NEGL <t> y))
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
		v.reset(OpAMD64SARXQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SARXQ x (ANDLconst [c] y))
	// cond: c & 63 == 63
	// result: (SARXQ x y)
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
		v.reset(OpAMD64SARXQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SARXQ x (NEGL <t> (ANDLconst [c] y)))
	// cond: c & 63 == 63
	// result: (SARXQ x (NEGL <t> y))
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
		v.reset(OpAMD64SARXQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SARXQ l:(MOVQload [off] {sym} ptr mem) x)
	// cond: canMergeLoad(v, l) && clobber(l)
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
		if !(canMergeLoad(v, l) && clobber(l)) {
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
	// match: (SETEQ (TESTL (SHLXL (MOVLconst [1]) x) y))
	// result: (SETAE (BTL x y))
	for {
		if v_0.Op != OpAMD64TESTL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpAMD64SHLXL {
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
	// match: (SETEQ (TESTQ (SHLXQ (MOVQconst [1]) x) y))
	// result: (SETAE (BTQ x y))
	for {
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpAMD64SHLXQ {
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
	// cond: isUint32PowerOfTwo(int64(c))
	// result: (SETAE (BTLconst [int8(log32(c))] x))
	for {
		if v_0.Op != OpAMD64TESTLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isUint32PowerOfTwo(int64(c))) {
			break
		}
		v.reset(OpAMD64SETAE)
		v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
		v0.AuxInt = int8ToAuxInt(int8(log32(c)))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETEQ (TESTQconst [c] x))
	// cond: isUint64PowerOfTwo(int64(c))
	// result: (SETAE (BTQconst [int8(log32(c))] x))
	for {
		if v_0.Op != OpAMD64TESTQconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isUint64PowerOfTwo(int64(c))) {
			break
		}
		v.reset(OpAMD64SETAE)
		v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
		v0.AuxInt = int8ToAuxInt(int8(log32(c)))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETEQ (TESTQ (MOVQconst [c]) x))
	// cond: isUint64PowerOfTwo(c)
	// result: (SETAE (BTQconst [int8(log64(c))] x))
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
			if !(isUint64PowerOfTwo(c)) {
				continue
			}
			v.reset(OpAMD64SETAE)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(int8(log64(c)))
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
	// match: (SETEQstore [off] {sym} ptr (TESTL (SHLXL (MOVLconst [1]) x) y) mem)
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
			if v_1_0.Op != OpAMD64SHLXL {
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
	// match: (SETEQstore [off] {sym} ptr (TESTQ (SHLXQ (MOVQconst [1]) x) y) mem)
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
			if v_1_0.Op != OpAMD64SHLXQ {
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
	// cond: isUint32PowerOfTwo(int64(c))
	// result: (SETAEstore [off] {sym} ptr (BTLconst [int8(log32(c))] x) mem)
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
		if !(isUint32PowerOfTwo(int64(c))) {
			break
		}
		v.reset(OpAMD64SETAEstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
		v0.AuxInt = int8ToAuxInt(int8(log32(c)))
		v0.AddArg(x)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETEQstore [off] {sym} ptr (TESTQconst [c] x) mem)
	// cond: isUint64PowerOfTwo(int64(c))
	// result: (SETAEstore [off] {sym} ptr (BTQconst [int8(log32(c))] x) mem)
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
		if !(isUint64PowerOfTwo(int64(c))) {
			break
		}
		v.reset(OpAMD64SETAEstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
		v0.AuxInt = int8ToAuxInt(int8(log32(c)))
		v0.AddArg(x)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETEQstore [off] {sym} ptr (TESTQ (MOVQconst [c]) x) mem)
	// cond: isUint64PowerOfTwo(c)
	// result: (SETAEstore [off] {sym} ptr (BTQconst [int8(log64(c))] x) mem)
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
			if !(isUint64PowerOfTwo(c)) {
				continue
			}
			v.reset(OpAMD64SETAEstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(int8(log64(c)))
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
	// match: (SETNE (TESTL (SHLXL (MOVLconst [1]) x) y))
	// result: (SETB (BTL x y))
	for {
		if v_0.Op != OpAMD64TESTL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpAMD64SHLXL {
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
	// match: (SETNE (TESTQ (SHLXQ (MOVQconst [1]) x) y))
	// result: (SETB (BTQ x y))
	for {
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpAMD64SHLXQ {
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
	// cond: isUint32PowerOfTwo(int64(c))
	// result: (SETB (BTLconst [int8(log32(c))] x))
	for {
		if v_0.Op != OpAMD64TESTLconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isUint32PowerOfTwo(int64(c))) {
			break
		}
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
		v0.AuxInt = int8ToAuxInt(int8(log32(c)))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETNE (TESTQconst [c] x))
	// cond: isUint64PowerOfTwo(int64(c))
	// result: (SETB (BTQconst [int8(log32(c))] x))
	for {
		if v_0.Op != OpAMD64TESTQconst {
			break
		}
		c := auxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isUint64PowerOfTwo(int64(c))) {
			break
		}
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
		v0.AuxInt = int8ToAuxInt(int8(log32(c)))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETNE (TESTQ (MOVQconst [c]) x))
	// cond: isUint64PowerOfTwo(c)
	// result: (SETB (BTQconst [int8(log64(c))] x))
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
			if !(isUint64PowerOfTwo(c)) {
				continue
			}
			v.reset(OpAMD64SETB)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(int8(log64(c)))
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
	// match: (SETNEstore [off] {sym} ptr (TESTL (SHLXL (MOVLconst [1]) x) y) mem)
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
			if v_1_0.Op != OpAMD64SHLXL {
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
	// match: (SETNEstore [off] {sym} ptr (TESTQ (SHLXQ (MOVQconst [1]) x) y) mem)
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
			if v_1_0.Op != OpAMD64SHLXQ {
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
	// cond: isUint32PowerOfTwo(int64(c))
	// result: (SETBstore [off] {sym} ptr (BTLconst [int8(log32(c))] x) mem)
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
		if !(isUint32PowerOfTwo(int64(c))) {
			break
		}
		v.reset(OpAMD64SETBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
		v0.AuxInt = int8ToAuxInt(int8(log32(c)))
		v0.AddArg(x)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETNEstore [off] {sym} ptr (TESTQconst [c] x) mem)
	// cond: isUint64PowerOfTwo(int64(c))
	// result: (SETBstore [off] {sym} ptr (BTQconst [int8(log32(c))] x) mem)
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
		if !(isUint64PowerOfTwo(int64(c))) {
			break
		}
		v.reset(OpAMD64SETBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
		v0.AuxInt = int8ToAuxInt(int8(log32(c)))
		v0.AddArg(x)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (SETNEstore [off] {sym} ptr (TESTQ (MOVQconst [c]) x) mem)
	// cond: isUint64PowerOfTwo(c)
	// result: (SETBstore [off] {sym} ptr (BTQconst [int8(log64(c))] x) mem)
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
			if !(isUint64PowerOfTwo(c)) {
				continue
			}
			v.reset(OpAMD64SETBstore)
			v.AuxInt = int32ToAuxInt(off)
			v.Aux = symToAux(sym)
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(int8(log64(c)))
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
	// match: (SHLL x y)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (SHLXL x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64SHLXL)
		v.AddArg2(x, y)
		return true
	}
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
	return false
}
func rewriteValueAMD64_OpAMD64SHLLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SHLLconst [1] (SHRLconst [1] x))
	// result: (BTRLconst [0] x)
	for {
		if auxIntToInt8(v.AuxInt) != 1 || v_0.Op != OpAMD64SHRLconst || auxIntToInt8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64BTRLconst)
		v.AuxInt = int8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
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
	// match: (SHLQ x y)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (SHLXQ x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64SHLXQ)
		v.AddArg2(x, y)
		return true
	}
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
	return false
}
func rewriteValueAMD64_OpAMD64SHLQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SHLQconst [1] (SHRQconst [1] x))
	// result: (BTRQconst [0] x)
	for {
		if auxIntToInt8(v.AuxInt) != 1 || v_0.Op != OpAMD64SHRQconst || auxIntToInt8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64BTRQconst)
		v.AuxInt = int8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
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
func rewriteValueAMD64_OpAMD64SHLXL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SHLXL x (MOVQconst [c]))
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
	// match: (SHLXL x (MOVLconst [c]))
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
	// match: (SHLXL x (ADDQconst [c] y))
	// cond: c & 31 == 0
	// result: (SHLXL x y)
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
		v.reset(OpAMD64SHLXL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHLXL x (NEGQ <t> (ADDQconst [c] y)))
	// cond: c & 31 == 0
	// result: (SHLXL x (NEGQ <t> y))
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
		v.reset(OpAMD64SHLXL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHLXL x (ANDQconst [c] y))
	// cond: c & 31 == 31
	// result: (SHLXL x y)
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
		v.reset(OpAMD64SHLXL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHLXL x (NEGQ <t> (ANDQconst [c] y)))
	// cond: c & 31 == 31
	// result: (SHLXL x (NEGQ <t> y))
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
		v.reset(OpAMD64SHLXL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHLXL x (ADDLconst [c] y))
	// cond: c & 31 == 0
	// result: (SHLXL x y)
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
		v.reset(OpAMD64SHLXL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHLXL x (NEGL <t> (ADDLconst [c] y)))
	// cond: c & 31 == 0
	// result: (SHLXL x (NEGL <t> y))
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
		v.reset(OpAMD64SHLXL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHLXL x (ANDLconst [c] y))
	// cond: c & 31 == 31
	// result: (SHLXL x y)
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
		v.reset(OpAMD64SHLXL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHLXL x (NEGL <t> (ANDLconst [c] y)))
	// cond: c & 31 == 31
	// result: (SHLXL x (NEGL <t> y))
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
		v.reset(OpAMD64SHLXL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHLXL l:(MOVLload [off] {sym} ptr mem) x)
	// cond: canMergeLoad(v, l) && clobber(l)
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
		if !(canMergeLoad(v, l) && clobber(l)) {
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
func rewriteValueAMD64_OpAMD64SHLXQ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SHLXQ x (MOVQconst [c]))
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
	// match: (SHLXQ x (MOVLconst [c]))
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
	// match: (SHLXQ x (ADDQconst [c] y))
	// cond: c & 63 == 0
	// result: (SHLXQ x y)
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
		v.reset(OpAMD64SHLXQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHLXQ x (NEGQ <t> (ADDQconst [c] y)))
	// cond: c & 63 == 0
	// result: (SHLXQ x (NEGQ <t> y))
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
		v.reset(OpAMD64SHLXQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHLXQ x (ANDQconst [c] y))
	// cond: c & 63 == 63
	// result: (SHLXQ x y)
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
		v.reset(OpAMD64SHLXQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHLXQ x (NEGQ <t> (ANDQconst [c] y)))
	// cond: c & 63 == 63
	// result: (SHLXQ x (NEGQ <t> y))
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
		v.reset(OpAMD64SHLXQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHLXQ x (ADDLconst [c] y))
	// cond: c & 63 == 0
	// result: (SHLXQ x y)
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
		v.reset(OpAMD64SHLXQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHLXQ x (NEGL <t> (ADDLconst [c] y)))
	// cond: c & 63 == 0
	// result: (SHLXQ x (NEGL <t> y))
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
		v.reset(OpAMD64SHLXQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHLXQ x (ANDLconst [c] y))
	// cond: c & 63 == 63
	// result: (SHLXQ x y)
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
		v.reset(OpAMD64SHLXQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHLXQ x (NEGL <t> (ANDLconst [c] y)))
	// cond: c & 63 == 63
	// result: (SHLXQ x (NEGL <t> y))
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
		v.reset(OpAMD64SHLXQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHLXQ l:(MOVQload [off] {sym} ptr mem) x)
	// cond: canMergeLoad(v, l) && clobber(l)
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
		if !(canMergeLoad(v, l) && clobber(l)) {
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
	// match: (SHRL x y)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (SHRXL x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64SHRXL)
		v.AddArg2(x, y)
		return true
	}
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
	return false
}
func rewriteValueAMD64_OpAMD64SHRLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SHRLconst [1] (SHLLconst [1] x))
	// result: (BTRLconst [31] x)
	for {
		if auxIntToInt8(v.AuxInt) != 1 || v_0.Op != OpAMD64SHLLconst || auxIntToInt8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64BTRLconst)
		v.AuxInt = int8ToAuxInt(31)
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
	// match: (SHRQ x y)
	// cond: buildcfg.GOAMD64 >= 3
	// result: (SHRXQ x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOAMD64 >= 3) {
			break
		}
		v.reset(OpAMD64SHRXQ)
		v.AddArg2(x, y)
		return true
	}
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
	return false
}
func rewriteValueAMD64_OpAMD64SHRQconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SHRQconst [1] (SHLQconst [1] x))
	// result: (BTRQconst [63] x)
	for {
		if auxIntToInt8(v.AuxInt) != 1 || v_0.Op != OpAMD64SHLQconst || auxIntToInt8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
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
func rewriteValueAMD64_OpAMD64SHRXL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SHRXL x (MOVQconst [c]))
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
	// match: (SHRXL x (MOVLconst [c]))
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
	// match: (SHRXL x (ADDQconst [c] y))
	// cond: c & 31 == 0
	// result: (SHRXL x y)
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
		v.reset(OpAMD64SHRXL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHRXL x (NEGQ <t> (ADDQconst [c] y)))
	// cond: c & 31 == 0
	// result: (SHRXL x (NEGQ <t> y))
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
		v.reset(OpAMD64SHRXL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHRXL x (ANDQconst [c] y))
	// cond: c & 31 == 31
	// result: (SHRXL x y)
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
		v.reset(OpAMD64SHRXL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHRXL x (NEGQ <t> (ANDQconst [c] y)))
	// cond: c & 31 == 31
	// result: (SHRXL x (NEGQ <t> y))
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
		v.reset(OpAMD64SHRXL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHRXL x (ADDLconst [c] y))
	// cond: c & 31 == 0
	// result: (SHRXL x y)
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
		v.reset(OpAMD64SHRXL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHRXL x (NEGL <t> (ADDLconst [c] y)))
	// cond: c & 31 == 0
	// result: (SHRXL x (NEGL <t> y))
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
		v.reset(OpAMD64SHRXL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHRXL x (ANDLconst [c] y))
	// cond: c & 31 == 31
	// result: (SHRXL x y)
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
		v.reset(OpAMD64SHRXL)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHRXL x (NEGL <t> (ANDLconst [c] y)))
	// cond: c & 31 == 31
	// result: (SHRXL x (NEGL <t> y))
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
		v.reset(OpAMD64SHRXL)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHRXL l:(MOVLload [off] {sym} ptr mem) x)
	// cond: canMergeLoad(v, l) && clobber(l)
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
		if !(canMergeLoad(v, l) && clobber(l)) {
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
func rewriteValueAMD64_OpAMD64SHRXQ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SHRXQ x (MOVQconst [c]))
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
	// match: (SHRXQ x (MOVLconst [c]))
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
	// match: (SHRXQ x (ADDQconst [c] y))
	// cond: c & 63 == 0
	// result: (SHRXQ x y)
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
		v.reset(OpAMD64SHRXQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHRXQ x (NEGQ <t> (ADDQconst [c] y)))
	// cond: c & 63 == 0
	// result: (SHRXQ x (NEGQ <t> y))
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
		v.reset(OpAMD64SHRXQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHRXQ x (ANDQconst [c] y))
	// cond: c & 63 == 63
	// result: (SHRXQ x y)
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
		v.reset(OpAMD64SHRXQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHRXQ x (NEGQ <t> (ANDQconst [c] y)))
	// cond: c & 63 == 63
	// result: (SHRXQ x (NEGQ <t> y))
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
		v.reset(OpAMD64SHRXQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHRXQ x (ADDLconst [c] y))
	// cond: c & 63 == 0
	// result: (SHRXQ x y)
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
		v.reset(OpAMD64SHRXQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHRXQ x (NEGL <t> (ADDLconst [c] y)))
	// cond: c & 63 == 0
	// result: (SHRXQ x (NEGL <t> y))
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
		v.reset(OpAMD64SHRXQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHRXQ x (ANDLconst [c] y))
	// cond: c & 63 == 63
	// result: (SHRXQ x y)
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
		v.reset(OpAMD64SHRXQ)
		v.AddArg2(x, y)
		return true
	}
	// match: (SHRXQ x (NEGL <t> (ANDLconst [c] y)))
	// cond: c & 63 == 63
	// result: (SHRXQ x (NEGL <t> y))
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
		v.reset(OpAMD64SHRXQ)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SHRXQ l:(MOVQload [off] {sym} ptr mem) x)
	// cond: canMergeLoad(v, l) && clobber(l)
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
		if !(canMergeLoad(v, l) && clobber(l)) {
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
	// match: (XORL (SHLXL (MOVLconst [1]) y) x)
	// result: (BTCL x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64SHLXL {
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
	// match: (XORL (MOVLconst [c]) x)
	// cond: isUint32PowerOfTwo(int64(c)) && uint64(c) >= 128
	// result: (BTCLconst [int8(log32(c))] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64MOVLconst {
				continue
			}
			c := auxIntToInt32(v_0.AuxInt)
			x := v_1
			if !(isUint32PowerOfTwo(int64(c)) && uint64(c) >= 128) {
				continue
			}
			v.reset(OpAMD64BTCLconst)
			v.AuxInt = int8ToAuxInt(int8(log32(c)))
			v.AddArg(x)
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
	// match: (XORLconst [c] x)
	// cond: isUint32PowerOfTwo(int64(c)) && uint64(c) >= 128
	// result: (BTCLconst [int8(log32(c))] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isUint32PowerOfTwo(int64(c)) && uint64(c) >= 128) {
			break
		}
		v.reset(OpAMD64BTCLconst)
		v.AuxInt = int8ToAuxInt(int8(log32(c)))
		v.AddArg(x)
		return true
	}
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
	// match: (XORLconst [c] (BTCLconst [d] x))
	// result: (XORLconst [c ^ 1<<uint32(d)] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64BTCLconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpAMD64XORLconst)
		v.AuxInt = int32ToAuxInt(c ^ 1<<uint32(d))
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
	// match: (XORQ (SHLXQ (MOVQconst [1]) y) x)
	// result: (BTCQ x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64SHLXQ {
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
	// cond: isUint64PowerOfTwo(c) && uint64(c) >= 128
	// result: (BTCQconst [int8(log64(c))] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpAMD64MOVQconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			x := v_1
			if !(isUint64PowerOfTwo(c) && uint64(c) >= 128) {
				continue
			}
			v.reset(OpAMD64BTCQconst)
			v.AuxInt = int8ToAuxInt(int8(log64(c)))
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
	// match: (XORQconst [c] x)
	// cond: isUint64PowerOfTwo(int64(c)) && uint64(c) >= 128
	// result: (BTCQconst [int8(log32(c))] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		x := v_0
		if !(isUint64PowerOfTwo(int64(c)) && uint64(c) >= 128) {
			break
		}
		v.reset(OpAMD64BTCQconst)
		v.AuxInt = int8ToAuxInt(int8(log32(c)))
		v.AddArg(x)
		return true
	}
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
	// match: (XORQconst [c] (BTCQconst [d] x))
	// cond: is32Bit(int64(c) ^ 1<<uint32(d))
	// result: (XORQconst [c ^ 1<<uint32(d)] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpAMD64BTCQconst {
			break
		}
		d := auxIntToInt8(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(int64(c) ^ 1<<uint32(d))) {
			break
		}
		v.reset(OpAMD64XORQconst)
		v.AuxInt = int32ToAuxInt(c ^ 1<<uint32(d))
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
	// result: (BSFL (BTSLconst <typ.UInt32> [16] x))
	for {
		x := v_0
		v.reset(OpAMD64BSFL)
		v0 := b.NewValue0(v.Pos, OpAMD64BTSLconst, typ.UInt32)
		v0.AuxInt = int8ToAuxInt(16)
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
	// result: (BSFL (BTSLconst <typ.UInt32> [ 8] x))
	for {
		x := v_0
		v.reset(OpAMD64BSFL)
		v0 := b.NewValue0(v.Pos, OpAMD64BTSLconst, typ.UInt32)
		v0.AuxInt = int8ToAuxInt(8)
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
	return false
}
func rewriteValueAMD64_OpLocalAddr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (LocalAddr {sym} base _)
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
	// cond: config.useSSE
	// result: (MOVOstore dst (MOVOload src mem) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 16 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		if !(config.useSSE) {
			break
		}
		v.reset(OpAMD64MOVOstore)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVOload, types.TypeInt128)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [16] dst src mem)
	// cond: !config.useSSE
	// result: (MOVQstore [8] dst (MOVQload [8] src mem) (MOVQstore dst (MOVQload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 16 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		if !(!config.useSSE) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [32] dst src mem)
	// result: (Move [16] (OffPtr <dst.Type> dst [16]) (OffPtr <src.Type> src [16]) (Move [16] dst src mem))
	for {
		if auxIntToInt64(v.AuxInt) != 32 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(16)
		v0 := b.NewValue0(v.Pos, OpOffPtr, dst.Type)
		v0.AuxInt = int64ToAuxInt(16)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpOffPtr, src.Type)
		v1.AuxInt = int64ToAuxInt(16)
		v1.AddArg(src)
		v2 := b.NewValue0(v.Pos, OpMove, types.TypeMem)
		v2.AuxInt = int64ToAuxInt(16)
		v2.AddArg3(dst, src, mem)
		v.AddArg3(v0, v1, v2)
		return true
	}
	// match: (Move [48] dst src mem)
	// cond: config.useSSE
	// result: (Move [32] (OffPtr <dst.Type> dst [16]) (OffPtr <src.Type> src [16]) (Move [16] dst src mem))
	for {
		if auxIntToInt64(v.AuxInt) != 48 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		if !(config.useSSE) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, OpOffPtr, dst.Type)
		v0.AuxInt = int64ToAuxInt(16)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpOffPtr, src.Type)
		v1.AuxInt = int64ToAuxInt(16)
		v1.AddArg(src)
		v2 := b.NewValue0(v.Pos, OpMove, types.TypeMem)
		v2.AuxInt = int64ToAuxInt(16)
		v2.AddArg3(dst, src, mem)
		v.AddArg3(v0, v1, v2)
		return true
	}
	// match: (Move [64] dst src mem)
	// cond: config.useSSE
	// result: (Move [32] (OffPtr <dst.Type> dst [32]) (OffPtr <src.Type> src [32]) (Move [32] dst src mem))
	for {
		if auxIntToInt64(v.AuxInt) != 64 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		if !(config.useSSE) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, OpOffPtr, dst.Type)
		v0.AuxInt = int64ToAuxInt(32)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpOffPtr, src.Type)
		v1.AuxInt = int64ToAuxInt(32)
		v1.AddArg(src)
		v2 := b.NewValue0(v.Pos, OpMove, types.TypeMem)
		v2.AuxInt = int64ToAuxInt(32)
		v2.AddArg3(dst, src, mem)
		v.AddArg3(v0, v1, v2)
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
	// cond: s == 11 || s >= 13 && s <= 15
	// result: (MOVQstore [int32(s-8)] dst (MOVQload [int32(s-8)] src mem) (MOVQstore dst (MOVQload src mem) mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s == 11 || s >= 13 && s <= 15) {
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
	// cond: s > 16 && s%16 != 0 && s%16 <= 8
	// result: (Move [s-s%16] (OffPtr <dst.Type> dst [s%16]) (OffPtr <src.Type> src [s%16]) (MOVQstore dst (MOVQload src mem) mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 16 && s%16 != 0 && s%16 <= 8) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(s - s%16)
		v0 := b.NewValue0(v.Pos, OpOffPtr, dst.Type)
		v0.AuxInt = int64ToAuxInt(s % 16)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpOffPtr, src.Type)
		v1.AuxInt = int64ToAuxInt(s % 16)
		v1.AddArg(src)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVQstore, types.TypeMem)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v3.AddArg2(src, mem)
		v2.AddArg3(dst, v3, mem)
		v.AddArg3(v0, v1, v2)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 16 && s%16 != 0 && s%16 > 8 && config.useSSE
	// result: (Move [s-s%16] (OffPtr <dst.Type> dst [s%16]) (OffPtr <src.Type> src [s%16]) (MOVOstore dst (MOVOload src mem) mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 16 && s%16 != 0 && s%16 > 8 && config.useSSE) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(s - s%16)
		v0 := b.NewValue0(v.Pos, OpOffPtr, dst.Type)
		v0.AuxInt = int64ToAuxInt(s % 16)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpOffPtr, src.Type)
		v1.AuxInt = int64ToAuxInt(s % 16)
		v1.AddArg(src)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVOstore, types.TypeMem)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVOload, types.TypeInt128)
		v3.AddArg2(src, mem)
		v2.AddArg3(dst, v3, mem)
		v.AddArg3(v0, v1, v2)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 16 && s%16 != 0 && s%16 > 8 && !config.useSSE
	// result: (Move [s-s%16] (OffPtr <dst.Type> dst [s%16]) (OffPtr <src.Type> src [s%16]) (MOVQstore [8] dst (MOVQload [8] src mem) (MOVQstore dst (MOVQload src mem) mem)))
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 16 && s%16 != 0 && s%16 > 8 && !config.useSSE) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(s - s%16)
		v0 := b.NewValue0(v.Pos, OpOffPtr, dst.Type)
		v0.AuxInt = int64ToAuxInt(s % 16)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpOffPtr, src.Type)
		v1.AuxInt = int64ToAuxInt(s % 16)
		v1.AddArg(src)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVQstore, types.TypeMem)
		v2.AuxInt = int32ToAuxInt(8)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v3.AuxInt = int32ToAuxInt(8)
		v3.AddArg2(src, mem)
		v4 := b.NewValue0(v.Pos, OpAMD64MOVQstore, types.TypeMem)
		v5 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v5.AddArg2(src, mem)
		v4.AddArg3(dst, v5, mem)
		v2.AddArg3(dst, v3, v4)
		v.AddArg3(v0, v1, v2)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 64 && s <= 16*64 && s%16 == 0 && !config.noDuffDevice && logLargeCopy(v, s)
	// result: (DUFFCOPY [s] dst src mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 64 && s <= 16*64 && s%16 == 0 && !config.noDuffDevice && logLargeCopy(v, s)) {
			break
		}
		v.reset(OpAMD64DUFFCOPY)
		v.AuxInt = int64ToAuxInt(s)
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: (s > 16*64 || config.noDuffDevice) && s%8 == 0 && logLargeCopy(v, s)
	// result: (REPMOVSQ dst src (MOVQconst [s/8]) mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !((s > 16*64 || config.noDuffDevice) && s%8 == 0 && logLargeCopy(v, s)) {
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
func rewriteValueAMD64_OpPanicBounds(v *Value) bool {
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
		v.reset(OpAMD64LoweredPanicBoundsA)
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
		v.reset(OpAMD64LoweredPanicBoundsB)
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
		v.reset(OpAMD64LoweredPanicBoundsC)
		v.AuxInt = int64ToAuxInt(kind)
		v.AddArg3(x, y, mem)
		return true
	}
	return false
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
		v.reset(OpAMD64MOVSDstore)
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
		v.reset(OpAMD64MOVSSstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8
	// result: (MOVQstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8) {
			break
		}
		v.reset(OpAMD64MOVQstore)
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
func rewriteValueAMD64_OpZero(v *Value) bool {
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
	// match: (Zero [s] destptr mem)
	// cond: s%8 != 0 && s > 8 && !config.useSSE
	// result: (Zero [s-s%8] (OffPtr <destptr.Type> destptr [s%8]) (MOVQstoreconst [makeValAndOff(0,0)] destptr mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		if !(s%8 != 0 && s > 8 && !config.useSSE) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = int64ToAuxInt(s - s%8)
		v0 := b.NewValue0(v.Pos, OpOffPtr, destptr.Type)
		v0.AuxInt = int64ToAuxInt(s % 8)
		v0.AddArg(destptr)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v1.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v1.AddArg2(destptr, mem)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Zero [16] destptr mem)
	// cond: !config.useSSE
	// result: (MOVQstoreconst [makeValAndOff(0,8)] destptr (MOVQstoreconst [makeValAndOff(0,0)] destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 16 {
			break
		}
		destptr := v_0
		mem := v_1
		if !(!config.useSSE) {
			break
		}
		v.reset(OpAMD64MOVQstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 8))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [24] destptr mem)
	// cond: !config.useSSE
	// result: (MOVQstoreconst [makeValAndOff(0,16)] destptr (MOVQstoreconst [makeValAndOff(0,8)] destptr (MOVQstoreconst [makeValAndOff(0,0)] destptr mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 24 {
			break
		}
		destptr := v_0
		mem := v_1
		if !(!config.useSSE) {
			break
		}
		v.reset(OpAMD64MOVQstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 16))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 8))
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v1.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v1.AddArg2(destptr, mem)
		v0.AddArg2(destptr, v1)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [32] destptr mem)
	// cond: !config.useSSE
	// result: (MOVQstoreconst [makeValAndOff(0,24)] destptr (MOVQstoreconst [makeValAndOff(0,16)] destptr (MOVQstoreconst [makeValAndOff(0,8)] destptr (MOVQstoreconst [makeValAndOff(0,0)] destptr mem))))
	for {
		if auxIntToInt64(v.AuxInt) != 32 {
			break
		}
		destptr := v_0
		mem := v_1
		if !(!config.useSSE) {
			break
		}
		v.reset(OpAMD64MOVQstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 24))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 16))
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v1.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 8))
		v2 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v2.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v2.AddArg2(destptr, mem)
		v1.AddArg2(destptr, v2)
		v0.AddArg2(destptr, v1)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s > 8 && s < 16 && config.useSSE
	// result: (MOVQstoreconst [makeValAndOff(0,int32(s-8))] destptr (MOVQstoreconst [makeValAndOff(0,0)] destptr mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		if !(s > 8 && s < 16 && config.useSSE) {
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
	// cond: s%16 != 0 && s > 16 && s%16 > 8 && config.useSSE
	// result: (Zero [s-s%16] (OffPtr <destptr.Type> destptr [s%16]) (MOVOstoreconst [makeValAndOff(0,0)] destptr mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		if !(s%16 != 0 && s > 16 && s%16 > 8 && config.useSSE) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = int64ToAuxInt(s - s%16)
		v0 := b.NewValue0(v.Pos, OpOffPtr, destptr.Type)
		v0.AuxInt = int64ToAuxInt(s % 16)
		v0.AddArg(destptr)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVOstoreconst, types.TypeMem)
		v1.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v1.AddArg2(destptr, mem)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s%16 != 0 && s > 16 && s%16 <= 8 && config.useSSE
	// result: (Zero [s-s%16] (OffPtr <destptr.Type> destptr [s%16]) (MOVOstoreconst [makeValAndOff(0,0)] destptr mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		if !(s%16 != 0 && s > 16 && s%16 <= 8 && config.useSSE) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = int64ToAuxInt(s - s%16)
		v0 := b.NewValue0(v.Pos, OpOffPtr, destptr.Type)
		v0.AuxInt = int64ToAuxInt(s % 16)
		v0.AddArg(destptr)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVOstoreconst, types.TypeMem)
		v1.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v1.AddArg2(destptr, mem)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Zero [16] destptr mem)
	// cond: config.useSSE
	// result: (MOVOstoreconst [makeValAndOff(0,0)] destptr mem)
	for {
		if auxIntToInt64(v.AuxInt) != 16 {
			break
		}
		destptr := v_0
		mem := v_1
		if !(config.useSSE) {
			break
		}
		v.reset(OpAMD64MOVOstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [32] destptr mem)
	// cond: config.useSSE
	// result: (MOVOstoreconst [makeValAndOff(0,16)] destptr (MOVOstoreconst [makeValAndOff(0,0)] destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 32 {
			break
		}
		destptr := v_0
		mem := v_1
		if !(config.useSSE) {
			break
		}
		v.reset(OpAMD64MOVOstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 16))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVOstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [48] destptr mem)
	// cond: config.useSSE
	// result: (MOVOstoreconst [makeValAndOff(0,32)] destptr (MOVOstoreconst [makeValAndOff(0,16)] destptr (MOVOstoreconst [makeValAndOff(0,0)] destptr mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 48 {
			break
		}
		destptr := v_0
		mem := v_1
		if !(config.useSSE) {
			break
		}
		v.reset(OpAMD64MOVOstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 32))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVOstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 16))
		v1 := b.NewValue0(v.Pos, OpAMD64MOVOstoreconst, types.TypeMem)
		v1.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v1.AddArg2(destptr, mem)
		v0.AddArg2(destptr, v1)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [64] destptr mem)
	// cond: config.useSSE
	// result: (MOVOstoreconst [makeValAndOff(0,48)] destptr (MOVOstoreconst [makeValAndOff(0,32)] destptr (MOVOstoreconst [makeValAndOff(0,16)] destptr (MOVOstoreconst [makeValAndOff(0,0)] destptr mem))))
	for {
		if auxIntToInt64(v.AuxInt) != 64 {
			break
		}
		destptr := v_0
		mem := v_1
		if !(config.useSSE) {
			break
		}
		v.reset(OpAMD64MOVOstoreconst)
		v.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 48))
		v0 := b.NewValue0(v.Pos, OpAMD64MOVOstoreconst, types.TypeMem)
		v0.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 32))
		v1 := b.NewValue0(v.Pos, OpAMD64MOVOstoreconst, types.TypeMem)
		v1.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 16))
		v2 := b.NewValue0(v.Pos, OpAMD64MOVOstoreconst, types.TypeMem)
		v2.AuxInt = valAndOffToAuxInt(makeValAndOff(0, 0))
		v2.AddArg2(destptr, mem)
		v1.AddArg2(destptr, v2)
		v0.AddArg2(destptr, v1)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s > 64 && s <= 1024 && s%16 == 0 && !config.noDuffDevice
	// result: (DUFFZERO [s] destptr mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		if !(s > 64 && s <= 1024 && s%16 == 0 && !config.noDuffDevice) {
			break
		}
		v.reset(OpAMD64DUFFZERO)
		v.AuxInt = int64ToAuxInt(s)
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: (s > 1024 || (config.noDuffDevice && s > 64 || !config.useSSE && s > 32)) && s%8 == 0
	// result: (REPSTOSQ destptr (MOVQconst [s/8]) (MOVQconst [0]) mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		if !((s > 1024 || (config.noDuffDevice && s > 64 || !config.useSSE && s > 32)) && s%8 == 0) {
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
		// match: (EQ (TESTL (SHLXL (MOVLconst [1]) x) y))
		// result: (UGE (BTL x y))
		for b.Controls[0].Op == OpAMD64TESTL {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				if v_0_0.Op != OpAMD64SHLXL {
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
		// match: (EQ (TESTQ (SHLXQ (MOVQconst [1]) x) y))
		// result: (UGE (BTQ x y))
		for b.Controls[0].Op == OpAMD64TESTQ {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				if v_0_0.Op != OpAMD64SHLXQ {
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
		// cond: isUint32PowerOfTwo(int64(c))
		// result: (UGE (BTLconst [int8(log32(c))] x))
		for b.Controls[0].Op == OpAMD64TESTLconst {
			v_0 := b.Controls[0]
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if !(isUint32PowerOfTwo(int64(c))) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpAMD64BTLconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(int8(log32(c)))
			v0.AddArg(x)
			b.resetWithControl(BlockAMD64UGE, v0)
			return true
		}
		// match: (EQ (TESTQconst [c] x))
		// cond: isUint64PowerOfTwo(int64(c))
		// result: (UGE (BTQconst [int8(log32(c))] x))
		for b.Controls[0].Op == OpAMD64TESTQconst {
			v_0 := b.Controls[0]
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if !(isUint64PowerOfTwo(int64(c))) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(int8(log32(c)))
			v0.AddArg(x)
			b.resetWithControl(BlockAMD64UGE, v0)
			return true
		}
		// match: (EQ (TESTQ (MOVQconst [c]) x))
		// cond: isUint64PowerOfTwo(c)
		// result: (UGE (BTQconst [int8(log64(c))] x))
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
				if !(isUint64PowerOfTwo(c)) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTQconst, types.TypeFlags)
				v0.AuxInt = int8ToAuxInt(int8(log64(c)))
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
	case BlockAMD64GE:
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
		// match: (NE (TESTL (SHLXL (MOVLconst [1]) x) y))
		// result: (ULT (BTL x y))
		for b.Controls[0].Op == OpAMD64TESTL {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				if v_0_0.Op != OpAMD64SHLXL {
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
		// match: (NE (TESTQ (SHLXQ (MOVQconst [1]) x) y))
		// result: (ULT (BTQ x y))
		for b.Controls[0].Op == OpAMD64TESTQ {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				if v_0_0.Op != OpAMD64SHLXQ {
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
		// cond: isUint32PowerOfTwo(int64(c))
		// result: (ULT (BTLconst [int8(log32(c))] x))
		for b.Controls[0].Op == OpAMD64TESTLconst {
			v_0 := b.Controls[0]
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if !(isUint32PowerOfTwo(int64(c))) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpAMD64BTLconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(int8(log32(c)))
			v0.AddArg(x)
			b.resetWithControl(BlockAMD64ULT, v0)
			return true
		}
		// match: (NE (TESTQconst [c] x))
		// cond: isUint64PowerOfTwo(int64(c))
		// result: (ULT (BTQconst [int8(log32(c))] x))
		for b.Controls[0].Op == OpAMD64TESTQconst {
			v_0 := b.Controls[0]
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if !(isUint64PowerOfTwo(int64(c))) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = int8ToAuxInt(int8(log32(c)))
			v0.AddArg(x)
			b.resetWithControl(BlockAMD64ULT, v0)
			return true
		}
		// match: (NE (TESTQ (MOVQconst [c]) x))
		// cond: isUint64PowerOfTwo(c)
		// result: (ULT (BTQconst [int8(log64(c))] x))
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
				if !(isUint64PowerOfTwo(c)) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpAMD64BTQconst, types.TypeFlags)
				v0.AuxInt = int8ToAuxInt(int8(log64(c)))
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
