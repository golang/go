// Code generated from gen/*Ops.go; DO NOT EDIT.

package ssa

import (
	"cmd/internal/obj"
	"cmd/internal/obj/arm"
	"cmd/internal/obj/arm64"
	"cmd/internal/obj/mips"
	"cmd/internal/obj/ppc64"
	"cmd/internal/obj/s390x"
	"cmd/internal/obj/x86"
)

const (
	BlockInvalid BlockKind = iota

	Block386EQ
	Block386NE
	Block386LT
	Block386LE
	Block386GT
	Block386GE
	Block386ULT
	Block386ULE
	Block386UGT
	Block386UGE
	Block386EQF
	Block386NEF
	Block386ORD
	Block386NAN

	BlockAMD64EQ
	BlockAMD64NE
	BlockAMD64LT
	BlockAMD64LE
	BlockAMD64GT
	BlockAMD64GE
	BlockAMD64ULT
	BlockAMD64ULE
	BlockAMD64UGT
	BlockAMD64UGE
	BlockAMD64EQF
	BlockAMD64NEF
	BlockAMD64ORD
	BlockAMD64NAN

	BlockARMEQ
	BlockARMNE
	BlockARMLT
	BlockARMLE
	BlockARMGT
	BlockARMGE
	BlockARMULT
	BlockARMULE
	BlockARMUGT
	BlockARMUGE

	BlockARM64EQ
	BlockARM64NE
	BlockARM64LT
	BlockARM64LE
	BlockARM64GT
	BlockARM64GE
	BlockARM64ULT
	BlockARM64ULE
	BlockARM64UGT
	BlockARM64UGE
	BlockARM64Z
	BlockARM64NZ
	BlockARM64ZW
	BlockARM64NZW

	BlockMIPSEQ
	BlockMIPSNE
	BlockMIPSLTZ
	BlockMIPSLEZ
	BlockMIPSGTZ
	BlockMIPSGEZ
	BlockMIPSFPT
	BlockMIPSFPF

	BlockMIPS64EQ
	BlockMIPS64NE
	BlockMIPS64LTZ
	BlockMIPS64LEZ
	BlockMIPS64GTZ
	BlockMIPS64GEZ
	BlockMIPS64FPT
	BlockMIPS64FPF

	BlockPPC64EQ
	BlockPPC64NE
	BlockPPC64LT
	BlockPPC64LE
	BlockPPC64GT
	BlockPPC64GE
	BlockPPC64FLT
	BlockPPC64FLE
	BlockPPC64FGT
	BlockPPC64FGE

	BlockS390XEQ
	BlockS390XNE
	BlockS390XLT
	BlockS390XLE
	BlockS390XGT
	BlockS390XGE
	BlockS390XGTF
	BlockS390XGEF

	BlockPlain
	BlockIf
	BlockDefer
	BlockRet
	BlockRetJmp
	BlockExit
	BlockFirst
)

var blockString = [...]string{
	BlockInvalid: "BlockInvalid",

	Block386EQ:  "EQ",
	Block386NE:  "NE",
	Block386LT:  "LT",
	Block386LE:  "LE",
	Block386GT:  "GT",
	Block386GE:  "GE",
	Block386ULT: "ULT",
	Block386ULE: "ULE",
	Block386UGT: "UGT",
	Block386UGE: "UGE",
	Block386EQF: "EQF",
	Block386NEF: "NEF",
	Block386ORD: "ORD",
	Block386NAN: "NAN",

	BlockAMD64EQ:  "EQ",
	BlockAMD64NE:  "NE",
	BlockAMD64LT:  "LT",
	BlockAMD64LE:  "LE",
	BlockAMD64GT:  "GT",
	BlockAMD64GE:  "GE",
	BlockAMD64ULT: "ULT",
	BlockAMD64ULE: "ULE",
	BlockAMD64UGT: "UGT",
	BlockAMD64UGE: "UGE",
	BlockAMD64EQF: "EQF",
	BlockAMD64NEF: "NEF",
	BlockAMD64ORD: "ORD",
	BlockAMD64NAN: "NAN",

	BlockARMEQ:  "EQ",
	BlockARMNE:  "NE",
	BlockARMLT:  "LT",
	BlockARMLE:  "LE",
	BlockARMGT:  "GT",
	BlockARMGE:  "GE",
	BlockARMULT: "ULT",
	BlockARMULE: "ULE",
	BlockARMUGT: "UGT",
	BlockARMUGE: "UGE",

	BlockARM64EQ:  "EQ",
	BlockARM64NE:  "NE",
	BlockARM64LT:  "LT",
	BlockARM64LE:  "LE",
	BlockARM64GT:  "GT",
	BlockARM64GE:  "GE",
	BlockARM64ULT: "ULT",
	BlockARM64ULE: "ULE",
	BlockARM64UGT: "UGT",
	BlockARM64UGE: "UGE",
	BlockARM64Z:   "Z",
	BlockARM64NZ:  "NZ",
	BlockARM64ZW:  "ZW",
	BlockARM64NZW: "NZW",

	BlockMIPSEQ:  "EQ",
	BlockMIPSNE:  "NE",
	BlockMIPSLTZ: "LTZ",
	BlockMIPSLEZ: "LEZ",
	BlockMIPSGTZ: "GTZ",
	BlockMIPSGEZ: "GEZ",
	BlockMIPSFPT: "FPT",
	BlockMIPSFPF: "FPF",

	BlockMIPS64EQ:  "EQ",
	BlockMIPS64NE:  "NE",
	BlockMIPS64LTZ: "LTZ",
	BlockMIPS64LEZ: "LEZ",
	BlockMIPS64GTZ: "GTZ",
	BlockMIPS64GEZ: "GEZ",
	BlockMIPS64FPT: "FPT",
	BlockMIPS64FPF: "FPF",

	BlockPPC64EQ:  "EQ",
	BlockPPC64NE:  "NE",
	BlockPPC64LT:  "LT",
	BlockPPC64LE:  "LE",
	BlockPPC64GT:  "GT",
	BlockPPC64GE:  "GE",
	BlockPPC64FLT: "FLT",
	BlockPPC64FLE: "FLE",
	BlockPPC64FGT: "FGT",
	BlockPPC64FGE: "FGE",

	BlockS390XEQ:  "EQ",
	BlockS390XNE:  "NE",
	BlockS390XLT:  "LT",
	BlockS390XLE:  "LE",
	BlockS390XGT:  "GT",
	BlockS390XGE:  "GE",
	BlockS390XGTF: "GTF",
	BlockS390XGEF: "GEF",

	BlockPlain:  "Plain",
	BlockIf:     "If",
	BlockDefer:  "Defer",
	BlockRet:    "Ret",
	BlockRetJmp: "RetJmp",
	BlockExit:   "Exit",
	BlockFirst:  "First",
}

func (k BlockKind) String() string { return blockString[k] }

const (
	OpInvalid Op = iota

	Op386ADDSS
	Op386ADDSD
	Op386SUBSS
	Op386SUBSD
	Op386MULSS
	Op386MULSD
	Op386DIVSS
	Op386DIVSD
	Op386MOVSSload
	Op386MOVSDload
	Op386MOVSSconst
	Op386MOVSDconst
	Op386MOVSSloadidx1
	Op386MOVSSloadidx4
	Op386MOVSDloadidx1
	Op386MOVSDloadidx8
	Op386MOVSSstore
	Op386MOVSDstore
	Op386MOVSSstoreidx1
	Op386MOVSSstoreidx4
	Op386MOVSDstoreidx1
	Op386MOVSDstoreidx8
	Op386ADDL
	Op386ADDLconst
	Op386ADDLcarry
	Op386ADDLconstcarry
	Op386ADCL
	Op386ADCLconst
	Op386SUBL
	Op386SUBLconst
	Op386SUBLcarry
	Op386SUBLconstcarry
	Op386SBBL
	Op386SBBLconst
	Op386MULL
	Op386MULLconst
	Op386HMULL
	Op386HMULLU
	Op386MULLQU
	Op386AVGLU
	Op386DIVL
	Op386DIVW
	Op386DIVLU
	Op386DIVWU
	Op386MODL
	Op386MODW
	Op386MODLU
	Op386MODWU
	Op386ANDL
	Op386ANDLconst
	Op386ORL
	Op386ORLconst
	Op386XORL
	Op386XORLconst
	Op386CMPL
	Op386CMPW
	Op386CMPB
	Op386CMPLconst
	Op386CMPWconst
	Op386CMPBconst
	Op386UCOMISS
	Op386UCOMISD
	Op386TESTL
	Op386TESTW
	Op386TESTB
	Op386TESTLconst
	Op386TESTWconst
	Op386TESTBconst
	Op386SHLL
	Op386SHLLconst
	Op386SHRL
	Op386SHRW
	Op386SHRB
	Op386SHRLconst
	Op386SHRWconst
	Op386SHRBconst
	Op386SARL
	Op386SARW
	Op386SARB
	Op386SARLconst
	Op386SARWconst
	Op386SARBconst
	Op386ROLLconst
	Op386ROLWconst
	Op386ROLBconst
	Op386NEGL
	Op386NOTL
	Op386BSFL
	Op386BSFW
	Op386BSRL
	Op386BSRW
	Op386BSWAPL
	Op386SQRTSD
	Op386SBBLcarrymask
	Op386SETEQ
	Op386SETNE
	Op386SETL
	Op386SETLE
	Op386SETG
	Op386SETGE
	Op386SETB
	Op386SETBE
	Op386SETA
	Op386SETAE
	Op386SETEQF
	Op386SETNEF
	Op386SETORD
	Op386SETNAN
	Op386SETGF
	Op386SETGEF
	Op386MOVBLSX
	Op386MOVBLZX
	Op386MOVWLSX
	Op386MOVWLZX
	Op386MOVLconst
	Op386CVTTSD2SL
	Op386CVTTSS2SL
	Op386CVTSL2SS
	Op386CVTSL2SD
	Op386CVTSD2SS
	Op386CVTSS2SD
	Op386PXOR
	Op386LEAL
	Op386LEAL1
	Op386LEAL2
	Op386LEAL4
	Op386LEAL8
	Op386MOVBload
	Op386MOVBLSXload
	Op386MOVWload
	Op386MOVWLSXload
	Op386MOVLload
	Op386MOVBstore
	Op386MOVWstore
	Op386MOVLstore
	Op386MOVBloadidx1
	Op386MOVWloadidx1
	Op386MOVWloadidx2
	Op386MOVLloadidx1
	Op386MOVLloadidx4
	Op386MOVBstoreidx1
	Op386MOVWstoreidx1
	Op386MOVWstoreidx2
	Op386MOVLstoreidx1
	Op386MOVLstoreidx4
	Op386MOVBstoreconst
	Op386MOVWstoreconst
	Op386MOVLstoreconst
	Op386MOVBstoreconstidx1
	Op386MOVWstoreconstidx1
	Op386MOVWstoreconstidx2
	Op386MOVLstoreconstidx1
	Op386MOVLstoreconstidx4
	Op386DUFFZERO
	Op386REPSTOSL
	Op386CALLstatic
	Op386CALLclosure
	Op386CALLinter
	Op386DUFFCOPY
	Op386REPMOVSL
	Op386InvertFlags
	Op386LoweredGetG
	Op386LoweredGetClosurePtr
	Op386LoweredNilCheck
	Op386MOVLconvert
	Op386FlagEQ
	Op386FlagLT_ULT
	Op386FlagLT_UGT
	Op386FlagGT_UGT
	Op386FlagGT_ULT
	Op386FCHS
	Op386MOVSSconst1
	Op386MOVSDconst1
	Op386MOVSSconst2
	Op386MOVSDconst2

	OpAMD64ADDSS
	OpAMD64ADDSD
	OpAMD64SUBSS
	OpAMD64SUBSD
	OpAMD64MULSS
	OpAMD64MULSD
	OpAMD64DIVSS
	OpAMD64DIVSD
	OpAMD64MOVSSload
	OpAMD64MOVSDload
	OpAMD64MOVSSconst
	OpAMD64MOVSDconst
	OpAMD64MOVSSloadidx1
	OpAMD64MOVSSloadidx4
	OpAMD64MOVSDloadidx1
	OpAMD64MOVSDloadidx8
	OpAMD64MOVSSstore
	OpAMD64MOVSDstore
	OpAMD64MOVSSstoreidx1
	OpAMD64MOVSSstoreidx4
	OpAMD64MOVSDstoreidx1
	OpAMD64MOVSDstoreidx8
	OpAMD64ADDSDmem
	OpAMD64ADDSSmem
	OpAMD64SUBSSmem
	OpAMD64SUBSDmem
	OpAMD64MULSSmem
	OpAMD64MULSDmem
	OpAMD64ADDQ
	OpAMD64ADDL
	OpAMD64ADDQconst
	OpAMD64ADDLconst
	OpAMD64SUBQ
	OpAMD64SUBL
	OpAMD64SUBQconst
	OpAMD64SUBLconst
	OpAMD64MULQ
	OpAMD64MULL
	OpAMD64MULQconst
	OpAMD64MULLconst
	OpAMD64HMULQ
	OpAMD64HMULL
	OpAMD64HMULQU
	OpAMD64HMULLU
	OpAMD64AVGQU
	OpAMD64DIVQ
	OpAMD64DIVL
	OpAMD64DIVW
	OpAMD64DIVQU
	OpAMD64DIVLU
	OpAMD64DIVWU
	OpAMD64MULQU2
	OpAMD64DIVQU2
	OpAMD64ANDQ
	OpAMD64ANDL
	OpAMD64ANDQconst
	OpAMD64ANDLconst
	OpAMD64ORQ
	OpAMD64ORL
	OpAMD64ORQconst
	OpAMD64ORLconst
	OpAMD64XORQ
	OpAMD64XORL
	OpAMD64XORQconst
	OpAMD64XORLconst
	OpAMD64CMPQ
	OpAMD64CMPL
	OpAMD64CMPW
	OpAMD64CMPB
	OpAMD64CMPQconst
	OpAMD64CMPLconst
	OpAMD64CMPWconst
	OpAMD64CMPBconst
	OpAMD64UCOMISS
	OpAMD64UCOMISD
	OpAMD64BTL
	OpAMD64BTQ
	OpAMD64BTLconst
	OpAMD64BTQconst
	OpAMD64TESTQ
	OpAMD64TESTL
	OpAMD64TESTW
	OpAMD64TESTB
	OpAMD64TESTQconst
	OpAMD64TESTLconst
	OpAMD64TESTWconst
	OpAMD64TESTBconst
	OpAMD64SHLQ
	OpAMD64SHLL
	OpAMD64SHLQconst
	OpAMD64SHLLconst
	OpAMD64SHRQ
	OpAMD64SHRL
	OpAMD64SHRW
	OpAMD64SHRB
	OpAMD64SHRQconst
	OpAMD64SHRLconst
	OpAMD64SHRWconst
	OpAMD64SHRBconst
	OpAMD64SARQ
	OpAMD64SARL
	OpAMD64SARW
	OpAMD64SARB
	OpAMD64SARQconst
	OpAMD64SARLconst
	OpAMD64SARWconst
	OpAMD64SARBconst
	OpAMD64ROLQ
	OpAMD64ROLL
	OpAMD64ROLW
	OpAMD64ROLB
	OpAMD64RORQ
	OpAMD64RORL
	OpAMD64RORW
	OpAMD64RORB
	OpAMD64ROLQconst
	OpAMD64ROLLconst
	OpAMD64ROLWconst
	OpAMD64ROLBconst
	OpAMD64ADDLmem
	OpAMD64ADDQmem
	OpAMD64SUBQmem
	OpAMD64SUBLmem
	OpAMD64ANDLmem
	OpAMD64ANDQmem
	OpAMD64ORQmem
	OpAMD64ORLmem
	OpAMD64XORQmem
	OpAMD64XORLmem
	OpAMD64NEGQ
	OpAMD64NEGL
	OpAMD64NOTQ
	OpAMD64NOTL
	OpAMD64BSFQ
	OpAMD64BSFL
	OpAMD64BSRQ
	OpAMD64BSRL
	OpAMD64CMOVQEQ
	OpAMD64CMOVLEQ
	OpAMD64BSWAPQ
	OpAMD64BSWAPL
	OpAMD64POPCNTQ
	OpAMD64POPCNTL
	OpAMD64SQRTSD
	OpAMD64SBBQcarrymask
	OpAMD64SBBLcarrymask
	OpAMD64SETEQ
	OpAMD64SETNE
	OpAMD64SETL
	OpAMD64SETLE
	OpAMD64SETG
	OpAMD64SETGE
	OpAMD64SETB
	OpAMD64SETBE
	OpAMD64SETA
	OpAMD64SETAE
	OpAMD64SETEQF
	OpAMD64SETNEF
	OpAMD64SETORD
	OpAMD64SETNAN
	OpAMD64SETGF
	OpAMD64SETGEF
	OpAMD64MOVBQSX
	OpAMD64MOVBQZX
	OpAMD64MOVWQSX
	OpAMD64MOVWQZX
	OpAMD64MOVLQSX
	OpAMD64MOVLQZX
	OpAMD64MOVLconst
	OpAMD64MOVQconst
	OpAMD64CVTTSD2SL
	OpAMD64CVTTSD2SQ
	OpAMD64CVTTSS2SL
	OpAMD64CVTTSS2SQ
	OpAMD64CVTSL2SS
	OpAMD64CVTSL2SD
	OpAMD64CVTSQ2SS
	OpAMD64CVTSQ2SD
	OpAMD64CVTSD2SS
	OpAMD64CVTSS2SD
	OpAMD64PXOR
	OpAMD64LEAQ
	OpAMD64LEAQ1
	OpAMD64LEAQ2
	OpAMD64LEAQ4
	OpAMD64LEAQ8
	OpAMD64LEAL
	OpAMD64MOVBload
	OpAMD64MOVBQSXload
	OpAMD64MOVWload
	OpAMD64MOVWQSXload
	OpAMD64MOVLload
	OpAMD64MOVLQSXload
	OpAMD64MOVQload
	OpAMD64MOVBstore
	OpAMD64MOVWstore
	OpAMD64MOVLstore
	OpAMD64MOVQstore
	OpAMD64MOVOload
	OpAMD64MOVOstore
	OpAMD64MOVBloadidx1
	OpAMD64MOVWloadidx1
	OpAMD64MOVWloadidx2
	OpAMD64MOVLloadidx1
	OpAMD64MOVLloadidx4
	OpAMD64MOVQloadidx1
	OpAMD64MOVQloadidx8
	OpAMD64MOVBstoreidx1
	OpAMD64MOVWstoreidx1
	OpAMD64MOVWstoreidx2
	OpAMD64MOVLstoreidx1
	OpAMD64MOVLstoreidx4
	OpAMD64MOVQstoreidx1
	OpAMD64MOVQstoreidx8
	OpAMD64MOVBstoreconst
	OpAMD64MOVWstoreconst
	OpAMD64MOVLstoreconst
	OpAMD64MOVQstoreconst
	OpAMD64MOVBstoreconstidx1
	OpAMD64MOVWstoreconstidx1
	OpAMD64MOVWstoreconstidx2
	OpAMD64MOVLstoreconstidx1
	OpAMD64MOVLstoreconstidx4
	OpAMD64MOVQstoreconstidx1
	OpAMD64MOVQstoreconstidx8
	OpAMD64DUFFZERO
	OpAMD64MOVOconst
	OpAMD64REPSTOSQ
	OpAMD64CALLstatic
	OpAMD64CALLclosure
	OpAMD64CALLinter
	OpAMD64DUFFCOPY
	OpAMD64REPMOVSQ
	OpAMD64InvertFlags
	OpAMD64LoweredGetG
	OpAMD64LoweredGetClosurePtr
	OpAMD64LoweredNilCheck
	OpAMD64MOVQconvert
	OpAMD64MOVLconvert
	OpAMD64FlagEQ
	OpAMD64FlagLT_ULT
	OpAMD64FlagLT_UGT
	OpAMD64FlagGT_UGT
	OpAMD64FlagGT_ULT
	OpAMD64MOVLatomicload
	OpAMD64MOVQatomicload
	OpAMD64XCHGL
	OpAMD64XCHGQ
	OpAMD64XADDLlock
	OpAMD64XADDQlock
	OpAMD64AddTupleFirst32
	OpAMD64AddTupleFirst64
	OpAMD64CMPXCHGLlock
	OpAMD64CMPXCHGQlock
	OpAMD64ANDBlock
	OpAMD64ORBlock

	OpARMADD
	OpARMADDconst
	OpARMSUB
	OpARMSUBconst
	OpARMRSB
	OpARMRSBconst
	OpARMMUL
	OpARMHMUL
	OpARMHMULU
	OpARMCALLudiv
	OpARMADDS
	OpARMADDSconst
	OpARMADC
	OpARMADCconst
	OpARMSUBS
	OpARMSUBSconst
	OpARMRSBSconst
	OpARMSBC
	OpARMSBCconst
	OpARMRSCconst
	OpARMMULLU
	OpARMMULA
	OpARMADDF
	OpARMADDD
	OpARMSUBF
	OpARMSUBD
	OpARMMULF
	OpARMMULD
	OpARMDIVF
	OpARMDIVD
	OpARMAND
	OpARMANDconst
	OpARMOR
	OpARMORconst
	OpARMXOR
	OpARMXORconst
	OpARMBIC
	OpARMBICconst
	OpARMMVN
	OpARMNEGF
	OpARMNEGD
	OpARMSQRTD
	OpARMCLZ
	OpARMREV
	OpARMRBIT
	OpARMSLL
	OpARMSLLconst
	OpARMSRL
	OpARMSRLconst
	OpARMSRA
	OpARMSRAconst
	OpARMSRRconst
	OpARMADDshiftLL
	OpARMADDshiftRL
	OpARMADDshiftRA
	OpARMSUBshiftLL
	OpARMSUBshiftRL
	OpARMSUBshiftRA
	OpARMRSBshiftLL
	OpARMRSBshiftRL
	OpARMRSBshiftRA
	OpARMANDshiftLL
	OpARMANDshiftRL
	OpARMANDshiftRA
	OpARMORshiftLL
	OpARMORshiftRL
	OpARMORshiftRA
	OpARMXORshiftLL
	OpARMXORshiftRL
	OpARMXORshiftRA
	OpARMXORshiftRR
	OpARMBICshiftLL
	OpARMBICshiftRL
	OpARMBICshiftRA
	OpARMMVNshiftLL
	OpARMMVNshiftRL
	OpARMMVNshiftRA
	OpARMADCshiftLL
	OpARMADCshiftRL
	OpARMADCshiftRA
	OpARMSBCshiftLL
	OpARMSBCshiftRL
	OpARMSBCshiftRA
	OpARMRSCshiftLL
	OpARMRSCshiftRL
	OpARMRSCshiftRA
	OpARMADDSshiftLL
	OpARMADDSshiftRL
	OpARMADDSshiftRA
	OpARMSUBSshiftLL
	OpARMSUBSshiftRL
	OpARMSUBSshiftRA
	OpARMRSBSshiftLL
	OpARMRSBSshiftRL
	OpARMRSBSshiftRA
	OpARMADDshiftLLreg
	OpARMADDshiftRLreg
	OpARMADDshiftRAreg
	OpARMSUBshiftLLreg
	OpARMSUBshiftRLreg
	OpARMSUBshiftRAreg
	OpARMRSBshiftLLreg
	OpARMRSBshiftRLreg
	OpARMRSBshiftRAreg
	OpARMANDshiftLLreg
	OpARMANDshiftRLreg
	OpARMANDshiftRAreg
	OpARMORshiftLLreg
	OpARMORshiftRLreg
	OpARMORshiftRAreg
	OpARMXORshiftLLreg
	OpARMXORshiftRLreg
	OpARMXORshiftRAreg
	OpARMBICshiftLLreg
	OpARMBICshiftRLreg
	OpARMBICshiftRAreg
	OpARMMVNshiftLLreg
	OpARMMVNshiftRLreg
	OpARMMVNshiftRAreg
	OpARMADCshiftLLreg
	OpARMADCshiftRLreg
	OpARMADCshiftRAreg
	OpARMSBCshiftLLreg
	OpARMSBCshiftRLreg
	OpARMSBCshiftRAreg
	OpARMRSCshiftLLreg
	OpARMRSCshiftRLreg
	OpARMRSCshiftRAreg
	OpARMADDSshiftLLreg
	OpARMADDSshiftRLreg
	OpARMADDSshiftRAreg
	OpARMSUBSshiftLLreg
	OpARMSUBSshiftRLreg
	OpARMSUBSshiftRAreg
	OpARMRSBSshiftLLreg
	OpARMRSBSshiftRLreg
	OpARMRSBSshiftRAreg
	OpARMCMP
	OpARMCMPconst
	OpARMCMN
	OpARMCMNconst
	OpARMTST
	OpARMTSTconst
	OpARMTEQ
	OpARMTEQconst
	OpARMCMPF
	OpARMCMPD
	OpARMCMPshiftLL
	OpARMCMPshiftRL
	OpARMCMPshiftRA
	OpARMCMPshiftLLreg
	OpARMCMPshiftRLreg
	OpARMCMPshiftRAreg
	OpARMCMPF0
	OpARMCMPD0
	OpARMMOVWconst
	OpARMMOVFconst
	OpARMMOVDconst
	OpARMMOVWaddr
	OpARMMOVBload
	OpARMMOVBUload
	OpARMMOVHload
	OpARMMOVHUload
	OpARMMOVWload
	OpARMMOVFload
	OpARMMOVDload
	OpARMMOVBstore
	OpARMMOVHstore
	OpARMMOVWstore
	OpARMMOVFstore
	OpARMMOVDstore
	OpARMMOVWloadidx
	OpARMMOVWloadshiftLL
	OpARMMOVWloadshiftRL
	OpARMMOVWloadshiftRA
	OpARMMOVWstoreidx
	OpARMMOVWstoreshiftLL
	OpARMMOVWstoreshiftRL
	OpARMMOVWstoreshiftRA
	OpARMMOVBreg
	OpARMMOVBUreg
	OpARMMOVHreg
	OpARMMOVHUreg
	OpARMMOVWreg
	OpARMMOVWnop
	OpARMMOVWF
	OpARMMOVWD
	OpARMMOVWUF
	OpARMMOVWUD
	OpARMMOVFW
	OpARMMOVDW
	OpARMMOVFWU
	OpARMMOVDWU
	OpARMMOVFD
	OpARMMOVDF
	OpARMCMOVWHSconst
	OpARMCMOVWLSconst
	OpARMSRAcond
	OpARMCALLstatic
	OpARMCALLclosure
	OpARMCALLinter
	OpARMLoweredNilCheck
	OpARMEqual
	OpARMNotEqual
	OpARMLessThan
	OpARMLessEqual
	OpARMGreaterThan
	OpARMGreaterEqual
	OpARMLessThanU
	OpARMLessEqualU
	OpARMGreaterThanU
	OpARMGreaterEqualU
	OpARMDUFFZERO
	OpARMDUFFCOPY
	OpARMLoweredZero
	OpARMLoweredMove
	OpARMLoweredGetClosurePtr
	OpARMMOVWconvert
	OpARMFlagEQ
	OpARMFlagLT_ULT
	OpARMFlagLT_UGT
	OpARMFlagGT_UGT
	OpARMFlagGT_ULT
	OpARMInvertFlags

	OpARM64ADD
	OpARM64ADDconst
	OpARM64SUB
	OpARM64SUBconst
	OpARM64MUL
	OpARM64MULW
	OpARM64MULH
	OpARM64UMULH
	OpARM64MULL
	OpARM64UMULL
	OpARM64DIV
	OpARM64UDIV
	OpARM64DIVW
	OpARM64UDIVW
	OpARM64MOD
	OpARM64UMOD
	OpARM64MODW
	OpARM64UMODW
	OpARM64FADDS
	OpARM64FADDD
	OpARM64FSUBS
	OpARM64FSUBD
	OpARM64FMULS
	OpARM64FMULD
	OpARM64FDIVS
	OpARM64FDIVD
	OpARM64AND
	OpARM64ANDconst
	OpARM64OR
	OpARM64ORconst
	OpARM64XOR
	OpARM64XORconst
	OpARM64BIC
	OpARM64BICconst
	OpARM64MVN
	OpARM64NEG
	OpARM64FNEGS
	OpARM64FNEGD
	OpARM64FSQRTD
	OpARM64REV
	OpARM64REVW
	OpARM64REV16W
	OpARM64RBIT
	OpARM64RBITW
	OpARM64CLZ
	OpARM64CLZW
	OpARM64SLL
	OpARM64SLLconst
	OpARM64SRL
	OpARM64SRLconst
	OpARM64SRA
	OpARM64SRAconst
	OpARM64RORconst
	OpARM64RORWconst
	OpARM64CMP
	OpARM64CMPconst
	OpARM64CMPW
	OpARM64CMPWconst
	OpARM64CMN
	OpARM64CMNconst
	OpARM64CMNW
	OpARM64CMNWconst
	OpARM64FCMPS
	OpARM64FCMPD
	OpARM64ADDshiftLL
	OpARM64ADDshiftRL
	OpARM64ADDshiftRA
	OpARM64SUBshiftLL
	OpARM64SUBshiftRL
	OpARM64SUBshiftRA
	OpARM64ANDshiftLL
	OpARM64ANDshiftRL
	OpARM64ANDshiftRA
	OpARM64ORshiftLL
	OpARM64ORshiftRL
	OpARM64ORshiftRA
	OpARM64XORshiftLL
	OpARM64XORshiftRL
	OpARM64XORshiftRA
	OpARM64BICshiftLL
	OpARM64BICshiftRL
	OpARM64BICshiftRA
	OpARM64CMPshiftLL
	OpARM64CMPshiftRL
	OpARM64CMPshiftRA
	OpARM64MOVDconst
	OpARM64FMOVSconst
	OpARM64FMOVDconst
	OpARM64MOVDaddr
	OpARM64MOVBload
	OpARM64MOVBUload
	OpARM64MOVHload
	OpARM64MOVHUload
	OpARM64MOVWload
	OpARM64MOVWUload
	OpARM64MOVDload
	OpARM64FMOVSload
	OpARM64FMOVDload
	OpARM64MOVBstore
	OpARM64MOVHstore
	OpARM64MOVWstore
	OpARM64MOVDstore
	OpARM64FMOVSstore
	OpARM64FMOVDstore
	OpARM64MOVBstorezero
	OpARM64MOVHstorezero
	OpARM64MOVWstorezero
	OpARM64MOVDstorezero
	OpARM64MOVBreg
	OpARM64MOVBUreg
	OpARM64MOVHreg
	OpARM64MOVHUreg
	OpARM64MOVWreg
	OpARM64MOVWUreg
	OpARM64MOVDreg
	OpARM64MOVDnop
	OpARM64SCVTFWS
	OpARM64SCVTFWD
	OpARM64UCVTFWS
	OpARM64UCVTFWD
	OpARM64SCVTFS
	OpARM64SCVTFD
	OpARM64UCVTFS
	OpARM64UCVTFD
	OpARM64FCVTZSSW
	OpARM64FCVTZSDW
	OpARM64FCVTZUSW
	OpARM64FCVTZUDW
	OpARM64FCVTZSS
	OpARM64FCVTZSD
	OpARM64FCVTZUS
	OpARM64FCVTZUD
	OpARM64FCVTSD
	OpARM64FCVTDS
	OpARM64CSELULT
	OpARM64CSELULT0
	OpARM64CALLstatic
	OpARM64CALLclosure
	OpARM64CALLinter
	OpARM64LoweredNilCheck
	OpARM64Equal
	OpARM64NotEqual
	OpARM64LessThan
	OpARM64LessEqual
	OpARM64GreaterThan
	OpARM64GreaterEqual
	OpARM64LessThanU
	OpARM64LessEqualU
	OpARM64GreaterThanU
	OpARM64GreaterEqualU
	OpARM64DUFFZERO
	OpARM64LoweredZero
	OpARM64DUFFCOPY
	OpARM64LoweredMove
	OpARM64LoweredGetClosurePtr
	OpARM64MOVDconvert
	OpARM64FlagEQ
	OpARM64FlagLT_ULT
	OpARM64FlagLT_UGT
	OpARM64FlagGT_UGT
	OpARM64FlagGT_ULT
	OpARM64InvertFlags
	OpARM64LDAR
	OpARM64LDARW
	OpARM64STLR
	OpARM64STLRW
	OpARM64LoweredAtomicExchange64
	OpARM64LoweredAtomicExchange32
	OpARM64LoweredAtomicAdd64
	OpARM64LoweredAtomicAdd32
	OpARM64LoweredAtomicCas64
	OpARM64LoweredAtomicCas32
	OpARM64LoweredAtomicAnd8
	OpARM64LoweredAtomicOr8

	OpMIPSADD
	OpMIPSADDconst
	OpMIPSSUB
	OpMIPSSUBconst
	OpMIPSMUL
	OpMIPSMULT
	OpMIPSMULTU
	OpMIPSDIV
	OpMIPSDIVU
	OpMIPSADDF
	OpMIPSADDD
	OpMIPSSUBF
	OpMIPSSUBD
	OpMIPSMULF
	OpMIPSMULD
	OpMIPSDIVF
	OpMIPSDIVD
	OpMIPSAND
	OpMIPSANDconst
	OpMIPSOR
	OpMIPSORconst
	OpMIPSXOR
	OpMIPSXORconst
	OpMIPSNOR
	OpMIPSNORconst
	OpMIPSNEG
	OpMIPSNEGF
	OpMIPSNEGD
	OpMIPSSQRTD
	OpMIPSSLL
	OpMIPSSLLconst
	OpMIPSSRL
	OpMIPSSRLconst
	OpMIPSSRA
	OpMIPSSRAconst
	OpMIPSCLZ
	OpMIPSSGT
	OpMIPSSGTconst
	OpMIPSSGTzero
	OpMIPSSGTU
	OpMIPSSGTUconst
	OpMIPSSGTUzero
	OpMIPSCMPEQF
	OpMIPSCMPEQD
	OpMIPSCMPGEF
	OpMIPSCMPGED
	OpMIPSCMPGTF
	OpMIPSCMPGTD
	OpMIPSMOVWconst
	OpMIPSMOVFconst
	OpMIPSMOVDconst
	OpMIPSMOVWaddr
	OpMIPSMOVBload
	OpMIPSMOVBUload
	OpMIPSMOVHload
	OpMIPSMOVHUload
	OpMIPSMOVWload
	OpMIPSMOVFload
	OpMIPSMOVDload
	OpMIPSMOVBstore
	OpMIPSMOVHstore
	OpMIPSMOVWstore
	OpMIPSMOVFstore
	OpMIPSMOVDstore
	OpMIPSMOVBstorezero
	OpMIPSMOVHstorezero
	OpMIPSMOVWstorezero
	OpMIPSMOVBreg
	OpMIPSMOVBUreg
	OpMIPSMOVHreg
	OpMIPSMOVHUreg
	OpMIPSMOVWreg
	OpMIPSMOVWnop
	OpMIPSCMOVZ
	OpMIPSCMOVZzero
	OpMIPSMOVWF
	OpMIPSMOVWD
	OpMIPSTRUNCFW
	OpMIPSTRUNCDW
	OpMIPSMOVFD
	OpMIPSMOVDF
	OpMIPSCALLstatic
	OpMIPSCALLclosure
	OpMIPSCALLinter
	OpMIPSLoweredAtomicLoad
	OpMIPSLoweredAtomicStore
	OpMIPSLoweredAtomicStorezero
	OpMIPSLoweredAtomicExchange
	OpMIPSLoweredAtomicAdd
	OpMIPSLoweredAtomicAddconst
	OpMIPSLoweredAtomicCas
	OpMIPSLoweredAtomicAnd
	OpMIPSLoweredAtomicOr
	OpMIPSLoweredZero
	OpMIPSLoweredMove
	OpMIPSLoweredNilCheck
	OpMIPSFPFlagTrue
	OpMIPSFPFlagFalse
	OpMIPSLoweredGetClosurePtr
	OpMIPSMOVWconvert

	OpMIPS64ADDV
	OpMIPS64ADDVconst
	OpMIPS64SUBV
	OpMIPS64SUBVconst
	OpMIPS64MULV
	OpMIPS64MULVU
	OpMIPS64DIVV
	OpMIPS64DIVVU
	OpMIPS64ADDF
	OpMIPS64ADDD
	OpMIPS64SUBF
	OpMIPS64SUBD
	OpMIPS64MULF
	OpMIPS64MULD
	OpMIPS64DIVF
	OpMIPS64DIVD
	OpMIPS64AND
	OpMIPS64ANDconst
	OpMIPS64OR
	OpMIPS64ORconst
	OpMIPS64XOR
	OpMIPS64XORconst
	OpMIPS64NOR
	OpMIPS64NORconst
	OpMIPS64NEGV
	OpMIPS64NEGF
	OpMIPS64NEGD
	OpMIPS64SLLV
	OpMIPS64SLLVconst
	OpMIPS64SRLV
	OpMIPS64SRLVconst
	OpMIPS64SRAV
	OpMIPS64SRAVconst
	OpMIPS64SGT
	OpMIPS64SGTconst
	OpMIPS64SGTU
	OpMIPS64SGTUconst
	OpMIPS64CMPEQF
	OpMIPS64CMPEQD
	OpMIPS64CMPGEF
	OpMIPS64CMPGED
	OpMIPS64CMPGTF
	OpMIPS64CMPGTD
	OpMIPS64MOVVconst
	OpMIPS64MOVFconst
	OpMIPS64MOVDconst
	OpMIPS64MOVVaddr
	OpMIPS64MOVBload
	OpMIPS64MOVBUload
	OpMIPS64MOVHload
	OpMIPS64MOVHUload
	OpMIPS64MOVWload
	OpMIPS64MOVWUload
	OpMIPS64MOVVload
	OpMIPS64MOVFload
	OpMIPS64MOVDload
	OpMIPS64MOVBstore
	OpMIPS64MOVHstore
	OpMIPS64MOVWstore
	OpMIPS64MOVVstore
	OpMIPS64MOVFstore
	OpMIPS64MOVDstore
	OpMIPS64MOVBstorezero
	OpMIPS64MOVHstorezero
	OpMIPS64MOVWstorezero
	OpMIPS64MOVVstorezero
	OpMIPS64MOVBreg
	OpMIPS64MOVBUreg
	OpMIPS64MOVHreg
	OpMIPS64MOVHUreg
	OpMIPS64MOVWreg
	OpMIPS64MOVWUreg
	OpMIPS64MOVVreg
	OpMIPS64MOVVnop
	OpMIPS64MOVWF
	OpMIPS64MOVWD
	OpMIPS64MOVVF
	OpMIPS64MOVVD
	OpMIPS64TRUNCFW
	OpMIPS64TRUNCDW
	OpMIPS64TRUNCFV
	OpMIPS64TRUNCDV
	OpMIPS64MOVFD
	OpMIPS64MOVDF
	OpMIPS64CALLstatic
	OpMIPS64CALLclosure
	OpMIPS64CALLinter
	OpMIPS64DUFFZERO
	OpMIPS64LoweredZero
	OpMIPS64LoweredMove
	OpMIPS64LoweredNilCheck
	OpMIPS64FPFlagTrue
	OpMIPS64FPFlagFalse
	OpMIPS64LoweredGetClosurePtr
	OpMIPS64MOVVconvert

	OpPPC64ADD
	OpPPC64ADDconst
	OpPPC64FADD
	OpPPC64FADDS
	OpPPC64SUB
	OpPPC64FSUB
	OpPPC64FSUBS
	OpPPC64MULLD
	OpPPC64MULLW
	OpPPC64MULHD
	OpPPC64MULHW
	OpPPC64MULHDU
	OpPPC64MULHWU
	OpPPC64FMUL
	OpPPC64FMULS
	OpPPC64FMADD
	OpPPC64FMADDS
	OpPPC64FMSUB
	OpPPC64FMSUBS
	OpPPC64SRAD
	OpPPC64SRAW
	OpPPC64SRD
	OpPPC64SRW
	OpPPC64SLD
	OpPPC64SLW
	OpPPC64ADDconstForCarry
	OpPPC64MaskIfNotCarry
	OpPPC64SRADconst
	OpPPC64SRAWconst
	OpPPC64SRDconst
	OpPPC64SRWconst
	OpPPC64SLDconst
	OpPPC64SLWconst
	OpPPC64ROTLconst
	OpPPC64ROTLWconst
	OpPPC64CNTLZD
	OpPPC64CNTLZW
	OpPPC64POPCNTD
	OpPPC64POPCNTW
	OpPPC64POPCNTB
	OpPPC64FDIV
	OpPPC64FDIVS
	OpPPC64DIVD
	OpPPC64DIVW
	OpPPC64DIVDU
	OpPPC64DIVWU
	OpPPC64FCTIDZ
	OpPPC64FCTIWZ
	OpPPC64FCFID
	OpPPC64FRSP
	OpPPC64Xf2i64
	OpPPC64Xi2f64
	OpPPC64AND
	OpPPC64ANDN
	OpPPC64OR
	OpPPC64ORN
	OpPPC64NOR
	OpPPC64XOR
	OpPPC64EQV
	OpPPC64NEG
	OpPPC64FNEG
	OpPPC64FSQRT
	OpPPC64FSQRTS
	OpPPC64ORconst
	OpPPC64XORconst
	OpPPC64ANDconst
	OpPPC64ANDCCconst
	OpPPC64MOVBreg
	OpPPC64MOVBZreg
	OpPPC64MOVHreg
	OpPPC64MOVHZreg
	OpPPC64MOVWreg
	OpPPC64MOVWZreg
	OpPPC64MOVBZload
	OpPPC64MOVHload
	OpPPC64MOVHZload
	OpPPC64MOVWload
	OpPPC64MOVWZload
	OpPPC64MOVDload
	OpPPC64FMOVDload
	OpPPC64FMOVSload
	OpPPC64MOVBstore
	OpPPC64MOVHstore
	OpPPC64MOVWstore
	OpPPC64MOVDstore
	OpPPC64FMOVDstore
	OpPPC64FMOVSstore
	OpPPC64MOVBstorezero
	OpPPC64MOVHstorezero
	OpPPC64MOVWstorezero
	OpPPC64MOVDstorezero
	OpPPC64MOVDaddr
	OpPPC64MOVDconst
	OpPPC64FMOVDconst
	OpPPC64FMOVSconst
	OpPPC64FCMPU
	OpPPC64CMP
	OpPPC64CMPU
	OpPPC64CMPW
	OpPPC64CMPWU
	OpPPC64CMPconst
	OpPPC64CMPUconst
	OpPPC64CMPWconst
	OpPPC64CMPWUconst
	OpPPC64Equal
	OpPPC64NotEqual
	OpPPC64LessThan
	OpPPC64FLessThan
	OpPPC64LessEqual
	OpPPC64FLessEqual
	OpPPC64GreaterThan
	OpPPC64FGreaterThan
	OpPPC64GreaterEqual
	OpPPC64FGreaterEqual
	OpPPC64LoweredGetClosurePtr
	OpPPC64LoweredNilCheck
	OpPPC64LoweredRound32F
	OpPPC64LoweredRound64F
	OpPPC64MOVDconvert
	OpPPC64CALLstatic
	OpPPC64CALLclosure
	OpPPC64CALLinter
	OpPPC64LoweredZero
	OpPPC64LoweredMove
	OpPPC64LoweredAtomicStore32
	OpPPC64LoweredAtomicStore64
	OpPPC64LoweredAtomicLoad32
	OpPPC64LoweredAtomicLoad64
	OpPPC64LoweredAtomicLoadPtr
	OpPPC64LoweredAtomicAdd32
	OpPPC64LoweredAtomicAdd64
	OpPPC64LoweredAtomicExchange32
	OpPPC64LoweredAtomicExchange64
	OpPPC64LoweredAtomicCas64
	OpPPC64LoweredAtomicCas32
	OpPPC64LoweredAtomicAnd8
	OpPPC64LoweredAtomicOr8
	OpPPC64InvertFlags
	OpPPC64FlagEQ
	OpPPC64FlagLT
	OpPPC64FlagGT

	OpS390XFADDS
	OpS390XFADD
	OpS390XFSUBS
	OpS390XFSUB
	OpS390XFMULS
	OpS390XFMUL
	OpS390XFDIVS
	OpS390XFDIV
	OpS390XFNEGS
	OpS390XFNEG
	OpS390XFMADDS
	OpS390XFMADD
	OpS390XFMSUBS
	OpS390XFMSUB
	OpS390XFMOVSload
	OpS390XFMOVDload
	OpS390XFMOVSconst
	OpS390XFMOVDconst
	OpS390XFMOVSloadidx
	OpS390XFMOVDloadidx
	OpS390XFMOVSstore
	OpS390XFMOVDstore
	OpS390XFMOVSstoreidx
	OpS390XFMOVDstoreidx
	OpS390XADD
	OpS390XADDW
	OpS390XADDconst
	OpS390XADDWconst
	OpS390XADDload
	OpS390XADDWload
	OpS390XSUB
	OpS390XSUBW
	OpS390XSUBconst
	OpS390XSUBWconst
	OpS390XSUBload
	OpS390XSUBWload
	OpS390XMULLD
	OpS390XMULLW
	OpS390XMULLDconst
	OpS390XMULLWconst
	OpS390XMULLDload
	OpS390XMULLWload
	OpS390XMULHD
	OpS390XMULHDU
	OpS390XDIVD
	OpS390XDIVW
	OpS390XDIVDU
	OpS390XDIVWU
	OpS390XMODD
	OpS390XMODW
	OpS390XMODDU
	OpS390XMODWU
	OpS390XAND
	OpS390XANDW
	OpS390XANDconst
	OpS390XANDWconst
	OpS390XANDload
	OpS390XANDWload
	OpS390XOR
	OpS390XORW
	OpS390XORconst
	OpS390XORWconst
	OpS390XORload
	OpS390XORWload
	OpS390XXOR
	OpS390XXORW
	OpS390XXORconst
	OpS390XXORWconst
	OpS390XXORload
	OpS390XXORWload
	OpS390XCMP
	OpS390XCMPW
	OpS390XCMPU
	OpS390XCMPWU
	OpS390XCMPconst
	OpS390XCMPWconst
	OpS390XCMPUconst
	OpS390XCMPWUconst
	OpS390XFCMPS
	OpS390XFCMP
	OpS390XSLD
	OpS390XSLW
	OpS390XSLDconst
	OpS390XSLWconst
	OpS390XSRD
	OpS390XSRW
	OpS390XSRDconst
	OpS390XSRWconst
	OpS390XSRAD
	OpS390XSRAW
	OpS390XSRADconst
	OpS390XSRAWconst
	OpS390XRLLGconst
	OpS390XRLLconst
	OpS390XNEG
	OpS390XNEGW
	OpS390XNOT
	OpS390XNOTW
	OpS390XFSQRT
	OpS390XSUBEcarrymask
	OpS390XSUBEWcarrymask
	OpS390XMOVDEQ
	OpS390XMOVDNE
	OpS390XMOVDLT
	OpS390XMOVDLE
	OpS390XMOVDGT
	OpS390XMOVDGE
	OpS390XMOVDGTnoinv
	OpS390XMOVDGEnoinv
	OpS390XMOVBreg
	OpS390XMOVBZreg
	OpS390XMOVHreg
	OpS390XMOVHZreg
	OpS390XMOVWreg
	OpS390XMOVWZreg
	OpS390XMOVDreg
	OpS390XMOVDnop
	OpS390XMOVDconst
	OpS390XCFDBRA
	OpS390XCGDBRA
	OpS390XCFEBRA
	OpS390XCGEBRA
	OpS390XCEFBRA
	OpS390XCDFBRA
	OpS390XCEGBRA
	OpS390XCDGBRA
	OpS390XLEDBR
	OpS390XLDEBR
	OpS390XMOVDaddr
	OpS390XMOVDaddridx
	OpS390XMOVBZload
	OpS390XMOVBload
	OpS390XMOVHZload
	OpS390XMOVHload
	OpS390XMOVWZload
	OpS390XMOVWload
	OpS390XMOVDload
	OpS390XMOVWBR
	OpS390XMOVDBR
	OpS390XMOVHBRload
	OpS390XMOVWBRload
	OpS390XMOVDBRload
	OpS390XMOVBstore
	OpS390XMOVHstore
	OpS390XMOVWstore
	OpS390XMOVDstore
	OpS390XMOVHBRstore
	OpS390XMOVWBRstore
	OpS390XMOVDBRstore
	OpS390XMVC
	OpS390XMOVBZloadidx
	OpS390XMOVHZloadidx
	OpS390XMOVWZloadidx
	OpS390XMOVDloadidx
	OpS390XMOVHBRloadidx
	OpS390XMOVWBRloadidx
	OpS390XMOVDBRloadidx
	OpS390XMOVBstoreidx
	OpS390XMOVHstoreidx
	OpS390XMOVWstoreidx
	OpS390XMOVDstoreidx
	OpS390XMOVHBRstoreidx
	OpS390XMOVWBRstoreidx
	OpS390XMOVDBRstoreidx
	OpS390XMOVBstoreconst
	OpS390XMOVHstoreconst
	OpS390XMOVWstoreconst
	OpS390XMOVDstoreconst
	OpS390XCLEAR
	OpS390XCALLstatic
	OpS390XCALLclosure
	OpS390XCALLinter
	OpS390XInvertFlags
	OpS390XLoweredGetG
	OpS390XLoweredGetClosurePtr
	OpS390XLoweredNilCheck
	OpS390XLoweredRound32F
	OpS390XLoweredRound64F
	OpS390XMOVDconvert
	OpS390XFlagEQ
	OpS390XFlagLT
	OpS390XFlagGT
	OpS390XMOVWZatomicload
	OpS390XMOVDatomicload
	OpS390XMOVWatomicstore
	OpS390XMOVDatomicstore
	OpS390XLAA
	OpS390XLAAG
	OpS390XAddTupleFirst32
	OpS390XAddTupleFirst64
	OpS390XLoweredAtomicCas32
	OpS390XLoweredAtomicCas64
	OpS390XLoweredAtomicExchange32
	OpS390XLoweredAtomicExchange64
	OpS390XFLOGR
	OpS390XSTMG2
	OpS390XSTMG3
	OpS390XSTMG4
	OpS390XSTM2
	OpS390XSTM3
	OpS390XSTM4
	OpS390XLoweredMove
	OpS390XLoweredZero

	OpAdd8
	OpAdd16
	OpAdd32
	OpAdd64
	OpAddPtr
	OpAdd32F
	OpAdd64F
	OpSub8
	OpSub16
	OpSub32
	OpSub64
	OpSubPtr
	OpSub32F
	OpSub64F
	OpMul8
	OpMul16
	OpMul32
	OpMul64
	OpMul32F
	OpMul64F
	OpDiv32F
	OpDiv64F
	OpHmul32
	OpHmul32u
	OpHmul64
	OpHmul64u
	OpMul32uhilo
	OpMul64uhilo
	OpAvg32u
	OpAvg64u
	OpDiv8
	OpDiv8u
	OpDiv16
	OpDiv16u
	OpDiv32
	OpDiv32u
	OpDiv64
	OpDiv64u
	OpDiv128u
	OpMod8
	OpMod8u
	OpMod16
	OpMod16u
	OpMod32
	OpMod32u
	OpMod64
	OpMod64u
	OpAnd8
	OpAnd16
	OpAnd32
	OpAnd64
	OpOr8
	OpOr16
	OpOr32
	OpOr64
	OpXor8
	OpXor16
	OpXor32
	OpXor64
	OpLsh8x8
	OpLsh8x16
	OpLsh8x32
	OpLsh8x64
	OpLsh16x8
	OpLsh16x16
	OpLsh16x32
	OpLsh16x64
	OpLsh32x8
	OpLsh32x16
	OpLsh32x32
	OpLsh32x64
	OpLsh64x8
	OpLsh64x16
	OpLsh64x32
	OpLsh64x64
	OpRsh8x8
	OpRsh8x16
	OpRsh8x32
	OpRsh8x64
	OpRsh16x8
	OpRsh16x16
	OpRsh16x32
	OpRsh16x64
	OpRsh32x8
	OpRsh32x16
	OpRsh32x32
	OpRsh32x64
	OpRsh64x8
	OpRsh64x16
	OpRsh64x32
	OpRsh64x64
	OpRsh8Ux8
	OpRsh8Ux16
	OpRsh8Ux32
	OpRsh8Ux64
	OpRsh16Ux8
	OpRsh16Ux16
	OpRsh16Ux32
	OpRsh16Ux64
	OpRsh32Ux8
	OpRsh32Ux16
	OpRsh32Ux32
	OpRsh32Ux64
	OpRsh64Ux8
	OpRsh64Ux16
	OpRsh64Ux32
	OpRsh64Ux64
	OpEq8
	OpEq16
	OpEq32
	OpEq64
	OpEqPtr
	OpEqInter
	OpEqSlice
	OpEq32F
	OpEq64F
	OpNeq8
	OpNeq16
	OpNeq32
	OpNeq64
	OpNeqPtr
	OpNeqInter
	OpNeqSlice
	OpNeq32F
	OpNeq64F
	OpLess8
	OpLess8U
	OpLess16
	OpLess16U
	OpLess32
	OpLess32U
	OpLess64
	OpLess64U
	OpLess32F
	OpLess64F
	OpLeq8
	OpLeq8U
	OpLeq16
	OpLeq16U
	OpLeq32
	OpLeq32U
	OpLeq64
	OpLeq64U
	OpLeq32F
	OpLeq64F
	OpGreater8
	OpGreater8U
	OpGreater16
	OpGreater16U
	OpGreater32
	OpGreater32U
	OpGreater64
	OpGreater64U
	OpGreater32F
	OpGreater64F
	OpGeq8
	OpGeq8U
	OpGeq16
	OpGeq16U
	OpGeq32
	OpGeq32U
	OpGeq64
	OpGeq64U
	OpGeq32F
	OpGeq64F
	OpAndB
	OpOrB
	OpEqB
	OpNeqB
	OpNot
	OpNeg8
	OpNeg16
	OpNeg32
	OpNeg64
	OpNeg32F
	OpNeg64F
	OpCom8
	OpCom16
	OpCom32
	OpCom64
	OpCtz32
	OpCtz64
	OpBitLen32
	OpBitLen64
	OpBswap32
	OpBswap64
	OpBitRev8
	OpBitRev16
	OpBitRev32
	OpBitRev64
	OpPopCount8
	OpPopCount16
	OpPopCount32
	OpPopCount64
	OpSqrt
	OpPhi
	OpCopy
	OpConvert
	OpConstBool
	OpConstString
	OpConstNil
	OpConst8
	OpConst16
	OpConst32
	OpConst64
	OpConst32F
	OpConst64F
	OpConstInterface
	OpConstSlice
	OpInitMem
	OpArg
	OpAddr
	OpSP
	OpSB
	OpLoad
	OpStore
	OpMove
	OpZero
	OpStoreWB
	OpMoveWB
	OpZeroWB
	OpClosureCall
	OpStaticCall
	OpInterCall
	OpSignExt8to16
	OpSignExt8to32
	OpSignExt8to64
	OpSignExt16to32
	OpSignExt16to64
	OpSignExt32to64
	OpZeroExt8to16
	OpZeroExt8to32
	OpZeroExt8to64
	OpZeroExt16to32
	OpZeroExt16to64
	OpZeroExt32to64
	OpTrunc16to8
	OpTrunc32to8
	OpTrunc32to16
	OpTrunc64to8
	OpTrunc64to16
	OpTrunc64to32
	OpCvt32to32F
	OpCvt32to64F
	OpCvt64to32F
	OpCvt64to64F
	OpCvt32Fto32
	OpCvt32Fto64
	OpCvt64Fto32
	OpCvt64Fto64
	OpCvt32Fto64F
	OpCvt64Fto32F
	OpRound32F
	OpRound64F
	OpIsNonNil
	OpIsInBounds
	OpIsSliceInBounds
	OpNilCheck
	OpGetG
	OpGetClosurePtr
	OpPtrIndex
	OpOffPtr
	OpSliceMake
	OpSlicePtr
	OpSliceLen
	OpSliceCap
	OpComplexMake
	OpComplexReal
	OpComplexImag
	OpStringMake
	OpStringPtr
	OpStringLen
	OpIMake
	OpITab
	OpIData
	OpStructMake0
	OpStructMake1
	OpStructMake2
	OpStructMake3
	OpStructMake4
	OpStructSelect
	OpArrayMake0
	OpArrayMake1
	OpArraySelect
	OpStoreReg
	OpLoadReg
	OpFwdRef
	OpUnknown
	OpVarDef
	OpVarKill
	OpVarLive
	OpKeepAlive
	OpInt64Make
	OpInt64Hi
	OpInt64Lo
	OpAdd32carry
	OpAdd32withcarry
	OpSub32carry
	OpSub32withcarry
	OpSignmask
	OpZeromask
	OpSlicemask
	OpCvt32Uto32F
	OpCvt32Uto64F
	OpCvt32Fto32U
	OpCvt64Fto32U
	OpCvt64Uto32F
	OpCvt64Uto64F
	OpCvt32Fto64U
	OpCvt64Fto64U
	OpSelect0
	OpSelect1
	OpAtomicLoad32
	OpAtomicLoad64
	OpAtomicLoadPtr
	OpAtomicStore32
	OpAtomicStore64
	OpAtomicStorePtrNoWB
	OpAtomicExchange32
	OpAtomicExchange64
	OpAtomicAdd32
	OpAtomicAdd64
	OpAtomicCompareAndSwap32
	OpAtomicCompareAndSwap64
	OpAtomicAnd8
	OpAtomicOr8
	OpClobber
)

var opcodeTable = [...]opInfo{
	{name: "OpInvalid"},

	{
		name:         "ADDSS",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		usesScratch:  true,
		asm:          x86.AADDSS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
				{1, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:         "ADDSD",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		asm:          x86.AADDSD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
				{1, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:         "SUBSS",
		argLen:       2,
		resultInArg0: true,
		usesScratch:  true,
		asm:          x86.ASUBSS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
				{1, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:         "SUBSD",
		argLen:       2,
		resultInArg0: true,
		asm:          x86.ASUBSD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
				{1, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:         "MULSS",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		usesScratch:  true,
		asm:          x86.AMULSS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
				{1, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:         "MULSD",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		asm:          x86.AMULSD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
				{1, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:         "DIVSS",
		argLen:       2,
		resultInArg0: true,
		usesScratch:  true,
		asm:          x86.ADIVSS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
				{1, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:         "DIVSD",
		argLen:       2,
		resultInArg0: true,
		asm:          x86.ADIVSD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
				{1, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:           "MOVSSload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVSS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:           "MOVSDload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVSD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:              "MOVSSconst",
		auxType:           auxFloat32,
		argLen:            0,
		rematerializeable: true,
		asm:               x86.AMOVSS,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:              "MOVSDconst",
		auxType:           auxFloat64,
		argLen:            0,
		rematerializeable: true,
		asm:               x86.AMOVSD,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:      "MOVSSloadidx1",
		auxType:   auxSymOff,
		argLen:    3,
		symEffect: SymRead,
		asm:       x86.AMOVSS,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:      "MOVSSloadidx4",
		auxType:   auxSymOff,
		argLen:    3,
		symEffect: SymRead,
		asm:       x86.AMOVSS,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:      "MOVSDloadidx1",
		auxType:   auxSymOff,
		argLen:    3,
		symEffect: SymRead,
		asm:       x86.AMOVSD,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:      "MOVSDloadidx8",
		auxType:   auxSymOff,
		argLen:    3,
		symEffect: SymRead,
		asm:       x86.AMOVSD,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:           "MOVSSstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVSS,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:           "MOVSDstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVSD,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:      "MOVSSstoreidx1",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       x86.AMOVSS,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{2, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:      "MOVSSstoreidx4",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       x86.AMOVSS,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{2, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:      "MOVSDstoreidx1",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       x86.AMOVSD,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{2, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:      "MOVSDstoreidx8",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       x86.AMOVSD,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{2, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:         "ADDL",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          x86.AADDL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 239}, // AX CX DX BX BP SI DI
				{0, 255}, // AX CX DX BX SP BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "ADDLconst",
		auxType:      auxInt32,
		argLen:       1,
		clobberFlags: true,
		asm:          x86.AADDL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 255}, // AX CX DX BX SP BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "ADDLcarry",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		asm:          x86.AADDL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
				{1, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "ADDLconstcarry",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		asm:          x86.AADDL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "ADCL",
		argLen:       3,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AADCL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
				{1, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "ADCLconst",
		auxType:      auxInt32,
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AADCL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SUBL",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASUBL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
				{1, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SUBLconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASUBL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SUBLcarry",
		argLen:       2,
		resultInArg0: true,
		asm:          x86.ASUBL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
				{1, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SUBLconstcarry",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		asm:          x86.ASUBL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SBBL",
		argLen:       3,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASBBL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
				{1, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SBBLconst",
		auxType:      auxInt32,
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASBBL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "MULL",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AIMULL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
				{1, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "MULLconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AIMULL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "HMULL",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          x86.AIMULL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},   // AX
				{1, 255}, // AX CX DX BX SP BP SI DI
			},
			clobbers: 1, // AX
			outputs: []outputInfo{
				{0, 4}, // DX
			},
		},
	},
	{
		name:         "HMULLU",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          x86.AMULL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},   // AX
				{1, 255}, // AX CX DX BX SP BP SI DI
			},
			clobbers: 1, // AX
			outputs: []outputInfo{
				{0, 4}, // DX
			},
		},
	},
	{
		name:         "MULLQU",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          x86.AMULL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},   // AX
				{1, 255}, // AX CX DX BX SP BP SI DI
			},
			outputs: []outputInfo{
				{0, 4}, // DX
				{1, 1}, // AX
			},
		},
	},
	{
		name:         "AVGLU",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
				{1, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "DIVL",
		argLen:       2,
		clobberFlags: true,
		asm:          x86.AIDIVL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},   // AX
				{1, 251}, // AX CX BX SP BP SI DI
			},
			clobbers: 4, // DX
			outputs: []outputInfo{
				{0, 1}, // AX
			},
		},
	},
	{
		name:         "DIVW",
		argLen:       2,
		clobberFlags: true,
		asm:          x86.AIDIVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},   // AX
				{1, 251}, // AX CX BX SP BP SI DI
			},
			clobbers: 4, // DX
			outputs: []outputInfo{
				{0, 1}, // AX
			},
		},
	},
	{
		name:         "DIVLU",
		argLen:       2,
		clobberFlags: true,
		asm:          x86.ADIVL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},   // AX
				{1, 251}, // AX CX BX SP BP SI DI
			},
			clobbers: 4, // DX
			outputs: []outputInfo{
				{0, 1}, // AX
			},
		},
	},
	{
		name:         "DIVWU",
		argLen:       2,
		clobberFlags: true,
		asm:          x86.ADIVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},   // AX
				{1, 251}, // AX CX BX SP BP SI DI
			},
			clobbers: 4, // DX
			outputs: []outputInfo{
				{0, 1}, // AX
			},
		},
	},
	{
		name:         "MODL",
		argLen:       2,
		clobberFlags: true,
		asm:          x86.AIDIVL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},   // AX
				{1, 251}, // AX CX BX SP BP SI DI
			},
			clobbers: 1, // AX
			outputs: []outputInfo{
				{0, 4}, // DX
			},
		},
	},
	{
		name:         "MODW",
		argLen:       2,
		clobberFlags: true,
		asm:          x86.AIDIVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},   // AX
				{1, 251}, // AX CX BX SP BP SI DI
			},
			clobbers: 1, // AX
			outputs: []outputInfo{
				{0, 4}, // DX
			},
		},
	},
	{
		name:         "MODLU",
		argLen:       2,
		clobberFlags: true,
		asm:          x86.ADIVL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},   // AX
				{1, 251}, // AX CX BX SP BP SI DI
			},
			clobbers: 1, // AX
			outputs: []outputInfo{
				{0, 4}, // DX
			},
		},
	},
	{
		name:         "MODWU",
		argLen:       2,
		clobberFlags: true,
		asm:          x86.ADIVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},   // AX
				{1, 251}, // AX CX BX SP BP SI DI
			},
			clobbers: 1, // AX
			outputs: []outputInfo{
				{0, 4}, // DX
			},
		},
	},
	{
		name:         "ANDL",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AANDL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
				{1, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "ANDLconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AANDL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "ORL",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AORL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
				{1, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "ORLconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AORL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "XORL",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AXORL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
				{1, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "XORLconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AXORL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "CMPL",
		argLen: 2,
		asm:    x86.ACMPL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 255}, // AX CX DX BX SP BP SI DI
				{1, 255}, // AX CX DX BX SP BP SI DI
			},
		},
	},
	{
		name:   "CMPW",
		argLen: 2,
		asm:    x86.ACMPW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 255}, // AX CX DX BX SP BP SI DI
				{1, 255}, // AX CX DX BX SP BP SI DI
			},
		},
	},
	{
		name:   "CMPB",
		argLen: 2,
		asm:    x86.ACMPB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 255}, // AX CX DX BX SP BP SI DI
				{1, 255}, // AX CX DX BX SP BP SI DI
			},
		},
	},
	{
		name:    "CMPLconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     x86.ACMPL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 255}, // AX CX DX BX SP BP SI DI
			},
		},
	},
	{
		name:    "CMPWconst",
		auxType: auxInt16,
		argLen:  1,
		asm:     x86.ACMPW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 255}, // AX CX DX BX SP BP SI DI
			},
		},
	},
	{
		name:    "CMPBconst",
		auxType: auxInt8,
		argLen:  1,
		asm:     x86.ACMPB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 255}, // AX CX DX BX SP BP SI DI
			},
		},
	},
	{
		name:        "UCOMISS",
		argLen:      2,
		usesScratch: true,
		asm:         x86.AUCOMISS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
				{1, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:        "UCOMISD",
		argLen:      2,
		usesScratch: true,
		asm:         x86.AUCOMISD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
				{1, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:        "TESTL",
		argLen:      2,
		commutative: true,
		asm:         x86.ATESTL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 255}, // AX CX DX BX SP BP SI DI
				{1, 255}, // AX CX DX BX SP BP SI DI
			},
		},
	},
	{
		name:        "TESTW",
		argLen:      2,
		commutative: true,
		asm:         x86.ATESTW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 255}, // AX CX DX BX SP BP SI DI
				{1, 255}, // AX CX DX BX SP BP SI DI
			},
		},
	},
	{
		name:        "TESTB",
		argLen:      2,
		commutative: true,
		asm:         x86.ATESTB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 255}, // AX CX DX BX SP BP SI DI
				{1, 255}, // AX CX DX BX SP BP SI DI
			},
		},
	},
	{
		name:    "TESTLconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     x86.ATESTL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 255}, // AX CX DX BX SP BP SI DI
			},
		},
	},
	{
		name:    "TESTWconst",
		auxType: auxInt16,
		argLen:  1,
		asm:     x86.ATESTW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 255}, // AX CX DX BX SP BP SI DI
			},
		},
	},
	{
		name:    "TESTBconst",
		auxType: auxInt8,
		argLen:  1,
		asm:     x86.ATESTB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 255}, // AX CX DX BX SP BP SI DI
			},
		},
	},
	{
		name:         "SHLL",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHLL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},   // CX
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SHLLconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHLL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SHRL",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHRL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},   // CX
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SHRW",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHRW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},   // CX
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SHRB",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHRB,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},   // CX
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SHRLconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHRL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SHRWconst",
		auxType:      auxInt16,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHRW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SHRBconst",
		auxType:      auxInt8,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHRB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SARL",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASARL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},   // CX
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SARW",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASARW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},   // CX
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SARB",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASARB,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},   // CX
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SARLconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASARL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SARWconst",
		auxType:      auxInt16,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASARW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SARBconst",
		auxType:      auxInt8,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASARB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "ROLLconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AROLL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "ROLWconst",
		auxType:      auxInt16,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AROLW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "ROLBconst",
		auxType:      auxInt8,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AROLB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "NEGL",
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ANEGL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "NOTL",
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ANOTL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "BSFL",
		argLen:       1,
		clobberFlags: true,
		asm:          x86.ABSFL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "BSFW",
		argLen:       1,
		clobberFlags: true,
		asm:          x86.ABSFW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "BSRL",
		argLen:       1,
		clobberFlags: true,
		asm:          x86.ABSRL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "BSRW",
		argLen:       1,
		clobberFlags: true,
		asm:          x86.ABSRW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "BSWAPL",
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ABSWAPL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "SQRTSD",
		argLen: 1,
		asm:    x86.ASQRTSD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:   "SBBLcarrymask",
		argLen: 1,
		asm:    x86.ASBBL,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "SETEQ",
		argLen: 1,
		asm:    x86.ASETEQ,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "SETNE",
		argLen: 1,
		asm:    x86.ASETNE,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "SETL",
		argLen: 1,
		asm:    x86.ASETLT,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "SETLE",
		argLen: 1,
		asm:    x86.ASETLE,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "SETG",
		argLen: 1,
		asm:    x86.ASETGT,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "SETGE",
		argLen: 1,
		asm:    x86.ASETGE,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "SETB",
		argLen: 1,
		asm:    x86.ASETCS,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "SETBE",
		argLen: 1,
		asm:    x86.ASETLS,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "SETA",
		argLen: 1,
		asm:    x86.ASETHI,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "SETAE",
		argLen: 1,
		asm:    x86.ASETCC,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SETEQF",
		argLen:       1,
		clobberFlags: true,
		asm:          x86.ASETEQ,
		reg: regInfo{
			clobbers: 1, // AX
			outputs: []outputInfo{
				{0, 238}, // CX DX BX BP SI DI
			},
		},
	},
	{
		name:         "SETNEF",
		argLen:       1,
		clobberFlags: true,
		asm:          x86.ASETNE,
		reg: regInfo{
			clobbers: 1, // AX
			outputs: []outputInfo{
				{0, 238}, // CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "SETORD",
		argLen: 1,
		asm:    x86.ASETPC,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "SETNAN",
		argLen: 1,
		asm:    x86.ASETPS,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "SETGF",
		argLen: 1,
		asm:    x86.ASETHI,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "SETGEF",
		argLen: 1,
		asm:    x86.ASETCC,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "MOVBLSX",
		argLen: 1,
		asm:    x86.AMOVBLSX,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "MOVBLZX",
		argLen: 1,
		asm:    x86.AMOVBLZX,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "MOVWLSX",
		argLen: 1,
		asm:    x86.AMOVWLSX,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "MOVWLZX",
		argLen: 1,
		asm:    x86.AMOVWLZX,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:              "MOVLconst",
		auxType:           auxInt32,
		argLen:            0,
		rematerializeable: true,
		asm:               x86.AMOVL,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:        "CVTTSD2SL",
		argLen:      1,
		usesScratch: true,
		asm:         x86.ACVTTSD2SL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:        "CVTTSS2SL",
		argLen:      1,
		usesScratch: true,
		asm:         x86.ACVTTSS2SL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:        "CVTSL2SS",
		argLen:      1,
		usesScratch: true,
		asm:         x86.ACVTSL2SS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:        "CVTSL2SD",
		argLen:      1,
		usesScratch: true,
		asm:         x86.ACVTSL2SD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:        "CVTSD2SS",
		argLen:      1,
		usesScratch: true,
		asm:         x86.ACVTSD2SS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:   "CVTSS2SD",
		argLen: 1,
		asm:    x86.ACVTSS2SD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:         "PXOR",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		asm:          x86.APXOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
				{1, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:              "LEAL",
		auxType:           auxSymOff,
		argLen:            1,
		rematerializeable: true,
		symEffect:         SymAddr,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:        "LEAL1",
		auxType:     auxSymOff,
		argLen:      2,
		commutative: true,
		symEffect:   SymAddr,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:      "LEAL2",
		auxType:   auxSymOff,
		argLen:    2,
		symEffect: SymAddr,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:      "LEAL4",
		auxType:   auxSymOff,
		argLen:    2,
		symEffect: SymAddr,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:      "LEAL8",
		auxType:   auxSymOff,
		argLen:    2,
		symEffect: SymAddr,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:           "MOVBload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVBLZX,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:           "MOVBLSXload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVBLSX,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:           "MOVWload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVWLZX,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:           "MOVWLSXload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVWLSX,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:           "MOVLload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:           "MOVBstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:           "MOVWstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:           "MOVLstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:        "MOVBloadidx1",
		auxType:     auxSymOff,
		argLen:      3,
		commutative: true,
		symEffect:   SymRead,
		asm:         x86.AMOVBLZX,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:        "MOVWloadidx1",
		auxType:     auxSymOff,
		argLen:      3,
		commutative: true,
		symEffect:   SymRead,
		asm:         x86.AMOVWLZX,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:      "MOVWloadidx2",
		auxType:   auxSymOff,
		argLen:    3,
		symEffect: SymRead,
		asm:       x86.AMOVWLZX,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:        "MOVLloadidx1",
		auxType:     auxSymOff,
		argLen:      3,
		commutative: true,
		symEffect:   SymRead,
		asm:         x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:      "MOVLloadidx4",
		auxType:   auxSymOff,
		argLen:    3,
		symEffect: SymRead,
		asm:       x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:        "MOVBstoreidx1",
		auxType:     auxSymOff,
		argLen:      4,
		commutative: true,
		symEffect:   SymWrite,
		asm:         x86.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{2, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:        "MOVWstoreidx1",
		auxType:     auxSymOff,
		argLen:      4,
		commutative: true,
		symEffect:   SymWrite,
		asm:         x86.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{2, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:      "MOVWstoreidx2",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       x86.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{2, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:        "MOVLstoreidx1",
		auxType:     auxSymOff,
		argLen:      4,
		commutative: true,
		symEffect:   SymWrite,
		asm:         x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{2, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:      "MOVLstoreidx4",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{2, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:           "MOVBstoreconst",
		auxType:        auxSymValAndOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:           "MOVWstoreconst",
		auxType:        auxSymValAndOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:           "MOVLstoreconst",
		auxType:        auxSymValAndOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:      "MOVBstoreconstidx1",
		auxType:   auxSymValAndOff,
		argLen:    3,
		symEffect: SymWrite,
		asm:       x86.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:      "MOVWstoreconstidx1",
		auxType:   auxSymValAndOff,
		argLen:    3,
		symEffect: SymWrite,
		asm:       x86.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:      "MOVWstoreconstidx2",
		auxType:   auxSymValAndOff,
		argLen:    3,
		symEffect: SymWrite,
		asm:       x86.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:      "MOVLstoreconstidx1",
		auxType:   auxSymValAndOff,
		argLen:    3,
		symEffect: SymWrite,
		asm:       x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:      "MOVLstoreconstidx4",
		auxType:   auxSymValAndOff,
		argLen:    3,
		symEffect: SymWrite,
		asm:       x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 255},   // AX CX DX BX SP BP SI DI
				{0, 65791}, // AX CX DX BX SP BP SI DI SB
			},
		},
	},
	{
		name:           "DUFFZERO",
		auxType:        auxInt64,
		argLen:         3,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 128}, // DI
				{1, 1},   // AX
			},
			clobbers: 130, // CX DI
		},
	},
	{
		name:           "REPSTOSL",
		argLen:         4,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 128}, // DI
				{1, 2},   // CX
				{2, 1},   // AX
			},
			clobbers: 130, // CX DI
		},
	},
	{
		name:         "CALLstatic",
		auxType:      auxSymOff,
		argLen:       1,
		clobberFlags: true,
		call:         true,
		symEffect:    SymNone,
		reg: regInfo{
			clobbers: 65519, // AX CX DX BX BP SI DI X0 X1 X2 X3 X4 X5 X6 X7
		},
	},
	{
		name:         "CALLclosure",
		auxType:      auxInt64,
		argLen:       3,
		clobberFlags: true,
		call:         true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 4},   // DX
				{0, 255}, // AX CX DX BX SP BP SI DI
			},
			clobbers: 65519, // AX CX DX BX BP SI DI X0 X1 X2 X3 X4 X5 X6 X7
		},
	},
	{
		name:         "CALLinter",
		auxType:      auxInt64,
		argLen:       2,
		clobberFlags: true,
		call:         true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			clobbers: 65519, // AX CX DX BX BP SI DI X0 X1 X2 X3 X4 X5 X6 X7
		},
	},
	{
		name:           "DUFFCOPY",
		auxType:        auxInt64,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		faultOnNilArg1: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 128}, // DI
				{1, 64},  // SI
			},
			clobbers: 194, // CX SI DI
		},
	},
	{
		name:           "REPMOVSL",
		argLen:         4,
		faultOnNilArg0: true,
		faultOnNilArg1: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 128}, // DI
				{1, 64},  // SI
				{2, 2},   // CX
			},
			clobbers: 194, // CX SI DI
		},
	},
	{
		name:   "InvertFlags",
		argLen: 1,
		reg:    regInfo{},
	},
	{
		name:   "LoweredGetG",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "LoweredGetClosurePtr",
		argLen: 0,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 4}, // DX
			},
		},
	},
	{
		name:           "LoweredNilCheck",
		argLen:         2,
		clobberFlags:   true,
		nilCheck:       true,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 255}, // AX CX DX BX SP BP SI DI
			},
		},
	},
	{
		name:   "MOVLconvert",
		argLen: 2,
		asm:    x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "FlagEQ",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagLT_ULT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagLT_UGT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagGT_UGT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagGT_ULT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FCHS",
		argLen: 1,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:    "MOVSSconst1",
		auxType: auxFloat32,
		argLen:  0,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:    "MOVSDconst1",
		auxType: auxFloat64,
		argLen:  0,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
		},
	},
	{
		name:   "MOVSSconst2",
		argLen: 1,
		asm:    x86.AMOVSS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},
	{
		name:   "MOVSDconst2",
		argLen: 1,
		asm:    x86.AMOVSD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 239}, // AX CX DX BX BP SI DI
			},
			outputs: []outputInfo{
				{0, 65280}, // X0 X1 X2 X3 X4 X5 X6 X7
			},
		},
	},

	{
		name:         "ADDSS",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		asm:          x86.AADDSS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{1, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:         "ADDSD",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		asm:          x86.AADDSD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{1, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:         "SUBSS",
		argLen:       2,
		resultInArg0: true,
		asm:          x86.ASUBSS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{1, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:         "SUBSD",
		argLen:       2,
		resultInArg0: true,
		asm:          x86.ASUBSD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{1, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:         "MULSS",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		asm:          x86.AMULSS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{1, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:         "MULSD",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		asm:          x86.AMULSD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{1, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:         "DIVSS",
		argLen:       2,
		resultInArg0: true,
		asm:          x86.ADIVSS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{1, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:         "DIVSD",
		argLen:       2,
		resultInArg0: true,
		asm:          x86.ADIVSD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{1, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:           "MOVSSload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVSS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:           "MOVSDload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVSD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:              "MOVSSconst",
		auxType:           auxFloat32,
		argLen:            0,
		rematerializeable: true,
		asm:               x86.AMOVSS,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:              "MOVSDconst",
		auxType:           auxFloat64,
		argLen:            0,
		rematerializeable: true,
		asm:               x86.AMOVSD,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:      "MOVSSloadidx1",
		auxType:   auxSymOff,
		argLen:    3,
		symEffect: SymRead,
		asm:       x86.AMOVSS,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:      "MOVSSloadidx4",
		auxType:   auxSymOff,
		argLen:    3,
		symEffect: SymRead,
		asm:       x86.AMOVSS,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:      "MOVSDloadidx1",
		auxType:   auxSymOff,
		argLen:    3,
		symEffect: SymRead,
		asm:       x86.AMOVSD,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:      "MOVSDloadidx8",
		auxType:   auxSymOff,
		argLen:    3,
		symEffect: SymRead,
		asm:       x86.AMOVSD,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:           "MOVSSstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVSS,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:           "MOVSDstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVSD,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:      "MOVSSstoreidx1",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       x86.AMOVSS,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{2, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:      "MOVSSstoreidx4",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       x86.AMOVSS,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{2, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:      "MOVSDstoreidx1",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       x86.AMOVSD,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{2, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:      "MOVSDstoreidx8",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       x86.AMOVSD,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{2, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:           "ADDSDmem",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            x86.AADDSD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:           "ADDSSmem",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            x86.AADDSS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:           "SUBSSmem",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            x86.ASUBSS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:           "SUBSDmem",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            x86.ASUBSD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:           "MULSSmem",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            x86.AMULSS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:           "MULSDmem",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            x86.AMULSD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:         "ADDQ",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          x86.AADDQ,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "ADDL",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          x86.AADDL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "ADDQconst",
		auxType:      auxInt64,
		argLen:       1,
		clobberFlags: true,
		asm:          x86.AADDQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "ADDLconst",
		auxType:      auxInt32,
		argLen:       1,
		clobberFlags: true,
		asm:          x86.AADDL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SUBQ",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASUBQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SUBL",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASUBL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SUBQconst",
		auxType:      auxInt64,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASUBQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SUBLconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASUBL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "MULQ",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AIMULQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "MULL",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AIMULL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "MULQconst",
		auxType:      auxInt64,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AIMULQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "MULLconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AIMULL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "HMULQ",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          x86.AIMULQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},     // AX
				{1, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			clobbers: 1, // AX
			outputs: []outputInfo{
				{0, 4}, // DX
			},
		},
	},
	{
		name:         "HMULL",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          x86.AIMULL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},     // AX
				{1, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			clobbers: 1, // AX
			outputs: []outputInfo{
				{0, 4}, // DX
			},
		},
	},
	{
		name:         "HMULQU",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          x86.AMULQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},     // AX
				{1, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			clobbers: 1, // AX
			outputs: []outputInfo{
				{0, 4}, // DX
			},
		},
	},
	{
		name:         "HMULLU",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          x86.AMULL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},     // AX
				{1, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			clobbers: 1, // AX
			outputs: []outputInfo{
				{0, 4}, // DX
			},
		},
	},
	{
		name:         "AVGQU",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "DIVQ",
		argLen:       2,
		clobberFlags: true,
		asm:          x86.AIDIVQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},     // AX
				{1, 65531}, // AX CX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 1}, // AX
				{1, 4}, // DX
			},
		},
	},
	{
		name:         "DIVL",
		argLen:       2,
		clobberFlags: true,
		asm:          x86.AIDIVL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},     // AX
				{1, 65531}, // AX CX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 1}, // AX
				{1, 4}, // DX
			},
		},
	},
	{
		name:         "DIVW",
		argLen:       2,
		clobberFlags: true,
		asm:          x86.AIDIVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},     // AX
				{1, 65531}, // AX CX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 1}, // AX
				{1, 4}, // DX
			},
		},
	},
	{
		name:         "DIVQU",
		argLen:       2,
		clobberFlags: true,
		asm:          x86.ADIVQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},     // AX
				{1, 65531}, // AX CX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 1}, // AX
				{1, 4}, // DX
			},
		},
	},
	{
		name:         "DIVLU",
		argLen:       2,
		clobberFlags: true,
		asm:          x86.ADIVL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},     // AX
				{1, 65531}, // AX CX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 1}, // AX
				{1, 4}, // DX
			},
		},
	},
	{
		name:         "DIVWU",
		argLen:       2,
		clobberFlags: true,
		asm:          x86.ADIVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},     // AX
				{1, 65531}, // AX CX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 1}, // AX
				{1, 4}, // DX
			},
		},
	},
	{
		name:         "MULQU2",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          x86.AMULQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1},     // AX
				{1, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 4}, // DX
				{1, 1}, // AX
			},
		},
	},
	{
		name:         "DIVQU2",
		argLen:       3,
		clobberFlags: true,
		asm:          x86.ADIVQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4},     // DX
				{1, 1},     // AX
				{2, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 1}, // AX
				{1, 4}, // DX
			},
		},
	},
	{
		name:         "ANDQ",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AANDQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "ANDL",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AANDL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "ANDQconst",
		auxType:      auxInt64,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AANDQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "ANDLconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AANDL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "ORQ",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AORQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "ORL",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AORL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "ORQconst",
		auxType:      auxInt64,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AORQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "ORLconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AORL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "XORQ",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AXORQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "XORL",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AXORL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "XORQconst",
		auxType:      auxInt64,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AXORQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "XORLconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AXORL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "CMPQ",
		argLen: 2,
		asm:    x86.ACMPQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "CMPL",
		argLen: 2,
		asm:    x86.ACMPL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "CMPW",
		argLen: 2,
		asm:    x86.ACMPW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "CMPB",
		argLen: 2,
		asm:    x86.ACMPB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:    "CMPQconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     x86.ACMPQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:    "CMPLconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     x86.ACMPL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:    "CMPWconst",
		auxType: auxInt16,
		argLen:  1,
		asm:     x86.ACMPW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:    "CMPBconst",
		auxType: auxInt8,
		argLen:  1,
		asm:     x86.ACMPB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "UCOMISS",
		argLen: 2,
		asm:    x86.AUCOMISS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{1, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:   "UCOMISD",
		argLen: 2,
		asm:    x86.AUCOMISD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{1, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:   "BTL",
		argLen: 2,
		asm:    x86.ABTL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "BTQ",
		argLen: 2,
		asm:    x86.ABTQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:    "BTLconst",
		auxType: auxInt8,
		argLen:  1,
		asm:     x86.ABTL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:    "BTQconst",
		auxType: auxInt8,
		argLen:  1,
		asm:     x86.ABTQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:        "TESTQ",
		argLen:      2,
		commutative: true,
		asm:         x86.ATESTQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:        "TESTL",
		argLen:      2,
		commutative: true,
		asm:         x86.ATESTL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:        "TESTW",
		argLen:      2,
		commutative: true,
		asm:         x86.ATESTW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:        "TESTB",
		argLen:      2,
		commutative: true,
		asm:         x86.ATESTB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:    "TESTQconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     x86.ATESTQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:    "TESTLconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     x86.ATESTL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:    "TESTWconst",
		auxType: auxInt16,
		argLen:  1,
		asm:     x86.ATESTW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:    "TESTBconst",
		auxType: auxInt8,
		argLen:  1,
		asm:     x86.ATESTB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SHLQ",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHLQ,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // CX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SHLL",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHLL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // CX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SHLQconst",
		auxType:      auxInt8,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHLQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SHLLconst",
		auxType:      auxInt8,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHLL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SHRQ",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHRQ,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // CX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SHRL",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHRL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // CX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SHRW",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHRW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // CX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SHRB",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHRB,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // CX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SHRQconst",
		auxType:      auxInt8,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHRQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SHRLconst",
		auxType:      auxInt8,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHRL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SHRWconst",
		auxType:      auxInt8,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHRW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SHRBconst",
		auxType:      auxInt8,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASHRB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SARQ",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASARQ,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // CX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SARL",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASARL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // CX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SARW",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASARW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // CX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SARB",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASARB,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // CX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SARQconst",
		auxType:      auxInt8,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASARQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SARLconst",
		auxType:      auxInt8,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASARL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SARWconst",
		auxType:      auxInt8,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASARW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SARBconst",
		auxType:      auxInt8,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ASARB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "ROLQ",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AROLQ,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // CX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "ROLL",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AROLL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // CX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "ROLW",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AROLW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // CX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "ROLB",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AROLB,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // CX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "RORQ",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ARORQ,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // CX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "RORL",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ARORL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // CX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "RORW",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ARORW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // CX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "RORB",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ARORB,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // CX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "ROLQconst",
		auxType:      auxInt8,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AROLQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "ROLLconst",
		auxType:      auxInt8,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AROLL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "ROLWconst",
		auxType:      auxInt8,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AROLW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "ROLBconst",
		auxType:      auxInt8,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.AROLB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "ADDLmem",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            x86.AADDL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519},      // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "ADDQmem",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            x86.AADDQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519},      // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "SUBQmem",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            x86.ASUBQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519},      // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "SUBLmem",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            x86.ASUBL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519},      // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "ANDLmem",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            x86.AANDL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519},      // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "ANDQmem",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            x86.AANDQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519},      // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "ORQmem",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            x86.AORQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519},      // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "ORLmem",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            x86.AORL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519},      // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "XORQmem",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            x86.AXORQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519},      // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "XORLmem",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            x86.AXORL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519},      // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "NEGQ",
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ANEGQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "NEGL",
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ANEGL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "NOTQ",
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ANOTQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "NOTL",
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ANOTL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "BSFQ",
		argLen: 1,
		asm:    x86.ABSFQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "BSFL",
		argLen: 1,
		asm:    x86.ABSFL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "BSRQ",
		argLen: 1,
		asm:    x86.ABSRQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "BSRL",
		argLen: 1,
		asm:    x86.ABSRL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "CMOVQEQ",
		argLen:       3,
		resultInArg0: true,
		asm:          x86.ACMOVQEQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "CMOVLEQ",
		argLen:       3,
		resultInArg0: true,
		asm:          x86.ACMOVLEQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "BSWAPQ",
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ABSWAPQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "BSWAPL",
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          x86.ABSWAPL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "POPCNTQ",
		argLen:       1,
		clobberFlags: true,
		asm:          x86.APOPCNTQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "POPCNTL",
		argLen:       1,
		clobberFlags: true,
		asm:          x86.APOPCNTL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "SQRTSD",
		argLen: 1,
		asm:    x86.ASQRTSD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:   "SBBQcarrymask",
		argLen: 1,
		asm:    x86.ASBBQ,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "SBBLcarrymask",
		argLen: 1,
		asm:    x86.ASBBL,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "SETEQ",
		argLen: 1,
		asm:    x86.ASETEQ,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "SETNE",
		argLen: 1,
		asm:    x86.ASETNE,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "SETL",
		argLen: 1,
		asm:    x86.ASETLT,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "SETLE",
		argLen: 1,
		asm:    x86.ASETLE,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "SETG",
		argLen: 1,
		asm:    x86.ASETGT,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "SETGE",
		argLen: 1,
		asm:    x86.ASETGE,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "SETB",
		argLen: 1,
		asm:    x86.ASETCS,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "SETBE",
		argLen: 1,
		asm:    x86.ASETLS,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "SETA",
		argLen: 1,
		asm:    x86.ASETHI,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "SETAE",
		argLen: 1,
		asm:    x86.ASETCC,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SETEQF",
		argLen:       1,
		clobberFlags: true,
		asm:          x86.ASETEQ,
		reg: regInfo{
			clobbers: 1, // AX
			outputs: []outputInfo{
				{0, 65518}, // CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:         "SETNEF",
		argLen:       1,
		clobberFlags: true,
		asm:          x86.ASETNE,
		reg: regInfo{
			clobbers: 1, // AX
			outputs: []outputInfo{
				{0, 65518}, // CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "SETORD",
		argLen: 1,
		asm:    x86.ASETPC,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "SETNAN",
		argLen: 1,
		asm:    x86.ASETPS,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "SETGF",
		argLen: 1,
		asm:    x86.ASETHI,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "SETGEF",
		argLen: 1,
		asm:    x86.ASETCC,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "MOVBQSX",
		argLen: 1,
		asm:    x86.AMOVBQSX,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "MOVBQZX",
		argLen: 1,
		asm:    x86.AMOVBLZX,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "MOVWQSX",
		argLen: 1,
		asm:    x86.AMOVWQSX,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "MOVWQZX",
		argLen: 1,
		asm:    x86.AMOVWLZX,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "MOVLQSX",
		argLen: 1,
		asm:    x86.AMOVLQSX,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "MOVLQZX",
		argLen: 1,
		asm:    x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:              "MOVLconst",
		auxType:           auxInt32,
		argLen:            0,
		rematerializeable: true,
		asm:               x86.AMOVL,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:              "MOVQconst",
		auxType:           auxInt64,
		argLen:            0,
		rematerializeable: true,
		asm:               x86.AMOVQ,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "CVTTSD2SL",
		argLen: 1,
		asm:    x86.ACVTTSD2SL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "CVTTSD2SQ",
		argLen: 1,
		asm:    x86.ACVTTSD2SQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "CVTTSS2SL",
		argLen: 1,
		asm:    x86.ACVTTSS2SL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "CVTTSS2SQ",
		argLen: 1,
		asm:    x86.ACVTTSS2SQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "CVTSL2SS",
		argLen: 1,
		asm:    x86.ACVTSL2SS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:   "CVTSL2SD",
		argLen: 1,
		asm:    x86.ACVTSL2SD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:   "CVTSQ2SS",
		argLen: 1,
		asm:    x86.ACVTSQ2SS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:   "CVTSQ2SD",
		argLen: 1,
		asm:    x86.ACVTSQ2SD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:   "CVTSD2SS",
		argLen: 1,
		asm:    x86.ACVTSD2SS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:   "CVTSS2SD",
		argLen: 1,
		asm:    x86.ACVTSS2SD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:         "PXOR",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		asm:          x86.APXOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{1, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:              "LEAQ",
		auxType:           auxSymOff,
		argLen:            1,
		rematerializeable: true,
		symEffect:         SymAddr,
		asm:               x86.ALEAQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:        "LEAQ1",
		auxType:     auxSymOff,
		argLen:      2,
		commutative: true,
		symEffect:   SymAddr,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:      "LEAQ2",
		auxType:   auxSymOff,
		argLen:    2,
		symEffect: SymAddr,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:      "LEAQ4",
		auxType:   auxSymOff,
		argLen:    2,
		symEffect: SymAddr,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:      "LEAQ8",
		auxType:   auxSymOff,
		argLen:    2,
		symEffect: SymAddr,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:              "LEAL",
		auxType:           auxSymOff,
		argLen:            1,
		rematerializeable: true,
		symEffect:         SymAddr,
		asm:               x86.ALEAL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "MOVBload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVBLZX,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "MOVBQSXload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVBQSX,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "MOVWload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVWLZX,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "MOVWQSXload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVWQSX,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "MOVLload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "MOVLQSXload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVLQSX,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "MOVQload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "MOVBstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:           "MOVWstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:           "MOVLstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:           "MOVQstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVQ,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:           "MOVOload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVUPS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:           "MOVOstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVUPS,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:        "MOVBloadidx1",
		auxType:     auxSymOff,
		argLen:      3,
		commutative: true,
		symEffect:   SymRead,
		asm:         x86.AMOVBLZX,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:        "MOVWloadidx1",
		auxType:     auxSymOff,
		argLen:      3,
		commutative: true,
		symEffect:   SymRead,
		asm:         x86.AMOVWLZX,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:      "MOVWloadidx2",
		auxType:   auxSymOff,
		argLen:    3,
		symEffect: SymRead,
		asm:       x86.AMOVWLZX,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:        "MOVLloadidx1",
		auxType:     auxSymOff,
		argLen:      3,
		commutative: true,
		symEffect:   SymRead,
		asm:         x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:      "MOVLloadidx4",
		auxType:   auxSymOff,
		argLen:    3,
		symEffect: SymRead,
		asm:       x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:        "MOVQloadidx1",
		auxType:     auxSymOff,
		argLen:      3,
		commutative: true,
		symEffect:   SymRead,
		asm:         x86.AMOVQ,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:      "MOVQloadidx8",
		auxType:   auxSymOff,
		argLen:    3,
		symEffect: SymRead,
		asm:       x86.AMOVQ,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:      "MOVBstoreidx1",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       x86.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{2, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:      "MOVWstoreidx1",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       x86.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{2, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:      "MOVWstoreidx2",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       x86.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{2, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:      "MOVLstoreidx1",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{2, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:      "MOVLstoreidx4",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{2, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:      "MOVQstoreidx1",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       x86.AMOVQ,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{2, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:      "MOVQstoreidx8",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       x86.AMOVQ,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{2, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:           "MOVBstoreconst",
		auxType:        auxSymValAndOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:           "MOVWstoreconst",
		auxType:        auxSymValAndOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:           "MOVLstoreconst",
		auxType:        auxSymValAndOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:           "MOVQstoreconst",
		auxType:        auxSymValAndOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            x86.AMOVQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:      "MOVBstoreconstidx1",
		auxType:   auxSymValAndOff,
		argLen:    3,
		symEffect: SymWrite,
		asm:       x86.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:      "MOVWstoreconstidx1",
		auxType:   auxSymValAndOff,
		argLen:    3,
		symEffect: SymWrite,
		asm:       x86.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:      "MOVWstoreconstidx2",
		auxType:   auxSymValAndOff,
		argLen:    3,
		symEffect: SymWrite,
		asm:       x86.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:      "MOVLstoreconstidx1",
		auxType:   auxSymValAndOff,
		argLen:    3,
		symEffect: SymWrite,
		asm:       x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:      "MOVLstoreconstidx4",
		auxType:   auxSymValAndOff,
		argLen:    3,
		symEffect: SymWrite,
		asm:       x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:      "MOVQstoreconstidx1",
		auxType:   auxSymValAndOff,
		argLen:    3,
		symEffect: SymWrite,
		asm:       x86.AMOVQ,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:      "MOVQstoreconstidx8",
		auxType:   auxSymValAndOff,
		argLen:    3,
		symEffect: SymWrite,
		asm:       x86.AMOVQ,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:           "DUFFZERO",
		auxType:        auxInt64,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 128},   // DI
				{1, 65536}, // X0
			},
			clobbers: 128, // DI
		},
	},
	{
		name:              "MOVOconst",
		auxType:           auxInt128,
		argLen:            0,
		rematerializeable: true,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 4294901760}, // X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
			},
		},
	},
	{
		name:           "REPSTOSQ",
		argLen:         4,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 128}, // DI
				{1, 2},   // CX
				{2, 1},   // AX
			},
			clobbers: 130, // CX DI
		},
	},
	{
		name:         "CALLstatic",
		auxType:      auxSymOff,
		argLen:       1,
		clobberFlags: true,
		call:         true,
		symEffect:    SymNone,
		reg: regInfo{
			clobbers: 4294967279, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
		},
	},
	{
		name:         "CALLclosure",
		auxType:      auxInt64,
		argLen:       3,
		clobberFlags: true,
		call:         true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 4},     // DX
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			clobbers: 4294967279, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
		},
	},
	{
		name:         "CALLinter",
		auxType:      auxInt64,
		argLen:       2,
		clobberFlags: true,
		call:         true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			clobbers: 4294967279, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15
		},
	},
	{
		name:           "DUFFCOPY",
		auxType:        auxInt64,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		faultOnNilArg1: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 128}, // DI
				{1, 64},  // SI
			},
			clobbers: 65728, // SI DI X0
		},
	},
	{
		name:           "REPMOVSQ",
		argLen:         4,
		faultOnNilArg0: true,
		faultOnNilArg1: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 128}, // DI
				{1, 64},  // SI
				{2, 2},   // CX
			},
			clobbers: 194, // CX SI DI
		},
	},
	{
		name:   "InvertFlags",
		argLen: 1,
		reg:    regInfo{},
	},
	{
		name:   "LoweredGetG",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "LoweredGetClosurePtr",
		argLen: 0,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 4}, // DX
			},
		},
	},
	{
		name:           "LoweredNilCheck",
		argLen:         2,
		clobberFlags:   true,
		nilCheck:       true,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65535}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "MOVQconvert",
		argLen: 2,
		asm:    x86.AMOVQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "MOVLconvert",
		argLen: 2,
		asm:    x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "FlagEQ",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagLT_ULT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagLT_UGT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagGT_UGT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagGT_ULT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:           "MOVLatomicload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "MOVQatomicload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            x86.AMOVQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "XCHGL",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		faultOnNilArg1: true,
		hasSideEffects: true,
		symEffect:      SymRdWr,
		asm:            x86.AXCHGL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519},      // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "XCHGQ",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		faultOnNilArg1: true,
		hasSideEffects: true,
		symEffect:      SymRdWr,
		asm:            x86.AXCHGQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519},      // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "XADDLlock",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		hasSideEffects: true,
		symEffect:      SymRdWr,
		asm:            x86.AXADDL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519},      // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "XADDQlock",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		hasSideEffects: true,
		symEffect:      SymRdWr,
		asm:            x86.AXADDQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65519},      // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{1, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
			outputs: []outputInfo{
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:   "AddTupleFirst32",
		argLen: 2,
		reg:    regInfo{},
	},
	{
		name:   "AddTupleFirst64",
		argLen: 2,
		reg:    regInfo{},
	},
	{
		name:           "CMPXCHGLlock",
		auxType:        auxSymOff,
		argLen:         4,
		clobberFlags:   true,
		faultOnNilArg0: true,
		hasSideEffects: true,
		symEffect:      SymRdWr,
		asm:            x86.ACMPXCHGL,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 1},     // AX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{2, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			clobbers: 1, // AX
			outputs: []outputInfo{
				{1, 0},
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "CMPXCHGQlock",
		auxType:        auxSymOff,
		argLen:         4,
		clobberFlags:   true,
		faultOnNilArg0: true,
		hasSideEffects: true,
		symEffect:      SymRdWr,
		asm:            x86.ACMPXCHGQ,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 1},     // AX
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{2, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
			clobbers: 1, // AX
			outputs: []outputInfo{
				{1, 0},
				{0, 65519}, // AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
			},
		},
	},
	{
		name:           "ANDBlock",
		auxType:        auxSymOff,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		hasSideEffects: true,
		symEffect:      SymRdWr,
		asm:            x86.AANDB,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},
	{
		name:           "ORBlock",
		auxType:        auxSymOff,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		hasSideEffects: true,
		symEffect:      SymRdWr,
		asm:            x86.AORB,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 65535},      // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15
				{0, 4295032831}, // AX CX DX BX SP BP SI DI R8 R9 R10 R11 R12 R13 R14 R15 SB
			},
		},
	},

	{
		name:        "ADD",
		argLen:      2,
		commutative: true,
		asm:         arm.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ADDconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 30719}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SUB",
		argLen: 2,
		asm:    arm.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SUBconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "RSB",
		argLen: 2,
		asm:    arm.ARSB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "RSBconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.ARSB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:        "MUL",
		argLen:      2,
		commutative: true,
		asm:         arm.AMUL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:        "HMUL",
		argLen:      2,
		commutative: true,
		asm:         arm.AMULL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:        "HMULU",
		argLen:      2,
		commutative: true,
		asm:         arm.AMULLU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "CALLudiv",
		argLen:       2,
		clobberFlags: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 2}, // R1
				{1, 1}, // R0
			},
			clobbers: 16396, // R2 R3 R14
			outputs: []outputInfo{
				{0, 1}, // R0
				{1, 2}, // R1
			},
		},
	},
	{
		name:        "ADDS",
		argLen:      2,
		commutative: true,
		asm:         arm.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ADDSconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:        "ADC",
		argLen:      3,
		commutative: true,
		asm:         arm.AADC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ADCconst",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.AADC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SUBS",
		argLen: 2,
		asm:    arm.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SUBSconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "RSBSconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.ARSB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SBC",
		argLen: 3,
		asm:    arm.ASBC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SBCconst",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ASBC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "RSCconst",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ARSC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:        "MULLU",
		argLen:      2,
		commutative: true,
		asm:         arm.AMULLU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MULA",
		argLen: 3,
		asm:    arm.AMULA,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:        "ADDF",
		argLen:      2,
		commutative: true,
		asm:         arm.AADDF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:        "ADDD",
		argLen:      2,
		commutative: true,
		asm:         arm.AADDD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "SUBF",
		argLen: 2,
		asm:    arm.ASUBF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "SUBD",
		argLen: 2,
		asm:    arm.ASUBD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:        "MULF",
		argLen:      2,
		commutative: true,
		asm:         arm.AMULF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:        "MULD",
		argLen:      2,
		commutative: true,
		asm:         arm.AMULD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "DIVF",
		argLen: 2,
		asm:    arm.ADIVF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "DIVD",
		argLen: 2,
		asm:    arm.ADIVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:        "AND",
		argLen:      2,
		commutative: true,
		asm:         arm.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ANDconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:        "OR",
		argLen:      2,
		commutative: true,
		asm:         arm.AORR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ORconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.AORR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:        "XOR",
		argLen:      2,
		commutative: true,
		asm:         arm.AEOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "XORconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.AEOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "BIC",
		argLen: 2,
		asm:    arm.ABIC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "BICconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.ABIC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MVN",
		argLen: 1,
		asm:    arm.AMVN,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "NEGF",
		argLen: 1,
		asm:    arm.ANEGF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "NEGD",
		argLen: 1,
		asm:    arm.ANEGD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "SQRTD",
		argLen: 1,
		asm:    arm.ASQRTD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "CLZ",
		argLen: 1,
		asm:    arm.ACLZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "REV",
		argLen: 1,
		asm:    arm.AREV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "RBIT",
		argLen: 1,
		asm:    arm.ARBIT,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SLL",
		argLen: 2,
		asm:    arm.ASLL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SLLconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.ASLL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SRL",
		argLen: 2,
		asm:    arm.ASRL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SRLconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.ASRL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SRA",
		argLen: 2,
		asm:    arm.ASRA,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SRAconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.ASRA,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SRRconst",
		auxType: auxInt32,
		argLen:  1,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ADDshiftLL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ADDshiftRL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ADDshiftRA",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SUBshiftLL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SUBshiftRL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SUBshiftRA",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "RSBshiftLL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ARSB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "RSBshiftRL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ARSB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "RSBshiftRA",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ARSB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ANDshiftLL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ANDshiftRL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ANDshiftRA",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ORshiftLL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.AORR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ORshiftRL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.AORR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ORshiftRA",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.AORR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "XORshiftLL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.AEOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "XORshiftRL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.AEOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "XORshiftRA",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.AEOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "XORshiftRR",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.AEOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "BICshiftLL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ABIC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "BICshiftRL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ABIC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "BICshiftRA",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ABIC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "MVNshiftLL",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.AMVN,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "MVNshiftRL",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.AMVN,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "MVNshiftRA",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.AMVN,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ADCshiftLL",
		auxType: auxInt32,
		argLen:  3,
		asm:     arm.AADC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ADCshiftRL",
		auxType: auxInt32,
		argLen:  3,
		asm:     arm.AADC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ADCshiftRA",
		auxType: auxInt32,
		argLen:  3,
		asm:     arm.AADC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SBCshiftLL",
		auxType: auxInt32,
		argLen:  3,
		asm:     arm.ASBC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SBCshiftRL",
		auxType: auxInt32,
		argLen:  3,
		asm:     arm.ASBC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SBCshiftRA",
		auxType: auxInt32,
		argLen:  3,
		asm:     arm.ASBC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "RSCshiftLL",
		auxType: auxInt32,
		argLen:  3,
		asm:     arm.ARSC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "RSCshiftRL",
		auxType: auxInt32,
		argLen:  3,
		asm:     arm.ARSC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "RSCshiftRA",
		auxType: auxInt32,
		argLen:  3,
		asm:     arm.ARSC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ADDSshiftLL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ADDSshiftRL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "ADDSshiftRA",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SUBSshiftLL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SUBSshiftRL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SUBSshiftRA",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "RSBSshiftLL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ARSB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "RSBSshiftRL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ARSB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "RSBSshiftRA",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ARSB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "ADDshiftLLreg",
		argLen: 3,
		asm:    arm.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "ADDshiftRLreg",
		argLen: 3,
		asm:    arm.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "ADDshiftRAreg",
		argLen: 3,
		asm:    arm.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SUBshiftLLreg",
		argLen: 3,
		asm:    arm.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SUBshiftRLreg",
		argLen: 3,
		asm:    arm.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SUBshiftRAreg",
		argLen: 3,
		asm:    arm.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "RSBshiftLLreg",
		argLen: 3,
		asm:    arm.ARSB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "RSBshiftRLreg",
		argLen: 3,
		asm:    arm.ARSB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "RSBshiftRAreg",
		argLen: 3,
		asm:    arm.ARSB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "ANDshiftLLreg",
		argLen: 3,
		asm:    arm.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "ANDshiftRLreg",
		argLen: 3,
		asm:    arm.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "ANDshiftRAreg",
		argLen: 3,
		asm:    arm.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "ORshiftLLreg",
		argLen: 3,
		asm:    arm.AORR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "ORshiftRLreg",
		argLen: 3,
		asm:    arm.AORR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "ORshiftRAreg",
		argLen: 3,
		asm:    arm.AORR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "XORshiftLLreg",
		argLen: 3,
		asm:    arm.AEOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "XORshiftRLreg",
		argLen: 3,
		asm:    arm.AEOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "XORshiftRAreg",
		argLen: 3,
		asm:    arm.AEOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "BICshiftLLreg",
		argLen: 3,
		asm:    arm.ABIC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "BICshiftRLreg",
		argLen: 3,
		asm:    arm.ABIC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "BICshiftRAreg",
		argLen: 3,
		asm:    arm.ABIC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MVNshiftLLreg",
		argLen: 2,
		asm:    arm.AMVN,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MVNshiftRLreg",
		argLen: 2,
		asm:    arm.AMVN,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MVNshiftRAreg",
		argLen: 2,
		asm:    arm.AMVN,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "ADCshiftLLreg",
		argLen: 4,
		asm:    arm.AADC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "ADCshiftRLreg",
		argLen: 4,
		asm:    arm.AADC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "ADCshiftRAreg",
		argLen: 4,
		asm:    arm.AADC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SBCshiftLLreg",
		argLen: 4,
		asm:    arm.ASBC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SBCshiftRLreg",
		argLen: 4,
		asm:    arm.ASBC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SBCshiftRAreg",
		argLen: 4,
		asm:    arm.ASBC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "RSCshiftLLreg",
		argLen: 4,
		asm:    arm.ARSC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "RSCshiftRLreg",
		argLen: 4,
		asm:    arm.ARSC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "RSCshiftRAreg",
		argLen: 4,
		asm:    arm.ARSC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "ADDSshiftLLreg",
		argLen: 3,
		asm:    arm.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "ADDSshiftRLreg",
		argLen: 3,
		asm:    arm.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "ADDSshiftRAreg",
		argLen: 3,
		asm:    arm.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SUBSshiftLLreg",
		argLen: 3,
		asm:    arm.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SUBSshiftRLreg",
		argLen: 3,
		asm:    arm.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SUBSshiftRAreg",
		argLen: 3,
		asm:    arm.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "RSBSshiftLLreg",
		argLen: 3,
		asm:    arm.ARSB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "RSBSshiftRLreg",
		argLen: 3,
		asm:    arm.ARSB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "RSBSshiftRAreg",
		argLen: 3,
		asm:    arm.ARSB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "CMP",
		argLen: 2,
		asm:    arm.ACMP,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
		},
	},
	{
		name:    "CMPconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.ACMP,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
		},
	},
	{
		name:   "CMN",
		argLen: 2,
		asm:    arm.ACMN,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
		},
	},
	{
		name:    "CMNconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.ACMN,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
		},
	},
	{
		name:        "TST",
		argLen:      2,
		commutative: true,
		asm:         arm.ATST,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
		},
	},
	{
		name:    "TSTconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.ATST,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
		},
	},
	{
		name:        "TEQ",
		argLen:      2,
		commutative: true,
		asm:         arm.ATEQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
		},
	},
	{
		name:    "TEQconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm.ATEQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
		},
	},
	{
		name:   "CMPF",
		argLen: 2,
		asm:    arm.ACMPF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "CMPD",
		argLen: 2,
		asm:    arm.ACMPD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:    "CMPshiftLL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ACMP,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
		},
	},
	{
		name:    "CMPshiftRL",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ACMP,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
		},
	},
	{
		name:    "CMPshiftRA",
		auxType: auxInt32,
		argLen:  2,
		asm:     arm.ACMP,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{1, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
		},
	},
	{
		name:   "CMPshiftLLreg",
		argLen: 3,
		asm:    arm.ACMP,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "CMPshiftRLreg",
		argLen: 3,
		asm:    arm.ACMP,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "CMPshiftRAreg",
		argLen: 3,
		asm:    arm.ACMP,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "CMPF0",
		argLen: 1,
		asm:    arm.ACMPF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "CMPD0",
		argLen: 1,
		asm:    arm.ACMPD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:              "MOVWconst",
		auxType:           auxInt32,
		argLen:            0,
		rematerializeable: true,
		asm:               arm.AMOVW,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:              "MOVFconst",
		auxType:           auxFloat64,
		argLen:            0,
		rematerializeable: true,
		asm:               arm.AMOVF,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:              "MOVDconst",
		auxType:           auxFloat64,
		argLen:            0,
		rematerializeable: true,
		asm:               arm.AMOVD,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:              "MOVWaddr",
		auxType:           auxSymOff,
		argLen:            1,
		rematerializeable: true,
		symEffect:         SymAddr,
		asm:               arm.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294975488}, // SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVBload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            arm.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVBUload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            arm.AMOVBU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVHload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            arm.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVHUload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            arm.AMOVHU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVWload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            arm.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVFload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            arm.AMOVF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:           "MOVDload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            arm.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:           "MOVBstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            arm.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 22527},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
			},
		},
	},
	{
		name:           "MOVHstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            arm.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 22527},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
			},
		},
	},
	{
		name:           "MOVWstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            arm.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 22527},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
			},
		},
	},
	{
		name:           "MOVFstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            arm.AMOVF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:           "MOVDstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            arm.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "MOVWloadidx",
		argLen: 3,
		asm:    arm.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 22527},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "MOVWloadshiftLL",
		auxType: auxInt32,
		argLen:  3,
		asm:     arm.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 22527},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "MOVWloadshiftRL",
		auxType: auxInt32,
		argLen:  3,
		asm:     arm.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 22527},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "MOVWloadshiftRA",
		auxType: auxInt32,
		argLen:  3,
		asm:     arm.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 22527},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVWstoreidx",
		argLen: 4,
		asm:    arm.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 22527},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{2, 22527},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
			},
		},
	},
	{
		name:    "MOVWstoreshiftLL",
		auxType: auxInt32,
		argLen:  4,
		asm:     arm.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 22527},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{2, 22527},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
			},
		},
	},
	{
		name:    "MOVWstoreshiftRL",
		auxType: auxInt32,
		argLen:  4,
		asm:     arm.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 22527},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{2, 22527},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
			},
		},
	},
	{
		name:    "MOVWstoreshiftRA",
		auxType: auxInt32,
		argLen:  4,
		asm:     arm.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 22527},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{2, 22527},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
				{0, 4294998015}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 SP R14 SB
			},
		},
	},
	{
		name:   "MOVBreg",
		argLen: 1,
		asm:    arm.AMOVBS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVBUreg",
		argLen: 1,
		asm:    arm.AMOVBU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVHreg",
		argLen: 1,
		asm:    arm.AMOVHS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVHUreg",
		argLen: 1,
		asm:    arm.AMOVHU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVWreg",
		argLen: 1,
		asm:    arm.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MOVWnop",
		argLen:       1,
		resultInArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVWF",
		argLen: 1,
		asm:    arm.AMOVWF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			clobbers: 2147483648, // F15
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "MOVWD",
		argLen: 1,
		asm:    arm.AMOVWD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			clobbers: 2147483648, // F15
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "MOVWUF",
		argLen: 1,
		asm:    arm.AMOVWF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			clobbers: 2147483648, // F15
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "MOVWUD",
		argLen: 1,
		asm:    arm.AMOVWD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			clobbers: 2147483648, // F15
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "MOVFW",
		argLen: 1,
		asm:    arm.AMOVFW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			clobbers: 2147483648, // F15
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVDW",
		argLen: 1,
		asm:    arm.AMOVDW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			clobbers: 2147483648, // F15
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVFWU",
		argLen: 1,
		asm:    arm.AMOVFW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			clobbers: 2147483648, // F15
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVDWU",
		argLen: 1,
		asm:    arm.AMOVDW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			clobbers: 2147483648, // F15
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVFD",
		argLen: 1,
		asm:    arm.AMOVFD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "MOVDF",
		argLen: 1,
		asm:    arm.AMOVDF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:         "CMOVWHSconst",
		auxType:      auxInt32,
		argLen:       2,
		resultInArg0: true,
		asm:          arm.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "CMOVWLSconst",
		auxType:      auxInt32,
		argLen:       2,
		resultInArg0: true,
		asm:          arm.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SRAcond",
		argLen: 3,
		asm:    arm.ASRA,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "CALLstatic",
		auxType:      auxSymOff,
		argLen:       1,
		clobberFlags: true,
		call:         true,
		symEffect:    SymNone,
		reg: regInfo{
			clobbers: 4294924287, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14 F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
		},
	},
	{
		name:         "CALLclosure",
		auxType:      auxInt64,
		argLen:       3,
		clobberFlags: true,
		call:         true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 128},   // R7
				{0, 29695}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 SP R14
			},
			clobbers: 4294924287, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14 F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
		},
	},
	{
		name:         "CALLinter",
		auxType:      auxInt64,
		argLen:       2,
		clobberFlags: true,
		call:         true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			clobbers: 4294924287, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14 F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
		},
	},
	{
		name:           "LoweredNilCheck",
		argLen:         2,
		nilCheck:       true,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
		},
	},
	{
		name:   "Equal",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "NotEqual",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "LessThan",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "LessEqual",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "GreaterThan",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "GreaterEqual",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "LessThanU",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "LessEqualU",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "GreaterThanU",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "GreaterEqualU",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "DUFFZERO",
		auxType:        auxInt64,
		argLen:         3,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 2}, // R1
				{1, 1}, // R0
			},
			clobbers: 16386, // R1 R14
		},
	},
	{
		name:           "DUFFCOPY",
		auxType:        auxInt64,
		argLen:         3,
		faultOnNilArg0: true,
		faultOnNilArg1: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4}, // R2
				{1, 2}, // R1
			},
			clobbers: 16391, // R0 R1 R2 R14
		},
	},
	{
		name:           "LoweredZero",
		auxType:        auxInt64,
		argLen:         4,
		clobberFlags:   true,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 2},     // R1
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			clobbers: 2, // R1
		},
	},
	{
		name:           "LoweredMove",
		auxType:        auxInt64,
		argLen:         4,
		clobberFlags:   true,
		faultOnNilArg0: true,
		faultOnNilArg1: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4},     // R2
				{1, 2},     // R1
				{2, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			clobbers: 6, // R1 R2
		},
	},
	{
		name:   "LoweredGetClosurePtr",
		argLen: 0,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 128}, // R7
			},
		},
	},
	{
		name:   "MOVWconvert",
		argLen: 2,
		asm:    arm.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 22527}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 g R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "FlagEQ",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagLT_ULT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagLT_UGT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagGT_UGT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagGT_ULT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "InvertFlags",
		argLen: 1,
		reg:    regInfo{},
	},

	{
		name:        "ADD",
		argLen:      2,
		commutative: true,
		asm:         arm64.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "ADDconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     arm64.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1878786047}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "SUB",
		argLen: 2,
		asm:    arm64.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "SUBconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     arm64.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:        "MUL",
		argLen:      2,
		commutative: true,
		asm:         arm64.AMUL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:        "MULW",
		argLen:      2,
		commutative: true,
		asm:         arm64.AMULW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:        "MULH",
		argLen:      2,
		commutative: true,
		asm:         arm64.ASMULH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:        "UMULH",
		argLen:      2,
		commutative: true,
		asm:         arm64.AUMULH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:        "MULL",
		argLen:      2,
		commutative: true,
		asm:         arm64.ASMULL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:        "UMULL",
		argLen:      2,
		commutative: true,
		asm:         arm64.AUMULL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "DIV",
		argLen: 2,
		asm:    arm64.ASDIV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "UDIV",
		argLen: 2,
		asm:    arm64.AUDIV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "DIVW",
		argLen: 2,
		asm:    arm64.ASDIVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "UDIVW",
		argLen: 2,
		asm:    arm64.AUDIVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "MOD",
		argLen: 2,
		asm:    arm64.AREM,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "UMOD",
		argLen: 2,
		asm:    arm64.AUREM,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "MODW",
		argLen: 2,
		asm:    arm64.AREMW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "UMODW",
		argLen: 2,
		asm:    arm64.AUREMW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:        "FADDS",
		argLen:      2,
		commutative: true,
		asm:         arm64.AFADDS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:        "FADDD",
		argLen:      2,
		commutative: true,
		asm:         arm64.AFADDD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "FSUBS",
		argLen: 2,
		asm:    arm64.AFSUBS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "FSUBD",
		argLen: 2,
		asm:    arm64.AFSUBD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:        "FMULS",
		argLen:      2,
		commutative: true,
		asm:         arm64.AFMULS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:        "FMULD",
		argLen:      2,
		commutative: true,
		asm:         arm64.AFMULD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "FDIVS",
		argLen: 2,
		asm:    arm64.AFDIVS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "FDIVD",
		argLen: 2,
		asm:    arm64.AFDIVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:        "AND",
		argLen:      2,
		commutative: true,
		asm:         arm64.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "ANDconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     arm64.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:        "OR",
		argLen:      2,
		commutative: true,
		asm:         arm64.AORR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "ORconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     arm64.AORR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:        "XOR",
		argLen:      2,
		commutative: true,
		asm:         arm64.AEOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "XORconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     arm64.AEOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "BIC",
		argLen: 2,
		asm:    arm64.ABIC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "BICconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     arm64.ABIC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "MVN",
		argLen: 1,
		asm:    arm64.AMVN,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "NEG",
		argLen: 1,
		asm:    arm64.ANEG,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "FNEGS",
		argLen: 1,
		asm:    arm64.AFNEGS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "FNEGD",
		argLen: 1,
		asm:    arm64.AFNEGD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "FSQRTD",
		argLen: 1,
		asm:    arm64.AFSQRTD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "REV",
		argLen: 1,
		asm:    arm64.AREV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "REVW",
		argLen: 1,
		asm:    arm64.AREVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "REV16W",
		argLen: 1,
		asm:    arm64.AREV16W,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "RBIT",
		argLen: 1,
		asm:    arm64.ARBIT,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "RBITW",
		argLen: 1,
		asm:    arm64.ARBITW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "CLZ",
		argLen: 1,
		asm:    arm64.ACLZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "CLZW",
		argLen: 1,
		asm:    arm64.ACLZW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "SLL",
		argLen: 2,
		asm:    arm64.ALSL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "SLLconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     arm64.ALSL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "SRL",
		argLen: 2,
		asm:    arm64.ALSR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "SRLconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     arm64.ALSR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "SRA",
		argLen: 2,
		asm:    arm64.AASR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "SRAconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     arm64.AASR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "RORconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     arm64.AROR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "RORWconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     arm64.ARORW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "CMP",
		argLen: 2,
		asm:    arm64.ACMP,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
		},
	},
	{
		name:    "CMPconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     arm64.ACMP,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
		},
	},
	{
		name:   "CMPW",
		argLen: 2,
		asm:    arm64.ACMPW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
		},
	},
	{
		name:    "CMPWconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm64.ACMPW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
		},
	},
	{
		name:   "CMN",
		argLen: 2,
		asm:    arm64.ACMN,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
		},
	},
	{
		name:    "CMNconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     arm64.ACMN,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
		},
	},
	{
		name:   "CMNW",
		argLen: 2,
		asm:    arm64.ACMNW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
		},
	},
	{
		name:    "CMNWconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     arm64.ACMNW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
		},
	},
	{
		name:   "FCMPS",
		argLen: 2,
		asm:    arm64.AFCMPS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "FCMPD",
		argLen: 2,
		asm:    arm64.AFCMPD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:    "ADDshiftLL",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "ADDshiftRL",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "ADDshiftRA",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "SUBshiftLL",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "SUBshiftRL",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "SUBshiftRA",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "ANDshiftLL",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "ANDshiftRL",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "ANDshiftRA",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "ORshiftLL",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.AORR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "ORshiftRL",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.AORR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "ORshiftRA",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.AORR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "XORshiftLL",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.AEOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "XORshiftRL",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.AEOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "XORshiftRA",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.AEOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "BICshiftLL",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.ABIC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "BICshiftRL",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.ABIC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "BICshiftRA",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.ABIC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:    "CMPshiftLL",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.ACMP,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
		},
	},
	{
		name:    "CMPshiftRL",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.ACMP,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
		},
	},
	{
		name:    "CMPshiftRA",
		auxType: auxInt64,
		argLen:  2,
		asm:     arm64.ACMP,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{1, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
		},
	},
	{
		name:              "MOVDconst",
		auxType:           auxInt64,
		argLen:            0,
		rematerializeable: true,
		asm:               arm64.AMOVD,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:              "FMOVSconst",
		auxType:           auxFloat64,
		argLen:            0,
		rematerializeable: true,
		asm:               arm64.AFMOVS,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:              "FMOVDconst",
		auxType:           auxFloat64,
		argLen:            0,
		rematerializeable: true,
		asm:               arm64.AFMOVD,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:              "MOVDaddr",
		auxType:           auxSymOff,
		argLen:            1,
		rematerializeable: true,
		symEffect:         SymAddr,
		asm:               arm64.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372037928517632}, // SP SB
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:           "MOVBload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            arm64.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:           "MOVBUload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            arm64.AMOVBU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:           "MOVHload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            arm64.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:           "MOVHUload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            arm64.AMOVHU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:           "MOVWload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            arm64.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:           "MOVWUload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            arm64.AMOVWU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:           "MOVDload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            arm64.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:           "FMOVSload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            arm64.AFMOVS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:           "FMOVDload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            arm64.AFMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:           "MOVBstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            arm64.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 805044223},           // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
		},
	},
	{
		name:           "MOVHstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            arm64.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 805044223},           // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
		},
	},
	{
		name:           "MOVWstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            arm64.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 805044223},           // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
		},
	},
	{
		name:           "MOVDstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            arm64.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 805044223},           // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
		},
	},
	{
		name:           "FMOVSstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            arm64.AFMOVS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
				{1, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:           "FMOVDstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            arm64.AFMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
				{1, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:           "MOVBstorezero",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            arm64.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
		},
	},
	{
		name:           "MOVHstorezero",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            arm64.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
		},
	},
	{
		name:           "MOVWstorezero",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            arm64.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
		},
	},
	{
		name:           "MOVDstorezero",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            arm64.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
		},
	},
	{
		name:   "MOVBreg",
		argLen: 1,
		asm:    arm64.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "MOVBUreg",
		argLen: 1,
		asm:    arm64.AMOVBU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "MOVHreg",
		argLen: 1,
		asm:    arm64.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "MOVHUreg",
		argLen: 1,
		asm:    arm64.AMOVHU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "MOVWreg",
		argLen: 1,
		asm:    arm64.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "MOVWUreg",
		argLen: 1,
		asm:    arm64.AMOVWU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "MOVDreg",
		argLen: 1,
		asm:    arm64.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:         "MOVDnop",
		argLen:       1,
		resultInArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "SCVTFWS",
		argLen: 1,
		asm:    arm64.ASCVTFWS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "SCVTFWD",
		argLen: 1,
		asm:    arm64.ASCVTFWD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "UCVTFWS",
		argLen: 1,
		asm:    arm64.AUCVTFWS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "UCVTFWD",
		argLen: 1,
		asm:    arm64.AUCVTFWD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "SCVTFS",
		argLen: 1,
		asm:    arm64.ASCVTFS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "SCVTFD",
		argLen: 1,
		asm:    arm64.ASCVTFD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "UCVTFS",
		argLen: 1,
		asm:    arm64.AUCVTFS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "UCVTFD",
		argLen: 1,
		asm:    arm64.AUCVTFD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "FCVTZSSW",
		argLen: 1,
		asm:    arm64.AFCVTZSSW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "FCVTZSDW",
		argLen: 1,
		asm:    arm64.AFCVTZSDW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "FCVTZUSW",
		argLen: 1,
		asm:    arm64.AFCVTZUSW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "FCVTZUDW",
		argLen: 1,
		asm:    arm64.AFCVTZUDW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "FCVTZSS",
		argLen: 1,
		asm:    arm64.AFCVTZSS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "FCVTZSD",
		argLen: 1,
		asm:    arm64.AFCVTZSD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "FCVTZUS",
		argLen: 1,
		asm:    arm64.AFCVTZUS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "FCVTZUD",
		argLen: 1,
		asm:    arm64.AFCVTZUD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "FCVTSD",
		argLen: 1,
		asm:    arm64.AFCVTSD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "FCVTDS",
		argLen: 1,
		asm:    arm64.AFCVTDS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 9223372034707292160}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "CSELULT",
		argLen: 3,
		asm:    arm64.ACSEL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
				{1, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "CSELULT0",
		argLen: 2,
		asm:    arm64.ACSEL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:         "CALLstatic",
		auxType:      auxSymOff,
		argLen:       1,
		clobberFlags: true,
		call:         true,
		symEffect:    SymNone,
		reg: regInfo{
			clobbers: 9223372035512336383, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
		},
	},
	{
		name:         "CALLclosure",
		auxType:      auxInt64,
		argLen:       3,
		clobberFlags: true,
		call:         true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 67108864},   // R26
				{0, 1744568319}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30 SP
			},
			clobbers: 9223372035512336383, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
		},
	},
	{
		name:         "CALLinter",
		auxType:      auxInt64,
		argLen:       2,
		clobberFlags: true,
		call:         true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
			clobbers: 9223372035512336383, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
		},
	},
	{
		name:           "LoweredNilCheck",
		argLen:         2,
		nilCheck:       true,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
		},
	},
	{
		name:   "Equal",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "NotEqual",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "LessThan",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "LessEqual",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "GreaterThan",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "GreaterEqual",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "LessThanU",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "LessEqualU",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "GreaterThanU",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "GreaterEqualU",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:           "DUFFZERO",
		auxType:        auxInt64,
		argLen:         2,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
			clobbers: 536936448, // R16 R30
		},
	},
	{
		name:           "LoweredZero",
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 65536},     // R16
				{1, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
			clobbers: 65536, // R16
		},
	},
	{
		name:           "DUFFCOPY",
		auxType:        auxInt64,
		argLen:         3,
		faultOnNilArg0: true,
		faultOnNilArg1: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 131072}, // R17
				{1, 65536},  // R16
			},
			clobbers: 537067520, // R16 R17 R30
		},
	},
	{
		name:           "LoweredMove",
		argLen:         4,
		clobberFlags:   true,
		faultOnNilArg0: true,
		faultOnNilArg1: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 131072},    // R17
				{1, 65536},     // R16
				{2, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
			clobbers: 196608, // R16 R17
		},
	},
	{
		name:   "LoweredGetClosurePtr",
		argLen: 0,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 67108864}, // R26
			},
		},
	},
	{
		name:   "MOVDconvert",
		argLen: 2,
		asm:    arm64.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 805044223}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:   "FlagEQ",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagLT_ULT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagLT_UGT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagGT_UGT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagGT_ULT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "InvertFlags",
		argLen: 1,
		reg:    regInfo{},
	},
	{
		name:           "LDAR",
		argLen:         2,
		faultOnNilArg0: true,
		asm:            arm64.ALDAR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:           "LDARW",
		argLen:         2,
		faultOnNilArg0: true,
		asm:            arm64.ALDARW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:           "STLR",
		argLen:         3,
		faultOnNilArg0: true,
		hasSideEffects: true,
		asm:            arm64.ASTLR,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 805044223},           // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
		},
	},
	{
		name:           "STLRW",
		argLen:         3,
		faultOnNilArg0: true,
		hasSideEffects: true,
		asm:            arm64.ASTLRW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 805044223},           // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
		},
	},
	{
		name:            "LoweredAtomicExchange64",
		argLen:          3,
		resultNotInArgs: true,
		faultOnNilArg0:  true,
		hasSideEffects:  true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 805044223},           // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:            "LoweredAtomicExchange32",
		argLen:          3,
		resultNotInArgs: true,
		faultOnNilArg0:  true,
		hasSideEffects:  true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 805044223},           // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:            "LoweredAtomicAdd64",
		argLen:          3,
		resultNotInArgs: true,
		faultOnNilArg0:  true,
		hasSideEffects:  true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 805044223},           // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:            "LoweredAtomicAdd32",
		argLen:          3,
		resultNotInArgs: true,
		faultOnNilArg0:  true,
		hasSideEffects:  true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 805044223},           // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:            "LoweredAtomicCas64",
		argLen:          4,
		resultNotInArgs: true,
		clobberFlags:    true,
		faultOnNilArg0:  true,
		hasSideEffects:  true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 805044223},           // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{2, 805044223},           // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:            "LoweredAtomicCas32",
		argLen:          4,
		resultNotInArgs: true,
		clobberFlags:    true,
		faultOnNilArg0:  true,
		hasSideEffects:  true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 805044223},           // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{2, 805044223},           // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
			outputs: []outputInfo{
				{0, 670826495}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 R30
			},
		},
	},
	{
		name:           "LoweredAtomicAnd8",
		argLen:         3,
		faultOnNilArg0: true,
		hasSideEffects: true,
		asm:            arm64.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 805044223},           // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
		},
	},
	{
		name:           "LoweredAtomicOr8",
		argLen:         3,
		faultOnNilArg0: true,
		hasSideEffects: true,
		asm:            arm64.AORR,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 805044223},           // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30
				{0, 9223372038733561855}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R19 R20 R21 R22 R23 R24 R25 R26 g R30 SP SB
			},
		},
	},

	{
		name:        "ADD",
		argLen:      2,
		commutative: true,
		asm:         mips.AADDU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{1, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:    "ADDconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     mips.AADDU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 536870910}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:   "SUB",
		argLen: 2,
		asm:    mips.ASUBU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{1, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:    "SUBconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     mips.ASUBU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:        "MUL",
		argLen:      2,
		commutative: true,
		asm:         mips.AMUL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{1, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			clobbers: 105553116266496, // HI LO
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:        "MULT",
		argLen:      2,
		commutative: true,
		asm:         mips.AMUL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{1, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 35184372088832}, // HI
				{1, 70368744177664}, // LO
			},
		},
	},
	{
		name:        "MULTU",
		argLen:      2,
		commutative: true,
		asm:         mips.AMULU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{1, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 35184372088832}, // HI
				{1, 70368744177664}, // LO
			},
		},
	},
	{
		name:   "DIV",
		argLen: 2,
		asm:    mips.ADIV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{1, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 35184372088832}, // HI
				{1, 70368744177664}, // LO
			},
		},
	},
	{
		name:   "DIVU",
		argLen: 2,
		asm:    mips.ADIVU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{1, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 35184372088832}, // HI
				{1, 70368744177664}, // LO
			},
		},
	},
	{
		name:        "ADDF",
		argLen:      2,
		commutative: true,
		asm:         mips.AADDF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
				{1, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:        "ADDD",
		argLen:      2,
		commutative: true,
		asm:         mips.AADDD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
				{1, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:   "SUBF",
		argLen: 2,
		asm:    mips.ASUBF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
				{1, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:   "SUBD",
		argLen: 2,
		asm:    mips.ASUBD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
				{1, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:        "MULF",
		argLen:      2,
		commutative: true,
		asm:         mips.AMULF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
				{1, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:        "MULD",
		argLen:      2,
		commutative: true,
		asm:         mips.AMULD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
				{1, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:   "DIVF",
		argLen: 2,
		asm:    mips.ADIVF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
				{1, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:   "DIVD",
		argLen: 2,
		asm:    mips.ADIVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
				{1, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:        "AND",
		argLen:      2,
		commutative: true,
		asm:         mips.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{1, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:    "ANDconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     mips.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:        "OR",
		argLen:      2,
		commutative: true,
		asm:         mips.AOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{1, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:    "ORconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     mips.AOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:        "XOR",
		argLen:      2,
		commutative: true,
		asm:         mips.AXOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{1, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:    "XORconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     mips.AXOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:        "NOR",
		argLen:      2,
		commutative: true,
		asm:         mips.ANOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{1, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:    "NORconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     mips.ANOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:   "NEG",
		argLen: 1,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:   "NEGF",
		argLen: 1,
		asm:    mips.ANEGF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:   "NEGD",
		argLen: 1,
		asm:    mips.ANEGD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:   "SQRTD",
		argLen: 1,
		asm:    mips.ASQRTD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:   "SLL",
		argLen: 2,
		asm:    mips.ASLL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{1, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:    "SLLconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     mips.ASLL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:   "SRL",
		argLen: 2,
		asm:    mips.ASRL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{1, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:    "SRLconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     mips.ASRL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:   "SRA",
		argLen: 2,
		asm:    mips.ASRA,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{1, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:    "SRAconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     mips.ASRA,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:   "CLZ",
		argLen: 1,
		asm:    mips.ACLZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:   "SGT",
		argLen: 2,
		asm:    mips.ASGT,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{1, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:    "SGTconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     mips.ASGT,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:   "SGTzero",
		argLen: 1,
		asm:    mips.ASGT,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:   "SGTU",
		argLen: 2,
		asm:    mips.ASGTU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{1, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:    "SGTUconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     mips.ASGTU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:   "SGTUzero",
		argLen: 1,
		asm:    mips.ASGTU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:   "CMPEQF",
		argLen: 2,
		asm:    mips.ACMPEQF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
				{1, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:   "CMPEQD",
		argLen: 2,
		asm:    mips.ACMPEQD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
				{1, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:   "CMPGEF",
		argLen: 2,
		asm:    mips.ACMPGEF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
				{1, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:   "CMPGED",
		argLen: 2,
		asm:    mips.ACMPGED,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
				{1, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:   "CMPGTF",
		argLen: 2,
		asm:    mips.ACMPGTF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
				{1, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:   "CMPGTD",
		argLen: 2,
		asm:    mips.ACMPGTD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
				{1, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:              "MOVWconst",
		auxType:           auxInt32,
		argLen:            0,
		rematerializeable: true,
		asm:               mips.AMOVW,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:              "MOVFconst",
		auxType:           auxFloat32,
		argLen:            0,
		rematerializeable: true,
		asm:               mips.AMOVF,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:              "MOVDconst",
		auxType:           auxFloat64,
		argLen:            0,
		rematerializeable: true,
		asm:               mips.AMOVD,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:              "MOVWaddr",
		auxType:           auxSymOff,
		argLen:            1,
		rematerializeable: true,
		symEffect:         SymAddr,
		asm:               mips.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 140737555464192}, // SP SB
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:           "MOVBload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            mips.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:           "MOVBUload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            mips.AMOVBU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:           "MOVHload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            mips.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:           "MOVHUload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            mips.AMOVHU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:           "MOVWload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            mips.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:           "MOVFload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            mips.AMOVF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:           "MOVDload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            mips.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:           "MOVBstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            mips.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 469762046},       // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
		},
	},
	{
		name:           "MOVHstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            mips.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 469762046},       // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
		},
	},
	{
		name:           "MOVWstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            mips.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 469762046},       // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
		},
	},
	{
		name:           "MOVFstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            mips.AMOVF,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 35183835217920},  // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
		},
	},
	{
		name:           "MOVDstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            mips.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 35183835217920},  // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
		},
	},
	{
		name:           "MOVBstorezero",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            mips.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
		},
	},
	{
		name:           "MOVHstorezero",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            mips.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
		},
	},
	{
		name:           "MOVWstorezero",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            mips.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
		},
	},
	{
		name:   "MOVBreg",
		argLen: 1,
		asm:    mips.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:   "MOVBUreg",
		argLen: 1,
		asm:    mips.AMOVBU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:   "MOVHreg",
		argLen: 1,
		asm:    mips.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:   "MOVHUreg",
		argLen: 1,
		asm:    mips.AMOVHU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:   "MOVWreg",
		argLen: 1,
		asm:    mips.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:         "MOVWnop",
		argLen:       1,
		resultInArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:         "CMOVZ",
		argLen:       3,
		resultInArg0: true,
		asm:          mips.ACMOVZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
				{1, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
				{2, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:         "CMOVZzero",
		argLen:       2,
		resultInArg0: true,
		asm:          mips.ACMOVZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
				{1, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:   "MOVWF",
		argLen: 1,
		asm:    mips.AMOVWF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:   "MOVWD",
		argLen: 1,
		asm:    mips.AMOVWD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:   "TRUNCFW",
		argLen: 1,
		asm:    mips.ATRUNCFW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:   "TRUNCDW",
		argLen: 1,
		asm:    mips.ATRUNCDW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:   "MOVFD",
		argLen: 1,
		asm:    mips.AMOVFD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:   "MOVDF",
		argLen: 1,
		asm:    mips.AMOVDF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
			outputs: []outputInfo{
				{0, 35183835217920}, // F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30
			},
		},
	},
	{
		name:         "CALLstatic",
		auxType:      auxSymOff,
		argLen:       1,
		clobberFlags: true,
		call:         true,
		symEffect:    SymNone,
		reg: regInfo{
			clobbers: 140737421246462, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31 F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30 HI LO
		},
	},
	{
		name:         "CALLclosure",
		auxType:      auxInt64,
		argLen:       3,
		clobberFlags: true,
		call:         true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 4194304},   // R22
				{0, 402653182}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP R31
			},
			clobbers: 140737421246462, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31 F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30 HI LO
		},
	},
	{
		name:         "CALLinter",
		auxType:      auxInt64,
		argLen:       2,
		clobberFlags: true,
		call:         true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
			clobbers: 140737421246462, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31 F0 F2 F4 F6 F8 F10 F12 F14 F16 F18 F20 F22 F24 F26 F28 F30 HI LO
		},
	},
	{
		name:           "LoweredAtomicLoad",
		argLen:         2,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:           "LoweredAtomicStore",
		argLen:         3,
		faultOnNilArg0: true,
		hasSideEffects: true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 469762046},       // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
		},
	},
	{
		name:           "LoweredAtomicStorezero",
		argLen:         2,
		faultOnNilArg0: true,
		hasSideEffects: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
		},
	},
	{
		name:            "LoweredAtomicExchange",
		argLen:          3,
		resultNotInArgs: true,
		faultOnNilArg0:  true,
		hasSideEffects:  true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 469762046},       // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:            "LoweredAtomicAdd",
		argLen:          3,
		resultNotInArgs: true,
		faultOnNilArg0:  true,
		hasSideEffects:  true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 469762046},       // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:            "LoweredAtomicAddconst",
		auxType:         auxInt32,
		argLen:          2,
		resultNotInArgs: true,
		faultOnNilArg0:  true,
		hasSideEffects:  true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:            "LoweredAtomicCas",
		argLen:          4,
		resultNotInArgs: true,
		faultOnNilArg0:  true,
		hasSideEffects:  true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 469762046},       // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{2, 469762046},       // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:           "LoweredAtomicAnd",
		argLen:         3,
		faultOnNilArg0: true,
		hasSideEffects: true,
		asm:            mips.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 469762046},       // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
		},
	},
	{
		name:           "LoweredAtomicOr",
		argLen:         3,
		faultOnNilArg0: true,
		hasSideEffects: true,
		asm:            mips.AOR,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 469762046},       // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
				{0, 140738025226238}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 SP g R31 SB
			},
		},
	},
	{
		name:           "LoweredZero",
		auxType:        auxInt32,
		argLen:         3,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 2},         // R1
				{1, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
			clobbers: 2, // R1
		},
	},
	{
		name:           "LoweredMove",
		auxType:        auxInt32,
		argLen:         4,
		faultOnNilArg0: true,
		faultOnNilArg1: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4},         // R2
				{1, 2},         // R1
				{2, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
			clobbers: 6, // R1 R2
		},
	},
	{
		name:           "LoweredNilCheck",
		argLen:         2,
		nilCheck:       true,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
		},
	},
	{
		name:   "FPFlagTrue",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:   "FPFlagFalse",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},
	{
		name:   "LoweredGetClosurePtr",
		argLen: 0,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 4194304}, // R22
			},
		},
	},
	{
		name:   "MOVWconvert",
		argLen: 2,
		asm:    mips.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 469762046}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 g R31
			},
			outputs: []outputInfo{
				{0, 335544318}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R28 R31
			},
		},
	},

	{
		name:        "ADDV",
		argLen:      2,
		commutative: true,
		asm:         mips.AADDVU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{1, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:    "ADDVconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     mips.AADDVU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 268435454}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:   "SUBV",
		argLen: 2,
		asm:    mips.ASUBVU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{1, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:    "SUBVconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     mips.ASUBVU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:        "MULV",
		argLen:      2,
		commutative: true,
		asm:         mips.AMULV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{1, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 1152921504606846976}, // HI
				{1, 2305843009213693952}, // LO
			},
		},
	},
	{
		name:        "MULVU",
		argLen:      2,
		commutative: true,
		asm:         mips.AMULVU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{1, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 1152921504606846976}, // HI
				{1, 2305843009213693952}, // LO
			},
		},
	},
	{
		name:   "DIVV",
		argLen: 2,
		asm:    mips.ADIVV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{1, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 1152921504606846976}, // HI
				{1, 2305843009213693952}, // LO
			},
		},
	},
	{
		name:   "DIVVU",
		argLen: 2,
		asm:    mips.ADIVVU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{1, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 1152921504606846976}, // HI
				{1, 2305843009213693952}, // LO
			},
		},
	},
	{
		name:        "ADDF",
		argLen:      2,
		commutative: true,
		asm:         mips.AADDF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:        "ADDD",
		argLen:      2,
		commutative: true,
		asm:         mips.AADDD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "SUBF",
		argLen: 2,
		asm:    mips.ASUBF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "SUBD",
		argLen: 2,
		asm:    mips.ASUBD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:        "MULF",
		argLen:      2,
		commutative: true,
		asm:         mips.AMULF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:        "MULD",
		argLen:      2,
		commutative: true,
		asm:         mips.AMULD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "DIVF",
		argLen: 2,
		asm:    mips.ADIVF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "DIVD",
		argLen: 2,
		asm:    mips.ADIVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:        "AND",
		argLen:      2,
		commutative: true,
		asm:         mips.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{1, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:    "ANDconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     mips.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:        "OR",
		argLen:      2,
		commutative: true,
		asm:         mips.AOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{1, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:    "ORconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     mips.AOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:        "XOR",
		argLen:      2,
		commutative: true,
		asm:         mips.AXOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{1, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:    "XORconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     mips.AXOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:        "NOR",
		argLen:      2,
		commutative: true,
		asm:         mips.ANOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{1, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:    "NORconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     mips.ANOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:   "NEGV",
		argLen: 1,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:   "NEGF",
		argLen: 1,
		asm:    mips.ANEGF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "NEGD",
		argLen: 1,
		asm:    mips.ANEGD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "SLLV",
		argLen: 2,
		asm:    mips.ASLLV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{1, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:    "SLLVconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     mips.ASLLV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:   "SRLV",
		argLen: 2,
		asm:    mips.ASRLV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{1, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:    "SRLVconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     mips.ASRLV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:   "SRAV",
		argLen: 2,
		asm:    mips.ASRAV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{1, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:    "SRAVconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     mips.ASRAV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:   "SGT",
		argLen: 2,
		asm:    mips.ASGT,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{1, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:    "SGTconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     mips.ASGT,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:   "SGTU",
		argLen: 2,
		asm:    mips.ASGTU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{1, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:    "SGTUconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     mips.ASGTU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:   "CMPEQF",
		argLen: 2,
		asm:    mips.ACMPEQF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "CMPEQD",
		argLen: 2,
		asm:    mips.ACMPEQD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "CMPGEF",
		argLen: 2,
		asm:    mips.ACMPGEF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "CMPGED",
		argLen: 2,
		asm:    mips.ACMPGED,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "CMPGTF",
		argLen: 2,
		asm:    mips.ACMPGTF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "CMPGTD",
		argLen: 2,
		asm:    mips.ACMPGTD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
				{1, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:              "MOVVconst",
		auxType:           auxInt64,
		argLen:            0,
		rematerializeable: true,
		asm:               mips.AMOVV,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:              "MOVFconst",
		auxType:           auxFloat64,
		argLen:            0,
		rematerializeable: true,
		asm:               mips.AMOVF,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:              "MOVDconst",
		auxType:           auxFloat64,
		argLen:            0,
		rematerializeable: true,
		asm:               mips.AMOVD,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:              "MOVVaddr",
		auxType:           auxSymOff,
		argLen:            1,
		rematerializeable: true,
		symEffect:         SymAddr,
		asm:               mips.AMOVV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4611686018460942336}, // SP SB
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:           "MOVBload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            mips.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:           "MOVBUload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            mips.AMOVBU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:           "MOVHload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            mips.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:           "MOVHUload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            mips.AMOVHU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:           "MOVWload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            mips.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:           "MOVWUload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            mips.AMOVWU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:           "MOVVload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            mips.AMOVV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:           "MOVFload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            mips.AMOVF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:           "MOVDload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            mips.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:           "MOVBstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            mips.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 234881022},           // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
			},
		},
	},
	{
		name:           "MOVHstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            mips.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 234881022},           // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
			},
		},
	},
	{
		name:           "MOVWstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            mips.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 234881022},           // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
			},
		},
	},
	{
		name:           "MOVVstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            mips.AMOVV,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 234881022},           // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
			},
		},
	},
	{
		name:           "MOVFstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            mips.AMOVF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
				{1, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:           "MOVDstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            mips.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
				{1, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:           "MOVBstorezero",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            mips.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
			},
		},
	},
	{
		name:           "MOVHstorezero",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            mips.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
			},
		},
	},
	{
		name:           "MOVWstorezero",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            mips.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
			},
		},
	},
	{
		name:           "MOVVstorezero",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            mips.AMOVV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4611686018695823358}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP g R31 SB
			},
		},
	},
	{
		name:   "MOVBreg",
		argLen: 1,
		asm:    mips.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:   "MOVBUreg",
		argLen: 1,
		asm:    mips.AMOVBU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:   "MOVHreg",
		argLen: 1,
		asm:    mips.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:   "MOVHUreg",
		argLen: 1,
		asm:    mips.AMOVHU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:   "MOVWreg",
		argLen: 1,
		asm:    mips.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:   "MOVWUreg",
		argLen: 1,
		asm:    mips.AMOVWU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:   "MOVVreg",
		argLen: 1,
		asm:    mips.AMOVV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:         "MOVVnop",
		argLen:       1,
		resultInArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:   "MOVWF",
		argLen: 1,
		asm:    mips.AMOVWF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "MOVWD",
		argLen: 1,
		asm:    mips.AMOVWD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "MOVVF",
		argLen: 1,
		asm:    mips.AMOVVF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "MOVVD",
		argLen: 1,
		asm:    mips.AMOVVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "TRUNCFW",
		argLen: 1,
		asm:    mips.ATRUNCFW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "TRUNCDW",
		argLen: 1,
		asm:    mips.ATRUNCDW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "TRUNCFV",
		argLen: 1,
		asm:    mips.ATRUNCFV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "TRUNCDV",
		argLen: 1,
		asm:    mips.ATRUNCDV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "MOVFD",
		argLen: 1,
		asm:    mips.AMOVFD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:   "MOVDF",
		argLen: 1,
		asm:    mips.AMOVDF,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
			outputs: []outputInfo{
				{0, 1152921504338411520}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31
			},
		},
	},
	{
		name:         "CALLstatic",
		auxType:      auxSymOff,
		argLen:       1,
		clobberFlags: true,
		call:         true,
		symEffect:    SymNone,
		reg: regInfo{
			clobbers: 4611686018393833470, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31 F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31 HI LO
		},
	},
	{
		name:         "CALLclosure",
		auxType:      auxInt64,
		argLen:       3,
		clobberFlags: true,
		call:         true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 4194304},   // R22
				{0, 201326590}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 SP R31
			},
			clobbers: 4611686018393833470, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31 F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31 HI LO
		},
	},
	{
		name:         "CALLinter",
		auxType:      auxInt64,
		argLen:       2,
		clobberFlags: true,
		call:         true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
			clobbers: 4611686018393833470, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31 F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31 HI LO
		},
	},
	{
		name:           "DUFFZERO",
		auxType:        auxInt64,
		argLen:         2,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
			clobbers: 134217730, // R1 R31
		},
	},
	{
		name:           "LoweredZero",
		auxType:        auxInt64,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 2},         // R1
				{1, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
			clobbers: 2, // R1
		},
	},
	{
		name:           "LoweredMove",
		auxType:        auxInt64,
		argLen:         4,
		clobberFlags:   true,
		faultOnNilArg0: true,
		faultOnNilArg1: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4},         // R2
				{1, 2},         // R1
				{2, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
			clobbers: 6, // R1 R2
		},
	},
	{
		name:           "LoweredNilCheck",
		argLen:         2,
		nilCheck:       true,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
		},
	},
	{
		name:   "FPFlagTrue",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:   "FPFlagFalse",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},
	{
		name:   "LoweredGetClosurePtr",
		argLen: 0,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 4194304}, // R22
			},
		},
	},
	{
		name:   "MOVVconvert",
		argLen: 2,
		asm:    mips.AMOVV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 234881022}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 g R31
			},
			outputs: []outputInfo{
				{0, 167772158}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R24 R25 R31
			},
		},
	},

	{
		name:        "ADD",
		argLen:      2,
		commutative: true,
		asm:         ppc64.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:      "ADDconst",
		auxType:   auxSymOff,
		argLen:    1,
		symEffect: SymAddr,
		asm:       ppc64.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:        "FADD",
		argLen:      2,
		commutative: true,
		asm:         ppc64.AFADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{1, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:        "FADDS",
		argLen:      2,
		commutative: true,
		asm:         ppc64.AFADDS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{1, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:   "SUB",
		argLen: 2,
		asm:    ppc64.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "FSUB",
		argLen: 2,
		asm:    ppc64.AFSUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{1, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:   "FSUBS",
		argLen: 2,
		asm:    ppc64.AFSUBS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{1, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:        "MULLD",
		argLen:      2,
		commutative: true,
		asm:         ppc64.AMULLD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:        "MULLW",
		argLen:      2,
		commutative: true,
		asm:         ppc64.AMULLW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:        "MULHD",
		argLen:      2,
		commutative: true,
		asm:         ppc64.AMULHD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:        "MULHW",
		argLen:      2,
		commutative: true,
		asm:         ppc64.AMULHW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:        "MULHDU",
		argLen:      2,
		commutative: true,
		asm:         ppc64.AMULHDU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:        "MULHWU",
		argLen:      2,
		commutative: true,
		asm:         ppc64.AMULHWU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:        "FMUL",
		argLen:      2,
		commutative: true,
		asm:         ppc64.AFMUL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{1, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:        "FMULS",
		argLen:      2,
		commutative: true,
		asm:         ppc64.AFMULS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{1, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:   "FMADD",
		argLen: 3,
		asm:    ppc64.AFMADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{1, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{2, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:   "FMADDS",
		argLen: 3,
		asm:    ppc64.AFMADDS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{1, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{2, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:   "FMSUB",
		argLen: 3,
		asm:    ppc64.AFMSUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{1, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{2, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:   "FMSUBS",
		argLen: 3,
		asm:    ppc64.AFMSUBS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{1, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{2, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:   "SRAD",
		argLen: 2,
		asm:    ppc64.ASRAD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "SRAW",
		argLen: 2,
		asm:    ppc64.ASRAW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "SRD",
		argLen: 2,
		asm:    ppc64.ASRD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "SRW",
		argLen: 2,
		asm:    ppc64.ASRW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "SLD",
		argLen: 2,
		asm:    ppc64.ASLD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "SLW",
		argLen: 2,
		asm:    ppc64.ASLW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:    "ADDconstForCarry",
		auxType: auxInt16,
		argLen:  1,
		asm:     ppc64.AADDC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			clobbers: 2147483648, // R31
		},
	},
	{
		name:   "MaskIfNotCarry",
		argLen: 1,
		asm:    ppc64.AADDME,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:    "SRADconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     ppc64.ASRAD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:    "SRAWconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     ppc64.ASRAW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:    "SRDconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     ppc64.ASRD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:    "SRWconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     ppc64.ASRW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:    "SLDconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     ppc64.ASLD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:    "SLWconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     ppc64.ASLW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:    "ROTLconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     ppc64.AROTL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:    "ROTLWconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     ppc64.AROTLW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:         "CNTLZD",
		argLen:       1,
		clobberFlags: true,
		asm:          ppc64.ACNTLZD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:         "CNTLZW",
		argLen:       1,
		clobberFlags: true,
		asm:          ppc64.ACNTLZW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "POPCNTD",
		argLen: 1,
		asm:    ppc64.APOPCNTD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "POPCNTW",
		argLen: 1,
		asm:    ppc64.APOPCNTW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "POPCNTB",
		argLen: 1,
		asm:    ppc64.APOPCNTB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "FDIV",
		argLen: 2,
		asm:    ppc64.AFDIV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{1, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:   "FDIVS",
		argLen: 2,
		asm:    ppc64.AFDIVS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{1, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:   "DIVD",
		argLen: 2,
		asm:    ppc64.ADIVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "DIVW",
		argLen: 2,
		asm:    ppc64.ADIVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "DIVDU",
		argLen: 2,
		asm:    ppc64.ADIVDU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "DIVWU",
		argLen: 2,
		asm:    ppc64.ADIVWU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "FCTIDZ",
		argLen: 1,
		asm:    ppc64.AFCTIDZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:   "FCTIWZ",
		argLen: 1,
		asm:    ppc64.AFCTIWZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:   "FCFID",
		argLen: 1,
		asm:    ppc64.AFCFID,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:   "FRSP",
		argLen: 1,
		asm:    ppc64.AFRSP,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:   "Xf2i64",
		argLen: 1,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "Xi2f64",
		argLen: 1,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:        "AND",
		argLen:      2,
		commutative: true,
		asm:         ppc64.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "ANDN",
		argLen: 2,
		asm:    ppc64.AANDN,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:        "OR",
		argLen:      2,
		commutative: true,
		asm:         ppc64.AOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "ORN",
		argLen: 2,
		asm:    ppc64.AORN,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:        "NOR",
		argLen:      2,
		commutative: true,
		asm:         ppc64.ANOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:        "XOR",
		argLen:      2,
		commutative: true,
		asm:         ppc64.AXOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:        "EQV",
		argLen:      2,
		commutative: true,
		asm:         ppc64.AEQV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "NEG",
		argLen: 1,
		asm:    ppc64.ANEG,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "FNEG",
		argLen: 1,
		asm:    ppc64.AFNEG,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:   "FSQRT",
		argLen: 1,
		asm:    ppc64.AFSQRT,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:   "FSQRTS",
		argLen: 1,
		asm:    ppc64.AFSQRTS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:    "ORconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     ppc64.AOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:    "XORconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     ppc64.AXOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:         "ANDconst",
		auxType:      auxInt64,
		argLen:       1,
		clobberFlags: true,
		asm:          ppc64.AANDCC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:    "ANDCCconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     ppc64.AANDCC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "MOVBreg",
		argLen: 1,
		asm:    ppc64.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "MOVBZreg",
		argLen: 1,
		asm:    ppc64.AMOVBZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "MOVHreg",
		argLen: 1,
		asm:    ppc64.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "MOVHZreg",
		argLen: 1,
		asm:    ppc64.AMOVHZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "MOVWreg",
		argLen: 1,
		asm:    ppc64.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "MOVWZreg",
		argLen: 1,
		asm:    ppc64.AMOVWZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "MOVBZload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            ppc64.AMOVBZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "MOVHload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            ppc64.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "MOVHZload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            ppc64.AMOVHZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "MOVWload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            ppc64.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "MOVWZload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            ppc64.AMOVWZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "MOVDload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            ppc64.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "FMOVDload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            ppc64.AFMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:           "FMOVSload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            ppc64.AFMOVS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:           "MOVBstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            ppc64.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "MOVHstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            ppc64.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "MOVWstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            ppc64.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "MOVDstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            ppc64.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "FMOVDstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            ppc64.AFMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{0, 1073733630},         // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "FMOVSstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            ppc64.AFMOVS,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{0, 1073733630},         // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "MOVBstorezero",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            ppc64.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "MOVHstorezero",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            ppc64.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "MOVWstorezero",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            ppc64.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "MOVDstorezero",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            ppc64.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:              "MOVDaddr",
		auxType:           auxSymOff,
		argLen:            1,
		rematerializeable: true,
		symEffect:         SymAddr,
		asm:               ppc64.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 6}, // SP SB
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:              "MOVDconst",
		auxType:           auxInt64,
		argLen:            0,
		rematerializeable: true,
		asm:               ppc64.AMOVD,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:              "FMOVDconst",
		auxType:           auxFloat64,
		argLen:            0,
		rematerializeable: true,
		asm:               ppc64.AFMOVD,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:              "FMOVSconst",
		auxType:           auxFloat32,
		argLen:            0,
		rematerializeable: true,
		asm:               ppc64.AFMOVS,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:   "FCMPU",
		argLen: 2,
		asm:    ppc64.AFCMPU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
				{1, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:   "CMP",
		argLen: 2,
		asm:    ppc64.ACMP,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "CMPU",
		argLen: 2,
		asm:    ppc64.ACMPU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "CMPW",
		argLen: 2,
		asm:    ppc64.ACMPW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "CMPWU",
		argLen: 2,
		asm:    ppc64.ACMPWU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:    "CMPconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     ppc64.ACMP,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:    "CMPUconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     ppc64.ACMPU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:    "CMPWconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     ppc64.ACMPW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:    "CMPWUconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     ppc64.ACMPWU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "Equal",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "NotEqual",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "LessThan",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "FLessThan",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "LessEqual",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "FLessEqual",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "GreaterThan",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "FGreaterThan",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "GreaterEqual",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "FGreaterEqual",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "LoweredGetClosurePtr",
		argLen: 0,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 2048}, // R11
			},
		},
	},
	{
		name:           "LoweredNilCheck",
		argLen:         2,
		clobberFlags:   true,
		nilCheck:       true,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			clobbers: 2147483648, // R31
		},
	},
	{
		name:         "LoweredRound32F",
		argLen:       1,
		resultInArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:         "LoweredRound64F",
		argLen:       1,
		resultInArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
			outputs: []outputInfo{
				{0, 576460743713488896}, // F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
			},
		},
	},
	{
		name:   "MOVDconvert",
		argLen: 2,
		asm:    ppc64.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:         "CALLstatic",
		auxType:      auxSymOff,
		argLen:       1,
		clobberFlags: true,
		call:         true,
		symEffect:    SymNone,
		reg: regInfo{
			clobbers: 576460745860964344, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29 g F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
		},
	},
	{
		name:         "CALLclosure",
		auxType:      auxInt64,
		argLen:       3,
		clobberFlags: true,
		call:         true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2048},       // R11
				{0, 1073733626}, // SP R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			clobbers: 576460745860964344, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29 g F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
		},
	},
	{
		name:         "CALLinter",
		auxType:      auxInt64,
		argLen:       2,
		clobberFlags: true,
		call:         true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			clobbers: 576460745860964344, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29 g F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26
		},
	},
	{
		name:           "LoweredZero",
		auxType:        auxInt64,
		argLen:         2,
		clobberFlags:   true,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 8}, // R3
			},
			clobbers: 8, // R3
		},
	},
	{
		name:           "LoweredMove",
		auxType:        auxInt64,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		faultOnNilArg1: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 8},  // R3
				{1, 16}, // R4
			},
			clobbers: 1944, // R3 R4 R7 R8 R9 R10
		},
	},
	{
		name:           "LoweredAtomicStore32",
		argLen:         3,
		faultOnNilArg0: true,
		hasSideEffects: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "LoweredAtomicStore64",
		argLen:         3,
		faultOnNilArg0: true,
		hasSideEffects: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "LoweredAtomicLoad32",
		argLen:         2,
		clobberFlags:   true,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "LoweredAtomicLoad64",
		argLen:         2,
		clobberFlags:   true,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "LoweredAtomicLoadPtr",
		argLen:         2,
		clobberFlags:   true,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:            "LoweredAtomicAdd32",
		argLen:          3,
		resultNotInArgs: true,
		clobberFlags:    true,
		faultOnNilArg0:  true,
		hasSideEffects:  true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:            "LoweredAtomicAdd64",
		argLen:          3,
		resultNotInArgs: true,
		clobberFlags:    true,
		faultOnNilArg0:  true,
		hasSideEffects:  true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:            "LoweredAtomicExchange32",
		argLen:          3,
		resultNotInArgs: true,
		clobberFlags:    true,
		faultOnNilArg0:  true,
		hasSideEffects:  true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:            "LoweredAtomicExchange64",
		argLen:          3,
		resultNotInArgs: true,
		clobberFlags:    true,
		faultOnNilArg0:  true,
		hasSideEffects:  true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:            "LoweredAtomicCas64",
		argLen:          4,
		resultNotInArgs: true,
		clobberFlags:    true,
		faultOnNilArg0:  true,
		hasSideEffects:  true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{2, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:            "LoweredAtomicCas32",
		argLen:          4,
		resultNotInArgs: true,
		clobberFlags:    true,
		faultOnNilArg0:  true,
		hasSideEffects:  true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{2, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
			outputs: []outputInfo{
				{0, 1073733624}, // R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "LoweredAtomicAnd8",
		argLen:         3,
		faultOnNilArg0: true,
		hasSideEffects: true,
		asm:            ppc64.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:           "LoweredAtomicOr8",
		argLen:         3,
		faultOnNilArg0: true,
		hasSideEffects: true,
		asm:            ppc64.AOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
				{1, 1073733630}, // SP SB R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29
			},
		},
	},
	{
		name:   "InvertFlags",
		argLen: 1,
		reg:    regInfo{},
	},
	{
		name:   "FlagEQ",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagLT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagGT",
		argLen: 0,
		reg:    regInfo{},
	},

	{
		name:         "FADDS",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AFADDS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:         "FADD",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AFADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:         "FSUBS",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AFSUBS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:         "FSUB",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AFSUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:         "FMULS",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		asm:          s390x.AFMULS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:         "FMUL",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		asm:          s390x.AFMUL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:         "FDIVS",
		argLen:       2,
		resultInArg0: true,
		asm:          s390x.AFDIVS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:         "FDIV",
		argLen:       2,
		resultInArg0: true,
		asm:          s390x.AFDIV,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:         "FNEGS",
		argLen:       1,
		clobberFlags: true,
		asm:          s390x.AFNEGS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:         "FNEG",
		argLen:       1,
		clobberFlags: true,
		asm:          s390x.AFNEG,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:         "FMADDS",
		argLen:       3,
		resultInArg0: true,
		asm:          s390x.AFMADDS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{2, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:         "FMADD",
		argLen:       3,
		resultInArg0: true,
		asm:          s390x.AFMADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{2, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:         "FMSUBS",
		argLen:       3,
		resultInArg0: true,
		asm:          s390x.AFMSUBS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{2, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:         "FMSUB",
		argLen:       3,
		resultInArg0: true,
		asm:          s390x.AFMSUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{2, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:           "FMOVSload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            s390x.AFMOVS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:           "FMOVDload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            s390x.AFMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:              "FMOVSconst",
		auxType:           auxFloat32,
		argLen:            0,
		rematerializeable: true,
		asm:               s390x.AFMOVS,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:              "FMOVDconst",
		auxType:           auxFloat64,
		argLen:            0,
		rematerializeable: true,
		asm:               s390x.AFMOVD,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:      "FMOVSloadidx",
		auxType:   auxSymOff,
		argLen:    3,
		symEffect: SymRead,
		asm:       s390x.AFMOVS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:      "FMOVDloadidx",
		auxType:   auxSymOff,
		argLen:    3,
		symEffect: SymRead,
		asm:       s390x.AFMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:           "FMOVSstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.AFMOVS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:           "FMOVDstore",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.AFMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:      "FMOVSstoreidx",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       s390x.AFMOVS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54270},      // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54270},      // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{2, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:      "FMOVDstoreidx",
		auxType:   auxSymOff,
		argLen:    4,
		symEffect: SymWrite,
		asm:       s390x.AFMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54270},      // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54270},      // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{2, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:         "ADD",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          s390x.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "ADDW",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          s390x.AADDW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "ADDconst",
		auxType:      auxInt64,
		argLen:       1,
		clobberFlags: true,
		asm:          s390x.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "ADDWconst",
		auxType:      auxInt32,
		argLen:       1,
		clobberFlags: true,
		asm:          s390x.AADDW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "ADDload",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            s390x.AADD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "ADDWload",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            s390x.AADDW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "SUB",
		argLen:       2,
		clobberFlags: true,
		asm:          s390x.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "SUBW",
		argLen:       2,
		clobberFlags: true,
		asm:          s390x.ASUBW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "SUBconst",
		auxType:      auxInt64,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "SUBWconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.ASUBW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "SUBload",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            s390x.ASUB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "SUBWload",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            s390x.ASUBW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MULLD",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AMULLD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MULLW",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AMULLW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MULLDconst",
		auxType:      auxInt64,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AMULLD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MULLWconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AMULLW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MULLDload",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            s390x.AMULLD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MULLWload",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            s390x.AMULLW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MULHD",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AMULHD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MULHDU",
		argLen:       2,
		commutative:  true,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AMULHDU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "DIVD",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.ADIVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "DIVW",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.ADIVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "DIVDU",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.ADIVDU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "DIVWU",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.ADIVWU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MODD",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AMODD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MODW",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AMODW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MODDU",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AMODDU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MODWU",
		argLen:       2,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AMODWU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "AND",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          s390x.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "ANDW",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          s390x.AANDW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "ANDconst",
		auxType:      auxInt64,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "ANDWconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AANDW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "ANDload",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            s390x.AAND,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "ANDWload",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            s390x.AANDW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "OR",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          s390x.AOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "ORW",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          s390x.AORW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "ORconst",
		auxType:      auxInt64,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "ORWconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AORW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "ORload",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            s390x.AOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "ORWload",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            s390x.AORW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "XOR",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          s390x.AXOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "XORW",
		argLen:       2,
		commutative:  true,
		clobberFlags: true,
		asm:          s390x.AXORW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "XORconst",
		auxType:      auxInt64,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AXOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "XORWconst",
		auxType:      auxInt32,
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		asm:          s390x.AXORW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "XORload",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            s390x.AXOR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "XORWload",
		auxType:        auxSymOff,
		argLen:         3,
		resultInArg0:   true,
		clobberFlags:   true,
		faultOnNilArg1: true,
		symEffect:      SymRead,
		asm:            s390x.AXORW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "CMP",
		argLen: 2,
		asm:    s390x.ACMP,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:   "CMPW",
		argLen: 2,
		asm:    s390x.ACMPW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:   "CMPU",
		argLen: 2,
		asm:    s390x.ACMPU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:   "CMPWU",
		argLen: 2,
		asm:    s390x.ACMPWU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:    "CMPconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     s390x.ACMP,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:    "CMPWconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     s390x.ACMPW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:    "CMPUconst",
		auxType: auxInt64,
		argLen:  1,
		asm:     s390x.ACMPU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:    "CMPWUconst",
		auxType: auxInt32,
		argLen:  1,
		asm:     s390x.ACMPWU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:   "FCMPS",
		argLen: 2,
		asm:    s390x.ACEBR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "FCMP",
		argLen: 2,
		asm:    s390x.AFCMPU,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
				{1, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "SLD",
		argLen: 2,
		asm:    s390x.ASLD,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 21502}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SLW",
		argLen: 2,
		asm:    s390x.ASLW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 21502}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SLDconst",
		auxType: auxInt8,
		argLen:  1,
		asm:     s390x.ASLD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SLWconst",
		auxType: auxInt8,
		argLen:  1,
		asm:     s390x.ASLW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SRD",
		argLen: 2,
		asm:    s390x.ASRD,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 21502}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SRW",
		argLen: 2,
		asm:    s390x.ASRW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 21502}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SRDconst",
		auxType: auxInt8,
		argLen:  1,
		asm:     s390x.ASRD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "SRWconst",
		auxType: auxInt8,
		argLen:  1,
		asm:     s390x.ASRW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "SRAD",
		argLen:       2,
		clobberFlags: true,
		asm:          s390x.ASRAD,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 21502}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "SRAW",
		argLen:       2,
		clobberFlags: true,
		asm:          s390x.ASRAW,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 21502}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "SRADconst",
		auxType:      auxInt8,
		argLen:       1,
		clobberFlags: true,
		asm:          s390x.ASRAD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "SRAWconst",
		auxType:      auxInt8,
		argLen:       1,
		clobberFlags: true,
		asm:          s390x.ASRAW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "RLLGconst",
		auxType: auxInt8,
		argLen:  1,
		asm:     s390x.ARLLG,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:    "RLLconst",
		auxType: auxInt8,
		argLen:  1,
		asm:     s390x.ARLL,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "NEG",
		argLen:       1,
		clobberFlags: true,
		asm:          s390x.ANEG,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "NEGW",
		argLen:       1,
		clobberFlags: true,
		asm:          s390x.ANEGW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "NOT",
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "NOTW",
		argLen:       1,
		resultInArg0: true,
		clobberFlags: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "FSQRT",
		argLen: 1,
		asm:    s390x.AFSQRT,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "SUBEcarrymask",
		argLen: 1,
		asm:    s390x.ASUBE,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "SUBEWcarrymask",
		argLen: 1,
		asm:    s390x.ASUBE,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MOVDEQ",
		argLen:       3,
		resultInArg0: true,
		asm:          s390x.AMOVDEQ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MOVDNE",
		argLen:       3,
		resultInArg0: true,
		asm:          s390x.AMOVDNE,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MOVDLT",
		argLen:       3,
		resultInArg0: true,
		asm:          s390x.AMOVDLT,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MOVDLE",
		argLen:       3,
		resultInArg0: true,
		asm:          s390x.AMOVDLE,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MOVDGT",
		argLen:       3,
		resultInArg0: true,
		asm:          s390x.AMOVDGT,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MOVDGE",
		argLen:       3,
		resultInArg0: true,
		asm:          s390x.AMOVDGE,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MOVDGTnoinv",
		argLen:       3,
		resultInArg0: true,
		asm:          s390x.AMOVDGT,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MOVDGEnoinv",
		argLen:       3,
		resultInArg0: true,
		asm:          s390x.AMOVDGE,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
				{1, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVBreg",
		argLen: 1,
		asm:    s390x.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVBZreg",
		argLen: 1,
		asm:    s390x.AMOVBZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVHreg",
		argLen: 1,
		asm:    s390x.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVHZreg",
		argLen: 1,
		asm:    s390x.AMOVHZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVWreg",
		argLen: 1,
		asm:    s390x.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVWZreg",
		argLen: 1,
		asm:    s390x.AMOVWZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVDreg",
		argLen: 1,
		asm:    s390x.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MOVDnop",
		argLen:       1,
		resultInArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:              "MOVDconst",
		auxType:           auxInt64,
		argLen:            0,
		rematerializeable: true,
		asm:               s390x.AMOVD,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "CFDBRA",
		argLen: 1,
		asm:    s390x.ACFDBRA,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "CGDBRA",
		argLen: 1,
		asm:    s390x.ACGDBRA,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "CFEBRA",
		argLen: 1,
		asm:    s390x.ACFEBRA,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "CGEBRA",
		argLen: 1,
		asm:    s390x.ACGEBRA,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "CEFBRA",
		argLen: 1,
		asm:    s390x.ACEFBRA,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "CDFBRA",
		argLen: 1,
		asm:    s390x.ACDFBRA,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "CEGBRA",
		argLen: 1,
		asm:    s390x.ACEGBRA,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "CDGBRA",
		argLen: 1,
		asm:    s390x.ACDGBRA,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "LEDBR",
		argLen: 1,
		asm:    s390x.ALEDBR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "LDEBR",
		argLen: 1,
		asm:    s390x.ALDEBR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:              "MOVDaddr",
		auxType:           auxSymOff,
		argLen:            1,
		rematerializeable: true,
		clobberFlags:      true,
		symEffect:         SymRead,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295000064}, // SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MOVDaddridx",
		auxType:      auxSymOff,
		argLen:       2,
		clobberFlags: true,
		symEffect:    SymRead,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295000064}, // SP SB
				{1, 54270},      // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVBZload",
		auxType:        auxSymOff,
		argLen:         2,
		clobberFlags:   true,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            s390x.AMOVBZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVBload",
		auxType:        auxSymOff,
		argLen:         2,
		clobberFlags:   true,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            s390x.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVHZload",
		auxType:        auxSymOff,
		argLen:         2,
		clobberFlags:   true,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            s390x.AMOVHZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVHload",
		auxType:        auxSymOff,
		argLen:         2,
		clobberFlags:   true,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            s390x.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVWZload",
		auxType:        auxSymOff,
		argLen:         2,
		clobberFlags:   true,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            s390x.AMOVWZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVWload",
		auxType:        auxSymOff,
		argLen:         2,
		clobberFlags:   true,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            s390x.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVDload",
		auxType:        auxSymOff,
		argLen:         2,
		clobberFlags:   true,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            s390x.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVWBR",
		argLen: 1,
		asm:    s390x.AMOVWBR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "MOVDBR",
		argLen: 1,
		asm:    s390x.AMOVDBR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVHBRload",
		auxType:        auxSymOff,
		argLen:         2,
		clobberFlags:   true,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            s390x.AMOVHBR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVWBRload",
		auxType:        auxSymOff,
		argLen:         2,
		clobberFlags:   true,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            s390x.AMOVWBR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVDBRload",
		auxType:        auxSymOff,
		argLen:         2,
		clobberFlags:   true,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            s390x.AMOVDBR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVBstore",
		auxType:        auxSymOff,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
				{1, 54271},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:           "MOVHstore",
		auxType:        auxSymOff,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
				{1, 54271},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:           "MOVWstore",
		auxType:        auxSymOff,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
				{1, 54271},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:           "MOVDstore",
		auxType:        auxSymOff,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
				{1, 54271},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:           "MOVHBRstore",
		auxType:        auxSymOff,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.AMOVHBR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:           "MOVWBRstore",
		auxType:        auxSymOff,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.AMOVWBR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:           "MOVDBRstore",
		auxType:        auxSymOff,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.AMOVDBR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:           "MVC",
		auxType:        auxSymValAndOff,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		faultOnNilArg1: true,
		symEffect:      SymNone,
		asm:            s390x.AMVC,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:         "MOVBZloadidx",
		auxType:      auxSymOff,
		argLen:       3,
		commutative:  true,
		clobberFlags: true,
		symEffect:    SymRead,
		asm:          s390x.AMOVBZ,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 54270},      // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MOVHZloadidx",
		auxType:      auxSymOff,
		argLen:       3,
		commutative:  true,
		clobberFlags: true,
		symEffect:    SymRead,
		asm:          s390x.AMOVHZ,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 54270},      // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MOVWZloadidx",
		auxType:      auxSymOff,
		argLen:       3,
		commutative:  true,
		clobberFlags: true,
		symEffect:    SymRead,
		asm:          s390x.AMOVWZ,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 54270},      // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MOVDloadidx",
		auxType:      auxSymOff,
		argLen:       3,
		commutative:  true,
		clobberFlags: true,
		symEffect:    SymRead,
		asm:          s390x.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 54270},      // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MOVHBRloadidx",
		auxType:      auxSymOff,
		argLen:       3,
		commutative:  true,
		clobberFlags: true,
		symEffect:    SymRead,
		asm:          s390x.AMOVHBR,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 54270},      // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MOVWBRloadidx",
		auxType:      auxSymOff,
		argLen:       3,
		commutative:  true,
		clobberFlags: true,
		symEffect:    SymRead,
		asm:          s390x.AMOVWBR,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 54270},      // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MOVDBRloadidx",
		auxType:      auxSymOff,
		argLen:       3,
		commutative:  true,
		clobberFlags: true,
		symEffect:    SymRead,
		asm:          s390x.AMOVDBR,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 54270},      // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "MOVBstoreidx",
		auxType:      auxSymOff,
		argLen:       4,
		commutative:  true,
		clobberFlags: true,
		symEffect:    SymWrite,
		asm:          s390x.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{2, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:         "MOVHstoreidx",
		auxType:      auxSymOff,
		argLen:       4,
		commutative:  true,
		clobberFlags: true,
		symEffect:    SymWrite,
		asm:          s390x.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{2, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:         "MOVWstoreidx",
		auxType:      auxSymOff,
		argLen:       4,
		commutative:  true,
		clobberFlags: true,
		symEffect:    SymWrite,
		asm:          s390x.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{2, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:         "MOVDstoreidx",
		auxType:      auxSymOff,
		argLen:       4,
		commutative:  true,
		clobberFlags: true,
		symEffect:    SymWrite,
		asm:          s390x.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{2, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:         "MOVHBRstoreidx",
		auxType:      auxSymOff,
		argLen:       4,
		commutative:  true,
		clobberFlags: true,
		symEffect:    SymWrite,
		asm:          s390x.AMOVHBR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{2, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:         "MOVWBRstoreidx",
		auxType:      auxSymOff,
		argLen:       4,
		commutative:  true,
		clobberFlags: true,
		symEffect:    SymWrite,
		asm:          s390x.AMOVWBR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{2, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:         "MOVDBRstoreidx",
		auxType:      auxSymOff,
		argLen:       4,
		commutative:  true,
		clobberFlags: true,
		symEffect:    SymWrite,
		asm:          s390x.AMOVDBR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{2, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:           "MOVBstoreconst",
		auxType:        auxSymValAndOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.AMOVB,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
		},
	},
	{
		name:           "MOVHstoreconst",
		auxType:        auxSymValAndOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.AMOVH,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
		},
	},
	{
		name:           "MOVWstoreconst",
		auxType:        auxSymValAndOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
		},
	},
	{
		name:           "MOVDstoreconst",
		auxType:        auxSymValAndOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
		},
	},
	{
		name:           "CLEAR",
		auxType:        auxSymValAndOff,
		argLen:         2,
		clobberFlags:   true,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.ACLEAR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21502}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:         "CALLstatic",
		auxType:      auxSymOff,
		argLen:       1,
		clobberFlags: true,
		call:         true,
		symEffect:    SymNone,
		reg: regInfo{
			clobbers: 4294923263, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
		},
	},
	{
		name:         "CALLclosure",
		auxType:      auxInt64,
		argLen:       3,
		clobberFlags: true,
		call:         true,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 4096},  // R12
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			clobbers: 4294923263, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
		},
	},
	{
		name:         "CALLinter",
		auxType:      auxInt64,
		argLen:       2,
		clobberFlags: true,
		call:         true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21502}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			clobbers: 4294923263, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
		},
	},
	{
		name:   "InvertFlags",
		argLen: 1,
		reg:    regInfo{},
	},
	{
		name:   "LoweredGetG",
		argLen: 1,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "LoweredGetClosurePtr",
		argLen: 0,
		reg: regInfo{
			outputs: []outputInfo{
				{0, 4096}, // R12
			},
		},
	},
	{
		name:           "LoweredNilCheck",
		argLen:         2,
		clobberFlags:   true,
		nilCheck:       true,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:         "LoweredRound32F",
		argLen:       1,
		resultInArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:         "LoweredRound64F",
		argLen:       1,
		resultInArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
			outputs: []outputInfo{
				{0, 4294901760}, // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15
			},
		},
	},
	{
		name:   "MOVDconvert",
		argLen: 2,
		asm:    s390x.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "FlagEQ",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagLT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:   "FlagGT",
		argLen: 0,
		reg:    regInfo{},
	},
	{
		name:           "MOVWZatomicload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            s390x.AMOVWZ,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVDatomicload",
		auxType:        auxSymOff,
		argLen:         2,
		faultOnNilArg0: true,
		symEffect:      SymRead,
		asm:            s390x.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "MOVWatomicstore",
		auxType:        auxSymOff,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		hasSideEffects: true,
		symEffect:      SymWrite,
		asm:            s390x.AMOVW,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
				{1, 54271},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:           "MOVDatomicstore",
		auxType:        auxSymOff,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		hasSideEffects: true,
		symEffect:      SymWrite,
		asm:            s390x.AMOVD,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
				{1, 54271},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:           "LAA",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		hasSideEffects: true,
		symEffect:      SymRdWr,
		asm:            s390x.ALAA,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
				{1, 54271},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "LAAG",
		auxType:        auxSymOff,
		argLen:         3,
		faultOnNilArg0: true,
		hasSideEffects: true,
		symEffect:      SymRdWr,
		asm:            s390x.ALAAG,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 4295021566}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP SB
				{1, 54271},      // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:   "AddTupleFirst32",
		argLen: 2,
		reg:    regInfo{},
	},
	{
		name:   "AddTupleFirst64",
		argLen: 2,
		reg:    regInfo{},
	},
	{
		name:           "LoweredAtomicCas32",
		auxType:        auxSymOff,
		argLen:         4,
		clobberFlags:   true,
		faultOnNilArg0: true,
		hasSideEffects: true,
		symEffect:      SymRdWr,
		asm:            s390x.ACS,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 1},     // R0
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{2, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			clobbers: 1, // R0
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "LoweredAtomicCas64",
		auxType:        auxSymOff,
		argLen:         4,
		clobberFlags:   true,
		faultOnNilArg0: true,
		hasSideEffects: true,
		symEffect:      SymRdWr,
		asm:            s390x.ACSG,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 1},     // R0
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{2, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			clobbers: 1, // R0
			outputs: []outputInfo{
				{1, 0},
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
		},
	},
	{
		name:           "LoweredAtomicExchange32",
		auxType:        auxSymOff,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		hasSideEffects: true,
		symEffect:      SymRdWr,
		asm:            s390x.ACS,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 1}, // R0
			},
		},
	},
	{
		name:           "LoweredAtomicExchange64",
		auxType:        auxSymOff,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		hasSideEffects: true,
		symEffect:      SymRdWr,
		asm:            s390x.ACSG,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
				{1, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			outputs: []outputInfo{
				{1, 0},
				{0, 1}, // R0
			},
		},
	},
	{
		name:         "FLOGR",
		argLen:       1,
		clobberFlags: true,
		asm:          s390x.AFLOGR,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 21503}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14
			},
			clobbers: 2, // R1
			outputs: []outputInfo{
				{0, 1}, // R0
			},
		},
	},
	{
		name:           "STMG2",
		auxType:        auxSymOff,
		argLen:         4,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.ASTMG,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // R1
				{2, 4},     // R2
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:           "STMG3",
		auxType:        auxSymOff,
		argLen:         5,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.ASTMG,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // R1
				{2, 4},     // R2
				{3, 8},     // R3
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:           "STMG4",
		auxType:        auxSymOff,
		argLen:         6,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.ASTMG,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // R1
				{2, 4},     // R2
				{3, 8},     // R3
				{4, 16},    // R4
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:           "STM2",
		auxType:        auxSymOff,
		argLen:         4,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.ASTMY,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // R1
				{2, 4},     // R2
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:           "STM3",
		auxType:        auxSymOff,
		argLen:         5,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.ASTMY,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // R1
				{2, 4},     // R2
				{3, 8},     // R3
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:           "STM4",
		auxType:        auxSymOff,
		argLen:         6,
		faultOnNilArg0: true,
		symEffect:      SymWrite,
		asm:            s390x.ASTMY,
		reg: regInfo{
			inputs: []inputInfo{
				{1, 2},     // R1
				{2, 4},     // R2
				{3, 8},     // R3
				{4, 16},    // R4
				{0, 54270}, // R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
		},
	},
	{
		name:           "LoweredMove",
		auxType:        auxInt64,
		argLen:         4,
		clobberFlags:   true,
		faultOnNilArg0: true,
		faultOnNilArg1: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 2},     // R1
				{1, 4},     // R2
				{2, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			clobbers: 6, // R1 R2
		},
	},
	{
		name:           "LoweredZero",
		auxType:        auxInt64,
		argLen:         3,
		clobberFlags:   true,
		faultOnNilArg0: true,
		reg: regInfo{
			inputs: []inputInfo{
				{0, 2},     // R1
				{1, 54271}, // R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R12 R14 SP
			},
			clobbers: 2, // R1
		},
	},

	{
		name:        "Add8",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Add16",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Add32",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Add64",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:    "AddPtr",
		argLen:  2,
		generic: true,
	},
	{
		name:        "Add32F",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Add64F",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:    "Sub8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Sub16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Sub32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Sub64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "SubPtr",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Sub32F",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Sub64F",
		argLen:  2,
		generic: true,
	},
	{
		name:        "Mul8",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Mul16",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Mul32",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Mul64",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Mul32F",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Mul64F",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:    "Div32F",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Div64F",
		argLen:  2,
		generic: true,
	},
	{
		name:        "Hmul32",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Hmul32u",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Hmul64",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Hmul64u",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Mul32uhilo",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Mul64uhilo",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:    "Avg32u",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Avg64u",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Div8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Div8u",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Div16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Div16u",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Div32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Div32u",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Div64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Div64u",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Div128u",
		argLen:  3,
		generic: true,
	},
	{
		name:    "Mod8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Mod8u",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Mod16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Mod16u",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Mod32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Mod32u",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Mod64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Mod64u",
		argLen:  2,
		generic: true,
	},
	{
		name:        "And8",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "And16",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "And32",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "And64",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Or8",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Or16",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Or32",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Or64",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Xor8",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Xor16",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Xor32",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Xor64",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:    "Lsh8x8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Lsh8x16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Lsh8x32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Lsh8x64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Lsh16x8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Lsh16x16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Lsh16x32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Lsh16x64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Lsh32x8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Lsh32x16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Lsh32x32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Lsh32x64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Lsh64x8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Lsh64x16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Lsh64x32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Lsh64x64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh8x8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh8x16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh8x32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh8x64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh16x8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh16x16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh16x32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh16x64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh32x8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh32x16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh32x32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh32x64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh64x8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh64x16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh64x32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh64x64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh8Ux8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh8Ux16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh8Ux32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh8Ux64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh16Ux8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh16Ux16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh16Ux32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh16Ux64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh32Ux8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh32Ux16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh32Ux32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh32Ux64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh64Ux8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh64Ux16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh64Ux32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Rsh64Ux64",
		argLen:  2,
		generic: true,
	},
	{
		name:        "Eq8",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Eq16",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Eq32",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Eq64",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "EqPtr",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:    "EqInter",
		argLen:  2,
		generic: true,
	},
	{
		name:    "EqSlice",
		argLen:  2,
		generic: true,
	},
	{
		name:        "Eq32F",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Eq64F",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Neq8",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Neq16",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Neq32",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Neq64",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "NeqPtr",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:    "NeqInter",
		argLen:  2,
		generic: true,
	},
	{
		name:    "NeqSlice",
		argLen:  2,
		generic: true,
	},
	{
		name:        "Neq32F",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Neq64F",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:    "Less8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Less8U",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Less16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Less16U",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Less32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Less32U",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Less64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Less64U",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Less32F",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Less64F",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Leq8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Leq8U",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Leq16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Leq16U",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Leq32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Leq32U",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Leq64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Leq64U",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Leq32F",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Leq64F",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Greater8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Greater8U",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Greater16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Greater16U",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Greater32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Greater32U",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Greater64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Greater64U",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Greater32F",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Greater64F",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Geq8",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Geq8U",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Geq16",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Geq16U",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Geq32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Geq32U",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Geq64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Geq64U",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Geq32F",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Geq64F",
		argLen:  2,
		generic: true,
	},
	{
		name:        "AndB",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "OrB",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "EqB",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "NeqB",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:    "Not",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Neg8",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Neg16",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Neg32",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Neg64",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Neg32F",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Neg64F",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Com8",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Com16",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Com32",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Com64",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Ctz32",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Ctz64",
		argLen:  1,
		generic: true,
	},
	{
		name:    "BitLen32",
		argLen:  1,
		generic: true,
	},
	{
		name:    "BitLen64",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Bswap32",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Bswap64",
		argLen:  1,
		generic: true,
	},
	{
		name:    "BitRev8",
		argLen:  1,
		generic: true,
	},
	{
		name:    "BitRev16",
		argLen:  1,
		generic: true,
	},
	{
		name:    "BitRev32",
		argLen:  1,
		generic: true,
	},
	{
		name:    "BitRev64",
		argLen:  1,
		generic: true,
	},
	{
		name:    "PopCount8",
		argLen:  1,
		generic: true,
	},
	{
		name:    "PopCount16",
		argLen:  1,
		generic: true,
	},
	{
		name:    "PopCount32",
		argLen:  1,
		generic: true,
	},
	{
		name:    "PopCount64",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Sqrt",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Phi",
		argLen:  -1,
		generic: true,
	},
	{
		name:    "Copy",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Convert",
		argLen:  2,
		generic: true,
	},
	{
		name:    "ConstBool",
		auxType: auxBool,
		argLen:  0,
		generic: true,
	},
	{
		name:    "ConstString",
		auxType: auxString,
		argLen:  0,
		generic: true,
	},
	{
		name:    "ConstNil",
		argLen:  0,
		generic: true,
	},
	{
		name:    "Const8",
		auxType: auxInt8,
		argLen:  0,
		generic: true,
	},
	{
		name:    "Const16",
		auxType: auxInt16,
		argLen:  0,
		generic: true,
	},
	{
		name:    "Const32",
		auxType: auxInt32,
		argLen:  0,
		generic: true,
	},
	{
		name:    "Const64",
		auxType: auxInt64,
		argLen:  0,
		generic: true,
	},
	{
		name:    "Const32F",
		auxType: auxFloat32,
		argLen:  0,
		generic: true,
	},
	{
		name:    "Const64F",
		auxType: auxFloat64,
		argLen:  0,
		generic: true,
	},
	{
		name:    "ConstInterface",
		argLen:  0,
		generic: true,
	},
	{
		name:    "ConstSlice",
		argLen:  0,
		generic: true,
	},
	{
		name:    "InitMem",
		argLen:  0,
		generic: true,
	},
	{
		name:      "Arg",
		auxType:   auxSymOff,
		argLen:    0,
		symEffect: SymNone,
		generic:   true,
	},
	{
		name:      "Addr",
		auxType:   auxSym,
		argLen:    1,
		symEffect: SymAddr,
		generic:   true,
	},
	{
		name:    "SP",
		argLen:  0,
		generic: true,
	},
	{
		name:    "SB",
		argLen:  0,
		generic: true,
	},
	{
		name:    "Load",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Store",
		auxType: auxTyp,
		argLen:  3,
		generic: true,
	},
	{
		name:    "Move",
		auxType: auxTypSize,
		argLen:  3,
		generic: true,
	},
	{
		name:    "Zero",
		auxType: auxTypSize,
		argLen:  2,
		generic: true,
	},
	{
		name:    "StoreWB",
		auxType: auxTyp,
		argLen:  3,
		generic: true,
	},
	{
		name:    "MoveWB",
		auxType: auxTypSize,
		argLen:  3,
		generic: true,
	},
	{
		name:    "ZeroWB",
		auxType: auxTypSize,
		argLen:  2,
		generic: true,
	},
	{
		name:    "ClosureCall",
		auxType: auxInt64,
		argLen:  3,
		call:    true,
		generic: true,
	},
	{
		name:      "StaticCall",
		auxType:   auxSymOff,
		argLen:    1,
		call:      true,
		symEffect: SymNone,
		generic:   true,
	},
	{
		name:    "InterCall",
		auxType: auxInt64,
		argLen:  2,
		call:    true,
		generic: true,
	},
	{
		name:    "SignExt8to16",
		argLen:  1,
		generic: true,
	},
	{
		name:    "SignExt8to32",
		argLen:  1,
		generic: true,
	},
	{
		name:    "SignExt8to64",
		argLen:  1,
		generic: true,
	},
	{
		name:    "SignExt16to32",
		argLen:  1,
		generic: true,
	},
	{
		name:    "SignExt16to64",
		argLen:  1,
		generic: true,
	},
	{
		name:    "SignExt32to64",
		argLen:  1,
		generic: true,
	},
	{
		name:    "ZeroExt8to16",
		argLen:  1,
		generic: true,
	},
	{
		name:    "ZeroExt8to32",
		argLen:  1,
		generic: true,
	},
	{
		name:    "ZeroExt8to64",
		argLen:  1,
		generic: true,
	},
	{
		name:    "ZeroExt16to32",
		argLen:  1,
		generic: true,
	},
	{
		name:    "ZeroExt16to64",
		argLen:  1,
		generic: true,
	},
	{
		name:    "ZeroExt32to64",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Trunc16to8",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Trunc32to8",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Trunc32to16",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Trunc64to8",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Trunc64to16",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Trunc64to32",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Cvt32to32F",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Cvt32to64F",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Cvt64to32F",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Cvt64to64F",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Cvt32Fto32",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Cvt32Fto64",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Cvt64Fto32",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Cvt64Fto64",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Cvt32Fto64F",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Cvt64Fto32F",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Round32F",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Round64F",
		argLen:  1,
		generic: true,
	},
	{
		name:    "IsNonNil",
		argLen:  1,
		generic: true,
	},
	{
		name:    "IsInBounds",
		argLen:  2,
		generic: true,
	},
	{
		name:    "IsSliceInBounds",
		argLen:  2,
		generic: true,
	},
	{
		name:    "NilCheck",
		argLen:  2,
		generic: true,
	},
	{
		name:    "GetG",
		argLen:  1,
		generic: true,
	},
	{
		name:    "GetClosurePtr",
		argLen:  0,
		generic: true,
	},
	{
		name:    "PtrIndex",
		argLen:  2,
		generic: true,
	},
	{
		name:    "OffPtr",
		auxType: auxInt64,
		argLen:  1,
		generic: true,
	},
	{
		name:    "SliceMake",
		argLen:  3,
		generic: true,
	},
	{
		name:    "SlicePtr",
		argLen:  1,
		generic: true,
	},
	{
		name:    "SliceLen",
		argLen:  1,
		generic: true,
	},
	{
		name:    "SliceCap",
		argLen:  1,
		generic: true,
	},
	{
		name:    "ComplexMake",
		argLen:  2,
		generic: true,
	},
	{
		name:    "ComplexReal",
		argLen:  1,
		generic: true,
	},
	{
		name:    "ComplexImag",
		argLen:  1,
		generic: true,
	},
	{
		name:    "StringMake",
		argLen:  2,
		generic: true,
	},
	{
		name:    "StringPtr",
		argLen:  1,
		generic: true,
	},
	{
		name:    "StringLen",
		argLen:  1,
		generic: true,
	},
	{
		name:    "IMake",
		argLen:  2,
		generic: true,
	},
	{
		name:    "ITab",
		argLen:  1,
		generic: true,
	},
	{
		name:    "IData",
		argLen:  1,
		generic: true,
	},
	{
		name:    "StructMake0",
		argLen:  0,
		generic: true,
	},
	{
		name:    "StructMake1",
		argLen:  1,
		generic: true,
	},
	{
		name:    "StructMake2",
		argLen:  2,
		generic: true,
	},
	{
		name:    "StructMake3",
		argLen:  3,
		generic: true,
	},
	{
		name:    "StructMake4",
		argLen:  4,
		generic: true,
	},
	{
		name:    "StructSelect",
		auxType: auxInt64,
		argLen:  1,
		generic: true,
	},
	{
		name:    "ArrayMake0",
		argLen:  0,
		generic: true,
	},
	{
		name:    "ArrayMake1",
		argLen:  1,
		generic: true,
	},
	{
		name:    "ArraySelect",
		auxType: auxInt64,
		argLen:  1,
		generic: true,
	},
	{
		name:    "StoreReg",
		argLen:  1,
		generic: true,
	},
	{
		name:    "LoadReg",
		argLen:  1,
		generic: true,
	},
	{
		name:      "FwdRef",
		auxType:   auxSym,
		argLen:    0,
		symEffect: SymNone,
		generic:   true,
	},
	{
		name:    "Unknown",
		argLen:  0,
		generic: true,
	},
	{
		name:      "VarDef",
		auxType:   auxSym,
		argLen:    1,
		symEffect: SymNone,
		generic:   true,
	},
	{
		name:      "VarKill",
		auxType:   auxSym,
		argLen:    1,
		symEffect: SymNone,
		generic:   true,
	},
	{
		name:      "VarLive",
		auxType:   auxSym,
		argLen:    1,
		symEffect: SymNone,
		generic:   true,
	},
	{
		name:    "KeepAlive",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Int64Make",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Int64Hi",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Int64Lo",
		argLen:  1,
		generic: true,
	},
	{
		name:        "Add32carry",
		argLen:      2,
		commutative: true,
		generic:     true,
	},
	{
		name:        "Add32withcarry",
		argLen:      3,
		commutative: true,
		generic:     true,
	},
	{
		name:    "Sub32carry",
		argLen:  2,
		generic: true,
	},
	{
		name:    "Sub32withcarry",
		argLen:  3,
		generic: true,
	},
	{
		name:    "Signmask",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Zeromask",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Slicemask",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Cvt32Uto32F",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Cvt32Uto64F",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Cvt32Fto32U",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Cvt64Fto32U",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Cvt64Uto32F",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Cvt64Uto64F",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Cvt32Fto64U",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Cvt64Fto64U",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Select0",
		argLen:  1,
		generic: true,
	},
	{
		name:    "Select1",
		argLen:  1,
		generic: true,
	},
	{
		name:    "AtomicLoad32",
		argLen:  2,
		generic: true,
	},
	{
		name:    "AtomicLoad64",
		argLen:  2,
		generic: true,
	},
	{
		name:    "AtomicLoadPtr",
		argLen:  2,
		generic: true,
	},
	{
		name:           "AtomicStore32",
		argLen:         3,
		hasSideEffects: true,
		generic:        true,
	},
	{
		name:           "AtomicStore64",
		argLen:         3,
		hasSideEffects: true,
		generic:        true,
	},
	{
		name:           "AtomicStorePtrNoWB",
		argLen:         3,
		hasSideEffects: true,
		generic:        true,
	},
	{
		name:           "AtomicExchange32",
		argLen:         3,
		hasSideEffects: true,
		generic:        true,
	},
	{
		name:           "AtomicExchange64",
		argLen:         3,
		hasSideEffects: true,
		generic:        true,
	},
	{
		name:           "AtomicAdd32",
		argLen:         3,
		hasSideEffects: true,
		generic:        true,
	},
	{
		name:           "AtomicAdd64",
		argLen:         3,
		hasSideEffects: true,
		generic:        true,
	},
	{
		name:           "AtomicCompareAndSwap32",
		argLen:         4,
		hasSideEffects: true,
		generic:        true,
	},
	{
		name:           "AtomicCompareAndSwap64",
		argLen:         4,
		hasSideEffects: true,
		generic:        true,
	},
	{
		name:           "AtomicAnd8",
		argLen:         3,
		hasSideEffects: true,
		generic:        true,
	},
	{
		name:           "AtomicOr8",
		argLen:         3,
		hasSideEffects: true,
		generic:        true,
	},
	{
		name:      "Clobber",
		auxType:   auxSymOff,
		argLen:    0,
		symEffect: SymNone,
		generic:   true,
	},
}

func (o Op) Asm() obj.As          { return opcodeTable[o].asm }
func (o Op) String() string       { return opcodeTable[o].name }
func (o Op) UsesScratch() bool    { return opcodeTable[o].usesScratch }
func (o Op) SymEffect() SymEffect { return opcodeTable[o].symEffect }
func (o Op) IsCall() bool         { return opcodeTable[o].call }

var registers386 = [...]Register{
	{0, x86.REG_AX, "AX"},
	{1, x86.REG_CX, "CX"},
	{2, x86.REG_DX, "DX"},
	{3, x86.REG_BX, "BX"},
	{4, x86.REGSP, "SP"},
	{5, x86.REG_BP, "BP"},
	{6, x86.REG_SI, "SI"},
	{7, x86.REG_DI, "DI"},
	{8, x86.REG_X0, "X0"},
	{9, x86.REG_X1, "X1"},
	{10, x86.REG_X2, "X2"},
	{11, x86.REG_X3, "X3"},
	{12, x86.REG_X4, "X4"},
	{13, x86.REG_X5, "X5"},
	{14, x86.REG_X6, "X6"},
	{15, x86.REG_X7, "X7"},
	{16, 0, "SB"},
}
var gpRegMask386 = regMask(239)
var fpRegMask386 = regMask(65280)
var specialRegMask386 = regMask(0)
var framepointerReg386 = int8(5)
var linkReg386 = int8(-1)
var registersAMD64 = [...]Register{
	{0, x86.REG_AX, "AX"},
	{1, x86.REG_CX, "CX"},
	{2, x86.REG_DX, "DX"},
	{3, x86.REG_BX, "BX"},
	{4, x86.REGSP, "SP"},
	{5, x86.REG_BP, "BP"},
	{6, x86.REG_SI, "SI"},
	{7, x86.REG_DI, "DI"},
	{8, x86.REG_R8, "R8"},
	{9, x86.REG_R9, "R9"},
	{10, x86.REG_R10, "R10"},
	{11, x86.REG_R11, "R11"},
	{12, x86.REG_R12, "R12"},
	{13, x86.REG_R13, "R13"},
	{14, x86.REG_R14, "R14"},
	{15, x86.REG_R15, "R15"},
	{16, x86.REG_X0, "X0"},
	{17, x86.REG_X1, "X1"},
	{18, x86.REG_X2, "X2"},
	{19, x86.REG_X3, "X3"},
	{20, x86.REG_X4, "X4"},
	{21, x86.REG_X5, "X5"},
	{22, x86.REG_X6, "X6"},
	{23, x86.REG_X7, "X7"},
	{24, x86.REG_X8, "X8"},
	{25, x86.REG_X9, "X9"},
	{26, x86.REG_X10, "X10"},
	{27, x86.REG_X11, "X11"},
	{28, x86.REG_X12, "X12"},
	{29, x86.REG_X13, "X13"},
	{30, x86.REG_X14, "X14"},
	{31, x86.REG_X15, "X15"},
	{32, 0, "SB"},
}
var gpRegMaskAMD64 = regMask(65519)
var fpRegMaskAMD64 = regMask(4294901760)
var specialRegMaskAMD64 = regMask(0)
var framepointerRegAMD64 = int8(5)
var linkRegAMD64 = int8(-1)
var registersARM = [...]Register{
	{0, arm.REG_R0, "R0"},
	{1, arm.REG_R1, "R1"},
	{2, arm.REG_R2, "R2"},
	{3, arm.REG_R3, "R3"},
	{4, arm.REG_R4, "R4"},
	{5, arm.REG_R5, "R5"},
	{6, arm.REG_R6, "R6"},
	{7, arm.REG_R7, "R7"},
	{8, arm.REG_R8, "R8"},
	{9, arm.REG_R9, "R9"},
	{10, arm.REGG, "g"},
	{11, arm.REG_R11, "R11"},
	{12, arm.REG_R12, "R12"},
	{13, arm.REGSP, "SP"},
	{14, arm.REG_R14, "R14"},
	{15, arm.REG_R15, "R15"},
	{16, arm.REG_F0, "F0"},
	{17, arm.REG_F1, "F1"},
	{18, arm.REG_F2, "F2"},
	{19, arm.REG_F3, "F3"},
	{20, arm.REG_F4, "F4"},
	{21, arm.REG_F5, "F5"},
	{22, arm.REG_F6, "F6"},
	{23, arm.REG_F7, "F7"},
	{24, arm.REG_F8, "F8"},
	{25, arm.REG_F9, "F9"},
	{26, arm.REG_F10, "F10"},
	{27, arm.REG_F11, "F11"},
	{28, arm.REG_F12, "F12"},
	{29, arm.REG_F13, "F13"},
	{30, arm.REG_F14, "F14"},
	{31, arm.REG_F15, "F15"},
	{32, 0, "SB"},
}
var gpRegMaskARM = regMask(21503)
var fpRegMaskARM = regMask(4294901760)
var specialRegMaskARM = regMask(0)
var framepointerRegARM = int8(-1)
var linkRegARM = int8(14)
var registersARM64 = [...]Register{
	{0, arm64.REG_R0, "R0"},
	{1, arm64.REG_R1, "R1"},
	{2, arm64.REG_R2, "R2"},
	{3, arm64.REG_R3, "R3"},
	{4, arm64.REG_R4, "R4"},
	{5, arm64.REG_R5, "R5"},
	{6, arm64.REG_R6, "R6"},
	{7, arm64.REG_R7, "R7"},
	{8, arm64.REG_R8, "R8"},
	{9, arm64.REG_R9, "R9"},
	{10, arm64.REG_R10, "R10"},
	{11, arm64.REG_R11, "R11"},
	{12, arm64.REG_R12, "R12"},
	{13, arm64.REG_R13, "R13"},
	{14, arm64.REG_R14, "R14"},
	{15, arm64.REG_R15, "R15"},
	{16, arm64.REG_R16, "R16"},
	{17, arm64.REG_R17, "R17"},
	{18, arm64.REG_R18, "R18"},
	{19, arm64.REG_R19, "R19"},
	{20, arm64.REG_R20, "R20"},
	{21, arm64.REG_R21, "R21"},
	{22, arm64.REG_R22, "R22"},
	{23, arm64.REG_R23, "R23"},
	{24, arm64.REG_R24, "R24"},
	{25, arm64.REG_R25, "R25"},
	{26, arm64.REG_R26, "R26"},
	{27, arm64.REGG, "g"},
	{28, arm64.REG_R29, "R29"},
	{29, arm64.REG_R30, "R30"},
	{30, arm64.REGSP, "SP"},
	{31, arm64.REG_F0, "F0"},
	{32, arm64.REG_F1, "F1"},
	{33, arm64.REG_F2, "F2"},
	{34, arm64.REG_F3, "F3"},
	{35, arm64.REG_F4, "F4"},
	{36, arm64.REG_F5, "F5"},
	{37, arm64.REG_F6, "F6"},
	{38, arm64.REG_F7, "F7"},
	{39, arm64.REG_F8, "F8"},
	{40, arm64.REG_F9, "F9"},
	{41, arm64.REG_F10, "F10"},
	{42, arm64.REG_F11, "F11"},
	{43, arm64.REG_F12, "F12"},
	{44, arm64.REG_F13, "F13"},
	{45, arm64.REG_F14, "F14"},
	{46, arm64.REG_F15, "F15"},
	{47, arm64.REG_F16, "F16"},
	{48, arm64.REG_F17, "F17"},
	{49, arm64.REG_F18, "F18"},
	{50, arm64.REG_F19, "F19"},
	{51, arm64.REG_F20, "F20"},
	{52, arm64.REG_F21, "F21"},
	{53, arm64.REG_F22, "F22"},
	{54, arm64.REG_F23, "F23"},
	{55, arm64.REG_F24, "F24"},
	{56, arm64.REG_F25, "F25"},
	{57, arm64.REG_F26, "F26"},
	{58, arm64.REG_F27, "F27"},
	{59, arm64.REG_F28, "F28"},
	{60, arm64.REG_F29, "F29"},
	{61, arm64.REG_F30, "F30"},
	{62, arm64.REG_F31, "F31"},
	{63, 0, "SB"},
}
var gpRegMaskARM64 = regMask(670826495)
var fpRegMaskARM64 = regMask(9223372034707292160)
var specialRegMaskARM64 = regMask(0)
var framepointerRegARM64 = int8(-1)
var linkRegARM64 = int8(29)
var registersMIPS = [...]Register{
	{0, mips.REG_R0, "R0"},
	{1, mips.REG_R1, "R1"},
	{2, mips.REG_R2, "R2"},
	{3, mips.REG_R3, "R3"},
	{4, mips.REG_R4, "R4"},
	{5, mips.REG_R5, "R5"},
	{6, mips.REG_R6, "R6"},
	{7, mips.REG_R7, "R7"},
	{8, mips.REG_R8, "R8"},
	{9, mips.REG_R9, "R9"},
	{10, mips.REG_R10, "R10"},
	{11, mips.REG_R11, "R11"},
	{12, mips.REG_R12, "R12"},
	{13, mips.REG_R13, "R13"},
	{14, mips.REG_R14, "R14"},
	{15, mips.REG_R15, "R15"},
	{16, mips.REG_R16, "R16"},
	{17, mips.REG_R17, "R17"},
	{18, mips.REG_R18, "R18"},
	{19, mips.REG_R19, "R19"},
	{20, mips.REG_R20, "R20"},
	{21, mips.REG_R21, "R21"},
	{22, mips.REG_R22, "R22"},
	{23, mips.REG_R24, "R24"},
	{24, mips.REG_R25, "R25"},
	{25, mips.REG_R28, "R28"},
	{26, mips.REGSP, "SP"},
	{27, mips.REGG, "g"},
	{28, mips.REG_R31, "R31"},
	{29, mips.REG_F0, "F0"},
	{30, mips.REG_F2, "F2"},
	{31, mips.REG_F4, "F4"},
	{32, mips.REG_F6, "F6"},
	{33, mips.REG_F8, "F8"},
	{34, mips.REG_F10, "F10"},
	{35, mips.REG_F12, "F12"},
	{36, mips.REG_F14, "F14"},
	{37, mips.REG_F16, "F16"},
	{38, mips.REG_F18, "F18"},
	{39, mips.REG_F20, "F20"},
	{40, mips.REG_F22, "F22"},
	{41, mips.REG_F24, "F24"},
	{42, mips.REG_F26, "F26"},
	{43, mips.REG_F28, "F28"},
	{44, mips.REG_F30, "F30"},
	{45, mips.REG_HI, "HI"},
	{46, mips.REG_LO, "LO"},
	{47, 0, "SB"},
}
var gpRegMaskMIPS = regMask(335544318)
var fpRegMaskMIPS = regMask(35183835217920)
var specialRegMaskMIPS = regMask(105553116266496)
var framepointerRegMIPS = int8(-1)
var linkRegMIPS = int8(28)
var registersMIPS64 = [...]Register{
	{0, mips.REG_R0, "R0"},
	{1, mips.REG_R1, "R1"},
	{2, mips.REG_R2, "R2"},
	{3, mips.REG_R3, "R3"},
	{4, mips.REG_R4, "R4"},
	{5, mips.REG_R5, "R5"},
	{6, mips.REG_R6, "R6"},
	{7, mips.REG_R7, "R7"},
	{8, mips.REG_R8, "R8"},
	{9, mips.REG_R9, "R9"},
	{10, mips.REG_R10, "R10"},
	{11, mips.REG_R11, "R11"},
	{12, mips.REG_R12, "R12"},
	{13, mips.REG_R13, "R13"},
	{14, mips.REG_R14, "R14"},
	{15, mips.REG_R15, "R15"},
	{16, mips.REG_R16, "R16"},
	{17, mips.REG_R17, "R17"},
	{18, mips.REG_R18, "R18"},
	{19, mips.REG_R19, "R19"},
	{20, mips.REG_R20, "R20"},
	{21, mips.REG_R21, "R21"},
	{22, mips.REG_R22, "R22"},
	{23, mips.REG_R24, "R24"},
	{24, mips.REG_R25, "R25"},
	{25, mips.REGSP, "SP"},
	{26, mips.REGG, "g"},
	{27, mips.REG_R31, "R31"},
	{28, mips.REG_F0, "F0"},
	{29, mips.REG_F1, "F1"},
	{30, mips.REG_F2, "F2"},
	{31, mips.REG_F3, "F3"},
	{32, mips.REG_F4, "F4"},
	{33, mips.REG_F5, "F5"},
	{34, mips.REG_F6, "F6"},
	{35, mips.REG_F7, "F7"},
	{36, mips.REG_F8, "F8"},
	{37, mips.REG_F9, "F9"},
	{38, mips.REG_F10, "F10"},
	{39, mips.REG_F11, "F11"},
	{40, mips.REG_F12, "F12"},
	{41, mips.REG_F13, "F13"},
	{42, mips.REG_F14, "F14"},
	{43, mips.REG_F15, "F15"},
	{44, mips.REG_F16, "F16"},
	{45, mips.REG_F17, "F17"},
	{46, mips.REG_F18, "F18"},
	{47, mips.REG_F19, "F19"},
	{48, mips.REG_F20, "F20"},
	{49, mips.REG_F21, "F21"},
	{50, mips.REG_F22, "F22"},
	{51, mips.REG_F23, "F23"},
	{52, mips.REG_F24, "F24"},
	{53, mips.REG_F25, "F25"},
	{54, mips.REG_F26, "F26"},
	{55, mips.REG_F27, "F27"},
	{56, mips.REG_F28, "F28"},
	{57, mips.REG_F29, "F29"},
	{58, mips.REG_F30, "F30"},
	{59, mips.REG_F31, "F31"},
	{60, mips.REG_HI, "HI"},
	{61, mips.REG_LO, "LO"},
	{62, 0, "SB"},
}
var gpRegMaskMIPS64 = regMask(167772158)
var fpRegMaskMIPS64 = regMask(1152921504338411520)
var specialRegMaskMIPS64 = regMask(3458764513820540928)
var framepointerRegMIPS64 = int8(-1)
var linkRegMIPS64 = int8(27)
var registersPPC64 = [...]Register{
	{0, ppc64.REG_R0, "R0"},
	{1, ppc64.REGSP, "SP"},
	{2, 0, "SB"},
	{3, ppc64.REG_R3, "R3"},
	{4, ppc64.REG_R4, "R4"},
	{5, ppc64.REG_R5, "R5"},
	{6, ppc64.REG_R6, "R6"},
	{7, ppc64.REG_R7, "R7"},
	{8, ppc64.REG_R8, "R8"},
	{9, ppc64.REG_R9, "R9"},
	{10, ppc64.REG_R10, "R10"},
	{11, ppc64.REG_R11, "R11"},
	{12, ppc64.REG_R12, "R12"},
	{13, ppc64.REG_R13, "R13"},
	{14, ppc64.REG_R14, "R14"},
	{15, ppc64.REG_R15, "R15"},
	{16, ppc64.REG_R16, "R16"},
	{17, ppc64.REG_R17, "R17"},
	{18, ppc64.REG_R18, "R18"},
	{19, ppc64.REG_R19, "R19"},
	{20, ppc64.REG_R20, "R20"},
	{21, ppc64.REG_R21, "R21"},
	{22, ppc64.REG_R22, "R22"},
	{23, ppc64.REG_R23, "R23"},
	{24, ppc64.REG_R24, "R24"},
	{25, ppc64.REG_R25, "R25"},
	{26, ppc64.REG_R26, "R26"},
	{27, ppc64.REG_R27, "R27"},
	{28, ppc64.REG_R28, "R28"},
	{29, ppc64.REG_R29, "R29"},
	{30, ppc64.REGG, "g"},
	{31, ppc64.REG_R31, "R31"},
	{32, ppc64.REG_F0, "F0"},
	{33, ppc64.REG_F1, "F1"},
	{34, ppc64.REG_F2, "F2"},
	{35, ppc64.REG_F3, "F3"},
	{36, ppc64.REG_F4, "F4"},
	{37, ppc64.REG_F5, "F5"},
	{38, ppc64.REG_F6, "F6"},
	{39, ppc64.REG_F7, "F7"},
	{40, ppc64.REG_F8, "F8"},
	{41, ppc64.REG_F9, "F9"},
	{42, ppc64.REG_F10, "F10"},
	{43, ppc64.REG_F11, "F11"},
	{44, ppc64.REG_F12, "F12"},
	{45, ppc64.REG_F13, "F13"},
	{46, ppc64.REG_F14, "F14"},
	{47, ppc64.REG_F15, "F15"},
	{48, ppc64.REG_F16, "F16"},
	{49, ppc64.REG_F17, "F17"},
	{50, ppc64.REG_F18, "F18"},
	{51, ppc64.REG_F19, "F19"},
	{52, ppc64.REG_F20, "F20"},
	{53, ppc64.REG_F21, "F21"},
	{54, ppc64.REG_F22, "F22"},
	{55, ppc64.REG_F23, "F23"},
	{56, ppc64.REG_F24, "F24"},
	{57, ppc64.REG_F25, "F25"},
	{58, ppc64.REG_F26, "F26"},
	{59, ppc64.REG_F27, "F27"},
	{60, ppc64.REG_F28, "F28"},
	{61, ppc64.REG_F29, "F29"},
	{62, ppc64.REG_F30, "F30"},
	{63, ppc64.REG_F31, "F31"},
}
var gpRegMaskPPC64 = regMask(1073733624)
var fpRegMaskPPC64 = regMask(576460743713488896)
var specialRegMaskPPC64 = regMask(0)
var framepointerRegPPC64 = int8(1)
var linkRegPPC64 = int8(-1)
var registersS390X = [...]Register{
	{0, s390x.REG_R0, "R0"},
	{1, s390x.REG_R1, "R1"},
	{2, s390x.REG_R2, "R2"},
	{3, s390x.REG_R3, "R3"},
	{4, s390x.REG_R4, "R4"},
	{5, s390x.REG_R5, "R5"},
	{6, s390x.REG_R6, "R6"},
	{7, s390x.REG_R7, "R7"},
	{8, s390x.REG_R8, "R8"},
	{9, s390x.REG_R9, "R9"},
	{10, s390x.REG_R10, "R10"},
	{11, s390x.REG_R11, "R11"},
	{12, s390x.REG_R12, "R12"},
	{13, s390x.REGG, "g"},
	{14, s390x.REG_R14, "R14"},
	{15, s390x.REGSP, "SP"},
	{16, s390x.REG_F0, "F0"},
	{17, s390x.REG_F1, "F1"},
	{18, s390x.REG_F2, "F2"},
	{19, s390x.REG_F3, "F3"},
	{20, s390x.REG_F4, "F4"},
	{21, s390x.REG_F5, "F5"},
	{22, s390x.REG_F6, "F6"},
	{23, s390x.REG_F7, "F7"},
	{24, s390x.REG_F8, "F8"},
	{25, s390x.REG_F9, "F9"},
	{26, s390x.REG_F10, "F10"},
	{27, s390x.REG_F11, "F11"},
	{28, s390x.REG_F12, "F12"},
	{29, s390x.REG_F13, "F13"},
	{30, s390x.REG_F14, "F14"},
	{31, s390x.REG_F15, "F15"},
	{32, 0, "SB"},
}
var gpRegMaskS390X = regMask(21503)
var fpRegMaskS390X = regMask(4294901760)
var specialRegMaskS390X = regMask(0)
var framepointerRegS390X = int8(-1)
var linkRegS390X = int8(14)
