// Code generated from _gen/*Ops.go using 'go generate'; DO NOT EDIT.

package block

const (
	BlockInvalid BlockKind = iota

	Block386EQ
	Block386NE
	Block386LT
	Block386LE
	Block386GT
	Block386GE
	Block386OS
	Block386OC
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
	BlockAMD64OS
	BlockAMD64OC
	BlockAMD64ULT
	BlockAMD64ULE
	BlockAMD64UGT
	BlockAMD64UGE
	BlockAMD64EQF
	BlockAMD64NEF
	BlockAMD64ORD
	BlockAMD64NAN
	BlockAMD64JUMPTABLE

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
	BlockARMLTnoov
	BlockARMLEnoov
	BlockARMGTnoov
	BlockARMGEnoov

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
	BlockARM64TBZ
	BlockARM64TBNZ
	BlockARM64FLT
	BlockARM64FLE
	BlockARM64FGT
	BlockARM64FGE
	BlockARM64LTnoov
	BlockARM64LEnoov
	BlockARM64GTnoov
	BlockARM64GEnoov
	BlockARM64JUMPTABLE

	BlockLOONG64EQZ
	BlockLOONG64NEZ
	BlockLOONG64LTZ
	BlockLOONG64LEZ
	BlockLOONG64GTZ
	BlockLOONG64GEZ
	BlockLOONG64FPT
	BlockLOONG64FPF
	BlockLOONG64BEQ
	BlockLOONG64BNE
	BlockLOONG64BGE
	BlockLOONG64BLT
	BlockLOONG64BGEU
	BlockLOONG64BLTU
	BlockLOONG64JUMPTABLE

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

	BlockRISCV64BEQ
	BlockRISCV64BNE
	BlockRISCV64BLT
	BlockRISCV64BGE
	BlockRISCV64BLTU
	BlockRISCV64BGEU
	BlockRISCV64BEQZ
	BlockRISCV64BNEZ
	BlockRISCV64BLEZ
	BlockRISCV64BGEZ
	BlockRISCV64BLTZ
	BlockRISCV64BGTZ

	BlockS390XBRC
	BlockS390XCRJ
	BlockS390XCGRJ
	BlockS390XCLRJ
	BlockS390XCLGRJ
	BlockS390XCIJ
	BlockS390XCGIJ
	BlockS390XCLIJ
	BlockS390XCLGIJ

	BlockPlain
	BlockIf
	BlockDefer
	BlockRet
	BlockRetJmp
	BlockExit
	BlockJumpTable
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
	Block386OS:  "OS",
	Block386OC:  "OC",
	Block386ULT: "ULT",
	Block386ULE: "ULE",
	Block386UGT: "UGT",
	Block386UGE: "UGE",
	Block386EQF: "EQF",
	Block386NEF: "NEF",
	Block386ORD: "ORD",
	Block386NAN: "NAN",

	BlockAMD64EQ:        "EQ",
	BlockAMD64NE:        "NE",
	BlockAMD64LT:        "LT",
	BlockAMD64LE:        "LE",
	BlockAMD64GT:        "GT",
	BlockAMD64GE:        "GE",
	BlockAMD64OS:        "OS",
	BlockAMD64OC:        "OC",
	BlockAMD64ULT:       "ULT",
	BlockAMD64ULE:       "ULE",
	BlockAMD64UGT:       "UGT",
	BlockAMD64UGE:       "UGE",
	BlockAMD64EQF:       "EQF",
	BlockAMD64NEF:       "NEF",
	BlockAMD64ORD:       "ORD",
	BlockAMD64NAN:       "NAN",
	BlockAMD64JUMPTABLE: "JUMPTABLE",

	BlockARMEQ:     "EQ",
	BlockARMNE:     "NE",
	BlockARMLT:     "LT",
	BlockARMLE:     "LE",
	BlockARMGT:     "GT",
	BlockARMGE:     "GE",
	BlockARMULT:    "ULT",
	BlockARMULE:    "ULE",
	BlockARMUGT:    "UGT",
	BlockARMUGE:    "UGE",
	BlockARMLTnoov: "LTnoov",
	BlockARMLEnoov: "LEnoov",
	BlockARMGTnoov: "GTnoov",
	BlockARMGEnoov: "GEnoov",

	BlockARM64EQ:        "EQ",
	BlockARM64NE:        "NE",
	BlockARM64LT:        "LT",
	BlockARM64LE:        "LE",
	BlockARM64GT:        "GT",
	BlockARM64GE:        "GE",
	BlockARM64ULT:       "ULT",
	BlockARM64ULE:       "ULE",
	BlockARM64UGT:       "UGT",
	BlockARM64UGE:       "UGE",
	BlockARM64Z:         "Z",
	BlockARM64NZ:        "NZ",
	BlockARM64ZW:        "ZW",
	BlockARM64NZW:       "NZW",
	BlockARM64TBZ:       "TBZ",
	BlockARM64TBNZ:      "TBNZ",
	BlockARM64FLT:       "FLT",
	BlockARM64FLE:       "FLE",
	BlockARM64FGT:       "FGT",
	BlockARM64FGE:       "FGE",
	BlockARM64LTnoov:    "LTnoov",
	BlockARM64LEnoov:    "LEnoov",
	BlockARM64GTnoov:    "GTnoov",
	BlockARM64GEnoov:    "GEnoov",
	BlockARM64JUMPTABLE: "JUMPTABLE",

	BlockLOONG64EQZ:       "EQZ",
	BlockLOONG64NEZ:       "NEZ",
	BlockLOONG64LTZ:       "LTZ",
	BlockLOONG64LEZ:       "LEZ",
	BlockLOONG64GTZ:       "GTZ",
	BlockLOONG64GEZ:       "GEZ",
	BlockLOONG64FPT:       "FPT",
	BlockLOONG64FPF:       "FPF",
	BlockLOONG64BEQ:       "BEQ",
	BlockLOONG64BNE:       "BNE",
	BlockLOONG64BGE:       "BGE",
	BlockLOONG64BLT:       "BLT",
	BlockLOONG64BGEU:      "BGEU",
	BlockLOONG64BLTU:      "BLTU",
	BlockLOONG64JUMPTABLE: "JUMPTABLE",

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

	BlockRISCV64BEQ:  "BEQ",
	BlockRISCV64BNE:  "BNE",
	BlockRISCV64BLT:  "BLT",
	BlockRISCV64BGE:  "BGE",
	BlockRISCV64BLTU: "BLTU",
	BlockRISCV64BGEU: "BGEU",
	BlockRISCV64BEQZ: "BEQZ",
	BlockRISCV64BNEZ: "BNEZ",
	BlockRISCV64BLEZ: "BLEZ",
	BlockRISCV64BGEZ: "BGEZ",
	BlockRISCV64BLTZ: "BLTZ",
	BlockRISCV64BGTZ: "BGTZ",

	BlockS390XBRC:   "BRC",
	BlockS390XCRJ:   "CRJ",
	BlockS390XCGRJ:  "CGRJ",
	BlockS390XCLRJ:  "CLRJ",
	BlockS390XCLGRJ: "CLGRJ",
	BlockS390XCIJ:   "CIJ",
	BlockS390XCGIJ:  "CGIJ",
	BlockS390XCLIJ:  "CLIJ",
	BlockS390XCLGIJ: "CLGIJ",

	BlockPlain:     "Plain",
	BlockIf:        "If",
	BlockDefer:     "Defer",
	BlockRet:       "Ret",
	BlockRetJmp:    "RetJmp",
	BlockExit:      "Exit",
	BlockJumpTable: "JumpTable",
	BlockFirst:     "First",
}

func (k BlockKind) String() string { return blockString[k] }
func (k BlockKind) AuxIntType() string {
	switch k {
	case BlockARM64TBZ:
		return "int64"
	case BlockARM64TBNZ:
		return "int64"
	case BlockS390XCIJ:
		return "int8"
	case BlockS390XCGIJ:
		return "int8"
	case BlockS390XCLIJ:
		return "uint8"
	case BlockS390XCLGIJ:
		return "uint8"
	}
	return ""
}
