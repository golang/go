// Copyright 2024 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package s390xasm

// Instructions with extended mnemonics fall under various categories.
// To handle each of them in one single function, various different
// structure types are defined as below. Corresponding instruction
// structures are created with the help of these base structures.
// Different instruction types are as below:

// Typ1 - Instructions having different base and extended mnemonic strings.
//
//	These instructions have single M-field value and single offset.
type typ1ExtndMnics struct {
	BaseOpStr string
	Value     uint8
	Offset    uint8
	ExtnOpStr string
}

// Typ2 - Instructions having couple of extra strings added to the base mnemonic string,
//
//	depending on the condition code evaluation.
//	These instructions have single M-field value and single offset.
type typ2ExtndMnics struct {
	Value     uint8
	Offset    uint8
	ExtnOpStr string
}

// Typ3 - Instructions having couple of extra strings added to the base mnemonic string,
//
//	depending on the condition code evaluation.
//	These instructions have two M-field values and two offsets.
type typ3ExtndMnics struct {
	Value1    uint8
	Value2    uint8
	Offset1   uint8
	Offset2   uint8
	ExtnOpStr string
}

// Typ4 - Instructions having different base and extended mnemonic strings.
//
//	These instructions have two M-field values and two offsets.
type typ4ExtndMnics struct {
	BaseOpStr string
	Value1    uint8
	Value2    uint8
	Offset1   uint8
	Offset2   uint8
	ExtnOpStr string
}

// Typ5 - Instructions having different base and extended mnemonic strings.
//
//	These instructions have three M-field values and three offsets.
type typ5ExtndMnics struct {
	BaseOpStr string
	Value1    uint8
	Value2    uint8
	Value3    uint8
	Offset1   uint8
	Offset2   uint8
	Offset3   uint8
	ExtnOpStr string
}

// "func Handleextndmnemonic" - This is the function where the extended mnemonic logic
// is implemented. This function defines various structures to keep a list of base
// instructions and their extended mnemonic strings. These structure will also have
// M-field values and offset values defined, based on their type.
// HandleExtndMnemonic takes "inst" structure as the input variable.
// Inst structure will have all the details related to an instruction. Based on the
// opcode base string, a switch-case statement is executed. In that, based on the
// M-field value and the offset value of that particular M-field, extended mnemonic
// string is either searched or constructed by adding couple of extra strings to the base
// opcode string from one of the structure defined below.
func HandleExtndMnemonic(inst *Inst) string {

	brnchInstrExtndMnics := []typ1ExtndMnics{
		//BIC - BRANCH INDIRECT ON CONDITION instruction
		typ1ExtndMnics{BaseOpStr: "bic", Value: 1, Offset: 0, ExtnOpStr: "bio"},
		typ1ExtndMnics{BaseOpStr: "bic", Value: 2, Offset: 0, ExtnOpStr: "bih"},
		typ1ExtndMnics{BaseOpStr: "bic", Value: 4, Offset: 0, ExtnOpStr: "bil"},
		typ1ExtndMnics{BaseOpStr: "bic", Value: 7, Offset: 0, ExtnOpStr: "bine"},
		typ1ExtndMnics{BaseOpStr: "bic", Value: 8, Offset: 0, ExtnOpStr: "bie"},
		typ1ExtndMnics{BaseOpStr: "bic", Value: 11, Offset: 0, ExtnOpStr: "binl"},
		typ1ExtndMnics{BaseOpStr: "bic", Value: 13, Offset: 0, ExtnOpStr: "binh"},
		typ1ExtndMnics{BaseOpStr: "bic", Value: 14, Offset: 0, ExtnOpStr: "bino"},
		typ1ExtndMnics{BaseOpStr: "bic", Value: 15, Offset: 0, ExtnOpStr: "bi"},

		//BCR - BRANCH ON CONDITION instruction
		typ1ExtndMnics{BaseOpStr: "bcr", Value: 0, Offset: 0, ExtnOpStr: "nopr"},
		typ1ExtndMnics{BaseOpStr: "bcr", Value: 1, Offset: 0, ExtnOpStr: "bor"},
		typ1ExtndMnics{BaseOpStr: "bcr", Value: 2, Offset: 0, ExtnOpStr: "bhr"},
		typ1ExtndMnics{BaseOpStr: "bcr", Value: 4, Offset: 0, ExtnOpStr: "blr"},
		typ1ExtndMnics{BaseOpStr: "bcr", Value: 7, Offset: 0, ExtnOpStr: "bner"},
		typ1ExtndMnics{BaseOpStr: "bcr", Value: 8, Offset: 0, ExtnOpStr: "ber"},
		typ1ExtndMnics{BaseOpStr: "bcr", Value: 11, Offset: 0, ExtnOpStr: "bnlr"},
		typ1ExtndMnics{BaseOpStr: "bcr", Value: 13, Offset: 0, ExtnOpStr: "bnhr"},
		typ1ExtndMnics{BaseOpStr: "bcr", Value: 14, Offset: 0, ExtnOpStr: "bnor"},
		typ1ExtndMnics{BaseOpStr: "bcr", Value: 15, Offset: 0, ExtnOpStr: "br"},

		//BC - BRANCH ON CONDITION instruction
		typ1ExtndMnics{BaseOpStr: "bc", Value: 0, Offset: 0, ExtnOpStr: "nopr"},
		typ1ExtndMnics{BaseOpStr: "bc", Value: 1, Offset: 0, ExtnOpStr: "bo"},
		typ1ExtndMnics{BaseOpStr: "bc", Value: 2, Offset: 0, ExtnOpStr: "bh"},
		typ1ExtndMnics{BaseOpStr: "bc", Value: 4, Offset: 0, ExtnOpStr: "bl"},
		typ1ExtndMnics{BaseOpStr: "bc", Value: 7, Offset: 0, ExtnOpStr: "bne"},
		typ1ExtndMnics{BaseOpStr: "bc", Value: 8, Offset: 0, ExtnOpStr: "be"},
		typ1ExtndMnics{BaseOpStr: "bc", Value: 11, Offset: 0, ExtnOpStr: "bnl"},
		typ1ExtndMnics{BaseOpStr: "bc", Value: 13, Offset: 0, ExtnOpStr: "bnh"},
		typ1ExtndMnics{BaseOpStr: "bc", Value: 14, Offset: 0, ExtnOpStr: "bno"},
		typ1ExtndMnics{BaseOpStr: "bc", Value: 15, Offset: 0, ExtnOpStr: "b"},

		//BRC - BRANCH RELATIVE ON CONDITION instruction
		typ1ExtndMnics{BaseOpStr: "brc", Value: 0, Offset: 0, ExtnOpStr: "jnop"},
		typ1ExtndMnics{BaseOpStr: "brc", Value: 1, Offset: 0, ExtnOpStr: "jo"},
		typ1ExtndMnics{BaseOpStr: "brc", Value: 2, Offset: 0, ExtnOpStr: "jh"},
		typ1ExtndMnics{BaseOpStr: "brc", Value: 4, Offset: 0, ExtnOpStr: "jl"},
		typ1ExtndMnics{BaseOpStr: "brc", Value: 7, Offset: 0, ExtnOpStr: "jne"},
		typ1ExtndMnics{BaseOpStr: "brc", Value: 8, Offset: 0, ExtnOpStr: "je"},
		typ1ExtndMnics{BaseOpStr: "brc", Value: 11, Offset: 0, ExtnOpStr: "jnl"},
		typ1ExtndMnics{BaseOpStr: "brc", Value: 13, Offset: 0, ExtnOpStr: "jnh"},
		typ1ExtndMnics{BaseOpStr: "brc", Value: 14, Offset: 0, ExtnOpStr: "jno"},
		typ1ExtndMnics{BaseOpStr: "brc", Value: 15, Offset: 0, ExtnOpStr: "j"},

		//BRCL - BRANCH RELATIVE ON CONDITION LONG instruction
		typ1ExtndMnics{BaseOpStr: "brcl", Value: 0, Offset: 0, ExtnOpStr: "jgnop"},
		typ1ExtndMnics{BaseOpStr: "brcl", Value: 1, Offset: 0, ExtnOpStr: "jgo"},
		typ1ExtndMnics{BaseOpStr: "brcl", Value: 2, Offset: 0, ExtnOpStr: "jgh"},
		typ1ExtndMnics{BaseOpStr: "brcl", Value: 4, Offset: 0, ExtnOpStr: "jgl"},
		typ1ExtndMnics{BaseOpStr: "brcl", Value: 7, Offset: 0, ExtnOpStr: "jgne"},
		typ1ExtndMnics{BaseOpStr: "brcl", Value: 8, Offset: 0, ExtnOpStr: "jge"},
		typ1ExtndMnics{BaseOpStr: "brcl", Value: 11, Offset: 0, ExtnOpStr: "jgnl"},
		typ1ExtndMnics{BaseOpStr: "brcl", Value: 13, Offset: 0, ExtnOpStr: "jgnh"},
		typ1ExtndMnics{BaseOpStr: "brcl", Value: 14, Offset: 0, ExtnOpStr: "jgno"},
		typ1ExtndMnics{BaseOpStr: "brcl", Value: 15, Offset: 0, ExtnOpStr: "jg"},
	}

	//Compare instructions
	cmpInstrExtndMnics := []typ2ExtndMnics{
		typ2ExtndMnics{Value: 2, Offset: 2, ExtnOpStr: "h"},
		typ2ExtndMnics{Value: 4, Offset: 2, ExtnOpStr: "l"},
		typ2ExtndMnics{Value: 6, Offset: 2, ExtnOpStr: "ne"},
		typ2ExtndMnics{Value: 8, Offset: 2, ExtnOpStr: "e"},
		typ2ExtndMnics{Value: 10, Offset: 2, ExtnOpStr: "nl"},
		typ2ExtndMnics{Value: 12, Offset: 2, ExtnOpStr: "nh"},
	}

	//Load and Store instructions
	ldSt_InstrExtndMnics := []typ2ExtndMnics{
		typ2ExtndMnics{Value: 1, Offset: 2, ExtnOpStr: "o"},
		typ2ExtndMnics{Value: 2, Offset: 2, ExtnOpStr: "h"},
		typ2ExtndMnics{Value: 3, Offset: 2, ExtnOpStr: "nle"},
		typ2ExtndMnics{Value: 4, Offset: 2, ExtnOpStr: "l"},
		typ2ExtndMnics{Value: 5, Offset: 2, ExtnOpStr: "nhe"},
		typ2ExtndMnics{Value: 6, Offset: 2, ExtnOpStr: "lh"},
		typ2ExtndMnics{Value: 7, Offset: 2, ExtnOpStr: "ne"},
		typ2ExtndMnics{Value: 8, Offset: 2, ExtnOpStr: "e"},
		typ2ExtndMnics{Value: 9, Offset: 2, ExtnOpStr: "nlh"},
		typ2ExtndMnics{Value: 10, Offset: 2, ExtnOpStr: "he"},
		typ2ExtndMnics{Value: 11, Offset: 2, ExtnOpStr: "nl"},
		typ2ExtndMnics{Value: 12, Offset: 2, ExtnOpStr: "le"},
		typ2ExtndMnics{Value: 13, Offset: 2, ExtnOpStr: "nh"},
		typ2ExtndMnics{Value: 14, Offset: 2, ExtnOpStr: "no"},
	}

	vecInstrExtndMnics := []typ2ExtndMnics{
		typ2ExtndMnics{Value: 0, Offset: 3, ExtnOpStr: "b"},
		typ2ExtndMnics{Value: 1, Offset: 3, ExtnOpStr: "h"},
		typ2ExtndMnics{Value: 2, Offset: 3, ExtnOpStr: "f"},
		typ2ExtndMnics{Value: 3, Offset: 3, ExtnOpStr: "g"},
		typ2ExtndMnics{Value: 4, Offset: 3, ExtnOpStr: "q"},
		typ2ExtndMnics{Value: 6, Offset: 3, ExtnOpStr: "lf"},
	}

	//VCEQ, VCH, VCHL
	vec2InstrExtndMnics := []typ3ExtndMnics{
		typ3ExtndMnics{Value1: 0, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "b"},
		typ3ExtndMnics{Value1: 1, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "h"},
		typ3ExtndMnics{Value1: 2, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "f"},
		typ3ExtndMnics{Value1: 3, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "g"},
		typ3ExtndMnics{Value1: 0, Value2: 1, Offset1: 3, Offset2: 4, ExtnOpStr: "bs"},
		typ3ExtndMnics{Value1: 1, Value2: 1, Offset1: 3, Offset2: 4, ExtnOpStr: "hs"},
		typ3ExtndMnics{Value1: 2, Value2: 1, Offset1: 3, Offset2: 4, ExtnOpStr: "fs"},
		typ3ExtndMnics{Value1: 3, Value2: 1, Offset1: 3, Offset2: 4, ExtnOpStr: "gs"},
	}

	//VFAE, VFEE, VFENE
	vec21InstrExtndMnics := []typ3ExtndMnics{
		typ3ExtndMnics{Value1: 0, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "b"},
		typ3ExtndMnics{Value1: 1, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "h"},
		typ3ExtndMnics{Value1: 2, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "f"},
		typ3ExtndMnics{Value1: 0, Value2: 1, Offset1: 3, Offset2: 4, ExtnOpStr: "bs"},
		typ3ExtndMnics{Value1: 1, Value2: 1, Offset1: 3, Offset2: 4, ExtnOpStr: "hs"},
		typ3ExtndMnics{Value1: 2, Value2: 1, Offset1: 3, Offset2: 4, ExtnOpStr: "fs"},
		typ3ExtndMnics{Value1: 0, Value2: 2, Offset1: 3, Offset2: 4, ExtnOpStr: "zb"},
		typ3ExtndMnics{Value1: 1, Value2: 2, Offset1: 3, Offset2: 4, ExtnOpStr: "zh"},
		typ3ExtndMnics{Value1: 2, Value2: 2, Offset1: 3, Offset2: 4, ExtnOpStr: "zf"},
		typ3ExtndMnics{Value1: 0, Value2: 3, Offset1: 3, Offset2: 4, ExtnOpStr: "zbs"},
		typ3ExtndMnics{Value1: 1, Value2: 3, Offset1: 3, Offset2: 4, ExtnOpStr: "zhs"},
		typ3ExtndMnics{Value1: 2, Value2: 3, Offset1: 3, Offset2: 4, ExtnOpStr: "zfs"},
	}

	vec3InstrExtndMnics := []typ3ExtndMnics{
		typ3ExtndMnics{Value1: 2, Value2: 0, Offset1: 2, Offset2: 3, ExtnOpStr: "sb"},
		typ3ExtndMnics{Value1: 3, Value2: 0, Offset1: 2, Offset2: 3, ExtnOpStr: "db"},
		typ3ExtndMnics{Value1: 4, Value2: 0, Offset1: 2, Offset2: 3, ExtnOpStr: "xb"},
	}

	vec4InstrExtndMnics := []typ4ExtndMnics{
		// VFA - VECTOR FP ADD
		typ4ExtndMnics{BaseOpStr: "vfa", Value1: 2, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "vfasb"},
		typ4ExtndMnics{BaseOpStr: "vfa", Value1: 3, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "vfadb"},
		typ4ExtndMnics{BaseOpStr: "vfa", Value1: 2, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfasb"},
		typ4ExtndMnics{BaseOpStr: "vfa", Value1: 3, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfadb"},
		typ4ExtndMnics{BaseOpStr: "vfa", Value1: 4, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfaxb"},

		// VFD - VECTOR FP DIVIDE
		typ4ExtndMnics{BaseOpStr: "vfd", Value1: 2, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "vfdsb"},
		typ4ExtndMnics{BaseOpStr: "vfd", Value1: 3, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "vfddb"},
		typ4ExtndMnics{BaseOpStr: "vfd", Value1: 2, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfdsb"},
		typ4ExtndMnics{BaseOpStr: "vfd", Value1: 3, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfddb"},
		typ4ExtndMnics{BaseOpStr: "vfd", Value1: 4, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfdxb"},

		// VFLL - VECTOR FP LOAD LENGTHENED
		typ4ExtndMnics{BaseOpStr: "vfll", Value1: 2, Value2: 0, Offset1: 2, Offset2: 3, ExtnOpStr: "vflfs"},
		typ4ExtndMnics{BaseOpStr: "vfll", Value1: 2, Value2: 8, Offset1: 2, Offset2: 3, ExtnOpStr: "wflls"},
		typ4ExtndMnics{BaseOpStr: "vfll", Value1: 3, Value2: 8, Offset1: 2, Offset2: 3, ExtnOpStr: "wflld"},

		// VFMAX - VECTOR FP MAXIMUM
		typ4ExtndMnics{BaseOpStr: "vfmax", Value1: 2, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "vfmaxsb"},
		typ4ExtndMnics{BaseOpStr: "vfmax", Value1: 3, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "vfmaxdb"},
		typ4ExtndMnics{BaseOpStr: "vfmax", Value1: 2, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfmaxsb"},
		typ4ExtndMnics{BaseOpStr: "vfmax", Value1: 3, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfmaxdb"},
		typ4ExtndMnics{BaseOpStr: "vfmax", Value1: 4, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfmaxxb"},

		// VFMIN - VECTOR FP MINIMUM
		typ4ExtndMnics{BaseOpStr: "vfmin", Value1: 2, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "vfminsb"},
		typ4ExtndMnics{BaseOpStr: "vfmin", Value1: 3, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "vfmindb"},
		typ4ExtndMnics{BaseOpStr: "vfmin", Value1: 2, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfminsb"},
		typ4ExtndMnics{BaseOpStr: "vfmin", Value1: 3, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfmindb"},
		typ4ExtndMnics{BaseOpStr: "vfmin", Value1: 4, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfminxb"},

		// VFM - VECTOR FP MULTIPLY
		typ4ExtndMnics{BaseOpStr: "vfm", Value1: 2, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "vfmsb"},
		typ4ExtndMnics{BaseOpStr: "vfm", Value1: 3, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "vfmdb"},
		typ4ExtndMnics{BaseOpStr: "vfm", Value1: 2, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfmsb"},
		typ4ExtndMnics{BaseOpStr: "vfm", Value1: 3, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfmdb"},
		typ4ExtndMnics{BaseOpStr: "vfm", Value1: 4, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfmxb"},

		// VFSQ - VECTOR FP SQUARE ROOT
		typ4ExtndMnics{BaseOpStr: "vfsq", Value1: 2, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "vfsqsb"},
		typ4ExtndMnics{BaseOpStr: "vfsq", Value1: 3, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "vfsqdb"},
		typ4ExtndMnics{BaseOpStr: "vfsq", Value1: 2, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfsqsb"},
		typ4ExtndMnics{BaseOpStr: "vfsq", Value1: 3, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfsqdb"},
		typ4ExtndMnics{BaseOpStr: "vfsq", Value1: 4, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfsqxb"},

		// VFS - VECTOR FP SUBTRACT
		typ4ExtndMnics{BaseOpStr: "vfs", Value1: 2, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "vfssb"},
		typ4ExtndMnics{BaseOpStr: "vfs", Value1: 3, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "vfsdb"},
		typ4ExtndMnics{BaseOpStr: "vfs", Value1: 2, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfssb"},
		typ4ExtndMnics{BaseOpStr: "vfs", Value1: 3, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfsdb"},
		typ4ExtndMnics{BaseOpStr: "vfs", Value1: 4, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wfsxb"},

		// VFTCI - VECTOR FP TEST DATA CLASS IMMEDIATE
		typ4ExtndMnics{BaseOpStr: "vftci", Value1: 2, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "vftcisb"},
		typ4ExtndMnics{BaseOpStr: "vftci", Value1: 3, Value2: 0, Offset1: 3, Offset2: 4, ExtnOpStr: "vftcidb"},
		typ4ExtndMnics{BaseOpStr: "vftci", Value1: 2, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wftcisb"},
		typ4ExtndMnics{BaseOpStr: "vftci", Value1: 3, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wftcidb"},
		typ4ExtndMnics{BaseOpStr: "vftci", Value1: 4, Value2: 8, Offset1: 3, Offset2: 4, ExtnOpStr: "wftcixb"},
	}

	vec6InstrExtndMnics := []typ5ExtndMnics{
		// VFCE - VECTOR FP COMPARE EQUAL
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 2, Value2: 0, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfcesb"},
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 2, Value2: 0, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfcesbs"},
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 3, Value2: 0, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfcedb"},
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 2, Value2: 8, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfcesb"},
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 2, Value2: 8, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfcesbs"},
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 3, Value2: 8, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfcedb"},
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 3, Value2: 8, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfcedbs"},
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 4, Value2: 8, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfcexb"},
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 4, Value2: 8, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfcexbs"},
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 2, Value2: 4, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfkesb"},
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 2, Value2: 4, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfkesbs"},
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 3, Value2: 4, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfkedb"},
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 3, Value2: 4, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfkedbs"},
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 2, Value2: 12, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfkesb"},
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 2, Value2: 12, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfkesbs"},
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 3, Value2: 12, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfkedb"},
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 3, Value2: 12, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfkedbs"},
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 4, Value2: 12, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfkexb"},
		typ5ExtndMnics{BaseOpStr: "vfce", Value1: 4, Value2: 12, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfkexbs"},

		// VFCH - VECTOR FP COMPARE HIGH
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 2, Value2: 0, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfchsb"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 2, Value2: 0, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfchsbs"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 3, Value2: 0, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfchdb"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 3, Value2: 0, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfchdbs"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 2, Value2: 8, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfchsb"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 2, Value2: 8, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfchsbs"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 3, Value2: 8, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfchdb"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 3, Value2: 8, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfchdbs"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 4, Value2: 8, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfchxb"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 4, Value2: 8, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfchxbs"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 2, Value2: 4, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfkhsb"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 2, Value2: 4, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfkhsbs"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 3, Value2: 4, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfkhdb"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 3, Value2: 4, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfkhdbs"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 2, Value2: 12, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfkhsb"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 2, Value2: 12, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfkhsbs"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 3, Value2: 12, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfkhdb"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 3, Value2: 12, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfkhdbs"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 4, Value2: 12, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfkhxb"},
		typ5ExtndMnics{BaseOpStr: "vfch", Value1: 4, Value2: 12, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfkhxbs"},

		// VFCHE - VECTOR FP COMPARE HIGH OR EQUAL
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 2, Value2: 0, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfchesb"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 2, Value2: 0, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfchesbs"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 3, Value2: 0, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfchedb"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 3, Value2: 0, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfchedbs"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 2, Value2: 8, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfchesb"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 2, Value2: 8, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfchesbs"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 3, Value2: 8, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfchedb"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 3, Value2: 8, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfchedbs"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 4, Value2: 8, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfchexb"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 4, Value2: 8, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfchexbs"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 2, Value2: 4, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfkhesb"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 2, Value2: 4, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfkhesbs"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 3, Value2: 4, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfkhedb"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 3, Value2: 4, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "vfkhedbs"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 2, Value2: 12, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfkhesb"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 2, Value2: 12, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfkhesbs"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 3, Value2: 12, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfkhedb"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 3, Value2: 12, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfkhedbs"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 4, Value2: 12, Value3: 0, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfkhexb"},
		typ5ExtndMnics{BaseOpStr: "vfche", Value1: 4, Value2: 12, Value3: 1, Offset1: 3, Offset2: 4, Offset3: 5, ExtnOpStr: "wfkhexbs"},

		// VFPSO - VECTOR FP PERFORM SIGN OPERATION
		typ5ExtndMnics{BaseOpStr: "vfpso", Value1: 2, Value2: 0, Value3: 0, Offset1: 2, Offset2: 3, Offset3: 4, ExtnOpStr: "vflcsb"},
		typ5ExtndMnics{BaseOpStr: "vfpso", Value1: 2, Value2: 8, Value3: 0, Offset1: 2, Offset2: 3, Offset3: 4, ExtnOpStr: "wflcsb"},
		typ5ExtndMnics{BaseOpStr: "vfpso", Value1: 2, Value2: 0, Value3: 1, Offset1: 2, Offset2: 3, Offset3: 4, ExtnOpStr: "vflnsb"},
		typ5ExtndMnics{BaseOpStr: "vfpso", Value1: 2, Value2: 8, Value3: 1, Offset1: 2, Offset2: 3, Offset3: 4, ExtnOpStr: "wflnsb"},
		typ5ExtndMnics{BaseOpStr: "vfpso", Value1: 2, Value2: 0, Value3: 2, Offset1: 2, Offset2: 3, Offset3: 4, ExtnOpStr: "vflpsb"},
		typ5ExtndMnics{BaseOpStr: "vfpso", Value1: 2, Value2: 8, Value3: 2, Offset1: 2, Offset2: 3, Offset3: 4, ExtnOpStr: "wflpsb"},
		typ5ExtndMnics{BaseOpStr: "vfpso", Value1: 3, Value2: 0, Value3: 0, Offset1: 2, Offset2: 3, Offset3: 4, ExtnOpStr: "vflcdb"},
		typ5ExtndMnics{BaseOpStr: "vfpso", Value1: 3, Value2: 8, Value3: 0, Offset1: 2, Offset2: 3, Offset3: 4, ExtnOpStr: "wflcdb"},
		typ5ExtndMnics{BaseOpStr: "vfpso", Value1: 3, Value2: 0, Value3: 1, Offset1: 2, Offset2: 3, Offset3: 4, ExtnOpStr: "vflndb"},
		typ5ExtndMnics{BaseOpStr: "vfpso", Value1: 3, Value2: 8, Value3: 1, Offset1: 2, Offset2: 3, Offset3: 4, ExtnOpStr: "wflndb"},
		typ5ExtndMnics{BaseOpStr: "vfpso", Value1: 3, Value2: 0, Value3: 2, Offset1: 2, Offset2: 3, Offset3: 4, ExtnOpStr: "vflpdb"},
		typ5ExtndMnics{BaseOpStr: "vfpso", Value1: 3, Value2: 8, Value3: 2, Offset1: 2, Offset2: 3, Offset3: 4, ExtnOpStr: "wflpdb"},
		typ5ExtndMnics{BaseOpStr: "vfpso", Value1: 4, Value2: 8, Value3: 0, Offset1: 2, Offset2: 3, Offset3: 4, ExtnOpStr: "wflcxb"},
		typ5ExtndMnics{BaseOpStr: "vfpso", Value1: 4, Value2: 8, Value3: 1, Offset1: 2, Offset2: 3, Offset3: 4, ExtnOpStr: "wflnxb"},
		typ5ExtndMnics{BaseOpStr: "vfpso", Value1: 4, Value2: 8, Value3: 2, Offset1: 2, Offset2: 3, Offset3: 4, ExtnOpStr: "wflpxb"},
	}

	vec7InstrExtndMnics := []typ4ExtndMnics{
		// VFMA - VECTOR FP MULTIPLY AND ADD
		typ4ExtndMnics{BaseOpStr: "vfma", Value1: 0, Value2: 2, Offset1: 4, Offset2: 5, ExtnOpStr: "vfmasb"},
		typ4ExtndMnics{BaseOpStr: "vfma", Value1: 0, Value2: 3, Offset1: 4, Offset2: 5, ExtnOpStr: "vfmadb"},
		typ4ExtndMnics{BaseOpStr: "vfma", Value1: 8, Value2: 2, Offset1: 4, Offset2: 5, ExtnOpStr: "wfmasb"},
		typ4ExtndMnics{BaseOpStr: "vfma", Value1: 8, Value2: 3, Offset1: 4, Offset2: 5, ExtnOpStr: "wfmadb"},
		typ4ExtndMnics{BaseOpStr: "vfma", Value1: 8, Value2: 4, Offset1: 4, Offset2: 5, ExtnOpStr: "wfmaxb"},

		// VFMS - VECTOR FP MULTIPLY AND SUBTRACT
		typ4ExtndMnics{BaseOpStr: "vfms", Value1: 0, Value2: 2, Offset1: 4, Offset2: 5, ExtnOpStr: "vfmssb"},
		typ4ExtndMnics{BaseOpStr: "vfms", Value1: 0, Value2: 3, Offset1: 4, Offset2: 5, ExtnOpStr: "vfmsdb"},
		typ4ExtndMnics{BaseOpStr: "vfms", Value1: 8, Value2: 2, Offset1: 4, Offset2: 5, ExtnOpStr: "wfmssb"},
		typ4ExtndMnics{BaseOpStr: "vfms", Value1: 8, Value2: 3, Offset1: 4, Offset2: 5, ExtnOpStr: "wfmsdb"},
		typ4ExtndMnics{BaseOpStr: "vfms", Value1: 8, Value2: 4, Offset1: 4, Offset2: 5, ExtnOpStr: "wfmsxb"},

		// VFNMA - VECTOR FP NEGATIVE MULTIPLY AND ADD
		typ4ExtndMnics{BaseOpStr: "vfnma", Value1: 0, Value2: 2, Offset1: 4, Offset2: 5, ExtnOpStr: "vfnmasb"},
		typ4ExtndMnics{BaseOpStr: "vfnma", Value1: 0, Value2: 3, Offset1: 4, Offset2: 5, ExtnOpStr: "vfnmadb"},
		typ4ExtndMnics{BaseOpStr: "vfnma", Value1: 8, Value2: 2, Offset1: 4, Offset2: 5, ExtnOpStr: "wfnmasb"},
		typ4ExtndMnics{BaseOpStr: "vfnma", Value1: 8, Value2: 3, Offset1: 4, Offset2: 5, ExtnOpStr: "wfnmadb"},
		typ4ExtndMnics{BaseOpStr: "vfnma", Value1: 8, Value2: 4, Offset1: 4, Offset2: 5, ExtnOpStr: "wfnmaxb"},

		// VFNMS - VECTOR FP NEGATIVE MULTIPLY AND SUBTRACT
		typ4ExtndMnics{BaseOpStr: "vfnms", Value1: 0, Value2: 2, Offset1: 4, Offset2: 5, ExtnOpStr: "vfnmssb"},
		typ4ExtndMnics{BaseOpStr: "vfnms", Value1: 0, Value2: 3, Offset1: 4, Offset2: 5, ExtnOpStr: "vfnmsdb"},
		typ4ExtndMnics{BaseOpStr: "vfnms", Value1: 8, Value2: 2, Offset1: 4, Offset2: 5, ExtnOpStr: "wfnmssb"},
		typ4ExtndMnics{BaseOpStr: "vfnms", Value1: 8, Value2: 3, Offset1: 4, Offset2: 5, ExtnOpStr: "wfnmsdb"},
		typ4ExtndMnics{BaseOpStr: "vfnms", Value1: 8, Value2: 4, Offset1: 4, Offset2: 5, ExtnOpStr: "wfnmsxb"},
	}

	opString := inst.Op.String()
	newOpStr := opString

	if inst.Enc == 0 {
		return ".long 0x0"
	} else if inst.Op == 0 {
		return "error: unknown instruction"
	}

	switch opString {
	// Case to handle all "branch" instructions with one M-field operand
	case "bic", "bcr", "bc", "brc", "brcl":

		for i := 0; i < len(brnchInstrExtndMnics); i++ {
			if opString == brnchInstrExtndMnics[i].BaseOpStr &&
				uint8(inst.Args[brnchInstrExtndMnics[i].Offset].(Mask)) == brnchInstrExtndMnics[i].Value {
				newOpStr = brnchInstrExtndMnics[i].ExtnOpStr
				removeArg(inst, int8(brnchInstrExtndMnics[i].Offset))
				break
			}
		}

	// Case to handle all "compare" instructions with one M-field operand
	case "crb", "cgrb", "crj", "cgrj", "crt", "cgrt", "cib", "cgib", "cij", "cgij", "cit", "cgit", "clrb", "clgrb",
		"clrj", "clgrj", "clrt", "clgrt", "clt", "clgt", "clib", "clgib", "clij", "clgij", "clfit", "clgit":

		for i := 0; i < len(cmpInstrExtndMnics); i++ {
			//For CLT and CLGT instructions, M-value is the second operand.
			//Hence, set the offset to "1"
			if opString == "clt" || opString == "clgt" {
				cmpInstrExtndMnics[i].Offset = 1
			}

			if uint8(inst.Args[cmpInstrExtndMnics[i].Offset].(Mask)) == cmpInstrExtndMnics[i].Value {
				newOpStr = opString + cmpInstrExtndMnics[i].ExtnOpStr
				removeArg(inst, int8(cmpInstrExtndMnics[i].Offset))
				break
			}
		}

	// Case to handle all "load" and "store" instructions with one M-field operand
	case "lochhi", "lochi", "locghi", "locfhr", "locfh", "locr", "locgr", "loc",
		"locg", "selr", "selgr", "selfhr", "stocfh", "stoc", "stocg":

		for i := 0; i < len(ldSt_InstrExtndMnics); i++ {

			//For LOCFH, LOC, LOCG, SELR, SELGR, SELFHR, STOCFH, STOC, STOCG instructions,
			//M-value is the forth operand. Hence, set the offset to "3"
			if opString == "locfh" || opString == "loc" || opString == "locg" || opString == "selr" || opString == "selgr" ||
				opString == "selfhr" || opString == "stocfh" || opString == "stoc" || opString == "stocg" {
				ldSt_InstrExtndMnics[i].Offset = 3
			}

			if uint8(inst.Args[ldSt_InstrExtndMnics[i].Offset].(Mask)) == ldSt_InstrExtndMnics[i].Value {
				newOpStr = opString + ldSt_InstrExtndMnics[i].ExtnOpStr
				removeArg(inst, int8(ldSt_InstrExtndMnics[i].Offset))
				break
			}
		}

	// Case to handle all "vector" instructions with one M-field operand
	case "vavg", "vavgl", "verllv", "veslv", "vesrav", "vesrlv", "vgfm", "vgm", "vmx", "vmxl", "vmrh", "vmrl", "vmn", "vmnl", "vrep",
		"vclz", "vctz", "vec", "vecl", "vlc", "vlp", "vpopct", "vrepi", "verim", "verll", "vesl", "vesra", "vesrl", "vgfma", "vlrep",
		"vlgv", "vlvg", "vlbrrep", "vler", "vlbr", "vstbr", "vster", "vpk", "vme", "vmh", "vmle", "vmlh", "vmlo", "vml", "vmo", "vmae",
		"vmale", "vmalo", "vmal", "vmah", "vmalh", "vmao", "vmph", "vmplh", "vupl", "vupll", "vscbi", "vs", "vsum", "vsumg", "vsumq",
		"va", "vacc":

		switch opString {

		case "vavg", "vavgl", "verllv", "veslv", "vesrav", "vesrlv", "vgfm", "vgm", "vmx", "vmxl", "vmrh", "vmrl", "vmn", "vmnl", "vrep":
			//M-field is 3rd arg for all these instructions. Hence, set the offset to "2"
			for i := 0; i < len(vecInstrExtndMnics)-2; i++ { // 0,1,2,3
				if uint8(inst.Args[vecInstrExtndMnics[i].Offset].(Mask)) == vecInstrExtndMnics[i].Value {
					newOpStr = opString + vecInstrExtndMnics[i].ExtnOpStr
					removeArg(inst, int8(vecInstrExtndMnics[i].Offset))
					break
				}
			}

		case "vclz", "vctz", "vec", "vecl", "vlc", "vlp", "vpopct", "vrepi":
			for i := 0; i < len(vecInstrExtndMnics)-2; i++ { //0,1,2,3
				if uint8(inst.Args[vecInstrExtndMnics[i].Offset-1].(Mask)) == vecInstrExtndMnics[i].Value {
					newOpStr = opString + vecInstrExtndMnics[i].ExtnOpStr
					removeArg(inst, int8(vecInstrExtndMnics[i].Offset-1))
					break
				}
			}

		case "verim", "verll", "vesl", "vesra", "vesrl", "vgfma", "vlrep":
			for i := 0; i < len(vecInstrExtndMnics)-2; i++ { //0,1,2,3
				if uint8(inst.Args[vecInstrExtndMnics[i].Offset+1].(Mask)) == vecInstrExtndMnics[i].Value {
					newOpStr = opString + vecInstrExtndMnics[i].ExtnOpStr
					removeArg(inst, int8(vecInstrExtndMnics[i].Offset+1))
					break
				}
			}

		case "vlgv", "vlvg":
			for i := 0; i < len(vecInstrExtndMnics)-2; i++ {
				if uint8(inst.Args[vecInstrExtndMnics[i].Offset+1].(Mask)) == vecInstrExtndMnics[i].Value {
					newOpStr = opString + vecInstrExtndMnics[i].ExtnOpStr
					removeArg(inst, int8(vecInstrExtndMnics[i].Offset+1))
					break
				}
			}

		case "vlbrrep", "vler", "vster":
			for i := 1; i < len(vecInstrExtndMnics)-2; i++ {
				if uint8(inst.Args[vecInstrExtndMnics[i].Offset+1].(Mask)) == vecInstrExtndMnics[i].Value {
					newOpStr = opString + vecInstrExtndMnics[i].ExtnOpStr
					removeArg(inst, int8(vecInstrExtndMnics[i].Offset+1))
					break
				}
			}

		case "vpk":
			for i := 1; i < len(vecInstrExtndMnics)-2; i++ {
				if uint8(inst.Args[vecInstrExtndMnics[i].Offset].(Mask)) == vecInstrExtndMnics[i].Value {
					newOpStr = opString + vecInstrExtndMnics[i].ExtnOpStr
					removeArg(inst, int8(vecInstrExtndMnics[i].Offset))
					break
				}
			}

		case "vlbr", "vstbr":
			for i := 1; i < len(vecInstrExtndMnics)-1; i++ {
				if uint8(inst.Args[vecInstrExtndMnics[i].Offset+1].(Mask)) == vecInstrExtndMnics[i].Value {
					newOpStr = opString + vecInstrExtndMnics[i].ExtnOpStr
					removeArg(inst, int8(vecInstrExtndMnics[i].Offset+1))
					break
				}
			}
		case "vme", "vmh", "vmle", "vmlh", "vmlo", "vmo":
			for i := 0; i < len(vecInstrExtndMnics)-3; i++ { //0,1,2
				if uint8(inst.Args[vecInstrExtndMnics[i].Offset].(Mask)) == vecInstrExtndMnics[i].Value {
					newOpStr = opString + vecInstrExtndMnics[i].ExtnOpStr
					removeArg(inst, int8(vecInstrExtndMnics[i].Offset))
					break
				}
			}

		case "vml":
			for i := 0; i < len(vecInstrExtndMnics)-3; i++ { //0,1,2
				if uint8(inst.Args[vecInstrExtndMnics[i].Offset].(Mask)) == vecInstrExtndMnics[i].Value {
					if uint8(inst.Args[vecInstrExtndMnics[i].Offset].(Mask)) == 1 {
						newOpStr = opString + string("hw")
					} else {
						newOpStr = opString + vecInstrExtndMnics[i].ExtnOpStr
					}
					removeArg(inst, int8(vecInstrExtndMnics[i].Offset))
					break
				}
			}

		case "vmae", "vmale", "vmalo", "vmal", "vmah", "vmalh", "vmao":
			for i := 0; i < len(vecInstrExtndMnics)-3; i++ { //0,1,2
				if uint8(inst.Args[vecInstrExtndMnics[i].Offset+1].(Mask)) == vecInstrExtndMnics[i].Value {
					newOpStr = opString + vecInstrExtndMnics[i].ExtnOpStr
					removeArg(inst, int8(vecInstrExtndMnics[i].Offset+1))
					break
				}
			}

		case "vmph", "vmplh", "vupl", "vupll": //0,1,2
			for i := 0; i < len(vecInstrExtndMnics)-3; i++ {
				if uint8(inst.Args[vecInstrExtndMnics[i].Offset-1].(Mask)) == vecInstrExtndMnics[i].Value {
					newOpStr = opString + vecInstrExtndMnics[i].ExtnOpStr
					removeArg(inst, int8(vecInstrExtndMnics[i].Offset-1))
					break
				}
			}

		case "vscbi", "vs", "va", "vacc": // 0,1,2,3,4
			for i := 0; i < len(vecInstrExtndMnics)-1; i++ {
				if uint8(inst.Args[vecInstrExtndMnics[i].Offset].(Mask)) == vecInstrExtndMnics[i].Value {
					newOpStr = opString + vecInstrExtndMnics[i].ExtnOpStr
					removeArg(inst, int8(vecInstrExtndMnics[i].Offset))
					break
				}
			}
		case "vsum", "vsumg":
			for i := 1; i < len(vecInstrExtndMnics)-4; i++ {
				if uint8(inst.Args[vecInstrExtndMnics[i].Offset].(Mask)) == vecInstrExtndMnics[i].Value {
					newOpStr = opString + vecInstrExtndMnics[i].ExtnOpStr
					removeArg(inst, int8(vecInstrExtndMnics[i].Offset))
					break
				}
			}
		case "vsumq":
			for i := 2; i < len(vecInstrExtndMnics)-2; i++ {
				if uint8(inst.Args[vecInstrExtndMnics[i].Offset].(Mask)) == vecInstrExtndMnics[i].Value {
					newOpStr = opString + vecInstrExtndMnics[i].ExtnOpStr
					removeArg(inst, int8(vecInstrExtndMnics[i].Offset))
					break
				}
			}
		}

	case "vllez":
		for i := 0; i < len(vecInstrExtndMnics); i++ {
			if i == 4 {
				continue
			}
			if uint8(inst.Args[vecInstrExtndMnics[i].Offset+1].(Mask)) == vecInstrExtndMnics[i].Value {
				newOpStr = opString + vecInstrExtndMnics[i].ExtnOpStr
				removeArg(inst, int8(vecInstrExtndMnics[i].Offset+1))
				break
			}
		}

	case "vgbm":
		if uint16(inst.Args[1].(Imm)) == uint16(0) {
			newOpStr = "vzeo"
			removeArg(inst, int8(1))
		} else if uint16(inst.Args[1].(Imm)) == uint16(0xFFFF) {
			newOpStr = "vone"
			removeArg(inst, int8(1))
		}
	case "vno":
		if uint8(inst.Args[1].(VReg)) == uint8(inst.Args[2].(VReg)) { //Bitwise Not instruction(VNOT)  if V2 equal to v3
			newOpStr = opString + "t"
			removeArg(inst, int8(2))
		}

	case "vmsl":
		if uint8(inst.Args[4].(Mask)) == uint8(3) {
			newOpStr = opString + "g"
			removeArg(inst, int8(4))
		}

	case "vflr":
		if uint8(inst.Args[2].(Mask)) == uint8(3) && ((inst.Args[3].(Mask)>>3)&0x1 == 0x1) {
			inst.Args[3] = (inst.Args[3].(Mask) ^ 0x8)
			newOpStr = "wflrd"
			removeArg(inst, int8(2))
		} else if uint8(inst.Args[2].(Mask)) == uint8(4) && ((inst.Args[3].(Mask)>>3)&0x1 == 0x1) {
			inst.Args[3] = (inst.Args[3].(Mask) ^ 0x8)
			newOpStr = "wflrx"
			removeArg(inst, int8(2))
		} else if uint8(inst.Args[2].(Mask)) == uint8(3) {
			newOpStr = "vflrd"
			removeArg(inst, int8(2))
		}

	case "vllebrz":
		if uint8(inst.Args[4].(Mask)) == uint8(1) {
			newOpStr = opString + "h"
			removeArg(inst, int8(4))
		} else if uint8(inst.Args[4].(Mask)) == uint8(2) {
			newOpStr = opString + "f"
			removeArg(inst, int8(4))
		} else if uint8(inst.Args[4].(Mask)) == uint8(3) {
			newOpStr = "ldrv"
			removeArg(inst, int8(4))
		} else if uint8(inst.Args[4].(Mask)) == uint8(6) {
			newOpStr = "lerv"
			removeArg(inst, int8(4))
		}

	case "vschp":
		if uint8(inst.Args[3].(Mask)) == uint8(2) {
			newOpStr = "vschsp"
			removeArg(inst, int8(3))
		} else if uint8(inst.Args[3].(Mask)) == uint8(3) {
			newOpStr = "vschdp"
			removeArg(inst, int8(3))
		} else if uint8(inst.Args[3].(Mask)) == uint8(4) {
			newOpStr = "vschxp"
			removeArg(inst, int8(3))
		}

	case "vsbcbi", "vsbi":
		if uint8(inst.Args[4].(Mask)) == uint8(4) {
			newOpStr = opString + vecInstrExtndMnics[4].ExtnOpStr
			removeArg(inst, int8(4))
		}

	case "vac", "vaccc":
		if uint8(inst.Args[4].(Mask)) == uint8(4) {
			newOpStr = opString + vecInstrExtndMnics[3].ExtnOpStr
			removeArg(inst, int8(3))
		}

	case "vceq", "vch", "vchl":
		for i := 0; i < len(vec2InstrExtndMnics)-6; i++ {
			if uint8(inst.Args[vec2InstrExtndMnics[i].Offset1].(Mask)) == vec2InstrExtndMnics[i].Value1 &&
				uint8(inst.Args[vec2InstrExtndMnics[i].Offset2].(Mask)) == vec2InstrExtndMnics[i].Value2 {
				newOpStr = opString + vec2InstrExtndMnics[i].ExtnOpStr
				removeArg(inst, int8(vec2InstrExtndMnics[i].Offset1))
				removeArg(inst, int8(vec2InstrExtndMnics[i].Offset2-1))
				break
			}
		}

	case "vpks", "vpkls":
		for i := 1; i < len(vec2InstrExtndMnics)-6; i++ {
			if i == 4 {
				continue
			}
			if uint8(inst.Args[vec2InstrExtndMnics[i].Offset1].(Mask)) == vec2InstrExtndMnics[i].Value1 &&
				uint8(inst.Args[vec2InstrExtndMnics[i].Offset2].(Mask)) == vec2InstrExtndMnics[i].Value2 {
				newOpStr = opString + vec2InstrExtndMnics[i].ExtnOpStr
				removeArg(inst, int8(vec2InstrExtndMnics[i].Offset1))
				removeArg(inst, int8(vec2InstrExtndMnics[i].Offset2-1))
				break
			}
		}
	case "vfee", "vfene":
		var check bool
		for i := 0; i < len(vec21InstrExtndMnics); i++ {
			if uint8(inst.Args[vec21InstrExtndMnics[i].Offset1].(Mask)) == vec21InstrExtndMnics[i].Value1 &&
				uint8(inst.Args[vec21InstrExtndMnics[i].Offset2].(Mask)) == vec21InstrExtndMnics[i].Value2 {
				newOpStr = opString + vec21InstrExtndMnics[i].ExtnOpStr
				removeArg(inst, int8(vec21InstrExtndMnics[i].Offset1))
				removeArg(inst, int8(vec21InstrExtndMnics[i].Offset2-1))
				check = true
				break
			}
		}
		if !check {
			if uint8(inst.Args[3].(Mask)) == 0 && (uint8(inst.Args[4].(Mask)) != uint8(0)) {
				newOpStr = opString + vec21InstrExtndMnics[0].ExtnOpStr
				removeArg(inst, int8(vec21InstrExtndMnics[0].Offset1))
			} else if uint8(inst.Args[3].(Mask)) == 1 && (uint8(inst.Args[4].(Mask)) != uint8(0)) {
				newOpStr = opString + vec21InstrExtndMnics[1].ExtnOpStr
				removeArg(inst, int8(vec21InstrExtndMnics[1].Offset1))
			} else if uint8(inst.Args[3].(Mask)) == 2 && (uint8(inst.Args[4].(Mask)) != uint8(0)) {
				newOpStr = opString + vec21InstrExtndMnics[2].ExtnOpStr
				removeArg(inst, int8(vec21InstrExtndMnics[2].Offset1))
			} else if uint8(inst.Args[4].(Mask)) == 0 {
				removeArg(inst, int8(vec21InstrExtndMnics[2].Offset2))
			}
		}

	case "vfae", "vstrc":
		off := uint8(0)
		var check bool
		if opString == "vstrc" {
			off = uint8(1)
		}
		for i := 0; i < len(vec21InstrExtndMnics)-9; i++ {
			if uint8(inst.Args[vec21InstrExtndMnics[i].Offset1+off].(Mask)) == vec21InstrExtndMnics[i].Value1 &&
				uint8(inst.Args[vec21InstrExtndMnics[i].Offset2+off].(Mask)) == vec21InstrExtndMnics[i].Value2 {
				newOpStr = opString + vec21InstrExtndMnics[i].ExtnOpStr
				removeArg(inst, int8(vec21InstrExtndMnics[i].Offset1+off))
				removeArg(inst, int8(vec21InstrExtndMnics[i].Offset2+off-1))
				check = true
				break
			}
		}

		for i := 0; !(check) && (i < len(vec21InstrExtndMnics)-9); i++ {
			if uint8(inst.Args[vec21InstrExtndMnics[i].Offset1+off].(Mask)) == vec21InstrExtndMnics[i].Value1 &&
				uint8(inst.Args[vec21InstrExtndMnics[i].Offset2+off].(Mask)) == vec21InstrExtndMnics[i].Value2 {
				newOpStr = opString + vec21InstrExtndMnics[i].ExtnOpStr
				removeArg(inst, int8(vec21InstrExtndMnics[i].Offset1+off))
				removeArg(inst, int8(vec21InstrExtndMnics[i].Offset2+off-1))
				check = true
				break
			}
		}
		//for i := 3; !(check) && (i < len(vec21InstrExtndMnics)); i++ {
		for i := len(vec21InstrExtndMnics) - 1; !(check) && (i > 2); i-- {
			if uint8(inst.Args[vec21InstrExtndMnics[i].Offset1+off].(Mask)) == vec21InstrExtndMnics[i].Value1 &&
				uint8(inst.Args[vec21InstrExtndMnics[i].Offset2+off].(Mask))&(vec21InstrExtndMnics[i].Value2) == vec21InstrExtndMnics[i].Value2 {
				x := uint8(inst.Args[vec21InstrExtndMnics[i].Offset2+off].(Mask)) ^ (vec21InstrExtndMnics[i].Value2)
				newOpStr = opString + vec21InstrExtndMnics[i].ExtnOpStr
				if x != 0 {
					inst.Args[vec21InstrExtndMnics[i].Offset2+off] = Mask(x)
					removeArg(inst, int8(vec21InstrExtndMnics[i].Offset1+off))
					check = true
					break
				} else {
					removeArg(inst, int8(vec21InstrExtndMnics[i].Offset1+off))
					removeArg(inst, int8(vec21InstrExtndMnics[i].Offset2+off-1))
					check = true
					break
				}
			}
		}
		if !check && inst.Args[4+off].(Mask) == Mask(0) {
			removeArg(inst, int8(4+off))
			break
		}

	case "vstrs":
		var check bool
		for i := 0; i < len(vec21InstrExtndMnics)-3; i++ {
			if uint8(inst.Args[vec21InstrExtndMnics[i].Offset1+1].(Mask)) == vec21InstrExtndMnics[i].Value1 &&
				uint8(inst.Args[vec21InstrExtndMnics[i].Offset2+1].(Mask)) == vec21InstrExtndMnics[i].Value2 {
				newOpStr = opString + vec21InstrExtndMnics[i].ExtnOpStr
				removeArg(inst, int8(vec21InstrExtndMnics[i].Offset1+1))
				removeArg(inst, int8(vec21InstrExtndMnics[i].Offset2))
				check = true
				break
			}
			if i == 2 {
				i = i + 3
			}
		}

		for i := 0; !(check) && (i < len(vec21InstrExtndMnics)-9); i++ {
			if uint8(inst.Args[vec21InstrExtndMnics[i].Offset1+1].(Mask)) == vec21InstrExtndMnics[i].Value1 &&
				uint8(inst.Args[vec21InstrExtndMnics[i].Offset2+1].(Mask)) != 0 {
				newOpStr = opString + vec21InstrExtndMnics[i].ExtnOpStr
				removeArg(inst, int8(vec21InstrExtndMnics[i].Offset1+1))
				break
			}
		}

	case "vistr":
		var check bool
		for i := 0; i < len(vec21InstrExtndMnics)-6; i++ {
			if uint8(inst.Args[vec21InstrExtndMnics[i].Offset1-1].(Mask)) == vec21InstrExtndMnics[i].Value1 &&
				uint8(inst.Args[vec21InstrExtndMnics[i].Offset2-1].(Mask)) == vec21InstrExtndMnics[i].Value2 {
				newOpStr = opString + vec21InstrExtndMnics[i].ExtnOpStr
				removeArg(inst, int8(vec21InstrExtndMnics[i].Offset1-1))
				removeArg(inst, int8(vec21InstrExtndMnics[i].Offset2-2))
				check = true
				break
			}
		}

		for i := 0; !(check) && (i < len(vec21InstrExtndMnics)-9); i++ {
			if uint8(inst.Args[vec21InstrExtndMnics[i].Offset1-1].(Mask)) == vec21InstrExtndMnics[i].Value1 &&
				uint8(inst.Args[vec21InstrExtndMnics[i].Offset2-1].(Mask)) != 0 {
				newOpStr = opString + vec21InstrExtndMnics[i].ExtnOpStr
				removeArg(inst, int8(vec21InstrExtndMnics[i].Offset1-1))
				break
			}
		}

		if uint8(inst.Args[3].(Mask)) == 0 {
			removeArg(inst, int8(3))
			break
		}

	case "vcfps":
		if inst.Args[2].(Mask) == Mask(2) && ((inst.Args[3].(Mask)>>3)&(0x1) == 1) {
			inst.Args[3] = Mask((inst.Args[3].(Mask)) ^ (0x8))
			newOpStr = "wcefb"
			removeArg(inst, int8(2))
			break
		} else if inst.Args[2].(Mask) == Mask(3) && ((inst.Args[3].(Mask)>>3)&(0x1) == 1) {
			inst.Args[3] = Mask((inst.Args[3].(Mask)) ^ (0x8))
			newOpStr = "wcdgb"
			removeArg(inst, int8(2))
			break
		} else if uint8(inst.Args[2].(Mask)) == uint8(2) {
			newOpStr = "vcefb"
			removeArg(inst, int8(2))
			break
		} else if uint8(inst.Args[2].(Mask)) == uint8(3) {
			newOpStr = "vcdgb"
			removeArg(inst, int8(2))
			break
		}

	case "vcfpl":
		if inst.Args[2].(Mask) == Mask(2) && ((inst.Args[3].(Mask)>>3)&(0x1) == 1) {
			inst.Args[3] = Mask((inst.Args[3].(Mask)) ^ (0x8))
			newOpStr = "wcelfb"
			removeArg(inst, int8(2))
			break
		} else if inst.Args[2].(Mask) == Mask(3) && ((inst.Args[3].(Mask)>>3)&(0x1) == 1) {
			inst.Args[3] = Mask((inst.Args[3].(Mask)) ^ (0x8))
			newOpStr = "wcdlgb"
			removeArg(inst, int8(2))
			break
		} else if inst.Args[2].(Mask) == Mask(2) {
			newOpStr = "vcelfb"
			removeArg(inst, int8(2))
			break
		} else if inst.Args[2].(Mask) == Mask(3) {
			newOpStr = "vcdlgb"
			removeArg(inst, int8(2))
			break
		}

	case "vcsfp":
		if inst.Args[2].(Mask) == Mask(2) && ((inst.Args[3].(Mask)>>3)&(0x1) == 1) {
			inst.Args[3] = Mask((inst.Args[3].(Mask)) ^ (0x8))
			newOpStr = "wcfeb"
			removeArg(inst, int8(2))
			break
		} else if inst.Args[2].(Mask) == Mask(3) && ((inst.Args[3].(Mask)>>3)&(0x1) == 1) {
			inst.Args[3] = Mask((inst.Args[3].(Mask)) ^ (0x8))
			newOpStr = "wcgdb"
			removeArg(inst, int8(2))
			break
		} else if inst.Args[2].(Mask) == Mask(2) {
			newOpStr = "vcfeb"
			removeArg(inst, int8(2))
			break
		} else if inst.Args[2].(Mask) == Mask(3) {
			newOpStr = "vcgdb"
			removeArg(inst, int8(2))
			break
		}

	case "vclfp":
		if inst.Args[2].(Mask) == Mask(2) && ((inst.Args[3].(Mask)>>3)&(0x1) == 1) {
			inst.Args[3] = Mask((inst.Args[3].(Mask)) ^ (0x8))
			newOpStr = "wclfeb"
			removeArg(inst, int8(2))
			break
		} else if inst.Args[2].(Mask) == Mask(3) && ((inst.Args[3].(Mask)>>3)&(0x1) == 1) {
			inst.Args[3] = Mask((inst.Args[3].(Mask)) ^ (0x8))
			newOpStr = "wclgdb"
			removeArg(inst, int8(2))
			break
		} else if inst.Args[2].(Mask) == Mask(2) {
			newOpStr = "vclfeb"
			removeArg(inst, int8(2))
			break
		} else if inst.Args[2].(Mask) == Mask(3) {
			newOpStr = "vclgdb"
			removeArg(inst, int8(2))
			break
		}

	case "vfi":
		if inst.Args[2].(Mask) == Mask(2) && ((inst.Args[3].(Mask)>>3)&(0x1) == 1) {
			newOpStr = "wfisb"
			removeArg(inst, int8(2))
			inst.Args[2] = Mask((inst.Args[2].(Mask)) ^ (0x8))
			break
		} else if inst.Args[2].(Mask) == Mask(3) && ((inst.Args[3].(Mask)>>3)&(0x3) == 1) {
			newOpStr = "wfidb"
			removeArg(inst, int8(2))
			inst.Args[2] = Mask((inst.Args[2].(Mask)) ^ (0x8))
			break
		} else if inst.Args[2].(Mask) == Mask(4) && ((inst.Args[3].(Mask)>>3)&(0x1) == 1) {
			newOpStr = "wfixb"
			removeArg(inst, int8(2))
			inst.Args[2] = Mask((inst.Args[2].(Mask)) ^ (0x8))
			break
		} else if inst.Args[2].(Mask) == Mask(2) {
			newOpStr = "vfisb"
			removeArg(inst, int8(2))
			break
		} else if inst.Args[2].(Mask) == Mask(3) {
			newOpStr = "vfidb"
			removeArg(inst, int8(2))
			break
		}

	// Case to handle few vector instructions with 2 M-field operands
	case "vfa", "vfd", "vfll", "vfmax", "vfmin", "vfm":
		for i := 0; i < len(vec4InstrExtndMnics); i++ {
			if opString == vec4InstrExtndMnics[i].BaseOpStr &&
				uint8(inst.Args[vec4InstrExtndMnics[i].Offset1].(Mask)) == vec4InstrExtndMnics[i].Value1 &&
				uint8(inst.Args[vec4InstrExtndMnics[i].Offset2].(Mask)) == vec4InstrExtndMnics[i].Value2 {
				newOpStr = vec4InstrExtndMnics[i].ExtnOpStr
				removeArg(inst, int8(vec4InstrExtndMnics[i].Offset1))
				removeArg(inst, int8(vec4InstrExtndMnics[i].Offset2-1))
				break
			}
		}

	// Case to handle few special "vector" instructions with 2 M-field operands
	case "wfc", "wfk":
		for i := 0; i < len(vec3InstrExtndMnics); i++ {
			if uint8(inst.Args[vec3InstrExtndMnics[i].Offset1].(Mask)) == vec3InstrExtndMnics[i].Value1 &&
				uint8(inst.Args[vec3InstrExtndMnics[i].Offset2].(Mask)) == vec3InstrExtndMnics[i].Value2 {
				newOpStr = opString + vec3InstrExtndMnics[i].ExtnOpStr
				removeArg(inst, int8(vec3InstrExtndMnics[i].Offset1))
				removeArg(inst, int8(vec3InstrExtndMnics[i].Offset2-1))
				break
			}
		}

	// Case to handle few vector instructions with 2 M-field operands
	case "vfma", "vfms", "vfnma", "vfnms":
		for i := 0; i < len(vec7InstrExtndMnics); i++ {
			if opString == vec7InstrExtndMnics[i].BaseOpStr &&
				uint8(inst.Args[vec7InstrExtndMnics[i].Offset1].(Mask)) == vec7InstrExtndMnics[i].Value1 &&
				uint8(inst.Args[vec7InstrExtndMnics[i].Offset2].(Mask)) == vec7InstrExtndMnics[i].Value2 {
				newOpStr = vec7InstrExtndMnics[i].ExtnOpStr
				removeArg(inst, int8(vec7InstrExtndMnics[i].Offset1))
				removeArg(inst, int8(vec7InstrExtndMnics[i].Offset2-1))
				break
			}
		}

	// List of instructions with 3 M-field operands.
	case "vfce", "vfch", "vfche", "vfpso":
		for i := 0; i < len(vec6InstrExtndMnics); i++ {
			if opString == vec6InstrExtndMnics[i].BaseOpStr &&
				uint8(inst.Args[vec6InstrExtndMnics[i].Offset1].(Mask)) == vec6InstrExtndMnics[i].Value1 &&
				uint8(inst.Args[vec6InstrExtndMnics[i].Offset2].(Mask)) == vec6InstrExtndMnics[i].Value2 &&
				uint8(inst.Args[vec6InstrExtndMnics[i].Offset3].(Mask)) == vec6InstrExtndMnics[i].Value3 {
				newOpStr = vec6InstrExtndMnics[i].ExtnOpStr
				removeArg(inst, int8(vec6InstrExtndMnics[i].Offset1))
				removeArg(inst, int8(vec6InstrExtndMnics[i].Offset2-1))
				removeArg(inst, int8(vec6InstrExtndMnics[i].Offset3-2))
				break
			}
		}

	default:
		return opString
	}
	return newOpStr
}

// This is the function that is called to print the disassembled instruction
// in the GNU (AT&T) syntax form.
func GNUSyntax(inst Inst, pc uint64) string {
	if inst.Enc == 0 {
		return ".long 0x0"
	} else if inst.Op == 0 {
		return "error: unknown instruction"
	}
	return inst.String(pc)
}

// removeArg removes the arg in inst.Args[index].
func removeArg(inst *Inst, index int8) {
	for i := int(index); i < len(inst.Args); i++ {
		if i+1 < len(inst.Args) {
			inst.Args[i] = inst.Args[i+1]
		} else {
			inst.Args[i] = nil
		}
	}
}
