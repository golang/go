// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"flag"
	"fmt"
	"os"
	"reflect"
	"regexp"
	"sort"
	"testing"
)

var arm64Path = flag.String("arm64Path", "", "Path to ARM64 XML definitions")

func requireEqual(t *testing.T, expected, actual any) bool {
	t.Helper()
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("❌ expected %v, got %v", expected, actual)
		return false
	}

	if expected != nil {
		switch reflect.TypeOf(expected).Kind() {
		case reflect.Slice, reflect.Array:
			t.Logf("✅ %v", expected)
		}
	}
	return true
}

func matchEqual(t *testing.T, expected, actual any) bool {
	t.Helper()
	eq := reflect.DeepEqual(expected, actual)
	if eq && expected != nil {
		switch reflect.TypeOf(expected).Kind() {
		case reflect.Slice, reflect.Array:
			t.Logf("✅ %v", expected)
		}
	}
	return eq
}

func arngs(t *testing.T, instr *Instruction, expectedArrangements []string, expectedShape ArngShape) {
	t.Helper()
	actualArrangements, actualShape := instr.Arrangements()
	actualStrings := make([]string, len(actualArrangements))
	for i, arr := range actualArrangements {
		actualStrings[i] = fmt.Sprintf("%s%d:%s", arr.baseType, arr.elemBits, arr.arrangement)
	}
	sort.Strings(actualStrings)
	sort.Strings(expectedArrangements)
	requireEqual(t, expectedArrangements, actualStrings)
	requireEqual(t, expectedShape, actualShape)
}

func ops(t *testing.T, instr *Instruction, expectedOps []string, equal func(*testing.T, any, any) bool) bool {
	t.Helper()
	templates := instr.templates()
	if !equal(t, 1, len(templates)) {
		return false
	}
	template := templates[0]
	operands := template.operands
	var actualOps []string
	for _, operand := range operands {
		opStr := fmt.Sprintf("%s:%d", operand.Type.String(), operand.AsmPos)
		actualOps = append(actualOps, opStr)
	}
	return equal(t, expectedOps, actualOps)
}

func matchOps(expectedOps []string) func(*testing.T, *Instruction) bool {
	return func(t *testing.T, instr *Instruction) bool {
		return ops(t, instr, expectedOps, matchEqual)
	}
}

func requireOps(expectedOps []string) func(*testing.T, *Instruction) bool {
	return func(t *testing.T, instr *Instruction) bool {
		return ops(t, instr, expectedOps, requireEqual)
	}
}

func requireArngs(expectedArngs []string, expectedShape ArngShape) func(*testing.T, *Instruction) {
	return func(t *testing.T, instr *Instruction) {
		arngs(t, instr, expectedArngs, expectedShape)
	}
}

func emitsDefs(expectedCount int) func(*testing.T, *Instruction) {
	return func(t *testing.T, instr *Instruction) {
		values := instr.EmitAll()
		requireEqual(t, expectedCount, len(values))
	}
}

var (
	// Operand patterns
	binary                   = []string{"VReg:0", "VReg:1", "VReg:2"}
	unary                    = []string{"VReg:0", "VReg:1"}
	twoArgsResultInArg0      = []string{"VReg:0", "VReg:0", "VReg:1"}
	unaryWithImm             = []string{"VReg:0", "Imm:2", "VReg:1"}
	unaryWithImmResultInArg0 = []string{"VReg:0", "Imm:2", "VReg:0", "VReg:1"}
	binaryWithImm            = []string{"VReg:0", "Imm:3", "VReg:1", "VReg:2"}
	elemToVreg               = []string{"VReg:0", "Imm:1", "VReg:1"}
	insertFromLane           = []string{"VReg:0", "Imm:0", "Imm:1", "VReg:0", "VReg:1"}
	insertFromGReg           = []string{"VReg:0", "Imm:0", "VReg:0", "GReg:1"}
	threeArgsResultInArg0    = []string{"VReg:0", "VReg:0", "VReg:1", "VReg:2"}
	fourOperands             = []string{"VReg:0", "VReg:1", "VReg:2", "VReg:3"}
	elemToGReg               = []string{"GReg:0", "Imm:1", "VReg:1"}

	// Arrangement patterns
	floatS32          = []string{"float32:4S"}
	floating          = []string{"float32:2S", "float32:4S", "float64:2D"}
	bitwise16B        = []string{"int8:16B", "uint8:16B"}
	integer2D         = []string{"int64:2D", "uint64:2D"}
	integerUpTo8Bits  = []string{"int8:16B", "int8:8B", "uint8:16B", "uint8:8B"}
	integerUpTo16Bits = append([]string{"int16:4H", "int16:8H", "uint16:4H", "uint16:8H"}, integerUpTo8Bits...)
	integerUpTo32Bits = append([]string{"int32:2S", "int32:4S", "uint32:2S", "uint32:4S"}, integerUpTo16Bits...)
	integerWideOnly   = []string{"int16:8H", "int32:4S", "int64:2D", "uint16:8H", "uint32:4S", "uint64:2D"}
	polynomialArrngs  = []string{"int8:8B", "int8:16B", "int64:1D", "int64:2D", "uint8:8B", "uint8:16B", "uint64:1D", "uint64:2D"}
	integer32And8Bits = append([]string{"int32:2S", "int32:4S", "uint32:2S", "uint32:4S"}, integerUpTo8Bits...)
	addvArngs         = append([]string{"int32:4S", "uint32:4S"}, integerUpTo16Bits...)
	integer           = append([]string{"int64:2D", "uint64:2D"}, integerUpTo32Bits...)
	integerWith1D     = append([]string{"int64:1D", "uint64:1D"}, integer...)
	allArngs          = append(append([]string{}, floating...), integer...)
	bitwise           = []string{
		"int8:16B", "uint8:16B", "int8:8B", "uint8:8B",
		"int16:16B", "uint16:16B", "int16:8B", "uint16:8B",
		"int32:16B", "uint32:16B", "int32:8B", "uint32:8B",
		"int64:16B", "uint64:16B",
	}
	fullwidth = []string{
		"float32:4S", "float64:2D",
		"int8:16B", "int16:8H", "int32:4S", "int64:2D",
		"uint8:16B", "uint16:8H", "uint32:4S", "uint64:2D",
	}
)

type Arm64InstructionTestSpec struct {
	Pattern     string                              // to match instruction mnemonics
	OpMatch     func(*testing.T, *Instruction) bool // returns true if operands match
	ArngTest    func(*testing.T, *Instruction)
	EmitAllTest func(*testing.T, *Instruction)
}

var arm64InstructionTests = []Arm64InstructionTestSpec{
	{"^((UQ|SQ)?ADD|(UQ|SQ)?SUB)$", requireOps(binary), requireArngs(integer, DefaultArngs), emitsDefs(14)},
	{"^ADDP$", matchOps(binary), requireArngs(integer, DefaultArngs), emitsDefs(14)},
	{"^ADDP$", matchOps(unary), requireArngs([]string{"int64:2D", "uint64:2D"}, DefaultArngs), emitsDefs(2)},
	{"^FADDP$", matchOps(binary), requireArngs(floating, DefaultArngs), emitsDefs(3)},
	{"^FADDP$", matchOps(unary), requireArngs([]string{"float32:2S", "float64:2D"}, DefaultArngs), emitsDefs(2)},
	{"^SABA$", matchOps(threeArgsResultInArg0), requireArngs(integerUpTo32Bits, DefaultArngs), emitsDefs(12)},
	{"^SABAL$", matchOps(threeArgsResultInArg0), requireArngs(integerUpTo32Bits, LongArngs), emitsDefs(12)},
	{"^F(ADD|SUB|DIV)$", requireOps(binary), requireArngs(floating, DefaultArngs), emitsDefs(3)},
	{"^(AND|ORR|EOR|BIC|ORN)$", matchOps(binary), requireArngs(bitwise, DefaultArngs), emitsDefs(14)},
	{"^NOT$", requireOps(unary), requireArngs(bitwise, DefaultArngs), emitsDefs(14)},
	{"^CM(GT|GE|EQ)$", matchOps(binary), requireArngs(integer, DefaultArngs), emitsDefs(14)},
	{"^CM(GT|GE|EQ)$", matchOps(unaryWithImm), requireArngs(integer, DefaultArngs), emitsDefs(14)},
	{"^CM(HI|HS|TST)$", requireOps(binary), requireArngs(integer, DefaultArngs), emitsDefs(14)},
	{"^CM(LT|LE)$", requireOps(unaryWithImm), requireArngs(integer, DefaultArngs), emitsDefs(14)},
	{"^FCM(GT|GE|EQ)$", matchOps(binary), requireArngs(floating, DefaultArngs), emitsDefs(3)},
	{"^FCM(GT|GE|EQ)$", matchOps(unaryWithImm), requireArngs(floating, DefaultArngs), emitsDefs(3)},
	{"^FCM(LT|LE)$", requireOps(unaryWithImm), requireArngs(floating, DefaultArngs), emitsDefs(3)},
	{"^(BIT|BIF|BSL)$", matchOps(threeArgsResultInArg0), requireArngs(bitwise, DefaultArngs), emitsDefs(14)},
	{"^BCAX$", matchOps(fourOperands), requireArngs(bitwise16B, DefaultArngs), emitsDefs(2)},
	{"^EOR3$", matchOps(fourOperands), requireArngs(bitwise16B, DefaultArngs), emitsDefs(2)},
	{"^RAX1$", matchOps(binary), requireArngs(integer2D, DefaultArngs), emitsDefs(2)},
	{"^XAR$", matchOps(binaryWithImm), requireArngs(integer2D, DefaultArngs), emitsDefs(2)},
	{"^DUP$", matchOps(elemToVreg), requireArngs(allArngs, DefaultArngs), emitsDefs(17)},
	{"^INS$", matchOps(insertFromLane), requireArngs(fullwidth, DefaultArngs), emitsDefs(10)},
	{"^INS$", matchOps(insertFromGReg), requireArngs(fullwidth, DefaultArngs), emitsDefs(10)},
	{"^UMOV$", matchOps(elemToGReg), requireArngs(integer, DefaultArngs), emitsDefs(14)},
	{"^EXT$", requireOps(binaryWithImm), requireArngs(integerUpTo8Bits, DefaultArngs), emitsDefs(4)},
	{"^TBL$", requireOps(binary), requireArngs(integerUpTo8Bits, DefaultArngs), emitsDefs(4)},
	{"^TBX$", requireOps(threeArgsResultInArg0), requireArngs(integerUpTo8Bits, DefaultArngs), emitsDefs(4)},
	{"^REV16$", requireOps(unary), requireArngs(integerUpTo8Bits, DefaultArngs), emitsDefs(4)},
	{"^REV32$", requireOps(unary), requireArngs(integerUpTo16Bits, DefaultArngs), emitsDefs(8)},
	{"^REV64$", requireOps(unary), requireArngs(integerUpTo32Bits, DefaultArngs), emitsDefs(12)},
	{"^(ZIP[12]|UZP[12]|TRN[12])$", requireOps(binary), requireArngs(integer, DefaultArngs), emitsDefs(14)},
	{"^(S|U)(MIN|MAX)P?$", requireOps(binary), requireArngs(integerUpTo32Bits, DefaultArngs), emitsDefs(12)},
	{"^((S|U)(MIN|MAX)|ADD)V$", requireOps(unary), requireArngs(addvArngs, DefaultArngs), emitsDefs(10)},
	{"^F(MIN|MAX)(NM)?$", requireOps(binary), requireArngs(floating, DefaultArngs), emitsDefs(3)},
	{"^F(MIN|MAX)(NM)?P$", matchOps(binary), requireArngs(floating, DefaultArngs), emitsDefs(3)},
	{"^F(MIN|MAX)(NM)?V$", requireOps(unary), requireArngs(floatS32, DefaultArngs), emitsDefs(1)},
	{"^(SQ)?(ABS|NEG)$", requireOps(unary), requireArngs(integer, DefaultArngs), emitsDefs(14)},
	{"^F(ABS|NEG)$", requireOps(unary), requireArngs(floating, DefaultArngs), emitsDefs(3)},
	{"^(S|U)SHL$", requireOps(binary), requireArngs(integer, DefaultArngs), emitsDefs(14)},
	{"^(S|U)QSHL$", matchOps(binary), requireArngs(integer, DefaultArngs), emitsDefs(14)},
	{"^(S|U)QSHL$", matchOps(unaryWithImm), requireArngs(integer, DefaultArngs), emitsDefs(14)},
	{"^SHL$", requireOps(unaryWithImm), requireArngs(integer, DefaultArngs), emitsDefs(14)},
	{"^(S|U)SHR$", requireOps(unaryWithImm), requireArngs(integer, DefaultArngs), emitsDefs(14)},
	{"^(S|U)SRA$", requireOps(unaryWithImmResultInArg0), requireArngs(integer, DefaultArngs), emitsDefs(14)},
	{"^(S|U)SHLL$", requireOps(unaryWithImm), requireArngs(integerUpTo32Bits, LongArngs), emitsDefs(12)},
	{"^SADALP$", matchOps(twoArgsResultInArg0), requireArngs(integerUpTo32Bits, LongArngs), emitsDefs(12)},
	{"^((S|U)ADDLP)$", requireOps(unary), requireArngs(integerUpTo32Bits, LongArngs), emitsDefs(12)},
	{"^(R?(ADD|SUB)HN)$", requireOps(binary), requireArngs(integerWideOnly, NarrowArngs), emitsDefs(6)},
	{"^SHRN$", requireOps(unaryWithImm), requireArngs(integerWideOnly, NarrowArngs), emitsDefs(6)},
	{"^(CLZ|CLS)$", requireOps(unary), requireArngs(integerUpTo32Bits, DefaultArngs), emitsDefs(12)},
	{"^(CNT|RBIT)$", requireOps(unary), requireArngs(integerUpTo8Bits, DefaultArngs), emitsDefs(4)},
	{"^(S|U)R?HADD$", matchOps(binary), requireArngs(integerUpTo32Bits, DefaultArngs), emitsDefs(12)},
	{"^F(RINT(N|P|M|Z)?|SQRT)$", requireOps(unary), requireArngs(floating, DefaultArngs), emitsDefs(3)},
	{"^FMUL$", matchOps(binary), requireArngs(floating, DefaultArngs), emitsDefs(3)},
	{"^F(MLA|MLS)$", matchOps(threeArgsResultInArg0), requireArngs(floating, DefaultArngs), emitsDefs(3)},
	{"^MUL$", matchOps(binary), requireArngs(integerUpTo32Bits, DefaultArngs), emitsDefs(12)},
	{"^((S|U)MULL)$", matchOps(binary), requireArngs(integerUpTo32Bits, LongArngs), emitsDefs(12)},
	{"^(MLA|MLS)$", matchOps(threeArgsResultInArg0), requireArngs(integerUpTo32Bits, DefaultArngs), emitsDefs(12)},
	{"^((S|U)Q)?XTN$", requireOps(unary), requireArngs(integerWideOnly, NarrowArngs), emitsDefs(6)},
	{"^(S|U)XTL$", requireOps(unary), requireArngs(integerUpTo32Bits, LongArngs), emitsDefs(12)},
	{"^FCVT[NMPZ](S|U)$", matchOps(unary), requireArngs(floating, DefaultArngs), emitsDefs(3)},
	{"^(S|U)CVTF$", matchOps(unary), requireArngs(floating, DefaultArngs), emitsDefs(3)},
	{"^(S|U)ADDW$", requireOps(binary), requireArngs(integerWideOnly, WideArngs), emitsDefs(6)},
	{"^(S|U)SUBW$", requireOps(binary), requireArngs(integerWideOnly, WideArngs), emitsDefs(6)},
	{"^FCVTL$", requireOps(unary), requireArngs([]string{"float32:2S", "float32:4S"}, LongArngs), emitsDefs(2)},
	{"^USDOT$", matchOps(threeArgsResultInArg0), requireArngs(integer32And8Bits, UnsupportedArngs), emitsDefs(0)},
	{"^PMULL$", matchOps(binary), requireArngs(polynomialArrngs, LongArngs), emitsDefs(8)},
}

func TestArm64Instructions(t *testing.T) {
	if *arm64Path == "" {
		t.Skip("ARM64 path not specified, use -arm64Path flag")
	}

	instructions, err := ParseInstructions(*arm64Path)
	if err != nil {
		t.Fatalf("ParseInstructions failed: %v", err)
	}
	t.Logf("parsed %d ARM64 instructions", len(instructions))

	for _, spec := range arm64InstructionTests {
		regex, err := regexp.Compile(spec.Pattern)
		requireEqual(t, error(nil), err)

		t.Run(spec.Pattern, func(t *testing.T) {
			var instrCount int
			var matches []*Instruction

			for _, instr := range instructions {
				if regex.MatchString(instr.Mnemonic()) {
					instrCount++
					if spec.OpMatch(t, instr) {
						matches = append(matches, instr)
					}
				}
			}
			requireEqual(t, true, len(matches) > 0)
			t.Logf("🔍 pattern %s: %d instructions, %d matched", spec.Pattern, instrCount, len(matches))

			for _, instr := range matches {
				t.Run(instr.Mnemonic(), func(t *testing.T) {
					requireEqual(t, "advsimd", instr.InstrClass())

					t.Run("Arrangements", func(t *testing.T) {
						spec.ArngTest(t, instr)
					})

					t.Run("EmitAll", func(t *testing.T) {
						spec.EmitAllTest(t, instr)
					})
				})
			}
		})
	}
}

func TestMain(m *testing.M) {
	flag.Parse()
	os.Exit(m.Run())
}
