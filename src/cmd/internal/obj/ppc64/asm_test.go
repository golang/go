// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ppc64

import (
	"bytes"
	"fmt"
	"internal/buildcfg"
	"internal/testenv"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"cmd/internal/obj"
	"cmd/internal/objabi"
)

var platformEnvs = [][]string{
	{"GOOS=aix", "GOARCH=ppc64"},
	{"GOOS=linux", "GOARCH=ppc64"},
	{"GOOS=linux", "GOARCH=ppc64le"},
}

const invalidPCAlignSrc = `
TEXT test(SB),0,$0-0
ADD $2, R3
PCALIGN $128
RET
`

const validPCAlignSrc = `
TEXT test(SB),0,$0-0
ADD $2, R3
PCALIGN $16
MOVD $8, R16
ADD $8, R4
PCALIGN $32
ADD $8, R3
PCALIGN $8
ADD $4, R8
RET
`

const x64pgm = `
TEXT test(SB),0,$0-0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
PNOP
`
const x32pgm = `
TEXT test(SB),0,$0-0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
PNOP
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
`

const x16pgm = `
TEXT test(SB),0,$0-0
OR R0, R0
OR R0, R0
OR R0, R0
PNOP
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
`

const x0pgm = `
TEXT test(SB),0,$0-0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
PNOP
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
`
const x64pgmA64 = `
TEXT test(SB),0,$0-0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
PNOP
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
PNOP
`

const x64pgmA32 = `
TEXT test(SB),0,$0-0
OR R0, R0
OR R0, R0
OR R0, R0
PNOP
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
OR R0, R0
PNOP
`

// Test that nops are inserted when crossing 64B boundaries, and
// alignment is adjusted to avoid crossing.
func TestPfxAlign(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	dir := t.TempDir()

	pgms := []struct {
		text   []byte
		align  string
		hasNop bool
	}{
		{[]byte(x0pgm), "align=0x0", false},     // No alignment or nop adjustments needed
		{[]byte(x16pgm), "align=0x20", false},   // Increased alignment needed
		{[]byte(x32pgm), "align=0x40", false},   // Worst case alignment needed
		{[]byte(x64pgm), "align=0x0", true},     // 0 aligned is default (16B) alignment
		{[]byte(x64pgmA64), "align=0x40", true}, // extra alignment + nop
		{[]byte(x64pgmA32), "align=0x20", true}, // extra alignment + nop
	}

	for _, pgm := range pgms {
		tmpfile := filepath.Join(dir, "x.s")
		err := os.WriteFile(tmpfile, pgm.text, 0644)
		if err != nil {
			t.Fatalf("can't write output: %v\n", err)
		}
		cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-S", "-o", filepath.Join(dir, "test.o"), tmpfile)
		cmd.Env = append(os.Environ(), "GOOS=linux", "GOARCH=ppc64le")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Errorf("Failed to compile %v: %v\n", pgm, err)
		}
		if !strings.Contains(string(out), pgm.align) {
			t.Errorf("Fatal, misaligned text with prefixed instructions:\n%s", out)
		}
		hasNop := strings.Contains(string(out), "00 00 00 60")
		if hasNop != pgm.hasNop {
			t.Errorf("Fatal, prefixed instruction is missing nop padding:\n%s", out)
		}
	}
}

// TestLarge generates a very large file to verify that large
// program builds successfully, and branches which exceed the
// range of BC are rewritten to reach.
func TestLarge(t *testing.T) {
	if testing.Short() {
		t.Skip("Skip in short mode")
	}
	testenv.MustHaveGoBuild(t)

	dir := t.TempDir()

	// A few interesting test cases for long conditional branch fixups
	tests := []struct {
		jmpinsn     string
		backpattern []string
		fwdpattern  []string
	}{
		// Test the interesting cases of conditional branch rewrites for too-far targets. Simple conditional
		// branches can be made to reach with one JMP insertion, compound conditionals require two.
		//
		// beq <-> bne conversion (insert one jump)
		{"BEQ",
			[]string{``,
				`0x20030 131120\s\(.*\)\tBC\t\$4,\sCR0EQ,\s131128`,
				`0x20034 131124\s\(.*\)\tJMP\t0`},
			[]string{``,
				`0x0000 00000\s\(.*\)\tBC\t\$4,\sCR0EQ,\s8`,
				`0x0004 00004\s\(.*\)\tJMP\t131128`},
		},
		{"BNE",
			[]string{``,
				`0x20030 131120\s\(.*\)\tBC\t\$12,\sCR0EQ,\s131128`,
				`0x20034 131124\s\(.*\)\tJMP\t0`},
			[]string{``,
				`0x0000 00000\s\(.*\)\tBC\t\$12,\sCR0EQ,\s8`,
				`0x0004 00004\s\(.*\)\tJMP\t131128`}},
		// bdnz (BC 16,0,tgt) <-> bdz (BC 18,0,+4) conversion (insert one jump)
		{"BC 16,0,",
			[]string{``,
				`0x20030 131120\s\(.*\)\tBC\t\$18,\sCR0LT,\s131128`,
				`0x20034 131124\s\(.*\)\tJMP\t0`},
			[]string{``,
				`0x0000 00000\s\(.*\)\tBC\t\$18,\sCR0LT,\s8`,
				`0x0004 00004\s\(.*\)\tJMP\t131128`}},
		{"BC 18,0,",
			[]string{``,
				`0x20030 131120\s\(.*\)\tBC\t\$16,\sCR0LT,\s131128`,
				`0x20034 131124\s\(.*\)\tJMP\t0`},
			[]string{``,
				`0x0000 00000\s\(.*\)\tBC\t\$16,\sCR0LT,\s8`,
				`0x0004 00004\s\(.*\)\tJMP\t131128`}},
		// bdnzt (BC 8,0,tgt) <-> bdnzt (BC 8,0,+4) conversion (insert two jumps)
		{"BC 8,0,",
			[]string{``,
				`0x20034 131124\s\(.*\)\tBC\t\$8,\sCR0LT,\s131132`,
				`0x20038 131128\s\(.*\)\tJMP\t131136`,
				`0x2003c 131132\s\(.*\)\tJMP\t0\n`},
			[]string{``,
				`0x0000 00000\s\(.*\)\tBC\t\$8,\sCR0LT,\s8`,
				`0x0004 00004\s\(.*\)\tJMP\t12`,
				`0x0008 00008\s\(.*\)\tJMP\t131136\n`}},
	}

	for _, test := range tests {
		// generate a very large function
		buf := bytes.NewBuffer(make([]byte, 0, 7000000))
		gen(buf, test.jmpinsn)

		tmpfile := filepath.Join(dir, "x.s")
		err := os.WriteFile(tmpfile, buf.Bytes(), 0644)
		if err != nil {
			t.Fatalf("can't write output: %v\n", err)
		}

		// Test on all supported ppc64 platforms
		for _, platenv := range platformEnvs {
			cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-S", "-o", filepath.Join(dir, "test.o"), tmpfile)
			cmd.Env = append(os.Environ(), platenv...)
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Errorf("Assemble failed (%v): %v, output: %s", platenv, err, out)
			}
			matched, err := regexp.MatchString(strings.Join(test.fwdpattern, "\n\t*"), string(out))
			if err != nil {
				t.Fatal(err)
			}
			if !matched {
				t.Errorf("Failed to detect long forward BC fixup in (%v):%s\n", platenv, out)
			}
			matched, err = regexp.MatchString(strings.Join(test.backpattern, "\n\t*"), string(out))
			if err != nil {
				t.Fatal(err)
			}
			if !matched {
				t.Errorf("Failed to detect long backward BC fixup in (%v):%s\n", platenv, out)
			}
		}
	}
}

// gen generates a very large program with a very long forward and backwards conditional branch.
func gen(buf *bytes.Buffer, jmpinsn string) {
	fmt.Fprintln(buf, "TEXT f(SB),0,$0-0")
	fmt.Fprintln(buf, "label_start:")
	fmt.Fprintln(buf, jmpinsn, "label_end")
	for i := 0; i < (1<<15 + 10); i++ {
		fmt.Fprintln(buf, "MOVD R0, R1")
	}
	fmt.Fprintln(buf, jmpinsn, "label_start")
	fmt.Fprintln(buf, "label_end:")
	fmt.Fprintln(buf, "MOVD R0, R1")
	fmt.Fprintln(buf, "RET")
}

// TestPCalign generates two asm files containing the
// PCALIGN directive, to verify correct values are and
// accepted, and incorrect values are flagged in error.
func TestPCalign(t *testing.T) {
	var pattern8 = `0x...8\s.*ADD\s..,\sR8`
	var pattern16 = `0x...[80]\s.*MOVD\s..,\sR16`
	var pattern32 = `0x...0\s.*ADD\s..,\sR3`

	testenv.MustHaveGoBuild(t)

	dir := t.TempDir()

	// generate a test with valid uses of PCALIGN

	tmpfile := filepath.Join(dir, "x.s")
	err := os.WriteFile(tmpfile, []byte(validPCAlignSrc), 0644)
	if err != nil {
		t.Fatalf("can't write output: %v\n", err)
	}

	// build generated file without errors and assemble it
	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-o", filepath.Join(dir, "x.o"), "-S", tmpfile)
	cmd.Env = append(os.Environ(), "GOARCH=ppc64le", "GOOS=linux")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("Build failed: %v, output: %s", err, out)
	}

	matched, err := regexp.MatchString(pattern8, string(out))
	if err != nil {
		t.Fatal(err)
	}
	if !matched {
		t.Errorf("The 8 byte alignment is not correct: %t, output:%s\n", matched, out)
	}

	matched, err = regexp.MatchString(pattern16, string(out))
	if err != nil {
		t.Fatal(err)
	}
	if !matched {
		t.Errorf("The 16 byte alignment is not correct: %t, output:%s\n", matched, out)
	}

	matched, err = regexp.MatchString(pattern32, string(out))
	if err != nil {
		t.Fatal(err)
	}
	if !matched {
		t.Errorf("The 32 byte alignment is not correct: %t, output:%s\n", matched, out)
	}

	// generate a test with invalid use of PCALIGN

	tmpfile = filepath.Join(dir, "xi.s")
	err = os.WriteFile(tmpfile, []byte(invalidPCAlignSrc), 0644)
	if err != nil {
		t.Fatalf("can't write output: %v\n", err)
	}

	// build test with errors and check for messages
	cmd = testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-o", filepath.Join(dir, "xi.o"), "-S", tmpfile)
	cmd.Env = append(os.Environ(), "GOARCH=ppc64le", "GOOS=linux")
	out, err = cmd.CombinedOutput()
	if !strings.Contains(string(out), "Unexpected alignment") {
		t.Errorf("Invalid alignment not detected for PCALIGN\n")
	}
}

// Verify register constants are correctly aligned. Much of the ppc64 assembler assumes masking out significant
// bits will produce a valid register number:
// REG_Rx & 31 == x
// REG_Fx & 31 == x
// REG_Vx & 31 == x
// REG_VSx & 63 == x
// REG_SPRx & 1023 == x
// REG_CRx & 7 == x
//
// VR and FPR disjointly overlap VSR, interpreting as VSR registers should produce the correctly overlapped VSR.
// REG_FPx & 63 == x
// REG_Vx & 63 == x + 32
func TestRegValueAlignment(t *testing.T) {
	tstFunc := func(rstart, rend, msk, rout int) {
		for i := rstart; i <= rend; i++ {
			if i&msk != rout {
				t.Errorf("%v is not aligned to 0x%X (expected %d, got %d)\n", rconv(i), msk, rout, rstart&msk)
			}
			rout++
		}
	}
	var testType = []struct {
		rstart int
		rend   int
		msk    int
		rout   int
	}{
		{REG_VS0, REG_VS63, 63, 0},
		{REG_R0, REG_R31, 31, 0},
		{REG_F0, REG_F31, 31, 0},
		{REG_V0, REG_V31, 31, 0},
		{REG_V0, REG_V31, 63, 32},
		{REG_F0, REG_F31, 63, 0},
		{REG_SPR0, REG_SPR0 + 1023, 1023, 0},
		{REG_CR0, REG_CR7, 7, 0},
		{REG_CR0LT, REG_CR7SO, 31, 0},
	}
	for _, t := range testType {
		tstFunc(t.rstart, t.rend, t.msk, t.rout)
	}
}

// Verify interesting obj.Addr arguments are classified correctly.
func TestAddrClassifier(t *testing.T) {
	type cmplx struct {
		pic     int
		pic_dyn int
		dyn     int
		nonpic  int
	}
	tsts := [...]struct {
		arg    obj.Addr
		output interface{}
	}{
		// Supported register type args
		{obj.Addr{Type: obj.TYPE_REG, Reg: REG_R1}, C_REG},
		{obj.Addr{Type: obj.TYPE_REG, Reg: REG_R2}, C_REGP},
		{obj.Addr{Type: obj.TYPE_REG, Reg: REG_F1}, C_FREG},
		{obj.Addr{Type: obj.TYPE_REG, Reg: REG_F2}, C_FREGP},
		{obj.Addr{Type: obj.TYPE_REG, Reg: REG_V2}, C_VREG},
		{obj.Addr{Type: obj.TYPE_REG, Reg: REG_VS1}, C_VSREG},
		{obj.Addr{Type: obj.TYPE_REG, Reg: REG_VS2}, C_VSREGP},
		{obj.Addr{Type: obj.TYPE_REG, Reg: REG_CR}, C_CREG},
		{obj.Addr{Type: obj.TYPE_REG, Reg: REG_CR1}, C_CREG},
		{obj.Addr{Type: obj.TYPE_REG, Reg: REG_CR1SO}, C_CRBIT},
		{obj.Addr{Type: obj.TYPE_REG, Reg: REG_SPR0}, C_SPR},
		{obj.Addr{Type: obj.TYPE_REG, Reg: REG_SPR0 + 8}, C_LR},
		{obj.Addr{Type: obj.TYPE_REG, Reg: REG_SPR0 + 9}, C_CTR},
		{obj.Addr{Type: obj.TYPE_REG, Reg: REG_FPSCR}, C_FPSCR},
		{obj.Addr{Type: obj.TYPE_REG, Reg: REG_A1}, C_AREG},

		// Memory type arguments.
		{obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_GOTREF}, C_ADDR},
		{obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_TOCREF}, C_ADDR},
		{obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_EXTERN, Sym: &obj.LSym{Type: objabi.STLSBSS}}, cmplx{C_TLS_IE, C_TLS_IE, C_TLS_LE, C_TLS_LE}},
		{obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_EXTERN, Sym: &obj.LSym{Type: objabi.SDATA}}, C_ADDR},
		{obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_AUTO}, C_SOREG},
		{obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_AUTO, Offset: BIG}, C_LOREG},
		{obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_AUTO, Offset: -BIG - 1}, C_LOREG},
		{obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_PARAM}, C_SOREG},
		{obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_PARAM, Offset: BIG}, C_LOREG},
		{obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_PARAM, Offset: -BIG - 33}, C_LOREG}, // 33 is FixedFrameSize-1
		{obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_NONE}, C_ZOREG},
		{obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_NONE, Index: REG_R4}, C_XOREG},
		{obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_NONE, Offset: 1}, C_SOREG},
		{obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_NONE, Offset: BIG}, C_LOREG},
		{obj.Addr{Type: obj.TYPE_MEM, Name: obj.NAME_NONE, Offset: -BIG - 33}, C_LOREG},

		// Misc (golang initializes -0.0 to 0.0, hence the obfuscation below)
		{obj.Addr{Type: obj.TYPE_TEXTSIZE}, C_TEXTSIZE},
		{obj.Addr{Type: obj.TYPE_FCONST, Val: 0.0}, C_ZCON},
		{obj.Addr{Type: obj.TYPE_FCONST, Val: math.Float64frombits(0x8000000000000000)}, C_S16CON},

		// Address type arguments
		{obj.Addr{Type: obj.TYPE_ADDR, Reg: REG_R0, Name: obj.NAME_NONE, Offset: 1}, C_SACON},
		{obj.Addr{Type: obj.TYPE_ADDR, Reg: REG_R0, Name: obj.NAME_NONE, Offset: BIG}, C_LACON},
		{obj.Addr{Type: obj.TYPE_ADDR, Reg: REG_R0, Name: obj.NAME_NONE, Offset: -BIG - 1}, C_LACON},
		{obj.Addr{Type: obj.TYPE_ADDR, Reg: REG_R0, Name: obj.NAME_NONE, Offset: 1 << 32}, C_DACON},
		{obj.Addr{Type: obj.TYPE_ADDR, Name: obj.NAME_EXTERN, Sym: &obj.LSym{Type: objabi.SDATA}}, C_LACON},
		{obj.Addr{Type: obj.TYPE_ADDR, Name: obj.NAME_STATIC, Sym: &obj.LSym{Type: objabi.SDATA}}, C_LACON},
		{obj.Addr{Type: obj.TYPE_ADDR, Reg: REG_R0, Name: obj.NAME_AUTO, Offset: 1}, C_SACON},
		{obj.Addr{Type: obj.TYPE_ADDR, Reg: REG_R0, Name: obj.NAME_AUTO, Offset: BIG}, C_LACON},
		{obj.Addr{Type: obj.TYPE_ADDR, Reg: REG_R0, Name: obj.NAME_AUTO, Offset: -BIG - 1}, C_LACON},
		{obj.Addr{Type: obj.TYPE_ADDR, Reg: REG_R0, Name: obj.NAME_PARAM, Offset: 1}, C_SACON},
		{obj.Addr{Type: obj.TYPE_ADDR, Reg: REG_R0, Name: obj.NAME_PARAM, Offset: BIG}, C_LACON},
		{obj.Addr{Type: obj.TYPE_ADDR, Reg: REG_R0, Name: obj.NAME_PARAM, Offset: -BIG - 33}, C_LACON}, // 33 is FixedFrameSize-1

		// Constant type arguments
		{obj.Addr{Type: obj.TYPE_CONST, Name: obj.NAME_NONE, Offset: 0}, C_ZCON},
		{obj.Addr{Type: obj.TYPE_CONST, Name: obj.NAME_NONE, Offset: 1}, C_U1CON},
		{obj.Addr{Type: obj.TYPE_CONST, Name: obj.NAME_NONE, Offset: 2}, C_U2CON},
		{obj.Addr{Type: obj.TYPE_CONST, Name: obj.NAME_NONE, Offset: 4}, C_U3CON},
		{obj.Addr{Type: obj.TYPE_CONST, Name: obj.NAME_NONE, Offset: 8}, C_U4CON},
		{obj.Addr{Type: obj.TYPE_CONST, Name: obj.NAME_NONE, Offset: 16}, C_U5CON},
		{obj.Addr{Type: obj.TYPE_CONST, Name: obj.NAME_NONE, Offset: 32}, C_U8CON},
		{obj.Addr{Type: obj.TYPE_CONST, Name: obj.NAME_NONE, Offset: 1 << 14}, C_U15CON},
		{obj.Addr{Type: obj.TYPE_CONST, Name: obj.NAME_NONE, Offset: 1 << 15}, C_U16CON},
		{obj.Addr{Type: obj.TYPE_CONST, Name: obj.NAME_NONE, Offset: 1 + 1<<16}, C_U31CON},
		{obj.Addr{Type: obj.TYPE_CONST, Name: obj.NAME_NONE, Offset: 1 << 31}, C_U32CON},
		{obj.Addr{Type: obj.TYPE_CONST, Name: obj.NAME_NONE, Offset: 1 << 32}, C_S34CON},
		{obj.Addr{Type: obj.TYPE_CONST, Name: obj.NAME_NONE, Offset: 1 << 33}, C_64CON},
		{obj.Addr{Type: obj.TYPE_CONST, Name: obj.NAME_NONE, Offset: -1}, C_S16CON},
		{obj.Addr{Type: obj.TYPE_CONST, Name: obj.NAME_NONE, Offset: -0x10001}, C_S32CON},
		{obj.Addr{Type: obj.TYPE_CONST, Name: obj.NAME_NONE, Offset: 0x10001}, C_U31CON},
		{obj.Addr{Type: obj.TYPE_CONST, Name: obj.NAME_NONE, Offset: -(1 << 33)}, C_S34CON},
		{obj.Addr{Type: obj.TYPE_CONST, Name: obj.NAME_NONE, Offset: -(1 << 34)}, C_64CON},

		// Branch like arguments
		{obj.Addr{Type: obj.TYPE_BRANCH, Sym: &obj.LSym{Type: objabi.SDATA}}, cmplx{C_BRA, C_BRAPIC, C_BRAPIC, C_BRA}},
		{obj.Addr{Type: obj.TYPE_BRANCH}, C_BRA},
	}

	pic_ctxt9 := ctxt9{ctxt: &obj.Link{Flag_shared: true, Arch: &Linkppc64}, autosize: 0}
	pic_dyn_ctxt9 := ctxt9{ctxt: &obj.Link{Flag_shared: true, Flag_dynlink: true, Arch: &Linkppc64}, autosize: 0}
	dyn_ctxt9 := ctxt9{ctxt: &obj.Link{Flag_dynlink: true, Arch: &Linkppc64}, autosize: 0}
	nonpic_ctxt9 := ctxt9{ctxt: &obj.Link{Arch: &Linkppc64}, autosize: 0}
	ctxts := [...]*ctxt9{&pic_ctxt9, &pic_dyn_ctxt9, &dyn_ctxt9, &nonpic_ctxt9}
	name := [...]string{"pic", "pic_dyn", "dyn", "nonpic"}
	for _, tst := range tsts {
		var expect []int
		switch tst.output.(type) {
		case cmplx:
			v := tst.output.(cmplx)
			expect = []int{v.pic, v.pic_dyn, v.dyn, v.nonpic}
		case int:
			expect = []int{tst.output.(int), tst.output.(int), tst.output.(int), tst.output.(int)}
		}
		for i := range ctxts {
			if output := ctxts[i].aclass(&tst.arg); output != expect[i] {
				t.Errorf("%s.aclass(%v) = %v, expected %v\n", name[i], tst.arg, DRconv(output), DRconv(expect[i]))
			}
		}
	}
}

// The optab size should remain constant when reinitializing the PPC64 assembler backend.
func TestOptabReinit(t *testing.T) {
	buildcfg.GOOS = "linux"
	buildcfg.GOARCH = "ppc64le"
	buildcfg.GOPPC64 = 8
	buildop(nil)
	optabLen := len(optab)
	buildcfg.GOPPC64 = 9
	buildop(nil)
	reinitOptabLen := len(optab)
	if reinitOptabLen != optabLen {
		t.Errorf("rerunning buildop changes optab size from %d to %d", optabLen, reinitOptabLen)
	}
}
