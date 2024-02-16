// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86

import (
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"internal/testenv"
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

type oclassTest struct {
	arg  *obj.Addr
	want int // Expected oclass return value for a given arg
}

// Filled inside init, because it's easier to do with helper functions.
var (
	oclassTestsAMD64 []*oclassTest
	oclassTests386   []*oclassTest
)

func init() {
	// Required for tests that access any of
	// opindex/ycover/reg/regrex global tables.
	var ctxt obj.Link
	instinit(&ctxt)

	regAddr := func(reg int16) *obj.Addr {
		return &obj.Addr{Type: obj.TYPE_REG, Reg: reg}
	}
	immAddr := func(v int64) *obj.Addr {
		return &obj.Addr{Type: obj.TYPE_CONST, Offset: v}
	}
	regListAddr := func(regFrom, regTo int16) *obj.Addr {
		return &obj.Addr{Type: obj.TYPE_REGLIST, Offset: EncodeRegisterRange(regFrom, regTo)}
	}
	memAddr := func(base, index int16) *obj.Addr {
		return &obj.Addr{Type: obj.TYPE_MEM, Reg: base, Index: index}
	}

	// TODO(quasilyte): oclass doesn't return Yxxx for X/Y regs with
	// ID higher than 7. We don't encode such instructions, but this
	// behavior seems inconsistent. It should probably either
	// never check for arch or do it in all cases.

	oclassTestsCommon := []*oclassTest{
		{&obj.Addr{Type: obj.TYPE_NONE}, Ynone},
		{&obj.Addr{Type: obj.TYPE_BRANCH}, Ybr},
		{&obj.Addr{Type: obj.TYPE_TEXTSIZE}, Ytextsize},

		{&obj.Addr{Type: obj.TYPE_INDIR, Name: obj.NAME_EXTERN}, Yindir},
		{&obj.Addr{Type: obj.TYPE_INDIR, Name: obj.NAME_GOTREF}, Yindir},

		{&obj.Addr{Type: obj.TYPE_ADDR, Name: obj.NAME_AUTO}, Yiauto},
		{&obj.Addr{Type: obj.TYPE_ADDR, Name: obj.NAME_PARAM}, Yiauto},
		{&obj.Addr{Type: obj.TYPE_ADDR, Name: obj.NAME_EXTERN}, Yiauto},
		{&obj.Addr{Type: obj.TYPE_ADDR, Sym: &obj.LSym{Name: "runtime.duff"}}, Yi32},
		{&obj.Addr{Type: obj.TYPE_ADDR, Offset: 4}, Yu7},
		{&obj.Addr{Type: obj.TYPE_ADDR, Offset: 255}, Yu8},

		{immAddr(0), Yi0},
		{immAddr(1), Yi1},
		{immAddr(2), Yu2},
		{immAddr(3), Yu2},
		{immAddr(4), Yu7},
		{immAddr(86), Yu7},
		{immAddr(127), Yu7},
		{immAddr(128), Yu8},
		{immAddr(200), Yu8},
		{immAddr(255), Yu8},
		{immAddr(-1), Yi8},
		{immAddr(-100), Yi8},
		{immAddr(-128), Yi8},

		{regAddr(REG_AL), Yal},
		{regAddr(REG_AX), Yax},
		{regAddr(REG_DL), Yrb},
		{regAddr(REG_DH), Yrb},
		{regAddr(REG_BH), Yrb},
		{regAddr(REG_CL), Ycl},
		{regAddr(REG_CX), Ycx},
		{regAddr(REG_DX), Yrx},
		{regAddr(REG_BX), Yrx},
		{regAddr(REG_F0), Yf0},
		{regAddr(REG_F3), Yrf},
		{regAddr(REG_F7), Yrf},
		{regAddr(REG_M0), Ymr},
		{regAddr(REG_M3), Ymr},
		{regAddr(REG_M7), Ymr},
		{regAddr(REG_X0), Yxr0},
		{regAddr(REG_X6), Yxr},
		{regAddr(REG_X13), Yxr},
		{regAddr(REG_X20), YxrEvex},
		{regAddr(REG_X31), YxrEvex},
		{regAddr(REG_Y0), Yyr},
		{regAddr(REG_Y6), Yyr},
		{regAddr(REG_Y13), Yyr},
		{regAddr(REG_Y20), YyrEvex},
		{regAddr(REG_Y31), YyrEvex},
		{regAddr(REG_Z0), Yzr},
		{regAddr(REG_Z6), Yzr},
		{regAddr(REG_K0), Yk0},
		{regAddr(REG_K5), Yknot0},
		{regAddr(REG_K7), Yknot0},
		{regAddr(REG_CS), Ycs},
		{regAddr(REG_SS), Yss},
		{regAddr(REG_DS), Yds},
		{regAddr(REG_ES), Yes},
		{regAddr(REG_FS), Yfs},
		{regAddr(REG_GS), Ygs},
		{regAddr(REG_TLS), Ytls},
		{regAddr(REG_GDTR), Ygdtr},
		{regAddr(REG_IDTR), Yidtr},
		{regAddr(REG_LDTR), Yldtr},
		{regAddr(REG_MSW), Ymsw},
		{regAddr(REG_TASK), Ytask},
		{regAddr(REG_CR0), Ycr0},
		{regAddr(REG_CR5), Ycr5},
		{regAddr(REG_CR8), Ycr8},
		{regAddr(REG_DR0), Ydr0},
		{regAddr(REG_DR5), Ydr5},
		{regAddr(REG_DR7), Ydr7},
		{regAddr(REG_TR0), Ytr0},
		{regAddr(REG_TR5), Ytr5},
		{regAddr(REG_TR7), Ytr7},

		{regListAddr(REG_X0, REG_X3), YxrEvexMulti4},
		{regListAddr(REG_X4, REG_X7), YxrEvexMulti4},
		{regListAddr(REG_Y0, REG_Y3), YyrEvexMulti4},
		{regListAddr(REG_Y4, REG_Y7), YyrEvexMulti4},
		{regListAddr(REG_Z0, REG_Z3), YzrMulti4},
		{regListAddr(REG_Z4, REG_Z7), YzrMulti4},

		{memAddr(REG_AL, REG_NONE), Ym},
		{memAddr(REG_AL, REG_SI), Ym},
		{memAddr(REG_SI, REG_CX), Ym},
		{memAddr(REG_DI, REG_X0), Yxvm},
		{memAddr(REG_DI, REG_X7), Yxvm},
		{memAddr(REG_DI, REG_Y0), Yyvm},
		{memAddr(REG_DI, REG_Y7), Yyvm},
		{memAddr(REG_DI, REG_Z0), Yzvm},
		{memAddr(REG_DI, REG_Z7), Yzvm},
	}

	oclassTestsAMD64 = []*oclassTest{
		{immAddr(-200), Ys32},
		{immAddr(500), Ys32},
		{immAddr(0x7FFFFFFF), Ys32},
		{immAddr(0x7FFFFFFF + 1), Yi32},
		{immAddr(0xFFFFFFFF), Yi32},
		{immAddr(0xFFFFFFFF + 1), Yi64},

		{regAddr(REG_BPB), Yrb},
		{regAddr(REG_SIB), Yrb},
		{regAddr(REG_DIB), Yrb},
		{regAddr(REG_R8B), Yrb},
		{regAddr(REG_R12B), Yrb},
		{regAddr(REG_R8), Yrl},
		{regAddr(REG_R13), Yrl},
		{regAddr(REG_R15), Yrl},
		{regAddr(REG_SP), Yrl},
		{regAddr(REG_SI), Yrl},
		{regAddr(REG_DI), Yrl},
		{regAddr(REG_Z13), Yzr},
		{regAddr(REG_Z20), Yzr},
		{regAddr(REG_Z31), Yzr},

		{regListAddr(REG_X10, REG_X13), YxrEvexMulti4},
		{regListAddr(REG_X24, REG_X27), YxrEvexMulti4},
		{regListAddr(REG_Y10, REG_Y13), YyrEvexMulti4},
		{regListAddr(REG_Y24, REG_Y27), YyrEvexMulti4},
		{regListAddr(REG_Z10, REG_Z13), YzrMulti4},
		{regListAddr(REG_Z24, REG_Z27), YzrMulti4},

		{memAddr(REG_DI, REG_X20), YxvmEvex},
		{memAddr(REG_DI, REG_X27), YxvmEvex},
		{memAddr(REG_DI, REG_Y20), YyvmEvex},
		{memAddr(REG_DI, REG_Y27), YyvmEvex},
		{memAddr(REG_DI, REG_Z20), Yzvm},
		{memAddr(REG_DI, REG_Z27), Yzvm},
	}

	oclassTests386 = []*oclassTest{
		{&obj.Addr{Type: obj.TYPE_ADDR, Name: obj.NAME_EXTERN, Sym: &obj.LSym{}}, Yi32},

		{immAddr(-200), Yi32},

		{regAddr(REG_SP), Yrl32},
		{regAddr(REG_SI), Yrl32},
		{regAddr(REG_DI), Yrl32},
	}

	// Add tests that are arch-independent for all sets.
	oclassTestsAMD64 = append(oclassTestsAMD64, oclassTestsCommon...)
	oclassTests386 = append(oclassTests386, oclassTestsCommon...)
}

func TestOclass(t *testing.T) {
	runTest := func(t *testing.T, ctxt *obj.Link, tests []*oclassTest) {
		var p obj.Prog
		for _, test := range tests {
			have := oclass(ctxt, &p, test.arg)
			if have != test.want {
				t.Errorf("oclass(%q):\nhave: %d\nwant: %d",
					obj.Dconv(&p, test.arg), have, test.want)
			}
		}
	}

	// TODO(quasilyte): test edge cases for Hsolaris, etc?

	t.Run("linux/AMD64", func(t *testing.T) {
		ctxtAMD64 := obj.Linknew(&Linkamd64)
		ctxtAMD64.Headtype = objabi.Hlinux // See #32028
		runTest(t, ctxtAMD64, oclassTestsAMD64)
	})

	t.Run("linux/386", func(t *testing.T) {
		ctxt386 := obj.Linknew(&Link386)
		ctxt386.Headtype = objabi.Hlinux // See #32028
		runTest(t, ctxt386, oclassTests386)
	})
}

func TestRegisterListEncDec(t *testing.T) {
	tests := []struct {
		printed string
		reg0    int16
		reg1    int16
	}{
		{"[R10-R13]", REG_R10, REG_R13},
		{"[X0-AX]", REG_X0, REG_AX},

		{"[X0-X3]", REG_X0, REG_X3},
		{"[X21-X24]", REG_X21, REG_X24},

		{"[Y0-Y3]", REG_Y0, REG_Y3},
		{"[Y21-Y24]", REG_Y21, REG_Y24},

		{"[Z0-Z3]", REG_Z0, REG_Z3},
		{"[Z21-Z24]", REG_Z21, REG_Z24},
	}

	for _, test := range tests {
		enc := EncodeRegisterRange(test.reg0, test.reg1)
		reg0, reg1 := decodeRegisterRange(enc)

		if int16(reg0) != test.reg0 {
			t.Errorf("%s reg0 mismatch: have %d, want %d",
				test.printed, reg0, test.reg0)
		}
		if int16(reg1) != test.reg1 {
			t.Errorf("%s reg1 mismatch: have %d, want %d",
				test.printed, reg1, test.reg1)
		}
		wantPrinted := test.printed
		if rlconv(enc) != wantPrinted {
			t.Errorf("%s string mismatch: have %s, want %s",
				test.printed, rlconv(enc), wantPrinted)
		}
	}
}

func TestRegIndex(t *testing.T) {
	tests := []struct {
		regFrom int
		regTo   int
	}{
		{REG_AL, REG_R15B},
		{REG_AX, REG_R15},
		{REG_M0, REG_M7},
		{REG_K0, REG_K7},
		{REG_X0, REG_X31},
		{REG_Y0, REG_Y31},
		{REG_Z0, REG_Z31},
	}

	for _, test := range tests {
		for index, reg := 0, test.regFrom; reg <= test.regTo; index, reg = index+1, reg+1 {
			have := regIndex(int16(reg))
			want := index
			if have != want {
				regName := rconv(int(reg))
				t.Errorf("regIndex(%s):\nhave: %d\nwant: %d",
					regName, have, want)
			}
		}
	}
}

// TestPCALIGN verifies the correctness of the PCALIGN by checking if the
// code can be aligned to the alignment value.
func TestPCALIGN(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	tmpfile := filepath.Join(dir, "test.s")
	tmpout := filepath.Join(dir, "test.o")

	var testCases = []struct {
		name string
		code string
		out  string
	}{
		{
			name: "8-byte alignment",
			code: "TEXT ·foo(SB),$0-0\nMOVQ $0, AX\nPCALIGN $8\nMOVQ $1, BX\nRET\n",
			out:  `0x0008\s00008\s\(.*\)\tMOVQ\t\$1,\sBX`,
		},
		{
			name: "16-byte alignment",
			code: "TEXT ·foo(SB),$0-0\nMOVQ $0, AX\nPCALIGN $16\nMOVQ $2, CX\nRET\n",
			out:  `0x0010\s00016\s\(.*\)\tMOVQ\t\$2,\sCX`,
		},
	}

	for _, test := range testCases {
		if err := os.WriteFile(tmpfile, []byte(test.code), 0644); err != nil {
			t.Fatal(err)
		}
		cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-S", "-o", tmpout, tmpfile)
		cmd.Env = append(os.Environ(), "GOARCH=amd64", "GOOS=linux")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Errorf("The %s build failed: %v, output: %s", test.name, err, out)
			continue
		}

		matched, err := regexp.MatchString(test.out, string(out))
		if err != nil {
			t.Fatal(err)
		}
		if !matched {
			t.Errorf("The %s testing failed!\ninput: %s\noutput: %s\n", test.name, test.code, out)
		}
	}
}
