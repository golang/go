// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

func TestSplitImm24uScaled(t *testing.T) {
	tests := []struct {
		v       int32
		shift   int
		wantErr bool
		wantHi  int32
		wantLo  int32
	}{
		{
			v:      0,
			shift:  0,
			wantHi: 0,
			wantLo: 0,
		},
		{
			v:      0x1001,
			shift:  0,
			wantHi: 0x1000,
			wantLo: 0x1,
		},
		{
			v:      0xffffff,
			shift:  0,
			wantHi: 0xfff000,
			wantLo: 0xfff,
		},
		{
			v:       0xffffff,
			shift:   1,
			wantErr: true,
		},
		{
			v:      0xfe,
			shift:  1,
			wantHi: 0x0,
			wantLo: 0x7f,
		},
		{
			v:      0x10fe,
			shift:  1,
			wantHi: 0x0,
			wantLo: 0x87f,
		},
		{
			v:      0x2002,
			shift:  1,
			wantHi: 0x2000,
			wantLo: 0x1,
		},
		{
			v:      0xfffffe,
			shift:  1,
			wantHi: 0xffe000,
			wantLo: 0xfff,
		},
		{
			v:      0x1000ffe,
			shift:  1,
			wantHi: 0xfff000,
			wantLo: 0xfff,
		},
		{
			v:       0x1001000,
			shift:   1,
			wantErr: true,
		},
		{
			v:       0xfffffe,
			shift:   2,
			wantErr: true,
		},
		{
			v:      0x4004,
			shift:  2,
			wantHi: 0x4000,
			wantLo: 0x1,
		},
		{
			v:      0xfffffc,
			shift:  2,
			wantHi: 0xffc000,
			wantLo: 0xfff,
		},
		{
			v:      0x1002ffc,
			shift:  2,
			wantHi: 0xfff000,
			wantLo: 0xfff,
		},
		{
			v:       0x1003000,
			shift:   2,
			wantErr: true,
		},
		{
			v:       0xfffffe,
			shift:   3,
			wantErr: true,
		},
		{
			v:      0x8008,
			shift:  3,
			wantHi: 0x8000,
			wantLo: 0x1,
		},
		{
			v:      0xfffff8,
			shift:  3,
			wantHi: 0xff8000,
			wantLo: 0xfff,
		},
		{
			v:      0x1006ff8,
			shift:  3,
			wantHi: 0xfff000,
			wantLo: 0xfff,
		},
		{
			v:       0x1007000,
			shift:   3,
			wantErr: true,
		},
	}
	for _, test := range tests {
		hi, lo, err := splitImm24uScaled(test.v, test.shift)
		switch {
		case err == nil && test.wantErr:
			t.Errorf("splitImm24uScaled(%v, %v) succeeded, want error", test.v, test.shift)
		case err != nil && !test.wantErr:
			t.Errorf("splitImm24uScaled(%v, %v) failed: %v", test.v, test.shift, err)
		case !test.wantErr:
			if got, want := hi, test.wantHi; got != want {
				t.Errorf("splitImm24uScaled(%x, %x) - got hi %x, want %x", test.v, test.shift, got, want)
			}
			if got, want := lo, test.wantLo; got != want {
				t.Errorf("splitImm24uScaled(%x, %x) - got lo %x, want %x", test.v, test.shift, got, want)
			}
		}
	}
	for shift := 0; shift <= 3; shift++ {
		for v := int32(0); v < 0xfff000+0xfff<<shift; v = v + 1<<shift {
			hi, lo, err := splitImm24uScaled(v, shift)
			if err != nil {
				t.Fatalf("splitImm24uScaled(%x, %x) failed: %v", v, shift, err)
			}
			if hi+lo<<shift != v {
				t.Fatalf("splitImm24uScaled(%x, %x) = (%x, %x) is incorrect", v, shift, hi, lo)
			}
		}
	}
}

// TestLarge generates a very large file to verify that large
// program builds successfully, in particular, too-far
// conditional branches are fixed, and also verify that the
// instruction's pc can be correctly aligned even when branches
// need to be fixed.
func TestLarge(t *testing.T) {
	if testing.Short() {
		t.Skip("Skip in short mode")
	}
	testenv.MustHaveGoBuild(t)

	dir, err := os.MkdirTemp("", "testlarge")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// generate a very large function
	buf := bytes.NewBuffer(make([]byte, 0, 7000000))
	gen(buf)

	tmpfile := filepath.Join(dir, "x.s")
	err = os.WriteFile(tmpfile, buf.Bytes(), 0644)
	if err != nil {
		t.Fatalf("can't write output: %v\n", err)
	}

	pattern := `0x0080\s00128\s\(.*\)\tMOVD\t\$3,\sR3`

	// assemble generated file
	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-S", "-o", filepath.Join(dir, "test.o"), tmpfile)
	cmd.Env = append(os.Environ(), "GOOS=linux")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("Assemble failed: %v, output: %s", err, out)
	}
	matched, err := regexp.MatchString(pattern, string(out))
	if err != nil {
		t.Fatal(err)
	}
	if !matched {
		t.Errorf("The alignment is not correct: %t, output:%s\n", matched, out)
	}

	// build generated file
	cmd = testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-o", filepath.Join(dir, "x.o"), tmpfile)
	cmd.Env = append(os.Environ(), "GOOS=linux")
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Errorf("Build failed: %v, output: %s", err, out)
	}
}

// gen generates a very large program, with a very far conditional branch.
func gen(buf *bytes.Buffer) {
	fmt.Fprintln(buf, "TEXT f(SB),0,$0-0")
	fmt.Fprintln(buf, "TBZ $5, R0, label")
	fmt.Fprintln(buf, "CBZ R0, label")
	fmt.Fprintln(buf, "BEQ label")
	fmt.Fprintln(buf, "PCALIGN $128")
	fmt.Fprintln(buf, "MOVD $3, R3")
	for i := 0; i < 1<<19; i++ {
		fmt.Fprintln(buf, "MOVD R0, R1")
	}
	fmt.Fprintln(buf, "label:")
	fmt.Fprintln(buf, "RET")
}

// Issue 20348.
func TestNoRet(t *testing.T) {
	dir, err := os.MkdirTemp("", "testnoret")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	tmpfile := filepath.Join(dir, "x.s")
	if err := os.WriteFile(tmpfile, []byte("TEXT ·stub(SB),$0-0\nNOP\n"), 0644); err != nil {
		t.Fatal(err)
	}
	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-o", filepath.Join(dir, "x.o"), tmpfile)
	cmd.Env = append(os.Environ(), "GOOS=linux")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Errorf("%v\n%s", err, out)
	}
}

// TestPCALIGN verifies the correctness of the PCALIGN by checking if the
// code can be aligned to the alignment value.
func TestPCALIGN(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir, err := os.MkdirTemp("", "testpcalign")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	tmpfile := filepath.Join(dir, "test.s")
	tmpout := filepath.Join(dir, "test.o")

	code1 := []byte("TEXT ·foo(SB),$0-0\nMOVD $0, R0\nPCALIGN $8\nMOVD $1, R1\nRET\n")
	code2 := []byte("TEXT ·foo(SB),$0-0\nMOVD $0, R0\nPCALIGN $16\nMOVD $2, R2\nRET\n")
	// If the output contains this pattern, the pc-offsite of "MOVD $1, R1" is 8 bytes aligned.
	out1 := `0x0008\s00008\s\(.*\)\tMOVD\t\$1,\sR1`
	// If the output contains this pattern, the pc-offsite of "MOVD $2, R2" is 16 bytes aligned.
	out2 := `0x0010\s00016\s\(.*\)\tMOVD\t\$2,\sR2`
	var testCases = []struct {
		name string
		code []byte
		out  string
	}{
		{"8-byte alignment", code1, out1},
		{"16-byte alignment", code2, out2},
	}

	for _, test := range testCases {
		if err := os.WriteFile(tmpfile, test.code, 0644); err != nil {
			t.Fatal(err)
		}
		cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-S", "-o", tmpout, tmpfile)
		cmd.Env = append(os.Environ(), "GOOS=linux")
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

func testvmovs() (r1, r2 uint64)
func testvmovd() (r1, r2 uint64)
func testvmovq() (r1, r2 uint64)

func TestVMOV(t *testing.T) {
	tests := []struct {
		op           string
		vmovFunc     func() (uint64, uint64)
		wantA, wantB uint64
	}{
		{"VMOVS", testvmovs, 0x80402010, 0},
		{"VMOVD", testvmovd, 0x7040201008040201, 0},
		{"VMOVQ", testvmovq, 0x7040201008040201, 0x3040201008040201},
	}
	for _, test := range tests {
		gotA, gotB := test.vmovFunc()
		if gotA != test.wantA || gotB != test.wantB {
			t.Errorf("%v: got: a=0x%x, b=0x%x, want: a=0x%x, b=0x%x", test.op, gotA, gotB, test.wantA, test.wantB)
		}
	}
}

func testmovk() uint64

// TestMOVK makes sure MOVK with a very large constant works. See issue 52261.
func TestMOVK(t *testing.T) {
	x := testmovk()
	want := uint64(40000 << 48)
	if x != want {
		t.Errorf("Got %x want %x\n", x, want)
	}
}
