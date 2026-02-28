// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// When using GOEXPERIMENT=boringcrypto, the test program links in the boringcrypto syso,
// which does not respect GOAMD64, so we skip the test if boringcrypto is enabled.
//go:build !boringcrypto

package amd64_test

import (
	"bufio"
	"debug/elf"
	"debug/macho"
	"errors"
	"fmt"
	"go/build"
	"internal/testenv"
	"io"
	"math"
	"math/bits"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

// Test to make sure that when building for GOAMD64=v1, we don't
// use any >v1 instructions.
func TestGoAMD64v1(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("amd64-only test")
	}
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skip("test only works on elf or macho platforms")
	}
	for _, tag := range build.Default.ToolTags {
		if tag == "amd64.v2" {
			t.Skip("compiling for GOAMD64=v2 or higher")
		}
	}
	if os.Getenv("TESTGOAMD64V1") != "" {
		t.Skip("recursive call")
	}

	// Make a binary which will be a modified version of the
	// currently running binary.
	dst, err := os.CreateTemp("", "TestGoAMD64v1")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer os.Remove(dst.Name())
	dst.Chmod(0500) // make executable

	// Clobber all the non-v1 opcodes.
	opcodes := map[string]bool{}
	var features []string
	for feature, opcodeList := range featureToOpcodes {
		if runtimeFeatures[feature] {
			features = append(features, fmt.Sprintf("cpu.%s=off", feature))
		}
		for _, op := range opcodeList {
			opcodes[op] = true
		}
	}
	clobber(t, os.Args[0], dst, opcodes)
	if err = dst.Close(); err != nil {
		t.Fatalf("can't close binary: %v", err)
	}

	// Run the resulting binary.
	cmd := testenv.Command(t, dst.Name())
	testenv.CleanCmdEnv(cmd)
	cmd.Env = append(cmd.Env, "TESTGOAMD64V1=yes")
	// Disable FIPS 140-3 mode, since it would detect the modified binary.
	cmd.Env = append(cmd.Env, fmt.Sprintf("GODEBUG=%s,fips140=off", strings.Join(features, ",")))
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("couldn't execute test: %s\n%s", err, out)
	}
	// Expect to see output of the form "PASS\n", unless the test binary
	// was compiled for coverage (in which case there will be an extra line).
	success := false
	lines := strings.Split(string(out), "\n")
	if len(lines) == 2 {
		success = lines[0] == "PASS" && lines[1] == ""
	} else if len(lines) == 3 {
		success = lines[0] == "PASS" &&
			strings.HasPrefix(lines[1], "coverage") && lines[2] == ""
	}
	if !success {
		t.Fatalf("test reported error: %s lines=%+v", string(out), lines)
	}
}

// Clobber copies the binary src to dst, replacing all the instructions in opcodes with
// faulting instructions.
func clobber(t *testing.T, src string, dst *os.File, opcodes map[string]bool) {
	// Run objdump to get disassembly.
	var re *regexp.Regexp
	var disasm io.Reader
	if false {
		// TODO: go tool objdump doesn't disassemble the bmi1 instructions
		// in question correctly. See issue 48584.
		cmd := testenv.Command(t, "go", "tool", "objdump", src)
		var err error
		disasm, err = cmd.StdoutPipe()
		if err != nil {
			t.Fatal(err)
		}
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := cmd.Wait(); err != nil {
				t.Error(err)
			}
		})
		re = regexp.MustCompile(`^[^:]*:[-\d]+\s+0x([\da-f]+)\s+([\da-f]+)\s+([A-Z]+)`)
	} else {
		// TODO: we're depending on platform-native objdump here. Hence the Skipf
		// below if it doesn't run for some reason.
		cmd := testenv.Command(t, "objdump", "-d", src)
		var err error
		disasm, err = cmd.StdoutPipe()
		if err != nil {
			t.Fatal(err)
		}
		if err := cmd.Start(); err != nil {
			if errors.Is(err, exec.ErrNotFound) {
				t.Skipf("can't run test due to missing objdump: %s", err)
			}
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := cmd.Wait(); err != nil {
				t.Error(err)
			}
		})
		re = regexp.MustCompile(`^\s*([\da-f]+):\s*((?:[\da-f][\da-f] )+)\s*([a-z\d]+)`)
	}

	// Find all the instruction addresses we need to edit.
	virtualEdits := map[uint64]bool{}
	scanner := bufio.NewScanner(disasm)
	for scanner.Scan() {
		line := scanner.Text()
		parts := re.FindStringSubmatch(line)
		if len(parts) == 0 {
			continue
		}
		addr, err := strconv.ParseUint(parts[1], 16, 64)
		if err != nil {
			continue // not a hex address
		}
		opcode := strings.ToLower(parts[3])
		if !opcodes[opcode] {
			continue
		}
		t.Logf("clobbering instruction %s", line)
		n := (len(parts[2]) - strings.Count(parts[2], " ")) / 2 // number of bytes in instruction encoding
		for i := 0; i < n; i++ {
			// Only really need to make the first byte faulting, but might
			// as well make all the bytes faulting.
			virtualEdits[addr+uint64(i)] = true
		}
	}

	// Figure out where in the binary the edits must be done.
	physicalEdits := map[uint64]bool{}
	if e, err := elf.Open(src); err == nil {
		for _, sec := range e.Sections {
			vaddr := sec.Addr
			paddr := sec.Offset
			size := sec.Size
			for a := range virtualEdits {
				if a >= vaddr && a < vaddr+size {
					physicalEdits[paddr+(a-vaddr)] = true
				}
			}
		}
	} else if m, err2 := macho.Open(src); err2 == nil {
		for _, sec := range m.Sections {
			vaddr := sec.Addr
			paddr := uint64(sec.Offset)
			size := sec.Size
			for a := range virtualEdits {
				if a >= vaddr && a < vaddr+size {
					physicalEdits[paddr+(a-vaddr)] = true
				}
			}
		}
	} else {
		t.Log(err)
		t.Log(err2)
		t.Fatal("executable format not elf or macho")
	}
	if len(virtualEdits) != len(physicalEdits) {
		t.Fatal("couldn't find an instruction in text sections")
	}

	// Copy source to destination, making edits along the way.
	f, err := os.Open(src)
	if err != nil {
		t.Fatal(err)
	}
	r := bufio.NewReader(f)
	w := bufio.NewWriter(dst)
	a := uint64(0)
	done := 0
	for {
		b, err := r.ReadByte()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal("can't read")
		}
		if physicalEdits[a] {
			b = 0xcc // INT3 opcode
			done++
		}
		err = w.WriteByte(b)
		if err != nil {
			t.Fatal("can't write")
		}
		a++
	}
	if done != len(physicalEdits) {
		t.Fatal("physical edits remaining")
	}
	w.Flush()
	f.Close()
}

func setOf(keys ...string) map[string]bool {
	m := make(map[string]bool, len(keys))
	for _, key := range keys {
		m[key] = true
	}
	return m
}

var runtimeFeatures = setOf(
	"adx", "aes", "avx", "avx2", "bmi1", "bmi2", "erms", "fma",
	"pclmulqdq", "popcnt", "rdtscp", "sse3", "sse41", "sse42", "ssse3",
)

var featureToOpcodes = map[string][]string{
	// Note: we include *q, *l, and plain opcodes here.
	// go tool objdump doesn't include a [QL] on popcnt instructions, until CL 351889
	// native objdump doesn't include [QL] on linux.
	"popcnt": {"popcntq", "popcntl", "popcnt"},
	"bmi1": {
		"andnq", "andnl", "andn",
		"blsiq", "blsil", "blsi",
		"blsmskq", "blsmskl", "blsmsk",
		"blsrq", "blsrl", "blsr",
		"tzcntq", "tzcntl", "tzcnt",
	},
	"bmi2": {
		"sarxq", "sarxl", "sarx",
		"shlxq", "shlxl", "shlx",
		"shrxq", "shrxl", "shrx",
	},
	"sse41": {
		"roundsd",
		"pinsrq", "pinsrl", "pinsrd", "pinsrb", "pinsr",
		"pextrq", "pextrl", "pextrd", "pextrb", "pextr",
		"pminsb", "pminsd", "pminuw", "pminud", // Note: ub and sw are ok.
		"pmaxsb", "pmaxsd", "pmaxuw", "pmaxud",
		"pmovzxbw", "pmovzxbd", "pmovzxbq", "pmovzxwd", "pmovzxwq", "pmovzxdq",
		"pmovsxbw", "pmovsxbd", "pmovsxbq", "pmovsxwd", "pmovsxwq", "pmovsxdq",
		"pblendvb",
	},
	"fma":   {"vfmadd231sd"},
	"movbe": {"movbeqq", "movbeq", "movbell", "movbel", "movbe"},
	"lzcnt": {"lzcntq", "lzcntl", "lzcnt"},
}

// Test to use POPCNT instruction, if available
func TestPopCnt(t *testing.T) {
	for _, tt := range []struct {
		x    uint64
		want int
	}{
		{0b00001111, 4},
		{0b00001110, 3},
		{0b00001100, 2},
		{0b00000000, 0},
	} {
		if got := bits.OnesCount64(tt.x); got != tt.want {
			t.Errorf("OnesCount64(%#x) = %d, want %d", tt.x, got, tt.want)
		}
		if got := bits.OnesCount32(uint32(tt.x)); got != tt.want {
			t.Errorf("OnesCount32(%#x) = %d, want %d", tt.x, got, tt.want)
		}
	}
}

// Test to use ANDN, if available
func TestAndNot(t *testing.T) {
	for _, tt := range []struct {
		x, y, want uint64
	}{
		{0b00001111, 0b00000011, 0b1100},
		{0b00001111, 0b00001100, 0b0011},
		{0b00000000, 0b00000000, 0b0000},
	} {
		if got := tt.x &^ tt.y; got != tt.want {
			t.Errorf("%#x &^ %#x = %#x, want %#x", tt.x, tt.y, got, tt.want)
		}
		if got := uint32(tt.x) &^ uint32(tt.y); got != uint32(tt.want) {
			t.Errorf("%#x &^ %#x = %#x, want %#x", tt.x, tt.y, got, tt.want)
		}
	}
}

// Test to use BLSI, if available
func TestBLSI(t *testing.T) {
	for _, tt := range []struct {
		x, want uint64
	}{
		{0b00001111, 0b001},
		{0b00001110, 0b010},
		{0b00001100, 0b100},
		{0b11000110, 0b010},
		{0b00000000, 0b000},
	} {
		if got := tt.x & -tt.x; got != tt.want {
			t.Errorf("%#x & (-%#x) = %#x, want %#x", tt.x, tt.x, got, tt.want)
		}
		if got := uint32(tt.x) & -uint32(tt.x); got != uint32(tt.want) {
			t.Errorf("%#x & (-%#x) = %#x, want %#x", tt.x, tt.x, got, tt.want)
		}
	}
}

// Test to use BLSMSK, if available
func TestBLSMSK(t *testing.T) {
	for _, tt := range []struct {
		x, want uint64
	}{
		{0b00001111, 0b001},
		{0b00001110, 0b011},
		{0b00001100, 0b111},
		{0b11000110, 0b011},
		{0b00000000, 1<<64 - 1},
	} {
		if got := tt.x ^ (tt.x - 1); got != tt.want {
			t.Errorf("%#x ^ (%#x-1) = %#x, want %#x", tt.x, tt.x, got, tt.want)
		}
		if got := uint32(tt.x) ^ (uint32(tt.x) - 1); got != uint32(tt.want) {
			t.Errorf("%#x ^ (%#x-1) = %#x, want %#x", tt.x, tt.x, got, uint32(tt.want))
		}
	}
}

// Test to use BLSR, if available
func TestBLSR(t *testing.T) {
	for _, tt := range []struct {
		x, want uint64
	}{
		{0b00001111, 0b00001110},
		{0b00001110, 0b00001100},
		{0b00001100, 0b00001000},
		{0b11000110, 0b11000100},
		{0b00000000, 0b00000000},
	} {
		if got := tt.x & (tt.x - 1); got != tt.want {
			t.Errorf("%#x & (%#x-1) = %#x, want %#x", tt.x, tt.x, got, tt.want)
		}
		if got := uint32(tt.x) & (uint32(tt.x) - 1); got != uint32(tt.want) {
			t.Errorf("%#x & (%#x-1) = %#x, want %#x", tt.x, tt.x, got, tt.want)
		}
	}
}

func TestTrailingZeros(t *testing.T) {
	for _, tt := range []struct {
		x    uint64
		want int
	}{
		{0b00001111, 0},
		{0b00001110, 1},
		{0b00001100, 2},
		{0b00001000, 3},
		{0b00000000, 64},
	} {
		if got := bits.TrailingZeros64(tt.x); got != tt.want {
			t.Errorf("TrailingZeros64(%#x) = %d, want %d", tt.x, got, tt.want)
		}
		want := tt.want
		if want == 64 {
			want = 32
		}
		if got := bits.TrailingZeros32(uint32(tt.x)); got != want {
			t.Errorf("TrailingZeros64(%#x) = %d, want %d", tt.x, got, want)
		}
	}
}

func TestRound(t *testing.T) {
	for _, tt := range []struct {
		x, want float64
	}{
		{1.4, 1},
		{1.5, 2},
		{1.6, 2},
		{2.4, 2},
		{2.5, 2},
		{2.6, 3},
	} {
		if got := math.RoundToEven(tt.x); got != tt.want {
			t.Errorf("RoundToEven(%f) = %f, want %f", tt.x, got, tt.want)
		}
	}
}

func TestFMA(t *testing.T) {
	for _, tt := range []struct {
		x, y, z, want float64
	}{
		{2, 3, 4, 10},
		{3, 4, 5, 17},
	} {
		if got := math.FMA(tt.x, tt.y, tt.z); got != tt.want {
			t.Errorf("FMA(%f,%f,%f) = %f, want %f", tt.x, tt.y, tt.z, got, tt.want)
		}
	}
}
