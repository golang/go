// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386 || amd64

package cpu_test

import (
	. "internal/cpu"
	"internal/godebug"
	"internal/testenv"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
)

func TestX86ifAVX2hasAVX(t *testing.T) {
	if X86.HasAVX2 && !X86.HasAVX {
		t.Fatalf("HasAVX expected true when HasAVX2 is true, got false")
	}
}

func TestX86ifAVX512FhasAVX2(t *testing.T) {
	if X86.HasAVX512F && !X86.HasAVX2 {
		t.Fatalf("HasAVX2 expected true when HasAVX512F is true, got false")
	}
}

func TestX86ifAVX512BWhasAVX512F(t *testing.T) {
	if X86.HasAVX512BW && !X86.HasAVX512F {
		t.Fatalf("HasAVX512F expected true when HasAVX512BW is true, got false")
	}
}

func TestX86ifAVX512VLhasAVX512F(t *testing.T) {
	if X86.HasAVX512VL && !X86.HasAVX512F {
		t.Fatalf("HasAVX512F expected true when HasAVX512VL is true, got false")
	}
}

func TestDisableSSE3(t *testing.T) {
	if GetGOAMD64level() > 1 {
		t.Skip("skipping test: can't run on GOAMD64>v1 machines")
	}
	runDebugOptionsTest(t, "TestSSE3DebugOption", "cpu.sse3=off")
}

func TestSSE3DebugOption(t *testing.T) {
	MustHaveDebugOptionsSupport(t)

	if godebug.New("#cpu.sse3").Value() != "off" {
		t.Skipf("skipping test: GODEBUG=cpu.sse3=off not set")
	}

	want := false
	if got := X86.HasSSE3; got != want {
		t.Errorf("X86.HasSSE3 expected %v, got %v", want, got)
	}
}

func TestX86VPCLMULQDQImplications(t *testing.T) {
	if X86.HasVPCLMULQDQ && !(X86.HasAVX && X86.HasPCLMULQDQ) {
		t.Errorf("HasVPCLMULQDQ is true, but HasAVX=%v, HasPCLMULQDQ=%v", X86.HasAVX, X86.HasPCLMULQDQ)
	}
	if X86.HasAVX512VPCLMULQDQ && !(X86.HasAVX512 && X86.HasVPCLMULQDQ) {
		t.Errorf("HasAVX512VPCLMULQDQ is true, but HasAVX512=%v, HasVPCLMULQDQ=%v", X86.HasAVX512, X86.HasVPCLMULQDQ)
	}
}

// TestAVX512VPCLMULQDQDebugOption checks that GODEBUG=cpu.avx512vpclmulqdq=off,
// the name golang.org/x/sys/cpu uses for 512-bit VPCLMULQDQ, is accepted and
// turns off only the 512-bit flag.
func TestAVX512VPCLMULQDQDebugOption(t *testing.T) {
	MustHaveDebugOptionsSupport(t)

	const wantEnv = "GO_CPU_TEST_WANT_VPCLMULQDQ"
	if godebug.New("#cpu.avx512vpclmulqdq").Value() == "off" {
		// This is the child process, or the test was run with the
		// setting.
		if X86.HasAVX512VPCLMULQDQ {
			t.Errorf("X86.HasAVX512VPCLMULQDQ expected false, got true")
		}
		if want := os.Getenv(wantEnv); want != "" && strconv.FormatBool(X86.HasVPCLMULQDQ) != want {
			t.Errorf("X86.HasVPCLMULQDQ expected %s, got %v", want, X86.HasVPCLMULQDQ)
		}
		return
	}

	cmd := exec.Command(testenv.Executable(t), "-test.run=^TestAVX512VPCLMULQDQDebugOption$")
	cmd.Env = append(cmd.Env, "GODEBUG=cpu.avx512vpclmulqdq=off", wantEnv+"="+strconv.FormatBool(X86.HasVPCLMULQDQ))
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("with GODEBUG=cpu.avx512vpclmulqdq=off: %v\n%s", err, out)
	}
	// The package reports options it doesn't know as it initializes.
	if strings.Contains(string(out), "unknown cpu feature") {
		t.Errorf("with GODEBUG=cpu.avx512vpclmulqdq=off, the package complained about the setting:\n%s", out)
	}
}

func TestDisableCascade(t *testing.T) {
	tests := []struct {
		name     string
		maxLevel int32 // 0 means any GOAMD64 level
		opt      string
	}{
		{"aes", 0, "cpu.aes=off"},
		{"pclmulqdq", 0, "cpu.pclmulqdq=off"},
		{"vpclmulqdq", 0, "cpu.vpclmulqdq=off"},
		{"avx", 2, "cpu.avx=off"},
		{"avx2", 2, "cpu.avx2=off"},
		{"avx512f", 3, "cpu.avx512f=off"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.maxLevel > 0 && GetGOAMD64level() > tc.maxLevel {
				t.Skipf("skipping test: can't run on GOAMD64>v%d", tc.maxLevel)
			}
			runDebugOptionsTest(t, "TestDisableCascadeOption", tc.opt)
		})
	}
}

func checkNoAVX512(t *testing.T) {
	t.Helper()
	flags := []struct {
		name string
		val  bool
	}{
		{"HasAVX512", X86.HasAVX512},
		{"HasAVX512F", X86.HasAVX512F},
		{"HasAVX512CD", X86.HasAVX512CD},
		{"HasAVX512BW", X86.HasAVX512BW},
		{"HasAVX512DQ", X86.HasAVX512DQ},
		{"HasAVX512VL", X86.HasAVX512VL},
		{"HasAVX512GFNI", X86.HasAVX512GFNI},
		{"HasAVX512BITALG", X86.HasAVX512BITALG},
		{"HasAVX512VPOPCNTDQ", X86.HasAVX512VPOPCNTDQ},
		{"HasAVX512VBMI", X86.HasAVX512VBMI},
		{"HasAVX512VBMI2", X86.HasAVX512VBMI2},
		{"HasAVX512VAES", X86.HasAVX512VAES},
		{"HasAVX512VNNI", X86.HasAVX512VNNI},
		{"HasAVX512VPCLMULQDQ", X86.HasAVX512VPCLMULQDQ},
	}
	for _, f := range flags {
		if f.val {
			t.Errorf("X86.%s expected false, got true", f.name)
		}
	}
}

func TestDisableCascadeOption(t *testing.T) {
	MustHaveDebugOptionsSupport(t)

	checked := false
	if godebug.New("#cpu.aes").Value() == "off" {
		checked = true
		if X86.HasAES {
			t.Errorf("X86.HasAES expected false, got true")
		}
		if X86.HasVAES {
			t.Errorf("X86.HasVAES expected false, got true")
		}
		if X86.HasAVX512VAES {
			t.Errorf("X86.HasAVX512VAES expected false, got true")
		}
	}
	if godebug.New("#cpu.pclmulqdq").Value() == "off" {
		checked = true
		if X86.HasPCLMULQDQ {
			t.Errorf("X86.HasPCLMULQDQ expected false, got true")
		}
		if X86.HasVPCLMULQDQ {
			t.Errorf("X86.HasVPCLMULQDQ expected false, got true")
		}
		if X86.HasAVX512VPCLMULQDQ {
			t.Errorf("X86.HasAVX512VPCLMULQDQ expected false, got true")
		}
	}
	if godebug.New("#cpu.vpclmulqdq").Value() == "off" {
		checked = true
		// Disabling VPCLMULQDQ must disable it at every vector width.
		if X86.HasVPCLMULQDQ {
			t.Errorf("X86.HasVPCLMULQDQ expected false, got true")
		}
		if X86.HasAVX512VPCLMULQDQ {
			t.Errorf("X86.HasAVX512VPCLMULQDQ expected false, got true")
		}
	}
	if godebug.New("#cpu.avx").Value() == "off" {
		checked = true
		if X86.HasAVX {
			t.Errorf("X86.HasAVX expected false, got true")
		}
		if X86.HasAVX2 {
			t.Errorf("X86.HasAVX2 expected false, got true")
		}
		if X86.HasFMA {
			t.Errorf("X86.HasFMA expected false, got true")
		}
		if X86.HasVAES {
			t.Errorf("X86.HasVAES expected false, got true")
		}
		if X86.HasVPCLMULQDQ {
			t.Errorf("X86.HasVPCLMULQDQ expected false, got true")
		}
		if X86.HasAVXVNNI {
			t.Errorf("X86.HasAVXVNNI expected false, got true")
		}
		checkNoAVX512(t)
	}
	if godebug.New("#cpu.avx2").Value() == "off" {
		checked = true
		if X86.HasAVX2 {
			t.Errorf("X86.HasAVX2 expected false, got true")
		}
		if X86.HasAVXVNNI {
			t.Errorf("X86.HasAVXVNNI expected false, got true")
		}
		checkNoAVX512(t)
	}
	if godebug.New("#cpu.avx512f").Value() == "off" {
		checked = true
		checkNoAVX512(t)
	}
	if !checked {
		t.Skipf("skipping test: no cascade GODEBUG cpu option set")
	}
}
