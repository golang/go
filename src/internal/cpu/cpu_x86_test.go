// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386 || amd64

package cpu_test

import (
	. "internal/cpu"
	"internal/godebug"
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

func TestX86ifGFNIhasAVX(t *testing.T) {
	if X86.HasGFNI && !X86.HasAVX {
		t.Fatalf("HasAVX expected true when HasGFNI is true, got false")
	}
}

func TestX86ifAVX512GFNIhasGFNI(t *testing.T) {
	// Skip if GODEBUG explicitly disabled gfni (directly or via cpu.all=off);
	// processOptions would have cleared X86.HasGFNI but not X86.HasAVX512GFNI
	// (which is not in the options table), so the invariant below would
	// spuriously fire even though the hardware reports both bits.
	if godebug.New("#cpu.gfni").Value() == "off" || godebug.New("#cpu.all").Value() == "off" {
		t.Skip("skipping test: GODEBUG=cpu.gfni=off set")
	}
	if X86.HasAVX512GFNI && !X86.HasGFNI {
		t.Fatalf("HasGFNI expected true when HasAVX512GFNI is true, got false")
	}
}

func TestX86ifCPUIDGFNIhasGFNI(t *testing.T) {
	if !X86.HasAVX {
		t.Skip("skipping test: requires AVX (and thus YMM OS support)")
	}
	// Skip if GODEBUG explicitly disabled gfni (directly or via cpu.all=off);
	// processOptions would have cleared X86.HasGFNI regardless of hardware
	// state, so the CPUID comparison below would spuriously fire.
	if godebug.New("#cpu.gfni").Value() == "off" || godebug.New("#cpu.all").Value() == "off" {
		t.Skip("skipping test: GODEBUG=cpu.gfni=off set")
	}
	maxID, _, _, _ := Cpuid(0, 0)
	if maxID < 7 {
		t.Skip("skipping test: CPUID leaf 7 unavailable")
	}
	_, _, ecx7, _ := Cpuid(7, 0)
	const gfniBit = 1 << 8
	if ecx7&gfniBit != 0 && !X86.HasGFNI {
		t.Fatalf("HasGFNI expected true when CPUID leaf 7 ECX bit 8 is set, got false")
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

func TestDisableGFNI(t *testing.T) {
	if !X86.HasGFNI {
		t.Skip("skipping test: host does not advertise GFNI")
	}
	runDebugOptionsTest(t, "TestGFNIDebugOption", "cpu.gfni=off")
}

func TestGFNIDebugOption(t *testing.T) {
	MustHaveDebugOptionsSupport(t)

	if godebug.New("#cpu.gfni").Value() != "off" {
		t.Skipf("skipping test: GODEBUG=cpu.gfni=off not set")
	}

	want := false
	if got := X86.HasGFNI; got != want {
		t.Errorf("X86.HasGFNI expected %v, got %v", want, got)
	}
}
