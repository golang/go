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
