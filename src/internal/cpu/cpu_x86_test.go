// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 386 amd64

package cpu_test

import (
	. "internal/cpu"
	"os"
	"runtime"
	"testing"
)

func TestX86ifAVX2hasAVX(t *testing.T) {
	if X86.HasAVX2 && !X86.HasAVX {
		t.Fatalf("HasAVX expected true when HasAVX2 is true, got false")
	}
}

func TestDisableSSE2(t *testing.T) {
	runDebugOptionsTest(t, "TestSSE2DebugOption", "cpu.sse2=off")
}

func TestSSE2DebugOption(t *testing.T) {
	MustHaveDebugOptionsSupport(t)

	if os.Getenv("GODEBUG") != "cpu.sse2=off" {
		t.Skipf("skipping test: GODEBUG=cpu.sse2=off not set")
	}

	want := runtime.GOARCH != "386" // SSE2 can only be disabled on 386.
	if got := X86.HasSSE2; got != want {
		t.Errorf("X86.HasSSE2 on %s expected %v, got %v", runtime.GOARCH, want, got)
	}
}

func TestDisableSSE3(t *testing.T) {
	runDebugOptionsTest(t, "TestSSE3DebugOption", "cpu.sse3=off")
}

func TestSSE3DebugOption(t *testing.T) {
	MustHaveDebugOptionsSupport(t)

	if os.Getenv("GODEBUG") != "cpu.sse3=off" {
		t.Skipf("skipping test: GODEBUG=cpu.sse3=off not set")
	}

	want := false
	if got := X86.HasSSE3; got != want {
		t.Errorf("X86.HasSSE3 expected %v, got %v", want, got)
	}
}
