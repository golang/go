// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 386 amd64 amd64p32

package cpu_test

import (
	. "internal/cpu"
	"os"
	"runtime"
	"testing"
)

func TestAMD64minimalFeatures(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		return
	}

	if !X86.HasSSE2 {
		t.Fatalf("HasSSE2 expected true, got false")
	}
}

func TestX86ifAVX2hasAVX(t *testing.T) {
	if X86.HasAVX2 && !X86.HasAVX {
		t.Fatalf("HasAVX expected true when HasAVX2 is true, got false")
	}
}

func TestDisableSSE2(t *testing.T) {
	runDebugOptionsTest(t, "TestSSE2DebugOption", "sse2=0")
}

func TestSSE2DebugOption(t *testing.T) {
	MustHaveDebugOptionsEnabled(t)

	if os.Getenv("GODEBUGCPU") != "sse2=0" {
		t.Skipf("skipping test: GODEBUGCPU=sse2=0 not set")
	}

	want := runtime.GOARCH != "386" // SSE2 can only be disabled on 386.
	if got := X86.HasSSE2; got != want {
		t.Errorf("X86.HasSSE2 on %s expected %v, got %v", runtime.GOARCH, want, got)
	}
}
