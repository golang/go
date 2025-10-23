// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"internal/testenv"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

// This is the function we'll be testing.
// It has a simple write barrier in it.
func setGlobalPointer() {
	globalPointer = nil
}

var globalPointer *int

func TestUnsafePoint(t *testing.T) {
	testenv.MustHaveExec(t)
	switch runtime.GOARCH {
	case "amd64", "arm64":
	default:
		t.Skipf("test not enabled for %s", runtime.GOARCH)
	}

	// Get a reference we can use to ask the runtime about
	// which of its instructions are unsafe preemption points.
	f := runtime.FuncForPC(reflect.ValueOf(setGlobalPointer).Pointer())

	// Disassemble the test function.
	// Note that normally "go test runtime" would strip symbols
	// and prevent this step from working. So there's a hack in
	// cmd/go/internal/test that exempts runtime tests from
	// symbol stripping.
	cmd := exec.Command(testenv.GoToolPath(t), "tool", "objdump", "-s", "setGlobalPointer", os.Args[0])
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("can't objdump %v:\n%s", err, out)
	}
	lines := strings.Split(string(out), "\n")[1:]

	// Walk through assembly instructions, checking preemptible flags.
	var entry uint64
	var startedWB bool
	var doneWB bool
	instructionCount := 0
	unsafeCount := 0
	for _, line := range lines {
		line = strings.TrimSpace(line)
		t.Logf("%s", line)
		parts := strings.Fields(line)
		if len(parts) < 4 {
			continue
		}
		if !strings.HasPrefix(parts[0], "unsafepoint_test.go:") {
			continue
		}
		pc, err := strconv.ParseUint(parts[1][2:], 16, 64)
		if err != nil {
			t.Fatalf("can't parse pc %s: %v", parts[1], err)
		}
		if entry == 0 {
			entry = pc
		}
		// Note that some platforms do ASLR, so the PCs in the disassembly
		// don't match PCs in the address space. Only offsets from function
		// entry make sense.
		unsafe := runtime.UnsafePoint(f.Entry() + uintptr(pc-entry))
		t.Logf("unsafe: %v\n", unsafe)
		instructionCount++
		if unsafe {
			unsafeCount++
		}

		// All the instructions inside the write barrier must be unpreemptible.
		if startedWB && !doneWB && !unsafe {
			t.Errorf("instruction %s must be marked unsafe, but isn't", parts[1])
		}

		// Detect whether we're in the write barrier.
		switch runtime.GOARCH {
		case "arm64":
			if parts[3] == "MOVWU" {
				// The unpreemptible region starts after the
				// load of runtime.writeBarrier.
				startedWB = true
			}
			if parts[3] == "MOVD" && parts[4] == "ZR," {
				// The unpreemptible region ends after the
				// write of nil.
				doneWB = true
			}
		case "amd64":
			if parts[3] == "CMPL" {
				startedWB = true
			}
			if parts[3] == "MOVQ" && parts[4] == "$0x0," {
				doneWB = true
			}
		}
	}

	if instructionCount == 0 {
		t.Errorf("no instructions")
	}
	if unsafeCount == instructionCount {
		t.Errorf("no interruptible instructions")
	}
	// Note that there are other instructions marked unpreemptible besides
	// just the ones required by the write barrier. Those include possibly
	// the preamble and postamble, as well as bleeding out from the
	// write barrier proper into adjacent instructions (in both directions).
	// Hopefully we can clean up the latter at some point.
}
