// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"internal/buildcfg"
	"internal/platform"
	"internal/testenv"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestABIWrapperNoRaceInstrumentation verifies that compiler-generated
// ABI0->ABIInternal wrappers are not instrumented under -race.
func TestABIWrapperNoRaceInstrumentation(t *testing.T) {
	if !platform.RaceDetectorSupported(runtime.GOOS, runtime.GOARCH) {
		t.Skipf("race detector not supported on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	if !buildcfg.Experiment.RegabiWrappers {
		t.Skipf("ABI wrappers not enabled on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	testenv.MustHaveGoBuild(t)

	dir := t.TempDir()
	src := filepath.Join(dir, "x.go")
	const code = `package main

import _ "unsafe"

var sink int

//go:linkname F
func F(x int) int {
	sink = x
	return x + 1
}
`
	if err := os.WriteFile(src, []byte(code), 0644); err != nil {
		t.Fatal(err)
	}

	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "compile", "-race", "-p=main", "-S", "-o", filepath.Join(dir, "x.o"), src)
	outBytes, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("compile failed: %v\n%s", err, outBytes)
	}
	out := string(outBytes)

	lines := strings.Split(out, "\n")
	var sawInternal, sawWrapper bool
	for i := 0; i < len(lines); i++ {
		if !strings.HasPrefix(lines[i], "main.F STEXT") {
			continue
		}
		end := i + 1
		for end < len(lines) && strings.HasPrefix(lines[end], "\t") {
			end++
		}
		body := strings.Join(lines[i:end], "\n")
		hasRace := strings.Contains(body, "runtime.racefuncenter") || strings.Contains(body, "runtime.racefuncexit")
		if strings.Contains(body, "ABIWRAPPER") {
			sawWrapper = true
			if hasRace {
				t.Errorf("main.F ABI wrapper unexpectedly instrumented:\n%s", body)
			}
		} else {
			sawInternal = true
			if !hasRace {
				t.Errorf("main.F (ABIInternal) not instrumented:\n%s", body)
			}
		}
		i = end - 1
	}
	if !sawInternal {
		t.Fatalf("no ABIInternal assembly found for main.F:\n%s", out)
	}
	if !sawWrapper {
		t.Fatalf("no ABIWRAPPER assembly found for main.F:\n%s", out)
	}
}
