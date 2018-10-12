// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu_test

import (
	. "internal/cpu"
	"internal/testenv"
	"os"
	"os/exec"
	"strings"
	"testing"
)

func MustHaveDebugOptionsSupport(t *testing.T) {
	if !DebugOptions {
		t.Skipf("skipping test: cpu feature options not supported by OS")
	}
}

func runDebugOptionsTest(t *testing.T, test string, options string) {
	MustHaveDebugOptionsSupport(t)

	testenv.MustHaveExec(t)

	env := "GODEBUGCPU=" + options

	cmd := exec.Command(os.Args[0], "-test.run="+test)
	cmd.Env = append(cmd.Env, env)

	output, err := cmd.CombinedOutput()
	lines := strings.Fields(string(output))
	lastline := lines[len(lines)-1]

	got := strings.TrimSpace(lastline)
	want := "PASS"
	if err != nil || got != want {
		t.Fatalf("%s with %s: want %s, got %v", test, env, want, got)
	}
}

func TestDisableAllCapabilities(t *testing.T) {
	runDebugOptionsTest(t, "TestAllCapabilitiesDisabled", "all=off")
}

func TestAllCapabilitiesDisabled(t *testing.T) {
	MustHaveDebugOptionsSupport(t)

	if os.Getenv("GODEBUGCPU") != "all=off" {
		t.Skipf("skipping test: GODEBUGCPU=all=off not set")
	}

	for _, o := range Options {
		if got := *o.Feature; got != false {
			t.Errorf("%v: expected false, got %v", o.Name, got)
		}
	}
}
