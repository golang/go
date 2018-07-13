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

func MustHaveDebugOptionsEnabled(t *testing.T) {
	if !DebugOptions {
		t.Skipf("skipping test: cpu feature options not enabled")
	}
}

func runDebugOptionsTest(t *testing.T, test string, options string) {
	MustHaveDebugOptionsEnabled(t)

	testenv.MustHaveExec(t)

	env := "GODEBUGCPU=" + options

	cmd := exec.Command(os.Args[0], "-test.run="+test)
	cmd.Env = append(cmd.Env, env)

	output, err := cmd.CombinedOutput()
	got := strings.TrimSpace(string(output))
	want := "PASS"
	if err != nil || got != want {
		t.Fatalf("%s with %s: want %s, got %v", test, env, want, got)
	}
}

func TestDisableAllCapabilities(t *testing.T) {
	runDebugOptionsTest(t, "TestAllCapabilitiesDisabled", "all=0")
}

func TestAllCapabilitiesDisabled(t *testing.T) {
	MustHaveDebugOptionsEnabled(t)

	if os.Getenv("GODEBUGCPU") != "all=0" {
		t.Skipf("skipping test: GODEBUGCPU=all=0 not set")
	}

	for _, o := range Options {
		if got := *o.Feature; got != false {
			t.Errorf("%v: expected false, got %v", o.Name, got)
		}
	}
}
