// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && arm64

package runtime_test

import (
	"internal/testenv"
	"testing"
	"time"
)

// TestR28ClobberSIGURG verifies that sigtramp recovers g from TLS when
// R28 is clobbered on the main thread. The child clobbers R28 and calls
// tgkill(SIGURG) in the same instruction sequence, guaranteeing the signal
// arrives while R28 holds a garbage value.
func TestR28ClobberSIGURG(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}

	cmd := testenv.CleanCmdEnv(testenv.Command(t, exe, "R28ClobberSIGURG"))
	cmd.WaitDelay = 5 * time.Second
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("crashed: sigtramp likely used clobbered R28:\n%s\n%v", out, err)
	}
}

// TestR28ClobberSIGURGClone is the same test on a newly cloned thread,
// verifying that clone sets up TLS for g recovery.
func TestR28ClobberSIGURGClone(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}

	cmd := testenv.CleanCmdEnv(testenv.Command(t, exe, "R28ClobberSIGURGClone"))
	cmd.WaitDelay = 5 * time.Second
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("crashed: TLS not set up on cloned thread:\n%s\n%v", out, err)
	}
}
