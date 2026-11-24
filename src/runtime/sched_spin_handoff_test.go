// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"testing"
)

func TestSchedulerSpinExitDiscoversEnqueuedWork(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skip("scheduler rendezvous test is validated on linux/amd64")
	}
	if runtime.NumCPU() < 2 {
		t.Skip("requires two CPUs")
	}
	exe, err := buildTestProg(t, "testprog", "-tags=runtime_test_sched")
	if err != nil {
		t.Fatal(err)
	}
	out, err := runBuiltTestProgErr(t, exe, "SchedulerSpinExitHandoff", "GOMAXPROCS=2", "GOGC=off", "GOMEMLIMIT=off")
	if err != nil || out != "OK\n" {
		t.Fatalf("spin-exit handoff failed (%v)\n%s", err, out)
	}
}
