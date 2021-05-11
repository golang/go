// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd
// +build darwin dragonfly freebsd linux netbsd openbsd

package syscall_test

import (
	"internal/testenv"
	"os"
	"os/exec"
	"syscall"
	"testing"
)

func TestExecPtrace(t *testing.T) {
	testenv.MustHaveExec(t)

	bin, err := exec.LookPath("sh")
	if err != nil {
		t.Skipf("skipped because sh is not available")
	}

	attr := &os.ProcAttr{
		Sys: &syscall.SysProcAttr{
			Ptrace: true,
		},
	}
	proc, err := os.StartProcess(bin, []string{bin}, attr)
	if err == nil {
		proc.Kill()
	}
	if err != nil && !os.IsPermission(err) {
		t.Fatalf("StartProcess with ptrace enabled failed: %v", err)
	}
}
