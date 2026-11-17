// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package goparity

import (
	"os/exec"
	"runtime"
	"syscall"
	"testing"
	"time"
)

func processGroupKillSupported() bool { return true }

func skipUnsupportedProcessGroupKill(t *testing.T) {
	t.Skipf("process-group kill is unsupported on %s; a timed-out parity run could leave grandchildren alive", runtime.GOOS)
}

func setNewProcessGroup(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
}

func killProcessGroup(cmd *exec.Cmd) {
	if cmd.Process == nil {
		return
	}
	_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
}

func procGroupGone(pgid int, wait time.Duration) bool {
	deadline := time.Now().Add(wait)
	for {
		if err := syscall.Kill(-pgid, syscall.Signal(0)); err == syscall.ESRCH {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(50 * time.Millisecond)
	}
}
