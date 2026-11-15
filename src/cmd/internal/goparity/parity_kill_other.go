// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !unix

package goparity

import (
	"os/exec"
	"runtime"
	"testing"
	"time"
)

func processGroupKillSupported() bool { return false }

func skipUnsupportedProcessGroupKill(t *testing.T) {
	t.Skipf("process-group kill is unsupported on %s; a timed-out parity run could leave grandchildren alive", runtime.GOOS)
}

func setNewProcessGroup(cmd *exec.Cmd) {}

func killProcessGroup(cmd *exec.Cmd) {
	if cmd.Process == nil {
		return
	}
	_ = cmd.Process.Kill()
}

func procGroupGone(pgid int, wait time.Duration) bool {
	return true
}
