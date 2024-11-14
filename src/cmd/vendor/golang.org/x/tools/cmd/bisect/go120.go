// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.20

package main

import (
	"os"
	"os/exec"
	"time"
)

func cmdInterrupt(cmd *exec.Cmd) {
	cmd.Cancel = func {
		// On timeout, send interrupt,
		// in hopes of shutting down process tree.
		// Ignore errors sending signal; it's all best effort
		// and not even implemented on Windows.
		// TODO(rsc): Maybe use a new process group and kill the whole group?
		cmd.Process.Signal(os.Interrupt)
		return nil
	}
	cmd.WaitDelay = 2 * time.Second
}
