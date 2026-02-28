// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package telemetry

import (
	"os/exec"
	"syscall"

	"golang.org/x/sys/windows"
)

func init() {
	daemonize = daemonizeWindows
}

func daemonizeWindows(cmd *exec.Cmd) {
	// Set DETACHED_PROCESS creation flag so that closing
	// the console window the parent process was run in
	// does not kill the child.
	// See documentation of creation flags in the Microsoft documentation:
	// https://learn.microsoft.com/en-us/windows/win32/procthread/process-creation-flags
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: windows.DETACHED_PROCESS,
	}
}
