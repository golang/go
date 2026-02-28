// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package configstore

import (
	"os/exec"
	"syscall"

	"golang.org/x/sys/windows"
)

func init() {
	needNoConsole = needNoConsoleWindows
}

func needNoConsoleWindows(cmd *exec.Cmd) {
	// The uploader main process is likely a daemonized process with no console.
	// (see x/telemetry/start_windows.go) The console creation behavior when
	// a parent is a console process without console is not clearly documented
	// but empirically we observed the new console is created and attached to the
	// subprocess in the default setup.
	//
	// Ensure no new console is attached to the subprocess by setting CREATE_NO_WINDOW.
	//   https://learn.microsoft.com/en-us/windows/console/creation-of-a-console
	//   https://learn.microsoft.com/en-us/windows/win32/procthread/process-creation-flags
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: windows.CREATE_NO_WINDOW,
	}
}
