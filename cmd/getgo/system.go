// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"os/exec"
	"runtime"
)

// arch contains either amd64 or 386.
var arch = func() string {
	cmd := exec.Command("uname", "-m") // "x86_64"
	if runtime.GOOS == "windows" {
		cmd = exec.Command("powershell", "-command", "(Get-WmiObject -Class Win32_ComputerSystem).SystemType") // "x64-based PC"
	}

	out, err := cmd.Output()
	if err != nil {
		// a sensible default?
		return "amd64"
	}
	if bytes.Contains(out, []byte("64")) {
		return "amd64"
	}
	return "386"
}()
