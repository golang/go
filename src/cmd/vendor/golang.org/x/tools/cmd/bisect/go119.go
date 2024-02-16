// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.20

package main

import "os/exec"

func cmdInterrupt(cmd *exec.Cmd) {
	// cmd.Cancel and cmd.WaitDelay not available before Go 1.20.
}
