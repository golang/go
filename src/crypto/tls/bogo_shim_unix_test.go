// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix && !wasm

package tls

import (
	"os"
	"syscall"
)

func pauseProcess() {
	pid := os.Getpid()
	process, _ := os.FindProcess(pid)
	process.Signal(syscall.SIGSTOP)
}
