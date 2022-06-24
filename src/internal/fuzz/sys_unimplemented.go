// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// If you update this constraint, also update cmd/internal/sys.FuzzSupported.
//
//go:build !darwin && !freebsd && !linux && !windows

package fuzz

import (
	"os"
	"os/exec"
)

type sharedMemSys struct{}

func sharedMemMapFile(f *os.File, size int, removeOnClose bool) (*sharedMem, error) {
	panic("not implemented")
}

func (m *sharedMem) Close() error {
	panic("not implemented")
}

func setWorkerComm(cmd *exec.Cmd, comm workerComm) {
	panic("not implemented")
}

func getWorkerComm() (comm workerComm, err error) {
	panic("not implemented")
}

func isInterruptError(err error) bool {
	panic("not implemented")
}

func terminationSignal(err error) (os.Signal, bool) {
	panic("not implemented")
}

func isCrashSignal(signal os.Signal) bool {
	panic("not implemented")
}
