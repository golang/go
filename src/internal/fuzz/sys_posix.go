// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows

package fuzz

import (
	"os"
	"os/exec"
)

// setWorkerComm configures communciation channels on the cmd that will
// run a worker process.
func setWorkerComm(cmd *exec.Cmd, fuzzIn, fuzzOut *os.File) {
	cmd.ExtraFiles = []*os.File{fuzzIn, fuzzOut}
}

// getWorkerComm returns communication channels in the worker process.
func getWorkerComm() (fuzzIn, fuzzOut *os.File, err error) {
	fuzzIn = os.NewFile(3, "fuzz_in")
	fuzzOut = os.NewFile(4, "fuzz_out")
	return fuzzIn, fuzzOut, nil
}
