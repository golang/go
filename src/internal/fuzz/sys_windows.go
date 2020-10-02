// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package fuzz

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
)

// setWorkerComm configures communciation channels on the cmd that will
// run a worker process.
func setWorkerComm(cmd *exec.Cmd, fuzzIn, fuzzOut *os.File) {
	syscall.SetHandleInformation(syscall.Handle(fuzzIn.Fd()), syscall.HANDLE_FLAG_INHERIT, 1)
	syscall.SetHandleInformation(syscall.Handle(fuzzOut.Fd()), syscall.HANDLE_FLAG_INHERIT, 1)
	cmd.Env = append(cmd.Env, fmt.Sprintf("GO_TEST_FUZZ_WORKER_HANDLES=%x,%x", fuzzIn.Fd(), fuzzOut.Fd()))
}

// getWorkerComm returns communication channels in the worker process.
func getWorkerComm() (fuzzIn *os.File, fuzzOut *os.File, err error) {
	v := os.Getenv("GO_TEST_FUZZ_WORKER_HANDLES")
	if v == "" {
		return nil, nil, fmt.Errorf("GO_TEST_FUZZ_WORKER_HANDLES not set")
	}
	parts := strings.Split(v, ",")
	if len(parts) != 2 {
		return nil, nil, fmt.Errorf("GO_TEST_FUZZ_WORKER_HANDLES has invalid value")
	}
	base := 16
	bitSize := 64
	in, err := strconv.ParseInt(parts[0], base, bitSize)
	if err != nil {
		return nil, nil, fmt.Errorf("GO_TEST_FUZZ_WORKER_HANDLES has invalid value: %v", err)
	}
	out, err := strconv.ParseInt(parts[1], base, bitSize)
	if err != nil {
		return nil, nil, fmt.Errorf("GO_TEST_FUZZ_WORKER_HANDLES has invalid value: %v", err)
	}
	fuzzIn = os.NewFile(uintptr(in), "fuzz_in")
	fuzzOut = os.NewFile(uintptr(out), "fuzz_out")
	return fuzzIn, fuzzOut, nil
}
