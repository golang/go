// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"os"
	"syscall"
)

func gettid() int {
	return syscall.Gettid()
}

func tidExists(tid int) (exists, supported bool) {
	stat, err := os.ReadFile(fmt.Sprintf("/proc/self/task/%d/stat", tid))
	if os.IsNotExist(err) {
		return false, true
	}
	// Check if it's a zombie thread.
	state := bytes.Fields(stat)[2]
	return !(len(state) == 1 && state[0] == 'Z'), true
}

func getcwd() (string, error) {
	if !syscall.ImplementsGetwd {
		return "", nil
	}
	// Use the syscall to get the current working directory.
	// This is imperative for checking for OS thread state
	// after an unshare since os.Getwd might just check the
	// environment, or use some other mechanism.
	var buf [4096]byte
	n, err := syscall.Getcwd(buf[:])
	if err != nil {
		return "", err
	}
	// Subtract one for null terminator.
	return string(buf[:n-1]), nil
}

func unshareFs() error {
	err := syscall.Unshare(syscall.CLONE_FS)
	if testenv.SyscallIsNotSupported(err) {
		return errNotPermitted
	}
	return err
}

func chdir(path string) error {
	return syscall.Chdir(path)
}
