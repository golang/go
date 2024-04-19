// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"io"
	"os"
	"syscall"
)

func gettid() int {
	return syscall.Gettid()
}

func tidExists(tid int) (exists, supported bool, err error) {
	// Open the magic proc status file for reading with the syscall package.
	// We want to identify certain valid errors very precisely.
	statusFile := fmt.Sprintf("/proc/self/task/%d/status", tid)
	fd, err := syscall.Open(statusFile, syscall.O_RDONLY, 0)
	if errno, ok := err.(syscall.Errno); ok {
		if errno == syscall.ENOENT || errno == syscall.ESRCH {
			return false, true, nil
		}
	}
	if err != nil {
		return false, false, err
	}
	f := os.NewFile(uintptr(fd), statusFile)
	defer f.Close()
	status, err := io.ReadAll(f)
	if err != nil {
		return false, false, err
	}
	lines := bytes.Split(status, []byte{'\n'})
	// Find the State line.
	stateLineIdx := -1
	for i, line := range lines {
		if bytes.HasPrefix(line, []byte("State:")) {
			stateLineIdx = i
			break
		}
	}
	if stateLineIdx < 0 {
		// Malformed status file?
		return false, false, fmt.Errorf("unexpected status file format: %s:\n%s", statusFile, status)
	}
	stateLine := bytes.SplitN(lines[stateLineIdx], []byte{':'}, 2)
	if len(stateLine) != 2 {
		// Malformed status file?
		return false, false, fmt.Errorf("unexpected status file format: %s:\n%s", statusFile, status)
	}
	// Check if it's a zombie thread.
	return !bytes.Contains(stateLine[1], []byte{'Z'}), true, nil
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
