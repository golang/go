// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || netbsd || openbsd

package osinfo

import (
	"strings"
	"syscall"
)

// Version returns the OS version name/number.
func Version() (string, error) {
	sysname, err := syscall.Sysctl("kern.ostype")
	if err != nil {
		return "", err
	}
	release, err := syscall.Sysctl("kern.osrelease")
	if err != nil {
		return "", err
	}
	version, err := syscall.Sysctl("kern.version")
	if err != nil {
		return "", err
	}

	// The version might have newlines or tabs; convert to spaces.
	version = strings.ReplaceAll(version, "\n", " ")
	version = strings.ReplaceAll(version, "\t", " ")
	version = strings.TrimSpace(version)

	machine, err := syscall.Sysctl("hw.machine")
	if err != nil {
		return "", err
	}

	ret := sysname + " " + release + " " + version + " " + machine
	return ret, nil
}
