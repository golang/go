// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package osinfo

import (
	"bytes"

	"golang.org/x/sys/unix"
)

func utsString(b []byte) string {
	i := bytes.IndexByte(b, 0)
	if i == -1 {
		return string(b)
	}
	return string(b[:i])
}

// Version returns the OS version name/number.
func Version() (string, error) {
	var uts unix.Utsname
	if err := unix.Uname(&uts); err != nil {
		return "", err
	}

	sysname := utsString(uts.Sysname[:])
	release := utsString(uts.Release[:])
	version := utsString(uts.Version[:])
	machine := utsString(uts.Machine[:])

	return sysname + " " + release + " " + version + " " + machine, nil
}
