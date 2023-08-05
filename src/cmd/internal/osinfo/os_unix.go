// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package osinfo

import "golang.org/x/sys/unix"

// Version returns the OS version name/number.
func Version() (string, error) {
	var uts unix.Utsname
	if err := unix.Uname(&uts); err != nil {
		return "", err
	}

	sysname := unix.ByteSliceToString(uts.Sysname[:])
	release := unix.ByteSliceToString(uts.Release[:])
	version := unix.ByteSliceToString(uts.Version[:])
	machine := unix.ByteSliceToString(uts.Machine[:])

	return sysname + " " + release + " " + version + " " + machine, nil
}
