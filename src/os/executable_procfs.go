// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux netbsd dragonfly nacl js,wasm

package os

import (
	"errors"
	"runtime"
)

// We query the executable path at init time to avoid the problem of
// readlink returns a path appended with " (deleted)" when the original
// binary gets deleted.
var executablePath, executablePathErr = func() (string, error) {
	var procfn string
	switch runtime.GOOS {
	default:
		return "", errors.New("Executable not implemented for " + runtime.GOOS)
	case "linux", "android":
		procfn = "/proc/self/exe"
	case "netbsd":
		procfn = "/proc/curproc/exe"
	case "dragonfly":
		procfn = "/proc/curproc/file"
	}
	return Readlink(procfn)
}()

func executable() (string, error) {
	return executablePath, executablePathErr
}
