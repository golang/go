// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo && (aix || dragonfly || freebsd || (linux && !android) || netbsd || openbsd)

package testpty

/*
#define _XOPEN_SOURCE 600
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
*/
import "C"

import "os"

func open() (pty *os.File, processTTY string, err error) {
	m, err := C.posix_openpt(C.O_RDWR)
	if m < 0 {
		return nil, "", ptyError("posix_openpt", err)
	}
	if res, err := C.grantpt(m); res < 0 {
		C.close(m)
		return nil, "", ptyError("grantpt", err)
	}
	if res, err := C.unlockpt(m); res < 0 {
		C.close(m)
		return nil, "", ptyError("unlockpt", err)
	}
	processTTY = C.GoString(C.ptsname(m))
	return os.NewFile(uintptr(m), "pty"), processTTY, nil
}
