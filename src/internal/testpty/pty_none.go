// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !(cgo && (aix || dragonfly || freebsd || (linux && !android) || netbsd || openbsd)) && !darwin

package testpty

import "os"

func open() (pty *os.File, processTTY string, err error) {
	return nil, "", ErrNotSupported
}
