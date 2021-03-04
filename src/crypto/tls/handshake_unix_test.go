// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris
// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package tls

import (
	"errors"
	"syscall"
)

func init() {
	isConnRefused = func(err error) bool {
		return errors.Is(err, syscall.ECONNREFUSED)
	}
}
