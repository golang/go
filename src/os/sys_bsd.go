// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || (js && wasm) || netbsd || openbsd
// +build darwin dragonfly freebsd js,wasm netbsd openbsd

package os

import "syscall"

func hostname() (name string, err error) {
	name, err = syscall.Sysctl("kern.hostname")
	if err != nil {
		return "", NewSyscallError("sysctl kern.hostname", err)
	}
	return name, nil
}
