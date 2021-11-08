// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || netbsd || openbsd

package os

import "syscall"

// Pipe returns a connected pair of Files; reads from r return bytes written to w.
// It returns the files and an error, if any.
func Pipe() (r *File, w *File, err error) {
	var p [2]int

	e := syscall.Pipe2(p[0:], syscall.O_CLOEXEC)
	if e != nil {
		return nil, nil, NewSyscallError("pipe", e)
	}

	return newFile(uintptr(p[0]), "|0", kindPipe), newFile(uintptr(p[1]), "|1", kindPipe), nil
}
