// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || (js && wasm)

package os

import "syscall"

// Pipe returns a connected pair of Files; reads from r return bytes written to w.
// It returns the files and an error, if any.
func Pipe() (r *File, w *File, err error) {
	var p [2]int

	// See ../syscall/exec.go for description of lock.
	syscall.ForkLock.RLock()
	e := syscall.Pipe(p[0:])
	if e != nil {
		syscall.ForkLock.RUnlock()
		return nil, nil, NewSyscallError("pipe", e)
	}
	syscall.CloseOnExec(p[0])
	syscall.CloseOnExec(p[1])
	syscall.ForkLock.RUnlock()

	return newFile(uintptr(p[0]), "|0", kindPipe), newFile(uintptr(p[1]), "|1", kindPipe), nil
}
