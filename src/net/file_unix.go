// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package net

import (
	"internal/poll"
	"os"
	"syscall"
)

const _SO_TYPE = syscall.SO_TYPE

func dupFileSocket(f *os.File) (int, error) {
	s, call, err := poll.DupCloseOnExec(int(f.Fd()))
	if err != nil {
		if call != "" {
			err = os.NewSyscallError(call, err)
		}
		return -1, err
	}
	if err := syscall.SetNonblock(s, true); err != nil {
		poll.CloseFunc(s)
		return -1, os.NewSyscallError("setnonblock", err)
	}
	return s, nil
}
