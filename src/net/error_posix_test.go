// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9
// +build !plan9

package net

import (
	"os"
	"syscall"
	"testing"
)

func TestSpuriousENOTAVAIL(t *testing.T) {
	for _, tt := range []struct {
		error
		ok bool
	}{
		{syscall.EADDRNOTAVAIL, true},
		{&os.SyscallError{Syscall: "syscall", Err: syscall.EADDRNOTAVAIL}, true},
		{&OpError{Op: "op", Err: syscall.EADDRNOTAVAIL}, true},
		{&OpError{Op: "op", Err: &os.SyscallError{Syscall: "syscall", Err: syscall.EADDRNOTAVAIL}}, true},

		{syscall.EINVAL, false},
		{&os.SyscallError{Syscall: "syscall", Err: syscall.EINVAL}, false},
		{&OpError{Op: "op", Err: syscall.EINVAL}, false},
		{&OpError{Op: "op", Err: &os.SyscallError{Syscall: "syscall", Err: syscall.EINVAL}}, false},
	} {
		if ok := spuriousENOTAVAIL(tt.error); ok != tt.ok {
			t.Errorf("spuriousENOTAVAIL(%v) = %v; want %v", tt.error, ok, tt.ok)
		}
	}
}
