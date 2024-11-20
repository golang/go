// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9

package os

import "syscall"

type syscallErrorType = syscall.Errno

const (
	errENOSYS = syscall.ENOSYS
	errERANGE = syscall.ERANGE
	errENOMEM = syscall.ENOMEM
)
