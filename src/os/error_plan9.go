// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "syscall"

type syscallErrorType = syscall.ErrorString

var errENOSYS = syscall.NewError("function not implemented")
var errERANGE = syscall.NewError("out of range")
