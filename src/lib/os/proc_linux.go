// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"os";
	"syscall";
)

var Args []string;	// provided by runtime
var Envs []string;	// provided by runtime

// Exit causes the current program to exit with the given status code.
// Conventionally, code zero indicates success, non-zero an error.
func Exit(code int) {
	syscall.Syscall(syscall.SYS_EXIT_GROUP, int64(code), 0, 0)
}

