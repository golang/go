// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "syscall"

func hostname() (name string, err error) {
	s, e := syscall.ComputerName()
	if e != nil {
		return "", NewSyscallError("ComputerName", e)
	}
	return s, nil
}
