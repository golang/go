// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd

package interp

import "syscall"

func init() {
	externals["syscall.Sysctl"] = ext۰syscall۰Sysctl
	externals["syscall.SysctlUint32"] = ext۰syscall۰SysctlUint32
}

func ext۰syscall۰Sysctl(fr *frame, args []value) value {
	r, err := syscall.Sysctl(args[0].(string))
	return tuple{r, wrapError(err)}
}

func ext۰syscall۰SysctlUint32(fr *frame, args []value) value {
	r, err := syscall.SysctlUint32(args[0].(string))
	return tuple{r, wrapError(err)}
}
