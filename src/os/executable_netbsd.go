// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

// From NetBSD's <sys/sysctl.h>
const (
	_CTL_KERN           = 1
	_KERN_PROC_ARGS     = 48
	_KERN_PROC_PATHNAME = 5
)

var executableMIB = [4]int32{_CTL_KERN, _KERN_PROC_ARGS, -1, _KERN_PROC_PATHNAME}
