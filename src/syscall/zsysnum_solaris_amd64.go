// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64,solaris

package syscall

// TODO(aram): remove these before Go 1.3.
const (
	SYS_EXECVE = 59
	SYS_FCNTL  = 62
)
