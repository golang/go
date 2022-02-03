// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package testenv

import "syscall"

// Sigquit is the signal to send to kill a hanging subprocess.
// Send SIGQUIT to get a stack trace.
var Sigquit = syscall.SIGQUIT
