// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package os

import "syscall"

// Some systems set an artificially low soft limit on open file count, for compatibility
// with code that uses select and its hard-coded maximum file descriptor
// (limited by the size of fd_set).
//
// Go does not use select, so it should not be subject to these limits.
// On some systems the limit is 256, which is very easy to run into,
// even in simple programs like gofmt when they parallelize walking
// a file tree.
//
// After a long discussion on go.dev/issue/46279, we decided the
// best approach was for Go to raise the limit unconditionally for itself,
// and then leave old software to set the limit back as needed.
// Code that really wants Go to leave the limit alone can set the hard limit,
// which Go of course has no choice but to respect.
func init() {
	var lim syscall.Rlimit
	if err := syscall.Getrlimit(syscall.RLIMIT_NOFILE, &lim); err == nil && lim.Cur != lim.Max {
		lim.Cur = lim.Max
		adjustFileLimit(&lim)
		syscall.Setrlimit(syscall.RLIMIT_NOFILE, &lim)
	}
}
