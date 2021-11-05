// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || freebsd || linux || netbsd || openbsd

package pprof

import (
	"syscall"
	"time"
)

func init() {
	diffCPUTimeImpl = diffCPUTimeRUsage
}

func diffCPUTimeRUsage(f func()) time.Duration {
	ok := true
	var before, after syscall.Rusage

	err := syscall.Getrusage(syscall.RUSAGE_SELF, &before)
	if err != nil {
		ok = false
	}

	f()

	err = syscall.Getrusage(syscall.RUSAGE_SELF, &after)
	if err != nil {
		ok = false
	}

	if !ok {
		return 0
	}

	return time.Duration((after.Utime.Nano() + after.Stime.Nano()) - (before.Utime.Nano() + before.Stime.Nano()))
}
