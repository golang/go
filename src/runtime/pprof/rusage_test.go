// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package pprof

import (
	"syscall"
	"time"
)

func init() {
	diffCPUTimeImpl = diffCPUTimeRUsage
}

func diffCPUTimeRUsage(f func()) (user, system time.Duration) {
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
		return 0, 0
	}

	user = time.Duration(after.Utime.Nano() - before.Utime.Nano())
	system = time.Duration(after.Stime.Nano() - before.Stime.Nano())
	return user, system
}
