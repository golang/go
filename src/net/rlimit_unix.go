// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || wasip1

package net

import "syscall"

// concurrentThreadsLimit returns the number of threads we permit to
// run concurrently doing DNS lookups via cgo. A DNS lookup may use a
// file descriptor so we limit this to less than the number of
// permitted open files. On some systems, notably Darwin, if
// getaddrinfo is unable to open a file descriptor it simply returns
// EAI_NONAME rather than a useful error. Limiting the number of
// concurrent getaddrinfo calls to less than the permitted number of
// file descriptors makes that error less likely. We don't bother to
// apply the same limit to DNS lookups run directly from Go, because
// there we will return a meaningful "too many open files" error.
func concurrentThreadsLimit() int {
	var rlim syscall.Rlimit
	if err := syscall.Getrlimit(syscall.RLIMIT_NOFILE, &rlim); err != nil {
		return 500
	}
	r := rlim.Cur
	if r > 500 {
		r = 500
	} else if r > 30 {
		r -= 30
	}
	return int(r)
}
