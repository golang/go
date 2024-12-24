// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package syscall

import (
	"sync/atomic"
)

// origRlimitNofile, if non-nil, is the original soft RLIMIT_NOFILE.
var origRlimitNofile atomic.Pointer[Rlimit]

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
	var lim Rlimit
	if err := Getrlimit(RLIMIT_NOFILE, &lim); err == nil && lim.Max > 0 && lim.Cur < lim.Max-1 {
		origRlimitNofile.Store(&lim)
		nlim := lim

		// We set Cur to Max - 1 so that we are more likely to
		// detect cases where another process uses prlimit
		// to change our resource limits. The theory is that
		// using prlimit to change to Cur == Max is more likely
		// than using prlimit to change to Cur == Max - 1.
		// The place we check for this is in exec_linux.go.
		nlim.Cur = nlim.Max - 1

		adjustFileLimit(&nlim)
		setrlimit(RLIMIT_NOFILE, &nlim)
	}
}

func Setrlimit(resource int, rlim *Rlimit) error {
	if resource == RLIMIT_NOFILE {
		// Store nil in origRlimitNofile to tell StartProcess
		// to not adjust the rlimit in the child process.
		origRlimitNofile.Store(nil)
	}
	return setrlimit(resource, rlim)
}
