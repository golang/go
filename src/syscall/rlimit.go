// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package syscall

import (
	"sync/atomic"
	_ "unsafe"
)

// origRlimitNofile, if non-nil, is the original soft RLIMIT_NOFILE.
//
// origRlimitNofile should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/opencontainers/runc
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname origRlimitNofile
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
	if err := Getrlimit(RLIMIT_NOFILE, &lim); err == nil && lim.Cur != lim.Max {
		origRlimitNofile.Store(&lim)
		nlim := lim
		nlim.Cur = nlim.Max
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
