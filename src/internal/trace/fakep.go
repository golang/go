// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

const (
	// Special P identifiers:
	FakeP    = 1000000 + iota
	TimerP   // depicts timer unblocks
	NetpollP // depicts network unblocks
	SyscallP // depicts returns from syscalls
	GCP      // depicts GC state
	ProfileP // depicts recording of CPU profile samples
)
