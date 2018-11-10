// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build plan9

package runtime

var netpollWaiters uint32

// Polls for ready network connections.
// Returns list of goroutines that become runnable.
func netpoll(block bool) (gp *g) {
	// Implementation for platforms that do not support
	// integrated network poller.
	return
}

func netpollinited() bool {
	return false
}
