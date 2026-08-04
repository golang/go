// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	_ "unsafe"
)

//go:linkname runtime_ignoreHangup internal/poll.runtime_ignoreHangup
func runtime_ignoreHangup() {
	getg().m.ignoreHangup = true
}

// A note is delivered at the next kernel exit, so sleep once for each
// one Cancel posted, or it arrives when nothing ignores it any more.
//
//go:linkname runtime_unignoreHangup internal/poll.runtime_unignoreHangup
func runtime_unignoreHangup(notes int) {
	for ; notes > 0; notes-- {
		sleep(0)
	}
	getg().m.ignoreHangup = false
}

func ignoredNote(note *byte) bool {
	if note == nil {
		return false
	}
	if gostringnocopy(note) != "hangup" {
		return false
	}
	return getg().m.ignoreHangup
}
