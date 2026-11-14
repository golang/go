// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

func logosClearDeferTryPoolLocked() {
	for t := sched.deferTryPool; t != nil; {
		next := t.ext
		t.ext = nil
		t = next
	}
	sched.deferTryPool = nil
}
