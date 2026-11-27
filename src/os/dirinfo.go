// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9

package os

// A dirInfo is released when its last reference is dropped. The File
// holds a reference while the dirInfo is attached to it, and each readdir
// call holds one while it uses the dirInfo, so Close or Seek can detach
// the dirInfo while a concurrent readdir is still reading from it.

// acquire adds a reference to d. It reports false if d has already been
// released.
func (d *dirInfo) acquire() bool {
	for {
		n := d.refs.Load()
		if n == 0 {
			return false
		}
		if d.refs.CompareAndSwap(n, n+1) {
			return true
		}
	}
}

// release drops a reference to d, and releases d if it was the last one.
func (d *dirInfo) release() {
	if d.refs.Add(-1) == 0 {
		d.close()
	}
}
