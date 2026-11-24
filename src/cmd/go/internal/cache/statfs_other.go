// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !linux

package cache

// statfs returns zeros (unknown) outside Linux, disabling low disk trims.
func statfs(dir string) (avail, total, bsize uint64) {
	return 0, 0, 0
}
