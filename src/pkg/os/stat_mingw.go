// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "syscall"

func isSymlink(stat *syscall.Stat_t) bool {
	panic("windows isSymlink not implemented")
}

func dirFromStat(name string, dir *Dir, lstat, stat *syscall.Stat_t) *Dir {
	panic("windows dirFromStat not implemented")
}
