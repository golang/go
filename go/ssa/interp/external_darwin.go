// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin

package interp

import "syscall"

func init() {
	externals["syscall.Sysctl"] = ext۰syscall۰Sysctl

	fillStat = func(st *syscall.Stat_t, stat structure) {
		stat[0] = st.Dev
		stat[1] = st.Mode
		stat[2] = st.Nlink
		stat[3] = st.Ino
		stat[4] = st.Uid
		stat[5] = st.Gid
		stat[6] = st.Rdev
		// TODO(adonovan): fix: copy Timespecs.
		// stat[8] = st.Atim
		// stat[9] = st.Mtim
		// stat[10] = st.Ctim
		stat[12] = st.Size
		stat[13] = st.Blocks
		stat[14] = st.Blksize
	}
}

func ext۰syscall۰Sysctl(fr *frame, args []value) value {
	r, err := syscall.Sysctl(args[0].(string))
	return tuple{r, wrapError(err)}
}
