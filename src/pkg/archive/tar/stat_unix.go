// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux darwin freebsd openbsd netbsd

package tar

import (
	"os"
	"syscall"
)

func init() {
	sysStat = statUnix
}

func statUnix(fi os.FileInfo, h *Header) error {
	sys, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return nil
	}
	h.Uid = int(sys.Uid)
	h.Gid = int(sys.Gid)
	// TODO(bradfitz): populate username & group.  os/user
	// doesn't cache LookupId lookups, and lacks group
	// lookup functions.
	h.AccessTime = statAtime(sys)
	h.ChangeTime = statCtime(sys)
	// TODO(bradfitz): major/minor device numbers?
	return nil
}
