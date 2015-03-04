// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux darwin dragonfly freebsd openbsd netbsd solaris

package tar

import (
	"os"
	"syscall"
)

func init() {
	sysStat = statUnix
}

func statUnix(fi os.FileInfo, h *Header) error {
	switch sys := fi.Sys().(type) {
	case *syscall.Stat_t:
		h.Uid = int(sys.Uid)
		h.Gid = int(sys.Gid)
		// TODO(bradfitz): populate username & group.  os/user
		// doesn't cache LookupId lookups, and lacks group
		// lookup functions.
		h.AccessTime = statAtime(sys)
		h.ChangeTime = statCtime(sys)
		// TODO(bradfitz): major/minor device numbers?
		if fi.Mode().IsRegular() && sys.Nlink > 1 {
			h.Typeflag = TypeLink
			h.Size = 0
			// TODO(vbatts): Linkname?
		}
	case *Header:
		// for the roundtrip logic
		h.Uid = sys.Uid
		h.Gid = sys.Gid
		h.Uname = sys.Uname
		h.Gname = sys.Gname
		h.AccessTime = sys.AccessTime
		h.ChangeTime = sys.ChangeTime
		if sys.Xattrs != nil {
			h.Xattrs = make(map[string]string)
			for k, v := range sys.Xattrs {
				h.Xattrs[k] = v
			}
		}
		if sys.Typeflag == TypeLink {
			// hard link
			h.Typeflag = TypeLink
			h.Size = 0
			h.Linkname = sys.Linkname
		}
	}
	return nil
}
