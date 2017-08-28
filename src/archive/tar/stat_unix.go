// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux darwin dragonfly freebsd openbsd netbsd solaris

package tar

import (
	"os"
	"os/user"
	"runtime"
	"strconv"
	"sync"
	"syscall"
)

func init() {
	sysStat = statUnix
}

// userMap and groupMap caches UID and GID lookups for performance reasons.
// The downside is that renaming uname or gname by the OS never takes effect.
var userMap, groupMap sync.Map // map[int]string

func statUnix(fi os.FileInfo, h *Header) error {
	sys, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return nil
	}
	h.Uid = int(sys.Uid)
	h.Gid = int(sys.Gid)

	// Best effort at populating Uname and Gname.
	// The os/user functions may fail for any number of reasons
	// (not implemented on that platform, cgo not enabled, etc).
	if u, ok := userMap.Load(h.Uid); ok {
		h.Uname = u.(string)
	} else if u, err := user.LookupId(strconv.Itoa(h.Uid)); err == nil {
		h.Uname = u.Username
		userMap.Store(h.Uid, h.Uname)
	}
	if g, ok := groupMap.Load(h.Gid); ok {
		h.Gname = g.(string)
	} else if g, err := user.LookupGroupId(strconv.Itoa(h.Gid)); err == nil {
		h.Gname = g.Name
		groupMap.Store(h.Gid, h.Gname)
	}

	h.AccessTime = statAtime(sys)
	h.ChangeTime = statCtime(sys)

	// Best effort at populating Devmajor and Devminor.
	if h.Typeflag == TypeChar || h.Typeflag == TypeBlock {
		dev := uint64(sys.Rdev) // May be int32 or uint32
		switch runtime.GOOS {
		case "linux":
			// Copied from golang.org/x/sys/unix/dev_linux.go.
			major := uint32((dev & 0x00000000000fff00) >> 8)
			major |= uint32((dev & 0xfffff00000000000) >> 32)
			minor := uint32((dev & 0x00000000000000ff) >> 0)
			minor |= uint32((dev & 0x00000ffffff00000) >> 12)
			h.Devmajor, h.Devminor = int64(major), int64(minor)
		default:
			// TODO: Implement others (see https://golang.org/issue/8106)
		}
	}
	return nil
}
