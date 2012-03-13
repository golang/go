// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin

package main

import (
	"archive/tar"
	"os"
	"syscall"
	"time"
)

func init() {
	sysStat = func(fi os.FileInfo, h *tar.Header) error {
		sys, ok := fi.Sys().(*syscall.Stat_t)
		if !ok {
			return nil
		}
		h.Uid = int(sys.Uid)
		h.Gid = int(sys.Gid)
		// TODO(bradfitz): populate username & group.  os/user
		// doesn't cache LookupId lookups, and lacks group
		// lookup functions.
		h.AccessTime = time.Unix(sys.Atimespec.Unix())
		h.ChangeTime = time.Unix(sys.Ctimespec.Unix())
		// TODO(bradfitz): major/minor device numbers?
		return nil
	}
}
