// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"syscall"
	"time"
)

// A fileStat is the implementation of FileInfo returned by Stat and Lstat.
type fileStat struct {
	name    string
	size    int64
	mode    FileMode
	modTime time.Time
	sys     any
}

func (fs *fileStat) Size() int64        { return fs.size }
func (fs *fileStat) Mode() FileMode     { return fs.mode }
func (fs *fileStat) ModTime() time.Time { return fs.modTime }
func (fs *fileStat) Sys() any           { return fs.sys }

func sameFile(fs1, fs2 *fileStat) bool {
	a := fs1.sys.(*syscall.Dir)
	b := fs2.sys.(*syscall.Dir)
	return a.Qid.Path == b.Qid.Path && a.Type == b.Type && a.Dev == b.Dev
}

const badFd = -1
