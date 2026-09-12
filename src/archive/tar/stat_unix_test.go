// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package tar

import (
	"io/fs"
	"math"
	"syscall"
	"testing"
	"time"
)

// statUnixFileInfo is a minimal fs.FileInfo whose Sys returns a
// *syscall.Stat_t, which is all statUnix inspects.
type statUnixFileInfo struct {
	sys *syscall.Stat_t
}

func (f statUnixFileInfo) Name() string       { return "" }
func (f statUnixFileInfo) Size() int64        { return 0 }
func (f statUnixFileInfo) Mode() fs.FileMode  { return 0 }
func (f statUnixFileInfo) ModTime() time.Time { return time.Time{} }
func (f statUnixFileInfo) IsDir() bool        { return false }
func (f statUnixFileInfo) Sys() any           { return f.sys }

func TestStatUnixUIDGIDOverflow(t *testing.T) {
	const id = uint32(math.MaxInt32) + 1
	fi := statUnixFileInfo{sys: &syscall.Stat_t{Uid: id, Gid: id}}

	var h Header
	err := statUnix(fi, &h, false)

	if int64(id) > math.MaxInt {
		// 32-bit int: the value cannot be represented without wrapping to
		// a negative number, so statUnix must report an error instead.
		if err == nil {
			t.Fatalf("statUnix(uid=%d) = nil error, want error", id)
		}
		return
	}

	// 64-bit int: the value fits, so it must round-trip unchanged and stay
	// non-negative.
	if err != nil {
		t.Fatalf("statUnix(uid=%d) = %v, want nil error", id, err)
	}
	if h.Uid < 0 || h.Gid < 0 {
		t.Errorf("statUnix gave negative Uid=%d Gid=%d, want non-negative", h.Uid, h.Gid)
	}
	if h.Uid != int(id) || h.Gid != int(id) {
		t.Errorf("statUnix gave Uid=%d Gid=%d, want %d", h.Uid, h.Gid, id)
	}
}
