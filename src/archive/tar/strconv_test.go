// Copyright 2024 The Go Authors. All rights reserved.
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

func TestStatUnixOverflow(t *testing.T) {
	// Only truly meaningful on 32-bit platforms, but the logic works everywhere.
	// We test if a large Uid/Gid is clamped correctly.
	largeUid := uint32(math.MaxInt32 + 1)

	sys := &syscall.Stat_t{
		Uid: largeUid,
		Gid: largeUid,
	}

	// This is a partial mock of fs.FileInfo just to provide Sys()
	fi := &statUnixTestFileInfo{sys: sys}
	h := &Header{}

	err := statUnix(fi, h, false)
	if err != nil {
		t.Fatalf("statUnix failed: %v", err)
	}

	if int64(h.Uid) < 0 {
		t.Errorf("expected non-negative Uid, got %d", h.Uid)
	}
	if int64(h.Gid) < 0 {
		t.Errorf("expected non-negative Gid, got %d", h.Gid)
	}
	
	expected := int(largeUid)
	if int64(largeUid) > math.MaxInt {
		expected = math.MaxInt
	}
	if h.Uid != expected {
		t.Errorf("expected Uid %d, got %d", expected, h.Uid)
	}
	if h.Gid != expected {
		t.Errorf("expected Gid %d, got %d", expected, h.Gid)
	}
}

// Full mock
type statUnixTestFileInfo struct {
	sys *syscall.Stat_t
}

func (s *statUnixTestFileInfo) Name() string       { return "test" }
func (s *statUnixTestFileInfo) Size() int64        { return 0 }
func (s *statUnixTestFileInfo) Mode() fs.FileMode  { return 0 }
func (s *statUnixTestFileInfo) ModTime() time.Time { return time.Time{} }
func (s *statUnixTestFileInfo) IsDir() bool        { return false }
func (s *statUnixTestFileInfo) Sys() any           { return s.sys }

// Note: ModTime and Mode technically return time.Time and fs.FileMode, 
// but since statUnix only calls Sys(), this loose mock is sufficient 
// if it implements an interface that matches in the minimal way needed (it just needs to implement fs.FileInfo). 
// Wait, we need it to implement fs.FileInfo exactly.