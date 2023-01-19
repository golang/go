// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package robustio_test

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"golang.org/x/tools/internal/robustio"
)

func TestFileInfo(t *testing.T) {
	// A nonexistent file has no ID.
	nonexistent := filepath.Join(t.TempDir(), "nonexistent")
	if _, _, err := robustio.GetFileID(nonexistent); err == nil {
		t.Fatalf("GetFileID(nonexistent) succeeded unexpectedly")
	}

	// A regular file has an ID.
	real := filepath.Join(t.TempDir(), "real")
	if err := os.WriteFile(real, nil, 0644); err != nil {
		t.Fatalf("can't create regular file: %v", err)
	}
	realID, realMtime, err := robustio.GetFileID(real)
	if err != nil {
		t.Fatalf("can't get ID of regular file: %v", err)
	}

	// Sleep so that we get a new mtime for subsequent writes.
	time.Sleep(2 * time.Second)

	// A second regular file has a different ID.
	real2 := filepath.Join(t.TempDir(), "real2")
	if err := os.WriteFile(real2, nil, 0644); err != nil {
		t.Fatalf("can't create second regular file: %v", err)
	}
	real2ID, real2Mtime, err := robustio.GetFileID(real2)
	if err != nil {
		t.Fatalf("can't get ID of second regular file: %v", err)
	}
	if realID == real2ID {
		t.Errorf("realID %+v == real2ID %+v", realID, real2ID)
	}
	if realMtime.Equal(real2Mtime) {
		t.Errorf("realMtime %v == real2Mtime %v", realMtime, real2Mtime)
	}

	// A symbolic link has the same ID as its target.
	if runtime.GOOS != "plan9" {
		symlink := filepath.Join(t.TempDir(), "symlink")
		if err := os.Symlink(real, symlink); err != nil {
			t.Fatalf("can't create symbolic link: %v", err)
		}
		symlinkID, symlinkMtime, err := robustio.GetFileID(symlink)
		if err != nil {
			t.Fatalf("can't get ID of symbolic link: %v", err)
		}
		if realID != symlinkID {
			t.Errorf("realID %+v != symlinkID %+v", realID, symlinkID)
		}
		if !realMtime.Equal(symlinkMtime) {
			t.Errorf("realMtime %v != symlinkMtime %v", realMtime, symlinkMtime)
		}
	}

	// Two hard-linked files have the same ID.
	if runtime.GOOS != "plan9" && runtime.GOOS != "android" {
		hardlink := filepath.Join(t.TempDir(), "hardlink")
		if err := os.Link(real, hardlink); err != nil {
			t.Fatal(err)
		}
		hardlinkID, hardlinkMtime, err := robustio.GetFileID(hardlink)
		if err != nil {
			t.Fatalf("can't get ID of hard link: %v", err)
		}
		if realID != hardlinkID {
			t.Errorf("realID %+v != hardlinkID %+v", realID, hardlinkID)
		}
		if !realMtime.Equal(hardlinkMtime) {
			t.Errorf("realMtime %v != hardlinkMtime %v", realMtime, hardlinkMtime)
		}
	}
}
