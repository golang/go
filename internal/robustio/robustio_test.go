// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package robustio_test

import (
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/tools/internal/robustio"
)

func TestFileID(t *testing.T) {
	// A nonexistent file has no ID.
	nonexistent := filepath.Join(t.TempDir(), "nonexistent")
	if _, err := robustio.GetFileID(nonexistent); err == nil {
		t.Fatalf("GetFileID(nonexistent) succeeded unexpectedly")
	}

	// A regular file has an ID.
	real := filepath.Join(t.TempDir(), "real")
	if err := os.WriteFile(real, nil, 0644); err != nil {
		t.Fatalf("can't create regular file: %v", err)
	}
	realID, err := robustio.GetFileID(real)
	if err != nil {
		t.Fatalf("can't get ID of regular file: %v", err)
	}

	// A second regular file has a different ID.
	real2 := filepath.Join(t.TempDir(), "real2")
	if err := os.WriteFile(real2, nil, 0644); err != nil {
		t.Fatalf("can't create second regular file: %v", err)
	}
	real2ID, err := robustio.GetFileID(real2)
	if err != nil {
		t.Fatalf("can't get ID of second regular file: %v", err)
	}
	if realID == real2ID {
		t.Errorf("realID %+v != real2ID %+v", realID, real2ID)
	}

	// A symbolic link has the same ID as its target.
	symlink := filepath.Join(t.TempDir(), "symlink")
	if err := os.Symlink(real, symlink); err != nil {
		t.Fatalf("can't create symbolic link: %v", err)
	}
	symlinkID, err := robustio.GetFileID(symlink)
	if err != nil {
		t.Fatalf("can't get ID of symbolic link: %v", err)
	}
	if realID != symlinkID {
		t.Errorf("realID %+v != symlinkID %+v", realID, symlinkID)
	}

	// Two hard-linked files have the same ID.
	hardlink := filepath.Join(t.TempDir(), "hardlink")
	if err := os.Link(real, hardlink); err != nil {
		t.Fatal(err)
	}
	hardlinkID, err := robustio.GetFileID(hardlink)
	if err != nil {
		t.Fatalf("can't get ID of hard link: %v", err)
	}
	if hardlinkID != realID {
		t.Errorf("realID %+v != hardlinkID %+v", realID, hardlinkID)
	}
}
