// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestPermissions(t *testing.T) {
	if os.Getuid() == 0 {
		t.Skip("skipping permission test when running as root")
	}

	dir := t.TempDir()
	fn := filepath.Join(dir, "perm.go")

	// Create a file that needs formatting without write permission.
	if err := os.WriteFile(filepath.Join(fn), []byte("  package main"), 0o400); err != nil {
		t.Fatal(err)
	}

	// Set mtime of the file in the past.
	past := time.Now().Add(-time.Hour)
	if err := os.Chtimes(fn, past, past); err != nil {
		t.Fatal(err)
	}

	info, err := os.Stat(fn)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { *write = false }()
	*write = true

	initParserMode()
	initRewrite()

	const maxWeight = 2 << 20
	var buf, errBuf strings.Builder
	s := newSequencer(maxWeight, &buf, &errBuf)
	s.Add(fileWeight(fn, info), func { r -> processFile(fn, info, nil, r) })
	if s.GetExitCode() == 0 {
		t.Fatal("rewrite of read-only file succeeded unexpectedly")
	}
	if errBuf.Len() > 0 {
		t.Log(errBuf)
	}

	info, err = os.Stat(fn)
	if err != nil {
		t.Fatal(err)
	}
	if !info.ModTime().Equal(past) {
		t.Errorf("after rewrite mod time is %v, want %v", info.ModTime(), past)
	}
}
