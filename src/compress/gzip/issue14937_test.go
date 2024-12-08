// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gzip

import (
	"internal/testenv"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestGZIPFilesHaveZeroMTimes checks that every .gz file in the tree
// has a zero MTIME. This is a requirement for the Debian maintainers
// to be able to have deterministic packages.
//
// To patch a .gz file, use the following command:
//
//	$ dd if=/dev/zero bs=1 seek=4 count=4 conv=notrunc of=filename.gz
//
// See https://golang.org/issue/14937.
func TestGZIPFilesHaveZeroMTimes(t *testing.T) {
	// To avoid spurious false positives due to untracked GZIP files that
	// may be in the user's GOROOT (Issue 18604), we only run this test on
	// the builders, which should have a clean checkout of the tree.
	if testenv.Builder() == "" {
		t.Skip("skipping test on non-builder")
	}
	testenv.MustHaveSource(t)

	goroot, err := filepath.EvalSymlinks(runtime.GOROOT())
	if err != nil {
		t.Fatal("error evaluating GOROOT: ", err)
	}
	var files []string
	err = filepath.WalkDir(goroot, func(path string, info fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(path, ".gz") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		if os.IsNotExist(err) {
			t.Skipf("skipping: GOROOT directory not found: %s", runtime.GOROOT())
		}
		t.Fatal("error collecting list of .gz files in GOROOT: ", err)
	}
	if len(files) == 0 {
		t.Fatal("expected to find some .gz files under GOROOT")
	}
	for _, path := range files {
		checkZeroMTime(t, path)
	}
}

func checkZeroMTime(t *testing.T, path string) {
	f, err := os.Open(path)
	if err != nil {
		t.Error(err)
		return
	}
	defer f.Close()
	gz, err := NewReader(f)
	if err != nil {
		t.Errorf("cannot read gzip file %s: %s", path, err)
		return
	}
	defer gz.Close()
	if !gz.ModTime.IsZero() {
		t.Errorf("gzip file %s has non-zero mtime (%s)", path, gz.ModTime)
	}
}
