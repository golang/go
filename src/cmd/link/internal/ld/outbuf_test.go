// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestMMap ensures that we can actually mmap on every supported platform.
func TestMMap(t *testing.T) {
	switch runtime.GOOS {
	default:
		t.Skip("unsupported OS")
	case "darwin", "dragonfly", "freebsd", "linux", "openbsd", "windows":
	}
	dir, err := ioutil.TempDir("", "TestMMap")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	filename := filepath.Join(dir, "foo.out")
	ob := NewOutBuf(nil)
	if err := ob.Open(filename); err != nil {
		t.Fatalf("error opening file: %v", err)
	}
	defer ob.Close()
	if err := ob.Mmap(1 << 20); err != nil {
		t.Errorf("error mmapping file %v", err)
	}
}
