// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"os"
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
	filename := "foo.out"
	ob := NewOutBuf(nil)
	if err := ob.Open(filename); err != nil {
		t.Errorf("error opening file: %v", err)
	}
	defer os.RemoveAll(filename)
	defer ob.Close()
	if err := ob.Mmap(1 << 20); err != nil {
		t.Errorf("error mmapping file %v", err)
	}
}
