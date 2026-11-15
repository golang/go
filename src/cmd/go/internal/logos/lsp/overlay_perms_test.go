// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"os"
	"path/filepath"
	"testing"
)

func TestOverlayBuffersStayOwnerOnly(t *testing.T) {
	ov, err := newOverlay(map[string][]byte{filepath.Join(t.TempDir(), "a.lgo"): []byte("package a\n")})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := ov.release(); err != nil {
			t.Errorf("releasing the overlay: %v", err)
		}
	}()
	var paths []string
	paths = append(paths, ov.json)
	for backing := range ov.back {
		paths = append(paths, backing, filepath.Dir(backing))
	}
	for _, p := range paths {
		fi, err := os.Stat(p)
		if err != nil {
			t.Fatal(err)
		}
		if fi.Mode().Perm()&0o077 != 0 {
			t.Errorf("overlay path %s mode %o grants group or other access", p, fi.Mode().Perm())
		}
	}
}
