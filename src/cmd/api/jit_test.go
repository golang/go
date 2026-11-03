// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"go/build"
	"internal/testenv"
	"path/filepath"
	"testing"
)

// Experimental packages do not belong in the unconditional Go 1 API corpus:
// they are absent when their experiment is disabled. Keep an independently
// checked API inventory so changes remain reviewable without promising a stable
// machine ABI or weakening the normal standard-library compatibility checks.
func TestJITAPI(t *testing.T) {
	c := build.Default
	c.ToolTags = append(append([]string(nil), c.ToolTags...), "goexperiment.jit")
	root := testenv.GOROOT(t)
	w := NewWalker(&c, filepath.Join(root, "src"))
	p, err := w.import_("runtime/jit")
	if err != nil {
		t.Fatal(err)
	}
	w.export(p)
	required := fileFeatures(filepath.Join(root, "api/experimental/jit.txt"), true)
	var difference bytes.Buffer
	if !compareAPI(&difference, w.Features(), required, nil) {
		t.Fatalf("experimental JIT API differences:\n%s", &difference)
	}
}
