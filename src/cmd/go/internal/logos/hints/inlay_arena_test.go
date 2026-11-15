// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hints

import (
	"slices"
	"testing"
)

func TestInlayArenaBadgeFromCompiler(t *testing.T) {
	f := compilerFixture(t, "sites.lgo")
	t.Setenv("GOEXPERIMENT", "nologosarena")
	r, _ := compileHints(t, f)
	if got := labelsAt(InlayHints(r, f.path, wholeFile), endOf(t, f.src, "make")); slices.Contains(got, "[arena]") {
		t.Errorf("default build: make([]byte, n) hints %q, want no [arena]", got)
	}
	t.Setenv("GOEXPERIMENT", "logosarena")
	ar, _ := compileHints(t, f)
	if got := labelsAt(InlayHints(ar, f.path, wholeFile), endOf(t, f.src, "make")); !slices.Contains(got, "[arena]") {
		t.Errorf("arena build: make([]byte, n) hints %q, want [arena]", got)
	}
}
