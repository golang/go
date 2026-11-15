// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"internal/logosfile"
	"testing"

	"cmd/compile/internal/syntax"
)

func TestOwnedConstraintHeaderByExtension(t *testing.T) {
	header := func(name string) string {
		return ownedConstraintHeader(&fileLower{baseName: name, file: &syntax.File{}})
	}
	want := header("x_linux_amd64.go")
	if want == "" {
		t.Fatal("the .go file name carries no constraint")
	}
	for _, ext := range logosfile.Exts() {
		for _, selector := range []string{"", "\uFE0F"} {
			name := "x_linux_amd64" + ext + selector
			if got := header(name); got != want {
				t.Errorf("ownedConstraintHeader(%q) = %q, want %q", name, got, want)
			}
		}
	}
}
