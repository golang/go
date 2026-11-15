// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"internal/logosfile"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
)

func TestUnsafeImportAllowedByEveryExtension(t *testing.T) {
	want := map[string]bool{"x_unsafe": true, "x_test": true, "x": false, "x_unsafe_other": false}
	exts := append(logosfile.Exts(), ".go")
	for _, ext := range exts {
		for _, selector := range []string{"", "\uFE0F"} {
			if ext == ".go" && selector != "" {
				continue
			}
			for stem, allowed := range want {
				name := stem + ext + selector
				file, _ := syntax.Parse(syntax.NewFileBase(name), strings.NewReader("package p\n"), nil, nil, syntax.LogosMode)
				if file == nil {
					t.Fatalf("%q did not parse", name)
				}
				if got := unsafeImportAllowed(file); got != allowed {
					t.Errorf("unsafeImportAllowed(%q) = %v, want %v", name, got, allowed)
				}
			}
		}
	}
}
