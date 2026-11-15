// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strings"
	"testing"
)

func TestUnionNoTwoTierTag(t *testing.T) {
	src := "package main\ntype U interface {\nint | string\n}\nfunc f(x U) U { return switch x.(type) {\ncase int => 1\ncase string => 2\ncase nil => 0\n} }\n"
	out := lowerSnippetSrc(t, "notag.lgo", src)
	if strings.Count(out, "type U ") != 1 {
		t.Fatalf("lowered output must hold one U decl, got %d: %q", strings.Count(out, "type U "), out)
	}
	start := strings.Index(out, "type U interface")
	if start < 0 {
		t.Fatalf("lowered output missing union interface decl: %q", out)
	}
	rest := out[start:]
	before, _, found := strings.Cut(rest, "}")
	if !found {
		t.Fatalf("lowered output missing closing brace: %q", out)
	}
	decl := before + "}"
	if !strings.Contains(decl, "int | string") && !strings.Contains(decl, "int|string") {
		t.Fatalf("lowered output missing member set: %q", out)
	}
	if strings.Contains(decl, "Tag") || strings.Contains(decl, "struct") {
		t.Fatalf("lowered output uses two-tier tag layout: %q", out)
	}
}
