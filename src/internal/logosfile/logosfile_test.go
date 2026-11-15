// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logosfile

import (
	"testing"

	"golang.org/x/text/unicode/norm"
)

func TestSplitEveryExtension(t *testing.T) {
	for _, ext := range Exts() {
		if norm.NFC.String(ext) != ext {
			t.Errorf("NFC(%q) differs from %q", ext, ext)
		}
		for _, suffix := range []string{"", "\uFE0E", "\uFE0F", "\uFE0E\uFE0F", "\uFE0F\uFE0E"} {
			for _, name := range []string{"x" + ext + suffix, "dir/sub.d/x_test" + ext + suffix, ext + suffix} {
				stem, got, ok := Split(name)
				if !ok || stem+got != name {
					t.Errorf("Split(%q) = %q, %q, %v", name, stem, got, ok)
				}
				if !Is(name) {
					t.Errorf("Is(%q) = false", name)
				}
			}
		}
	}
}

func TestSplitRejectsOtherNames(t *testing.T) {
	for _, name := range []string{"", "x", "x.go", "x.lgo.go", "x.lgox", "xlgo", "x.lgo/y", ".ʕ◔π◔ʔ", "x.⚡.txt", "x.\uFE0F", "main.txt", "x.lgo.bak", "main.⚡x"} {
		if Is(name) {
			t.Errorf("Is(%q) = true", name)
		}
	}
}

func TestExtsReturnsACopy(t *testing.T) {
	got := Exts()
	if len(got) != 4 || got[0] != Canonical {
		t.Fatalf("Exts() = %q", got)
	}
	got[0] = ".go"
	if Exts()[0] != Canonical {
		t.Fatal("Exts() shares its backing array")
	}
}

func TestExtAndIsAlias(t *testing.T) {
	for _, ext := range Exts() {
		for _, suffix := range []string{"", "\uFE0E", "\uFE0F", "\uFE0F\uFE0E"} {
			name := "dir.d/x_test" + ext + suffix
			if got := Ext(name); got != ext {
				t.Errorf("Ext(%q) = %q, want %q", name, got, ext)
			}
			if got, want := IsAlias(name), ext != Canonical; got != want {
				t.Errorf("IsAlias(%q) = %v, want %v", name, got, want)
			}
		}
	}
	for _, name := range []string{"x.go", "x", "x.lgo.go"} {
		if Ext(name) != "" || IsAlias(name) {
			t.Errorf("%q is not loGo/s source", name)
		}
	}
}
