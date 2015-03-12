// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

import (
	"reflect"
	"sort"
	"strings"
	"testing"
)

var typeTests = initMimeForTests()

func TestTypeByExtension(t *testing.T) {
	for ext, want := range typeTests {
		val := TypeByExtension(ext)
		if val != want {
			t.Errorf("TypeByExtension(%q) = %q, want %q", ext, val, want)
		}
	}
}

func TestTypeByExtensionCase(t *testing.T) {
	const custom = "test/test; charset=iso-8859-1"
	const caps = "test/test; WAS=ALLCAPS"
	if err := AddExtensionType(".TEST", caps); err != nil {
		t.Fatalf("error %s for AddExtension(%s)", err, custom)
	}
	if err := AddExtensionType(".tesT", custom); err != nil {
		t.Fatalf("error %s for AddExtension(%s)", err, custom)
	}

	// case-sensitive lookup
	if got := TypeByExtension(".tesT"); got != custom {
		t.Fatalf("for .tesT, got %q; want %q", got, custom)
	}
	if got := TypeByExtension(".TEST"); got != caps {
		t.Fatalf("for .TEST, got %q; want %s", got, caps)
	}

	// case-insensitive
	if got := TypeByExtension(".TesT"); got != custom {
		t.Fatalf("for .TesT, got %q; want %q", got, custom)
	}
}

func TestExtensionsByType(t *testing.T) {
	for want, typ := range typeTests {
		val, err := ExtensionsByType(typ)
		if err != nil {
			t.Errorf("error %s for ExtensionsByType(%q)", err, typ)
			continue
		}
		if len(val) != 1 {
			t.Errorf("ExtensionsByType(%q) = %v; expected exactly 1 entry", typ, val)
			continue
		}
		// We always expect lower case, test data includes upper-case.
		want = strings.ToLower(want)
		if val[0] != want {
			t.Errorf("ExtensionsByType(%q) = %q, want %q", typ, val[0], want)
		}
	}
}

func TestExtensionsByTypeMultiple(t *testing.T) {
	const typ = "text/html"
	exts, err := ExtensionsByType(typ)
	if err != nil {
		t.Fatalf("ExtensionsByType(%q) error: %v", typ, err)
	}
	sort.Strings(exts)
	if want := []string{".htm", ".html"}; !reflect.DeepEqual(exts, want) {
		t.Errorf("ExtensionsByType(%q) = %v; want %v", typ, exts, want)
	}
}

func TestExtensionsByTypeNoDuplicates(t *testing.T) {
	const (
		typ = "text/html"
		ext = ".html"
	)
	AddExtensionType(ext, typ)
	AddExtensionType(ext, typ)
	exts, err := ExtensionsByType(typ)
	if err != nil {
		t.Fatalf("ExtensionsByType(%q) error: %v", typ, err)
	}
	count := 0
	for _, v := range exts {
		if v == ext {
			count++
		}
	}
	if count != 1 {
		t.Errorf("ExtensionsByType(%q) = %v; want %v once", typ, exts, ext)
	}
}

func TestLookupMallocs(t *testing.T) {
	n := testing.AllocsPerRun(10000, func() {
		TypeByExtension(".html")
		TypeByExtension(".HtML")
	})
	if n > 0 {
		t.Errorf("allocs = %v; want 0", n)
	}
}
