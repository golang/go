// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

import (
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

func TestLookupMallocs(t *testing.T) {
	n := testing.AllocsPerRun(10000, func() {
		TypeByExtension(".html")
		TypeByExtension(".HtML")
	})
	if n > 0 {
		t.Errorf("allocs = %v; want 0", n)
	}
}
