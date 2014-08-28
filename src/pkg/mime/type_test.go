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

func TestCustomExtension(t *testing.T) {
	custom := "test/test; charset=iso-8859-1"
	if error := AddExtensionType(".tesT", custom); error != nil {
		t.Fatalf("error %s for AddExtension(%s)", error, custom)
	}
	// test with same capitalization
	if registered := TypeByExtension(".tesT"); registered != custom {
		t.Fatalf("registered %s instead of %s", registered, custom)
	}
	// test with different capitalization
	if registered := TypeByExtension(".Test"); registered != custom {
		t.Fatalf("registered %s instead of %s", registered, custom)
	}
}
