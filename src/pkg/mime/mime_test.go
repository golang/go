// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for type.go

package mime

import "testing"

var typeTests = map[string]string{
	".t1":  "application/test",
	".t2":  "text/test; charset=utf-8",
	".png": "image/png",
}

func TestType(t *testing.T) {
	typeFiles = []string{"test.types"}

	for ext, want := range typeTests {
		val := TypeByExtension(ext)
		if val != want {
			t.Errorf("TypeByExtension(%q) = %q, want %q", ext, val, want)
		}

	}
}
