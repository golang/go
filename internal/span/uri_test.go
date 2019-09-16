// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package span_test

import (
	"path/filepath"
	"testing"

	"golang.org/x/tools/internal/span"
)

// TestURI tests the conversion between URIs and filenames. The test cases
// include Windows-style URIs and filepaths, but we avoid having OS-specific
// tests by using only forward slashes, assuming that the standard library
// functions filepath.ToSlash and filepath.FromSlash do not need testing.
func TestURI(t *testing.T) {
	for _, test := range []string{
		``,
		`C:/Windows/System32`,
		`C:/Go/src/bob.go`,
		`c:/Go/src/bob.go`,
		`/path/to/dir`,
		`/a/b/c/src/bob.go`,
		`c:/Go/src/bob george/george/george.go`,
	} {
		testPath := filepath.FromSlash(test)
		expectPath := testPath
		if len(test) > 0 && test[0] == '/' {
			if abs, err := filepath.Abs(expectPath); err == nil {
				expectPath = abs
			}
		}
		expectURI := filepath.ToSlash(expectPath)
		if len(expectURI) > 0 {
			if expectURI[0] != '/' {
				expectURI = "/" + expectURI
			}
			expectURI = "file://" + expectURI
		}
		uri := span.FileURI(testPath)
		if expectURI != string(uri) {
			t.Errorf("ToURI: expected %s, got %s", expectURI, uri)
		}
		filename := uri.Filename()
		if expectPath != filename {
			t.Errorf("Filename: expected %s, got %s", expectPath, filename)
		}
	}
}
