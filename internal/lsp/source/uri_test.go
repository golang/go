// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"testing"
)

// TestURI tests the conversion between URIs and filenames. The test cases
// include Windows-style URIs and filepaths, but we avoid having OS-specific
// tests by using only forward slashes, assuming that the standard library
// functions filepath.ToSlash and filepath.FromSlash do not need testing.
func TestURI(t *testing.T) {
	for _, tt := range []struct {
		uri      URI
		filename string
	}{
		{
			uri:      URI(`file:///C:/Windows/System32`),
			filename: `C:/Windows/System32`,
		},
		{
			uri:      URI(`file:///C:/Go/src/bob.go`),
			filename: `C:/Go/src/bob.go`,
		},
		{
			uri:      URI(`file:///c:/Go/src/bob.go`),
			filename: `c:/Go/src/bob.go`,
		},
		{
			uri:      URI(`file:///path/to/dir`),
			filename: `/path/to/dir`,
		},
		{
			uri:      URI(`file:///a/b/c/src/bob.go`),
			filename: `/a/b/c/src/bob.go`,
		},
	} {
		if string(tt.uri) != toURI(tt.filename).String() {
			t.Errorf("ToURI: expected %s, got %s", tt.uri, ToURI(tt.filename))
		}
		filename, err := filename(tt.uri)
		if err != nil {
			t.Fatal(err)
		}
		if tt.filename != filename {
			t.Errorf("Filename: expected %s, got %s", tt.filename, filename)
		}
	}
}
