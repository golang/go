// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importpath

import (
	"testing"
)

func TestImportPath(t *testing.T) {
	tests := []string{
		"net",
		"net/http",
		"a/b/c",
	}
	for _, tc := range tests {
		if !IsValidImport(tc) {
			t.Errorf("expected %q to be valid import path", tc)
		}
	}
}

func TestImportPathInvalid(t *testing.T) {
	tests := []string{
		"",
		"foo bar",
		"\uFFFD",
		"hello!",
	}
	for _, tc := range tests {
		if IsValidImport(tc) {
			t.Errorf("expected %q to be invalid import path", tc)
		}
	}
}
