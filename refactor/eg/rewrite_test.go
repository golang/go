// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eg

import "testing"

func TestVendorlessImportPath(t *testing.T) {
	tests := []struct {
		path  string
		ipath string
	}{
		{"a/b/c", "a/b/c"},
		{"a/b/vendor/c", "c"},
		{"a/vendor/b/c", "b/c"},
		{"vendor/a/b/c", "a/b/c"},
		{"a/b/vendor/vendor/c", "c"},
		{"a/vendor/b/vendor/c", "c"},
		{"vendor/a/b/vendor/c", "c"},
		{"vendor/a/vendor/b/c", "b/c"},
		{"a/vendor/vendor/b/c", "b/c"},
		{"vendor/vendor/a/b/c", "a/b/c"},

		{"a/b/notvendor/c", "a/b/notvendor/c"},
		{"a/b/vendors/c", "a/b/vendors/c"},
		{"a/b/notvendors/c", "a/b/notvendors/c"},
		{"notvendor/a/b/c", "notvendor/a/b/c"},
	}

	for _, tt := range tests {
		if have, want := vendorlessImportPath(tt.path), tt.ipath; have != want {
			t.Errorf("vendorlessImportPath(%q); %q != %q", tt.path, have, want)
		}
	}
}
