// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sumweb

import "testing"

var encodeTests = []struct {
	path string
	enc  string // empty means same as path
}{
	{path: "ascii.com/abcdefghijklmnopqrstuvwxyz.-+/~_0123456789"},
	{path: "github.com/GoogleCloudPlatform/omega", enc: "github.com/!google!cloud!platform/omega"},
}

func TestEncodePath(t *testing.T) {
	// Check encodings.
	for _, tt := range encodeTests {
		enc, err := encodePath(tt.path)
		if err != nil {
			t.Errorf("encodePath(%q): unexpected error: %v", tt.path, err)
			continue
		}
		want := tt.enc
		if want == "" {
			want = tt.path
		}
		if enc != want {
			t.Errorf("encodePath(%q) = %q, want %q", tt.path, enc, want)
		}
	}
}

var badDecode = []string{
	"github.com/GoogleCloudPlatform/omega",
	"github.com/!google!cloud!platform!/omega",
	"github.com/!0google!cloud!platform/omega",
	"github.com/!_google!cloud!platform/omega",
	"github.com/!!google!cloud!platform/omega",
}

func TestDecodePath(t *testing.T) {
	// Check invalid decodings.
	for _, bad := range badDecode {
		_, err := decodePath(bad)
		if err == nil {
			t.Errorf("DecodePath(%q): succeeded, want error (invalid decoding)", bad)
		}
	}

	// Check encodings.
	for _, tt := range encodeTests {
		enc := tt.enc
		if enc == "" {
			enc = tt.path
		}
		path, err := decodePath(enc)
		if err != nil {
			t.Errorf("decodePath(%q): unexpected error: %v", enc, err)
			continue
		}
		if path != tt.path {
			t.Errorf("decodePath(%q) = %q, want %q", enc, path, tt.path)
		}
	}
}
