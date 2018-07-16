// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package module

import "testing"

var checkTests = []struct {
	path    string
	version string
	ok      bool
}{
	{"rsc.io/quote", "0.1.0", false},
	{"rsc io/quote", "v1.0.0", false},

	{"github.com/go-yaml/yaml", "v0.8.0", true},
	{"github.com/go-yaml/yaml", "v1.0.0", true},
	{"github.com/go-yaml/yaml", "v2.0.0", false},
	{"github.com/go-yaml/yaml", "v2.1.5", false},
	{"github.com/go-yaml/yaml", "v3.0.0", false},

	{"github.com/go-yaml/yaml/v2", "v1.0.0", false},
	{"github.com/go-yaml/yaml/v2", "v2.0.0", true},
	{"github.com/go-yaml/yaml/v2", "v2.1.5", true},
	{"github.com/go-yaml/yaml/v2", "v3.0.0", false},

	{"gopkg.in/yaml.v0", "v0.8.0", true},
	{"gopkg.in/yaml.v0", "v1.0.0", false},
	{"gopkg.in/yaml.v0", "v2.0.0", false},
	{"gopkg.in/yaml.v0", "v2.1.5", false},
	{"gopkg.in/yaml.v0", "v3.0.0", false},

	{"gopkg.in/yaml.v1", "v0.8.0", false},
	{"gopkg.in/yaml.v1", "v1.0.0", true},
	{"gopkg.in/yaml.v1", "v2.0.0", false},
	{"gopkg.in/yaml.v1", "v2.1.5", false},
	{"gopkg.in/yaml.v1", "v3.0.0", false},

	{"gopkg.in/yaml.v2", "v1.0.0", false},
	{"gopkg.in/yaml.v2", "v2.0.0", true},
	{"gopkg.in/yaml.v2", "v2.1.5", true},
	{"gopkg.in/yaml.v2", "v3.0.0", false},
}

func TestCheck(t *testing.T) {
	for _, tt := range checkTests {
		err := Check(tt.path, tt.version)
		if tt.ok && err != nil {
			t.Errorf("Check(%q, %q) = %v, wanted nil error", tt.path, tt.version, err)
		} else if !tt.ok && err == nil {
			t.Errorf("Check(%q, %q) succeeded, wanted error", tt.path, tt.version)
		}
	}
}

var checkPathTests = []struct {
	path     string
	ok       bool
	importOK bool
}{
	{"x.y/z", true, true},
	{"x.y", true, true},

	{"", false, false},
	{"x.y/\xFFz", false, false},
	{"/x.y/z", false, false},
	{"x./z", false, false},
	{".x/z", false, false},
	{"-x/z", false, true},
	{"x..y/z", false, false},
	{"x.y/z/../../w", false, false},
	{"x.y//z", false, false},
	{"x.y/z//w", false, false},
	{"x.y/z/", false, false},

	{"x.y/z/v0", false, true},
	{"x.y/z/v1", false, true},
	{"x.y/z/v2", true, true},
	{"x.y/z/v2.0", false, true},
	{"X.y/z", false, true},

	{"!x.y/z", false, false},
	{"_x.y/z", false, true},
	{"x.y!/z", false, false},
	{"x.y\"/z", false, false},
	{"x.y#/z", false, false},
	{"x.y$/z", false, false},
	{"x.y%/z", false, false},
	{"x.y&/z", false, false},
	{"x.y'/z", false, false},
	{"x.y(/z", false, false},
	{"x.y)/z", false, false},
	{"x.y*/z", false, false},
	{"x.y+/z", false, true},
	{"x.y,/z", false, false},
	{"x.y-/z", true, true},
	{"x.y./zt", false, false},
	{"x.y:/z", false, false},
	{"x.y;/z", false, false},
	{"x.y</z", false, false},
	{"x.y=/z", false, false},
	{"x.y>/z", false, false},
	{"x.y?/z", false, false},
	{"x.y@/z", false, false},
	{"x.y[/z", false, false},
	{"x.y\\/z", false, false},
	{"x.y]/z", false, false},
	{"x.y^/z", false, false},
	{"x.y_/z", false, true},
	{"x.y`/z", false, false},
	{"x.y{/z", false, false},
	{"x.y}/z", false, false},
	{"x.y~/z", false, true},
	{"x.y/z!", false, false},
	{"x.y/z\"", false, false},
	{"x.y/z#", false, false},
	{"x.y/z$", false, false},
	{"x.y/z%", false, false},
	{"x.y/z&", false, false},
	{"x.y/z'", false, false},
	{"x.y/z(", false, false},
	{"x.y/z)", false, false},
	{"x.y/z*", false, false},
	{"x.y/z+", true, true},
	{"x.y/z,", false, false},
	{"x.y/z-", true, true},
	{"x.y/z.t", true, true},
	{"x.y/z/t", true, true},
	{"x.y/z:", false, false},
	{"x.y/z;", false, false},
	{"x.y/z<", false, false},
	{"x.y/z=", false, false},
	{"x.y/z>", false, false},
	{"x.y/z?", false, false},
	{"x.y/z@", false, false},
	{"x.y/z[", false, false},
	{"x.y/z\\", false, false},
	{"x.y/z]", false, false},
	{"x.y/z^", false, false},
	{"x.y/z_", true, true},
	{"x.y/z`", false, false},
	{"x.y/z{", false, false},
	{"x.y/z}", false, false},
	{"x.y/z~", true, true},
	{"x.y/x.foo", true, true},
	{"x.y/aux.foo", false, false},
	{"x.y/prn", false, false},
	{"x.y/prn2", true, true},
	{"x.y/com", true, true},
	{"x.y/com1", false, false},
	{"x.y/com1.txt", false, false},
	{"x.y/calm1", true, true},
	{"github.com/!123/logrus", false, false},

	// TODO: CL 41822 allowed Unicode letters in old "go get"
	// without due consideration of the implications, and only on github.com (!).
	// For now, we disallow non-ASCII characters in module mode,
	// in both module paths and general import paths,
	// until we can get the implications right.
	// When we do, we'll enable them everywhere, not just for GitHub.
	{"github.com/user/unicode/испытание", false, false},
}

func TestCheckPath(t *testing.T) {
	for _, tt := range checkPathTests {
		err := CheckPath(tt.path)
		if tt.ok && err != nil {
			t.Errorf("CheckPath(%q) = %v, wanted nil error", tt.path, err)
		} else if !tt.ok && err == nil {
			t.Errorf("CheckPath(%q) succeeded, wanted error", tt.path)
		}

		err = CheckImportPath(tt.path)
		if tt.importOK && err != nil {
			t.Errorf("CheckImportPath(%q) = %v, wanted nil error", tt.path, err)
		} else if !tt.importOK && err == nil {
			t.Errorf("CheckImportPath(%q) succeeded, wanted error", tt.path)
		}
	}
}

var splitPathVersionTests = []struct {
	pathPrefix string
	version    string
}{
	{"x.y/z", ""},
	{"x.y/z", "/v2"},
	{"x.y/z", "/v3"},
	{"gopkg.in/yaml", ".v0"},
	{"gopkg.in/yaml", ".v1"},
	{"gopkg.in/yaml", ".v2"},
	{"gopkg.in/yaml", ".v3"},
}

func TestSplitPathVersion(t *testing.T) {
	for _, tt := range splitPathVersionTests {
		pathPrefix, version, ok := SplitPathVersion(tt.pathPrefix + tt.version)
		if pathPrefix != tt.pathPrefix || version != tt.version || !ok {
			t.Errorf("SplitPathVersion(%q) = %q, %q, %v, want %q, %q, true", tt.pathPrefix+tt.version, pathPrefix, version, ok, tt.pathPrefix, tt.version)
		}
	}

	for _, tt := range checkPathTests {
		pathPrefix, version, ok := SplitPathVersion(tt.path)
		if pathPrefix+version != tt.path {
			t.Errorf("SplitPathVersion(%q) = %q, %q, %v, doesn't add to input", tt.path, pathPrefix, version, ok)
		}
	}
}

var encodeTests = []struct {
	path string
	enc  string // empty means same as path
}{
	{path: "ascii.com/abcdefghijklmnopqrstuvwxyz.-+/~_0123456789"},
	{path: "github.com/GoogleCloudPlatform/omega", enc: "github.com/!google!cloud!platform/omega"},
}

func TestEncodePath(t *testing.T) {
	// Check invalid paths.
	for _, tt := range checkPathTests {
		if !tt.ok {
			_, err := EncodePath(tt.path)
			if err == nil {
				t.Errorf("EncodePath(%q): succeeded, want error (invalid path)", tt.path)
			}
		}
	}

	// Check encodings.
	for _, tt := range encodeTests {
		enc, err := EncodePath(tt.path)
		if err != nil {
			t.Errorf("EncodePath(%q): unexpected error: %v", tt.path, err)
			continue
		}
		want := tt.enc
		if want == "" {
			want = tt.path
		}
		if enc != want {
			t.Errorf("EncodePath(%q) = %q, want %q", tt.path, enc, want)
		}
	}
}

var badDecode = []string{
	"github.com/GoogleCloudPlatform/omega",
	"github.com/!google!cloud!platform!/omega",
	"github.com/!0google!cloud!platform/omega",
	"github.com/!_google!cloud!platform/omega",
	"github.com/!!google!cloud!platform/omega",
	"",
}

func TestDecodePath(t *testing.T) {
	// Check invalid decodings.
	for _, bad := range badDecode {
		_, err := DecodePath(bad)
		if err == nil {
			t.Errorf("DecodePath(%q): succeeded, want error (invalid decoding)", bad)
		}
	}

	// Check invalid paths (or maybe decodings).
	for _, tt := range checkPathTests {
		if !tt.ok {
			path, err := DecodePath(tt.path)
			if err == nil {
				t.Errorf("DecodePath(%q) = %q, want error (invalid path)", tt.path, path)
			}
		}
	}

	// Check encodings.
	for _, tt := range encodeTests {
		enc := tt.enc
		if enc == "" {
			enc = tt.path
		}
		path, err := DecodePath(enc)
		if err != nil {
			t.Errorf("DecodePath(%q): unexpected error: %v", enc, err)
			continue
		}
		if path != tt.path {
			t.Errorf("DecodePath(%q) = %q, want %q", enc, path, tt.path)
		}
	}
}
