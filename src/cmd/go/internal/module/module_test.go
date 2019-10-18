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

	// For gopkg.in, .v1 means v1 only (not v0).
	// But early versions of vgo still generated v0 pseudo-versions for it.
	// Even though now we'd generate those as v1 pseudo-versions,
	// we accept the old pseudo-versions to avoid breaking existing go.mod files.
	// For example gopkg.in/yaml.v2@v2.2.1's go.mod requires check.v1 at a v0 pseudo-version.
	{"gopkg.in/check.v1", "v0.0.0", false},
	{"gopkg.in/check.v1", "v0.0.0-20160102150405-abcdef123456", true},

	{"gopkg.in/yaml.v2", "v1.0.0", false},
	{"gopkg.in/yaml.v2", "v2.0.0", true},
	{"gopkg.in/yaml.v2", "v2.1.5", true},
	{"gopkg.in/yaml.v2", "v3.0.0", false},

	{"rsc.io/quote", "v17.0.0", false},
	{"rsc.io/quote", "v17.0.0+incompatible", true},
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
	fileOK   bool
}{
	{"x.y/z", true, true, true},
	{"x.y", true, true, true},

	{"", false, false, false},
	{"x.y/\xFFz", false, false, false},
	{"/x.y/z", false, false, false},
	{"x./z", false, false, false},
	{".x/z", false, false, true},
	{"-x/z", false, false, false},
	{"x..y/z", true, true, true},
	{"x.y/z/../../w", false, false, false},
	{"x.y//z", false, false, false},
	{"x.y/z//w", false, false, false},
	{"x.y/z/", false, false, false},

	{"x.y/z/v0", false, true, true},
	{"x.y/z/v1", false, true, true},
	{"x.y/z/v2", true, true, true},
	{"x.y/z/v2.0", false, true, true},
	{"X.y/z", false, true, true},

	{"!x.y/z", false, false, true},
	{"_x.y/z", false, true, true},
	{"x.y!/z", false, false, true},
	{"x.y\"/z", false, false, false},
	{"x.y#/z", false, false, true},
	{"x.y$/z", false, false, true},
	{"x.y%/z", false, false, true},
	{"x.y&/z", false, false, true},
	{"x.y'/z", false, false, false},
	{"x.y(/z", false, false, true},
	{"x.y)/z", false, false, true},
	{"x.y*/z", false, false, false},
	{"x.y+/z", false, true, true},
	{"x.y,/z", false, false, true},
	{"x.y-/z", true, true, true},
	{"x.y./zt", false, false, false},
	{"x.y:/z", false, false, false},
	{"x.y;/z", false, false, false},
	{"x.y</z", false, false, false},
	{"x.y=/z", false, false, true},
	{"x.y>/z", false, false, false},
	{"x.y?/z", false, false, false},
	{"x.y@/z", false, false, true},
	{"x.y[/z", false, false, true},
	{"x.y\\/z", false, false, false},
	{"x.y]/z", false, false, true},
	{"x.y^/z", false, false, true},
	{"x.y_/z", false, true, true},
	{"x.y`/z", false, false, false},
	{"x.y{/z", false, false, true},
	{"x.y}/z", false, false, true},
	{"x.y~/z", false, true, true},
	{"x.y/z!", false, false, true},
	{"x.y/z\"", false, false, false},
	{"x.y/z#", false, false, true},
	{"x.y/z$", false, false, true},
	{"x.y/z%", false, false, true},
	{"x.y/z&", false, false, true},
	{"x.y/z'", false, false, false},
	{"x.y/z(", false, false, true},
	{"x.y/z)", false, false, true},
	{"x.y/z*", false, false, false},
	{"x.y/z+", true, true, true},
	{"x.y/z,", false, false, true},
	{"x.y/z-", true, true, true},
	{"x.y/z.t", true, true, true},
	{"x.y/z/t", true, true, true},
	{"x.y/z:", false, false, false},
	{"x.y/z;", false, false, false},
	{"x.y/z<", false, false, false},
	{"x.y/z=", false, false, true},
	{"x.y/z>", false, false, false},
	{"x.y/z?", false, false, false},
	{"x.y/z@", false, false, true},
	{"x.y/z[", false, false, true},
	{"x.y/z\\", false, false, false},
	{"x.y/z]", false, false, true},
	{"x.y/z^", false, false, true},
	{"x.y/z_", true, true, true},
	{"x.y/z`", false, false, false},
	{"x.y/z{", false, false, true},
	{"x.y/z}", false, false, true},
	{"x.y/z~", true, true, true},
	{"x.y/x.foo", true, true, true},
	{"x.y/aux.foo", false, false, false},
	{"x.y/prn", false, false, false},
	{"x.y/prn2", true, true, true},
	{"x.y/com", true, true, true},
	{"x.y/com1", false, false, false},
	{"x.y/com1.txt", false, false, false},
	{"x.y/calm1", true, true, true},
	{"github.com/!123/logrus", false, false, true},

	// TODO: CL 41822 allowed Unicode letters in old "go get"
	// without due consideration of the implications, and only on github.com (!).
	// For now, we disallow non-ASCII characters in module mode,
	// in both module paths and general import paths,
	// until we can get the implications right.
	// When we do, we'll enable them everywhere, not just for GitHub.
	{"github.com/user/unicode/испытание", false, false, true},

	{".../x", false, false, false},
	{"../x", false, false, false},
	{"./y", false, false, false},
	{"x:y", false, false, false},
	{`\temp\foo`, false, false, false},
	{".gitignore", false, false, true},
	{".github/ISSUE_TEMPLATE", false, false, true},
	{"x☺y", false, false, false},
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

		err = CheckFilePath(tt.path)
		if tt.fileOK && err != nil {
			t.Errorf("CheckFilePath(%q) = %v, wanted nil error", tt.path, err)
		} else if !tt.fileOK && err == nil {
			t.Errorf("CheckFilePath(%q) succeeded, wanted error", tt.path)
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
	{"x.y/v", ""},
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

var escapeTests = []struct {
	path string
	esc  string // empty means same as path
}{
	{path: "ascii.com/abcdefghijklmnopqrstuvwxyz.-+/~_0123456789"},
	{path: "github.com/GoogleCloudPlatform/omega", esc: "github.com/!google!cloud!platform/omega"},
}

func TestEscapePath(t *testing.T) {
	// Check invalid paths.
	for _, tt := range checkPathTests {
		if !tt.ok {
			_, err := EscapePath(tt.path)
			if err == nil {
				t.Errorf("EscapePath(%q): succeeded, want error (invalid path)", tt.path)
			}
		}
	}

	// Check encodings.
	for _, tt := range escapeTests {
		esc, err := EscapePath(tt.path)
		if err != nil {
			t.Errorf("EscapePath(%q): unexpected error: %v", tt.path, err)
			continue
		}
		want := tt.esc
		if want == "" {
			want = tt.path
		}
		if esc != want {
			t.Errorf("EscapePath(%q) = %q, want %q", tt.path, esc, want)
		}
	}
}

var badUnescape = []string{
	"github.com/GoogleCloudPlatform/omega",
	"github.com/!google!cloud!platform!/omega",
	"github.com/!0google!cloud!platform/omega",
	"github.com/!_google!cloud!platform/omega",
	"github.com/!!google!cloud!platform/omega",
	"",
}

func TestUnescapePath(t *testing.T) {
	// Check invalid decodings.
	for _, bad := range badUnescape {
		_, err := UnescapePath(bad)
		if err == nil {
			t.Errorf("UnescapePath(%q): succeeded, want error (invalid decoding)", bad)
		}
	}

	// Check invalid paths (or maybe decodings).
	for _, tt := range checkPathTests {
		if !tt.ok {
			path, err := UnescapePath(tt.path)
			if err == nil {
				t.Errorf("UnescapePath(%q) = %q, want error (invalid path)", tt.path, path)
			}
		}
	}

	// Check encodings.
	for _, tt := range escapeTests {
		esc := tt.esc
		if esc == "" {
			esc = tt.path
		}
		path, err := UnescapePath(esc)
		if err != nil {
			t.Errorf("UnescapePath(%q): unexpected error: %v", esc, err)
			continue
		}
		if path != tt.path {
			t.Errorf("UnescapePath(%q) = %q, want %q", esc, path, tt.path)
		}
	}
}

func TestMatchPathMajor(t *testing.T) {
	for _, test := range []struct {
		v, pathMajor string
		want         bool
	}{
		{"v0.0.0", "", true},
		{"v0.0.0", "/v2", false},
		{"v0.0.0", ".v0", true},
		{"v0.0.0-20190510104115-cbcb75029529", ".v1", true},
		{"v1.0.0", "/v2", false},
		{"v1.0.0", ".v1", true},
		{"v1.0.0", ".v1-unstable", true},
		{"v2.0.0+incompatible", "", true},
		{"v2.0.0", "", false},
		{"v2.0.0", "/v2", true},
		{"v2.0.0", ".v2", true},
	} {
		if got := MatchPathMajor(test.v, test.pathMajor); got != test.want {
			t.Errorf("MatchPathMajor(%q, %q) = %v, want %v", test.v, test.pathMajor, got, test.want)
		}
	}
}
