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
	path string
	ok   bool
}{
	{"x.y/z", true},
	{"x.y", true},

	{"", false},
	{"x.y/\xFFz", false},
	{"/x.y/z", false},
	{"x./z", false},
	{".x/z", false},
	{"-x/z", false},
	{"x..y/z", false},
	{"x.y/z/../../w", false},
	{"x.y//z", false},
	{"x.y/z//w", false},
	{"x.y/z/", false},

	{"x.y/z/v0", false},
	{"x.y/z/v1", false},
	{"x.y/z/v2", true},
	{"x.y/z/v2.0", false},

	{"!x.y/z", false},
	{"_x.y/z", false},
	{"x.y!/z", false},
	{"x.y\"/z", false},
	{"x.y#/z", false},
	{"x.y$/z", false},
	{"x.y%/z", false},
	{"x.y&/z", false},
	{"x.y'/z", false},
	{"x.y(/z", false},
	{"x.y)/z", false},
	{"x.y*/z", false},
	{"x.y+/z", false},
	{"x.y,/z", false},
	{"x.y-/z", true},
	{"x.y./zt", false},
	{"x.y:/z", false},
	{"x.y;/z", false},
	{"x.y</z", false},
	{"x.y=/z", false},
	{"x.y>/z", false},
	{"x.y?/z", false},
	{"x.y@/z", false},
	{"x.y[/z", false},
	{"x.y\\/z", false},
	{"x.y]/z", false},
	{"x.y^/z", false},
	{"x.y_/z", false},
	{"x.y`/z", false},
	{"x.y{/z", false},
	{"x.y}/z", false},
	{"x.y~/z", false},
	{"x.y/z!", false},
	{"x.y/z\"", false},
	{"x.y/z#", false},
	{"x.y/z$", false},
	{"x.y/z%", false},
	{"x.y/z&", false},
	{"x.y/z'", false},
	{"x.y/z(", false},
	{"x.y/z)", false},
	{"x.y/z*", false},
	{"x.y/z+", true},
	{"x.y/z,", true},
	{"x.y/z-", true},
	{"x.y/z.t", true},
	{"x.y/z/t", true},
	{"x.y/z:", false},
	{"x.y/z;", false},
	{"x.y/z<", false},
	{"x.y/z=", false},
	{"x.y/z>", false},
	{"x.y/z?", false},
	{"x.y/z@", false},
	{"x.y/z[", false},
	{"x.y/z\\", false},
	{"x.y/z]", false},
	{"x.y/z^", false},
	{"x.y/z_", true},
	{"x.y/z`", false},
	{"x.y/z{", false},
	{"x.y/z}", false},
	{"x.y/z~", true},
}

func TestCheckPath(t *testing.T) {
	for _, tt := range checkPathTests {
		err := CheckPath(tt.path)
		if tt.ok && err != nil {
			t.Errorf("CheckPath(%q) = %v, wanted nil error", tt.path, err)
		} else if !tt.ok && err == nil {
			t.Errorf("CheckPath(%q) succeeded, wanted error", tt.path)
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
