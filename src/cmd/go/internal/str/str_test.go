// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package str

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

var foldDupTests = []struct {
	list   []string
	f1, f2 string
}{
	{StringList("math/rand", "math/big"), "", ""},
	{StringList("math", "strings"), "", ""},
	{StringList("strings"), "", ""},
	{StringList("strings", "strings"), "strings", "strings"},
	{StringList("Rand", "rand", "math", "math/rand", "math/Rand"), "Rand", "rand"},
}

func TestFoldDup(t *testing.T) {
	for _, tt := range foldDupTests {
		f1, f2 := FoldDup(tt.list)
		if f1 != tt.f1 || f2 != tt.f2 {
			t.Errorf("foldDup(%q) = %q, %q, want %q, %q", tt.list, f1, f2, tt.f1, tt.f2)
		}
	}
}

func TestHasPathPrefix(t *testing.T) {
	type testCase struct {
		s, prefix string
		want      bool
	}
	for _, tt := range []testCase{
		{"", "", true},
		{"", "/", false},
		{"foo", "", true},
		{"foo", "/", false},
		{"foo", "foo", true},
		{"foo", "foo/", false},
		{"foo", "/foo", false},
		{"foo/bar", "", true},
		{"foo/bar", "foo", true},
		{"foo/bar", "foo/", true},
		{"foo/bar", "/foo", false},
		{"foo/bar", "foo/bar", true},
		{"foo/bar", "foo/bar/", false},
		{"foo/bar", "/foo/bar", false},
	} {
		got := HasPathPrefix(tt.s, tt.prefix)
		if got != tt.want {
			t.Errorf("HasPathPrefix(%q, %q) = %v; want %v", tt.s, tt.prefix, got, tt.want)
		}
	}
}

func TestTrimFilePathPrefixSlash(t *testing.T) {
	if os.PathSeparator != '/' {
		t.Skipf("test requires slash-separated file paths")
	}

	type testCase struct {
		s, prefix, want string
	}
	for _, tt := range []testCase{
		{"/", "", "/"},
		{"/", "/", ""},
		{"/foo", "", "/foo"},
		{"/foo", "/", "foo"},
		{"/foo", "/foo", ""},
		{"/foo/bar", "/foo", "bar"},
		{"/foo/bar", "/foo/", "bar"},
		{"/foo/", "/", "foo/"},
		{"/foo/", "/foo", ""},
		{"/foo/", "/foo/", ""},

		// if prefix is not s's prefix, return s
		{"", "/", ""},
		{"/foo", "/bar", "/foo"},
		{"/foo", "/foo/bar", "/foo"},
		{"foo", "/foo", "foo"},
		{"/foo", "foo", "/foo"},
		{"/foo", "/foo/", "/foo"},
	} {
		got := TrimFilePathPrefix(tt.s, tt.prefix)
		if got == tt.want {
			t.Logf("TrimFilePathPrefix(%q, %q) = %q", tt.s, tt.prefix, got)
		} else {
			t.Errorf("TrimFilePathPrefix(%q, %q) = %q, want %q", tt.s, tt.prefix, got, tt.want)
		}

		if HasFilePathPrefix(tt.s, tt.prefix) {
			joined := filepath.Join(tt.prefix, got)
			if clean := filepath.Clean(tt.s); joined != clean {
				t.Errorf("filepath.Join(%q, %q) = %q, want %q", tt.prefix, got, joined, clean)
			}
		}
	}
}

func TestTrimFilePathPrefixWindows(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skipf("test requires Windows file paths")
	}
	type testCase struct {
		s, prefix, want string
	}
	for _, tt := range []testCase{
		{`\`, ``, `\`},
		{`\`, `\`, ``},
		{`C:`, `C:`, ``},
		{`C:\`, `C:`, `\`},
		{`C:\`, `C:\`, ``},
		{`C:\foo`, ``, `C:\foo`},
		{`C:\foo`, `C:`, `\foo`},
		{`C:\foo`, `C:\`, `foo`},
		{`C:\foo`, `C:\foo`, ``},
		{`C:\foo\`, `C:\foo`, ``},
		{`C:\foo\bar`, `C:\foo`, `bar`},
		{`C:\foo\bar`, `C:\foo\`, `bar`},
		// if prefix is not s's prefix, return s
		{`C:\foo`, `C:\bar`, `C:\foo`},
		{`C:\foo`, `C:\foo\bar`, `C:\foo`},
		{`C:`, `C:\`, `C:`},
		// if volumes are different, return s
		{`C:`, ``, `C:`},
		{`C:\`, ``, `C:\`},
		{`C:\foo`, ``, `C:\foo`},
		{`C:\foo`, `\foo`, `C:\foo`},
		{`C:\foo`, `D:\foo`, `C:\foo`},

		//UNC path
		{`\\host\share\foo`, `\\host\share`, `foo`},
		{`\\host\share\foo`, `\\host\share\`, `foo`},
		{`\\host\share\foo`, `\\host\share\foo`, ``},
		{`\\host\share\foo\bar`, `\\host\share\foo`, `bar`},
		{`\\host\share\foo\bar`, `\\host\share\foo\`, `bar`},
		// if prefix is not s's prefix, return s
		{`\\host\share\foo`, `\\host\share\bar`, `\\host\share\foo`},
		{`\\host\share\foo`, `\\host\share\foo\bar`, `\\host\share\foo`},
		// if either host or share name is different, return s
		{`\\host\share\foo`, ``, `\\host\share\foo`},
		{`\\host\share\foo`, `\foo`, `\\host\share\foo`},
		{`\\host\share\foo`, `\\host\other\`, `\\host\share\foo`},
		{`\\host\share\foo`, `\\other\share\`, `\\host\share\foo`},
		{`\\host\share\foo`, `\\host\`, `\\host\share\foo`},
		{`\\host\share\foo`, `\share\`, `\\host\share\foo`},

		// only volume names are case-insensitive
		{`C:\foo`, `c:`, `\foo`},
		{`C:\foo`, `c:\foo`, ``},
		{`c:\foo`, `C:`, `\foo`},
		{`c:\foo`, `C:\foo`, ``},
		{`C:\foo`, `C:\Foo`, `C:\foo`},
		{`\\Host\Share\foo`, `\\host\share`, `foo`},
		{`\\Host\Share\foo`, `\\host\share\foo`, ``},
		{`\\host\share\foo`, `\\Host\Share`, `foo`},
		{`\\host\share\foo`, `\\Host\Share\foo`, ``},
		{`\\Host\Share\foo`, `\\Host\Share\Foo`, `\\Host\Share\foo`},
	} {
		got := TrimFilePathPrefix(tt.s, tt.prefix)
		if got == tt.want {
			t.Logf("TrimFilePathPrefix(%#q, %#q) = %#q", tt.s, tt.prefix, got)
		} else {
			t.Errorf("TrimFilePathPrefix(%#q, %#q) = %#q, want %#q", tt.s, tt.prefix, got, tt.want)
		}

		if HasFilePathPrefix(tt.s, tt.prefix) {
			// Although TrimFilePathPrefix is only case-insensitive in the volume name,
			// what we care about in testing Join is that absolute paths remain
			// absolute and relative paths remaining relative — there is no harm in
			// over-normalizing letters in the comparison, so we use EqualFold.
			joined := filepath.Join(tt.prefix, got)
			if clean := filepath.Clean(tt.s); !strings.EqualFold(joined, clean) {
				t.Errorf("filepath.Join(%#q, %#q) = %#q, want %#q", tt.prefix, got, joined, clean)
			}
		}
	}
}
