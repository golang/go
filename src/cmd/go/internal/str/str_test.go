// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package str

import (
	"os"
	"runtime"
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

type trimFilePathPrefixTest struct {
	s, prefix, want string
}

func TestTrimFilePathPrefixSlash(t *testing.T) {
	if os.PathSeparator != '/' {
		t.Skipf("test requires slash-separated file paths")
	}
	tests := []trimFilePathPrefixTest{
		{"/foo", "", "foo"},
		{"/foo", "/", "foo"},
		{"/foo", "/foo", ""},
		{"/foo/bar", "/foo", "bar"},
		{"/foo/bar", "/foo/", "bar"},
		// if prefix is not s's prefix, return s
		{"/foo", "/bar", "/foo"},
		{"/foo", "/foo/bar", "/foo"},
	}

	for _, tt := range tests {
		if got := TrimFilePathPrefix(tt.s, tt.prefix); got != tt.want {
			t.Errorf("TrimFilePathPrefix(%q, %q) = %q, want %q", tt.s, tt.prefix, got, tt.want)
		}
	}
}

func TestTrimFilePathPrefixWindows(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skipf("test requires Windows file paths")
	}
	tests := []trimFilePathPrefixTest{
		{`C:\foo`, `C:`, `foo`},
		{`C:\foo`, `C:\`, `foo`},
		{`C:\foo`, `C:\foo`, ``},
		{`C:\foo\bar`, `C:\foo`, `bar`},
		{`C:\foo\bar`, `C:\foo\`, `bar`},
		// if prefix is not s's prefix, return s
		{`C:\foo`, `C:\bar`, `C:\foo`},
		{`C:\foo`, `C:\foo\bar`, `C:\foo`},
		// if volumes are different, return s
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
	}

	for _, tt := range tests {
		if got := TrimFilePathPrefix(tt.s, tt.prefix); got != tt.want {
			t.Errorf("TrimFilePathPrefix(%q, %q) = %q, want %q", tt.s, tt.prefix, got, tt.want)
		}
	}
}
