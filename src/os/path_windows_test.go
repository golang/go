// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"os"
	"testing"
)

func TestFixLongPath(t *testing.T) {
	for _, test := range []struct{ in, want string }{
		{`C:\foo.txt`, `\\?\C:\foo.txt`},
		{`C:/foo.txt`, `\\?\C:\foo.txt`},
		{`C:\foo\\bar\.\baz\\`, `\\?\C:\foo\bar\baz`},
		{`C:\`, `\\?\C:\`}, // drives must have a trailing slash
		{`\\unc\path`, `\\unc\path`},
		{`foo.txt`, `foo.txt`},
		{`C:foo.txt`, `C:foo.txt`},
		{`c:\foo\..\bar\baz`, `c:\foo\..\bar\baz`},
		{`\\?\c:\windows\foo.txt`, `\\?\c:\windows\foo.txt`},
		{`\\?\c:\windows/foo.txt`, `\\?\c:\windows/foo.txt`},
	} {
		if got := os.FixLongPath(test.in); got != test.want {
			t.Errorf("fixLongPath(%q) = %q; want %q", test.in, got, test.want)
		}
	}
}
