// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"regexp"
)

func init() {
	register(googlecodeFix)
}

var googlecodeFix = fix{
	"googlecode",
	"2011-11-21",
	googlecode,
	`Rewrite Google Code imports from the deprecated form
"foo.googlecode.com/vcs/path" to "code.google.com/p/foo/path".
`,
}

var googlecodeRe = regexp.MustCompile(`^([a-z0-9\-]+)\.googlecode\.com/(svn|git|hg)(/[a-z0-9A-Z_.\-/]+)?$`)

func googlecode(f *ast.File) bool {
	fixed := false

	for _, s := range f.Imports {
		old := importPath(s)
		if m := googlecodeRe.FindStringSubmatch(old); m != nil {
			new := "code.google.com/p/" + m[1] + m[3]
			if rewriteImport(f, old, new) {
				fixed = true
			}
		}
	}

	return fixed
}
