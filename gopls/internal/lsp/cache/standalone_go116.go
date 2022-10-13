// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.16
// +build go1.16

package cache

import (
	"go/build/constraint"
	"go/parser"
	"go/token"
)

// isStandaloneFile reports whether a file with the given contents should be
// considered a 'standalone main file', meaning a package that consists of only
// a single file.
func isStandaloneFile(src []byte, standaloneTags []string) bool {
	f, err := parser.ParseFile(token.NewFileSet(), "", src, parser.PackageClauseOnly|parser.ParseComments)
	if err != nil {
		return false
	}

	if f.Name == nil || f.Name.Name != "main" {
		return false
	}

	for _, cg := range f.Comments {
		// Even with PackageClauseOnly the parser consumes the semicolon following
		// the package clause, so we must guard against comments that come after
		// the package name.
		if cg.Pos() > f.Name.Pos() {
			continue
		}
		for _, comment := range cg.List {
			if c, err := constraint.Parse(comment.Text); err == nil {
				if tag, ok := c.(*constraint.TagExpr); ok {
					for _, t := range standaloneTags {
						if t == tag.Tag {
							return true
						}
					}
				}
			}
		}
	}

	return false
}
