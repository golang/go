// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast

import (
	"go/token"
	"sort"
	"strconv"
)

// SortImports sorts runs of consecutive import lines in import blocks in f.
func SortImports(fset *token.FileSet, f *File) {
	for _, d := range f.Decls {
		d, ok := d.(*GenDecl)
		if !ok || d.Tok != token.IMPORT {
			// Not an import declaration, so we're done.
			// Imports are always first.
			break
		}

		if d.Lparen == token.NoPos {
			// Not a block: sorted by default.
			continue
		}

		// Identify and sort runs of specs on successive lines.
		i := 0
		for j, s := range d.Specs {
			if j > i && fset.Position(s.Pos()).Line > 1+fset.Position(d.Specs[j-1].End()).Line {
				// j begins a new run.  End this one.
				sortSpecs(fset, f, d.Specs[i:j])
				i = j
			}
		}
		sortSpecs(fset, f, d.Specs[i:])
	}
}

func importPath(s Spec) string {
	t, err := strconv.Unquote(s.(*ImportSpec).Path.Value)
	if err == nil {
		return t
	}
	return ""
}

type posSpan struct {
	Start token.Pos
	End   token.Pos
}

func sortSpecs(fset *token.FileSet, f *File, specs []Spec) {
	// Avoid work if already sorted (also catches < 2 entries).
	sorted := true
	for i, s := range specs {
		if i > 0 && importPath(specs[i-1]) > importPath(s) {
			sorted = false
			break
		}
	}
	if sorted {
		return
	}

	// Record positions for specs.
	pos := make([]posSpan, len(specs))
	for i, s := range specs {
		// Cannot use s.End(), because it looks at len(s.Path.Value),
		// and that string might have gotten longer or shorter.
		// Instead, use s.Pos()+1, which is guaranteed to be > s.Pos()
		// and still before the original end of the string, since any
		// string literal must be at least 2 characters ("" or ``).
		pos[i] = posSpan{s.Pos(), s.Pos() + 1}
	}

	// Identify comments in this range.
	// Any comment from pos[0].Start to the final line counts.
	lastLine := fset.Position(pos[len(pos)-1].End).Line
	cstart := len(f.Comments)
	cend := len(f.Comments)
	for i, g := range f.Comments {
		if g.Pos() < pos[0].Start {
			continue
		}
		if i < cstart {
			cstart = i
		}
		if fset.Position(g.End()).Line > lastLine {
			cend = i
			break
		}
	}
	comments := f.Comments[cstart:cend]

	// Assign each comment to the import spec preceding it.
	importComment := map[*ImportSpec][]*CommentGroup{}
	specIndex := 0
	for _, g := range comments {
		for specIndex+1 < len(specs) && pos[specIndex+1].Start <= g.Pos() {
			specIndex++
		}
		s := specs[specIndex].(*ImportSpec)
		importComment[s] = append(importComment[s], g)
	}

	// Sort the import specs by import path.
	// Reassign the import paths to have the same position sequence.
	// Reassign each comment to abut the end of its spec.
	// Sort the comments by new position.
	sort.Sort(byImportPath(specs))
	for i, s := range specs {
		s := s.(*ImportSpec)
		if s.Name != nil {
			s.Name.NamePos = pos[i].Start
		}
		s.Path.ValuePos = pos[i].Start
		s.EndPos = pos[i].End
		for _, g := range importComment[s] {
			for _, c := range g.List {
				c.Slash = pos[i].End
			}
		}
	}
	sort.Sort(byCommentPos(comments))
}

type byImportPath []Spec // slice of *ImportSpec

func (x byImportPath) Len() int           { return len(x) }
func (x byImportPath) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x byImportPath) Less(i, j int) bool { return importPath(x[i]) < importPath(x[j]) }

type byCommentPos []*CommentGroup

func (x byCommentPos) Len() int           { return len(x) }
func (x byCommentPos) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x byCommentPos) Less(i, j int) bool { return x[i].Pos() < x[j].Pos() }
