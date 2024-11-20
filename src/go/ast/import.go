// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast

import (
	"cmp"
	"go/token"
	"slices"
	"strconv"
)

// SortImports sorts runs of consecutive import lines in import blocks in f.
// It also removes duplicate imports when it is possible to do so without data loss.
func SortImports(fset *token.FileSet, f *File) {
	for _, d := range f.Decls {
		d, ok := d.(*GenDecl)
		if !ok || d.Tok != token.IMPORT {
			// Not an import declaration, so we're done.
			// Imports are always first.
			break
		}

		if !d.Lparen.IsValid() {
			// Not a block: sorted by default.
			continue
		}

		// Identify and sort runs of specs on successive lines.
		i := 0
		specs := d.Specs[:0]
		for j, s := range d.Specs {
			if j > i && lineAt(fset, s.Pos()) > 1+lineAt(fset, d.Specs[j-1].End()) {
				// j begins a new run. End this one.
				specs = append(specs, sortSpecs(fset, f, d.Specs[i:j])...)
				i = j
			}
		}
		specs = append(specs, sortSpecs(fset, f, d.Specs[i:])...)
		d.Specs = specs

		// Deduping can leave a blank line before the rparen; clean that up.
		if len(d.Specs) > 0 {
			lastSpec := d.Specs[len(d.Specs)-1]
			lastLine := lineAt(fset, lastSpec.Pos())
			rParenLine := lineAt(fset, d.Rparen)
			for rParenLine > lastLine+1 {
				rParenLine--
				fset.File(d.Rparen).MergeLine(rParenLine)
			}
		}
	}

	// Make File.Imports order consistent.
	f.Imports = f.Imports[:0]
	for _, decl := range f.Decls {
		if decl, ok := decl.(*GenDecl); ok && decl.Tok == token.IMPORT {
			for _, spec := range decl.Specs {
				f.Imports = append(f.Imports, spec.(*ImportSpec))
			}
		}
	}
}

func lineAt(fset *token.FileSet, pos token.Pos) int {
	return fset.PositionFor(pos, false).Line
}

func importPath(s Spec) string {
	t, err := strconv.Unquote(s.(*ImportSpec).Path.Value)
	if err == nil {
		return t
	}
	return ""
}

func importName(s Spec) string {
	n := s.(*ImportSpec).Name
	if n == nil {
		return ""
	}
	return n.Name
}

func importComment(s Spec) string {
	c := s.(*ImportSpec).Comment
	if c == nil {
		return ""
	}
	return c.Text()
}

// collapse indicates whether prev may be removed, leaving only next.
func collapse(prev, next Spec) bool {
	if importPath(next) != importPath(prev) || importName(next) != importName(prev) {
		return false
	}
	return prev.(*ImportSpec).Comment == nil
}

type posSpan struct {
	Start token.Pos
	End   token.Pos
}

type cgPos struct {
	left bool // true if comment is to the left of the spec, false otherwise.
	cg   *CommentGroup
}

func sortSpecs(fset *token.FileSet, f *File, specs []Spec) []Spec {
	// Can't short-circuit here even if specs are already sorted,
	// since they might yet need deduplication.
	// A lone import, however, may be safely ignored.
	if len(specs) <= 1 {
		return specs
	}

	// Record positions for specs.
	pos := make([]posSpan, len(specs))
	for i, s := range specs {
		pos[i] = posSpan{s.Pos(), s.End()}
	}

	// Identify comments in this range.
	begSpecs := pos[0].Start
	endSpecs := pos[len(pos)-1].End
	beg := fset.File(begSpecs).LineStart(lineAt(fset, begSpecs))
	endLine := lineAt(fset, endSpecs)
	endFile := fset.File(endSpecs)
	var end token.Pos
	if endLine == endFile.LineCount() {
		end = endSpecs
	} else {
		end = endFile.LineStart(endLine + 1) // beginning of next line
	}
	first := len(f.Comments)
	last := -1
	for i, g := range f.Comments {
		if g.End() >= end {
			break
		}
		// g.End() < end
		if beg <= g.Pos() {
			// comment is within the range [beg, end[ of import declarations
			if i < first {
				first = i
			}
			if i > last {
				last = i
			}
		}
	}

	var comments []*CommentGroup
	if last >= 0 {
		comments = f.Comments[first : last+1]
	}

	// Assign each comment to the import spec on the same line.
	importComments := map[*ImportSpec][]cgPos{}
	specIndex := 0
	for _, g := range comments {
		for specIndex+1 < len(specs) && pos[specIndex+1].Start <= g.Pos() {
			specIndex++
		}
		var left bool
		// A block comment can appear before the first import spec.
		if specIndex == 0 && pos[specIndex].Start > g.Pos() {
			left = true
		} else if specIndex+1 < len(specs) && // Or it can appear on the left of an import spec.
			lineAt(fset, pos[specIndex].Start)+1 == lineAt(fset, g.Pos()) {
			specIndex++
			left = true
		}
		s := specs[specIndex].(*ImportSpec)
		importComments[s] = append(importComments[s], cgPos{left: left, cg: g})
	}

	// Sort the import specs by import path.
	// Remove duplicates, when possible without data loss.
	// Reassign the import paths to have the same position sequence.
	// Reassign each comment to the spec on the same line.
	// Sort the comments by new position.
	slices.SortFunc(specs, func(a, b Spec) int {
		ipath := importPath(a)
		jpath := importPath(b)
		r := cmp.Compare(ipath, jpath)
		if r != 0 {
			return r
		}
		iname := importName(a)
		jname := importName(b)
		r = cmp.Compare(iname, jname)
		if r != 0 {
			return r
		}
		return cmp.Compare(importComment(a), importComment(b))
	})

	// Dedup. Thanks to our sorting, we can just consider
	// adjacent pairs of imports.
	deduped := specs[:0]
	for i, s := range specs {
		if i == len(specs)-1 || !collapse(s, specs[i+1]) {
			deduped = append(deduped, s)
		} else {
			p := s.Pos()
			fset.File(p).MergeLine(lineAt(fset, p))
		}
	}
	specs = deduped

	// Fix up comment positions
	for i, s := range specs {
		s := s.(*ImportSpec)
		if s.Name != nil {
			s.Name.NamePos = pos[i].Start
		}
		s.Path.ValuePos = pos[i].Start
		s.EndPos = pos[i].End
		for _, g := range importComments[s] {
			for _, c := range g.cg.List {
				if g.left {
					c.Slash = pos[i].Start - 1
				} else {
					// An import spec can have both block comment and a line comment
					// to its right. In that case, both of them will have the same pos.
					// But while formatting the AST, the line comment gets moved to
					// after the block comment.
					c.Slash = pos[i].End
				}
			}
		}
	}

	slices.SortFunc(comments, func(a, b *CommentGroup) int {
		return cmp.Compare(a.Pos(), b.Pos())
	})

	return specs
}
