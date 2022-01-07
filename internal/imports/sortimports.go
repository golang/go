// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Hacked up copy of go/ast/import.go

package imports

import (
	"go/ast"
	"go/token"
	"log"
	"sort"
	"strconv"
)

// sortImports sorts runs of consecutive import lines in import blocks in f.
// It also removes duplicate imports when it is possible to do so without data loss.
func sortImports(localPrefix string, fset *token.FileSet, f *ast.File) {
	for i, d := range f.Decls {
		d, ok := d.(*ast.GenDecl)
		if !ok || d.Tok != token.IMPORT {
			// Not an import declaration, so we're done.
			// Imports are always first.
			break
		}

		if len(d.Specs) == 0 {
			// Empty import block, remove it.
			f.Decls = append(f.Decls[:i], f.Decls[i+1:]...)
		}

		if !d.Lparen.IsValid() {
			// Not a block: sorted by default.
			continue
		}

		// Identify and sort runs of specs on successive lines.
		i := 0
		specs := d.Specs[:0]
		for j, s := range d.Specs {
			if j > i && fset.Position(s.Pos()).Line > 1+fset.Position(d.Specs[j-1].End()).Line {
				// j begins a new run.  End this one.
				specs = append(specs, sortSpecs(localPrefix, fset, f, d.Specs[i:j])...)
				i = j
			}
		}
		specs = append(specs, sortSpecs(localPrefix, fset, f, d.Specs[i:])...)
		d.Specs = specs

		// Deduping can leave a blank line before the rparen; clean that up.
		if len(d.Specs) > 0 {
			lastSpec := d.Specs[len(d.Specs)-1]
			lastLine := fset.Position(lastSpec.Pos()).Line
			if rParenLine := fset.Position(d.Rparen).Line; rParenLine > lastLine+1 {
				fset.File(d.Rparen).MergeLine(rParenLine - 1)
			}
		}
	}
}

// mergeImports merges all the import declarations into the first one.
// Taken from golang.org/x/tools/ast/astutil.
// This does not adjust line numbers properly
func mergeImports(fset *token.FileSet, f *ast.File) {
	if len(f.Decls) <= 1 {
		return
	}

	// Merge all the import declarations into the first one.
	var first *ast.GenDecl
	for i := 0; i < len(f.Decls); i++ {
		decl := f.Decls[i]
		gen, ok := decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.IMPORT || declImports(gen, "C") {
			continue
		}
		if first == nil {
			first = gen
			continue // Don't touch the first one.
		}
		// We now know there is more than one package in this import
		// declaration. Ensure that it ends up parenthesized.
		first.Lparen = first.Pos()
		// Move the imports of the other import declaration to the first one.
		for _, spec := range gen.Specs {
			spec.(*ast.ImportSpec).Path.ValuePos = first.Pos()
			first.Specs = append(first.Specs, spec)
		}
		f.Decls = append(f.Decls[:i], f.Decls[i+1:]...)
		i--
	}
}

// declImports reports whether gen contains an import of path.
// Taken from golang.org/x/tools/ast/astutil.
func declImports(gen *ast.GenDecl, path string) bool {
	if gen.Tok != token.IMPORT {
		return false
	}
	for _, spec := range gen.Specs {
		impspec := spec.(*ast.ImportSpec)
		if importPath(impspec) == path {
			return true
		}
	}
	return false
}

func importPath(s ast.Spec) string {
	t, err := strconv.Unquote(s.(*ast.ImportSpec).Path.Value)
	if err == nil {
		return t
	}
	return ""
}

func importName(s ast.Spec) string {
	n := s.(*ast.ImportSpec).Name
	if n == nil {
		return ""
	}
	return n.Name
}

func importComment(s ast.Spec) string {
	c := s.(*ast.ImportSpec).Comment
	if c == nil {
		return ""
	}
	return c.Text()
}

// collapse indicates whether prev may be removed, leaving only next.
func collapse(prev, next ast.Spec) bool {
	if importPath(next) != importPath(prev) || importName(next) != importName(prev) {
		return false
	}
	return prev.(*ast.ImportSpec).Comment == nil
}

type posSpan struct {
	Start token.Pos
	End   token.Pos
}

func sortSpecs(localPrefix string, fset *token.FileSet, f *ast.File, specs []ast.Spec) []ast.Spec {
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
	importComment := map[*ast.ImportSpec][]*ast.CommentGroup{}
	specIndex := 0
	for _, g := range comments {
		for specIndex+1 < len(specs) && pos[specIndex+1].Start <= g.Pos() {
			specIndex++
		}
		s := specs[specIndex].(*ast.ImportSpec)
		importComment[s] = append(importComment[s], g)
	}

	// Sort the import specs by import path.
	// Remove duplicates, when possible without data loss.
	// Reassign the import paths to have the same position sequence.
	// Reassign each comment to abut the end of its spec.
	// Sort the comments by new position.
	sort.Sort(byImportSpec{localPrefix, specs})

	// Dedup. Thanks to our sorting, we can just consider
	// adjacent pairs of imports.
	deduped := specs[:0]
	for i, s := range specs {
		if i == len(specs)-1 || !collapse(s, specs[i+1]) {
			deduped = append(deduped, s)
		} else {
			p := s.Pos()
			fset.File(p).MergeLine(fset.Position(p).Line)
		}
	}
	specs = deduped

	// Fix up comment positions
	for i, s := range specs {
		s := s.(*ast.ImportSpec)
		if s.Name != nil {
			s.Name.NamePos = pos[i].Start
		}
		s.Path.ValuePos = pos[i].Start
		s.EndPos = pos[i].End
		nextSpecPos := pos[i].End

		for _, g := range importComment[s] {
			for _, c := range g.List {
				c.Slash = pos[i].End
				nextSpecPos = c.End()
			}
		}
		if i < len(specs)-1 {
			pos[i+1].Start = nextSpecPos
			pos[i+1].End = nextSpecPos
		}
	}

	sort.Sort(byCommentPos(comments))

	// Fixup comments can insert blank lines, because import specs are on different lines.
	// We remove those blank lines here by merging import spec to the first import spec line.
	firstSpecLine := fset.Position(specs[0].Pos()).Line
	for _, s := range specs[1:] {
		p := s.Pos()
		line := fset.File(p).Line(p)
		for previousLine := line - 1; previousLine >= firstSpecLine; {
			// MergeLine can panic. Avoid the panic at the cost of not removing the blank line
			// golang/go#50329
			if previousLine > 0 && previousLine < fset.File(p).LineCount() {
				fset.File(p).MergeLine(previousLine)
				previousLine--
			} else {
				// try to gather some data to diagnose how this could happen
				req := "Please report what the imports section of your go file looked like."
				log.Printf("panic avoided: first:%d line:%d previous:%d max:%d. %s",
					firstSpecLine, line, previousLine, fset.File(p).LineCount(), req)
			}
		}
	}
	return specs
}

type byImportSpec struct {
	localPrefix string
	specs       []ast.Spec // slice of *ast.ImportSpec
}

func (x byImportSpec) Len() int      { return len(x.specs) }
func (x byImportSpec) Swap(i, j int) { x.specs[i], x.specs[j] = x.specs[j], x.specs[i] }
func (x byImportSpec) Less(i, j int) bool {
	ipath := importPath(x.specs[i])
	jpath := importPath(x.specs[j])

	igroup := importGroup(x.localPrefix, ipath)
	jgroup := importGroup(x.localPrefix, jpath)
	if igroup != jgroup {
		return igroup < jgroup
	}

	if ipath != jpath {
		return ipath < jpath
	}
	iname := importName(x.specs[i])
	jname := importName(x.specs[j])

	if iname != jname {
		return iname < jname
	}
	return importComment(x.specs[i]) < importComment(x.specs[j])
}

type byCommentPos []*ast.CommentGroup

func (x byCommentPos) Len() int           { return len(x) }
func (x byCommentPos) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x byCommentPos) Less(i, j int) bool { return x[i].Pos() < x[j].Pos() }
