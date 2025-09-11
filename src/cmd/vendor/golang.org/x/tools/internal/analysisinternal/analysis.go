// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package analysisinternal provides gopls' internal analyses with a
// number of helper functions that operate on typed syntax trees.
package analysisinternal

import (
	"bytes"
	"cmp"
	"fmt"
	"go/ast"
	"go/printer"
	"go/scanner"
	"go/token"
	"go/types"
	"iter"
	pathpkg "path"
	"slices"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/moreiters"
	"golang.org/x/tools/internal/typesinternal"
)

// Deprecated: this heuristic is ill-defined.
// TODO(adonovan): move to sole use in gopls/internal/cache.
func TypeErrorEndPos(fset *token.FileSet, src []byte, start token.Pos) token.Pos {
	// Get the end position for the type error.
	file := fset.File(start)
	if file == nil {
		return start
	}
	if offset := file.PositionFor(start, false).Offset; offset > len(src) {
		return start
	} else {
		src = src[offset:]
	}

	// Attempt to find a reasonable end position for the type error.
	//
	// TODO(rfindley): the heuristic implemented here is unclear. It looks like
	// it seeks the end of the primary operand starting at start, but that is not
	// quite implemented (for example, given a func literal this heuristic will
	// return the range of the func keyword).
	//
	// We should formalize this heuristic, or deprecate it by finally proposing
	// to add end position to all type checker errors.
	//
	// Nevertheless, ensure that the end position at least spans the current
	// token at the cursor (this was golang/go#69505).
	end := start
	{
		var s scanner.Scanner
		fset := token.NewFileSet()
		f := fset.AddFile("", fset.Base(), len(src))
		s.Init(f, src, nil /* no error handler */, scanner.ScanComments)
		pos, tok, lit := s.Scan()
		if tok != token.SEMICOLON && token.Pos(f.Base()) <= pos && pos <= token.Pos(f.Base()+f.Size()) {
			off := file.Offset(pos) + len(lit)
			src = src[off:]
			end += token.Pos(off)
		}
	}

	// Look for bytes that might terminate the current operand. See note above:
	// this is imprecise.
	if width := bytes.IndexAny(src, " \n,():;[]+-*/"); width > 0 {
		end += token.Pos(width)
	}
	return end
}

// WalkASTWithParent walks the AST rooted at n. The semantics are
// similar to ast.Inspect except it does not call f(nil).
func WalkASTWithParent(n ast.Node, f func(n ast.Node, parent ast.Node) bool) {
	var ancestors []ast.Node
	ast.Inspect(n, func(n ast.Node) (recurse bool) {
		if n == nil {
			ancestors = ancestors[:len(ancestors)-1]
			return false
		}

		var parent ast.Node
		if len(ancestors) > 0 {
			parent = ancestors[len(ancestors)-1]
		}
		ancestors = append(ancestors, n)
		return f(n, parent)
	})
}

// MatchingIdents finds the names of all identifiers in 'node' that match any of the given types.
// 'pos' represents the position at which the identifiers may be inserted. 'pos' must be within
// the scope of each of identifier we select. Otherwise, we will insert a variable at 'pos' that
// is unrecognized.
func MatchingIdents(typs []types.Type, node ast.Node, pos token.Pos, info *types.Info, pkg *types.Package) map[types.Type][]string {

	// Initialize matches to contain the variable types we are searching for.
	matches := make(map[types.Type][]string)
	for _, typ := range typs {
		if typ == nil {
			continue // TODO(adonovan): is this reachable?
		}
		matches[typ] = nil // create entry
	}

	seen := map[types.Object]struct{}{}
	ast.Inspect(node, func(n ast.Node) bool {
		if n == nil {
			return false
		}
		// Prevent circular definitions. If 'pos' is within an assignment statement, do not
		// allow any identifiers in that assignment statement to be selected. Otherwise,
		// we could do the following, where 'x' satisfies the type of 'f0':
		//
		// x := fakeStruct{f0: x}
		//
		if assign, ok := n.(*ast.AssignStmt); ok && pos > assign.Pos() && pos <= assign.End() {
			return false
		}
		if n.End() > pos {
			return n.Pos() <= pos
		}
		ident, ok := n.(*ast.Ident)
		if !ok || ident.Name == "_" {
			return true
		}
		obj := info.Defs[ident]
		if obj == nil || obj.Type() == nil {
			return true
		}
		if _, ok := obj.(*types.TypeName); ok {
			return true
		}
		// Prevent duplicates in matches' values.
		if _, ok = seen[obj]; ok {
			return true
		}
		seen[obj] = struct{}{}
		// Find the scope for the given position. Then, check whether the object
		// exists within the scope.
		innerScope := pkg.Scope().Innermost(pos)
		if innerScope == nil {
			return true
		}
		_, foundObj := innerScope.LookupParent(ident.Name, pos)
		if foundObj != obj {
			return true
		}
		// The object must match one of the types that we are searching for.
		// TODO(adonovan): opt: use typeutil.Map?
		if names, ok := matches[obj.Type()]; ok {
			matches[obj.Type()] = append(names, ident.Name)
		} else {
			// If the object type does not exactly match
			// any of the target types, greedily find the first
			// target type that the object type can satisfy.
			for typ := range matches {
				if equivalentTypes(obj.Type(), typ) {
					matches[typ] = append(matches[typ], ident.Name)
				}
			}
		}
		return true
	})
	return matches
}

func equivalentTypes(want, got types.Type) bool {
	if types.Identical(want, got) {
		return true
	}
	// Code segment to help check for untyped equality from (golang/go#32146).
	if rhs, ok := want.(*types.Basic); ok && rhs.Info()&types.IsUntyped > 0 {
		if lhs, ok := got.Underlying().(*types.Basic); ok {
			return rhs.Info()&types.IsConstType == lhs.Info()&types.IsConstType
		}
	}
	return types.AssignableTo(want, got)
}

// A ReadFileFunc is a function that returns the
// contents of a file, such as [os.ReadFile].
type ReadFileFunc = func(filename string) ([]byte, error)

// CheckedReadFile returns a wrapper around a Pass.ReadFile
// function that performs the appropriate checks.
func CheckedReadFile(pass *analysis.Pass, readFile ReadFileFunc) ReadFileFunc {
	return func(filename string) ([]byte, error) {
		if err := CheckReadable(pass, filename); err != nil {
			return nil, err
		}
		return readFile(filename)
	}
}

// CheckReadable enforces the access policy defined by the ReadFile field of [analysis.Pass].
func CheckReadable(pass *analysis.Pass, filename string) error {
	if slices.Contains(pass.OtherFiles, filename) ||
		slices.Contains(pass.IgnoredFiles, filename) {
		return nil
	}
	for _, f := range pass.Files {
		if pass.Fset.File(f.FileStart).Name() == filename {
			return nil
		}
	}
	return fmt.Errorf("Pass.ReadFile: %s is not among OtherFiles, IgnoredFiles, or names of Files", filename)
}

// AddImport checks whether this file already imports pkgpath and
// that import is in scope at pos. If so, it returns the name under
// which it was imported and a zero edit. Otherwise, it adds a new
// import of pkgpath, using a name derived from the preferred name,
// and returns the chosen name, a prefix to be concatenated with member
// to form a qualified name, and the edit for the new import.
//
// In the special case that pkgpath is dot-imported then member, the
// identifier for which the import is being added, is consulted. If
// member is not shadowed at pos, AddImport returns (".", "", nil).
// (AddImport accepts the caller's implicit claim that the imported
// package declares member.)
//
// It does not mutate its arguments.
func AddImport(info *types.Info, file *ast.File, preferredName, pkgpath, member string, pos token.Pos) (name, prefix string, newImport []analysis.TextEdit) {
	// Find innermost enclosing lexical block.
	scope := info.Scopes[file].Innermost(pos)
	if scope == nil {
		panic("no enclosing lexical block")
	}

	// Is there an existing import of this package?
	// If so, are we in its scope? (not shadowed)
	for _, spec := range file.Imports {
		pkgname := info.PkgNameOf(spec)
		if pkgname != nil && pkgname.Imported().Path() == pkgpath {
			name = pkgname.Name()
			if name == "." {
				// The scope of ident must be the file scope.
				if s, _ := scope.LookupParent(member, pos); s == info.Scopes[file] {
					return name, "", nil
				}
			} else if _, obj := scope.LookupParent(name, pos); obj == pkgname {
				return name, name + ".", nil
			}
		}
	}

	// We must add a new import.
	// Ensure we have a fresh name.
	newName := FreshName(scope, pos, preferredName)

	// Create a new import declaration either before the first existing
	// declaration (which must exist), including its comments; or
	// inside the declaration, if it is an import group.
	//
	// Use a renaming import whenever the preferred name is not
	// available, or the chosen name does not match the last
	// segment of its path.
	newText := fmt.Sprintf("%q", pkgpath)
	if newName != preferredName || newName != pathpkg.Base(pkgpath) {
		newText = fmt.Sprintf("%s %q", newName, pkgpath)
	}
	decl0 := file.Decls[0]
	var before ast.Node = decl0
	switch decl0 := decl0.(type) {
	case *ast.GenDecl:
		if decl0.Doc != nil {
			before = decl0.Doc
		}
	case *ast.FuncDecl:
		if decl0.Doc != nil {
			before = decl0.Doc
		}
	}
	if gd, ok := before.(*ast.GenDecl); ok && gd.Tok == token.IMPORT && gd.Rparen.IsValid() {
		// Have existing grouped import ( ... ) decl.
		if IsStdPackage(pkgpath) && len(gd.Specs) > 0 {
			// Add spec for a std package before
			// first existing spec, followed by
			// a blank line if the next one is non-std.
			first := gd.Specs[0].(*ast.ImportSpec)
			pos = first.Pos()
			if !IsStdPackage(first.Path.Value) {
				newText += "\n"
			}
			newText += "\n\t"
		} else {
			// Add spec at end of group.
			pos = gd.Rparen
			newText = "\t" + newText + "\n"
		}
	} else {
		// No import decl, or non-grouped import.
		// Add a new import decl before first decl.
		// (gofmt will merge multiple import decls.)
		pos = before.Pos()
		newText = "import " + newText + "\n\n"
	}
	return newName, newName + ".", []analysis.TextEdit{{
		Pos:     pos,
		End:     pos,
		NewText: []byte(newText),
	}}
}

// FreshName returns the name of an identifier that is undefined
// at the specified position, based on the preferred name.
func FreshName(scope *types.Scope, pos token.Pos, preferred string) string {
	newName := preferred
	for i := 0; ; i++ {
		if _, obj := scope.LookupParent(newName, pos); obj == nil {
			break // fresh
		}
		newName = fmt.Sprintf("%s%d", preferred, i)
	}
	return newName
}

// Format returns a string representation of the node n.
func Format(fset *token.FileSet, n ast.Node) string {
	var buf strings.Builder
	printer.Fprint(&buf, fset, n) // ignore errors
	return buf.String()
}

// Imports returns true if path is imported by pkg.
func Imports(pkg *types.Package, path string) bool {
	for _, imp := range pkg.Imports() {
		if imp.Path() == path {
			return true
		}
	}
	return false
}

// IsTypeNamed reports whether t is (or is an alias for) a
// package-level defined type with the given package path and one of
// the given names. It returns false if t is nil.
//
// This function avoids allocating the concatenation of "pkg.Name",
// which is important for the performance of syntax matching.
func IsTypeNamed(t types.Type, pkgPath string, names ...string) bool {
	if named, ok := types.Unalias(t).(*types.Named); ok {
		tname := named.Obj()
		return tname != nil &&
			typesinternal.IsPackageLevel(tname) &&
			tname.Pkg().Path() == pkgPath &&
			slices.Contains(names, tname.Name())
	}
	return false
}

// IsPointerToNamed reports whether t is (or is an alias for) a pointer to a
// package-level defined type with the given package path and one of the given
// names. It returns false if t is not a pointer type.
func IsPointerToNamed(t types.Type, pkgPath string, names ...string) bool {
	r := typesinternal.Unpointer(t)
	if r == t {
		return false
	}
	return IsTypeNamed(r, pkgPath, names...)
}

// IsFunctionNamed reports whether obj is a package-level function
// defined in the given package and has one of the given names.
// It returns false if obj is nil.
//
// This function avoids allocating the concatenation of "pkg.Name",
// which is important for the performance of syntax matching.
func IsFunctionNamed(obj types.Object, pkgPath string, names ...string) bool {
	f, ok := obj.(*types.Func)
	return ok &&
		typesinternal.IsPackageLevel(obj) &&
		f.Pkg().Path() == pkgPath &&
		f.Type().(*types.Signature).Recv() == nil &&
		slices.Contains(names, f.Name())
}

// IsMethodNamed reports whether obj is a method defined on a
// package-level type with the given package and type name, and has
// one of the given names. It returns false if obj is nil.
//
// This function avoids allocating the concatenation of "pkg.TypeName.Name",
// which is important for the performance of syntax matching.
func IsMethodNamed(obj types.Object, pkgPath string, typeName string, names ...string) bool {
	if fn, ok := obj.(*types.Func); ok {
		if recv := fn.Type().(*types.Signature).Recv(); recv != nil {
			_, T := typesinternal.ReceiverNamed(recv)
			return T != nil &&
				IsTypeNamed(T, pkgPath, typeName) &&
				slices.Contains(names, fn.Name())
		}
	}
	return false
}

// ValidateFixes validates the set of fixes for a single diagnostic.
// Any error indicates a bug in the originating analyzer.
//
// It updates fixes so that fixes[*].End.IsValid().
//
// It may be used as part of an analysis driver implementation.
func ValidateFixes(fset *token.FileSet, a *analysis.Analyzer, fixes []analysis.SuggestedFix) error {
	fixMessages := make(map[string]bool)
	for i := range fixes {
		fix := &fixes[i]
		if fixMessages[fix.Message] {
			return fmt.Errorf("analyzer %q suggests two fixes with same Message (%s)", a.Name, fix.Message)
		}
		fixMessages[fix.Message] = true
		if err := validateFix(fset, fix); err != nil {
			return fmt.Errorf("analyzer %q suggests invalid fix (%s): %v", a.Name, fix.Message, err)
		}
	}
	return nil
}

// validateFix validates a single fix.
// Any error indicates a bug in the originating analyzer.
//
// It updates fix so that fix.End.IsValid().
func validateFix(fset *token.FileSet, fix *analysis.SuggestedFix) error {

	// Stably sort edits by Pos. This ordering puts insertions
	// (end = start) before deletions (end > start) at the same
	// point, but uses a stable sort to preserve the order of
	// multiple insertions at the same point.
	slices.SortStableFunc(fix.TextEdits, func(x, y analysis.TextEdit) int {
		if sign := cmp.Compare(x.Pos, y.Pos); sign != 0 {
			return sign
		}
		return cmp.Compare(x.End, y.End)
	})

	var prev *analysis.TextEdit
	for i := range fix.TextEdits {
		edit := &fix.TextEdits[i]

		// Validate edit individually.
		start := edit.Pos
		file := fset.File(start)
		if file == nil {
			return fmt.Errorf("no token.File for TextEdit.Pos (%v)", edit.Pos)
		}
		fileEnd := token.Pos(file.Base() + file.Size())
		if end := edit.End; end.IsValid() {
			if end < start {
				return fmt.Errorf("TextEdit.Pos (%v) > TextEdit.End (%v)", edit.Pos, edit.End)
			}
			endFile := fset.File(end)
			if endFile != file && end < fileEnd+10 {
				// Relax the checks below in the special case when the end position
				// is only slightly beyond EOF, as happens when End is computed
				// (as in ast.{Struct,Interface}Type) rather than based on
				// actual token positions. In such cases, truncate end to EOF.
				//
				// This is a workaround for #71659; see:
				// https://github.com/golang/go/issues/71659#issuecomment-2651606031
				// A better fix would be more faithful recording of token
				// positions (or their absence) in the AST.
				edit.End = fileEnd
				continue
			}
			if endFile == nil {
				return fmt.Errorf("no token.File for TextEdit.End (%v; File(start).FileEnd is %d)", end, file.Base()+file.Size())
			}
			if endFile != file {
				return fmt.Errorf("edit #%d spans files (%v and %v)",
					i, file.Position(edit.Pos), endFile.Position(edit.End))
			}
		} else {
			edit.End = start // update the SuggestedFix
		}
		if eof := fileEnd; edit.End > eof {
			return fmt.Errorf("end is (%v) beyond end of file (%v)", edit.End, eof)
		}

		// Validate the sequence of edits:
		// properly ordered, no overlapping deletions
		if prev != nil && edit.Pos < prev.End {
			xpos := fset.Position(prev.Pos)
			xend := fset.Position(prev.End)
			ypos := fset.Position(edit.Pos)
			yend := fset.Position(edit.End)
			return fmt.Errorf("overlapping edits to %s (%d:%d-%d:%d and %d:%d-%d:%d)",
				xpos.Filename,
				xpos.Line, xpos.Column,
				xend.Line, xend.Column,
				ypos.Line, ypos.Column,
				yend.Line, yend.Column,
			)
		}
		prev = edit
	}

	return nil
}

// CanImport reports whether one package is allowed to import another.
//
// TODO(adonovan): allow customization of the accessibility relation
// (e.g. for Bazel).
func CanImport(from, to string) bool {
	// TODO(adonovan): better segment hygiene.
	if to == "internal" || strings.HasPrefix(to, "internal/") {
		// Special case: only std packages may import internal/...
		// We can't reliably know whether we're in std, so we
		// use a heuristic on the first segment.
		first, _, _ := strings.Cut(from, "/")
		if strings.Contains(first, ".") {
			return false // example.com/foo ∉ std
		}
		if first == "testdata" {
			return false // testdata/foo ∉ std
		}
	}
	if strings.HasSuffix(to, "/internal") {
		return strings.HasPrefix(from, to[:len(to)-len("/internal")])
	}
	if i := strings.LastIndex(to, "/internal/"); i >= 0 {
		return strings.HasPrefix(from, to[:i])
	}
	return true
}

// DeleteStmt returns the edits to remove the [ast.Stmt] identified by
// curStmt, if it is contained within a BlockStmt, CaseClause,
// CommClause, or is the STMT in switch STMT; ... {...}. It returns nil otherwise.
func DeleteStmt(fset *token.FileSet, curStmt inspector.Cursor) []analysis.TextEdit {
	stmt := curStmt.Node().(ast.Stmt)
	// if the stmt is on a line by itself delete the whole line
	// otherwise just delete the statement.

	// this logic would be a lot simpler with the file contents, and somewhat simpler
	// if the cursors included the comments.

	tokFile := fset.File(stmt.Pos())
	lineOf := tokFile.Line
	stmtStartLine, stmtEndLine := lineOf(stmt.Pos()), lineOf(stmt.End())

	var from, to token.Pos
	// bounds of adjacent syntax/comments on same line, if any
	limits := func(left, right token.Pos) {
		if lineOf(left) == stmtStartLine {
			from = left
		}
		if lineOf(right) == stmtEndLine {
			to = right
		}
	}
	// TODO(pjw): there are other places a statement might be removed:
	// IfStmt = "if" [ SimpleStmt ";" ] Expression Block [ "else" ( IfStmt | Block ) ] .
	// (removing the blocks requires more rewriting than this routine would do)
	// CommCase   = "case" ( SendStmt | RecvStmt ) | "default" .
	// (removing the stmt requires more rewriting, and it's unclear what the user means)
	switch parent := curStmt.Parent().Node().(type) {
	case *ast.SwitchStmt:
		limits(parent.Switch, parent.Body.Lbrace)
	case *ast.TypeSwitchStmt:
		limits(parent.Switch, parent.Body.Lbrace)
		if parent.Assign == stmt {
			return nil // don't let the user break the type switch
		}
	case *ast.BlockStmt:
		limits(parent.Lbrace, parent.Rbrace)
	case *ast.CommClause:
		limits(parent.Colon, curStmt.Parent().Parent().Node().(*ast.BlockStmt).Rbrace)
		if parent.Comm == stmt {
			return nil // maybe the user meant to remove the entire CommClause?
		}
	case *ast.CaseClause:
		limits(parent.Colon, curStmt.Parent().Parent().Node().(*ast.BlockStmt).Rbrace)
	case *ast.ForStmt:
		limits(parent.For, parent.Body.Lbrace)

	default:
		return nil // not one of ours
	}

	if prev, found := curStmt.PrevSibling(); found && lineOf(prev.Node().End()) == stmtStartLine {
		from = prev.Node().End() // preceding statement ends on same line
	}
	if next, found := curStmt.NextSibling(); found && lineOf(next.Node().Pos()) == stmtEndLine {
		to = next.Node().Pos() // following statement begins on same line
	}
	// and now for the comments
Outer:
	for _, cg := range enclosingFile(curStmt).Comments {
		for _, co := range cg.List {
			if lineOf(co.End()) < stmtStartLine {
				continue
			} else if lineOf(co.Pos()) > stmtEndLine {
				break Outer // no more are possible
			}
			if lineOf(co.End()) == stmtStartLine && co.End() < stmt.Pos() {
				if !from.IsValid() || co.End() > from {
					from = co.End()
					continue // maybe there are more
				}
			}
			if lineOf(co.Pos()) == stmtEndLine && co.Pos() > stmt.End() {
				if !to.IsValid() || co.Pos() < to {
					to = co.Pos()
					continue // maybe there are more
				}
			}
		}
	}
	// if either from or to is valid, just remove the statement
	// otherwise remove the line
	edit := analysis.TextEdit{Pos: stmt.Pos(), End: stmt.End()}
	if from.IsValid() || to.IsValid() {
		// remove just the statement.
		// we can't tell if there is a ; or whitespace right after the statement
		// ideally we'd like to remove the former and leave the latter
		// (if gofmt has run, there likely won't be a ;)
		// In type switches we know there's a semicolon somewhere after the statement,
		// but the extra work for this special case is not worth it, as gofmt will fix it.
		return []analysis.TextEdit{edit}
	}
	// remove the whole line
	for lineOf(edit.Pos) == stmtStartLine {
		edit.Pos--
	}
	edit.Pos++ // get back tostmtStartLine
	for lineOf(edit.End) == stmtEndLine {
		edit.End++
	}
	return []analysis.TextEdit{edit}
}

// Comments returns an iterator over the comments overlapping the specified interval.
func Comments(file *ast.File, start, end token.Pos) iter.Seq[*ast.Comment] {
	// TODO(adonovan): optimize use binary O(log n) instead of linear O(n) search.
	return func(yield func(*ast.Comment) bool) {
		for _, cg := range file.Comments {
			for _, co := range cg.List {
				if co.Pos() > end {
					return
				}
				if co.End() < start {
					continue
				}

				if !yield(co) {
					return
				}
			}
		}
	}
}

// IsStdPackage reports whether the specified package path belongs to a
// package in the standard library (including internal dependencies).
func IsStdPackage(path string) bool {
	// A standard package has no dot in its first segment.
	// (It may yet have a dot, e.g. "vendor/golang.org/x/foo".)
	slash := strings.IndexByte(path, '/')
	if slash < 0 {
		slash = len(path)
	}
	return !strings.Contains(path[:slash], ".") && path != "testdata"
}

// Range returns an [analysis.Range] for the specified start and end positions.
func Range(pos, end token.Pos) analysis.Range {
	return tokenRange{pos, end}
}

// tokenRange is an implementation of the [analysis.Range] interface.
type tokenRange struct{ StartPos, EndPos token.Pos }

func (r tokenRange) Pos() token.Pos { return r.StartPos }
func (r tokenRange) End() token.Pos { return r.EndPos }

// enclosingFile returns the syntax tree for the file enclosing c.
func enclosingFile(c inspector.Cursor) *ast.File {
	c, _ = moreiters.First(c.Enclosing((*ast.File)(nil)))
	return c.Node().(*ast.File)
}
