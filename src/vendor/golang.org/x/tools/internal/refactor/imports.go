// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package refactor

// This file defines operations for computing edits to imports.

import (
	"go/ast"
	"go/token"
	"go/types"
	pathpkg "path"
	"strconv"

	"golang.org/x/tools/internal/packagepath"
)

// AddImport returns the prefix (either "pkg." or "") that should be
// used to qualify references to the desired symbol (member) imported
// from the specified package, plus any necessary edits to the file's
// import declaration to add a new import.
//
// If poses is non-empty, AddImport ensures that the chosen package prefix
// is accessible and unshadowed at all specified positions.
//
// If the import already exists, and is accessible at all poses, AddImport
// returns the existing name and no edits. (If the existing import is
// a dot import, the prefix is "".)
//
// Otherwise, it adds a new import, using a local name derived from
// the preferred name that is unshadowed at all poses. To request a blank
// import, use a preferredName of "_", and discard the prefix result;
// member is ignored in this case.
//
// AddImport accepts the caller's implicit claim that the imported
// package declares member.
//
// AddImport does not mutate its arguments.
func AddImport(info *types.Info, file *ast.File, preferredName, pkgpath, member string, poses ...token.Pos) (prefix string, edits []Edit) {
	fileScope := info.Scopes[file]
	if fileScope == nil {
		panic("no file scope")
	}

	scopes := make([]*types.Scope, len(poses))
	for i, pos := range poses {
		// fileScope is a half-open interval [FileStart, FileEnd), so
		// fileScope.Contains(FileEnd) is false. If pos is at EOF
		// (e.g. a parser-synthesized Rbrace or insertion at FileEnd),
		// look up the innermost scope at FileEnd-1 (golang/go#82678).
		if pos == fileScope.End() && pos > fileScope.Pos() {
			pos--
		}
		scopes[i] = fileScope.Innermost(pos)
		if scopes[i] == nil {
			panic("no enclosing lexical block")
		}
	}

	// Is there an existing import of this package?
	// If so, are we in its scope at all poses? (not shadowed)
specs:
	for _, spec := range file.Imports {
		pkgname := info.PkgNameOf(spec)
		if pkgname != nil && pkgname.Imported().Path() == pkgpath {
			if preferredName == "_" {
				// Request for blank import; any existing import will do.
				return "", nil
			}
			name := pkgname.Name()
			prefix, ident := name+".", name
			if name == "." {
				prefix, ident = "", member
			}
			// The scope of ident must be the file scope at all poses.
			for i, pos := range poses {
				if s, _ := scopes[i].LookupParent(ident, pos); s != fileScope {
					continue specs
				}
			}
			return prefix, nil
		}
	}

	// We must add a new import.

	// Ensure we have a fresh name unshadowed at all poses and file scope.
	newName := preferredName
	if preferredName != "_" {
		// No position scope check needed, only check file scope.
		if len(poses) == 0 {
			scopes = []*types.Scope{fileScope}
			poses = []token.Pos{token.NoPos}
		}
		newName = freshName(scopes, poses, preferredName)
		prefix = newName + "."
	}

	// Use a renaming import whenever the preferred name is not
	// available, or the chosen name does not match the last
	// segment of its path.
	if newName == preferredName && newName == pathpkg.Base(pkgpath) {
		newName = ""
	}

	return prefix, AddImportEdits(file, newName, pkgpath)
}

// AddImportEdits returns the edits to add an import of the specified
// package, without any analysis of whether this is necessary or safe.
// If name is nonempty, it is used as an explicit [ImportSpec.Name].
//
// A sequence of calls to AddImportEdits that each add the file's
// first import (or in a file that does not have a grouped import) may
// result in multiple import declarations, rather than a single one
// with multiple ImportSpecs. However, a subsequent run of
// x/tools/cmd/goimports ([imports.Process]) will combine them.
//
// AddImportEdits does not mutate the AST.
func AddImportEdits(file *ast.File, name, pkgpath string) []Edit {
	newText := strconv.Quote(pkgpath)
	if name != "" {
		newText = name + " " + newText
	}

	// Create a new import declaration either before the first existing
	// declaration (if it exists), including its comments; or at the end of the
	// file (if there are no decls); or inside the declaration, if it is an
	// import group.
	var (
		before token.Pos
		decl0  ast.Decl
	)
	if len(file.Decls) > 0 {
		decl0 = file.Decls[0]
		before = decl0.Pos()
		switch decl0 := decl0.(type) {
		case *ast.GenDecl:
			if decl0.Doc != nil {
				before = decl0.Doc.Pos()
			}
		case *ast.FuncDecl:
			if decl0.Doc != nil {
				before = decl0.Doc.Pos()
			}
		}
	} else {
		before = file.FileEnd
	}
	var pos token.Pos
	if gd, ok := decl0.(*ast.GenDecl); ok && gd.Tok == token.IMPORT && gd.Rparen.IsValid() {
		// Have existing grouped import ( ... ) decl.
		if packagepath.MaybeStdPackage(pkgpath) && len(gd.Specs) > 0 {
			// Add spec for a std package before
			// first existing spec, followed by
			// a blank line if the next one is non-std.
			first := gd.Specs[0].(*ast.ImportSpec)
			pos = first.Pos()
			if !packagepath.MaybeStdPackage(first.Path.Value) {
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
		//
		// TODO(adonovan): do better here; plunder the
		// mergeImports logic from [imports.Process].
		pos = before
		newText = "import " + newText + "\n\n"
	}
	return []Edit{{
		Pos:     pos,
		End:     pos,
		NewText: []byte(newText),
	}}
}
