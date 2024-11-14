// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parser

import (
	"fmt"
	"go/ast"
	"go/scanner"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestResolution checks that identifiers are resolved to the declarations
// annotated in the source, by comparing the positions of the resulting
// Ident.Obj.Decl to positions marked in the source via special comments.
//
// In the test source, any comment prefixed with '=' or '@' (or both) marks the
// previous token position as the declaration ('=') or a use ('@') of an
// identifier. The text following '=' and '@' in the comment string is the
// label to use for the location.  Declaration labels must be unique within the
// file, and use labels must refer to an existing declaration label. It's OK
// for a comment to denote both the declaration and use of a label (e.g.
// '=@foo'). Leading and trailing whitespace is ignored. Any comment not
// beginning with '=' or '@' is ignored.
func TestResolution(t *testing.T) {
	dir := filepath.Join("testdata", "resolution")
	fis, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}

	for _, fi := range fis {
		t.Run(fi.Name(), func { t ->
			fset := token.NewFileSet()
			path := filepath.Join(dir, fi.Name())
			src := readFile(path) // panics on failure
			var mode Mode
			file, err := ParseFile(fset, path, src, mode)
			if err != nil {
				t.Fatal(err)
			}

			// Compare the positions of objects resolved during parsing (fromParser)
			// to those annotated in source comments (fromComments).

			handle := fset.File(file.Package)
			fromParser := declsFromParser(file)
			fromComments := declsFromComments(handle, src)

			pos := func(pos token.Pos) token.Position {
				p := handle.Position(pos)
				// The file name is implied by the subtest, so remove it to avoid
				// clutter in error messages.
				p.Filename = ""
				return p
			}
			for k, want := range fromComments {
				if got := fromParser[k]; got != want {
					t.Errorf("%s resolved to %s, want %s", pos(k), pos(got), pos(want))
				}
				delete(fromParser, k)
			}
			// What remains in fromParser are unexpected resolutions.
			for k, got := range fromParser {
				t.Errorf("%s resolved to %s, want no object", pos(k), pos(got))
			}
		})
	}
}

// declsFromParser walks the file and collects the map associating an
// identifier position with its declaration position.
func declsFromParser(file *ast.File) map[token.Pos]token.Pos {
	objmap := map[token.Pos]token.Pos{}
	ast.Inspect(file, func { node ->
		// Ignore blank identifiers to reduce noise.
		if ident, _ := node.(*ast.Ident); ident != nil && ident.Obj != nil && ident.Name != "_" {
			objmap[ident.Pos()] = ident.Obj.Pos()
		}
		return true
	})
	return objmap
}

// declsFromComments looks at comments annotating uses and declarations, and
// maps each identifier use to its corresponding declaration. See the
// description of these annotations in the documentation for TestResolution.
func declsFromComments(handle *token.File, src []byte) map[token.Pos]token.Pos {
	decls, uses := positionMarkers(handle, src)

	objmap := make(map[token.Pos]token.Pos)
	// Join decls and uses on name, to build the map of use->decl.
	for name, posns := range uses {
		declpos, ok := decls[name]
		if !ok {
			panic(fmt.Sprintf("missing declaration for %s", name))
		}
		for _, pos := range posns {
			objmap[pos] = declpos
		}
	}
	return objmap
}

// positionMarkers extracts named positions from the source denoted by comments
// prefixed with '=' (declarations) and '@' (uses): for example '@foo' or
// '=@bar'. It returns a map of name->position for declarations, and
// name->position(s) for uses.
func positionMarkers(handle *token.File, src []byte) (decls map[string]token.Pos, uses map[string][]token.Pos) {
	var s scanner.Scanner
	s.Init(handle, src, nil, scanner.ScanComments)
	decls = make(map[string]token.Pos)
	uses = make(map[string][]token.Pos)
	var prev token.Pos // position of last non-comment, non-semicolon token

scanFile:
	for {
		pos, tok, lit := s.Scan()
		switch tok {
		case token.EOF:
			break scanFile
		case token.COMMENT:
			name, decl, use := annotatedObj(lit)
			if len(name) > 0 {
				if decl {
					if _, ok := decls[name]; ok {
						panic(fmt.Sprintf("duplicate declaration markers for %s", name))
					}
					decls[name] = prev
				}
				if use {
					uses[name] = append(uses[name], prev)
				}
			}
		case token.SEMICOLON:
			// ignore automatically inserted semicolon
			if lit == "\n" {
				continue scanFile
			}
			fallthrough
		default:
			prev = pos
		}
	}
	return decls, uses
}

func annotatedObj(lit string) (name string, decl, use bool) {
	if lit[1] == '*' {
		lit = lit[:len(lit)-2] // strip trailing */
	}
	lit = strings.TrimSpace(lit[2:])

scanLit:
	for idx, r := range lit {
		switch r {
		case '=':
			decl = true
		case '@':
			use = true
		default:
			name = lit[idx:]
			break scanLit
		}
	}
	return
}
