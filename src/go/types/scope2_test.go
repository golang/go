// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"fmt"
	"go/ast"
	"go/token"
	"reflect"
	"regexp"
	"strings"
	"testing"

	. "go/types"
)

// TestScopeLookupParent ensures that (*Scope).LookupParent returns
// the correct result at various positions with the source.
func TestScopeLookupParent(t *testing.T) {
	fset := token.NewFileSet()
	imports := make(testImporter)
	conf := Config{Importer: imports}
	var info Info
	makePkg := func(path string, files ...*ast.File) {
		var err error
		imports[path], err = conf.Check(path, fset, files, &info)
		if err != nil {
			t.Fatal(err)
		}
	}

	makePkg("lib", mustParse(fset, "package lib; var X int"))
	// Each /*name=kind:line*/ comment makes the test look up the
	// name at that point and checks that it resolves to a decl of
	// the specified kind and line number.  "undef" means undefined.
	// Note that type switch case clauses with an empty body (but for
	// comments) need the ";" to ensure that the recorded scope extends
	// past the comments.
	mainSrc := `
/*lib=pkgname:5*/ /*X=var:1*/ /*Pi=const:8*/ /*T=typename:9*/ /*Y=var:10*/ /*F=func:12*/
package main

import "lib"
import . "lib"

const Pi = 3.1415
type T struct{}
var Y, _ = lib.X, X

func F[T *U, U any](param1, param2 int) /*param1=undef*/ (res1 /*res1=undef*/, res2 int) /*param1=var:12*/ /*res1=var:12*/ /*U=typename:12*/ {
	const pi, e = 3.1415, /*pi=undef*/ 2.71828 /*pi=const:13*/ /*e=const:13*/
	type /*t=undef*/ t /*t=typename:14*/ *t
	print(Y) /*Y=var:10*/
	x, Y := Y, /*x=undef*/ /*Y=var:10*/ Pi /*x=var:16*/ /*Y=var:16*/ ; _ = x; _ = Y
	var F = /*F=func:12*/ F[*int, int] /*F=var:17*/ ; _ = F

	var a []int
	for i, x := range a /*i=undef*/ /*x=var:16*/ { _ = i; _ = x }

	var i interface{}
	switch y := i.(type) { /*y=undef*/
	case /*y=undef*/ int /*y=undef*/ : /*y=var:23*/ ;
	case float32, /*y=undef*/ float64 /*y=undef*/ : /*y=var:23*/ ;
	default /*y=undef*/ : /*y=var:23*/
		println(y)
	}
	/*y=undef*/

        switch int := i.(type) {
        case /*int=typename:0*/ int /*int=typename:0*/ : /*int=var:31*/
        	println(int)
        default /*int=typename:0*/ : /*int=var:31*/ ;
        }

	_ = param1
	_ = res1
	return
}
/*main=undef*/
`

	info.Uses = make(map[*ast.Ident]Object)
	f := mustParse(fset, mainSrc)
	makePkg("main", f)
	mainScope := imports["main"].Scope()
	rx := regexp.MustCompile(`^/\*(\w*)=([\w:]*)\*/$`)
	for _, group := range f.Comments {
		for _, comment := range group.List {
			// Parse the assertion in the comment.
			m := rx.FindStringSubmatch(comment.Text)
			if m == nil {
				t.Errorf("%s: bad comment: %s",
					fset.Position(comment.Pos()), comment.Text)
				continue
			}
			name, want := m[1], m[2]

			// Look up the name in the innermost enclosing scope.
			inner := mainScope.Innermost(comment.Pos())
			if inner == nil {
				t.Errorf("%s: at %s: can't find innermost scope",
					fset.Position(comment.Pos()), comment.Text)
				continue
			}
			got := "undef"
			if _, obj := inner.LookupParent(name, comment.Pos()); obj != nil {
				kind := strings.ToLower(strings.TrimPrefix(reflect.TypeOf(obj).String(), "*types."))
				got = fmt.Sprintf("%s:%d", kind, fset.Position(obj.Pos()).Line)
			}
			if got != want {
				t.Errorf("%s: at %s: %s resolved to %s, want %s",
					fset.Position(comment.Pos()), comment.Text, name, got, want)
			}
		}
	}

	// Check that for each referring identifier,
	// a lookup of its name on the innermost
	// enclosing scope returns the correct object.

	for id, wantObj := range info.Uses {
		inner := mainScope.Innermost(id.Pos())
		if inner == nil {
			t.Errorf("%s: can't find innermost scope enclosing %q",
				fset.Position(id.Pos()), id.Name)
			continue
		}

		// Exclude selectors and qualified identifiers---lexical
		// refs only.  (Ideally, we'd see if the AST parent is a
		// SelectorExpr, but that requires PathEnclosingInterval
		// from golang.org/x/tools/go/ast/astutil.)
		if id.Name == "X" {
			continue
		}

		_, gotObj := inner.LookupParent(id.Name, id.Pos())
		if gotObj != wantObj {
			// Print the scope tree of mainScope in case of error.
			var printScopeTree func(indent string, s *Scope)
			printScopeTree = func(indent string, s *Scope) {
				t.Logf("%sscope %s %v-%v = %v",
					indent,
					ScopeComment(s),
					s.Pos(),
					s.End(),
					s.Names())
				for i := range s.NumChildren() {
					printScopeTree(indent+"  ", s.Child(i))
				}
			}
			printScopeTree("", mainScope)

			t.Errorf("%s: Scope(%s).LookupParent(%s@%v) got %v, want %v [scopePos=%v]",
				fset.Position(id.Pos()),
				ScopeComment(inner),
				id.Name,
				id.Pos(),
				gotObj,
				wantObj,
				ObjectScopePos(wantObj))
			continue
		}
	}
}
