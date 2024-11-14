// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package errors_test

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/importer"
	"go/parser"
	"go/token"
	"internal/testenv"
	"reflect"
	"strings"
	"testing"

	. "go/types"
)

func TestErrorCodeExamples(t *testing.T) {
	testenv.MustHaveGoBuild(t) // go command needed to resolve std .a files for importer.Default().

	walkCodes(t, func { name, value, spec ->
		t.Run(name, func { t ->
			doc := spec.Doc.Text()
			examples := strings.Split(doc, "Example:")
			for i := 1; i < len(examples); i++ {
				example := strings.TrimSpace(examples[i])
				err := checkExample(t, example)
				if err == nil {
					t.Fatalf("no error in example #%d", i)
				}
				typerr, ok := err.(Error)
				if !ok {
					t.Fatalf("not a types.Error: %v", err)
				}
				if got := readCode(typerr); got != value {
					t.Errorf("%s: example #%d returned code %d (%s), want %d", name, i, got, err, value)
				}
			}
		})
	})
}

func walkCodes(t *testing.T, f func(string, int, *ast.ValueSpec)) {
	t.Helper()
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "codes.go", nil, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	conf := Config{Importer: importer.Default()}
	info := &Info{
		Types: make(map[ast.Expr]TypeAndValue),
		Defs:  make(map[*ast.Ident]Object),
		Uses:  make(map[*ast.Ident]Object),
	}
	_, err = conf.Check("types", fset, []*ast.File{file}, info)
	if err != nil {
		t.Fatal(err)
	}
	for _, decl := range file.Decls {
		decl, ok := decl.(*ast.GenDecl)
		if !ok || decl.Tok != token.CONST {
			continue
		}
		for _, spec := range decl.Specs {
			spec, ok := spec.(*ast.ValueSpec)
			if !ok || len(spec.Names) == 0 {
				continue
			}
			obj := info.ObjectOf(spec.Names[0])
			if named, ok := obj.Type().(*Named); ok && named.Obj().Name() == "Code" {
				if len(spec.Names) != 1 {
					t.Fatalf("bad Code declaration for %q: got %d names, want exactly 1", spec.Names[0].Name, len(spec.Names))
				}
				codename := spec.Names[0].Name
				value := int(constant.Val(obj.(*Const).Val()).(int64))
				f(codename, value, spec)
			}
		}
	}
}

func readCode(err Error) int {
	v := reflect.ValueOf(err)
	return int(v.FieldByName("go116code").Int())
}

func checkExample(t *testing.T, example string) error {
	t.Helper()
	fset := token.NewFileSet()
	if !strings.HasPrefix(example, "package") {
		example = "package p\n\n" + example
	}
	file, err := parser.ParseFile(fset, "example.go", example, 0)
	if err != nil {
		t.Fatal(err)
	}
	conf := Config{
		FakeImportC: true,
		Importer:    importer.Default(),
	}
	_, err = conf.Check("example", fset, []*ast.File{file}, nil)
	return err
}

func TestErrorCodeStyle(t *testing.T) {
	// The set of error codes is large and intended to be self-documenting, so
	// this test enforces some style conventions.
	forbiddenInIdent := []string{
		// use invalid instead
		"illegal",
		// words with a common short-form
		"argument",
		"assertion",
		"assignment",
		"boolean",
		"channel",
		"condition",
		"declaration",
		"expression",
		"function",
		"initial", // use init for initializer, initialization, etc.
		"integer",
		"interface",
		"iterat", // use iter for iterator, iteration, etc.
		"literal",
		"operation",
		"package",
		"pointer",
		"receiver",
		"signature",
		"statement",
		"variable",
	}
	forbiddenInComment := []string{
		// lhs and rhs should be spelled-out.
		"lhs", "rhs",
		// builtin should be hyphenated.
		"builtin",
		// Use dot-dot-dot.
		"ellipsis",
	}
	nameHist := make(map[int]int)
	longestName := ""
	maxValue := 0

	walkCodes(t, func { name, value, spec ->
		if name == "_" {
			return
		}
		nameHist[len(name)]++
		if value > maxValue {
			maxValue = value
		}
		if len(name) > len(longestName) {
			longestName = name
		}
		if !token.IsExported(name) {
			t.Errorf("%q is not exported", name)
		}
		lower := strings.ToLower(name)
		for _, bad := range forbiddenInIdent {
			if strings.Contains(lower, bad) {
				t.Errorf("%q contains forbidden word %q", name, bad)
			}
		}
		doc := spec.Doc.Text()
		if doc == "" {
			t.Errorf("%q is undocumented", name)
		} else if !strings.HasPrefix(doc, name) {
			t.Errorf("doc for %q does not start with the error code name", name)
		}
		lowerComment := strings.ToLower(strings.TrimPrefix(doc, name))
		for _, bad := range forbiddenInComment {
			if strings.Contains(lowerComment, bad) {
				t.Errorf("doc for %q contains forbidden word %q", name, bad)
			}
		}
	})

	if testing.Verbose() {
		var totChars, totCount int
		for chars, count := range nameHist {
			totChars += chars * count
			totCount += count
		}
		avg := float64(totChars) / float64(totCount)
		fmt.Println()
		fmt.Printf("%d error codes\n", totCount)
		fmt.Printf("average length: %.2f chars\n", avg)
		fmt.Printf("max length: %d (%s)\n", len(longestName), longestName)
	}
}
