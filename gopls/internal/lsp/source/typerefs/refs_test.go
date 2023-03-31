// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typerefs_test

import (
	"context"
	"fmt"
	"go/token"
	"sort"
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/source/typerefs"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/testenv"
)

func TestRefs(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		label     string
		srcs      []string            // source for the local package; package name must be p
		imports   map[string]string   // for simplicity: importPath -> pkgID/pkgName (we set pkgName == pkgID)
		want      map[string][]string // decl name -> id.<decl name>
		go118     bool                // test uses generics
		allowErrs bool                // whether we expect parsing errors
	}{
		{
			label: "empty package",
			want:  map[string][]string{},
		},
		{
			label: "fields",
			srcs: []string{`
package p

type A struct{ b B }
type B func(c C) (d D)
type C int
type D int

// Should not be referenced by field names.
type b int
type c int
type d int
`},
			want: map[string][]string{
				"A": {"p.B"},
				"B": {"p.C", "p.D"},
			},
		},
		{
			label: "embedding",
			srcs: []string{`
package p

type A struct{
	B
	_ struct {
		C
	}
	D
}
type B int
type C int
type D interface{
	B
}
`},
			want: map[string][]string{
				"A": {"p.B", "p.C", "p.D"},
				"D": {"p.B"},
			},
		},
		{
			label: "constraint embedding",
			srcs: []string{`
package p

type A interface{
	int | B | ~C
	struct{D}
}

type B int
type C int
type D int
`},
			want: map[string][]string{
				"A": {"p.B", "p.C", "p.D"},
			},
			go118: true,
		},
		{
			label: "funcs",
			srcs: []string{`
package p

type A int
type B int
const C B = 2
func F(A) B {
	return C
}
var V = F(W)
var W A
`},
			want: map[string][]string{
				"C": {"p.B"},
				"F": {"p.A", "p.B"},
				"V": {"p.F", "p.W"}, // p.W edge can't be eliminated: F could be builtin or generic
				"W": {"p.A"},
			},
		},
		{
			label: "methods",
			srcs: []string{`package p

type A int
type B int
`, `package p

func (A) M(B)
func (*B) M(A)
`},
			want: map[string][]string{
				"A": {"p.B"},
				"B": {"p.A"},
			},
		},
		{
			label: "initializers",
			srcs: []string{`
package p

var a b = c // type does not depend on c
type b int
var c = d // type does depend on d
var d b

var e = d + a

var f = func() b { return e }

var g = struct{
	a b
	_ [unsafe.Sizeof(g)]int
}{}

var h = (d + a + c*c)

var i = (a+c).f
`},
			want: map[string][]string{
				"a": {"p.b"},
				"c": {"p.d"},
				"d": {"p.b"},
				"e": {"p.a", "p.d"},
				"f": {"p.b"},
				"g": {"p.b", "p.g"}, // p.g edge could be skipped
				"h": {"p.a", "p.c", "p.d"},
				"i": {"p.a", "p.c"},
			},
		},
		{
			label: "builtins",
			srcs: []string{`package p

var A = new(B)
type B struct{}

type C chan D
type D int

type S []T
type T int
var U = append(([]*S)(nil), new(T))

type X map[K]V
type K int
type V int

var Z = make(map[K]A)

// close, delete, and panic cannot occur outside of statements
`},
			want: map[string][]string{
				"A": {"p.B"},
				"C": {"p.D"},
				"S": {"p.T"},
				"U": {"p.S", "p.T"}, // p.T edge could be eliminated
				"X": {"p.K", "p.V"},
				"Z": {"p.A", "p.K"},
			},
		},
		{
			label: "builtin shadowing",
			srcs: []string{`package p

var A = new(B)
func new() c
type c int
`},
			want: map[string][]string{
				"A":   {"p.new"},
				"new": {"p.c"},
			},
		},
		{
			label: "named forwarding",
			srcs: []string{`package p

type A B
type B C
type C int
`},
			want: map[string][]string{
				"A": {"p.B"},
				"B": {"p.C"},
			},
		},
		{
			label: "aliases",
			srcs: []string{`package p

type A = B
type B = C
type C = int
`},
			want: map[string][]string{
				"A": {"p.B"},
				"B": {"p.C"},
			},
		},
		{
			label: "array length",
			srcs: []string{`package p

import "unsafe"

type A [unsafe.Sizeof(B{C})]int
type A2 [unsafe.Sizeof(B{f:C})]int // use a KeyValueExpr
type B struct{ f int }
var C = 0

type D [unsafe.Sizeof(struct{ f E })]int
type E int

type F [3]G
type G [C]int
`},
			want: map[string][]string{
				"A":  {"p.B"},
				"A2": {"p.B"},
				"D":  {"p.E"},
				"F":  {"p.G"},
				"G":  {"p.C"},
			},
		},
		{
			label: "imports",
			srcs: []string{`package p

import (
	"q"
	r2 "r"
	"s" // note: package name is t
	"z"
)

type A struct {
	q.Q
	r2.R
	s.S // invalid ref
	z.Z // note: shadowed below
}

type B struct {
	r.R // invalid ref
	t.T
}

var x int = q.V
var y = q.V.W

type z interface{}
`},
			imports: map[string]string{"q": "q", "r": "r", "s": "t", "z": "z"},
			want: map[string][]string{
				"A": {"p.z", "q.Q", "r.R", "z.Z"},
				"B": {"t.T"},
				"y": {"q.V"},
			},
		},
		{
			label: "import blank",
			srcs: []string{`package p

import _ "q"

type A q.Q
`},
			imports: map[string]string{"q": "q"},
			want:    map[string][]string{},
		},
		{
			label: "import dot",
			srcs: []string{`package p

import . "q"

type A q.Q // not actually an edge, since q is imported .
type B struct {
	C // assumed to be an edge to q
	D // resolved to package decl
}


type E error // unexported, therefore must be universe.error
type F Field
var G = Field.X
`, `package p

type D interface{}
`},
			imports: map[string]string{"q": "q"},
			want: map[string][]string{
				"B": {"p.D", "q.C"},
				"F": {"q.Field"},
				"G": {"q.Field"},
			},
		},
		{
			label: "typeparams",
			srcs: []string{`package p

type A[T any] struct {
	t T
	b B
}

type B int

func F1[T any](T, B)
func F2[T C]()(T, B)

type T int

type C any

func F3[T1 ~[]T2, T2 ~[]T3](t1 T1, t2 T2)
type T3 any
`, `package p

func (A[B]) M(C) {}
`},
			want: map[string][]string{
				"A":  {"p.B", "p.C"},
				"F1": {"p.B"},
				"F2": {"p.B", "p.C"},
				"F3": {"p.T3"},
			},
			go118: true,
		},
		{
			label: "instances",
			srcs: []string{`package p

type A[T any] struct{}
type B[T1, T2 any] struct{}

type C A[int]
type D B[int, A[E]]
type E int
`},
			want: map[string][]string{
				"C": {"p.A"},
				"D": {"p.A", "p.B", "p.E"},
			},
			go118: true,
		},
		{
			label: "duplicate decls",
			srcs: []string{`package p

import "a"

type a a.A
type b int
type C a.A
func (C) Foo(x) {} // invalid parameter, but that does not matter
type C b
func (C) Bar(y) {} // invalid parameter, but that does not matter

var x, y int
`},
			imports: map[string]string{"a": "a", "b": "b"}, // "b" import should not matter, since it isn't in this file
			want: map[string][]string{
				"a": {"a.A", "p.a"},
				"C": {"a.A", "p.a", "p.b", "p.x", "p.y"},
			},
		},
		{
			label: "invalid decls",
			srcs: []string{`package p

type A B

func () Foo(B){}

var B
`},
			want: map[string][]string{
				"A":   {"p.B"},
				"Foo": {"p.B"},
			},
			allowErrs: true,
		},
	}

	for _, test := range tests {
		t.Run(test.label, func(t *testing.T) {
			if test.go118 {
				testenv.NeedsGo1Point(t, 18)
			}

			var pgfs []*source.ParsedGoFile
			for i, src := range test.srcs {
				uri := span.URI(fmt.Sprintf("file:///%d.go", i))
				pgf, _ := cache.ParseGoSrc(ctx, token.NewFileSet(), uri, []byte(src), source.ParseFull)
				if !test.allowErrs && pgf.ParseErr != nil {
					t.Fatalf("ParseGoSrc(...) returned parse errors: %v", pgf.ParseErr)
				}
				pgfs = append(pgfs, pgf)
			}

			imports := make(map[source.ImportPath]*source.Metadata)
			for path, m := range test.imports {
				imports[source.ImportPath(path)] = &source.Metadata{
					ID:   source.PackageID(m),
					Name: source.PackageName(m),
				}
			}

			index := typerefs.NewPackageIndex()
			refs := typerefs.Refs(pgfs, "p", imports, index)

			got := make(map[string][]string)
			for name, refs := range refs {
				var srefs []string
				for _, ref := range refs {
					id, name := ref.Unpack(index)
					srefs = append(srefs, fmt.Sprintf("%s.%s", id, name))
				}
				sort.Strings(srefs)
				got[name] = srefs
			}

			if diff := cmp.Diff(test.want, got); diff != "" {
				t.Errorf("Refs(...) returned unexpected refs (-want +got):\n%s", diff)
			}
		})
	}
}
