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

// TestRefs checks that the analysis reports, for each exported member
// of the test package ("p"), its correct dependencies on exported
// members of its direct imports (e.g. "ext").
func TestRefs(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		label     string
		srcs      []string            // source for the local package; package name must be p
		imports   map[string]string   // for simplicity: importPath -> pkgID/pkgName (we set pkgName == pkgID); 'ext' is always available.
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

import "ext"

type A struct{ b B }
type B func(c C) (d D)
type C ext.C
type D ext.D

// Should not be referenced by field names.
type b ext.B_
type c int.C_
type d ext.D_
`},
			want: map[string][]string{
				"A": {"ext.C", "ext.D"},
				"B": {"ext.C", "ext.D"},
				"C": {"ext.C"},
				"D": {"ext.D"},
			},
		},
		{
			label: "embedding",
			srcs: []string{`
package p

import "ext"

type A struct{
	B
	_ struct {
		C
	}
	D
}
type B ext.B
type C ext.C
type D interface{
	B
}
`},
			want: map[string][]string{
				"A": {"ext.B", "ext.C"},
				"B": {"ext.B"},
				"C": {"ext.C"},
				"D": {"ext.B"},
			},
		},
		{
			label: "constraint embedding",
			srcs: []string{`
package p

import "ext"

type A interface{
	int | B | ~C
	struct{D}
}

type B ext.B
type C ext.C
type D ext.D
`},
			want: map[string][]string{
				"A": {"ext.B", "ext.C", "ext.D"},
				"B": {"ext.B"},
				"C": {"ext.C"},
				"D": {"ext.D"},
			},
			go118: true,
		},
		{
			label: "funcs",
			srcs: []string{`
package p

import "ext"

type A ext.A
type B ext.B
const C B = 2
func F(A) B {
	return C
}
var V = F(W)
var W A
`},
			want: map[string][]string{
				"A": {"ext.A"},
				"B": {"ext.B"},
				"C": {"ext.B"},
				"F": {"ext.A", "ext.B"},
				"V": {
					"ext.A", // via F
					"ext.B", // via W: can't be eliminated: F could be builtin or generic
				},
				"W": {"ext.A"},
			},
		},
		{
			label: "methods",
			srcs: []string{`package p

import "ext"

type A ext.A
type B ext.B
`, `package p

func (A) M(B)
func (*B) M(A)
`},
			want: map[string][]string{
				"A": {"ext.A", "ext.B"},
				"B": {"ext.A", "ext.B"},
			},
		},
		{
			label: "initializers",
			srcs: []string{`
package p

import "ext"

var A b = C // type does not depend on C
type b ext.B
var C = d // type does depend on D
var d b

var e = d + a

var F = func() B { return E }

var G = struct{
	A b
	_ [unsafe.Sizeof(ext.V)]int // array size + Sizeof creates edge to a var
	_ [unsafe.Sizeof(G)]int // creates a self edge; doesn't affect output though
}{}

var H = (D + A + C*C)

var I = (A+C).F
`},
			want: map[string][]string{
				"A": {"ext.B"},
				"C": {"ext.B"},          // via d
				"G": {"ext.B", "ext.V"}, // via b,C
				"H": {"ext.B"},          // via d,A,C
				"I": {"ext.B"},
			},
		},
		{
			label: "builtins",
			srcs: []string{`package p

import "ext"

var A = new(b)
type b struct{ ext.B }

type C chan d
type d ext.D

type S []ext.S
type t ext.T
var U = append(([]*S)(nil), new(t))

type X map[k]v
type k ext.K
type v ext.V

var Z = make(map[k]A)

// close, delete, and panic cannot occur outside of statements
`},
			want: map[string][]string{
				"A": {"ext.B"},
				"C": {"ext.D"},
				"S": {"ext.S"},
				"U": {"ext.S", "ext.T"}, // ext.T edge could be eliminated
				"X": {"ext.K", "ext.V"},
				"Z": {"ext.B", "ext.K"},
			},
		},
		{
			label: "builtin shadowing",
			srcs: []string{`package p

import "ext"

var A = new(ext.B)
func new() c
type c ext.C
`},
			want: map[string][]string{
				"A": {"ext.B", "ext.C"},
			},
		},
		{
			label: "named forwarding",
			srcs: []string{`package p

import "ext"

type A B
type B c
type c ext.C
`},
			want: map[string][]string{
				"A": {"ext.C"},
				"B": {"ext.C"},
			},
		},
		{
			label: "aliases",
			srcs: []string{`package p

import "ext"

type A = B
type B = C
type C = ext.C
`},
			want: map[string][]string{
				"A": {"ext.C"},
				"B": {"ext.C"},
				"C": {"ext.C"},
			},
		},
		{
			label: "array length",
			srcs: []string{`package p

import "ext"
import "unsafe"

type A [unsafe.Sizeof(ext.B{ext.C})]int
type A2 [unsafe.Sizeof(ext.B{f:ext.C})]int // use a KeyValueExpr

type D [unsafe.Sizeof(struct{ f E })]int
type E ext.E

type F [3]G
type G [ext.C]int
`},
			want: map[string][]string{
				"A":  {"ext.B"}, // no ext.C: doesn't enter CompLit
				"A2": {"ext.B"}, // ditto
				"D":  {"ext.E"},
				"E":  {"ext.E"},
				"F":  {"ext.C"},
				"G":  {"ext.C"},
			},
		},
		{
			label: "imports",
			srcs: []string{`package p

import "ext"

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
	z.Z // references both external z.Z as well as package-level type z
}

type B struct {
	r.R // invalid ref
	t.T
}

var X int = q.V // X={}: no descent into RHS of 'var v T = rhs'
var Y = q.V.W

type z ext.Z
`},
			imports: map[string]string{"q": "q", "r": "r", "s": "t", "z": "z"},
			want: map[string][]string{
				"A": {"ext.Z", "q.Q", "r.R", "z.Z"},
				"B": {"t.T"},
				"Y": {"q.V"},
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

import "ext"
import "q"

type D ext.D
`},
			imports: map[string]string{"q": "q"},
			want: map[string][]string{
				"B": {"ext.D", "q.C"},
				"D": {"ext.D"},
				"F": {"q.Field"},
				"G": {"q.Field"},
			},
		},
		{
			label: "typeparams",
			srcs: []string{`package p

import "ext"

type A[T any] struct {
	t T
	b B
}

type B ext.B

func F1[T any](T, B)
func F2[T C]()(T, B)

type T ext.T

type C ext.C

func F3[T1 ~[]T2, T2 ~[]T3](t1 T1, t2 T2)
type T3 ext.T3
`, `package p

func (A[B]) M(C) {}
`},
			want: map[string][]string{
				"A":  {"ext.B", "ext.C"},
				"B":  {"ext.B"},
				"C":  {"ext.C"},
				"F1": {"ext.B"},
				"F2": {"ext.B", "ext.C"},
				"F3": {"ext.T3"},
				"T":  {"ext.T"},
				"T3": {"ext.T3"},
			},
			go118: true,
		},
		{
			label: "instances",
			srcs: []string{`package p

import "ext"

type A[T any] ext.A
type B[T1, T2 any] ext.B

type C A[int]
type D B[int, A[E]]
type E ext.E
`},
			want: map[string][]string{
				"A": {"ext.A"},
				"B": {"ext.B"},
				"C": {"ext.A"},
				"D": {"ext.A", "ext.B", "ext.E"},
				"E": {"ext.E"},
			},
			go118: true,
		},
		{
			label: "duplicate decls",
			srcs: []string{`package p

import "a"
import "ext"

type a a.A
type A a
type b ext.B
type C a.A
func (C) Foo(x) {} // invalid parameter, but that does not matter
type C b
func (C) Bar(y) {} // invalid parameter, but that does not matter

var x ext.X
var y ext.Y
`},
			imports: map[string]string{"a": "a", "b": "b"}, // "b" import should not matter, since it isn't in this file
			want: map[string][]string{
				"A": {"a.A"},
				"C": {"a.A", "ext.B", "ext.X", "ext.Y"},
			},
		},
		{
			label: "invalid decls",
			srcs: []string{`package p

import "ext"

type A B

func () Foo(B){}

var B struct{ ext.B
`},
			want: map[string][]string{
				"A":   {"ext.B"},
				"B":   {"ext.B"},
				"Foo": {"ext.B"},
			},
			allowErrs: true,
		},
		{
			label: "unmapped receiver",
			srcs: []string{`package p

type P struct{}

func (a) x(P)
`},
			want:      map[string][]string{},
			allowErrs: true,
		},
		{
			label: "SCC special case",
			srcs: []string{`package p

import "ext"

type X Y
type Y struct { Z; *X }
type Z map[ext.A]ext.B
`},
			want: map[string][]string{
				"X": {"ext.A", "ext.B"},
				"Y": {"ext.A", "ext.B"},
				"Z": {"ext.A", "ext.B"},
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

			imports := map[source.ImportPath]*source.Metadata{
				"ext": {ID: "ext", Name: "ext"}, // this one comes for free
			}
			for path, m := range test.imports {
				imports[source.ImportPath(path)] = &source.Metadata{
					ID:   source.PackageID(m),
					Name: source.PackageName(m),
				}
			}

			data := typerefs.Encode(pgfs, "p", imports)

			got := make(map[string][]string)
			index := typerefs.NewPackageIndex()
			for _, class := range typerefs.Decode(index, "p", data) {
				// We redundantly expand out the name x refs cross product
				// here since that's what the existing tests expect.
				for _, name := range class.Decls {
					var syms []string
					for _, sym := range class.Refs {
						syms = append(syms, fmt.Sprintf("%s.%s", sym.PackageID(index), sym.Name))
					}
					sort.Strings(syms)
					got[name] = syms
				}
			}

			if diff := cmp.Diff(test.want, got); diff != "" {
				t.Errorf("Refs(...) returned unexpected refs (-want +got):\n%s", diff)
			}
		})
	}
}
