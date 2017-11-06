// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rename

import (
	"bytes"
	"fmt"
	"go/build"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"

	"golang.org/x/tools/go/buildutil"
)

// TODO(adonovan): test reported source positions, somehow.

func TestConflicts(t *testing.T) {
	defer func(savedWriteFile func(string, []byte) error, savedReportError func(token.Position, string)) {
		writeFile = savedWriteFile
		reportError = savedReportError
	}(writeFile, reportError)
	writeFile = func(string, []byte) error { return nil }

	var ctxt *build.Context
	for _, test := range []struct {
		ctxt             *build.Context // nil => use previous
		offset, from, to string         // values of the -offset/-from and -to flags
		want             string         // regexp to match conflict errors, or "OK"
	}{
		// init() checks
		{
			ctxt: fakeContext(map[string][]string{
				"fmt": {`package fmt; type Stringer interface { String() }`},
				"main": {`
package main

import foo "fmt"

var v foo.Stringer

func f() { v.String(); f() }
`,
					`package main; var w int`},
			}),
			from: "main.v", to: "init",
			want: `you cannot have a var at package level named "init"`,
		},
		{
			from: "main.f", to: "init",
			want: `renaming this func "f" to "init" would make it a package initializer.*` +
				`but references to it exist`,
		},
		{
			from: "/go/src/main/0.go::foo", to: "init",
			want: `"init" is not a valid imported package name`,
		},

		// Export checks
		{
			from: "fmt.Stringer", to: "stringer",
			want: `renaming this type "Stringer" to "stringer" would make it unexported.*` +
				`breaking references from packages such as "main"`,
		},
		{
			from: "(fmt.Stringer).String", to: "string",
			want: `renaming this method "String" to "string" would make it unexported.*` +
				`breaking references from packages such as "main"`,
		},

		// Lexical scope checks
		{
			// file/package conflict, same file
			from: "main.v", to: "foo",
			want: `renaming this var "v" to "foo" would conflict.*` +
				`with this imported package name`,
		},
		{
			// file/package conflict, same file
			from: "main::foo", to: "v",
			want: `renaming this imported package name "foo" to "v" would conflict.*` +
				`with this package member var`,
		},
		{
			// file/package conflict, different files
			from: "main.w", to: "foo",
			want: `renaming this var "w" to "foo" would conflict.*` +
				`with this imported package name`,
		},
		{
			// file/package conflict, different files
			from: "main::foo", to: "w",
			want: `renaming this imported package name "foo" to "w" would conflict.*` +
				`with this package member var`,
		},
		{
			ctxt: main(`
package main

var x, z int

func f(y int) {
	print(x)
	print(y)
}

func g(w int) {
	print(x)
	x := 1
	print(x)
}`),
			from: "main.x", to: "y",
			want: `renaming this var "x" to "y".*` +
				`would cause this reference to become shadowed.*` +
				`by this intervening var definition`,
		},
		{
			from: "main.g::x", to: "w",
			want: `renaming this var "x" to "w".*` +
				`conflicts with var in same block`,
		},
		{
			from: "main.f::y", to: "x",
			want: `renaming this var "y" to "x".*` +
				`would shadow this reference.*` +
				`to the var declared here`,
		},
		{
			from: "main.g::w", to: "x",
			want: `renaming this var "w" to "x".*` +
				`conflicts with var in same block`,
		},
		{
			from: "main.z", to: "y", want: "OK",
		},

		// Label checks
		{
			ctxt: main(`
package main

func f() {
foo:
	goto foo
bar:
	goto bar
	func(x int) {
	wiz:
		goto wiz
	}(0)
}
`),
			from: "main.f::foo", to: "bar",
			want: `renaming this label "foo" to "bar".*` +
				`would conflict with this one`,
		},
		{
			from: "main.f::foo", to: "wiz", want: "OK",
		},
		{
			from: "main.f::wiz", to: "x", want: "OK",
		},
		{
			from: "main.f::x", to: "wiz", want: "OK",
		},
		{
			from: "main.f::wiz", to: "foo", want: "OK",
		},

		// Struct fields
		{
			ctxt: main(`
package main

type U struct { u int }
type V struct { v int }

func (V) x() {}

type W (struct {
	U
	V
	w int
})

func f() {
	var w W
	print(w.u) // NB: there is no selection of w.v
	var _ struct { yy, zz int }
}
`),
			// field/field conflict in named struct declaration
			from: "(main.W).U", to: "w",
			want: `renaming this field "U" to "w".*` +
				`would conflict with this field`,
		},
		{
			// rename type used as embedded field
			// => rename field
			// => field/field conflict
			// This is an entailed renaming;
			// it would be nice if we checked source positions.
			from: "main.U", to: "w",
			want: `renaming this field "U" to "w".*` +
				`would conflict with this field`,
		},
		{
			// field/field conflict in unnamed struct declaration
			from: "main.f::zz", to: "yy",
			want: `renaming this field "zz" to "yy".*` +
				`would conflict with this field`,
		},

		// Now we test both directions of (u,v) (u,w) (v,w) (u,x) (v,x).
		// Too bad we don't test position info...
		{
			// field/field ambiguity at same promotion level ('from' selection)
			from: "(main.U).u", to: "v",
			want: `renaming this field "u" to "v".*` +
				`would make this reference ambiguous.*` +
				`with this field`,
		},
		{
			// field/field ambiguity at same promotion level ('to' selection)
			from: "(main.V).v", to: "u",
			want: `renaming this field "v" to "u".*` +
				`would make this reference ambiguous.*` +
				`with this field`,
		},
		{
			// field/method conflict at different promotion level ('from' selection)
			from: "(main.U).u", to: "w",
			want: `renaming this field "u" to "w".*` +
				`would change the referent of this selection.*` +
				`of this field`,
		},
		{
			// field/field shadowing at different promotion levels ('to' selection)
			from: "(main.W).w", to: "u",
			want: `renaming this field "w" to "u".*` +
				`would shadow this selection.*` +
				`of the field declared here`,
		},
		{
			from: "(main.V).v", to: "w",
			want: "OK", // since no selections are made ambiguous
		},
		{
			from: "(main.W).w", to: "v",
			want: "OK", // since no selections are made ambiguous
		},
		{
			// field/method ambiguity at same promotion level ('from' selection)
			from: "(main.U).u", to: "x",
			want: `renaming this field "u" to "x".*` +
				`would make this reference ambiguous.*` +
				`with this method`,
		},
		{
			// field/field ambiguity at same promotion level ('to' selection)
			from: "(main.V).x", to: "u",
			want: `renaming this method "x" to "u".*` +
				`would make this reference ambiguous.*` +
				`with this field`,
		},
		{
			// field/method conflict at named struct declaration
			from: "(main.V).v", to: "x",
			want: `renaming this field "v" to "x".*` +
				`would conflict with this method`,
		},
		{
			// field/method conflict at named struct declaration
			from: "(main.V).x", to: "v",
			want: `renaming this method "x" to "v".*` +
				`would conflict with this field`,
		},

		// Methods
		{
			ctxt: main(`
package main
type C int
func (C) f()
func (C) g()
type D int
func (*D) f()
func (*D) g()
type I interface { f(); g() }
type J interface { I; h() }
var _ I = new(D)
var _ interface {f()} = C(0)
`),
			from: "(main.I).f", to: "g",
			want: `renaming this interface method "f" to "g".*` +
				`would conflict with this method`,
		},
		{
			from: `("main".I).f`, to: "h", // NB: exercises quoted import paths too
			want: `renaming this interface method "f" to "h".*` +
				`would conflict with this method.*` +
				`in named interface type "J"`,
		},
		{
			// type J interface { h; h() } is not a conflict, amusingly.
			from: "main.I", to: "h",
			want: `OK`,
		},
		{
			from: "(main.J).h", to: "f",
			want: `renaming this interface method "h" to "f".*` +
				`would conflict with this method`,
		},
		{
			from: "(main.C).f", to: "e",
			want: `renaming this method "f" to "e".*` +
				`would make main.C no longer assignable to interface{f..}.*` +
				`(rename interface{f..}.f if you intend to change both types)`,
		},
		{
			from: "(main.D).g", to: "e",
			want: `renaming this method "g" to "e".*` +
				`would make \*main.D no longer assignable to interface I.*` +
				`(rename main.I.g if you intend to change both types)`,
		},
		{
			from: "(main.I).f", to: "e",
			want: `OK`,
		},
		// Indirect C/I method coupling via another concrete type D.
		{
			ctxt: main(`
package main
type I interface { f() }
type C int
func (C) f()
type D struct{C}
var _ I = D{}
`),
			from: "(main.C).f", to: "F",
			want: `renaming this method "f" to "F".*` +
				`would make main.D no longer assignable to interface I.*` +
				`(rename main.I.f if you intend to change both types)`,
		},
		// Renaming causes promoted method to become shadowed; C no longer satisfies I.
		{
			ctxt: main(`
package main
type I interface { f() }
type C struct { I }
func (C) g() int
var _ I = C{}
`),
			from: "main.I.f", to: "g",
			want: `renaming this method "f" to "g".*` +
				`would change the g method of main.C invoked via interface main.I.*` +
				`from \(main.I\).g.*` +
				`to \(main.C\).g`,
		},
		// Renaming causes promoted method to become ambiguous; C no longer satisfies I.
		{
			ctxt: main(`
package main
type I interface{f()}
type C int
func (C) f()
type D int
func (D) g()
type E struct{C;D}
var _ I = E{}
`),
			from: "main.I.f", to: "g",
			want: `renaming this method "f" to "g".*` +
				`would make the g method of main.E invoked via interface main.I ambiguous.*` +
				`with \(main.D\).g`,
		},
	} {
		var conflicts []string
		reportError = func(posn token.Position, message string) {
			conflicts = append(conflicts, message)
		}
		if test.ctxt != nil {
			ctxt = test.ctxt
		}
		err := Main(ctxt, test.offset, test.from, test.to)
		var prefix string
		if test.offset == "" {
			prefix = fmt.Sprintf("-from %q -to %q", test.from, test.to)
		} else {
			prefix = fmt.Sprintf("-offset %q -to %q", test.offset, test.to)
		}
		if err == ConflictError {
			got := strings.Join(conflicts, "\n")
			if false {
				t.Logf("%s: %s", prefix, got)
			}
			pattern := "(?s:" + test.want + ")" // enable multi-line matching
			if !regexp.MustCompile(pattern).MatchString(got) {
				t.Errorf("%s: conflict does not match pattern:\n"+
					"Conflict:\t%s\n"+
					"Pattern: %s",
					prefix, got, test.want)
			}
		} else if err != nil {
			t.Errorf("%s: unexpected error: %s", prefix, err)
		} else if test.want != "OK" {
			t.Errorf("%s: unexpected success, want conflicts matching:\n%s",
				prefix, test.want)
		}
	}
}

func TestInvalidIdentifiers(t *testing.T) {
	ctxt := fakeContext(map[string][]string{
		"main": {`
package main

func f() { }
`}})

	for _, test := range []struct {
		from, to string // values of the -offset/-from and -to flags
		want     string // expected error message
	}{
		{
			from: "main.f", to: "_",
			want: `-to "_": not a valid identifier`,
		},
		{
			from: "main.f", to: "123",
			want: `-to "123": not a valid identifier`,
		},
		{
			from: "main.f", to: "for",
			want: `-to "for": not a valid identifier`,
		},
		{
			from: "switch", to: "v",
			want: `-from "switch": invalid expression`,
		},
	} {
		err := Main(ctxt, "", test.from, test.to)
		prefix := fmt.Sprintf("-from %q -to %q", test.from, test.to)
		if err == nil {
			t.Errorf("%s: expected error %q", prefix, test.want)
		} else if err.Error() != test.want {
			t.Errorf("%s: unexpected error\nwant: %s\n got: %s", prefix, test.want, err.Error())
		}
	}
}

func TestRewrites(t *testing.T) {
	defer func(savedWriteFile func(string, []byte) error) {
		writeFile = savedWriteFile
	}(writeFile)

	var ctxt *build.Context
	for _, test := range []struct {
		ctxt             *build.Context    // nil => use previous
		offset, from, to string            // values of the -from/-offset and -to flags
		want             map[string]string // contents of updated files
	}{
		// Elimination of renaming import.
		{
			ctxt: fakeContext(map[string][]string{
				"foo": {`package foo; type T int`},
				"main": {`package main

import foo2 "foo"

var _ foo2.T
`},
			}),
			from: "main::foo2", to: "foo",
			want: map[string]string{
				"/go/src/main/0.go": `package main

import "foo"

var _ foo.T
`,
			},
		},
		// Introduction of renaming import.
		{
			ctxt: fakeContext(map[string][]string{
				"foo": {`package foo; type T int`},
				"main": {`package main

import "foo"

var _ foo.T
`},
			}),
			offset: "/go/src/main/0.go:#36", to: "foo2", // the "foo" in foo.T
			want: map[string]string{
				"/go/src/main/0.go": `package main

import foo2 "foo"

var _ foo2.T
`,
			},
		},
		// Renaming of package-level member.
		{
			from: "foo.T", to: "U",
			want: map[string]string{
				"/go/src/main/0.go": `package main

import "foo"

var _ foo.U
`,
				"/go/src/foo/0.go": `package foo

type U int
`,
			},
		},
		// Rename package-level func plus doc
		{
			ctxt: main(`package main

// Foo is a no-op.
// Calling Foo does nothing.
func Foo() {
}
`),
			from: "main.Foo", to: "FooBar",
			want: map[string]string{
				"/go/src/main/0.go": `package main

// FooBar is a no-op.
// Calling FooBar does nothing.
func FooBar() {
}
`,
			},
		},
		// Rename method plus doc
		{
			ctxt: main(`package main

type Foo struct{}

// Bar does nothing.
func (Foo) Bar() {
}
`),
			from: "main.Foo.Bar", to: "Baz",
			want: map[string]string{
				"/go/src/main/0.go": `package main

type Foo struct{}

// Baz does nothing.
func (Foo) Baz() {
}
`,
			},
		},
		// Rename type spec plus doc
		{
			ctxt: main(`package main

type (
	// Test but not Testing.
	Test struct{}
)
`),
			from: "main.Test", to: "Type",
			want: map[string]string{
				"/go/src/main/0.go": `package main

type (
	// Type but not Testing.
	Type struct{}
)
`,
			},
		},
		// Rename type in gen decl plus doc
		{
			ctxt: main(`package main

// T is a test type.
type T struct{}
`),
			from: "main.T", to: "Type",
			want: map[string]string{
				"/go/src/main/0.go": `package main

// Type is a test type.
type Type struct{}
`,
			},
		},
		// Rename value spec with doc
		{
			ctxt: main(`package main

const (
	// C is the speed of light.
	C = 2.998e8
)
`),
			from: "main.C", to: "Lightspeed",
			want: map[string]string{
				"/go/src/main/0.go": `package main

const (
	// Lightspeed is the speed of light.
	Lightspeed = 2.998e8
)
`,
			},
		},
		// Rename value inside gen decl with doc
		{
			ctxt: main(`package main

var out *string
`),
			from: "main.out", to: "discard",
			want: map[string]string{
				"/go/src/main/0.go": `package main

var discard *string
`,
			},
		},
		// Rename field plus doc
		{
			ctxt: main(`package main

type Struct struct {
	// Field is a struct field.
	Field string
}
`),
			from: "main.Struct.Field", to: "Foo",
			want: map[string]string{
				"/go/src/main/0.go": `package main

type Struct struct {
	// Foo is a struct field.
	Foo string
}
`,
			},
		},
		// Label renamings.
		{
			ctxt: main(`package main
func f() {
loop:
	loop := 0
	go func() {
	loop:
		goto loop
	}()
	loop++
	goto loop
}
`),
			offset: "/go/src/main/0.go:#25", to: "loop2", // def of outer label "loop"
			want: map[string]string{
				"/go/src/main/0.go": `package main

func f() {
loop2:
	loop := 0
	go func() {
	loop:
		goto loop
	}()
	loop++
	goto loop2
}
`,
			},
		},
		{
			offset: "/go/src/main/0.go:#70", to: "loop2", // ref to inner label "loop"
			want: map[string]string{
				"/go/src/main/0.go": `package main

func f() {
loop:
	loop := 0
	go func() {
	loop2:
		goto loop2
	}()
	loop++
	goto loop
}
`,
			},
		},
		// Renaming of type used as embedded field.
		{
			ctxt: main(`package main

type T int
type U struct { T }

var _ = U{}.T
`),
			from: "main.T", to: "T2",
			want: map[string]string{
				"/go/src/main/0.go": `package main

type T2 int
type U struct{ T2 }

var _ = U{}.T2
`,
			},
		},
		// Renaming of embedded field.
		{
			ctxt: main(`package main

type T int
type U struct { T }

var _ = U{}.T
`),
			offset: "/go/src/main/0.go:#58", to: "T2", // T in "U{}.T"
			want: map[string]string{
				"/go/src/main/0.go": `package main

type T2 int
type U struct{ T2 }

var _ = U{}.T2
`,
			},
		},
		// Renaming of pointer embedded field.
		{
			ctxt: main(`package main

type T int
type U struct { *T }

var _ = U{}.T
`),
			offset: "/go/src/main/0.go:#59", to: "T2", // T in "U{}.T"
			want: map[string]string{
				"/go/src/main/0.go": `package main

type T2 int
type U struct{ *T2 }

var _ = U{}.T2
`,
			},
		},

		// Lexical scope tests.
		{
			ctxt: main(`package main

var y int

func f() {
	print(y)
	y := ""
	print(y)
}
`),
			from: "main.y", to: "x",
			want: map[string]string{
				"/go/src/main/0.go": `package main

var x int

func f() {
	print(x)
	y := ""
	print(y)
}
`,
			},
		},
		{
			from: "main.f::y", to: "x",
			want: map[string]string{
				"/go/src/main/0.go": `package main

var y int

func f() {
	print(y)
	x := ""
	print(x)
}
`,
			},
		},
		// Renaming of typeswitch vars (a corner case).
		{
			ctxt: main(`package main

func f(z interface{}) {
	switch y := z.(type) {
	case int:
		print(y)
	default:
		print(y)
	}
}
`),
			offset: "/go/src/main/0.go:#46", to: "x", // def of y
			want: map[string]string{
				"/go/src/main/0.go": `package main

func f(z interface{}) {
	switch x := z.(type) {
	case int:
		print(x)
	default:
		print(x)
	}
}
`},
		},
		{
			offset: "/go/src/main/0.go:#81", to: "x", // ref of y in case int
			want: map[string]string{
				"/go/src/main/0.go": `package main

func f(z interface{}) {
	switch x := z.(type) {
	case int:
		print(x)
	default:
		print(x)
	}
}
`},
		},
		{
			offset: "/go/src/main/0.go:#102", to: "x", // ref of y in default case
			want: map[string]string{
				"/go/src/main/0.go": `package main

func f(z interface{}) {
	switch x := z.(type) {
	case int:
		print(x)
	default:
		print(x)
	}
}
`},
		},

		// Renaming of embedded field that is a qualified reference.
		// (Regression test for bug 8924.)
		{
			ctxt: fakeContext(map[string][]string{
				"foo": {`package foo; type T int`},
				"main": {`package main

import "foo"

type _ struct{ *foo.T }
`},
			}),
			offset: "/go/src/main/0.go:#48", to: "U", // the "T" in *foo.T
			want: map[string]string{
				"/go/src/foo/0.go": `package foo

type U int
`,
				"/go/src/main/0.go": `package main

import "foo"

type _ struct{ *foo.U }
`,
			},
		},

		// Renaming of embedded field that is a qualified reference with the '-from' flag.
		// (Regression test for bug 12038.)
		{
			ctxt: fakeContext(map[string][]string{
				"foo": {`package foo; type T int`},
				"main": {`package main

import "foo"

type V struct{ *foo.T }
`},
			}),
			from: "(main.V).T", to: "U", // the "T" in *foo.T
			want: map[string]string{
				"/go/src/foo/0.go": `package foo

type U int
`,
				"/go/src/main/0.go": `package main

import "foo"

type V struct{ *foo.U }
`,
			},
		},
		{
			ctxt: fakeContext(map[string][]string{
				"foo": {`package foo; type T int`},
				"main": {`package main

import "foo"

type V struct{ foo.T }
`},
			}),
			from: "(main.V).T", to: "U", // the "T" in *foo.T
			want: map[string]string{
				"/go/src/foo/0.go": `package foo

type U int
`,
				"/go/src/main/0.go": `package main

import "foo"

type V struct{ foo.U }
`,
			},
		},

		// Interface method renaming.
		{
			ctxt: fakeContext(map[string][]string{
				"main": {`
package main
type I interface {
	f()
}
type J interface { f(); g() }
type A int
func (A) f()
type B int
func (B) f()
func (B) g()
type C int
func (C) f()
func (C) g()
var _, _ I = A(0), B(0)
var _, _ J = B(0), C(0)
`,
				},
			}),
			offset: "/go/src/main/0.go:#34", to: "F", // abstract method I.f
			want: map[string]string{
				"/go/src/main/0.go": `package main

type I interface {
	F()
}
type J interface {
	F()
	g()
}
type A int

func (A) F()

type B int

func (B) F()
func (B) g()

type C int

func (C) F()
func (C) g()

var _, _ I = A(0), B(0)
var _, _ J = B(0), C(0)
`,
			},
		},
		{
			offset: "/go/src/main/0.go:#59", to: "F", // abstract method J.f
			want: map[string]string{
				"/go/src/main/0.go": `package main

type I interface {
	F()
}
type J interface {
	F()
	g()
}
type A int

func (A) F()

type B int

func (B) F()
func (B) g()

type C int

func (C) F()
func (C) g()

var _, _ I = A(0), B(0)
var _, _ J = B(0), C(0)
`,
			},
		},
		{
			offset: "/go/src/main/0.go:#64", to: "G", // abstract method J.g
			want: map[string]string{
				"/go/src/main/0.go": `package main

type I interface {
	f()
}
type J interface {
	f()
	G()
}
type A int

func (A) f()

type B int

func (B) f()
func (B) G()

type C int

func (C) f()
func (C) G()

var _, _ I = A(0), B(0)
var _, _ J = B(0), C(0)
`,
			},
		},
		// Indirect coupling of I.f to C.f from D->I assignment and anonymous field of D.
		{
			ctxt: fakeContext(map[string][]string{
				"main": {`
package main
type I interface {
	f()
}
type C int
func (C) f()
type D struct{C}
var _ I = D{}
`,
				},
			}),
			offset: "/go/src/main/0.go:#34", to: "F", // abstract method I.f
			want: map[string]string{
				"/go/src/main/0.go": `package main

type I interface {
	F()
}
type C int

func (C) F()

type D struct{ C }

var _ I = D{}
`,
			},
		},
		// Interface embedded in struct.  No conflict if C need not satisfy I.
		{
			ctxt: fakeContext(map[string][]string{
				"main": {`
package main
type I interface {
	f()
}
type C struct{I}
func (C) g() int
var _ int = C{}.g()
`,
				},
			}),
			offset: "/go/src/main/0.go:#34", to: "g", // abstract method I.f
			want: map[string]string{
				"/go/src/main/0.go": `package main

type I interface {
	g()
}
type C struct{ I }

func (C) g() int

var _ int = C{}.g()
`,
			},
		},
		// A type assertion causes method coupling iff signatures match.
		{
			ctxt: fakeContext(map[string][]string{
				"main": {`package main
type I interface{
	f()
}
type J interface{
	f()
}
var _ = I(nil).(J)
`,
				},
			}),
			offset: "/go/src/main/0.go:#32", to: "g", // abstract method I.f
			want: map[string]string{
				"/go/src/main/0.go": `package main

type I interface {
	g()
}
type J interface {
	g()
}

var _ = I(nil).(J)
`,
			},
		},
		// Impossible type assertion: no method coupling.
		{
			ctxt: fakeContext(map[string][]string{
				"main": {`package main
type I interface{
	f()
}
type J interface{
	f()int
}
var _ = I(nil).(J)
`,
				},
			}),
			offset: "/go/src/main/0.go:#32", to: "g", // abstract method I.f
			want: map[string]string{
				"/go/src/main/0.go": `package main

type I interface {
	g()
}
type J interface {
	f() int
}

var _ = I(nil).(J)
`,
			},
		},
		// Impossible type assertion: no method coupling C.f<->J.f.
		{
			ctxt: fakeContext(map[string][]string{
				"main": {`package main
type I interface{
	f()
}
type C int
func (C) f()
type J interface{
	f()int
}
var _ = I(C(0)).(J)
`,
				},
			}),
			offset: "/go/src/main/0.go:#32", to: "g", // abstract method I.f
			want: map[string]string{
				"/go/src/main/0.go": `package main

type I interface {
	g()
}
type C int

func (C) g()

type J interface {
	f() int
}

var _ = I(C(0)).(J)
`,
			},
		},
		// Progress after "soft" type errors (Go issue 14596).
		{
			ctxt: fakeContext(map[string][]string{
				"main": {`package main

func main() {
	var unused, x int
	print(x)
}
`,
				},
			}),
			offset: "/go/src/main/0.go:#54", to: "y", // var x
			want: map[string]string{
				"/go/src/main/0.go": `package main

func main() {
	var unused, y int
	print(y)
}
`,
			},
		},
	} {
		if test.ctxt != nil {
			ctxt = test.ctxt
		}

		got := make(map[string]string)
		writeFile = func(filename string, content []byte) error {
			got[filepath.ToSlash(filename)] = string(content)
			return nil
		}

		err := Main(ctxt, test.offset, test.from, test.to)
		var prefix string
		if test.offset == "" {
			prefix = fmt.Sprintf("-from %q -to %q", test.from, test.to)
		} else {
			prefix = fmt.Sprintf("-offset %q -to %q", test.offset, test.to)
		}
		if err != nil {
			t.Errorf("%s: unexpected error: %s", prefix, err)
			continue
		}

		for file, wantContent := range test.want {
			gotContent, ok := got[file]
			delete(got, file)
			if !ok {
				t.Errorf("%s: file %s not rewritten", prefix, file)
				continue
			}
			if gotContent != wantContent {
				t.Errorf("%s: rewritten file %s does not match expectation; got <<<%s>>>\n"+
					"want <<<%s>>>", prefix, file, gotContent, wantContent)
			}
		}
		// got should now be empty
		for file := range got {
			t.Errorf("%s: unexpected rewrite of file %s", prefix, file)
		}
	}
}

func TestDiff(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skipf("diff tool non-existent for windows on builders")
	}
	if runtime.GOOS == "plan9" {
		t.Skipf("plan9 diff tool doesn't support -u flag")
	}

	defer func() {
		Diff = false
		stdout = os.Stdout
	}()
	Diff = true
	stdout = new(bytes.Buffer)

	if err := Main(&build.Default, "", `"golang.org/x/tools/refactor/rename".justHereForTestingDiff`, "Foo"); err != nil {
		t.Fatal(err)
	}

	// NB: there are tabs in the string literal!
	if !strings.Contains(stdout.(fmt.Stringer).String(), `
-func justHereForTestingDiff() {
-	justHereForTestingDiff()
+func Foo() {
+	Foo()
 }
`) {
		t.Errorf("unexpected diff:\n<<%s>>", stdout)
	}
}

func justHereForTestingDiff() {
	justHereForTestingDiff()
}

// ---------------------------------------------------------------------

// Simplifying wrapper around buildutil.FakeContext for packages whose
// filenames are sequentially numbered (%d.go).  pkgs maps a package
// import path to its list of file contents.
func fakeContext(pkgs map[string][]string) *build.Context {
	pkgs2 := make(map[string]map[string]string)
	for path, files := range pkgs {
		filemap := make(map[string]string)
		for i, contents := range files {
			filemap[fmt.Sprintf("%d.go", i)] = contents
		}
		pkgs2[path] = filemap
	}
	return buildutil.FakeContext(pkgs2)
}

// helper for single-file main packages with no imports.
func main(content string) *build.Context {
	return fakeContext(map[string][]string{"main": {content}})
}
