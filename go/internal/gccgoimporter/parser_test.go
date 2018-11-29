// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Except for this comment, this file is a verbatim copy of the file
// with the same name in $GOROOT/src/go/internal/gccgoimporter.

package gccgoimporter

import (
	"bytes"
	"go/types"
	"strings"
	"testing"
	"text/scanner"
)

var typeParserTests = []struct {
	id, typ, want, underlying, methods string
}{
	{id: "foo", typ: "<type -1>", want: "int8"},
	{id: "foo", typ: "<type 1 *<type -19>>", want: "*error"},
	{id: "foo", typ: "<type 1 *any>", want: "unsafe.Pointer"},
	{id: "foo", typ: "<type 1 \"Bar\" <type 2 *<type 1>>>", want: "foo.Bar", underlying: "*foo.Bar"},
	{id: "foo", typ: "<type 1 \"bar.Foo\" \"bar\" <type -1>\nfunc (? <type 1>) M ();\n>", want: "bar.Foo", underlying: "int8", methods: "func (bar.Foo).M()"},
	{id: "foo", typ: "<type 1 \".bar.foo\" \"bar\" <type -1>>", want: "bar.foo", underlying: "int8"},
	{id: "foo", typ: "<type 1 []<type -1>>", want: "[]int8"},
	{id: "foo", typ: "<type 1 [42]<type -1>>", want: "[42]int8"},
	{id: "foo", typ: "<type 1 map [<type -1>] <type -2>>", want: "map[int8]int16"},
	{id: "foo", typ: "<type 1 chan <type -1>>", want: "chan int8"},
	{id: "foo", typ: "<type 1 chan <- <type -1>>", want: "<-chan int8"},
	{id: "foo", typ: "<type 1 chan -< <type -1>>", want: "chan<- int8"},
	{id: "foo", typ: "<type 1 struct { I8 <type -1>; I16 <type -2> \"i16\"; }>", want: "struct{I8 int8; I16 int16 \"i16\"}"},
	{id: "foo", typ: "<type 1 interface { Foo (a <type -1>, b <type -2>) <type -1>; Bar (? <type -2>, ? ...<type -1>) (? <type -2>, ? <type -1>); Baz (); }>", want: "interface{Bar(int16, ...int8) (int16, int8); Baz(); Foo(a int8, b int16) int8}"},
	{id: "foo", typ: "<type 1 (? <type -1>) <type -2>>", want: "func(int8) int16"},
}

func TestTypeParser(t *testing.T) {
	for _, test := range typeParserTests {
		var p parser
		p.init("test.gox", strings.NewReader(test.typ), make(map[string]*types.Package))
		p.version = "v2"
		p.pkgname = test.id
		p.pkgpath = test.id
		p.maybeCreatePackage()
		typ := p.parseType(p.pkg)

		if p.tok != scanner.EOF {
			t.Errorf("expected full parse, stopped at %q", p.lit)
		}

		// interfaces must be explicitly completed
		if ityp, _ := typ.(*types.Interface); ityp != nil {
			ityp.Complete()
		}

		got := typ.String()
		if got != test.want {
			t.Errorf("got type %q, expected %q", got, test.want)
		}

		if test.underlying != "" {
			underlying := typ.Underlying().String()
			if underlying != test.underlying {
				t.Errorf("got underlying type %q, expected %q", underlying, test.underlying)
			}
		}

		if test.methods != "" {
			nt := typ.(*types.Named)
			var buf bytes.Buffer
			for i := 0; i != nt.NumMethods(); i++ {
				buf.WriteString(nt.Method(i).String())
			}
			methods := buf.String()
			if methods != test.methods {
				t.Errorf("got methods %q, expected %q", methods, test.methods)
			}
		}
	}
}
