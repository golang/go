// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	"bytes"
	"fmt"
	"go/parser"
	"go/token"
	"reflect"
	"sort"
	"testing"

	"golang.org/x/tools/go/expect"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/internal/typeparams"
)

// TestGenericBodies tests that bodies of generic functions and methods containing
// different constructs can be built in BuilderMode(0).
//
// Each test specifies the contents of package containing a single go file.
// Each call print(arg0, arg1, ...) to the builtin print function
// in ssa is correlated a comment at the end of the line of the form:
//
//	//@ types(a, b, c)
//
// where a, b and c are the types of the arguments to the print call
// serialized using go/types.Type.String().
// See x/tools/go/expect for details on the syntax.
func TestGenericBodies(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("TestGenericBodies requires type parameters")
	}
	for _, contents := range []string{
		`
		package p00

		func f(x int) {
			var i interface{}
			print(i, 0) //@ types("interface{}", int)
			print()     //@ types()
			print(x)    //@ types(int)
		}
		`,
		`
		package p01

		func f[T any](x T) {
			print(x) //@ types(T)
		}
		`,
		`
		package p02

		func f[T ~int]() {
			var x T
			print(x) //@ types(T)
		}
		`,
		`
		package p03

		func a[T ~[4]byte](x T) {
			for k, v := range x {
				print(x, k, v) //@ types(T, int, byte)
			}
		}
		func b[T ~*[4]byte](x T) {
			for k, v := range x {
				print(x, k, v) //@ types(T, int, byte)
			}
		}
		func c[T ~[]byte](x T) {
			for k, v := range x {
				print(x, k, v) //@ types(T, int, byte)
			}
		}
		func d[T ~string](x T) {
			for k, v := range x {
				print(x, k, v) //@ types(T, int, rune)
			}
		}
		func e[T ~map[int]string](x T) {
			for k, v := range x {
				print(x, k, v) //@ types(T, int, string)
			}
		}
		func f[T ~chan string](x T) {
			for v := range x {
				print(x, v) //@ types(T, string)
			}
		}

		func From() {
			type A [4]byte
			print(a[A]) //@ types("func(x p03.A)")

			type B *[4]byte
			print(b[B]) //@ types("func(x p03.B)")

			type C []byte
			print(c[C]) //@ types("func(x p03.C)")

			type D string
			print(d[D]) //@ types("func(x p03.D)")

			type E map[int]string
			print(e[E]) //@ types("func(x p03.E)")

			type F chan string
			print(f[F]) //@ types("func(x p03.F)")
		}
		`,
		`
		package p05

		func f[S any, T ~chan S](x T) {
			for v := range x {
				print(x, v) //@ types(T, S)
			}
		}

		func From() {
			type F chan string
			print(f[string, F]) //@ types("func(x p05.F)")
		}
		`,
		`
		package p06

		func fibonacci[T ~chan int](c, quit T) {
			x, y := 0, 1
			for {
				select {
				case c <- x:
					x, y = y, x+y
				case <-quit:
					print(c, quit, x, y) //@ types(T, T, int, int)
					return
				}
			}
		}
		func start[T ~chan int](c, quit T) {
			go func() {
				for i := 0; i < 10; i++ {
					print(<-c) //@ types(int)
				}
				quit <- 0
			}()
		}
		func From() {
			type F chan int
			c := make(F)
			quit := make(F)
			print(start[F], c, quit)     //@ types("func(c p06.F, quit p06.F)", "p06.F", "p06.F")
			print(fibonacci[F], c, quit) //@ types("func(c p06.F, quit p06.F)", "p06.F", "p06.F")
		}
		`,
		`
		package p07

		func f[T ~struct{ x int; y string }](i int) T {
			u := []T{ T{0, "lorem"},  T{1, "ipsum"}}
			return u[i]
		}
		func From() {
			type S struct{ x int; y string }
			print(f[S])     //@ types("func(i int) p07.S")
		}
		`,
		`
		package p08

		func f[T ~[4]int8](x T, l, h int) []int8 {
			return x[l:h]
		}
		func g[T ~*[4]int16](x T, l, h int) []int16 {
			return x[l:h]
		}
		func h[T ~[]int32](x T, l, h int) T {
			return x[l:h]
		}
		func From() {
			type F [4]int8
			type G *[4]int16
			type H []int32
			print(f[F](F{}, 0, 0))  //@ types("[]int8")
			print(g[G](nil, 0, 0)) //@ types("[]int16")
			print(h[H](nil, 0, 0)) //@ types("p08.H")
		}
		`,
		`
		package p09

		func h[E any, T ~[]E](x T, l, h int) []E {
			s := x[l:h]
			print(s) //@ types("T")
			return s
		}
		func From() {
			type H []int32
			print(h[int32, H](nil, 0, 0)) //@ types("[]int32")
		}
		`,
		`
		package p10

		// Test "make" builtin with different forms on core types and
		// when capacities are constants or variable.
		func h[E any, T ~[]E](m, n int) {
			print(make(T, 3))    //@ types(T)
			print(make(T, 3, 5)) //@ types(T)
			print(make(T, m))    //@ types(T)
			print(make(T, m, n)) //@ types(T)
		}
		func i[K comparable, E any, T ~map[K]E](m int) {
			print(make(T))    //@ types(T)
			print(make(T, 5)) //@ types(T)
			print(make(T, m)) //@ types(T)
		}
		func j[E any, T ~chan E](m int) {
			print(make(T))    //@ types(T)
			print(make(T, 6)) //@ types(T)
			print(make(T, m)) //@ types(T)
		}
		func From() {
			type H []int32
			h[int32, H](3, 4)
			type I map[int8]H
			i[int8, H, I](5)
			type J chan I
			j[I, J](6)
		}
		`,
		`
		package p11

		func h[T ~[4]int](x T) {
			print(len(x), cap(x)) //@ types(int, int)
		}
		func i[T ~[4]byte | []int | ~chan uint8](x T) {
			print(len(x), cap(x)) //@ types(int, int)
		}
		func j[T ~[4]int | any | map[string]int]() {
			print(new(T)) //@ types("*T")
		}
		func k[T ~[4]int | any | map[string]int](x T) {
			print(x) //@ types(T)
			panic(x)
		}
		`,
		`
		package p12

		func f[E any, F ~func() E](x F) {
			print(x, x()) //@ types(F, E)
		}
		func From() {
			type T func() int
			f[int, T](func() int { return 0 })
			f[int, func() int](func() int { return 1 })
		}
		`,
		`
		package p13

		func f[E any, M ~map[string]E](m M) {
			y, ok := m["lorem"]
			print(m, y, ok) //@ types(M, E, bool)
		}
		func From() {
			type O map[string][]int
			f(O{"lorem": []int{0, 1, 2, 3}})
		}
		`,
		`
		package p14

		func a[T interface{ []int64 | [5]int64 }](x T) int64 {
			print(x, x[2], x[3]) //@ types(T, int64, int64)
			x[2] = 5
			return x[3]
		}
		func b[T interface{ []byte | string }](x T) byte {
			print(x, x[3]) //@ types(T, byte)
			return x[3]
		}
		func c[T interface{ []byte }](x T) byte {
			print(x, x[2], x[3]) //@ types(T, byte, byte)
			x[2] = 'b'
			return x[3]
		}
		func d[T interface{ map[int]int64 }](x T) int64 {
			print(x, x[2], x[3]) //@ types(T, int64, int64)
			x[2] = 43
			return x[3]
		}
		func e[T ~string](t T) {
			print(t, t[0]) //@ types(T, uint8)
		}
		func f[T ~string|[]byte](t T) {
			print(t, t[0]) //@ types(T, uint8)
		}
		func g[T []byte](t T) {
			print(t, t[0]) //@ types(T, byte)
		}
		func h[T ~[4]int|[]int](t T) {
			print(t, t[0]) //@ types(T, int)
		}
		func i[T ~[4]int|*[4]int|[]int](t T) {
			print(t, t[0]) //@ types(T, int)
		}
		func j[T ~[4]int|*[4]int|[]int](t T) {
			print(t, &t[0]) //@ types(T, "*int")
		}
		`,
		`
		package p15

		type MyInt int
		type Other int
		type MyInterface interface{ foo() }

		// ChangeType tests
		func ct0(x int) { v := MyInt(x);  print(x, v) /*@ types(int, "p15.MyInt")*/ }
		func ct1[T MyInt | Other, S int ](x S) { v := T(x);  print(x, v) /*@ types(S, T)*/ }
		func ct2[T int, S MyInt | int ](x S) { v := T(x); print(x, v) /*@ types(S, T)*/ }
		func ct3[T MyInt | Other, S MyInt | int ](x S) { v := T(x) ; print(x, v) /*@ types(S, T)*/ }

		// Convert tests
		func co0[T int | int8](x MyInt) { v := T(x); print(x, v) /*@ types("p15.MyInt", T)*/}
		func co1[T int | int8](x T) { v := MyInt(x); print(x, v) /*@ types(T, "p15.MyInt")*/ }
		func co2[S, T int | int8](x T) { v := S(x); print(x, v) /*@ types(T, S)*/ }

		// MakeInterface tests
		func mi0[T MyInterface](x T) { v := MyInterface(x); print(x, v) /*@ types(T, "p15.MyInterface")*/ }

		// NewConst tests
		func nc0[T any]() { v := (*T)(nil); print(v) /*@ types("*T")*/}

		// SliceToArrayPointer
		func sl0[T *[4]int | *[2]int](x []int) { v := T(x); print(x, v) /*@ types("[]int", T)*/ }
		func sl1[T *[4]int | *[2]int, S []int](x S) { v := T(x); print(x, v) /*@ types(S, T)*/ }
		`,
		`
		package p16

		func c[T interface{ foo() string }](x T) {
			print(x, x.foo, x.foo())  /*@ types(T, "func() string", string)*/
		}
		`,
		`
		package p17

		func eq[T comparable](t T, i interface{}) bool {
			return t == i
		}
		`,
		// TODO(59983): investigate why writing g.c panics in (*FieldAddr).String.
		`
		package p18

		type S struct{ f int }
		func c[P *S]() []P { return []P{{f: 1}} }
		`,
		`
		package p19

		func sign[bytes []byte | string](s bytes) (bool, bool) {
			neg := false
			if len(s) > 0 && (s[0] == '-' || s[0] == '+') {
				neg = s[0] == '-'
				s = s[1:]
			}
			return !neg, len(s) > 0
		}
		`,
		`package p20

		func digits[bytes []byte | string](s bytes) bool {
			for _, c := range []byte(s) {
				if c < '0' || '9' < c {
					return false
				}
			}
			return true
		}
		`,
		`
		package p21

		type E interface{}

		func Foo[T E, PT interface{ *T }]() T {
			pt := PT(new(T))
			x := *pt
			print(x)  /*@ types(T)*/
			return x
		}
		`,
		`
		package p22

		func f[M any, PM *M](p PM) {
			var m M
			*p = m
			print(m)  /*@ types(M)*/
			print(p)  /*@ types(PM)*/
		}
		`,
		`
		package p23

		type A struct{int}
		func (*A) Marker() {}

		type B struct{string}
		func (*B) Marker() {}

		type C struct{float32}
		func (*C) Marker() {}

		func process[T interface {
			*A
			*B
			*C
			Marker()
		}](v T) {
			v.Marker()
			a := *(any(v).(*A)); print(a)  /*@ types("p23.A")*/
			b := *(any(v).(*B)); print(b)  /*@ types("p23.B")*/
			c := *(any(v).(*C)); print(c)  /*@ types("p23.C")*/
		}
		`,
		`
		package p24

		func a[T any](f func() [4]T) {
			x := len(f())
			print(x) /*@ types("int")*/
		}

		func b[T [4]any](f func() T) {
			x := len(f())
			print(x) /*@ types("int")*/
		}

		func c[T any](f func() *[4]T) {
			x := len(f())
			print(x) /*@ types("int")*/
		}

		func d[T *[4]any](f func() T) {
			x := len(f())
			print(x) /*@ types("int")*/
		}
		`,
		`
		package p25

		func a[T any]() {
			var f func() [4]T
			for i, v := range f() {
				print(i, v) /*@ types("int", "T")*/
			}
		}

		func b[T [4]any](f func() T) {
			for i, v := range f() {
				print(i, v) /*@ types("int", "any")*/
			}
		}

		func c[T any](f func() *[4]T) {
			for i, v := range f() {
				print(i, v) /*@ types("int", "T")*/
			}
		}

		func d[T *[4]any](f func() T) {
			for i, v := range f() {
				print(i, v) /*@ types("int", "any")*/
			}
		}
		`,
	} {
		contents := contents
		pkgname := packageName(t, contents)
		t.Run(pkgname, func(t *testing.T) {
			// Parse
			conf := loader.Config{ParserMode: parser.ParseComments}
			f, err := conf.ParseFile("file.go", contents)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			conf.CreateFromFiles(pkgname, f)

			// Load
			lprog, err := conf.Load()
			if err != nil {
				t.Fatalf("Load: %v", err)
			}

			// Create and build SSA
			prog := ssa.NewProgram(lprog.Fset, ssa.SanityCheckFunctions)
			for _, info := range lprog.AllPackages {
				if info.TransitivelyErrorFree {
					prog.CreatePackage(info.Pkg, info.Files, &info.Info, info.Importable)
				}
			}
			p := prog.Package(lprog.Package(pkgname).Pkg)
			p.Build()

			// Collect calls to the builtin print function.
			probes := make(map[*ssa.CallCommon]*ssa.Function)
			for _, mem := range p.Members {
				if fn, ok := mem.(*ssa.Function); ok {
					for _, bb := range fn.Blocks {
						for _, i := range bb.Instrs {
							if i, ok := i.(ssa.CallInstruction); ok {
								call := i.Common()
								if b, ok := call.Value.(*ssa.Builtin); ok && b.Name() == "print" {
									probes[i.Common()] = fn
								}
							}
						}
					}
				}
			}

			// Collect all notes in f, i.e. comments starting with "//@ types".
			notes, err := expect.ExtractGo(prog.Fset, f)
			if err != nil {
				t.Errorf("expect.ExtractGo: %v", err)
			}

			// Matches each probe with a note that has the same line.
			sameLine := func(x, y token.Pos) bool {
				xp := prog.Fset.Position(x)
				yp := prog.Fset.Position(y)
				return xp.Filename == yp.Filename && xp.Line == yp.Line
			}
			expectations := make(map[*ssa.CallCommon]*expect.Note)
			for call := range probes {
				var match *expect.Note
				for _, note := range notes {
					if note.Name == "types" && sameLine(call.Pos(), note.Pos) {
						match = note // first match is good enough.
						break
					}
				}
				if match != nil {
					expectations[call] = match
				} else {
					t.Errorf("Unmatched probe: %v", call)
				}
			}

			// Check each expectation.
			for call, note := range expectations {
				var args []string
				for _, a := range call.Args {
					args = append(args, a.Type().String())
				}
				if got, want := fmt.Sprint(args), fmt.Sprint(note.Args); got != want {
					t.Errorf("Arguments to print() were expected to be %q. got %q", want, got)
					logFunction(t, probes[call])
				}
			}
		})
	}
}

// TestInstructionString tests serializing instructions via Instruction.String().
func TestInstructionString(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("TestInstructionString requires type parameters")
	}
	// Tests (ssa.Instruction).String(). Instructions are from a single go file.
	// The Instructions tested are those that match a comment of the form:
	//
	//	//@ instrs(f, kind, strs...)
	//
	// where f is the name of the function, kind is the type of the instructions matched
	// within the function, and tests that the String() value for all of the instructions
	// matched of String() is strs (in some order).
	// See x/tools/go/expect for details on the syntax.

	const contents = `
	package p

	//@ instrs("f0", "*ssa.TypeAssert")
	//@ instrs("f0", "*ssa.Call", "print(nil:interface{}, 0:int)")
	func f0(x int) { // non-generic smoke test.
		var i interface{}
		print(i, 0)
	}

	//@ instrs("f1", "*ssa.Alloc", "local T (u)")
	//@ instrs("f1", "*ssa.FieldAddr", "&t0.x [#0]")
	func f1[T ~struct{ x string }]() T {
		u := T{"lorem"}
		return u
	}

	//@ instrs("f1b", "*ssa.Alloc", "new T (complit)")
	//@ instrs("f1b", "*ssa.FieldAddr", "&t0.x [#0]")
	func f1b[T ~struct{ x string }]() *T {
		u := &T{"lorem"}
		return u
	}

	//@ instrs("f2", "*ssa.TypeAssert", "typeassert t0.(interface{})")
	//@ instrs("f2", "*ssa.Call", "invoke x.foo()")
	func f2[T interface{ foo() string }](x T) {
		_ = x.foo
		_ = x.foo()
	}

	//@ instrs("f3", "*ssa.TypeAssert", "typeassert t0.(interface{})")
	//@ instrs("f3", "*ssa.Call", "invoke x.foo()")
	func f3[T interface{ foo() string; comparable }](x T) {
		_ = x.foo
		_ = x.foo()
	}

	//@ instrs("f4", "*ssa.BinOp", "t1 + 1:int", "t2 < 4:int")
	//@ instrs("f4", "*ssa.Call", "f()", "print(t2, t4)")
	func f4[T [4]string](f func() T) {
		for i, v := range f() {
			print(i, v)
		}
	}

	//@ instrs("f5", "*ssa.Call", "nil:func()()")
	func f5() {
		var f func()
		f()
	}

	type S struct{ f int }

	//@ instrs("f6", "*ssa.Alloc", "new [1]P (slicelit)", "new S (complit)")
	//@ instrs("f6", "*ssa.IndexAddr", "&t0[0:int]")
	//@ instrs("f6", "*ssa.FieldAddr", "&t2.f [#0]")
	func f6[P *S]() []P { return []P{{f: 1}} }

	//@ instrs("f7", "*ssa.Alloc", "local S (complit)")
	//@ instrs("f7", "*ssa.FieldAddr", "&t0.f [#0]")
	func f7[T any, S struct{f T}](x T) S { return S{f: x} }

	//@ instrs("f8", "*ssa.Alloc", "new [1]P (slicelit)", "new struct{f T} (complit)")
	//@ instrs("f8", "*ssa.IndexAddr", "&t0[0:int]")
	//@ instrs("f8", "*ssa.FieldAddr", "&t2.f [#0]")
	func f8[T any, P *struct{f T}](x T) []P { return []P{{f: x}} }

	//@ instrs("f9", "*ssa.Alloc", "new [1]PS (slicelit)", "new S (complit)")
	//@ instrs("f9", "*ssa.IndexAddr", "&t0[0:int]")
	//@ instrs("f9", "*ssa.FieldAddr", "&t2.f [#0]")
	func f9[T any, S struct{f T}, PS *S](x T) {
		_ = []PS{{f: x}}
	}

	//@ instrs("f10", "*ssa.FieldAddr", "&t0.x [#0]")
	//@ instrs("f10", "*ssa.Store", "*t0 = *new(T):T", "*t1 = 4:int")
	func f10[T ~struct{ x, y int }]() T {
		var u T
		u = T{x: 4}
		return u
	}

	//@ instrs("f11", "*ssa.FieldAddr", "&t1.y [#1]")
	//@ instrs("f11", "*ssa.Store", "*t1 = *new(T):T", "*t2 = 5:int")
	func f11[T ~struct{ x, y int }, PT *T]() PT {
		var u PT = new(T)
		*u = T{y: 5}
		return u
	}

	//@ instrs("f12", "*ssa.Alloc", "new struct{f T} (complit)")
	//@ instrs("f12", "*ssa.MakeMap", "make map[P]bool 1:int")
	func f12[T any, P *struct{f T}](x T) map[P]bool { return map[P]bool{{}: true} }

	//@ instrs("f13", "*ssa.IndexAddr", "&v[0:int]")
	//@ instrs("f13", "*ssa.Store", "*t0 = 7:int", "*v = *new(A):A")
	func f13[A [3]int, PA *A](v PA) {
		*v = A{7}
	}

	//@ instrs("f14", "*ssa.Call", "invoke t1.Set(0:int)")
	func f14[T any, PT interface {
		Set(int)
		*T
	}]() {
		var t T
		p := PT(&t)
		p.Set(0)
	}

	//@ instrs("f15", "*ssa.MakeClosure", "make closure (interface{Set(int); *T}).Set$bound [t1]")
	func f15[T any, PT interface {
		Set(int)
		*T
	}]() func(int) {
		var t T
		p := PT(&t)
		return p.Set
	}
	`

	// Parse
	conf := loader.Config{ParserMode: parser.ParseComments}
	const fname = "p.go"
	f, err := conf.ParseFile(fname, contents)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	conf.CreateFromFiles("p", f)

	// Load
	lprog, err := conf.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Create and build SSA
	prog := ssa.NewProgram(lprog.Fset, ssa.SanityCheckFunctions)
	for _, info := range lprog.AllPackages {
		if info.TransitivelyErrorFree {
			prog.CreatePackage(info.Pkg, info.Files, &info.Info, info.Importable)
		}
	}
	p := prog.Package(lprog.Package("p").Pkg)
	p.Build()

	// Collect all notes in f, i.e. comments starting with "//@ instr".
	notes, err := expect.ExtractGo(prog.Fset, f)
	if err != nil {
		t.Errorf("expect.ExtractGo: %v", err)
	}

	// Expectation is a {function, type string} -> {want, matches}
	// where matches is all Instructions.String() that match the key.
	// Each expecation is that some permutation of matches is wants.
	type expKey struct {
		function string
		kind     string
	}
	type expValue struct {
		wants   []string
		matches []string
	}
	expectations := make(map[expKey]*expValue)
	for _, note := range notes {
		if note.Name == "instrs" {
			if len(note.Args) < 2 {
				t.Error("Had @instrs annotation without at least 2 arguments")
				continue
			}
			fn, kind := fmt.Sprint(note.Args[0]), fmt.Sprint(note.Args[1])
			var wants []string
			for _, arg := range note.Args[2:] {
				wants = append(wants, fmt.Sprint(arg))
			}
			expectations[expKey{fn, kind}] = &expValue{wants, nil}
		}
	}

	// Collect all Instructions that match the expectations.
	for _, mem := range p.Members {
		if fn, ok := mem.(*ssa.Function); ok {
			for _, bb := range fn.Blocks {
				for _, i := range bb.Instrs {
					kind := fmt.Sprintf("%T", i)
					if e := expectations[expKey{fn.Name(), kind}]; e != nil {
						e.matches = append(e.matches, i.String())
					}
				}
			}
		}
	}

	// Check each expectation.
	for key, value := range expectations {
		fn, ok := p.Members[key.function].(*ssa.Function)
		if !ok {
			t.Errorf("Expectation on %s does not match a member in %s", key.function, p.Pkg.Name())
		}
		got, want := value.matches, value.wants
		sort.Strings(got)
		sort.Strings(want)
		if !reflect.DeepEqual(want, got) {
			t.Errorf("Within %s wanted instructions of kind %s: %q. got %q", key.function, key.kind, want, got)
			logFunction(t, fn)
		}
	}
}

// packageName is a test helper to extract the package name from a string
// containing the content of a go file.
func packageName(t testing.TB, content string) string {
	f, err := parser.ParseFile(token.NewFileSet(), "", content, parser.PackageClauseOnly)
	if err != nil {
		t.Fatalf("parsing the file %q failed with error: %s", content, err)
	}
	return f.Name.Name
}

func logFunction(t testing.TB, fn *ssa.Function) {
	// TODO: Consider adding a ssa.Function.GoString() so this can be logged to t via '%#v'.
	var buf bytes.Buffer
	ssa.WriteFunction(&buf, fn)
	t.Log(buf.String())
}
