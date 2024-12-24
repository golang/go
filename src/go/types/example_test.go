// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Only run where builders (build.golang.org) have
// access to compiled packages for import.
//
//go:build !android && !ios && !js && !wasip1

package types_test

// This file shows examples of basic usage of the go/types API.
//
// To locate a Go package, use (*go/build.Context).Import.
// To load, parse, and type-check a complete Go program
// from source, use golang.org/x/tools/go/loader.

import (
	"fmt"
	"go/ast"
	"go/format"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"log"
	"regexp"
	"slices"
	"strings"
)

// ExampleScope prints the tree of Scopes of a package created from a
// set of parsed files.
func ExampleScope() {
	// Parse the source files for a package.
	fset := token.NewFileSet()
	var files []*ast.File
	for _, src := range []string{
		`package main
import "fmt"
func main() {
	freezing := FToC(-18)
	fmt.Println(freezing, Boiling) }
`,
		`package main
import "fmt"
type Celsius float64
func (c Celsius) String() string { return fmt.Sprintf("%g°C", c) }
func FToC(f float64) Celsius { return Celsius(f - 32 / 9 * 5) }
const Boiling Celsius = 100
func Unused() { {}; {{ var x int; _ = x }} } // make sure empty block scopes get printed
`,
	} {
		files = append(files, mustParse(fset, src))
	}

	// Type-check a package consisting of these files.
	// Type information for the imported "fmt" package
	// comes from $GOROOT/pkg/$GOOS_$GOOARCH/fmt.a.
	conf := types.Config{Importer: importer.Default()}
	pkg, err := conf.Check("temperature", fset, files, nil)
	if err != nil {
		log.Fatal(err)
	}

	// Print the tree of scopes.
	// For determinism, we redact addresses.
	var buf strings.Builder
	pkg.Scope().WriteTo(&buf, 0, true)
	rx := regexp.MustCompile(` 0x[a-fA-F\d]*`)
	fmt.Println(rx.ReplaceAllString(buf.String(), ""))

	// Output:
	// package "temperature" scope {
	// .  const temperature.Boiling temperature.Celsius
	// .  type temperature.Celsius float64
	// .  func temperature.FToC(f float64) temperature.Celsius
	// .  func temperature.Unused()
	// .  func temperature.main()
	// .  main scope {
	// .  .  package fmt
	// .  .  function scope {
	// .  .  .  var freezing temperature.Celsius
	// .  .  }
	// .  }
	// .  main scope {
	// .  .  package fmt
	// .  .  function scope {
	// .  .  .  var c temperature.Celsius
	// .  .  }
	// .  .  function scope {
	// .  .  .  var f float64
	// .  .  }
	// .  .  function scope {
	// .  .  .  block scope {
	// .  .  .  }
	// .  .  .  block scope {
	// .  .  .  .  block scope {
	// .  .  .  .  .  var x int
	// .  .  .  .  }
	// .  .  .  }
	// .  .  }
	// .  }
	// }
}

// ExampleMethodSet prints the method sets of various types.
func ExampleMethodSet() {
	// Parse a single source file.
	const input = `
package temperature
import "fmt"
type Celsius float64
func (c Celsius) String() string  { return fmt.Sprintf("%g°C", c) }
func (c *Celsius) SetF(f float64) { *c = Celsius(f - 32 / 9 * 5) }

type S struct { I; m int }
type I interface { m() byte }
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "celsius.go", input, 0)
	if err != nil {
		log.Fatal(err)
	}

	// Type-check a package consisting of this file.
	// Type information for the imported packages
	// comes from $GOROOT/pkg/$GOOS_$GOOARCH/fmt.a.
	conf := types.Config{Importer: importer.Default()}
	pkg, err := conf.Check("temperature", fset, []*ast.File{f}, nil)
	if err != nil {
		log.Fatal(err)
	}

	// Print the method sets of Celsius and *Celsius.
	celsius := pkg.Scope().Lookup("Celsius").Type()
	for _, t := range []types.Type{celsius, types.NewPointer(celsius)} {
		fmt.Printf("Method set of %s:\n", t)
		for m := range types.NewMethodSet(t).Methods() {
			fmt.Println(m)
		}
		fmt.Println()
	}

	// Print the method set of S.
	styp := pkg.Scope().Lookup("S").Type()
	fmt.Printf("Method set of %s:\n", styp)
	fmt.Println(types.NewMethodSet(styp))

	// Output:
	// Method set of temperature.Celsius:
	// method (temperature.Celsius) String() string
	//
	// Method set of *temperature.Celsius:
	// method (*temperature.Celsius) SetF(f float64)
	// method (*temperature.Celsius) String() string
	//
	// Method set of temperature.S:
	// MethodSet {}
}

// ExampleInfo prints various facts recorded by the type checker in a
// types.Info struct: definitions of and references to each named object,
// and the type, value, and mode of every expression in the package.
func ExampleInfo() {
	// Parse a single source file.
	const input = `
package fib

type S string

var a, b, c = len(b), S(c), "hello"

func fib(x int) int {
	if x < 2 {
		return x
	}
	return fib(x-1) - fib(x-2)
}`
	// We need a specific fileset in this test below for positions.
	// Cannot use typecheck helper.
	fset := token.NewFileSet()
	f := mustParse(fset, input)

	// Type-check the package.
	// We create an empty map for each kind of input
	// we're interested in, and Check populates them.
	info := types.Info{
		Types: make(map[ast.Expr]types.TypeAndValue),
		Defs:  make(map[*ast.Ident]types.Object),
		Uses:  make(map[*ast.Ident]types.Object),
	}
	var conf types.Config
	pkg, err := conf.Check("fib", fset, []*ast.File{f}, &info)
	if err != nil {
		log.Fatal(err)
	}

	// Print package-level variables in initialization order.
	fmt.Printf("InitOrder: %v\n\n", info.InitOrder)

	// For each named object, print the line and
	// column of its definition and each of its uses.
	fmt.Println("Defs and Uses of each named object:")
	usesByObj := make(map[types.Object][]string)
	for id, obj := range info.Uses {
		posn := fset.Position(id.Pos())
		lineCol := fmt.Sprintf("%d:%d", posn.Line, posn.Column)
		usesByObj[obj] = append(usesByObj[obj], lineCol)
	}
	var items []string
	for obj, uses := range usesByObj {
		slices.Sort(uses)
		item := fmt.Sprintf("%s:\n  defined at %s\n  used at %s",
			types.ObjectString(obj, types.RelativeTo(pkg)),
			fset.Position(obj.Pos()),
			strings.Join(uses, ", "))
		items = append(items, item)
	}
	slices.Sort(items) // sort by line:col, in effect
	fmt.Println(strings.Join(items, "\n"))
	fmt.Println()

	fmt.Println("Types and Values of each expression:")
	items = nil
	for expr, tv := range info.Types {
		var buf strings.Builder
		posn := fset.Position(expr.Pos())
		tvstr := tv.Type.String()
		if tv.Value != nil {
			tvstr += " = " + tv.Value.String()
		}
		// line:col | expr | mode : type = value
		fmt.Fprintf(&buf, "%2d:%2d | %-19s | %-7s : %s",
			posn.Line, posn.Column, exprString(fset, expr),
			mode(tv), tvstr)
		items = append(items, buf.String())
	}
	slices.Sort(items)
	fmt.Println(strings.Join(items, "\n"))

	// Output:
	// InitOrder: [c = "hello" b = S(c) a = len(b)]
	//
	// Defs and Uses of each named object:
	// builtin len:
	//   defined at -
	//   used at 6:15
	// func fib(x int) int:
	//   defined at fib:8:6
	//   used at 12:20, 12:9
	// type S string:
	//   defined at fib:4:6
	//   used at 6:23
	// type int:
	//   defined at -
	//   used at 8:12, 8:17
	// type string:
	//   defined at -
	//   used at 4:8
	// var b S:
	//   defined at fib:6:8
	//   used at 6:19
	// var c string:
	//   defined at fib:6:11
	//   used at 6:25
	// var x int:
	//   defined at fib:8:10
	//   used at 10:10, 12:13, 12:24, 9:5
	//
	// Types and Values of each expression:
	//  4: 8 | string              | type    : string
	//  6:15 | len                 | builtin : func(fib.S) int
	//  6:15 | len(b)              | value   : int
	//  6:19 | b                   | var     : fib.S
	//  6:23 | S                   | type    : fib.S
	//  6:23 | S(c)                | value   : fib.S
	//  6:25 | c                   | var     : string
	//  6:29 | "hello"             | value   : string = "hello"
	//  8:12 | int                 | type    : int
	//  8:17 | int                 | type    : int
	//  9: 5 | x                   | var     : int
	//  9: 5 | x < 2               | value   : untyped bool
	//  9: 9 | 2                   | value   : int = 2
	// 10:10 | x                   | var     : int
	// 12: 9 | fib                 | value   : func(x int) int
	// 12: 9 | fib(x - 1)          | value   : int
	// 12: 9 | fib(x-1) - fib(x-2) | value   : int
	// 12:13 | x                   | var     : int
	// 12:13 | x - 1               | value   : int
	// 12:15 | 1                   | value   : int = 1
	// 12:20 | fib                 | value   : func(x int) int
	// 12:20 | fib(x - 2)          | value   : int
	// 12:24 | x                   | var     : int
	// 12:24 | x - 2               | value   : int
	// 12:26 | 2                   | value   : int = 2
}

func mode(tv types.TypeAndValue) string {
	switch {
	case tv.IsVoid():
		return "void"
	case tv.IsType():
		return "type"
	case tv.IsBuiltin():
		return "builtin"
	case tv.IsNil():
		return "nil"
	case tv.Assignable():
		if tv.Addressable() {
			return "var"
		}
		return "mapindex"
	case tv.IsValue():
		return "value"
	default:
		return "unknown"
	}
}

func exprString(fset *token.FileSet, expr ast.Expr) string {
	var buf strings.Builder
	format.Node(&buf, fset, expr)
	return buf.String()
}
