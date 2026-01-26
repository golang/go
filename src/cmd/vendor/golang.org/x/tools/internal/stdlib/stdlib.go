// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run generate.go

// Package stdlib provides a table of all exported symbols in the
// standard library, along with the version at which they first
// appeared. It also provides the import graph of std packages.
package stdlib

import (
	"fmt"
	"strings"
)

type Symbol struct {
	Name    string
	Kind    Kind
	Version Version // Go version that first included the symbol
	// Signature provides the type of a function (defined only for Kind=Func).
	// Imported types are denoted as pkg.T; pkg is not fully qualified.
	// TODO(adonovan): use an unambiguous encoding that is parseable.
	//
	// Example2:
	//    func[M ~map[K]V, K comparable, V any](m M) M
	//    func(fi fs.FileInfo, link string) (*Header, error)
	Signature string // if Kind == stdlib.Func
}

// A Kind indicates the kind of a symbol:
// function, variable, constant, type, and so on.
type Kind int8

const (
	Invalid Kind = iota // Example name:
	Type                // "Buffer"
	Func                // "Println"
	Var                 // "EOF"
	Const               // "Pi"
	Field               // "Point.X"
	Method              // "(*Buffer).Grow" or "(Reader).Read"
)

func (kind Kind) String() string {
	return [...]string{
		Invalid: "invalid",
		Type:    "type",
		Func:    "func",
		Var:     "var",
		Const:   "const",
		Field:   "field",
		Method:  "method",
	}[kind]
}

// A Version represents a version of Go of the form "go1.%d".
type Version int8

// String returns a version string of the form "go1.23", without allocating.
func (v Version) String() string { return versions[v] }

var versions [30]string // (increase constant as needed)

func init() {
	for i := range versions {
		versions[i] = fmt.Sprintf("go1.%d", i)
	}
}

// HasPackage reports whether the specified package path is part of
// the standard library's public API.
func HasPackage(path string) bool {
	_, ok := PackageSymbols[path]
	return ok
}

// SplitField splits the field symbol name into type and field
// components. It must be called only on Field symbols.
//
// Example: "File.Package" -> ("File", "Package")
func (sym *Symbol) SplitField() (typename, name string) {
	if sym.Kind != Field {
		panic("not a field")
	}
	typename, name, _ = strings.Cut(sym.Name, ".")
	return
}

// SplitMethod splits the method symbol name into pointer, receiver,
// and method components. It must be called only on Method symbols.
//
// Example: "(*Buffer).Grow" -> (true, "Buffer", "Grow")
func (sym *Symbol) SplitMethod() (ptr bool, recv, name string) {
	if sym.Kind != Method {
		panic("not a method")
	}
	recv, name, _ = strings.Cut(sym.Name, ".")
	recv = recv[len("(") : len(recv)-len(")")]
	ptr = recv[0] == '*'
	if ptr {
		recv = recv[len("*"):]
	}
	return
}
