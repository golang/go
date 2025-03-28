// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stdlib

// This file provides the API for the import graph of the standard library.
//
// Be aware that the compiler-generated code for every package
// implicitly depends on package "runtime" and a handful of others
// (see runtimePkgs in GOROOT/src/cmd/internal/objabi/pkgspecial.go).

import (
	"encoding/binary"
	"iter"
	"slices"
	"strings"
)

// Imports returns the sequence of packages directly imported by the
// named standard packages, in name order.
// The imports of an unknown package are the empty set.
//
// The graph is built into the application and may differ from the
// graph in the Go source tree being analyzed by the application.
func Imports(pkgs ...string) iter.Seq[string] {
	return func(yield func(string) bool) {
		for _, pkg := range pkgs {
			if i, ok := find(pkg); ok {
				var depIndex uint64
				for data := []byte(deps[i].deps); len(data) > 0; {
					delta, n := binary.Uvarint(data)
					depIndex += delta
					if !yield(deps[depIndex].name) {
						return
					}
					data = data[n:]
				}
			}
		}
	}
}

// Dependencies returns the set of all dependencies of the named
// standard packages, including the initial package,
// in a deterministic topological order.
// The dependencies of an unknown package are the empty set.
//
// The graph is built into the application and may differ from the
// graph in the Go source tree being analyzed by the application.
func Dependencies(pkgs ...string) iter.Seq[string] {
	return func(yield func(string) bool) {
		for _, pkg := range pkgs {
			if i, ok := find(pkg); ok {
				var seen [1 + len(deps)/8]byte // bit set of seen packages
				var visit func(i int) bool
				visit = func(i int) bool {
					bit := byte(1) << (i % 8)
					if seen[i/8]&bit == 0 {
						seen[i/8] |= bit
						var depIndex uint64
						for data := []byte(deps[i].deps); len(data) > 0; {
							delta, n := binary.Uvarint(data)
							depIndex += delta
							if !visit(int(depIndex)) {
								return false
							}
							data = data[n:]
						}
						if !yield(deps[i].name) {
							return false
						}
					}
					return true
				}
				if !visit(i) {
					return
				}
			}
		}
	}
}

// find returns the index of pkg in the deps table.
func find(pkg string) (int, bool) {
	return slices.BinarySearchFunc(deps[:], pkg, func(p pkginfo, n string) int {
		return strings.Compare(p.name, n)
	})
}
