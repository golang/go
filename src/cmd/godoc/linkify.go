// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements LinkifyText which introduces
// links for identifiers pointing to their declarations.
// The approach does not cover all cases because godoc
// doesn't have complete type information, but it's
// reasonably good for browsing.

package main

import (
	"fmt"
	"go/ast"
	"go/token"
	"io"
	"strconv"
)

// LinkifyText HTML-escapes source text and writes it to w.
// Identifiers that are in a "use" position (i.e., that are
// not being declared), are wrapped with HTML links pointing
// to the respective declaration, if possible. Comments are
// formatted the same way as with FormatText.
//
func LinkifyText(w io.Writer, text []byte, n ast.Node) {
	links := links(n)

	i := 0        // links index
	open := false // status of html tag
	linkWriter := func(w io.Writer, _ int, start bool) {
		// end tag
		if !start {
			if open {
				fmt.Fprintf(w, `</a>`)
				open = false
			}
			return
		}

		// start tag
		open = false
		if i < len(links) {
			switch info := links[i]; {
			case info.path != "" && info.ident == nil:
				// package path
				fmt.Fprintf(w, `<a href="/pkg/%s/">`, info.path)
				open = true
			case info.path != "" && info.ident != nil:
				// qualified identifier
				fmt.Fprintf(w, `<a href="/pkg/%s/#%s">`, info.path, info.ident.Name)
				open = true
			case info.path == "" && info.ident != nil:
				// locally declared identifier
				fmt.Fprintf(w, `<a href="#%s">`, info.ident.Name)
				open = true
			}
			i++
		}
	}

	idents := tokenSelection(text, token.IDENT)
	comments := tokenSelection(text, token.COMMENT)
	FormatSelections(w, text, linkWriter, idents, selectionTag, comments)
}

// A link describes the (HTML) link information for an identifier.
// The zero value of a link represents "no link".
//
type link struct {
	path  string
	ident *ast.Ident
}

// links returns the list of links for the identifiers used
// by node in the same order as they appear in the source.
//
func links(node ast.Node) (list []link) {
	defs := defs(node)

	// NOTE: We are expecting ast.Inspect to call the
	//       callback function in source text order.
	ast.Inspect(node, func(node ast.Node) bool {
		switch n := node.(type) {
		case *ast.Ident:
			info := link{}
			if !defs[n] {
				if n.Obj == nil && predeclared[n.Name] {
					info.path = builtinPkgPath
				}
				info.ident = n
			}
			list = append(list, info)
			return false
		case *ast.SelectorExpr:
			// Detect qualified identifiers of the form pkg.ident.
			// If anything fails we return true and collect individual
			// identifiers instead.
			if x, _ := n.X.(*ast.Ident); x != nil {
				// x must be a package for a qualified identifier
				if obj := x.Obj; obj != nil && obj.Kind == ast.Pkg {
					if spec, _ := obj.Decl.(*ast.ImportSpec); spec != nil {
						// spec.Path.Value is the import path
						if path, err := strconv.Unquote(spec.Path.Value); err == nil {
							// Register two links, one for the package
							// and one for the qualified identifier.
							info := link{path: path}
							list = append(list, info)
							info.ident = n.Sel
							list = append(list, info)
							return false
						}
					}
				}
			}
		}
		return true
	})

	return
}

// defs returns the set of identifiers that are declared ("defined") by node.
func defs(node ast.Node) map[*ast.Ident]bool {
	m := make(map[*ast.Ident]bool)

	ast.Inspect(node, func(node ast.Node) bool {
		switch n := node.(type) {
		case *ast.Field:
			for _, n := range n.Names {
				m[n] = true
			}
		case *ast.ImportSpec:
			if name := n.Name; name != nil {
				m[name] = true
			}
		case *ast.ValueSpec:
			for _, n := range n.Names {
				m[n] = true
			}
		case *ast.TypeSpec:
			m[n.Name] = true
		case *ast.FuncDecl:
			m[n.Name] = true
		case *ast.AssignStmt:
			// Short variable declarations only show up if we apply
			// this code to all source code (as opposed to exported
			// declarations only).
			if n.Tok == token.DEFINE {
				// Some of the lhs variables may be re-declared,
				// so technically they are not defs. We don't
				// care for now.
				for _, x := range n.Lhs {
					// Each lhs expression should be an
					// ident, but we are conservative and check.
					if n, _ := x.(*ast.Ident); n != nil {
						m[n] = true
					}
				}
			}
		}
		return true
	})

	return m
}

// The predeclared map represents the set of all predeclared identifiers.
var predeclared = map[string]bool{
	"bool":       true,
	"byte":       true,
	"complex64":  true,
	"complex128": true,
	"error":      true,
	"float32":    true,
	"float64":    true,
	"int":        true,
	"int8":       true,
	"int16":      true,
	"int32":      true,
	"int64":      true,
	"rune":       true,
	"string":     true,
	"uint":       true,
	"uint8":      true,
	"uint16":     true,
	"uint32":     true,
	"uint64":     true,
	"uintptr":    true,
	"true":       true,
	"false":      true,
	"iota":       true,
	"nil":        true,
	"append":     true,
	"cap":        true,
	"close":      true,
	"complex":    true,
	"copy":       true,
	"delete":     true,
	"imag":       true,
	"len":        true,
	"make":       true,
	"new":        true,
	"panic":      true,
	"print":      true,
	"println":    true,
	"real":       true,
	"recover":    true,
}
