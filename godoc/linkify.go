// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements LinkifyText which introduces
// links for identifiers pointing to their declarations.
// The approach does not cover all cases because godoc
// doesn't have complete type information, but it's
// reasonably good for browsing.

package godoc

import (
	"fmt"
	"go/ast"
	"go/doc"
	"go/token"
	"io"
	"strconv"

	"golang.org/x/tools/internal/typeparams"
)

// LinkifyText HTML-escapes source text and writes it to w.
// Identifiers that are in a "use" position (i.e., that are
// not being declared), are wrapped with HTML links pointing
// to the respective declaration, if possible. Comments are
// formatted the same way as with FormatText.
//
func LinkifyText(w io.Writer, text []byte, n ast.Node) {
	links := linksFor(n)

	i := 0     // links index
	prev := "" // prev HTML tag
	linkWriter := func(w io.Writer, _ int, start bool) {
		// end tag
		if !start {
			if prev != "" {
				fmt.Fprintf(w, `</%s>`, prev)
				prev = ""
			}
			return
		}

		// start tag
		prev = ""
		if i < len(links) {
			switch info := links[i]; {
			case info.path != "" && info.name == "":
				// package path
				fmt.Fprintf(w, `<a href="/pkg/%s/">`, info.path)
				prev = "a"
			case info.path != "" && info.name != "":
				// qualified identifier
				fmt.Fprintf(w, `<a href="/pkg/%s/#%s">`, info.path, info.name)
				prev = "a"
			case info.path == "" && info.name != "":
				// local identifier
				if info.isVal {
					fmt.Fprintf(w, `<span id="%s">`, info.name)
					prev = "span"
				} else if ast.IsExported(info.name) {
					fmt.Fprintf(w, `<a href="#%s">`, info.name)
					prev = "a"
				}
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
	path, name string // package path, identifier name
	isVal      bool   // identifier is defined in a const or var declaration
}

// linksFor returns the list of links for the identifiers used
// by node in the same order as they appear in the source.
//
func linksFor(node ast.Node) (links []link) {
	// linkMap tracks link information for each ast.Ident node. Entries may
	// be created out of source order (for example, when we visit a parent
	// definition node). These links are appended to the returned slice when
	// their ast.Ident nodes are visited.
	linkMap := make(map[*ast.Ident]link)

	typeParams := make(map[string]bool)

	ast.Inspect(node, func(node ast.Node) bool {
		switch n := node.(type) {
		case *ast.Field:
			for _, n := range n.Names {
				linkMap[n] = link{}
			}
		case *ast.ImportSpec:
			if name := n.Name; name != nil {
				linkMap[name] = link{}
			}
		case *ast.ValueSpec:
			for _, n := range n.Names {
				linkMap[n] = link{name: n.Name, isVal: true}
			}
		case *ast.FuncDecl:
			linkMap[n.Name] = link{}
			if n.Recv != nil {
				recv := n.Recv.List[0].Type
				if r, isstar := recv.(*ast.StarExpr); isstar {
					recv = r.X
				}
				switch x := recv.(type) {
				case *ast.IndexExpr:
					if ident, _ := x.Index.(*ast.Ident); ident != nil {
						typeParams[ident.Name] = true
					}
				case *typeparams.IndexListExpr:
					for _, index := range x.Indices {
						if ident, _ := index.(*ast.Ident); ident != nil {
							typeParams[ident.Name] = true
						}
					}
				}
			}
		case *ast.TypeSpec:
			linkMap[n.Name] = link{}
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
						linkMap[n] = link{isVal: true}
					}
				}
			}
		case *ast.SelectorExpr:
			// Detect qualified identifiers of the form pkg.ident.
			// If anything fails we return true and collect individual
			// identifiers instead.
			if x, _ := n.X.(*ast.Ident); x != nil {
				// Create links only if x is a qualified identifier.
				if obj := x.Obj; obj != nil && obj.Kind == ast.Pkg {
					if spec, _ := obj.Decl.(*ast.ImportSpec); spec != nil {
						// spec.Path.Value is the import path
						if path, err := strconv.Unquote(spec.Path.Value); err == nil {
							// Register two links, one for the package
							// and one for the qualified identifier.
							linkMap[x] = link{path: path}
							linkMap[n.Sel] = link{path: path, name: n.Sel.Name}
						}
					}
				}
			}
		case *ast.CompositeLit:
			// Detect field names within composite literals. These links should
			// be prefixed by the type name.
			fieldPath := ""
			prefix := ""
			switch typ := n.Type.(type) {
			case *ast.Ident:
				prefix = typ.Name + "."
			case *ast.SelectorExpr:
				if x, _ := typ.X.(*ast.Ident); x != nil {
					// Create links only if x is a qualified identifier.
					if obj := x.Obj; obj != nil && obj.Kind == ast.Pkg {
						if spec, _ := obj.Decl.(*ast.ImportSpec); spec != nil {
							// spec.Path.Value is the import path
							if path, err := strconv.Unquote(spec.Path.Value); err == nil {
								// Register two links, one for the package
								// and one for the qualified identifier.
								linkMap[x] = link{path: path}
								linkMap[typ.Sel] = link{path: path, name: typ.Sel.Name}
								fieldPath = path
								prefix = typ.Sel.Name + "."
							}
						}
					}
				}
			}
			for _, e := range n.Elts {
				if kv, ok := e.(*ast.KeyValueExpr); ok {
					if k, ok := kv.Key.(*ast.Ident); ok {
						// Note: there is some syntactic ambiguity here. We cannot determine
						// if this is a struct literal or a map literal without type
						// information. We assume struct literal.
						name := prefix + k.Name
						linkMap[k] = link{path: fieldPath, name: name}
					}
				}
			}
		case *ast.Ident:
			if l, ok := linkMap[n]; ok {
				links = append(links, l)
			} else {
				l := link{name: n.Name}
				if n.Obj == nil {
					if doc.IsPredeclared(n.Name) {
						l.path = builtinPkgPath
					} else {
						if typeParams[n.Name] {
							// If a type parameter was declared then do not generate a link.
							// Doing this is necessary because type parameter identifiers do not
							// have their Decl recorded sometimes, see
							// https://golang.org/issue/50956.
							l = link{}
						}
					}
				} else {
					if n.Obj.Kind == ast.Typ {
						if _, isfield := n.Obj.Decl.(*ast.Field); isfield {
							// If an identifier is a type declared in a field assume it is a type
							// parameter and do not generate a link.
							l = link{}
						}
					}
				}
				links = append(links, l)
			}
		}
		return true
	})
	return
}
