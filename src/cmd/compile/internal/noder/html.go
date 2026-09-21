// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"html"
	"os"
	"path/filepath"
	"reflect"
	"strings"
)

// An HTMLWriter dumps syntax nodes to multicolumn HTML, similar to what the
// ssa backend does for GOSSAFUNC.
type HTMLWriter struct {
	ir.HTMLWriterBase

	Decl *syntax.FuncDecl
	pkg  *types2.Package
	file *syntax.File
	info *types2.Info
}

func NewHTMLWriter(pkg *types2.Package, file *syntax.File, info *types2.Info, path string, decl *syntax.FuncDecl, cfgMask string) *HTMLWriter {
	path = strings.ReplaceAll(path, "/", string(filepath.Separator))
	out, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		panic(err)
	}
	reportPath := path
	if !filepath.IsAbs(reportPath) {
		pwd, err := os.Getwd()
		if err != nil {
			panic(err)
		}
		reportPath = filepath.Join(pwd, path)
	}
	h := HTMLWriter{
		pkg:  pkg,
		file: file,
		info: info,
		Decl: decl,
	}
	h.Init(out, reportPath, h.DeclHTML)
	h.start()
	return &h
}

func (w *HTMLWriter) pkgFuncName() string {
	p := w.pkg.Path()
	if p == "" {
		p = base.Ctxt.Pkgpath
	}
	return p + "." + w.Decl.Name.Value
}

func (w *HTMLWriter) start() {
	if w == nil {
		return
	}
	escName := html.EscapeString(w.pkgFuncName())
	w.Print("<!DOCTYPE html>")
	w.Print("<html>")
	w.Printf(`<head>
<meta name="generator" content="AST display for %s">
<meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
%s
%s
<title>AST display for %s</title>
</head>`, escName, ir.CSS, ir.JS("checked", "rangefunc"), escName)
	w.Print("<body>")
	w.Print("<h1>")
	w.Print(escName)
	w.Print("</h1>")
	w.Print(`
<a href="#" onclick="toggle_visibility('help');return false;" id="helplink">help</a>
<div id="help">

<p>
Click anywhere on a node (with "cell" cursor) to outline a node and all of its subtrees.
</p>
<p>
Click on a name (with "crosshair" cursor) to highlight every occurrence of a name.
(Note that all the name nodes are the same node, so those also all outline together).
</p>
<p>
Click on a file, line, or column (with "crosshair" cursor) to highlight positions
in that file, at that file:line, or at that file:line:column, respectively.<br>Inlined
locations are not treated as a single location, but as a sequence of locations that
can be independently highlighted.
</p>
<p>
Click on a ` + ir.DownArrow + ` to collapse a subtree, or on a ` + ir.RightArrow + ` to expand a subtree.
</p>
<p>
Non-tree attributes, like scope and type lookups, are displayed in italics.  Those may
also be clicked to highlight identity relationships within and between phases.
</p>

</div>
<label for="dark-mode-button" style="margin-left: 15px; cursor: pointer;">darkmode</label>
<input type="checkbox" onclick="toggleDarkMode();" id="dark-mode-button" style="cursor: pointer" />
`)
	w.Print("<table>")
	w.Print("<tr>")
}

func (w *HTMLWriter) DeclHTML(phase string) func() {
	return func() {
		w.Print("<pre>") // use pre for formatting to preserve indentation
		w.dumpScopeHTML(w.pkg.Scope(), 1, false)
		w.dumpScopeHTML(w.info.Scopes[w.file], 1, false)
		w.dumpNodeHTML(w.Decl, 1, "")
		w.Print("</pre>")
	}
}

func (h *HTMLWriter) dumpNodesHTML(list []syntax.Node, depth int) {
	if len(list) == 0 {
		h.Print(" <nil>")
		return
	}

	for _, n := range list {
		h.dumpNodeHTML(n, depth, "")
	}
}

func isValid(t types2.Type) bool {
	return t != nil && types2.Unalias(t) != types2.Typ[types2.Invalid]
}

const indentString = ".  "

func (w *HTMLWriter) indent(n int) {
	w.Print("\n")
	for range n {
		w.Print(indentString)
	}
}

// indentForToggle prints indentation to w.
func (h *HTMLWriter) indentForToggle(depth int, hasChildren bool) {
	h.Print("\n")
	if depth == 0 {
		return
	}
	for range depth - 1 {
		h.Print(indentString)
	}
	if hasChildren {
		// Remove 2 spaces, which have similar rendered width to
		// leading ir.DownArrow and trailing space.
		h.Print(indentString[:len(indentString)-2])
	} else {
		h.Print(indentString)
	}
}

// dumpScopeHTML writes a string representation of the scope to w,
// with the scope elements sorted by name.
// The level of indentation is controlled by n >= 0, with
// n == 0 for no indentation.
// If recurse is set, it also writes nested (children) scopes.
func (h *HTMLWriter) dumpScopeHTML(s *types2.Scope, depth int, recur bool) {
	hasChildren := true // TODO detect empty scopes
	h.indentForToggle(depth, hasChildren)
	if hasChildren {
		h.Printf("<span class=\"n%d scope\">", h.CanonId(s))
		defer h.Printf("</span>")
		// NOTE TRAILING SPACE after </span>! See indentForToggle above.
		h.Print(`<span class="toggle" onclick="toggle_node(this)">` + ir.DownArrow + `</span> `)
	}
	h.Printf("scope %s %p", html.EscapeString(s.Comment()), s)

	if hasChildren {
		h.Print(`<span class="node-body">`)
		defer h.Print(`</span>`)

		for _, name := range s.Names() {
			obj := s.Lookup(name)
			h.dumpOutlineNodeHTML(depth+1, "", obj)
		}

		if recur {
			for i := range s.NumChildren() {
				c := s.Child(i)
				h.dumpScopeHTML(c, depth+1, recur)
			}
		}
	}
}

func (h *HTMLWriter) dumpOutlineNodeHTML(depth int, pfx string, obj fmt.Stringer) {
	h.indentForToggle(depth, false)
	h.Printf("<span class=\"n%d outline-node\" style=\"font-style: italic;\">%s%s</span>",
		h.CanonId(obj), pfx, html.EscapeString(obj.String()))
}

func (h *HTMLWriter) dumpNodeHTML(n syntax.Node, depth int, prefix string) {
	hasChildren := h.nodeHasChildren(n)
	h.indentForToggle(depth, hasChildren)

	if depth > 40 {
		h.Print("...")
		return
	}

	if n == nil {
		h.Print("NilSyntaxNode")
		return
	}

	h.Printf("<span class=\"n%d outline-node\">", h.CanonId(n))
	defer h.Printf("</span>")

	if hasChildren {
		// NOTE TRAILING SPACE after </span>! See indentForToggle above.
		h.Print(`<span class="toggle" onclick="toggle_node(this)">` + ir.DownArrow + `</span> `) // NOTE TRAILING SPACE after </span>!
	}

	opName := strings.TrimPrefix(fmt.Sprintf("%T", n), "*syntax.")

	if prefix != "" {
		h.Printf("%s", html.EscapeString(prefix))
	}

	switch n := n.(type) {
	case *syntax.BasicLit:
		h.Printf("%s-%v", opName, html.EscapeString(n.Value))
		h.dumpNodeHeaderHTML(n)
		return

	case *syntax.Name:
		name := n.Value
		hash := sha256.Sum256([]byte(name))
		symID := "sym-" + hex.EncodeToString(hash[:6])
		h.Printf("%s-<span class=\"%s variable-name\">%s</span>", opName, symID, html.EscapeString(name))
		h.dumpNodeHeaderHTML(n)
		if hasChildren {
			h.Print(`<span class="node-body">`)
			defer h.Print(`</span>`)

			if obj := h.info.ObjectOf(n); obj != nil {
				h.dumpOutlineNodeHTML(depth+1, "objectOf=", obj)
			}
			if typ := h.info.TypeOf(n); isValid(typ) {
				h.dumpOutlineNodeHTML(depth+1, "typeOf=", typ)
			}
		}
		return

	case syntax.Expr:
		h.Printf("%s", opName)
		h.dumpNodeHeaderHTML(n)
		if hasChildren {
			h.Print(`<span class="node-body">`)
			defer h.Print(`</span>`)

			if typ := h.info.TypeOf(n); isValid(typ) {
				h.dumpOutlineNodeHTML(depth+1, "typeOf=", typ)
			}
		}

	default:
		h.Printf("%s", opName)
		h.dumpNodeHeaderHTML(n)
		if hasChildren {
			h.Print(`<span class="node-body">`)
			defer h.Print(`</span>`)
		}
	}

	if s := h.info.Scopes[n]; s != nil && s.Len() > 0 {
		h.dumpScopeHTML(s, depth+1, false)
	}

	v := reflect.ValueOf(n).Elem()
	t := v.Type()
	nf := t.NumField()
	for i := 0; i < nf; i++ {
		tf := t.Field(i)
		vf := v.Field(i)
		if tf.PkgPath != "" {
			continue
		}
		switch tf.Type.Kind() {
		case reflect.Interface, reflect.Ptr, reflect.Slice:
			if vf.IsNil() {
				continue
			}
		}
		name := strings.TrimSuffix(tf.Name, "_")

		switch val := vf.Interface().(type) {
		case syntax.Node:
			if name != "" {
				h.dumpNodeHTML(val, depth+1, name+": ")
			} else {
				h.dumpNodeHTML(val, depth+1, "")
			}
		default:
			if vf.Kind() == reflect.Slice && vf.Type().Elem().Implements(nodeType) {
				if vf.Len() == 0 {
					continue
				}
				if name != "" {
					for i := range vf.Len() {
						h.dumpNodeHTML(vf.Index(i).Interface().(syntax.Node), depth+1,
							fmt.Sprintf("%s[%d]: ", name, i))
					}
				} else {
					for i := range vf.Len() {
						h.dumpNodeHTML(vf.Index(i).Interface().(syntax.Node), depth+1, "")
					}
				}
			}
		}
	}
}

var nodeType = reflect.TypeFor[syntax.Node]()

func (h *HTMLWriter) nodeHasChildren(n syntax.Node) bool {
	if n == nil {
		return false
	}
	switch x := n.(type) {
	case *syntax.BasicLit:
		return false
	case *syntax.Name:
		return h.info.ObjectOf(x) != nil || isValid(h.info.TypeOf(x))
	case syntax.Expr:
		if isValid(h.info.TypeOf(x)) {
			return true
		}
	}

	v := reflect.ValueOf(n).Elem()
	t := reflect.TypeOf(n).Elem()
	nf := t.NumField()
	for i := 0; i < nf; i++ {
		tf := t.Field(i)
		vf := v.Field(i)
		if tf.PkgPath != "" {
			continue
		}
		switch tf.Type.Kind() {
		case reflect.Interface, reflect.Ptr, reflect.Slice:
			if vf.IsNil() {
				continue
			}
		}
		switch vf.Interface().(type) {
		case syntax.Node:
			return true
		default:
			if vf.Kind() == reflect.Slice && vf.Type().Elem().Implements(nodeType) && vf.Len() > 0 {
				return true
			}
		}
	}
	return false
}

func (h *HTMLWriter) dumpNodeHeaderHTML(n syntax.Node) {
	v := reflect.ValueOf(n).Elem()
	t := v.Type()
	nf := t.NumField()
	for i := 0; i < nf; i++ {
		tf := t.Field(i)
		if tf.PkgPath != "" {
			continue
		}
		k := tf.Type.Kind()
		if reflect.Bool <= k && k <= reflect.Complex128 || k == reflect.String {
			name := strings.TrimSuffix(tf.Name, "_")
			if name == "Value" {
				continue
			}
			vf := v.Field(i)
			vfi := vf.Interface()
			if vf.IsZero() {
				continue
			}
			if vfi == true {
				h.Printf(" %s", name)
			} else {
				h.Printf(" %s:%+v", name, html.EscapeString(fmt.Sprint(vf.Interface())))
			}
		}
	}

	if n.Pos().IsKnown() {
		h.Print(" <span class=\"line-number\">")
		file := n.Pos().Base().Filename()
		if file != "" {
			hash := sha256.Sum256([]byte(file))
			fileID := "loc-" + hex.EncodeToString(hash[:6])
			lineID := fmt.Sprintf("%s-L%d", fileID, n.Pos().Line())
			colID := fmt.Sprintf("%s-C%d", lineID, n.Pos().Col())

			h.Printf("<span class=\"%s line-number\">%s</span>:", fileID, html.EscapeString(filepath.Base(file)))
			h.Printf("<span class=\"%s %s line-number\">%d</span>:", lineID, fileID, n.Pos().Line())
			h.Printf("<span class=\"%s %s %s line-number\">%d</span>", colID, lineID, fileID, n.Pos().Col())
		} else {
			h.Printf("%v", html.EscapeString(n.Pos().String()))
		}
		h.Print("</span>")
	}
}
