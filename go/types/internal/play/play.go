// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The play program is a playground for go/types: a simple web-based
// text editor into which the user can enter a Go program, select a
// region, and see type information about it.
//
// It is intended for convenient exploration and debugging of
// go/types. The command and its web interface are not officially
// supported and they may be changed arbitrarily in the future.
package main

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"go/types"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/packages"
)

// TODO(adonovan):
// - show line numbers next to textarea.
// - show a (tree) breakdown of the representation of the expression's type.
// - mention this in the go/types tutorial.
// - display versions of go/types and go command.

func main() {
	http.HandleFunc("/", handleRoot)
	http.HandleFunc("/main.js", handleJS)
	http.HandleFunc("/main.css", handleCSS)
	http.HandleFunc("/select.json", handleSelectJSON)
	const addr = "localhost:9999"
	log.Printf("Listening on http://%s", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}

func handleSelectJSON(w http.ResponseWriter, req *http.Request) {
	// Parse request.
	if err := req.ParseForm(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	startOffset, err := strconv.Atoi(req.Form.Get("start"))
	if err != nil {
		http.Error(w, fmt.Sprintf("start: %v", err), http.StatusBadRequest)
		return
	}
	endOffset, err := strconv.Atoi(req.Form.Get("end"))
	if err != nil {
		http.Error(w, fmt.Sprintf("end: %v", err), http.StatusBadRequest)
		return
	}

	// Write Go program to temporary file.
	f, err := os.CreateTemp("", "play-*.go")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if _, err := io.Copy(f, req.Body); err != nil {
		f.Close() // ignore error
		http.Error(w, fmt.Sprintf("can't read body: %v", err), http.StatusInternalServerError)
		return
	}
	if err := f.Close(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer func() {
		_ = os.Remove(f.Name()) // ignore error
	}()

	// Load and type check it.
	cfg := &packages.Config{
		Fset: token.NewFileSet(),
		Mode: packages.NeedTypes | packages.NeedSyntax | packages.NeedTypesInfo,
		Dir:  filepath.Dir(f.Name()),
	}
	pkgs, err := packages.Load(cfg, "file="+f.Name())
	if err != nil {
		http.Error(w, fmt.Sprintf("load: %v", err), http.StatusInternalServerError)
		return
	}
	pkg := pkgs[0]

	// -- Format the response --

	out := new(strings.Builder)

	// Parse/type error information.
	if len(pkg.Errors) > 0 {
		fmt.Fprintf(out, "Errors:\n")
		for _, err := range pkg.Errors {
			fmt.Fprintf(out, "%s: %s\n", err.Pos, err.Msg)
		}
		fmt.Fprintf(out, "\n")
	}

	fset := pkg.Fset
	file := pkg.Syntax[0]
	tokFile := fset.File(file.Pos())
	startPos := tokFile.Pos(startOffset)
	endPos := tokFile.Pos(endOffset)

	// Syntax information
	path, exact := astutil.PathEnclosingInterval(file, startPos, endPos)
	fmt.Fprintf(out, "Path enclosing interval #%d-%d [exact=%t]:\n",
		startOffset, endOffset, exact)
	var innermostExpr ast.Expr
	for i, n := range path {
		// TODO(adonovan): turn these into links to highlight the source.
		start, end := fset.Position(n.Pos()), fset.Position(n.End())
		fmt.Fprintf(out, "[%d] %T @ %d:%d-%d:%d (#%d-%d)\n",
			i, n,
			start.Line, start.Column, end.Line,
			end.Column, start.Offset, end.Offset)
		if e, ok := n.(ast.Expr); ok && innermostExpr == nil {
			innermostExpr = e
		}
	}
	fmt.Fprintf(out, "\n")

	// Expression type information
	if innermostExpr != nil {
		if tv, ok := pkg.TypesInfo.Types[innermostExpr]; ok {
			// TODO(adonovan): show tv.mode.
			// e.g. IsVoid IsType IsBuiltin IsValue IsNil Addressable Assignable HasOk
			fmt.Fprintf(out, "%T has type %v", innermostExpr, tv.Type)
			if tv.Value != nil {
				fmt.Fprintf(out, " and constant value %v", tv.Value)
			}
			fmt.Fprintf(out, "\n\n")
		}
	}

	// Object type information.
	switch n := path[0].(type) {
	case *ast.Ident:
		if obj, ok := pkg.TypesInfo.Defs[n]; ok {
			if obj == nil {
				fmt.Fprintf(out, "nil def") // e.g. package name, "_", type switch
			} else {
				formatObj(out, fset, "def", obj)
			}
		}
		if obj, ok := pkg.TypesInfo.Uses[n]; ok {
			formatObj(out, fset, "use", obj)
		}
	default:
		if obj, ok := pkg.TypesInfo.Implicits[n]; ok {
			formatObj(out, fset, "implicit def", obj)
		}
	}
	fmt.Fprintf(out, "\n")

	// Syntax debug output.
	ast.Fprint(out, fset, path[0], nil) // ignore errors
	fmt.Fprintf(out, "\n")

	// Pretty-print of selected syntax.
	format.Node(out, fset, path[0])

	// Clean up the messy temp file name.
	outStr := strings.ReplaceAll(out.String(), f.Name(), "play.go")

	// Send response.
	var respJSON struct {
		Out string
	}
	respJSON.Out = outStr

	data, _ := json.Marshal(respJSON) // can't fail
	w.Write(data)                     // ignore error
}

func formatObj(out *strings.Builder, fset *token.FileSet, ref string, obj types.Object) {
	// e.g. *types.Func -> "func"
	kind := strings.ToLower(strings.TrimPrefix(reflect.TypeOf(obj).String(), "*types."))

	// Show origin of generics, and refine kind.
	var origin types.Object
	switch obj := obj.(type) {
	case *types.Var:
		if obj.IsField() {
			kind = "field"
		}
		origin = obj.Origin()

	case *types.Func:
		if obj.Type().(*types.Signature).Recv() != nil {
			kind = "method"
		}
		origin = obj.Origin()

	case *types.TypeName:
		if obj.IsAlias() {
			kind = "type alias"
		}
		if named, ok := obj.Type().(*types.Named); ok {
			origin = named.Obj()
		}
	}

	fmt.Fprintf(out, "%s of %s %s of type %v declared at %v",
		ref, kind, obj.Name(), obj.Type(), fset.Position(obj.Pos()))
	if origin != nil && origin != obj {
		fmt.Fprintf(out, " (instantiation of %v)", origin.Type())
	}
	fmt.Fprintf(out, "\n")
}

func handleRoot(w http.ResponseWriter, req *http.Request) { io.WriteString(w, mainHTML) }
func handleJS(w http.ResponseWriter, req *http.Request)   { io.WriteString(w, mainJS) }
func handleCSS(w http.ResponseWriter, req *http.Request)  { io.WriteString(w, mainCSS) }

// TODO(adonovan): avoid CSS reliance on quirks mode and enable strict mode (<!DOCTYPE html>).
const mainHTML = `<html>
<head>
<script src="/main.js"></script>
<link rel="stylesheet" href="/main.css"></link>
</head>
<body onload="onLoad()">
<h1>go/types playground</h1>
<p>Select an expression to see information about it.</p>
<textarea rows='25' id='src'>
package main

import "fmt"

func main() {
	fmt.Println("Hello, world!")
}
</textarea>
<div id='out'/>
</body>
</html>
`

const mainJS = `
function onSelectionChange() {
	var start = document.activeElement.selectionStart;
	var end = document.activeElement.selectionEnd;
	var req = new XMLHttpRequest();
	req.open("POST", "/select.json?start=" + start + "&end=" + end, false);
	req.send(document.activeElement.value);
	var resp = JSON.parse(req.responseText);
	document.getElementById('out').innerText = resp.Out;
}

function onLoad() {
	document.getElementById("src").addEventListener('select', onSelectionChange)
}
`

const mainCSS = `
textarea { width: 6in; }
body { color: gray; }
div#out { font-family: monospace; font-size: 80%; }
`
