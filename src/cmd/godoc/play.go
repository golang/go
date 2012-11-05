// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Common Playground functionality.

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"net/http"
)

// The server that will service compile and share requests.
const playgroundBaseURL = "http://play.golang.org"

func registerPlaygroundHandlers(mux *http.ServeMux) {
	if *showPlayground {
		mux.HandleFunc("/compile", bounceToPlayground)
		mux.HandleFunc("/share", bounceToPlayground)
	} else {
		mux.HandleFunc("/compile", disabledHandler)
		mux.HandleFunc("/share", disabledHandler)
	}
	http.HandleFunc("/fmt", fmtHandler)
}

type fmtResponse struct {
	Body  string
	Error string
}

// fmtHandler takes a Go program in its "body" form value, formats it with
// standard gofmt formatting, and writes a fmtResponse as a JSON object.
func fmtHandler(w http.ResponseWriter, r *http.Request) {
	resp := new(fmtResponse)
	body, err := gofmt(r.FormValue("body"))
	if err != nil {
		resp.Error = err.Error()
	} else {
		resp.Body = body
	}
	json.NewEncoder(w).Encode(resp)
}

// gofmt takes a Go program, formats it using the standard Go formatting
// rules, and returns it or an error.
func gofmt(body string) (string, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "prog.go", body, parser.ParseComments)
	if err != nil {
		return "", err
	}
	ast.SortImports(fset, f)
	var buf bytes.Buffer
	config := printer.Config{
		Mode:     printer.UseSpaces | printer.TabIndent,
		Tabwidth: 8,
	}
	err = config.Fprint(&buf, fset, f)
	if err != nil {
		return "", err
	}
	return buf.String(), nil
}

// disabledHandler serves a 501 "Not Implemented" response.
func disabledHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusNotImplemented)
	fmt.Fprint(w, "This functionality is not available via local godoc.")
}
