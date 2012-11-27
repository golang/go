// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Common Playground functionality.

package main

import (
	"encoding/json"
	"fmt"
	"go/format"
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
	body, err := format.Source([]byte(r.FormValue("body")))
	if err != nil {
		resp.Error = err.Error()
	} else {
		resp.Body = string(body)
	}
	json.NewEncoder(w).Encode(resp)
}

// disabledHandler serves a 501 "Not Implemented" response.
func disabledHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusNotImplemented)
	fmt.Fprint(w, "This functionality is not available via local godoc.")
}
