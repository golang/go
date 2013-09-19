// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package playground registers HTTP handlers at "/compile" and "/share" that
// proxy requests to the golang.org playground service.
// This package may be used unaltered on App Engine.
package playground

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
)

const baseURL = "http://play.golang.org"

func init() {
	http.HandleFunc("/compile", bounce)
	http.HandleFunc("/share", bounce)
}

func bounce(w http.ResponseWriter, r *http.Request) {
	b := new(bytes.Buffer)
	if err := passThru(b, r); err != nil {
		http.Error(w, "Server error.", http.StatusInternalServerError)
		report(r, err)
		return
	}
	io.Copy(w, b)
}

func passThru(w io.Writer, req *http.Request) error {
	defer req.Body.Close()
	url := baseURL + req.URL.Path
	r, err := client(req).Post(url, req.Header.Get("Content-type"), req.Body)
	if err != nil {
		return fmt.Errorf("making POST request: %v", err)
	}
	defer r.Body.Close()
	if _, err := io.Copy(w, r.Body); err != nil {
		return fmt.Errorf("copying response Body: %v", err)
	}
	return nil
}
