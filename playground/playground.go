// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package playground registers HTTP handlers at "/compile" and "/share" that
// proxy requests to the golang.org playground service.
// This package may be used unaltered on App Engine Standard with Go 1.11+ runtime.
package playground // import "golang.org/x/tools/playground"

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"golang.org/x/tools/godoc/golangorgenv"
)

const baseURL = "https://play.golang.org"

func init() {
	http.HandleFunc("/compile", bounce)
	http.HandleFunc("/share", bounce)
}

func bounce(w http.ResponseWriter, r *http.Request) {
	b := new(bytes.Buffer)
	if err := passThru(b, r); os.IsPermission(err) {
		http.Error(w, "403 Forbidden", http.StatusForbidden)
		log.Println(err)
		return
	} else if err != nil {
		http.Error(w, "500 Internal Server Error", http.StatusInternalServerError)
		log.Println(err)
		return
	}
	io.Copy(w, b)
}

func passThru(w io.Writer, req *http.Request) error {
	if req.URL.Path == "/share" && googleCN(req) {
		return os.ErrPermission
	}
	defer req.Body.Close()
	url := baseURL + req.URL.Path
	ctx, cancel := context.WithTimeout(req.Context(), 60*time.Second)
	defer cancel()
	r, err := post(ctx, url, req.Header.Get("Content-Type"), req.Body)
	if err != nil {
		return fmt.Errorf("making POST request: %v", err)
	}
	defer r.Body.Close()
	if _, err := io.Copy(w, r.Body); err != nil {
		return fmt.Errorf("copying response Body: %v", err)
	}
	return nil
}

func post(ctx context.Context, url, contentType string, body io.Reader) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodPost, url, body)
	if err != nil {
		return nil, fmt.Errorf("http.NewRequest: %v", err)
	}
	req.Header.Set("Content-Type", contentType)
	return http.DefaultClient.Do(req.WithContext(ctx))
}

// googleCN reports whether request r is considered
// to be served from golang.google.cn.
func googleCN(r *http.Request) bool {
	if r.FormValue("googlecn") != "" {
		return true
	}
	if strings.HasSuffix(r.Host, ".cn") {
		return true
	}
	if !golangorgenv.CheckCountry() {
		return false
	}
	switch r.Header.Get("X-Appengine-Country") {
	case "", "ZZ", "CN":
		return true
	}
	return false
}
