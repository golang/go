// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package playground registers an HTTP handler at "/compile" that
// proxies requests to the golang.org playground service.
// This package may be used unaltered on App Engine Standard with Go 1.11+ runtime.
package playground // import "golang.org/x/tools/playground"

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

const baseURL = "https://play.golang.org"

func init() {
	http.Handle("/compile", Proxy())
}

// Proxy returns a handler that can be registered on /compile to proxy requests to the Go playground.
//
// This package already contains a func init that does:
//
//	func init() {
//		http.Handle("/compile", Proxy())
//	}
//
// Proxy may be useful for servers that use HTTP muxes other than the default mux.
func Proxy() http.Handler {
	return http.HandlerFunc(bounce)
}

func bounce(w http.ResponseWriter, r *http.Request) {
	b := new(bytes.Buffer)
	if err := passThru(b, r); err != nil {
		http.Error(w, "500 Internal Server Error", http.StatusInternalServerError)
		log.Println(err)
		return
	}
	io.Copy(w, b)
}

func passThru(w io.Writer, req *http.Request) error {
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
