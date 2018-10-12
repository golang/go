// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package proxy proxies requests to the playground's compile and share handlers.
// It is designed to run only on the instance of godoc that serves golang.org.
package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"strings"
	"time"

	"golang.org/x/tools/godoc/env"
)

const playgroundURL = "https://play.golang.org"

type Request struct {
	Body string
}

type Response struct {
	Errors string
	Events []Event
}

type Event struct {
	Message string
	Kind    string        // "stdout" or "stderr"
	Delay   time.Duration // time to wait before printing Message
}

const expires = 7 * 24 * time.Hour // 1 week
var cacheControlHeader = fmt.Sprintf("public, max-age=%d", int(expires.Seconds()))

func RegisterHandlers(mux *http.ServeMux) {
	mux.HandleFunc("/compile", compile)
	mux.HandleFunc("/share", share)
}

func compile(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "I only answer to POST requests.", http.StatusMethodNotAllowed)
		return
	}

	ctx := r.Context()

	body := r.FormValue("body")
	res := &Response{}
	req := &Request{Body: body}
	if err := makeCompileRequest(ctx, req, res); err != nil {
		log.Printf("ERROR compile error: %v", err)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}

	var out interface{}
	switch r.FormValue("version") {
	case "2":
		out = res
	default: // "1"
		out = struct {
			CompileErrors string `json:"compile_errors"`
			Output        string `json:"output"`
		}{res.Errors, flatten(res.Events)}
	}
	b, err := json.Marshal(out)
	if err != nil {
		log.Printf("ERROR encoding response: %v", err)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}

	expiresTime := time.Now().Add(expires).UTC()
	w.Header().Set("Expires", expiresTime.Format(time.RFC1123))
	w.Header().Set("Cache-Control", cacheControlHeader)
	w.Write(b)
}

// makePlaygroundRequest sends the given Request to the playground compile
// endpoint and stores the response in the given Response.
func makeCompileRequest(ctx context.Context, req *Request, res *Response) error {
	reqJ, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("marshalling request: %v", err)
	}
	hReq, _ := http.NewRequest("POST", playgroundURL+"/compile", bytes.NewReader(reqJ))
	hReq.Header.Set("Content-Type", "application/json")
	hReq = hReq.WithContext(ctx)

	r, err := http.DefaultClient.Do(hReq)
	if err != nil {
		return fmt.Errorf("making request: %v", err)
	}
	defer r.Body.Close()

	if r.StatusCode != http.StatusOK {
		b, _ := ioutil.ReadAll(r.Body)
		return fmt.Errorf("bad status: %v body:\n%s", r.Status, b)
	}

	if err := json.NewDecoder(r.Body).Decode(res); err != nil {
		return fmt.Errorf("unmarshalling response: %v", err)
	}
	return nil
}

// flatten takes a sequence of Events and returns their contents, concatenated.
func flatten(seq []Event) string {
	var buf bytes.Buffer
	for _, e := range seq {
		buf.WriteString(e.Message)
	}
	return buf.String()
}

func share(w http.ResponseWriter, r *http.Request) {
	if googleCN(r) {
		http.Error(w, http.StatusText(http.StatusForbidden), http.StatusForbidden)
		return
	}

	// HACK(cbro): use a simple proxy rather than httputil.ReverseProxy because of Issue #28168.
	// TODO: investigate using ReverseProxy with a Director, unsetting whatever's necessary to make that work.
	req, _ := http.NewRequest("POST", playgroundURL+"/share", r.Body)
	req.Header.Set("Content-Type", r.Header.Get("Content-Type"))
	req = req.WithContext(r.Context())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Printf("ERROR share error: %v", err)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}
	copyHeader := func(k string) {
		if v := resp.Header.Get(k); v != "" {
			w.Header().Set(k, v)
		}
	}
	copyHeader("Content-Type")
	copyHeader("Content-Length")
	defer resp.Body.Close()
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

func googleCN(r *http.Request) bool {
	if r.FormValue("googlecn") != "" {
		return true
	}
	if !env.IsProd() {
		return false
	}
	if strings.HasSuffix(r.Host, ".cn") {
		return true
	}
	switch r.Header.Get("X-AppEngine-Country") {
	case "", "ZZ", "CN":
		return true
	}
	return false
}
