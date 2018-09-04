// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package proxy proxies requests to the playground's compile and share handlers.
// It is designed to run only on the instance of godoc that serves golang.org.
package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/tools/godoc/env"
)

const playgroundURL = "https://play.golang.org"

var proxy *httputil.ReverseProxy

func init() {
	target, _ := url.Parse(playgroundURL)
	proxy = httputil.NewSingleHostReverseProxy(target)
}

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
	proxy.ServeHTTP(w, r)
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
