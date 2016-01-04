// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

// Package proxy proxies requests to the sandbox compiler service and the
// playground share handler.
// It is designed to run only on the instance of godoc that serves golang.org.
package proxy

import (
	"bytes"
	"crypto/sha1"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httputil"
	"net/url"
	"time"

	"golang.org/x/net/context"

	"google.golang.org/appengine"
	"google.golang.org/appengine/log"
	"google.golang.org/appengine/memcache"
	"google.golang.org/appengine/urlfetch"
)

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

const (
	// We need to use HTTP here for "reasons", but the traffic isn't
	// sensitive and it only travels across Google's internal network
	// so we should be OK.
	sandboxURL    = "http://sandbox.golang.org/compile"
	playgroundURL = "http://play.golang.org"
)

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

	c := appengine.NewContext(r)

	body := r.FormValue("body")
	res := &Response{}
	key := cacheKey(body)
	if _, err := memcache.Gob.Get(c, key, res); err != nil {
		if err != memcache.ErrCacheMiss {
			log.Errorf(c, "getting response cache: %v", err)
		}

		req := &Request{Body: body}
		if err := makeSandboxRequest(c, req, res); err != nil {
			log.Errorf(c, "compile error: %v", err)
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			return
		}

		item := &memcache.Item{Key: key, Object: res}
		if err := memcache.Gob.Set(c, item); err != nil {
			log.Errorf(c, "setting response cache: %v", err)
		}
	}

	expiresTime := time.Now().Add(expires).UTC()
	w.Header().Set("Expires", expiresTime.Format(time.RFC1123))
	w.Header().Set("Cache-Control", cacheControlHeader)

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
	if err := json.NewEncoder(w).Encode(out); err != nil {
		log.Errorf(c, "encoding response: %v", err)
	}
}

// makeSandboxRequest sends the given Request to the sandbox
// and stores the response in the given Response.
func makeSandboxRequest(c context.Context, req *Request, res *Response) error {
	reqJ, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("marshalling request: %v", err)
	}
	r, err := urlfetch.Client(c).Post(sandboxURL, "application/json", bytes.NewReader(reqJ))
	if err != nil {
		return fmt.Errorf("making request: %v", err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		b, _ := ioutil.ReadAll(r.Body)
		return fmt.Errorf("bad status: %v body:\n%s", r.Status, b)
	}
	err = json.NewDecoder(r.Body).Decode(res)
	if err != nil {
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

func cacheKey(body string) string {
	h := sha1.New()
	io.WriteString(h, body)
	return fmt.Sprintf("prog-%x", h.Sum(nil))
}

func share(w http.ResponseWriter, r *http.Request) {
	if !allowShare(r) {
		http.Error(w, "Forbidden", http.StatusForbidden)
		return
	}
	target, _ := url.Parse(playgroundURL)
	p := httputil.NewSingleHostReverseProxy(target)
	p.Transport = &urlfetch.Transport{Context: appengine.NewContext(r)}
	p.ServeHTTP(w, r)
}

func allowShare(r *http.Request) bool {
	if appengine.IsDevAppServer() {
		return true
	}
	switch r.Header.Get("X-AppEngine-Country") {
	case "", "ZZ", "CN":
		return false
	}
	return true
}
