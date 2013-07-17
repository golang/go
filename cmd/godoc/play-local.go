// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Stand-alone godoc Playground functionality.

// +build !appengine

package main

import (
	"io"
	"net/http"
	"net/url"
)

var playgroundScheme, playgroundHost string

func init() {
	u, err := url.Parse(playgroundBaseURL)
	if err != nil {
		panic(err)
	}
	playgroundScheme = u.Scheme
	playgroundHost = u.Host
}

// bounceToPlayground forwards the request to play.golang.org.
func bounceToPlayground(w http.ResponseWriter, req *http.Request) {
	defer req.Body.Close()
	req.URL.Scheme = playgroundScheme
	req.URL.Host = playgroundHost
	resp, err := http.Post(req.URL.String(), req.Header.Get("Content-type"), req.Body)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
	resp.Body.Close()
}
