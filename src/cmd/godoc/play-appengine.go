// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// App Engine godoc Playground functionality.

// +build appengine

package main

import (
	"io"
	"net/http"

	"appengine"
	"appengine/urlfetch"
)

func bounceToPlayground(w http.ResponseWriter, req *http.Request) {
	c := appengine.NewContext(req)
	client := urlfetch.Client(c)
	url := playgroundBaseURL + req.URL.Path
	defer req.Body.Close()
	resp, err := client.Post(url, req.Header.Get("Content-type"), req.Body)
	if err != nil {
		http.Error(w, "Internal Server Error", 500)
		c.Errorf("making POST request: %v", err)
		return
	}
	defer resp.Body.Close()
	if _, err := io.Copy(w, resp.Body); err != nil {
		http.Error(w, "Internal Server Error", 500)
		c.Errorf("making POST request: %v", err)
	}
}
