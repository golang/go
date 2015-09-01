// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package playground

import (
	"net/http"

	"appengine"
	"appengine/urlfetch"
)

func init() {
	onAppengine = !appengine.IsDevAppServer()
}

func client(r *http.Request) *http.Client {
	return urlfetch.Client(appengine.NewContext(r))
}

func report(r *http.Request, err error) {
	appengine.NewContext(r).Errorf("%v", err)
}
