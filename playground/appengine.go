// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package playground

import (
	"context"
	"io"
	"net/http"

	"google.golang.org/appengine"
	"google.golang.org/appengine/log"
	"google.golang.org/appengine/urlfetch"
)

func init() {
	onAppengine = !appengine.IsDevAppServer()
}

func contextFunc(r *http.Request) context.Context {
	return appengine.NewContext(r)
}

func post(ctx context.Context, url, contentType string, body io.Reader) (*http.Response, error) {
	return urlfetch.Client(ctx).Post(url, contentType, body)
}

func report(r *http.Request, err error) {
	ctx := appengine.NewContext(r)
	log.Errorf(ctx, "%v", err)
}
