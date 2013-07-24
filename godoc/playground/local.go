// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !appengine

package playground

import (
	"log"
	"net/http"
)

func client(r *http.Request) *http.Client {
	return http.DefaultClient
}

func report(r *http.Request, err error) {
	log.Println(err)
}
