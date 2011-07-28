// errchk $G $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 1979
// used to get internal compiler error too

package main

import (
	"http"
	"io/ioutil"
	"os"
)

func makeHandler(fn func(http.ResponseWriter, *http.Request, string)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request)  // ERROR "syntax error"
}

type Page struct {
	Title string
	Body []byte
}

