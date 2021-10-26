// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the httpresponse checker.

//go:build go1.18

package typeparams

import (
	"log"
	"net/http"
)

func badHTTPGet[T any](url string) {
	res, err := http.Get(url)
	defer res.Body.Close() // want "using res before checking for errors"
	if err != nil {
		log.Fatal(err)
	}
}

func mkClient[T any]() *T {
	return nil
}

func badClientHTTPGet() {
	client := mkClient[http.Client]()
	res, _ := client.Get("")
	defer res.Body.Close() // want "using res before checking for errors"
}

// User-defined type embedded "http.Client"
type S[P any] struct {
	http.Client
}

func unmatchedClientTypeName(client S[string]) {
	res, _ := client.Get("")
	defer res.Body.Close() // the name of client's type doesn't match "*http.Client"
}

// User-defined Client type
type C[P any] interface {
	Get(url string) (resp *P, err error)
}

func userDefinedClientType(client C[http.Response]) {
	resp, _ := client.Get("http://foo.com")
	defer resp.Body.Close() // "client" is not of type "*http.Client"
}
