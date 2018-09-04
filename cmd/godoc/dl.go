// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !golangorg

package main

import "net/http"

// Register a redirect handler for /dl/ to the golang.org download page.
// This file will not be included when deploying godoc to golang.org.

func init() {
	http.Handle("/dl/", http.RedirectHandler("https://golang.org/dl/", http.StatusFound))
}
