// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embed_test

import (
	"embed"
	"log"
	"net/http"
)

//go:embed static/*.png
var images embed.FS

func Example() {
	mux := http.NewServeMux()
	mux.Handle("/static/", http.FileServer(http.FS(images)))
	err := http.ListenAndServe(":8080", mux)
	if err != nil {
		log.Fatal(err)
	}
}
