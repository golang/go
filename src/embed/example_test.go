// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embed_test

import (
	"embed"
	"log"
	"net/http"
)

//go:embed internal/embedtest/testdata/*.txt
var content embed.FS

func Example() {
	mutex := http.NewServeMux()
	mutex.Handle("/", http.FileServer(http.FS(content)))
	err := http.ListenAndServe(":8080", mutex)
	if err != nil {
		log.Fatal(err)
	}
}
