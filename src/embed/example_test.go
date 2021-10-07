// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embed_test

import (
	"embed"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

//go:embed internal/embedtest/testdata/*.txt
var content embed.FS

func Example() {
	addr := "127.0.0.1:8080"

	go func() {
		mutex := http.NewServeMux()
		mutex.Handle("/", http.FileServer(http.FS(content)))
		err := http.ListenAndServe(addr, mutex)
		if err != nil {
			log.Fatalf("http server start failed: %v", err)
		}
	}()

	time.Sleep(time.Second)

	url := fmt.Sprintf("http://%s/internal/embedtest/testdata/hello.txt", addr)
	resp, err := http.Get(url)
	if err != nil {
		log.Fatalf("http client request failed: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Fatalf("read response body failed: %v", err)
	}

	fmt.Printf("%s", body)

	// Output:
	// hello, world
}
