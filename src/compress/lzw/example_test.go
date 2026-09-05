// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzw_test

import (
	"bytes"
	"compress/lzw"
	"fmt"
	"io"
	"log"
)

func Example() {
	const original = "Hello, Gophers!"

	var compressed bytes.Buffer
	// LSB and an 8-bit literal width are the settings used by GIF.
	w := lzw.NewWriter(&compressed, lzw.LSB, 8)
	if _, err := w.Write([]byte(original)); err != nil {
		log.Fatal(err)
	}
	if err := w.Close(); err != nil {
		log.Fatal(err)
	}
	r := lzw.NewReader(&compressed, lzw.LSB, 8)
	decoded, err := io.ReadAll(r)
	if err != nil {
		log.Fatal(err)
	}
	if err := r.Close(); err != nil {
		log.Fatal(err)
	}

	fmt.Println(string(decoded))
	// Output: Hello, Gophers!
}
