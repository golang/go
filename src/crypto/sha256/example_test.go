// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sha256_test

import (
	"crypto/sha256"
	"fmt"
	"io"
	"log"
	"os"
)

func ExampleSum256() {
	sum := sha256.Sum256([]byte("hello world\n"))
	fmt.Printf("%x", sum)
	// Output: a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447
}

func ExampleNew() {
	h := sha256.New()
	h.Write([]byte("hello world\n"))
	fmt.Printf("%x", h.Sum(nil))
	// Output: a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447
}

func ExampleNew_file() {
	f, err := os.Open("file.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%x", h.Sum(nil))
}
