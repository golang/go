// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"log"
	"os"
)

func main() {
	wd, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Open(wd)
	if err != nil {
		log.Fatal(err)
	}
	dirnames1, err := f.Readdirnames(0)
	if err != nil {
		log.Fatal(err)
	}

	ret, err := f.Seek(0, 0)
	if err != nil {
		log.Fatal(err)
	}
	if ret != 0 {
		log.Fatalf("seek result not zero: %d", ret)
	}

	dirnames2, err := f.Readdirnames(0)
	if err != nil {
		log.Fatal(err)
		return
	}

	if len(dirnames1) != len(dirnames2) {
		log.Fatalf("listings have different lengths: %d and %d\n", len(dirnames1), len(dirnames2))
	}
	for i, n1 := range dirnames1 {
		n2 := dirnames2[i]
		if n1 != n2 {
			log.Fatalf("different name i=%d n1=%s n2=%s\n", i, n1, n2)
		}
	}
}
