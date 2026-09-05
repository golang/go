// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fstest_test

import (
	"fmt"
	"io/fs"
	"log"
	"testing/fstest"
)

func ExampleMapFS() {
	fsys := fstest.MapFS{
		"hello.txt": {
			Data: []byte("hello"),
		},
		"docs/readme.txt": {
			Data: []byte("documentation"),
		},
		"readme.link": {
			Data: []byte("docs/readme.txt"),
			Mode: fs.ModeSymlink,
		},
	}

	target, err := fs.ReadLink(fsys, "readme.link")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("readme.link ->", target)

	data, err := fs.ReadFile(fsys, "readme.link")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(data))

	entries, err := fs.ReadDir(fsys, ".")
	if err != nil {
		log.Fatal(err)
	}
	for _, entry := range entries {
		fmt.Println(entry.Name())
	}

	// Output:
	// readme.link -> docs/readme.txt
	// documentation
	// docs
	// hello.txt
	// readme.link
}
