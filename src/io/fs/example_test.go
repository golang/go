// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fs_test

import (
	"fmt"
	"io/fs"
	"log"
	"os"
)

func ExampleWalkDir() {
	root := "/usr/local/go/bin"
	fileSystem := os.DirFS(root)

	fs.WalkDir(fileSystem, ".", func { path, d, err ->
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(path)
		return nil
	})
}
