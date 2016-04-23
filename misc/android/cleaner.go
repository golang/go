// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Cleaner removes anything from /data/local/tmp/goroot not on a builtin list.
// Used by androidtest.bash.
package main

import (
	"log"
	"os"
	"path/filepath"
	"strings"
)

func main() {
	const goroot = "/data/local/tmp/goroot"
	expect := make(map[string]bool)
	for _, f := range strings.Split(files, "\n") {
		expect[filepath.Join(goroot, f)] = true
	}

	err := filepath.Walk(goroot, func(path string, info os.FileInfo, err error) error {
		if expect[path] {
			return nil
		}
		log.Printf("removing %s", path)
		if err := os.RemoveAll(path); err != nil {
			return err
		}
		if info.IsDir() {
			return filepath.SkipDir
		}
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
}
