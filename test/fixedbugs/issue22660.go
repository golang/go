// run

//go:build !js && !wasip1 && gc

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func main() {
	f, err := os.CreateTemp("", "issue22660.go")
	if err != nil {
		log.Fatal(err)
	}
	f.Close()
	defer os.Remove(f.Name())

	// path must appear in error messages even if we strip them with -trimpath
	path := filepath.Join("users", "xxx", "go")
	filename := filepath.Join(path, "foo.go")
	var src bytes.Buffer
	fmt.Fprintf(&src, "//line %s:1\n", filename)
	if err := os.WriteFile(f.Name(), src.Bytes(), 0660); err != nil {
		log.Fatal(err)
	}

	out, err := exec.Command("go", "tool", "compile", "-p=p", fmt.Sprintf("-trimpath=%s", path), f.Name()).CombinedOutput()
	if err == nil {
		// The file only contains a line directive, w/o a package clause.
		log.Fatalf("expected compiling %s to fail with syntax error", f.Name())
	}

	// The error message position depends on the line directive.
	// The resolved path is <tempdir>/filename, so the directive's components
	// must appear as a path suffix in the error message's error position
	// (go.dev/issue/70478).
	if !strings.Contains(string(out), string(filepath.Separator)+filename) {
		log.Fatalf("expected full path and filename (%s) in error message, got:\n%s", filename, out)
	}
}
