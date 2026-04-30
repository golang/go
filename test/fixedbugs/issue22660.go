// run

//go:build !js && !wasip1 && gc

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func main() {
	f, err := ioutil.TempFile("", "issue22660.go")
	if err != nil {
		log.Fatal(err)
	}
	f.Close()
	defer os.Remove(f.Name())

	// path components from the //line directive must survive -trimpath and
	// appear in error messages (since go.dev/issue/70478, as a path-component
	// suffix of the resolved path, not as the bare prefix).
	path := filepath.Join("users", "xxx", "go")
	var src bytes.Buffer
	fmt.Fprintf(&src, "//line %s:1\n", filepath.Join(path, "foo.go"))

	if err := ioutil.WriteFile(f.Name(), src.Bytes(), 0660); err != nil {
		log.Fatal(err)
	}

	out, err := exec.Command("go", "tool", "compile", "-p=p", fmt.Sprintf("-trimpath=%s", path), f.Name()).CombinedOutput()
	if err == nil {
		log.Fatalf("expected compiling %s to fail", f.Name())
	}

	// After #70478 the resolved path is <tempdir>/users/xxx/go/foo.go,
	// so the directive's components must appear as a path suffix.
	want := filepath.Join(path, "foo.go")
	if !strings.Contains(string(out), string(filepath.Separator)+want) {
		log.Fatalf("expected path component (%s) in error message, got:\n%s", want, out)
	}
}
