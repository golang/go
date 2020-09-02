// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testgodefs

import (
	"bytes"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// We are testing cgo -godefs, which translates Go files that use
// import "C" into Go files with Go definitions of types defined in the
// import "C" block.  Add more tests here.
var filePrefixes = []string{
	"anonunion",
	"bitfields",
	"issue8478",
	"fieldtypedef",
	"issue37479",
	"issue37621",
	"issue38649",
	"issue39534",
}

func TestGoDefs(t *testing.T) {
	testdata, err := filepath.Abs("testdata")
	if err != nil {
		t.Fatal(err)
	}

	gopath, err := ioutil.TempDir("", "testgodefs-gopath")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(gopath)

	dir := filepath.Join(gopath, "src", "testgodefs")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}

	for _, fp := range filePrefixes {
		cmd := exec.Command("go", "tool", "cgo",
			"-godefs",
			"-srcdir", testdata,
			"-objdir", dir,
			fp+".go")
		cmd.Stderr = new(bytes.Buffer)

		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("%s: %v\n%s", strings.Join(cmd.Args, " "), err, cmd.Stderr)
		}

		if err := ioutil.WriteFile(filepath.Join(dir, fp+"_defs.go"), out, 0644); err != nil {
			t.Fatal(err)
		}
	}

	main, err := ioutil.ReadFile(filepath.Join("testdata", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(dir, "main.go"), main, 0644); err != nil {
		t.Fatal(err)
	}

	if err := ioutil.WriteFile(filepath.Join(dir, "go.mod"), []byte("module testgodefs\ngo 1.14\n"), 0644); err != nil {
		t.Fatal(err)
	}

	// Use 'go run' to build and run the resulting binary in a single step,
	// instead of invoking 'go build' and the resulting binary separately, so that
	// this test can pass on mobile builders, which do not copy artifacts back
	// from remote invocations.
	cmd := exec.Command("go", "run", ".")
	cmd.Env = append(os.Environ(), "GOPATH="+gopath)
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("%s [%s]: %v\n%s", strings.Join(cmd.Args, " "), dir, err, out)
	}
}
