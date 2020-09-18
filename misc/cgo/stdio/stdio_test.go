// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stdio_test

import (
	"bytes"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestMain(m *testing.M) {
	log.SetFlags(log.Lshortfile)
	os.Exit(testMain(m))
}

func testMain(m *testing.M) int {
	GOPATH, err := ioutil.TempDir("", "cgostdio")
	if err != nil {
		log.Panic(err)
	}
	defer os.RemoveAll(GOPATH)
	os.Setenv("GOPATH", GOPATH)

	// Copy testdata into GOPATH/src/cgostdio, along with a go.mod file
	// declaring the same path.
	modRoot := filepath.Join(GOPATH, "src", "cgostdio")
	if err := overlayDir(modRoot, "testdata"); err != nil {
		log.Panic(err)
	}
	if err := os.Chdir(modRoot); err != nil {
		log.Panic(err)
	}
	os.Setenv("PWD", modRoot)
	if err := ioutil.WriteFile("go.mod", []byte("module cgostdio\n"), 0666); err != nil {
		log.Panic(err)
	}

	return m.Run()
}

func TestTestRun(t *testing.T) {
	if os.Getenv("GOOS") == "android" {
		t.Skip("subpackage stdio is not available on android")
	}
	out, err := exec.Command("go", "env", "GOROOT").Output()
	if err != nil {
		t.Fatal(err)
	}
	GOROOT := string(bytes.TrimSpace(out))

	cmd := exec.Command("go", "run", filepath.Join(GOROOT, "test", "run.go"), "-", ".")
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s: %s\n%s", strings.Join(cmd.Args, " "), err, out)
	}
	t.Logf("%s:\n%s", strings.Join(cmd.Args, " "), out)
}
