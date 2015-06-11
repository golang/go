// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"io/ioutil"
	"os"
	"os/exec"
	"runtime"
	"testing"
)

func TestNoteReading(t *testing.T) {
	// No file system access on these systems.
	switch sys := runtime.GOOS + "/" + runtime.GOARCH; sys {
	case "darwin/arm", "darwin/arm64", "nacl/386", "nacl/amd64p32", "nacl/arm":
		t.Skipf("skipping on %s/%s - no file system", runtime.GOOS, runtime.GOARCH)
	}
	if runtime.GOOS == "android" {
		t.Skipf("skipping; requires go tool")
	}

	// TODO: Replace with new test scaffolding by iant.
	d, err := ioutil.TempDir("", "go-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(d)

	out, err := exec.Command("go", "build", "-o", d+"/go.exe", "cmd/go").CombinedOutput()
	if err != nil {
		t.Fatalf("go build cmd/go: %v\n%s", err, out)
	}

	const buildID = "TestNoteReading-Build-ID"
	out, err = exec.Command(d+"/go.exe", "build", "-ldflags", "-buildid="+buildID, "-o", d+"/hello.exe", "../../../test/helloworld.go").CombinedOutput()
	if err != nil {
		t.Fatalf("go build hello: %v\n%s", err, out)
	}

	id, err := readBuildIDFromBinary(d + "/hello.exe")
	if err != nil {
		t.Fatalf("reading build ID from hello binary: %v", err)
	}

	if id != buildID {
		t.Fatalf("buildID in hello binary = %q, want %q", id, buildID)
	}
}
