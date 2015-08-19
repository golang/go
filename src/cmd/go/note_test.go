// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	main "cmd/go"
	"runtime"
	"testing"
)

func TestNoteReading(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.tempFile("hello.go", `package main; func main() { print("hello, world\n") }`)
	const buildID = "TestNoteReading-Build-ID"
	tg.run("build", "-ldflags", "-buildid="+buildID, "-o", tg.path("hello.exe"), tg.path("hello.go"))
	id, err := main.ReadBuildIDFromBinary(tg.path("hello.exe"))
	if err != nil {
		t.Fatalf("reading build ID from hello binary: %v", err)
	}
	if id != buildID {
		t.Fatalf("buildID in hello binary = %q, want %q", id, buildID)
	}

	if runtime.GOOS == "linux" && runtime.GOARCH == "ppc64le" {
		t.Skipf("skipping - golang.org/issue/11184")
	}

	switch runtime.GOOS {
	case "plan9":
		// no external linking
		t.Logf("no external linking - skipping linkmode=external test")

	case "solaris":
		t.Logf("skipping - golang.org/issue/12178")

	default:
		tg.run("build", "-ldflags", "-buildid="+buildID+" -linkmode=external", "-o", tg.path("hello.exe"), tg.path("hello.go"))
		id, err := main.ReadBuildIDFromBinary(tg.path("hello.exe"))
		if err != nil {
			t.Fatalf("reading build ID from hello binary (linkmode=external): %v", err)
		}
		if id != buildID {
			t.Fatalf("buildID in hello binary = %q, want %q (linkmode=external)", id, buildID)
		}
	}
}
