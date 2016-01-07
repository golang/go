// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	main "cmd/go"
	"go/build"
	"runtime"
	"testing"
)

func TestNoteReading(t *testing.T) {
	testNoteReading(t)
}

func TestNoteReading2K(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skipf("2kB is not enough on %s", runtime.GOOS)
	}
	// Set BuildIDReadSize to 2kB to exercise Mach-O parsing more strictly.
	defer func(old int) {
		main.BuildIDReadSize = old
	}(main.BuildIDReadSize)
	main.BuildIDReadSize = 2 * 1024

	testNoteReading(t)
}

func testNoteReading(t *testing.T) {
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

	switch {
	case !build.Default.CgoEnabled:
		t.Skipf("skipping - no cgo, so assuming external linking not available")
	case runtime.GOOS == "linux" && (runtime.GOARCH == "ppc64le" || runtime.GOARCH == "ppc64"):
		t.Skipf("skipping - external linking not supported, golang.org/issue/11184")
	case runtime.GOOS == "linux" && (runtime.GOARCH == "mips64le" || runtime.GOARCH == "mips64"):
		t.Skipf("skipping - external linking not supported, golang.org/issue/12560")
	case runtime.GOOS == "openbsd" && runtime.GOARCH == "arm":
		t.Skipf("skipping - external linking not supported, golang.org/issue/10619")
	case runtime.GOOS == "plan9":
		t.Skipf("skipping - external linking not supported")
	}

	tg.run("build", "-ldflags", "-buildid="+buildID+" -linkmode=external", "-o", tg.path("hello.exe"), tg.path("hello.go"))
	id, err = main.ReadBuildIDFromBinary(tg.path("hello.exe"))
	if err != nil {
		t.Fatalf("reading build ID from hello binary (linkmode=external): %v", err)
	}
	if id != buildID {
		t.Fatalf("buildID in hello binary = %q, want %q (linkmode=external)", id, buildID)
	}
}
