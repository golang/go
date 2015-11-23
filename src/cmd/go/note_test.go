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
	if runtime.GOOS == "dragonfly" {
		t.Skipf("TestNoteReading is broken on dragonfly - golang.org/issue/13364", runtime.GOOS)
	}
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
