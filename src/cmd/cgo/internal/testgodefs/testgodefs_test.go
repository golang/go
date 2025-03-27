// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testgodefs

import (
	"bytes"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
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
	"issue48396",
}

func TestGoDefs(t *testing.T) {
	testenv.MustHaveGoRun(t)
	testenv.MustHaveCGO(t)

	testdata, err := filepath.Abs("testdata")
	if err != nil {
		t.Fatal(err)
	}

	gopath, err := os.MkdirTemp("", "testgodefs-gopath")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(gopath)

	dir := filepath.Join(gopath, "src", "testgodefs")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}

	for _, fp := range filePrefixes {
		cmd := exec.Command(testenv.GoToolPath(t), "tool", "cgo",
			"-godefs",
			"-srcdir", testdata,
			"-objdir", dir,
			fp+".go")
		cmd.Stderr = new(bytes.Buffer)

		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("%s: %v\n%s", strings.Join(cmd.Args, " "), err, cmd.Stderr)
		}

		fn := fp + "_defs.go"
		if err := os.WriteFile(filepath.Join(dir, fn), out, 0644); err != nil {
			t.Fatal(err)
		}

		// Verify that command line arguments are not rewritten in the generated comment,
		// see go.dev/issue/52063
		hasGeneratedByComment := false
		for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
			cgoExe := "cgo"
			if runtime.GOOS == "windows" {
				cgoExe = "cgo.exe"
			}
			if !strings.HasPrefix(line, "// "+cgoExe+" -godefs") {
				continue
			}
			if want := "// " + cgoExe + " " + strings.Join(cmd.Args[3:], " "); line != want {
				t.Errorf("%s: got generated comment %q, want %q", fn, line, want)
			}
			hasGeneratedByComment = true
			break
		}

		if !hasGeneratedByComment {
			t.Errorf("%s: comment with generating cgo -godefs command not found", fn)
		}
	}

	main, err := os.ReadFile(filepath.Join("testdata", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"), main, 0644); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module testgodefs\ngo 1.14\n"), 0644); err != nil {
		t.Fatal(err)
	}

	// Use 'go run' to build and run the resulting binary in a single step,
	// instead of invoking 'go build' and the resulting binary separately, so that
	// this test can pass on mobile builders, which do not copy artifacts back
	// from remote invocations.
	cmd := exec.Command(testenv.GoToolPath(t), "run", ".")
	cmd.Env = append(os.Environ(), "GOPATH="+gopath)
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("%s [%s]: %v\n%s", strings.Join(cmd.Args, " "), dir, err, out)
	}
}
