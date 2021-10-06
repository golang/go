// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"debug/pe"
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestUndefinedRelocErrors(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// When external linking, symbols may be defined externally, so we allow
	// undefined symbols and let external linker resolve. Skip the test.
	testenv.MustInternalLink(t)

	t.Parallel()

	out, err := exec.Command(testenv.GoToolPath(t), "build", "./testdata/issue10978").CombinedOutput()
	if err == nil {
		t.Fatal("expected build to fail")
	}

	wantErrors := map[string]int{
		// Main function has dedicated error message.
		"function main is undeclared in the main package": 1,

		// Single error reporting per each symbol.
		// This way, duplicated messages are not reported for
		// multiple relocations with a same name.
		"main.defined1: relocation target main.undefined not defined": 1,
		"main.defined2: relocation target main.undefined not defined": 1,
	}
	unexpectedErrors := map[string]int{}

	for _, l := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(l, "#") || l == "" {
			continue
		}
		matched := ""
		for want := range wantErrors {
			if strings.Contains(l, want) {
				matched = want
				break
			}
		}
		if matched != "" {
			wantErrors[matched]--
		} else {
			unexpectedErrors[l]++
		}
	}

	for want, n := range wantErrors {
		switch {
		case n > 0:
			t.Errorf("unmatched error: %s (x%d)", want, n)
		case n < 0:
			t.Errorf("extra errors: %s (x%d)", want, -n)
		}
	}
	for unexpected, n := range unexpectedErrors {
		t.Errorf("unexpected error: %s (x%d)", unexpected, n)
	}
}

const carchiveSrcText = `
package main

//export GoFunc
func GoFunc() {
	println(42)
}

func main() {
}
`

func TestArchiveBuildInvokeWithExec(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)

	// run this test on just a small set of platforms (no need to test it
	// across the board given the nature of the test).
	pair := runtime.GOOS + "-" + runtime.GOARCH
	switch pair {
	case "darwin-amd64", "darwin-arm64", "linux-amd64", "freebsd-amd64":
	default:
		t.Skip("no need for test on " + pair)
	}
	switch runtime.GOOS {
	case "openbsd", "windows":
		t.Skip("c-archive unsupported")
	}
	dir := t.TempDir()

	srcfile := filepath.Join(dir, "test.go")
	arfile := filepath.Join(dir, "test.a")
	if err := ioutil.WriteFile(srcfile, []byte(carchiveSrcText), 0666); err != nil {
		t.Fatal(err)
	}

	ldf := fmt.Sprintf("-ldflags=-v -tmpdir=%s", dir)
	argv := []string{"build", "-buildmode=c-archive", "-o", arfile, ldf, srcfile}
	out, err := exec.Command(testenv.GoToolPath(t), argv...).CombinedOutput()
	if err != nil {
		t.Fatalf("build failure: %s\n%s\n", err, string(out))
	}

	found := false
	const want = "invoking archiver with syscall.Exec"
	for _, l := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(l, want) {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("expected '%s' in -v output, got:\n%s\n", want, string(out))
	}
}

func TestLargeTextSectionSplitting(t *testing.T) {
	switch runtime.GOARCH {
	case "ppc64", "ppc64le":
	case "arm64":
		if runtime.GOOS == "darwin" {
			break
		}
		fallthrough
	default:
		t.Skipf("text section splitting is not done in %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	t.Parallel()
	dir := t.TempDir()

	// NB: the use of -ldflags=-debugtextsize=1048576 tells the linker to
	// split text sections at a size threshold of 1M instead of the
	// architected limit of 67M or larger. The choice of building cmd/go
	// is arbitrary; we just need something sufficiently large that uses
	// external linking.
	exe := filepath.Join(dir, "go.exe")
	out, err := exec.Command(testenv.GoToolPath(t), "build", "-o", exe, "-ldflags=-linkmode=external -debugtextsize=1048576", "cmd/go").CombinedOutput()
	if err != nil {
		t.Fatalf("build failure: %s\n%s\n", err, string(out))
	}

	// Check that we did split text sections.
	out, err = exec.Command(testenv.GoToolPath(t), "tool", "nm", exe).CombinedOutput()
	if err != nil {
		t.Fatalf("nm failure: %s\n%s\n", err, string(out))
	}
	if !bytes.Contains(out, []byte("runtime.text.1")) {
		t.Errorf("runtime.text.1 not found, text section not split?")
	}

	// Result should be runnable.
	_, err = exec.Command(exe, "version").CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}
}

func TestWindowsBuildmodeCSharedASLR(t *testing.T) {
	platform := fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH)
	switch platform {
	case "windows/amd64", "windows/386":
	default:
		t.Skip("skipping windows amd64/386 only test")
	}

	testenv.MustHaveCGO(t)

	t.Run("aslr", func(t *testing.T) {
		testWindowsBuildmodeCSharedASLR(t, true)
	})
	t.Run("no-aslr", func(t *testing.T) {
		testWindowsBuildmodeCSharedASLR(t, false)
	})
}

func testWindowsBuildmodeCSharedASLR(t *testing.T, useASLR bool) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)

	dir := t.TempDir()

	srcfile := filepath.Join(dir, "test.go")
	objfile := filepath.Join(dir, "test.dll")
	if err := ioutil.WriteFile(srcfile, []byte(`package main; func main() { print("hello") }`), 0666); err != nil {
		t.Fatal(err)
	}
	argv := []string{"build", "-buildmode=c-shared"}
	if !useASLR {
		argv = append(argv, "-ldflags", "-aslr=false")
	}
	argv = append(argv, "-o", objfile, srcfile)
	out, err := exec.Command(testenv.GoToolPath(t), argv...).CombinedOutput()
	if err != nil {
		t.Fatalf("build failure: %s\n%s\n", err, string(out))
	}

	f, err := pe.Open(objfile)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var dc uint16
	switch oh := f.OptionalHeader.(type) {
	case *pe.OptionalHeader32:
		dc = oh.DllCharacteristics
	case *pe.OptionalHeader64:
		dc = oh.DllCharacteristics
		hasHEVA := (dc & pe.IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA) != 0
		if useASLR && !hasHEVA {
			t.Error("IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA flag is not set")
		} else if !useASLR && hasHEVA {
			t.Error("IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA flag should not be set")
		}
	default:
		t.Fatalf("unexpected optional header type of %T", f.OptionalHeader)
	}
	hasASLR := (dc & pe.IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE) != 0
	if useASLR && !hasASLR {
		t.Error("IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE flag is not set")
	} else if !useASLR && hasASLR {
		t.Error("IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE flag should not be set")
	}
}

// TestMemProfileCheck tests that cmd/link sets
// runtime.disableMemoryProfiling if the runtime.MemProfile
// symbol is unreachable after deadcode (and not dynlinking).
// The runtime then uses that to set the default value of
// runtime.MemProfileRate, which this test checks.
func TestMemProfileCheck(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	tests := []struct {
		name    string
		prog    string
		wantOut string
	}{
		{
			"no_memprofile",
			`
package main
import "runtime"
func main() {
	println(runtime.MemProfileRate)
}
`,
			"0",
		},
		{
			"with_memprofile",
			`
package main
import "runtime"
func main() {
	runtime.MemProfile(nil, false)
	println(runtime.MemProfileRate)
}
`,
			"524288",
		},
		{
			"with_memprofile_indirect",
			`
package main
import "runtime"
var f = runtime.MemProfile
func main() {
	if f == nil {
		panic("no f")
	}
	println(runtime.MemProfileRate)
}
`,
			"524288",
		},
		{
			"with_memprofile_runtime_pprof",
			`
package main
import "runtime"
import "runtime/pprof"
func main() {
        _ = pprof.Profiles()
	println(runtime.MemProfileRate)
}
`,
			"524288",
		},
		{
			"with_memprofile_http_pprof",
			`
package main
import "runtime"
import _ "net/http/pprof"
func main() {
	println(runtime.MemProfileRate)
}
`,
			"524288",
		},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			tempDir := t.TempDir()
			src := filepath.Join(tempDir, "x.go")
			if err := ioutil.WriteFile(src, []byte(tt.prog), 0644); err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command(testenv.GoToolPath(t), "run", src)
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatal(err)
			}
			got := strings.TrimSpace(string(out))
			if got != tt.wantOut {
				t.Errorf("got %q; want %q", got, tt.wantOut)
			}
		})
	}
}
