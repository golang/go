// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package so_test

import (
	"cmd/cgo/internal/cgotest"
	"internal/testenv"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestSO(t *testing.T) {
	testSO(t, "so")
}

func TestSOVar(t *testing.T) {
	testSO(t, "sovar")
}

func testSO(t *testing.T, dir string) {
	if runtime.GOOS == "ios" {
		t.Skip("iOS disallows dynamic loading of user libraries")
	}
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveExec(t)
	testenv.MustHaveCGO(t)

	GOPATH, err := os.MkdirTemp("", "cgosotest")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(GOPATH)

	modRoot := filepath.Join(GOPATH, "src", "cgosotest")
	if err := cgotest.OverlayDir(modRoot, filepath.Join("testdata", dir)); err != nil {
		log.Panic(err)
	}
	if err := os.WriteFile(filepath.Join(modRoot, "go.mod"), []byte("module cgosotest\n"), 0666); err != nil {
		log.Panic(err)
	}

	cmd := exec.Command("go", "env", "CC", "GOGCCFLAGS")
	cmd.Dir = modRoot
	cmd.Stderr = new(strings.Builder)
	cmd.Env = append(os.Environ(), "GOPATH="+GOPATH)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("%s: %v\n%s", strings.Join(cmd.Args, " "), err, cmd.Stderr)
	}
	lines := strings.Split(string(out), "\n")
	if len(lines) != 3 || lines[2] != "" {
		t.Fatalf("Unexpected output from %s:\n%s", strings.Join(cmd.Args, " "), lines)
	}

	cc := lines[0]
	if cc == "" {
		t.Fatal("CC environment variable (go env CC) cannot be empty")
	}
	gogccflags := strings.Split(lines[1], " ")

	// build shared object
	ext := "so"
	args := append(gogccflags, "-shared")
	switch runtime.GOOS {
	case "darwin", "ios":
		ext = "dylib"
		args = append(args, "-undefined", "suppress", "-flat_namespace")
	case "windows":
		ext = "dll"
		args = append(args, "-DEXPORT_DLL")
		// At least in mingw-clang it is not permitted to just name a .dll
		// on the command line. You must name the corresponding import
		// library instead, even though the dll is used when the executable is run.
		args = append(args, "-Wl,-out-implib,libcgosotest.a")
	case "aix":
		ext = "so.1"
	}
	sofname := "libcgosotest." + ext
	args = append(args, "-o", sofname, "cgoso_c.c")

	cmd = exec.Command(cc, args...)
	cmd.Dir = modRoot
	cmd.Env = append(os.Environ(), "GOPATH="+GOPATH)
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s: %s\n%s", strings.Join(cmd.Args, " "), err, out)
	}
	t.Logf("%s:\n%s", strings.Join(cmd.Args, " "), out)

	if runtime.GOOS == "aix" {
		// Shared object must be wrapped by an archive
		cmd = exec.Command("ar", "-X64", "-q", "libcgosotest.a", "libcgosotest.so.1")
		cmd.Dir = modRoot
		out, err = cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("%s: %s\n%s", strings.Join(cmd.Args, " "), err, out)
		}
	}

	cmd = exec.Command("go", "build", "-o", "main.exe", "main.go")
	cmd.Dir = modRoot
	cmd.Env = append(os.Environ(), "GOPATH="+GOPATH)
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s: %s\n%s", strings.Join(cmd.Args, " "), err, out)
	}
	t.Logf("%s:\n%s", strings.Join(cmd.Args, " "), out)

	cmd = exec.Command("./main.exe")
	cmd.Dir = modRoot
	cmd.Env = append(os.Environ(), "GOPATH="+GOPATH)
	if runtime.GOOS != "windows" {
		s := "LD_LIBRARY_PATH"
		if runtime.GOOS == "darwin" || runtime.GOOS == "ios" {
			s = "DYLD_LIBRARY_PATH"
		}
		cmd.Env = append(os.Environ(), s+"=.")

		// On FreeBSD 64-bit architectures, the 32-bit linker looks for
		// different environment variables.
		if runtime.GOOS == "freebsd" && runtime.GOARCH == "386" {
			cmd.Env = append(cmd.Env, "LD_32_LIBRARY_PATH=.")
		}
	}
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s: %s\n%s", strings.Join(cmd.Args, " "), err, out)
	}
	t.Logf("%s:\n%s", strings.Join(cmd.Args, " "), out)
}
