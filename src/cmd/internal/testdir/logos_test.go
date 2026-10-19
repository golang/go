// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testdir_test

// loGo/s harness for GOROOT/test/logos (M0-S6).
//
// Directory map, stated once: files ending in .lgo run with the logos
// binary (bin/logos beside the go tool under test); files ending in .go
// under test/logos/** run with go under the same directives, except under
// test/logos/lint/, where they run with `logos lint` (M9-S2 owns that
// subcommand and its first files). A .go twin that needs an optimization
// tier carries `// run -O=<level>`; routeLogosHarness sends it here and it
// builds through `go build` with the level intact, never through `go run`
// or `go tool compile`, and never to logos (a -O belongs to cmd/go).
// cmd/go does not accept -O yet: M13-S1 owns enabling it and the first -O
// twin, so no committed file uses -O today.
//
// testdir_test.go carries four hook lines for this harness: the "logos"
// directory entry, the testFilesForDir enumeration call, the runLogos
// branch in run(), and the testOutputStem call in checkExpectedOutput.
// Every upstream case body stays as it is. Every .lgo action below stages
// the file in a throwaway module and drives it with the logos binary, so
// language selection (LogosMode parsing, Config.Logos checking) matches
// real use. `go tool compile` is never used for .lgo files: it would
// bypass logos file recognition and break spec mapping.

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"io/fs"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
)

// testFilesForDir lists the test files for one Test directory. Ordinary
// directories keep the upstream *.go listing; "logos" is walked
// recursively so test/logos/** files run as Test/logos/<relpath>.
func testFilesForDir(t *testing.T, dir string) []string {
	if dir != "logos" {
		return goFiles(t, dir)
	}
	root := filepath.Join(testenv.GOROOT(t), "test", "logos")
	var names []string
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || strings.HasPrefix(d.Name(), ".") {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if strings.HasSuffix(rel, ".go") || strings.HasSuffix(rel, ".lgo") {
			if shardMatch(rel) {
				names = append(names, rel)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	slices.Sort(names)
	return names
}

// logosToolPath returns the logos binary beside the go tool under test.
// The M0-S1 dist step installs $GOROOT/bin/logos next to $GOROOT/bin/go;
// without a rebuilt toolchain the logos subtests cannot run.
func logosToolPath(t *testing.T) string {
	t.Helper()
	name := "logos"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	p := filepath.Join(filepath.Dir(goTool), name)
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("logos binary not found at %s: %v (rebuild the toolchain with make.bash)", p, err)
	}
	return p
}

// testOutputStem strips a Go or loGo/s source extension so
// checkExpectedOutput maps hello.lgo to hello.out exactly as it maps
// hello.go to hello.out. Behavior for .go is unchanged.
func testOutputStem(filename string) string {
	if strings.HasSuffix(filename, ".lgo") {
		return filename[:len(filename)-len(".lgo")]
	}
	if strings.HasSuffix(filename, ".go") {
		return filename[:len(filename)-len(".go")]
	}
	return filename
}

// routeLogosHarness reports whether a test/logos case runs through
// runLogos: .lgo files (logos binary), test/logos/lint/ files
// (`logos lint`, M9-S2), and .go twins carrying -O=<level> under a `run`
// recipe, which build through `go build`. Every other .go twin keeps the
// upstream handlers, so its flags keep flowing to `go run`/`go build`.
func routeLogosHarness(goFile, action string, flags []string) bool {
	if strings.HasSuffix(goFile, ".lgo") || strings.HasPrefix(goFile, "lint/") {
		return true
	}
	if action != "run" || !strings.HasSuffix(goFile, ".go") {
		return false
	}
	for _, f := range flags {
		if strings.HasPrefix(f, "-O=") {
			return true
		}
	}
	return false
}

// logosCase bundles one parsed test/logos recipe for runLogos.
type logosCase struct {
	action string
	args   []string
	flags  []string
	runenv []string
	src    []byte
	auto   bool
	tim    int
}

// runLogos runs one test/logos case. The file is staged under its base
// name in a throwaway module; .lgo actions drive the logos binary, and
// test/logos/lint/*.go files drive `logos lint` (M9-S2).
func (t test) runLogos(c logosCase) error {
	base := path.Base(t.goFile) // t.goFile uses slashes: "run/hello.lgo"
	isLgo := strings.HasSuffix(base, ".lgo")

	tempDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(tempDir, base), c.src, 0644); err != nil {
		t.Fatal(err)
	}

	runcmd := func(args ...string) ([]byte, error) {
		cmd := exec.Command(args[0], args[1:]...)
		var buf bytes.Buffer
		cmd.Stdout = &buf
		cmd.Stderr = &buf
		cmd.Env = append(os.Environ(), "GOENV=off", "GOFLAGS=")
		cmd.Dir = tempDir
		// Set PWD to match Dir to speed up os.Getwd in the child process.
		cmd.Env = append(cmd.Env, "PWD="+tempDir)
		cmd.Env = append(cmd.Env, "STDLIB_IMPORTCFG="+stdlibImportcfgFile())
		cmd.Env = append(cmd.Env, c.runenv...)

		var err error
		if c.tim != 0 {
			err = cmd.Start()
			if err == nil {
				tick := time.NewTimer(time.Duration(c.tim) * time.Second)
				done := make(chan error)
				go func() {
					done <- cmd.Wait()
				}()
				select {
				case err = <-done:
					// ok
				case <-tick.C:
					cmd.Process.Signal(os.Interrupt)
					time.Sleep(1 * time.Second)
					cmd.Process.Kill()
					<-done
					err = errTimeout
				}
				tick.Stop()
			}
		} else {
			err = cmd.Run()
		}
		if err != nil && err != errTimeout {
			err = fmt.Errorf("%s\n%s", err, buf.Bytes())
		}
		return buf.Bytes(), err
	}

	// Split -O=<level> out: it is a go-build-only flag for the .go twins
	// (M13-S1 owns cmd/go acceptance). It is never passed to logos, and
	// never reaches `go run` or `go tool compile`: .go twins carrying it
	// take the `go build` path below.
	var buildFlags, optFlags []string
	for _, f := range c.flags {
		if strings.HasPrefix(f, "-O=") {
			optFlags = append(optFlags, f)
		} else {
			buildFlags = append(buildFlags, f)
		}
	}

	// test/logos/lint/ belongs to M9-S2: its .go files run with
	// `logos lint`, never with go. M9-S2 owns the exact subcommand
	// shape and the first files; this routing only reserves it.
	if strings.HasPrefix(t.goFile, "lint/") {
		out, err := runcmd(append([]string{logosToolPath(t.T), "lint", base}, buildFlags...)...)
		if err != nil {
			return err
		}
		return t.checkExpectedOutput(out)
	}

	if !isLgo {
		// .go twin carrying -O=<level> (routeLogosHarness sends only
		// those here): build through `go build` with the level intact,
		// then run the binary, mirroring the upstream buildrun recipe.
		cmd := []string{goTool, "build", t.goGcflags(), "-o", "a.exe"}
		cmd = append(cmd, optFlags...)
		cmd = append(cmd, filepath.Join(tempDir, base))
		if _, err := runcmd(cmd...); err != nil {
			return err
		}
		out, err := runcmd(append([]string{"./a.exe"}, c.args...)...)
		if err != nil {
			return err
		}
		return t.checkExpectedOutput(out)
	}

	if err := os.WriteFile(filepath.Join(tempDir, "go.mod"), []byte("module logostest\n\ngo 1.27\n"), 0644); err != nil {
		t.Fatal(err)
	}

	switch c.action {
	case "run":
		cmd := []string{logosToolPath(t.T), "run", t.goGcflags()}
		cmd = append(cmd, buildFlags...)
		cmd = append(cmd, ".")
		cmd = append(cmd, c.args...)
		out, err := runcmd(cmd...)
		if err != nil {
			return err
		}
		return t.checkExpectedOutput(out)

	case "compile", "build":
		cmd := []string{logosToolPath(t.T), "build"}
		cmd = append(cmd, buildFlags...)
		if c.action == "build" {
			cmd = append(cmd, "-o", "a.exe")
		}
		cmd = append(cmd, ".")
		_, err := runcmd(cmd...)
		return err

	case "errorcheck":
		cmd := []string{logosToolPath(t.T), "build"}
		cmd = append(cmd, buildFlags...)
		cmd = append(cmd, ".")
		out, err := runcmd(cmd...)
		if err == nil {
			return fmt.Errorf("compilation succeeded unexpectedly\n%s", out)
		}
		full := filepath.Join(tempDir, base)
		// Cover both the bare and "./"-prefixed spellings a build may
		// use for the file name; the second pair's full side names no
		// readable file and contributes no ERROR entries, only prefix
		// normalization.
		return t.errorCheck(string(out), c.auto, full, base, "./"+base, base)

	case "runoutput":
		cmd := []string{logosToolPath(t.T), "run", t.goGcflags()}
		cmd = append(cmd, buildFlags...)
		cmd = append(cmd, ".")
		out, err := runcmd(cmd...)
		if err != nil {
			return err
		}
		tfile := filepath.Join(tempDir, "tmp__.go")
		if err := os.WriteFile(tfile, out, 0666); err != nil {
			t.Fatalf("write tempfile: %v", err)
		}
		cmd = []string{goTool, "run", t.goGcflags()}
		cmd = append(cmd, tfile)
		out, err = runcmd(cmd...)
		if err != nil {
			return err
		}
		return t.checkExpectedOutput(out)

	default:
		t.Fatalf("unknown logos pattern: %q", c.action)
		panic("unreachable")
	}
}
