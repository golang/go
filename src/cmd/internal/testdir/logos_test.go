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
// tier carries `// run -O=<level>`; it builds through `go build` with the
// level intact alongside every other directive flag, never through
// `go run` or `go tool compile`, and never to logos (a -O belongs to
// cmd/go). cmd/go does not accept -O yet: M13-S1 owns enabling it and the
// first -O twin, so no committed file uses -O today.
//
// testdir_test.go carries hook lines for this harness: the "logos"
// directory entry, the testFilesForDir enumeration call, the runLogosCase
// branch in run(), and the testOutputStem call in checkExpectedOutput.
// Every upstream case body stays as it is. Every .lgo action below stages
// the file in a throwaway module and drives it with the logos binary, so
// language selection matches real use. `go tool compile` is never used
// for .lgo files: it would bypass logos file recognition and break spec
// mapping. .lgo compile and errorcheck compiler flags travel as -gcflags
// (always with -e, plus the ssa/check injection for errorcheck), because
// `logos build` does not define compiler flags such as -m or -d.

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

// routeLogosHarness reports whether a test case runs through runLogos.
// Only test/logos cases route here (dir == "logos"), so plain go keeps
// upstream behavior everywhere else (I3): .lgo files run with the logos
// binary, test/logos/lint/ files run with `logos lint` (M9-S2), and every
// other .go twin under test/logos runs with go through runLogos (C1),
// where its base name is staged. Twins carrying -O=<level> build through
// `go build` with the level intact.
func routeLogosHarness(dir, goFile, action string, flags []string) bool {
	if dir != "logos" {
		return false
	}
	// Only single-file actions run here; directory actions (*dir, rundir,
	// runindir) and the remaining recipes keep the upstream handlers,
	// which read their inputs from the source tree rather than staging
	// one file.
	switch action {
	case "run", "compile", "build", "errorcheck", "runoutput":
	default:
		return false
	}
	if strings.HasSuffix(goFile, ".lgo") || strings.HasPrefix(goFile, "lint/") {
		return true
	}
	if strings.HasSuffix(goFile, ".go") {
		return true
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

// runLogosCase runs one test/logos recipe through runLogos. It exists so
// the hook in testdir_test.go stays a single call (M6).
func (t test) runLogosCase(action string, args, flags, runenv []string, src []byte, auto bool, tim int) error {
	return t.runLogos(logosCase{
		action: action,
		args:   args,
		flags:  flags,
		runenv: runenv,
		src:    src,
		auto:   auto,
		tim:    tim,
	})
}

// splitOptFlags splits -O=<level> out of directive flags: -O is a
// go-build-only flag for the .go twins (M13-S1 owns cmd/go acceptance).
// It is never passed to logos, and never reaches `go run` or
// `go tool compile`.
func splitOptFlags(flags []string) (buildFlags, optFlags []string) {
	for _, f := range flags {
		if strings.HasPrefix(f, "-O=") {
			optFlags = append(optFlags, f)
		} else {
			buildFlags = append(buildFlags, f)
		}
	}
	return buildFlags, optFlags
}

// logosGCFlags builds the single -gcflags value for .lgo compile and
// errorcheck (I1): `logos build` does not define compiler flags such as
// -m or -d=..., so they must travel as -gcflags. -e is always present so
// errors past the 10-error cutoff still match ERROR comments; the shared
// GO_GCFLAGS base stays in the same value so the standard library is not
// rebuilt.
func logosGCFlags(compilerFlags []string) string {
	base := strings.TrimSpace(os.Getenv("GO_GCFLAGS"))
	parts := make([]string, 0, len(compilerFlags)+2)
	if base != "" {
		parts = append(parts, base)
	}
	hasE := false
	for _, f := range compilerFlags {
		if f == "-e" {
			hasE = true
			break
		}
	}
	if !hasE {
		parts = append(parts, "-e")
	}
	parts = append(parts, compilerFlags...)
	return "-gcflags=all=" + strings.Join(parts, " ")
}

// logosGoBuildCmd builds the `go build` argv for a .go twin carrying
// -O=<level> (I2): the -O flags and every other directive flag reach the
// build, before the file argument.
func logosGoBuildCmd(gcflags, file string, buildFlags, optFlags []string) []string {
	cmd := []string{goTool, "build", gcflags, "-o", "a.exe"}
	cmd = append(cmd, optFlags...)
	cmd = append(cmd, buildFlags...)
	cmd = append(cmd, file)
	return cmd
}

// runLogos runs one test/logos case. The file is staged under its base
// name in a throwaway module; .lgo actions drive the logos binary, .go
// twins drive go, and test/logos/lint/*.go files drive `logos lint`
// (M9-S2).
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

	buildFlags, optFlags := splitOptFlags(c.flags)

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
		return t.runLogosGo(c, base, tempDir, buildFlags, optFlags, runcmd)
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

	case "compile":
		cmd := []string{logosToolPath(t.T), "build", logosGCFlags(buildFlags), "."}
		_, err := runcmd(cmd...)
		return err

	case "build":
		cmd := []string{logosToolPath(t.T), "build"}
		cmd = append(cmd, buildFlags...)
		cmd = append(cmd, "-o", "a.exe", ".")
		_, err := runcmd(cmd...)
		return err

	case "errorcheck":
		flags := slices.Clone(buildFlags)
		found := false
		for i, f := range flags {
			if strings.HasPrefix(f, "-d=") {
				flags[i] = f + ",ssa/check/on"
				found = true
				break
			}
		}
		if !found {
			flags = append(flags, "-d=ssa/check/on")
		}
		cmd := []string{logosToolPath(t.T), "build", logosGCFlags(flags), "."}
		out, err := runcmd(cmd...)
		if err == nil {
			return fmt.Errorf("compilation succeeded unexpectedly\n%s", out)
		}
		full := filepath.Join(tempDir, base)
		// Keep the "./"-prefixed pair: the unmatched-error filter in
		// errorCheck uses HasPrefix (not matchPrefix), so without it
		// "./"-prefixed leftovers would drop silently instead of
		// failing the test.
		return t.errorCheck(string(out), c.auto, full, base, "./"+base, base)

	case "runoutput":
		t.runoutputGate <- true
		defer func() { <-t.runoutputGate }()
		cmd := []string{logosToolPath(t.T), "run", t.goGcflags()}
		cmd = append(cmd, buildFlags...)
		cmd = append(cmd, ".")
		cmd = append(cmd, c.args...)
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

// runLogosGo runs a .go twin under test/logos with go (C1). The file is
// already staged under its base name: the upstream path would write
// tempDir/run/x.go with only tempDir/test present and fail with ENOENT,
// so every twin routes here. Twins carrying -O=<level> build through
// `go build` with the level and every other directive flag intact (I2);
// plain twins mirror the upstream single-file recipes with the staged
// base path.
func (t test) runLogosGo(c logosCase, base, tempDir string, buildFlags, optFlags []string, runcmd runCmd) error {
	if len(optFlags) > 0 {
		cmd := logosGoBuildCmd(t.goGcflags(), filepath.Join(tempDir, base), buildFlags, optFlags)
		if _, err := runcmd(cmd...); err != nil {
			return err
		}
		out, err := runcmd(append([]string{"./a.exe"}, c.args...)...)
		if err != nil {
			return err
		}
		return t.checkExpectedOutput(out)
	}

	full := filepath.Join(tempDir, base)
	switch c.action {
	case "run":
		cmd := []string{goTool, "run", t.goGcflags()}
		cmd = append(cmd, buildFlags...)
		cmd = append(cmd, full)
		cmd = append(cmd, c.args...)
		out, err := runcmd(cmd...)
		if err != nil {
			return err
		}
		return t.checkExpectedOutput(out)

	case "build":
		cmd := []string{goTool, "build", t.goGcflags()}
		cmd = append(cmd, buildFlags...)
		cmd = append(cmd, "-o", "a.exe", full)
		_, err := runcmd(cmd...)
		return err

	case "compile":
		cmd := []string{goTool, "tool", "compile", "-e", "-p=p", "-importcfg=" + stdlibImportcfgFile()}
		cmd = append(cmd, buildFlags...)
		cmd = append(cmd, full)
		_, err := runcmd(cmd...)
		return err

	case "errorcheck":
		flags := slices.Clone(buildFlags)
		found := false
		for i, f := range flags {
			if strings.HasPrefix(f, "-d=") {
				flags[i] = f + ",ssa/check/on"
				found = true
				break
			}
		}
		if !found {
			flags = append(flags, "-d=ssa/check/on")
		}
		cmdline := []string{goTool, "tool", "compile", "-p=p", "-d=panic", "-C", "-e", "-importcfg=" + stdlibImportcfgFile(), "-o", "a.o"}
		cmdline = append(cmdline, flags...)
		cmdline = append(cmdline, full)
		out, err := runcmd(cmdline...)
		if err == nil {
			return fmt.Errorf("compilation succeeded unexpectedly\n%s", out)
		}
		return t.errorCheck(string(out), c.auto, full, base)

	case "runoutput":
		t.runoutputGate <- true
		defer func() { <-t.runoutputGate }()
		cmd := []string{goTool, "run", t.goGcflags()}
		cmd = append(cmd, buildFlags...)
		cmd = append(cmd, full)
		cmd = append(cmd, c.args...)
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
		t.Fatalf("unknown logos go pattern: %q", c.action)
		panic("unreachable")
	}
}

func TestRouteLogosHarnessScopesToLogos(t *testing.T) {
	// I3: -O= outside test/logos keeps upstream behavior.
	if routeLogosHarness("fixedbugs", "bug000.go", "run", []string{"-O=3"}) {
		t.Fatal("routeLogosHarness routed an upstream -O= file to the logos harness")
	}
	// C1: every .go twin under test/logos routes to runLogos, not only -O.
	if !routeLogosHarness("logos", "run/twin_go.go", "run", nil) {
		t.Fatal("routeLogosHarness did not route a plain test/logos .go twin")
	}
	if !routeLogosHarness("logos", "run/x.go", "run", []string{"-O=3"}) {
		t.Fatal("routeLogosHarness did not route a -O= twin")
	}
	if !routeLogosHarness("logos", "run/hello.lgo", "run", nil) {
		t.Fatal("routeLogosHarness did not route a .lgo file")
	}
	// Directory actions keep the upstream handlers, which read from the
	// source tree instead of staging one file.
	if routeLogosHarness("logos", "run/x.go", "rundir", nil) {
		t.Fatal("routeLogosHarness routed a directory action to the logos harness")
	}
	if routeLogosHarness("logos", "run/x.lgo", "compile", nil) == false {
		t.Fatal("routeLogosHarness did not route a .lgo compile action")
	}
}

func TestLogosGoBuildPreservesFlags(t *testing.T) {
	// I2: the -O build keeps every other directive flag before the file.
	cmd := logosGoBuildCmd("-gcflags=all=", "x.go", []string{"-gcflags=-l"}, []string{"-O=3"})
	foundO, foundGC, fileLast := false, false, false
	for i, a := range cmd {
		if a == "-O=3" {
			foundO = true
		}
		if a == "-gcflags=-l" {
			foundGC = true
		}
		if a == "x.go" && i == len(cmd)-1 {
			fileLast = true
		}
	}
	if !foundO || !foundGC || !fileLast {
		t.Fatalf("logosGoBuildCmd dropped flags: %q", cmd)
	}
}

func TestLogosGCFlagsHasE(t *testing.T) {
	// I1: compiler flags travel as -gcflags with -e always present.
	got := logosGCFlags([]string{"-m"})
	if !strings.HasPrefix(got, "-gcflags=") || !strings.Contains(got, "-e") || !strings.Contains(got, "-m") {
		t.Fatalf("logosGCFlags = %q, want -gcflags with -e and -m", got)
	}
}
