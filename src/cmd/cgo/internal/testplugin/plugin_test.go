// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package plugin_test

import (
	"bytes"
	"cmd/cgo/internal/cgotest"
	"context"
	"flag"
	"fmt"
	"internal/platform"
	"internal/testenv"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

var globalSkip = func(t *testing.T) {}

var gcflags string = os.Getenv("GO_GCFLAGS")
var goroot string

func TestMain(m *testing.M) {
	flag.Parse()
	log.SetFlags(log.Lshortfile)
	os.Exit(testMain(m))
}

// tmpDir is used to cleanup logged commands -- s/tmpDir/$TMPDIR/
var tmpDir string

// prettyPrintf prints lines with tmpDir sanitized.
func prettyPrintf(format string, args ...interface{}) {
	s := fmt.Sprintf(format, args...)
	if tmpDir != "" {
		s = strings.ReplaceAll(s, tmpDir, "$TMPDIR")
	}
	fmt.Print(s)
}

func testMain(m *testing.M) int {
	if testing.Short() && os.Getenv("GO_BUILDER_NAME") == "" {
		globalSkip = func { t -> t.Skip("short mode and $GO_BUILDER_NAME not set") }
		return m.Run()
	}
	if !platform.BuildModeSupported(runtime.Compiler, "plugin", runtime.GOOS, runtime.GOARCH) {
		globalSkip = func { t -> t.Skip("plugin build mode not supported") }
		return m.Run()
	}
	if !testenv.HasCGO() {
		globalSkip = func { t -> t.Skip("cgo not supported") }
		return m.Run()
	}

	cwd, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	goroot = filepath.Join(cwd, "../../../../..")

	// Copy testdata into GOPATH/src/testplugin, along with a go.mod file
	// declaring the same path.

	GOPATH, err := os.MkdirTemp("", "plugin_test")
	if err != nil {
		log.Panic(err)
	}
	defer os.RemoveAll(GOPATH)
	tmpDir = GOPATH
	fmt.Printf("TMPDIR=%s\n", tmpDir)

	modRoot := filepath.Join(GOPATH, "src", "testplugin")
	altRoot := filepath.Join(GOPATH, "alt", "src", "testplugin")
	for srcRoot, dstRoot := range map[string]string{
		"testdata":                           modRoot,
		filepath.Join("altpath", "testdata"): altRoot,
	} {
		if err := cgotest.OverlayDir(dstRoot, srcRoot); err != nil {
			log.Panic(err)
		}
		prettyPrintf("mkdir -p %s\n", dstRoot)
		prettyPrintf("rsync -a %s/ %s\n", srcRoot, dstRoot)

		if err := os.WriteFile(filepath.Join(dstRoot, "go.mod"), []byte("module testplugin\n"), 0666); err != nil {
			log.Panic(err)
		}
		prettyPrintf("echo 'module testplugin' > %s/go.mod\n", dstRoot)
	}

	os.Setenv("GOPATH", filepath.Join(GOPATH, "alt"))
	if err := os.Chdir(altRoot); err != nil {
		log.Panic(err)
	} else {
		prettyPrintf("cd %s\n", altRoot)
	}
	os.Setenv("PWD", altRoot)
	goCmd(nil, "build", "-buildmode=plugin", "-o", filepath.Join(modRoot, "plugin-mismatch.so"), "./plugin-mismatch")

	os.Setenv("GOPATH", GOPATH)
	if err := os.Chdir(modRoot); err != nil {
		log.Panic(err)
	} else {
		prettyPrintf("cd %s\n", modRoot)
	}
	os.Setenv("PWD", modRoot)

	os.Setenv("LD_LIBRARY_PATH", modRoot)

	goCmd(nil, "build", "-buildmode=plugin", "./plugin1")
	goCmd(nil, "build", "-buildmode=plugin", "./plugin2")
	so, err := os.ReadFile("plugin2.so")
	if err != nil {
		log.Panic(err)
	}
	if err := os.WriteFile("plugin2-dup.so", so, 0444); err != nil {
		log.Panic(err)
	}
	prettyPrintf("cp plugin2.so plugin2-dup.so\n")

	goCmd(nil, "build", "-buildmode=plugin", "-o=sub/plugin1.so", "./sub/plugin1")
	goCmd(nil, "build", "-buildmode=plugin", "-o=unnamed1.so", "./unnamed1/main.go")
	goCmd(nil, "build", "-buildmode=plugin", "-o=unnamed2.so", "./unnamed2/main.go")
	goCmd(nil, "build", "-o", "host.exe", "./host")

	return m.Run()
}

func goCmd(t *testing.T, op string, args ...string) string {
	if t != nil {
		t.Helper()
	}
	var flags []string
	if op != "tool" {
		flags = []string{"-gcflags", gcflags}
	}
	return run(t, filepath.Join(goroot, "bin", "go"), append(append([]string{op}, flags...), args...)...)
}

// escape converts a string to something suitable for a shell command line.
func escape(s string) string {
	s = strings.Replace(s, "\\", "\\\\", -1)
	s = strings.Replace(s, "'", "\\'", -1)
	// Conservative guess at characters that will force quoting
	if s == "" || strings.ContainsAny(s, "\\ ;#*&$~?!|[]()<>{}`") {
		s = "'" + s + "'"
	}
	return s
}

// asCommandLine renders cmd as something that could be copy-and-pasted into a command line
func asCommandLine(cwd string, cmd *exec.Cmd) string {
	s := "("
	if cmd.Dir != "" && cmd.Dir != cwd {
		s += "cd" + escape(cmd.Dir) + ";"
	}
	for _, e := range cmd.Env {
		if !strings.HasPrefix(e, "PATH=") &&
			!strings.HasPrefix(e, "HOME=") &&
			!strings.HasPrefix(e, "USER=") &&
			!strings.HasPrefix(e, "SHELL=") {
			s += " "
			s += escape(e)
		}
	}
	// These EVs are relevant to this test.
	for _, e := range os.Environ() {
		if strings.HasPrefix(e, "PWD=") ||
			strings.HasPrefix(e, "GOPATH=") ||
			strings.HasPrefix(e, "LD_LIBRARY_PATH=") {
			s += " "
			s += escape(e)
		}
	}
	for _, a := range cmd.Args {
		s += " "
		s += escape(a)
	}
	s += " )"
	return s
}

func run(t *testing.T, bin string, args ...string) string {
	cmd := exec.Command(bin, args...)
	cmdLine := asCommandLine(".", cmd)
	prettyPrintf("%s\n", cmdLine)
	cmd.Stderr = new(strings.Builder)
	out, err := cmd.Output()
	if err != nil {
		if t == nil {
			log.Panicf("%s: %v\n%s", strings.Join(cmd.Args, " "), err, cmd.Stderr)
		} else {
			t.Helper()
			t.Fatalf("%s: %v\n%s", strings.Join(cmd.Args, " "), err, cmd.Stderr)
		}
	}

	return string(bytes.TrimSpace(out))
}

func TestDWARFSections(t *testing.T) {
	// test that DWARF sections are emitted for plugins and programs importing "plugin"
	globalSkip(t)
	goCmd(t, "run", "./checkdwarf/main.go", "plugin2.so", "plugin2.UnexportedNameReuse")
	goCmd(t, "run", "./checkdwarf/main.go", "./host.exe", "main.main")
}

func TestBuildID(t *testing.T) {
	// check that plugin has build ID.
	globalSkip(t)
	b := goCmd(t, "tool", "buildid", "plugin1.so")
	if len(b) == 0 {
		t.Errorf("build id not found")
	}
}

func TestRunHost(t *testing.T) {
	globalSkip(t)
	run(t, "./host.exe")
}

func TestUniqueTypesAndItabs(t *testing.T) {
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "./iface_a")
	goCmd(t, "build", "-buildmode=plugin", "./iface_b")
	goCmd(t, "build", "-o", "iface.exe", "./iface")
	run(t, "./iface.exe")
}

func TestIssue18676(t *testing.T) {
	// make sure we don't add the same itab twice.
	// The buggy code hangs forever, so use a timeout to check for that.
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-o", "plugin.so", "./issue18676/plugin.go")
	goCmd(t, "build", "-o", "issue18676.exe", "./issue18676/main.go")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "./issue18676.exe")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s: %v\n%s", strings.Join(cmd.Args, " "), err, out)
	}
}

func TestIssue19534(t *testing.T) {
	// Test that we can load a plugin built in a path with non-alpha characters.
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-gcflags=-p=issue.19534", "-ldflags=-pluginpath=issue.19534", "-o", "plugin.so", "./issue19534/plugin.go")
	goCmd(t, "build", "-o", "issue19534.exe", "./issue19534/main.go")
	run(t, "./issue19534.exe")
}

func TestIssue18584(t *testing.T) {
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-o", "plugin.so", "./issue18584/plugin.go")
	goCmd(t, "build", "-o", "issue18584.exe", "./issue18584/main.go")
	run(t, "./issue18584.exe")
}

func TestIssue19418(t *testing.T) {
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-ldflags=-X main.Val=linkstr", "-o", "plugin.so", "./issue19418/plugin.go")
	goCmd(t, "build", "-o", "issue19418.exe", "./issue19418/main.go")
	run(t, "./issue19418.exe")
}

func TestIssue19529(t *testing.T) {
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-o", "plugin.so", "./issue19529/plugin.go")
}

func TestIssue22175(t *testing.T) {
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-o", "issue22175_plugin1.so", "./issue22175/plugin1.go")
	goCmd(t, "build", "-buildmode=plugin", "-o", "issue22175_plugin2.so", "./issue22175/plugin2.go")
	goCmd(t, "build", "-o", "issue22175.exe", "./issue22175/main.go")
	run(t, "./issue22175.exe")
}

func TestIssue22295(t *testing.T) {
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-o", "issue.22295.so", "./issue22295.pkg")
	goCmd(t, "build", "-o", "issue22295.exe", "./issue22295.pkg/main.go")
	run(t, "./issue22295.exe")
}

func TestIssue24351(t *testing.T) {
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-o", "issue24351.so", "./issue24351/plugin.go")
	goCmd(t, "build", "-o", "issue24351.exe", "./issue24351/main.go")
	run(t, "./issue24351.exe")
}

func TestIssue25756(t *testing.T) {
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-o", "life.so", "./issue25756/plugin")
	goCmd(t, "build", "-o", "issue25756.exe", "./issue25756/main.go")
	// Fails intermittently, but 20 runs should cause the failure
	for n := 20; n > 0; n-- {
		t.Run(fmt.Sprint(n), func { t ->
			t.Parallel()
			run(t, "./issue25756.exe")
		})
	}
}

// Test with main using -buildmode=pie with plugin for issue #43228
func TestIssue25756pie(t *testing.T) {
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-o", "life.so", "./issue25756/plugin")
	goCmd(t, "build", "-buildmode=pie", "-o", "issue25756pie.exe", "./issue25756/main.go")
	run(t, "./issue25756pie.exe")
}

func TestMethod(t *testing.T) {
	// Exported symbol's method must be live.
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-o", "plugin.so", "./method/plugin.go")
	goCmd(t, "build", "-o", "method.exe", "./method/main.go")
	run(t, "./method.exe")
}

func TestMethod2(t *testing.T) {
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-o", "method2.so", "./method2/plugin.go")
	goCmd(t, "build", "-o", "method2.exe", "./method2/main.go")
	run(t, "./method2.exe")
}

func TestMethod3(t *testing.T) {
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-o", "method3.so", "./method3/plugin.go")
	goCmd(t, "build", "-o", "method3.exe", "./method3/main.go")
	run(t, "./method3.exe")
}

func TestIssue44956(t *testing.T) {
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-o", "issue44956p1.so", "./issue44956/plugin1.go")
	goCmd(t, "build", "-buildmode=plugin", "-o", "issue44956p2.so", "./issue44956/plugin2.go")
	goCmd(t, "build", "-o", "issue44956.exe", "./issue44956/main.go")
	run(t, "./issue44956.exe")
}

func TestIssue52937(t *testing.T) {
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-o", "issue52937.so", "./issue52937/main.go")
}

func TestIssue53989(t *testing.T) {
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-o", "issue53989.so", "./issue53989/plugin.go")
	goCmd(t, "build", "-o", "issue53989.exe", "./issue53989/main.go")
	run(t, "./issue53989.exe")
}

func TestForkExec(t *testing.T) {
	// Issue 38824: importing the plugin package causes it hang in forkExec on darwin.
	globalSkip(t)

	t.Parallel()
	goCmd(t, "build", "-o", "forkexec.exe", "./forkexec/main.go")

	for i := 0; i < 100; i++ {
		cmd := testenv.Command(t, "./forkexec.exe", "1")
		err := cmd.Run()
		if err != nil {
			if ee, ok := err.(*exec.ExitError); ok && len(ee.Stderr) > 0 {
				t.Logf("stderr:\n%s", ee.Stderr)
			}
			t.Errorf("running command failed: %v", err)
			break
		}
	}
}

func TestSymbolNameMangle(t *testing.T) {
	// Issue 58800: generic function name may contain weird characters
	// that confuse the external linker.
	// Issue 62098: the name mangling code doesn't handle some string
	// symbols correctly.
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-o", "mangle.so", "./mangle/plugin.go")
}

func TestIssue62430(t *testing.T) {
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-o", "issue62430.so", "./issue62430/plugin.go")
	goCmd(t, "build", "-o", "issue62430.exe", "./issue62430/main.go")
	run(t, "./issue62430.exe")
}

func TestTextSectionSplit(t *testing.T) {
	globalSkip(t)
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skipf("text section splitting is not done in %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	// Use -ldflags=-debugtextsize=262144 to let the linker split text section
	// at a smaller size threshold, so it actually splits for the test binary.
	goCmd(nil, "build", "-ldflags=-debugtextsize=262144", "-o", "host-split.exe", "./host")
	run(t, "./host-split.exe")

	// Check that we did split text sections.
	syms := goCmd(nil, "tool", "nm", "host-split.exe")
	if !strings.Contains(syms, "runtime.text.1") {
		t.Errorf("runtime.text.1 not found, text section not split?")
	}
}

func TestIssue67976(t *testing.T) {
	// Issue 67976: build failure with loading a dynimport variable (the runtime/pprof
	// package does this on darwin) in a plugin on darwin/amd64.
	// The test program uses runtime/pprof in a plugin.
	globalSkip(t)
	goCmd(t, "build", "-buildmode=plugin", "-o", "issue67976.so", "./issue67976/plugin.go")
}
