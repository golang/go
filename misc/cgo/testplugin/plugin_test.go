// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package plugin_test

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var gcflags string = os.Getenv("GO_GCFLAGS")

func TestMain(m *testing.M) {
	flag.Parse()
	if testing.Short() && os.Getenv("GO_BUILDER_NAME") == "" {
		fmt.Printf("SKIP - short mode and $GO_BUILDER_NAME not set\n")
		os.Exit(0)
	}
	log.SetFlags(log.Lshortfile)
	os.Exit(testMain(m))
}

func testMain(m *testing.M) int {
	// Copy testdata into GOPATH/src/testplugin, along with a go.mod file
	// declaring the same path.

	GOPATH, err := ioutil.TempDir("", "plugin_test")
	if err != nil {
		log.Panic(err)
	}
	defer os.RemoveAll(GOPATH)

	modRoot := filepath.Join(GOPATH, "src", "testplugin")
	altRoot := filepath.Join(GOPATH, "alt", "src", "testplugin")
	for srcRoot, dstRoot := range map[string]string{
		"testdata":                           modRoot,
		filepath.Join("altpath", "testdata"): altRoot,
	} {
		if err := overlayDir(dstRoot, srcRoot); err != nil {
			log.Panic(err)
		}
		if err := ioutil.WriteFile(filepath.Join(dstRoot, "go.mod"), []byte("module testplugin\n"), 0666); err != nil {
			log.Panic(err)
		}
	}

	os.Setenv("GOPATH", filepath.Join(GOPATH, "alt"))
	if err := os.Chdir(altRoot); err != nil {
		log.Panic(err)
	}
	os.Setenv("PWD", altRoot)
	goCmd(nil, "build", "-buildmode=plugin", "-o", filepath.Join(modRoot, "plugin-mismatch.so"), "./plugin-mismatch")

	os.Setenv("GOPATH", GOPATH)
	if err := os.Chdir(modRoot); err != nil {
		log.Panic(err)
	}
	os.Setenv("PWD", modRoot)

	os.Setenv("LD_LIBRARY_PATH", modRoot)

	goCmd(nil, "build", "-buildmode=plugin", "./plugin1")
	goCmd(nil, "build", "-buildmode=plugin", "./plugin2")
	so, err := ioutil.ReadFile("plugin2.so")
	if err != nil {
		log.Panic(err)
	}
	if err := ioutil.WriteFile("plugin2-dup.so", so, 0444); err != nil {
		log.Panic(err)
	}

	goCmd(nil, "build", "-buildmode=plugin", "-o=sub/plugin1.so", "./sub/plugin1")
	goCmd(nil, "build", "-buildmode=plugin", "-o=unnamed1.so", "./unnamed1/main.go")
	goCmd(nil, "build", "-buildmode=plugin", "-o=unnamed2.so", "./unnamed2/main.go")
	goCmd(nil, "build", "-o", "host.exe", "./host")

	return m.Run()
}

func goCmd(t *testing.T, op string, args ...string) {
	if t != nil {
		t.Helper()
	}
	run(t, "go", append([]string{op, "-gcflags", gcflags}, args...)...)
}

func run(t *testing.T, bin string, args ...string) string {
	cmd := exec.Command(bin, args...)
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
	goCmd(t, "run", "./checkdwarf/main.go", "plugin2.so", "plugin2.UnexportedNameReuse")
	goCmd(t, "run", "./checkdwarf/main.go", "./host.exe", "main.main")
}

func TestRunHost(t *testing.T) {
	run(t, "./host.exe")
}

func TestUniqueTypesAndItabs(t *testing.T) {
	goCmd(t, "build", "-buildmode=plugin", "./iface_a")
	goCmd(t, "build", "-buildmode=plugin", "./iface_b")
	goCmd(t, "build", "-o", "iface.exe", "./iface")
	run(t, "./iface.exe")
}

func TestIssue18676(t *testing.T) {
	// make sure we don't add the same itab twice.
	// The buggy code hangs forever, so use a timeout to check for that.
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
	goCmd(t, "build", "-buildmode=plugin", "-ldflags='-pluginpath=issue.19534'", "-o", "plugin.so", "./issue19534/plugin.go")
	goCmd(t, "build", "-o", "issue19534.exe", "./issue19534/main.go")
	run(t, "./issue19534.exe")
}

func TestIssue18584(t *testing.T) {
	goCmd(t, "build", "-buildmode=plugin", "-o", "plugin.so", "./issue18584/plugin.go")
	goCmd(t, "build", "-o", "issue18584.exe", "./issue18584/main.go")
	run(t, "./issue18584.exe")
}

func TestIssue19418(t *testing.T) {
	goCmd(t, "build", "-buildmode=plugin", "-ldflags=-X main.Val=linkstr", "-o", "plugin.so", "./issue19418/plugin.go")
	goCmd(t, "build", "-o", "issue19418.exe", "./issue19418/main.go")
	run(t, "./issue19418.exe")
}

func TestIssue19529(t *testing.T) {
	goCmd(t, "build", "-buildmode=plugin", "-o", "plugin.so", "./issue19529/plugin.go")
}

func TestIssue22175(t *testing.T) {
	goCmd(t, "build", "-buildmode=plugin", "-o", "issue22175_plugin1.so", "./issue22175/plugin1.go")
	goCmd(t, "build", "-buildmode=plugin", "-o", "issue22175_plugin2.so", "./issue22175/plugin2.go")
	goCmd(t, "build", "-o", "issue22175.exe", "./issue22175/main.go")
	run(t, "./issue22175.exe")
}

func TestIssue22295(t *testing.T) {
	goCmd(t, "build", "-buildmode=plugin", "-o", "issue.22295.so", "./issue22295.pkg")
	goCmd(t, "build", "-o", "issue22295.exe", "./issue22295.pkg/main.go")
	run(t, "./issue22295.exe")
}

func TestIssue24351(t *testing.T) {
	goCmd(t, "build", "-buildmode=plugin", "-o", "issue24351.so", "./issue24351/plugin.go")
	goCmd(t, "build", "-o", "issue24351.exe", "./issue24351/main.go")
	run(t, "./issue24351.exe")
}

func TestIssue25756(t *testing.T) {
	goCmd(t, "build", "-buildmode=plugin", "-o", "life.so", "./issue25756/plugin")
	goCmd(t, "build", "-o", "issue25756.exe", "./issue25756/main.go")
	// Fails intermittently, but 20 runs should cause the failure
	for n := 20; n > 0; n-- {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			t.Parallel()
			run(t, "./issue25756.exe")
		})
	}
}
