// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for vendoring semantics.

package main_test

import (
	"bytes"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestVendorImports(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.setenv("GO15VENDOREXPERIMENT", "1")
	tg.run("list", "-f", "{{.ImportPath}} {{.Imports}}", "vend/...")
	want := `
		vend [vend/vendor/p r]
		vend/hello [fmt vend/vendor/strings]
		vend/subdir [vend/vendor/p r]
		vend/vendor/p []
		vend/vendor/q []
		vend/vendor/strings []
		vend/x [vend/x/vendor/p vend/vendor/q vend/x/vendor/r]
		vend/x/invalid [vend/x/invalid/vendor/foo]
		vend/x/vendor/p []
		vend/x/vendor/p/p [notfound]
		vend/x/vendor/r []
	`
	want = strings.Replace(want+"\t", "\n\t\t", "\n", -1)
	want = strings.TrimPrefix(want, "\n")

	have := tg.stdout.String()

	if have != want {
		t.Errorf("incorrect go list output:\n%s", diffSortedOutputs(have, want))
	}
}

func TestVendorRun(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.setenv("GO15VENDOREXPERIMENT", "1")
	tg.cd(filepath.Join(tg.pwd(), "testdata/src/vend/hello"))
	tg.run("run", "hello.go")
	tg.grepStdout("hello, world", "missing hello world output")
}

func TestVendorGOPATH(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	changeVolume := func(s string, f func(s string) string) string {
		vol := filepath.VolumeName(s)
		return f(vol) + s[len(vol):]
	}
	gopath := changeVolume(filepath.Join(tg.pwd(), "testdata"), strings.ToLower)
	tg.setenv("GOPATH", gopath)
	tg.setenv("GO15VENDOREXPERIMENT", "1")
	cd := changeVolume(filepath.Join(tg.pwd(), "testdata/src/vend/hello"), strings.ToUpper)
	tg.cd(cd)
	tg.run("run", "hello.go")
	tg.grepStdout("hello, world", "missing hello world output")
}

func TestVendorTest(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.setenv("GO15VENDOREXPERIMENT", "1")
	tg.cd(filepath.Join(tg.pwd(), "testdata/src/vend/hello"))
	tg.run("test", "-v")
	tg.grepStdout("TestMsgInternal", "missing use in internal test")
	tg.grepStdout("TestMsgExternal", "missing use in external test")
}

func TestVendorInvalid(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.setenv("GO15VENDOREXPERIMENT", "1")

	tg.runFail("build", "vend/x/invalid")
	tg.grepStderr("must be imported as foo", "missing vendor import error")
}

func TestVendorImportError(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.setenv("GO15VENDOREXPERIMENT", "1")

	tg.runFail("build", "vend/x/vendor/p/p")

	re := regexp.MustCompile(`cannot find package "notfound" in any of:
	.*[\\/]testdata[\\/]src[\\/]vend[\\/]x[\\/]vendor[\\/]notfound \(vendor tree\)
	.*[\\/]testdata[\\/]src[\\/]vend[\\/]vendor[\\/]notfound \(vendor tree\)
	.*[\\/]src[\\/]notfound \(from \$GOROOT\)
	.*[\\/]testdata[\\/]src[\\/]notfound \(from \$GOPATH\)`)

	if !re.MatchString(tg.stderr.String()) {
		t.Errorf("did not find expected search list in error text")
	}
}

// diffSortedOutput prepares a diff of the already sorted outputs haveText and wantText.
// The diff shows common lines prefixed by a tab, lines present only in haveText
// prefixed by "unexpected: ", and lines present only in wantText prefixed by "missing: ".
func diffSortedOutputs(haveText, wantText string) string {
	var diff bytes.Buffer
	have := splitLines(haveText)
	want := splitLines(wantText)
	for len(have) > 0 || len(want) > 0 {
		if len(want) == 0 || len(have) > 0 && have[0] < want[0] {
			fmt.Fprintf(&diff, "unexpected: %s\n", have[0])
			have = have[1:]
			continue
		}
		if len(have) == 0 || len(want) > 0 && want[0] < have[0] {
			fmt.Fprintf(&diff, "missing: %s\n", want[0])
			want = want[1:]
			continue
		}
		fmt.Fprintf(&diff, "\t%s\n", want[0])
		want = want[1:]
		have = have[1:]
	}
	return diff.String()
}

func splitLines(s string) []string {
	x := strings.Split(s, "\n")
	if x[len(x)-1] == "" {
		x = x[:len(x)-1]
	}
	return x
}
