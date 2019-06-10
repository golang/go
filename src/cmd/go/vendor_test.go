// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for vendoring semantics.

package main_test

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestVendorImports(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("list", "-f", "{{.ImportPath}} {{.Imports}}", "vend/...", "vend/vendor/...", "vend/x/vendor/...")
	want := `
		vend [vend/vendor/p r]
		vend/dir1 []
		vend/hello [fmt vend/vendor/strings]
		vend/subdir [vend/vendor/p r]
		vend/x [vend/x/vendor/p vend/vendor/q vend/x/vendor/r vend/dir1 vend/vendor/vend/dir1/dir2]
		vend/x/invalid [vend/x/invalid/vendor/foo]
		vend/vendor/p []
		vend/vendor/q []
		vend/vendor/strings []
		vend/vendor/vend/dir1/dir2 []
		vend/x/vendor/p []
		vend/x/vendor/p/p [notfound]
		vend/x/vendor/r []
	`
	want = strings.ReplaceAll(want+"\t", "\n\t\t", "\n")
	want = strings.TrimPrefix(want, "\n")

	have := tg.stdout.String()

	if have != want {
		t.Errorf("incorrect go list output:\n%s", diffSortedOutputs(have, want))
	}
}

func TestVendorBuild(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("build", "vend/x")
}

func TestVendorRun(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
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
	cd := changeVolume(filepath.Join(tg.pwd(), "testdata/src/vend/hello"), strings.ToUpper)
	tg.cd(cd)
	tg.run("run", "hello.go")
	tg.grepStdout("hello, world", "missing hello world output")
}

func TestVendorTest(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.cd(filepath.Join(tg.pwd(), "testdata/src/vend/hello"))
	tg.run("test", "-v")
	tg.grepStdout("TestMsgInternal", "missing use in internal test")
	tg.grepStdout("TestMsgExternal", "missing use in external test")
}

func TestVendorInvalid(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))

	tg.runFail("build", "vend/x/invalid")
	tg.grepStderr("must be imported as foo", "missing vendor import error")
}

func TestVendorImportError(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))

	tg.runFail("build", "vend/x/vendor/p/p")

	re := regexp.MustCompile(`cannot find package "notfound" in any of:
	.*[\\/]testdata[\\/]src[\\/]vend[\\/]x[\\/]vendor[\\/]notfound \(vendor tree\)
	.*[\\/]testdata[\\/]src[\\/]vend[\\/]vendor[\\/]notfound
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

func TestVendorGet(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.tempFile("src/v/m.go", `
		package main
		import ("fmt"; "vendor.org/p")
		func main() {
			fmt.Println(p.C)
		}`)
	tg.tempFile("src/v/m_test.go", `
		package main
		import ("fmt"; "testing"; "vendor.org/p")
		func TestNothing(t *testing.T) {
			fmt.Println(p.C)
		}`)
	tg.tempFile("src/v/vendor/vendor.org/p/p.go", `
		package p
		const C = 1`)
	tg.setenv("GOPATH", tg.path("."))
	tg.cd(tg.path("src/v"))
	tg.run("run", "m.go")
	tg.run("test")
	tg.run("list", "-f", "{{.Imports}}")
	tg.grepStdout("v/vendor/vendor.org/p", "import not in vendor directory")
	tg.run("list", "-f", "{{.TestImports}}")
	tg.grepStdout("v/vendor/vendor.org/p", "test import not in vendor directory")
	tg.run("get", "-d")
	tg.run("get", "-t", "-d")
}

func TestVendorGetUpdate(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveExecPath(t, "git")

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.run("get", "github.com/rsc/go-get-issue-11864")
	tg.run("get", "-u", "github.com/rsc/go-get-issue-11864")
}

func TestVendorGetU(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveExecPath(t, "git")

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.run("get", "-u", "github.com/rsc/go-get-issue-11864")
}

func TestVendorGetTU(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveExecPath(t, "git")

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.run("get", "-t", "-u", "github.com/rsc/go-get-issue-11864/...")
}

func TestVendorGetBadVendor(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveExecPath(t, "git")

	for _, suffix := range []string{"bad/imp", "bad/imp2", "bad/imp3", "..."} {
		t.Run(suffix, func(t *testing.T) {
			tg := testgo(t)
			defer tg.cleanup()
			tg.makeTempdir()
			tg.setenv("GOPATH", tg.path("."))
			tg.runFail("get", "-t", "-u", "github.com/rsc/go-get-issue-18219/"+suffix)
			tg.grepStderr("must be imported as", "did not find error about vendor import")
			tg.mustNotExist(tg.path("src/github.com/rsc/vendor"))
		})
	}
}

func TestGetSubmodules(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveExecPath(t, "git")

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.run("get", "-d", "github.com/rsc/go-get-issue-12612")
	tg.run("get", "-u", "-d", "github.com/rsc/go-get-issue-12612")
	tg.mustExist(tg.path("src/github.com/rsc/go-get-issue-12612/vendor/golang.org/x/crypto/.git"))
}

func TestVendorCache(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata/testvendor"))
	tg.runFail("build", "p")
	tg.grepStderr("must be imported as x", "did not fail to build p")
}

func TestVendorTest2(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveExecPath(t, "git")

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.run("get", "github.com/rsc/go-get-issue-11864")

	// build -i should work
	tg.run("build", "-i", "github.com/rsc/go-get-issue-11864")
	tg.run("build", "-i", "github.com/rsc/go-get-issue-11864/t")

	// test -i should work like build -i (golang.org/issue/11988)
	tg.run("test", "-i", "github.com/rsc/go-get-issue-11864")
	tg.run("test", "-i", "github.com/rsc/go-get-issue-11864/t")

	// test should work too
	tg.run("test", "github.com/rsc/go-get-issue-11864")
	tg.run("test", "github.com/rsc/go-get-issue-11864/t")

	// external tests should observe internal test exports (golang.org/issue/11977)
	tg.run("test", "github.com/rsc/go-get-issue-11864/vendor/vendor.org/tx2")
}

func TestVendorTest3(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveExecPath(t, "git")

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.run("get", "github.com/clsung/go-vendor-issue-14613")

	tg.run("build", "-o", tg.path("a.out"), "-i", "github.com/clsung/go-vendor-issue-14613")

	// test folder should work
	tg.run("test", "-i", "github.com/clsung/go-vendor-issue-14613")
	tg.run("test", "github.com/clsung/go-vendor-issue-14613")

	// test with specified _test.go should work too
	tg.cd(filepath.Join(tg.path("."), "src"))
	tg.run("test", "-i", "github.com/clsung/go-vendor-issue-14613/vendor_test.go")
	tg.run("test", "github.com/clsung/go-vendor-issue-14613/vendor_test.go")

	// test with imported and not used
	tg.run("test", "-i", "github.com/clsung/go-vendor-issue-14613/vendor/mylibtesttest/myapp/myapp_test.go")
	tg.runFail("test", "github.com/clsung/go-vendor-issue-14613/vendor/mylibtesttest/myapp/myapp_test.go")
	tg.grepStderr("imported and not used:", `should say "imported and not used"`)
}

func TestVendorList(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveExecPath(t, "git")

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.run("get", "github.com/rsc/go-get-issue-11864")

	tg.run("list", "-f", `{{join .TestImports "\n"}}`, "github.com/rsc/go-get-issue-11864/t")
	tg.grepStdout("go-get-issue-11864/vendor/vendor.org/p", "did not find vendor-expanded p")

	tg.run("list", "-f", `{{join .XTestImports "\n"}}`, "github.com/rsc/go-get-issue-11864/tx")
	tg.grepStdout("go-get-issue-11864/vendor/vendor.org/p", "did not find vendor-expanded p")

	tg.run("list", "-f", `{{join .XTestImports "\n"}}`, "github.com/rsc/go-get-issue-11864/vendor/vendor.org/tx2")
	tg.grepStdout("go-get-issue-11864/vendor/vendor.org/tx2", "did not find vendor-expanded tx2")

	tg.run("list", "-f", `{{join .XTestImports "\n"}}`, "github.com/rsc/go-get-issue-11864/vendor/vendor.org/tx3")
	tg.grepStdout("go-get-issue-11864/vendor/vendor.org/tx3", "did not find vendor-expanded tx3")
}

func TestVendor12156(t *testing.T) {
	// Former index out of range panic.
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata/testvendor2"))
	tg.cd(filepath.Join(tg.pwd(), "testdata/testvendor2/src/p"))
	tg.runFail("build", "p.go")
	tg.grepStderrNot("panic", "panicked")
	tg.grepStderr(`cannot find package "x"`, "wrong error")
}

// Module legacy support does path rewriting very similar to vendoring.

func TestLegacyMod(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata/modlegacy"))
	tg.run("list", "-f", "{{.Imports}}", "old/p1")
	tg.grepStdout("new/p1", "old/p1 should import new/p1")
	tg.run("list", "-f", "{{.Imports}}", "new/p1")
	tg.grepStdout("new/p2", "new/p1 should import new/p2 (not new/v2/p2)")
	tg.grepStdoutNot("new/v2", "new/p1 should NOT import new/v2*")
	tg.grepStdout("new/sub/x/v1/y", "new/p1 should import new/sub/x/v1/y (not new/sub/v2/x/v1/y)")
	tg.grepStdoutNot("new/sub/v2", "new/p1 should NOT import new/sub/v2*")
	tg.grepStdout("new/sub/inner/x", "new/p1 should import new/sub/inner/x (no rewrites)")
	tg.run("build", "old/p1", "new/p1")
}

func TestLegacyModGet(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveExecPath(t, "git")

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("d1"))
	tg.run("get", "vcs-test.golang.org/git/modlegacy1-old.git/p1")
	tg.run("list", "-f", "{{.Deps}}", "vcs-test.golang.org/git/modlegacy1-old.git/p1")
	tg.grepStdout("new.git/p2", "old/p1 should depend on new/p2")
	tg.grepStdoutNot("new.git/v2/p2", "old/p1 should NOT depend on new/v2/p2")
	tg.run("build", "vcs-test.golang.org/git/modlegacy1-old.git/p1", "vcs-test.golang.org/git/modlegacy1-new.git/p1")

	tg.setenv("GOPATH", tg.path("d2"))

	tg.must(os.RemoveAll(tg.path("d2")))
	tg.run("get", "github.com/rsc/vgotest5")
	tg.run("get", "github.com/rsc/vgotest4")
	tg.run("get", "github.com/myitcv/vgo_example_compat")

	if testing.Short() {
		return
	}

	tg.must(os.RemoveAll(tg.path("d2")))
	tg.run("get", "github.com/rsc/vgotest4")
	tg.run("get", "github.com/rsc/vgotest5")
	tg.run("get", "github.com/myitcv/vgo_example_compat")

	tg.must(os.RemoveAll(tg.path("d2")))
	tg.run("get", "github.com/rsc/vgotest4", "github.com/rsc/vgotest5")
	tg.run("get", "github.com/myitcv/vgo_example_compat")

	tg.must(os.RemoveAll(tg.path("d2")))
	tg.run("get", "github.com/rsc/vgotest5", "github.com/rsc/vgotest4")
	tg.run("get", "github.com/myitcv/vgo_example_compat")

	tg.must(os.RemoveAll(tg.path("d2")))
	tg.run("get", "github.com/myitcv/vgo_example_compat")
	tg.run("get", "github.com/rsc/vgotest4", "github.com/rsc/vgotest5")

	pkgs := []string{"github.com/myitcv/vgo_example_compat", "github.com/rsc/vgotest4", "github.com/rsc/vgotest5"}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 3; k++ {
				if i == j || i == k || k == j {
					continue
				}
				tg.must(os.RemoveAll(tg.path("d2")))
				tg.run("get", pkgs[i], pkgs[j], pkgs[k])
			}
		}
	}
}
