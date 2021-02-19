// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// No testdata on Android.

//go:build !android
// +build !android

package main

import (
	"bytes"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"golang.org/x/tools/internal/testenv"
)

// TODO(adonovan):
// - test introduction of renaming imports.
// - test induced failures of rewriteFile.

// Guide to the test packages:
//
// new.com/one		-- canonical name for old.com/one
// old.com/one		-- non-canonical; has import comment "new.com/one"
// old.com/bad		-- has a parse error
// fruit.io/orange	\
// fruit.io/banana	 } orange -> pear -> banana -> titanic.biz/bar
// fruit.io/pear	/
// titanic.biz/bar	-- domain is sinking; package has jumped ship to new.com/bar
// titanic.biz/foo	-- domain is sinking but package has no import comment yet

var gopath = filepath.Join(cwd, "testdata")

func init() {
	if err := os.Setenv("GOPATH", gopath); err != nil {
		log.Fatal(err)
	}

	// This test currently requires GOPATH mode.
	// Explicitly disabling module mode should suffix, but
	// we'll also turn off GOPROXY just for good measure.
	if err := os.Setenv("GO111MODULE", "off"); err != nil {
		log.Fatal(err)
	}
	if err := os.Setenv("GOPROXY", "off"); err != nil {
		log.Fatal(err)
	}
}

func TestFixImports(t *testing.T) {
	testenv.NeedsTool(t, "go")

	defer func() {
		stderr = os.Stderr
		*badDomains = "code.google.com"
		*replaceFlag = ""
	}()

	for i, test := range []struct {
		packages    []string // packages to rewrite, "go list" syntax
		badDomains  string   // -baddomains flag
		replaceFlag string   // -replace flag
		wantOK      bool
		wantStderr  string
		wantRewrite map[string]string
	}{
		// #0. No errors.
		{
			packages:   []string{"all"},
			badDomains: "code.google.com",
			wantOK:     true,
			wantStderr: `
testdata/src/old.com/bad/bad.go:2:43: expected 'package', found 'EOF'
fruit.io/banana
	fixed: old.com/one -> new.com/one
	fixed: titanic.biz/bar -> new.com/bar
`,
			wantRewrite: map[string]string{
				"$GOPATH/src/fruit.io/banana/banana.go": `package banana

import (
	_ "new.com/bar"
	_ "new.com/one"
	_ "titanic.biz/foo"
)`,
			},
		},
		// #1. No packages needed rewriting.
		{
			packages:   []string{"titanic.biz/...", "old.com/...", "new.com/..."},
			badDomains: "code.google.com",
			wantOK:     true,
			wantStderr: `
testdata/src/old.com/bad/bad.go:2:43: expected 'package', found 'EOF'
`,
		},
		// #2. Some packages without import comments matched bad domains.
		{
			packages:   []string{"all"},
			badDomains: "titanic.biz",
			wantOK:     false,
			wantStderr: `
testdata/src/old.com/bad/bad.go:2:43: expected 'package', found 'EOF'
fruit.io/banana
	testdata/src/fruit.io/banana/banana.go:6: import "titanic.biz/foo"
	fixed: old.com/one -> new.com/one
	fixed: titanic.biz/bar -> new.com/bar
	ERROR: titanic.biz/foo has no import comment
	imported directly by:
		fruit.io/pear
	imported indirectly by:
		fruit.io/orange
`,
			wantRewrite: map[string]string{
				"$GOPATH/src/fruit.io/banana/banana.go": `package banana

import (
	_ "new.com/bar"
	_ "new.com/one"
	_ "titanic.biz/foo"
)`,
			},
		},
		// #3. The -replace flag lets user supply missing import comments.
		{
			packages:    []string{"all"},
			replaceFlag: "titanic.biz/foo=new.com/foo",
			wantOK:      true,
			wantStderr: `
testdata/src/old.com/bad/bad.go:2:43: expected 'package', found 'EOF'
fruit.io/banana
	fixed: old.com/one -> new.com/one
	fixed: titanic.biz/bar -> new.com/bar
	fixed: titanic.biz/foo -> new.com/foo
`,
			wantRewrite: map[string]string{
				"$GOPATH/src/fruit.io/banana/banana.go": `package banana

import (
	_ "new.com/bar"
	_ "new.com/foo"
	_ "new.com/one"
)`,
			},
		},
		// #4. The -replace flag supports wildcards.
		//     An explicit import comment takes precedence.
		{
			packages:    []string{"all"},
			replaceFlag: "titanic.biz/...=new.com/...",
			wantOK:      true,
			wantStderr: `
testdata/src/old.com/bad/bad.go:2:43: expected 'package', found 'EOF'
fruit.io/banana
	fixed: old.com/one -> new.com/one
	fixed: titanic.biz/bar -> new.com/bar
	fixed: titanic.biz/foo -> new.com/foo
`,
			wantRewrite: map[string]string{
				"$GOPATH/src/fruit.io/banana/banana.go": `package banana

import (
	_ "new.com/bar"
	_ "new.com/foo"
	_ "new.com/one"
)`,
			},
		},
		// #5. The -replace flag trumps -baddomains.
		{
			packages:    []string{"all"},
			badDomains:  "titanic.biz",
			replaceFlag: "titanic.biz/foo=new.com/foo",
			wantOK:      true,
			wantStderr: `
testdata/src/old.com/bad/bad.go:2:43: expected 'package', found 'EOF'
fruit.io/banana
	fixed: old.com/one -> new.com/one
	fixed: titanic.biz/bar -> new.com/bar
	fixed: titanic.biz/foo -> new.com/foo
`,
			wantRewrite: map[string]string{
				"$GOPATH/src/fruit.io/banana/banana.go": `package banana

import (
	_ "new.com/bar"
	_ "new.com/foo"
	_ "new.com/one"
)`,
			},
		},
	} {
		*badDomains = test.badDomains
		*replaceFlag = test.replaceFlag

		stderr = new(bytes.Buffer)
		gotRewrite := make(map[string]string)
		writeFile = func(filename string, content []byte, mode os.FileMode) error {
			filename = strings.Replace(filename, gopath, "$GOPATH", 1)
			filename = filepath.ToSlash(filename)
			gotRewrite[filename] = string(bytes.TrimSpace(content))
			return nil
		}

		if runtime.GOOS == "windows" {
			test.wantStderr = strings.Replace(test.wantStderr, `testdata/src/old.com/bad/bad.go`, `testdata\src\old.com\bad\bad.go`, -1)
			test.wantStderr = strings.Replace(test.wantStderr, `testdata/src/fruit.io/banana/banana.go`, `testdata\src\fruit.io\banana\banana.go`, -1)
		}
		test.wantStderr = strings.TrimSpace(test.wantStderr)

		// Check status code.
		if fiximports(test.packages...) != test.wantOK {
			t.Errorf("#%d. fiximports() = %t", i, !test.wantOK)
		}

		// Compare stderr output.
		if got := strings.TrimSpace(stderr.(*bytes.Buffer).String()); got != test.wantStderr {
			if strings.Contains(got, "vendor/golang_org/x/text/unicode/norm") {
				t.Skip("skipping known-broken test; see golang.org/issue/17417")
			}
			t.Errorf("#%d. stderr: got <<\n%s\n>>, want <<\n%s\n>>",
				i, got, test.wantStderr)
		}

		// Compare rewrites.
		for k, v := range gotRewrite {
			if test.wantRewrite[k] != v {
				t.Errorf("#%d. rewrite[%s] = <<%s>>, want <<%s>>",
					i, k, v, test.wantRewrite[k])
			}
			delete(test.wantRewrite, k)
		}
		for k, v := range test.wantRewrite {
			t.Errorf("#%d. rewrite[%s] missing, want <<%s>>", i, k, v)
		}
	}
}

// TestDryRun tests that the -n flag suppresses calls to writeFile.
func TestDryRun(t *testing.T) {
	testenv.NeedsTool(t, "go")

	*dryrun = true
	defer func() { *dryrun = false }() // restore
	stderr = new(bytes.Buffer)
	writeFile = func(filename string, content []byte, mode os.FileMode) error {
		t.Fatalf("writeFile(%s) called in dryrun mode", filename)
		return nil
	}

	if !fiximports("all") {
		t.Fatalf("fiximports failed: %s", stderr)
	}
}
