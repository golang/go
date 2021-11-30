// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Incomplete std lib sources on Android.

//go:build !android
// +build !android

package importgraph_test

import (
	"fmt"
	"go/build"
	"os"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/refactor/importgraph"

	_ "crypto/hmac" // just for test, below
)

const this = "golang.org/x/tools/refactor/importgraph"

func TestBuild(t *testing.T) {
	exported := packagestest.Export(t, packagestest.GOPATH, []packagestest.Module{
		{Name: "golang.org/x/tools/refactor/importgraph", Files: packagestest.MustCopyFileTree(".")}})
	defer exported.Cleanup()

	var gopath string
	for _, env := range exported.Config.Env {
		eq := strings.Index(env, "=")
		if eq == 0 {
			// We sometimes see keys with a single leading "=" in the environment on Windows.
			// TODO(#49886): What is the correct way to parse them in general?
			eq = strings.Index(env[1:], "=") + 1
		}
		if eq < 0 {
			t.Fatalf("invalid variable in exported.Config.Env: %q", env)
		}
		k := env[:eq]
		v := env[eq+1:]
		if k == "GOPATH" {
			gopath = v
		}

		if os.Getenv(k) == v {
			continue
		}
		defer func(prev string, prevOK bool) {
			if !prevOK {
				if err := os.Unsetenv(k); err != nil {
					t.Fatal(err)
				}
			} else {
				if err := os.Setenv(k, prev); err != nil {
					t.Fatal(err)
				}
			}
		}(os.LookupEnv(k))

		if err := os.Setenv(k, v); err != nil {
			t.Fatal(err)
		}
		t.Logf("%s=%s", k, v)
	}
	if gopath == "" {
		t.Fatal("Failed to fish GOPATH out of env: ", exported.Config.Env)
	}

	var buildContext = build.Default
	buildContext.GOPATH = gopath
	buildContext.Dir = exported.Config.Dir

	forward, reverse, errs := importgraph.Build(&buildContext)
	for path, err := range errs {
		t.Errorf("%s: %s", path, err)
	}
	if t.Failed() {
		return
	}

	// Log the complete graph before the errors, so that the errors are near the
	// end of the log (where we expect them to be).
	nodePrinted := map[string]bool{}
	printNode := func(direction string, from string) {
		key := fmt.Sprintf("%s[%q]", direction, from)
		if nodePrinted[key] {
			return
		}
		nodePrinted[key] = true

		var g importgraph.Graph
		switch direction {
		case "forward":
			g = forward
		case "reverse":
			g = reverse
		default:
			t.Helper()
			t.Fatalf("bad direction: %q", direction)
		}

		t.Log(key)
		var pkgs []string
		for pkg := range g[from] {
			pkgs = append(pkgs, pkg)
		}
		sort.Strings(pkgs)
		for _, pkg := range pkgs {
			t.Logf("\t%s", pkg)
		}
	}

	if testing.Verbose() {
		printNode("forward", this)
		printNode("reverse", this)
	}

	// Test direct edges.
	// We throw in crypto/hmac to prove that external test files
	// (such as this one) are inspected.
	for _, p := range []string{"go/build", "testing", "crypto/hmac"} {
		if !forward[this][p] {
			printNode("forward", this)
			t.Errorf("forward[%q][%q] not found", this, p)
		}
		if !reverse[p][this] {
			printNode("reverse", p)
			t.Errorf("reverse[%q][%q] not found", p, this)
		}
	}

	// Test non-existent direct edges
	for _, p := range []string{"errors", "reflect"} {
		if forward[this][p] {
			printNode("forward", this)
			t.Errorf("unexpected: forward[%q][%q] found", this, p)
		}
		if reverse[p][this] {
			printNode("reverse", p)
			t.Errorf("unexpected: reverse[%q][%q] found", p, this)
		}
	}

	// Test Search is reflexive.
	if !forward.Search(this)[this] {
		printNode("forward", this)
		t.Errorf("irreflexive: forward.Search(importgraph)[importgraph] not found")
	}
	if !reverse.Search(this)[this] {
		printNode("reverse", this)
		t.Errorf("irrefexive: reverse.Search(importgraph)[importgraph] not found")
	}

	// Test Search is transitive.  (There is no direct edge to these packages.)
	for _, p := range []string{"errors", "reflect", "unsafe"} {
		if !forward.Search(this)[p] {
			printNode("forward", this)
			t.Errorf("intransitive: forward.Search(importgraph)[%s] not found", p)
		}
		if !reverse.Search(p)[this] {
			printNode("reverse", p)
			t.Errorf("intransitive: reverse.Search(%s)[importgraph] not found", p)
		}
	}

	// Test strongly-connected components.  Because A's external
	// test package can depend on B, and vice versa, most of the
	// standard libraries are mutually dependent when their external
	// tests are considered.
	//
	// For any nodes x, y in the same SCC, y appears in the results
	// of both forward and reverse searches starting from x
	if !forward.Search("fmt")["io"] ||
		!forward.Search("io")["fmt"] ||
		!reverse.Search("fmt")["io"] ||
		!reverse.Search("io")["fmt"] {
		printNode("forward", "fmt")
		printNode("forward", "io")
		printNode("reverse", "fmt")
		printNode("reverse", "io")
		t.Errorf("fmt and io are not mutually reachable despite being in the same SCC")
	}
}
