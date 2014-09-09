package importgraph_test

import (
	"go/build"
	"runtime"
	"sort"
	"testing"

	"code.google.com/p/go.tools/refactor/importgraph"

	_ "crypto/hmac" // just for test, below
)

const this = "code.google.com/p/go.tools/refactor/importgraph"

func TestBuild(t *testing.T) {
	saved := runtime.GOMAXPROCS(8) // Build is highly parallel
	defer runtime.GOMAXPROCS(saved)

	forward, reverse, errors := importgraph.Build(&build.Default)

	// Test direct edges.
	// We throw in crypto/hmac to prove that external test files
	// (such as this one) are inspected.
	for _, p := range []string{"go/build", "runtime", "testing", "crypto/hmac"} {
		if !forward[this][p] {
			t.Errorf("forward[importgraph][%s] not found", p)
		}
		if !reverse[p][this] {
			t.Errorf("reverse[%s][importgraph] not found", p)
		}
	}

	// Test non-existent direct edges
	for _, p := range []string{"fmt", "errors", "reflect"} {
		if forward[this][p] {
			t.Errorf("unexpected: forward[importgraph][%s] found", p)
		}
		if reverse[p][this] {
			t.Errorf("unexpected: reverse[%s][importgraph] found", p)
		}
	}

	// Test Search is reflexive.
	if !forward.Search(this)[this] {
		t.Errorf("irreflexive: forward.Search(importgraph)[importgraph] not found")
	}
	if !reverse.Search(this)[this] {
		t.Errorf("irrefexive: reverse.Search(importgraph)[importgraph] not found")
	}

	// Test Search is transitive.  (There is no direct edge to these packages.)
	for _, p := range []string{"errors", "reflect", "unsafe"} {
		if !forward.Search(this)[p] {
			t.Errorf("intransitive: forward.Search(importgraph)[%s] not found", p)
		}
		if !reverse.Search(p)[this] {
			t.Errorf("intransitive: reverse.Search(%s)[importgraph] not found", p)
		}
	}

	// debugging
	if false {
		for path, err := range errors {
			t.Logf("%s: %s", path, err)
		}
		printSorted := func(direction string, g importgraph.Graph, start string) {
			t.Log(direction)
			var pkgs []string
			for pkg := range g.Search(start) {
				pkgs = append(pkgs, pkg)
			}
			sort.Strings(pkgs)
			for _, pkg := range pkgs {
				t.Logf("\t%s", pkg)
			}
		}
		printSorted("forward", forward, this)
		printSorted("forward", reverse, this)
	}
}
