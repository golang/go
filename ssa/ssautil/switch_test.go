// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssautil_test

import (
	"go/parser"
	"strings"
	"testing"

	"code.google.com/p/go.tools/importer"
	"code.google.com/p/go.tools/ssa"
	"code.google.com/p/go.tools/ssa/ssautil"
)

func TestSwitches(t *testing.T) {
	imp := importer.New(new(importer.Config)) // (uses GCImporter)
	f, err := parser.ParseFile(imp.Fset, "testdata/switches.go", nil, parser.ParseComments)
	if err != nil {
		t.Error(err)
		return
	}

	mainInfo := imp.CreatePackage("main", f)

	prog := ssa.NewProgram(imp.Fset, 0)
	if err := prog.CreatePackages(imp); err != nil {
		t.Error(err)
		return
	}
	mainPkg := prog.Package(mainInfo.Pkg)
	mainPkg.Build()

	for _, mem := range mainPkg.Members {
		if fn, ok := mem.(*ssa.Function); ok {
			if fn.Synthetic != "" {
				continue // e.g. init()
			}
			// Each (multi-line) "switch" comment within
			// this function must match the printed form
			// of a ConstSwitch.
			var wantSwitches []string
			for _, c := range f.Comments {
				if fn.Syntax().Pos() <= c.Pos() && c.Pos() < fn.Syntax().End() {
					text := strings.TrimSpace(c.Text())
					if strings.HasPrefix(text, "switch ") {
						wantSwitches = append(wantSwitches, text)
					}
				}
			}

			switches := ssautil.Switches(fn)
			if len(switches) != len(wantSwitches) {
				t.Errorf("in %s, found %d switches, want %d", fn, len(switches), len(wantSwitches))
			}
			for i, sw := range switches {
				got := sw.String()
				if i >= len(wantSwitches) {
					continue
				}
				want := wantSwitches[i]
				if got != want {
					t.Errorf("in %s, found switch %d: got <<%s>>, want <<%s>>", fn, i, got, want)
				}
			}
		}
	}
}
