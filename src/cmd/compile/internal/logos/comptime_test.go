// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func TestCfCopyNameShape(t *testing.T) {
	pkg := types2.NewPackage("example.com/p", "p")
	obj := types2.NewFunc(syntax.Pos{}, pkg, "Autowire", nil)
	if got, want := cfCopyName(obj, 7), "Autowire_logosct7"; got != want {
		t.Fatalf("cfCopyName = %q, want %q", got, want)
	}
	if cfCopyInfix != "_logosct" {
		t.Fatalf("cfCopyInfix = %q, want %q (logosCtASCIIInfix in cmd/link/internal/ld must match)", cfCopyInfix, "_logosct")
	}

	parse := func(name string) error {
		var errs []error
		_, _ = syntax.Parse(syntax.NewFileBase("cfcopy.lgo"),
			strings.NewReader("package p\nfunc "+name+"() {}\n"),
			func(err error) { errs = append(errs, err) },
			nil, syntax.CheckBranches|syntax.LogosMode)
		if len(errs) == 0 {
			return nil
		}
		return errs[0]
	}
	if err := parse(cfCopyName(obj, 0)); err != nil {
		t.Fatalf("copy name does not parse: %v", err)
	}
	if err := parse("Autowire·logosct0"); err == nil {
		t.Fatalf("middle-dot copy name parses, want a parse error")
	}
}
