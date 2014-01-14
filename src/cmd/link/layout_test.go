// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestLayout(t *testing.T) {
	p := Prog{GOOS: "darwin", GOARCH: "amd64", StartSym: "text_start"}
	p.omitRuntime = true
	p.Error = func(s string) { t.Error(s) }
	var buf bytes.Buffer
	const obj = "testdata/layout.6"
	p.link(&buf, obj)
	if p.NumError > 0 {
		return // already reported
	}
	if len(p.Dead) > 0 {
		t.Errorf("%s: unexpected dead symbols %v", obj, p.Dead)
		return
	}

	for _, sym := range p.SymOrder {
		if p.isAuto(sym.SymID) {
			continue
		}
		if sym.Section == nil {
			t.Errorf("%s: symbol %s is missing section", obj, sym)
			continue
		}
		i := strings.Index(sym.Name, "_")
		if i < 0 {
			t.Errorf("%s: unexpected symbol %s", obj, sym)
			continue
		}
		if sym.Section.Name != sym.Name[:i] {
			t.Errorf("%s: symbol %s in section %s, want %s", obj, sym, sym.Section.Name, sym.Name[:i])
		}
	}
}
