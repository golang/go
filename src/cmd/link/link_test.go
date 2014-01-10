// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"debug/goobj"
	"testing"
)

func TestLinkHello(t *testing.T) {
	p := &Prog{
		GOOS:   "darwin",
		GOARCH: "amd64",
		Error:  func(s string) { t.Error(s) },
	}
	var buf bytes.Buffer
	p.link(&buf, "testdata/hello.6")
	if p.NumError > 0 {
		return
	}
	if len(p.Syms) != 2 || p.Syms[goobj.SymID{"_rt0_go", 0}] == nil || p.Syms[goobj.SymID{"hello", 1}] == nil {
		t.Errorf("Syms = %v, want [_rt0_go hello<1>]", p.Syms)
	}

	checkGolden(t, buf.Bytes(), "testdata/link.hello.darwin.amd64")

	// uncomment to leave file behind for execution:
	// ioutil.WriteFile("a.out", buf.Bytes(), 0777)
}
