// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"debug/goobj"
	"io/ioutil"
	"testing"
)

func TestLinkHello(t *testing.T) {
	p := &Prog{
		GOOS:     "darwin",
		GOARCH:   "amd64",
		Error:    func(s string) { t.Error(s) },
		StartSym: "_rt0_go",
	}
	var buf bytes.Buffer
	p.link(&buf, "testdata/hello.6")
	if p.NumError > 0 {
		return
	}
	if p.Syms[goobj.SymID{"_rt0_go", 0}] == nil || p.Syms[goobj.SymID{"hello", 1}] == nil {
		t.Errorf("Syms = %v, want at least [_rt0_go hello<1>]", p.Syms)
	}

	// uncomment to leave file behind for execution:
	if false {
		ioutil.WriteFile("a.out", buf.Bytes(), 0777)
	}
	checkGolden(t, buf.Bytes(), "testdata/link.hello.darwin.amd64")
}
