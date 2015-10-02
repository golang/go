// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import (
	"fmt"
	"os"
	"strings"
)

// go-specific code shared across loaders (5l, 6l, 8l).

var (
	Framepointer_enabled int
	Fieldtrack_enabled   int
)

// Toolchain experiments.
// These are controlled by the GOEXPERIMENT environment
// variable recorded when the toolchain is built.
// This list is also known to cmd/gc.
var exper = []struct {
	name string
	val  *int
}{
	{"fieldtrack", &Fieldtrack_enabled},
	{"framepointer", &Framepointer_enabled},
}

func addexp(s string) {
	for i := 0; i < len(exper); i++ {
		if exper[i].name == s {
			if exper[i].val != nil {
				*exper[i].val = 1
			}
			return
		}
	}

	fmt.Printf("unknown experiment %s\n", s)
	os.Exit(2)
}

func init() {
	for _, f := range strings.Split(goexperiment, ",") {
		if f != "" {
			addexp(f)
		}
	}
}

func Nopout(p *Prog) {
	p.As = ANOP
	p.Scond = 0
	p.From = Addr{}
	p.From3 = nil
	p.Reg = 0
	p.To = Addr{}
}

func Nocache(p *Prog) {
	p.Optab = 0
	p.From.Class = 0
	if p.From3 != nil {
		p.From3.Class = 0
	}
	p.To.Class = 0
}

func Expstring() string {
	buf := "X"
	for i := range exper {
		if *exper[i].val != 0 {
			buf += "," + exper[i].name
		}
	}
	if buf == "X" {
		buf += ",none"
	}
	return "X:" + buf[2:]
}
