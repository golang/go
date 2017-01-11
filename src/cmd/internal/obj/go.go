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
	framepointer_enabled     int
	Fieldtrack_enabled       int
	Preemptibleloops_enabled int
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
	{"framepointer", &framepointer_enabled},
	{"preemptibleloops", &Preemptibleloops_enabled},
}

func addexp(s string) {
	// Could do general integer parsing here, but the runtime copy doesn't yet.
	v := 1
	name := s
	if len(name) > 2 && name[:2] == "no" {
		v = 0
		name = name[2:]
	}
	for i := 0; i < len(exper); i++ {
		if exper[i].name == name {
			if exper[i].val != nil {
				*exper[i].val = v
			}
			return
		}
	}

	fmt.Printf("unknown experiment %s\n", s)
	os.Exit(2)
}

func init() {
	framepointer_enabled = 1 // default
	for _, f := range strings.Split(goexperiment, ",") {
		if f != "" {
			addexp(f)
		}
	}
}

func Framepointer_enabled(goos, goarch string) bool {
	return framepointer_enabled != 0 && goarch == "amd64" && goos != "nacl"
}

func Nopout(p *Prog) {
	p.As = ANOP
	p.Scond = 0
	p.From = Addr{}
	p.From3 = nil
	p.Reg = 0
	p.To = Addr{}
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
