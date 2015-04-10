// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import (
	"fmt"
	"math"
	"os"
	"strings"
)

// go-specific code shared across loaders (5l, 6l, 8l).

var Framepointer_enabled int

var Fieldtrack_enabled int

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

// replace all "". with pkg.
func Expandpkg(t0 string, pkg string) string {
	return strings.Replace(t0, `"".`, pkg+".", -1)
}

func double2ieee(ieee *uint64, f float64) {
	*ieee = math.Float64bits(f)
}

func Nopout(p *Prog) {
	p.As = ANOP
	p.Scond = 0
	p.From = Addr{}
	p.From3 = Addr{}
	p.Reg = 0
	p.To = Addr{}
}

func Nocache(p *Prog) {
	p.Optab = 0
	p.From.Class = 0
	p.From3.Class = 0
	p.To.Class = 0
}

/*
 *	bv.c
 */

/*
 *	closure.c
 */

/*
 *	const.c
 */

/*
 *	cplx.c
 */

/*
 *	dcl.c
 */

/*
 *	esc.c
 */

/*
 *	export.c
 */

/*
 *	fmt.c
 */

/*
 *	gen.c
 */

/*
 *	init.c
 */

/*
 *	inl.c
 */

/*
 *	lex.c
 */
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
