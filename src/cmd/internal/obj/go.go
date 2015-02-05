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

var fieldtrack_enabled int

var Zprog Prog

// Toolchain experiments.
// These are controlled by the GOEXPERIMENT environment
// variable recorded when the toolchain is built.
// This list is also known to cmd/gc.
var exper = []struct {
	name string
	val  *int
}{
	struct {
		name string
		val  *int
	}{"fieldtrack", &fieldtrack_enabled},
	struct {
		name string
		val  *int
	}{"framepointer", &Framepointer_enabled},
}

func addexp(s string) {
	var i int

	for i = 0; i < len(exper); i++ {
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

func linksetexp() {
	for _, f := range strings.Split(goexperiment, ",") {
		if f != "" {
			addexp(f)
		}
	}
}

func expstring() string {
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

// replace all "". with pkg.
func expandpkg(t0 string, pkg string) string {
	return strings.Replace(t0, `"".`, pkg+".", -1)
}

func double2ieee(ieee *uint64, f float64) {
	*ieee = math.Float64bits(f)
}
