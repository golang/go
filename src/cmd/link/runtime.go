// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Generation of runtime-accessible data structures.
// See also debug.go.

package main

import "debug/goobj"

func (p *Prog) runtime() {
	p.pclntab()

	// TODO: Implement garbage collection data.
	p.addSym(&Sym{
		Sym: &goobj.Sym{
			SymID: goobj.SymID{Name: "runtime.gcdata"},
			Kind:  goobj.SRODATA,
		},
	})
	p.addSym(&Sym{
		Sym: &goobj.Sym{
			SymID: goobj.SymID{Name: "runtime.gcbss"},
			Kind:  goobj.SRODATA,
		},
	})
}
