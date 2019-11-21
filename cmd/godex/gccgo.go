// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements access to gccgo-generated export data.

package main

import (
	"go/importer"
	"go/token"
	"go/types"
)

func init() {
	register("gccgo", importer.ForCompiler(token.NewFileSet(), "gccgo", nil))
}

// Print the extra gccgo compiler data for this package, if it exists.
func (p *printer) printGccgoExtra(pkg *types.Package) {
	// Disabled for now.
	// TODO(gri) address this at some point.

	// if initdata, ok := initmap[pkg]; ok {
	// 	p.printf("/*\npriority %d\n", initdata.Priority)

	// 	p.printDecl("init", len(initdata.Inits), func() {
	// 		for _, init := range initdata.Inits {
	// 			p.printf("%s %s %d\n", init.Name, init.InitFunc, init.Priority)
	// 		}
	// 	})

	// 	p.print("*/\n")
	// }
}
