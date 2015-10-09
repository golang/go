// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.5

// This file implements access to gccgo-generated export data.

package main

import (
	"golang.org/x/tools/go/gccgoimporter"
	"golang.org/x/tools/go/types"
)

var (
	initmap = make(map[*types.Package]gccgoimporter.InitData)
)

func init() {
	incpaths := []string{"/"}

	// importer for default gccgo
	var inst gccgoimporter.GccgoInstallation
	inst.InitFromDriver("gccgo")
	register("gccgo", inst.GetImporter(incpaths, initmap))
}

// Print the extra gccgo compiler data for this package, if it exists.
func (p *printer) printGccgoExtra(pkg *types.Package) {
	if initdata, ok := initmap[pkg]; ok {
		p.printf("/*\npriority %d\n", initdata.Priority)

		p.printDecl("init", len(initdata.Inits), func() {
			for _, init := range initdata.Inits {
				p.printf("%s %s %d\n", init.Name, init.InitFunc, init.Priority)
			}
		})

		p.print("*/\n")
	}
}
