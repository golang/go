// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/objabi"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"fmt"
	"sort"
)

// Inittasks finds inittask records, figures out a good
// order to execute them in, and emits that order for the
// runtime to use.
//
// An inittask represents the initialization code that needs
// to be run for a package. For package p, the p..inittask
// symbol contains a list of init functions to run, both
// explicit user init functions and implicit compiler-generated
// init functions for initializing global variables like maps.
//
// In addition, inittask records have dependencies between each
// other, mirroring the import dependencies. So if package p
// imports package q, then there will be a dependency p -> q.
// We can't initialize package p until after package q has
// already been initialized.
//
// Package dependencies are encoded with relocations. If package
// p imports package q, then package p's inittask record will
// have a R_INITORDER relocation pointing to package q's inittask
// record. See cmd/compile/internal/pkginit/init.go.
//
// This function computes an ordering of all of the inittask
// records so that the order respects all the dependencies,
// and given that restriction, orders the inittasks in
// lexicographic order.
func (ctxt *Link) inittasks() {
	switch ctxt.BuildMode {
	case BuildModeExe, BuildModePIE, BuildModeCArchive, BuildModeCShared:
		// Normally the inittask list will be run on program startup.
		ctxt.mainInittasks = ctxt.inittaskSym([]string{"main..inittask"}, "go:main.inittasks")
	case BuildModePlugin:
		// For plugins, the list will be run on plugin load.
		ctxt.mainInittasks = ctxt.inittaskSym([]string{fmt.Sprintf("%s..inittask", objabi.PathToPrefix(*flagPluginPath))}, "go:plugin.inittasks")
		// Make symbol local so multiple plugins don't clobber each other's inittask list.
		ctxt.loader.SetAttrLocal(ctxt.mainInittasks, true)
	case BuildModeShared:
		// For a shared library, all packages are roots.
		var roots []string
		for _, lib := range ctxt.Library {
			roots = append(roots, fmt.Sprintf("%s..inittask", objabi.PathToPrefix(lib.Pkg)))
		}
		ctxt.mainInittasks = ctxt.inittaskSym(roots, "go:shlib.inittasks")
		// Make symbol local so multiple plugins don't clobber each other's inittask list.
		ctxt.loader.SetAttrLocal(ctxt.mainInittasks, true)
	default:
		Exitf("unhandled build mode %d", ctxt.BuildMode)
	}

	// If the runtime is one of the packages we are building,
	// initialize the runtime_inittasks variable.
	ldr := ctxt.loader
	if ldr.Lookup("runtime.runtime_inittasks", 0) != 0 {
		t := ctxt.inittaskSym([]string{"runtime..inittask"}, "go:runtime.inittasks")

		// This slice header is already defined in runtime/proc.go, so we update it here with new contents.
		sh := ldr.Lookup("runtime.runtime_inittasks", 0)
		sb := ldr.MakeSymbolUpdater(sh)
		sb.SetSize(0)
		sb.SetType(sym.SNOPTRDATA) // Could be SRODATA, but see issue 58857.
		sb.AddAddr(ctxt.Arch, t)
		sb.AddUint(ctxt.Arch, uint64(ldr.SymSize(t)/int64(ctxt.Arch.PtrSize)))
		sb.AddUint(ctxt.Arch, uint64(ldr.SymSize(t)/int64(ctxt.Arch.PtrSize)))
	}
}

// inittaskSym builds a symbol containing pointers to all the inittasks
// that need to be run, given a list of root inittask symbols.
func (ctxt *Link) inittaskSym(rootNames []string, symName string) loader.Sym {
	ldr := ctxt.loader
	var roots []loader.Sym
	for _, n := range rootNames {
		p := ldr.Lookup(n, 0)
		if p != 0 {
			roots = append(roots, p)
		}
	}
	if len(roots) == 0 {
		// Nothing to do
		return 0
	}

	// Edges record dependencies between packages.
	// {from,to} is in edges if from's package imports to's package.
	// This list is used to implement reverse edge lookups.
	type edge struct {
		from, to loader.Sym
	}
	var edges []edge

	// List of packages that are ready to schedule. We use a lexicographic
	// ordered heap to pick the lexically earliest uninitialized but
	// inititalizeable package at each step.
	var h lexHeap

	// m maps from an inittask symbol for package p to the number of
	// p's direct imports that have not yet been scheduled.
	m := map[loader.Sym]int{}

	// Find all reachable inittask records from the roots.
	// Keep track of the dependency edges between them in edges.
	// Keep track of how many imports each package has in m.
	// q is the list of found but not yet explored packages.
	var q []loader.Sym
	for _, p := range roots {
		m[p] = 0
		q = append(q, p)
	}
	for len(q) > 0 {
		x := q[len(q)-1]
		q = q[:len(q)-1]
		relocs := ldr.Relocs(x)
		n := relocs.Count()
		ndeps := 0
		for i := 0; i < n; i++ {
			r := relocs.At(i)
			if r.Type() != objabi.R_INITORDER {
				continue
			}
			ndeps++
			s := r.Sym()
			edges = append(edges, edge{from: x, to: s})
			if _, ok := m[s]; ok {
				continue // already found
			}
			q = append(q, s)
			m[s] = 0 // mark as found
		}
		m[x] = ndeps
		if ndeps == 0 {
			h.push(ldr, x)
		}
	}

	// Sort edges so we can look them up by edge destination.
	sort.Slice(edges, func(i, j int) bool {
		return edges[i].to < edges[j].to
	})

	// Figure out the schedule.
	sched := ldr.MakeSymbolBuilder(symName)
	sched.SetType(sym.SNOPTRDATA) // Could be SRODATA, but see isue 58857.
	for !h.empty() {
		// Pick the lexicographically first initializable package.
		s := h.pop(ldr)

		// Add s to the schedule.
		if ldr.SymSize(s) > 8 {
			// Note: don't add s if it has no functions to run. We need
			// s during linking to compute an ordering, but the runtime
			// doesn't need to know about it. About 1/2 of stdlib packages
			// fit in this bucket.
			sched.AddAddr(ctxt.Arch, s)
		}

		// Find all incoming edges into s.
		a := sort.Search(len(edges), func(i int) bool { return edges[i].to >= s })
		b := sort.Search(len(edges), func(i int) bool { return edges[i].to > s })

		// Decrement the import count for all packages that import s.
		// If the count reaches 0, that package is now ready to schedule.
		for _, e := range edges[a:b] {
			m[e.from]--
			if m[e.from] == 0 {
				h.push(ldr, e.from)
			}
		}
	}

	for s, n := range m {
		if n != 0 {
			Exitf("inittask for %s is not schedulable %d", ldr.SymName(s), n)
		}
	}
	return sched.Sym()
}
