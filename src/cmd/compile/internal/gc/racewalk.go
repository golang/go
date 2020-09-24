// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"cmd/internal/sys"
)

// The racewalk pass is currently handled in three parts.
//
// First, for flag_race, it inserts calls to racefuncenter and
// racefuncexit at the start and end (respectively) of each
// function. This is handled below.
//
// Second, during buildssa, it inserts appropriate instrumentation
// calls immediately before each memory load or store. This is handled
// by the (*state).instrument method in ssa.go, so here we just set
// the Func.InstrumentBody flag as needed. For background on why this
// is done during SSA construction rather than a separate SSA pass,
// see issue #19054.
//
// Third we remove calls to racefuncenter and racefuncexit, for leaf
// functions without instrumented operations. This is done as part of
// ssa opt pass via special rule.

// TODO(dvyukov): do not instrument initialization as writes:
// a := make([]int, 10)

// Do not instrument the following packages at all,
// at best instrumentation would cause infinite recursion.
var omit_pkgs = []string{
	"runtime/internal/atomic",
	"runtime/internal/sys",
	"runtime/internal/math",
	"runtime",
	"runtime/race",
	"runtime/msan",
	"internal/cpu",
}

// Don't insert racefuncenterfp/racefuncexit into the following packages.
// Memory accesses in the packages are either uninteresting or will cause false positives.
var norace_inst_pkgs = []string{"sync", "sync/atomic"}

func ispkgin(pkgs []string) bool {
	if base.Ctxt.Pkgpath != "" {
		for _, p := range pkgs {
			if base.Ctxt.Pkgpath == p {
				return true
			}
		}
	}

	return false
}

func instrument(fn *ir.Func) {
	if fn.Pragma&ir.Norace != 0 || (fn.Sym().Linksym() != nil && fn.Sym().Linksym().ABIWrapper()) {
		return
	}

	if !base.Flag.Race || !ispkgin(norace_inst_pkgs) {
		fn.SetInstrumentBody(true)
	}

	if base.Flag.Race {
		lno := base.Pos
		base.Pos = src.NoXPos

		if thearch.LinkArch.Arch.Family != sys.AMD64 {
			fn.Enter.Prepend(mkcall("racefuncenterfp", nil, nil))
			fn.Exit.Append(mkcall("racefuncexit", nil, nil))
		} else {

			// nodpc is the PC of the caller as extracted by
			// getcallerpc. We use -widthptr(FP) for x86.
			// This only works for amd64. This will not
			// work on arm or others that might support
			// race in the future.
			nodpc := nodfp.CloneName()
			nodpc.SetType(types.Types[types.TUINTPTR])
			nodpc.SetFrameOffset(int64(-Widthptr))
			fn.Dcl = append(fn.Dcl, nodpc)
			fn.Enter.Prepend(mkcall("racefuncenter", nil, nil, nodpc))
			fn.Exit.Append(mkcall("racefuncexit", nil, nil))
		}
		base.Pos = lno
	}
}
