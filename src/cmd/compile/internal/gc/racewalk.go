// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
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
	if myimportpath != "" {
		for _, p := range pkgs {
			if myimportpath == p {
				return true
			}
		}
	}

	return false
}

func instrument(fn *Node) {
	if fn.Func.Pragma&Norace != 0 {
		return
	}

	if !flag_race || !ispkgin(norace_inst_pkgs) {
		fn.Func.SetInstrumentBody(true)
	}

	if flag_race {
		lno := lineno
		lineno = src.NoXPos

		if thearch.LinkArch.Arch.Family != sys.AMD64 {
			fn.Func.Enter.Prepend(mkcall("racefuncenterfp", nil, nil))
			fn.Func.Exit.Append(mkcall("racefuncexit", nil, nil))
		} else {

			// nodpc is the PC of the caller as extracted by
			// getcallerpc. We use -widthptr(FP) for x86.
			// This only works for amd64. This will not
			// work on arm or others that might support
			// race in the future.
			nodpc := nodfp.copy()
			nodpc.Type = types.Types[TUINTPTR]
			nodpc.Xoffset = int64(-Widthptr)
			fn.Func.Dcl = append(fn.Func.Dcl, nodpc)
			fn.Func.Enter.Prepend(mkcall("racefuncenter", nil, nil, nodpc))
			fn.Func.Exit.Append(mkcall("racefuncexit", nil, nil))
		}
		lineno = lno
	}
}
