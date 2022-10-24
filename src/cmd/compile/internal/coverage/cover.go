// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package coverage

// This package contains support routines for coverage "fixup" in the
// compiler, which happens when compiling a package whose source code
// has been run through "cmd/cover" to add instrumentation. The two
// important entry points are FixupVars (called prior to package init
// generation) and FixupInit (called following package init
// generation).

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/objabi"
	"internal/coverage"
	"strconv"
	"strings"
)

// Names records state information collected in the first fixup
// phase so that it can be passed to the second fixup phase.
type Names struct {
	MetaVar     *ir.Name
	PkgIdVar    *ir.Name
	InitFn      *ir.Func
	CounterMode coverage.CounterMode
	CounterGran coverage.CounterGranularity
}

// FixupVars is the first of two entry points for coverage compiler
// fixup. It collects and returns the package ID and meta-data
// variables being used for this "-cover" build, along with the
// coverage counter mode and granularity. It also reclassifies selected
// variables (for example, tagging coverage counter variables with
// flags so that they can be handled properly downstream).
func FixupVars() Names {
	metaVarName := base.Flag.Cfg.CoverageInfo.MetaVar
	pkgIdVarName := base.Flag.Cfg.CoverageInfo.PkgIdVar
	counterMode := base.Flag.Cfg.CoverageInfo.CounterMode
	counterGran := base.Flag.Cfg.CoverageInfo.CounterGranularity
	counterPrefix := base.Flag.Cfg.CoverageInfo.CounterPrefix
	var metavar *ir.Name
	var pkgidvar *ir.Name

	ckTypSanity := func(nm *ir.Name, tag string) {
		if nm.Type() == nil || nm.Type().HasPointers() {
			base.Fatalf("unsuitable %s %q mentioned in coveragecfg, improper type '%v'", tag, nm.Sym().Name, nm.Type())
		}
	}

	for _, n := range typecheck.Target.Decls {
		as, ok := n.(*ir.AssignStmt)
		if !ok {
			continue
		}
		nm, ok := as.X.(*ir.Name)
		if !ok {
			continue
		}
		s := nm.Sym()
		switch s.Name {
		case metaVarName:
			metavar = nm
			ckTypSanity(nm, "metavar")
			nm.MarkReadonly()
			continue
		case pkgIdVarName:
			pkgidvar = nm
			ckTypSanity(nm, "pkgidvar")
			nm.SetCoverageAuxVar(true)
			s := nm.Linksym()
			s.Type = objabi.SCOVERAGE_AUXVAR
			continue
		}
		if strings.HasPrefix(s.Name, counterPrefix) {
			ckTypSanity(nm, "countervar")
			nm.SetCoverageCounter(true)
			s := nm.Linksym()
			s.Type = objabi.SCOVERAGE_COUNTER
		}
	}
	cm := coverage.ParseCounterMode(counterMode)
	if cm == coverage.CtrModeInvalid {
		base.Fatalf("bad setting %q for covermode in coveragecfg:",
			counterMode)
	}
	var cg coverage.CounterGranularity
	switch counterGran {
	case "perblock":
		cg = coverage.CtrGranularityPerBlock
	case "perfunc":
		cg = coverage.CtrGranularityPerFunc
	default:
		base.Fatalf("bad setting %q for covergranularity in coveragecfg:",
			counterGran)
	}

	return Names{
		MetaVar:     metavar,
		PkgIdVar:    pkgidvar,
		CounterMode: cm,
		CounterGran: cg,
	}
}

// FixupInit is the second main entry point for coverage compiler
// fixup. It adds calls to the pkg init function as appropriate to
// register coverage-related variables with the runtime.
func FixupInit(cnames Names) {
	for _, n := range typecheck.Target.Decls {
		if fn, ok := n.(*ir.Func); ok && ir.FuncName(fn) == "init" {
			cnames.InitFn = fn
			break
		}
	}
	if cnames.InitFn == nil {
		panic("unexpected (no init func for -cover build)")
	}

	hashv, len := metaHashAndLen()
	if cnames.CounterMode != coverage.CtrModeTestMain {
		registerMeta(cnames, hashv, len)
	}
	if base.Ctxt.Pkgpath == "main" {
		addInitHookCall(cnames.InitFn, cnames.CounterMode)
	}
}

func metaHashAndLen() ([16]byte, int) {

	// Read meta-data hash from config entry.
	mhash := base.Flag.Cfg.CoverageInfo.MetaHash
	if len(mhash) != 32 {
		base.Fatalf("unexpected: got metahash length %d want 32", len(mhash))
	}
	var hv [16]byte
	for i := 0; i < 16; i++ {
		nib := string(mhash[i*2 : i*2+2])
		x, err := strconv.ParseInt(nib, 16, 32)
		if err != nil {
			base.Fatalf("metahash bad byte %q", nib)
		}
		hv[i] = byte(x)
	}

	// Return hash and meta-data len
	return hv, base.Flag.Cfg.CoverageInfo.MetaLen
}

func registerMeta(cnames Names, hashv [16]byte, mdlen int) {
	// Materialize expression for hash (an array literal)
	pos := cnames.InitFn.Pos()
	elist := make([]ir.Node, 0, 16)
	for i := 0; i < 16; i++ {
		elem := ir.NewInt(int64(hashv[i]))
		elist = append(elist, elem)
	}
	ht := types.NewArray(types.Types[types.TUINT8], 16)
	hashx := ir.NewCompLitExpr(pos, ir.OCOMPLIT, ht, elist)

	// Materalize expression corresponding to address of the meta-data symbol.
	mdax := typecheck.NodAddr(cnames.MetaVar)
	mdauspx := typecheck.ConvNop(mdax, types.Types[types.TUNSAFEPTR])

	// Materialize expression for length.
	lenx := ir.NewInt(int64(mdlen)) // untyped

	// Generate a call to runtime.addCovMeta, e.g.
	//
	//   pkgIdVar = runtime.addCovMeta(&sym, len, hash, pkgpath, pkid, cmode, cgran)
	//
	fn := typecheck.LookupRuntime("addCovMeta")
	pkid := coverage.HardCodedPkgID(base.Ctxt.Pkgpath)
	pkIdNode := ir.NewInt(int64(pkid))
	cmodeNode := ir.NewInt(int64(cnames.CounterMode))
	cgranNode := ir.NewInt(int64(cnames.CounterGran))
	pkPathNode := ir.NewString(base.Ctxt.Pkgpath)
	callx := typecheck.Call(pos, fn, []ir.Node{mdauspx, lenx, hashx,
		pkPathNode, pkIdNode, cmodeNode, cgranNode}, false)
	assign := callx
	if pkid == coverage.NotHardCoded {
		assign = typecheck.Stmt(ir.NewAssignStmt(pos, cnames.PkgIdVar, callx))
	}

	// Tack the call onto the start of our init function. We do this
	// early in the init since it's possible that instrumented function
	// bodies (with counter updates) might be inlined into init.
	cnames.InitFn.Body.Prepend(assign)
}

// addInitHookCall generates a call to runtime/coverage.initHook() and
// inserts it into the package main init function, which will kick off
// the process for coverage data writing (emit meta data, and register
// an exit hook to emit counter data).
func addInitHookCall(initfn *ir.Func, cmode coverage.CounterMode) {
	typecheck.InitCoverage()
	pos := initfn.Pos()
	istest := cmode == coverage.CtrModeTestMain
	initf := typecheck.LookupCoverage("initHook")
	istestNode := ir.NewBool(istest)
	args := []ir.Node{istestNode}
	callx := typecheck.Call(pos, initf, args, false)
	initfn.Body.Append(callx)
}
