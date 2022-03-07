// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package coverage

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

// Fixup is the main entry point for coverage compiler fixup. It
// collects and reclassifies the variables mentioned in the
// -coveragecfg file, then adds calls to the pkg init function as
// appropriate to register the proper variables with the runtime.
func Fixup() {
	metavar, pkgIdVar, initfn, covermode, covergran :=
		fixupMetaAndCounterVariables()
	hashv, len := metaHashAndLen()
	if covermode != coverage.CtrModeTestMain {
		registerMeta(metavar, initfn, hashv, len,
			pkgIdVar, covermode, covergran)
	}
	if base.Ctxt.Pkgpath == "main" {
		addInitHookCall(initfn, covermode)
	}
}

// fixupMetaAndCounterVariables collects and returns the package ID
// and meta-data variables being used for this "-cover" build, along
// with the init function for the package and the coverage mode. It
// also reclassifies certain variables (for example, tagging coverage
// counter variables with flags so that they can be handled properly
// downstream).
func fixupMetaAndCounterVariables() (*ir.Name, *ir.Name, *ir.Func, coverage.CounterMode, coverage.CounterGranularity) {
	metaVarName := base.Flag.Cfg.CoverageInfo.MetaVar
	pkgIdVarName := base.Flag.Cfg.CoverageInfo.PkgIdVar
	counterMode := base.Flag.Cfg.CoverageInfo.CounterMode
	counterGran := base.Flag.Cfg.CoverageInfo.CounterGranularity
	counterPrefix := base.Flag.Cfg.CoverageInfo.CounterPrefix
	var metavar *ir.Name
	var pkgidvar *ir.Name
	var initfn *ir.Func

	ckTypSanity := func(nm *ir.Name, tag string) {
		if nm.Type() == nil || nm.Type().HasPointers() {
			base.Fatalf("unsuitable %s %q mentioned in coveragecfg, improper type '%v'", tag, nm.Sym().Name, nm.Type())
		}
	}

	for _, n := range typecheck.Target.Decls {
		if fn, ok := n.(*ir.Func); ok && ir.FuncName(fn) == "init" {
			if initfn != nil {
				panic("unexpected")
			}
			initfn = fn
			continue
		}
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

	return metavar, pkgidvar, initfn, cm, cg
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

func registerMeta(mdname *ir.Name, initfn *ir.Func, hash [16]byte, mdlen int, pkgIdVar *ir.Name, cmode coverage.CounterMode, cgran coverage.CounterGranularity) {
	// Materialize expression for hash (an array literal)
	pos := initfn.Pos()
	elist := make([]ir.Node, 0, 16)
	for i := 0; i < 16; i++ {
		elem := ir.NewInt(int64(hash[i]))
		elist = append(elist, elem)
	}
	ht := types.NewArray(types.Types[types.TUINT8], 16)
	hashx := ir.NewCompLitExpr(pos, ir.OCOMPLIT, ht, elist)

	// Materalize expression corresponding to address of the meta-data symbol.
	mdax := typecheck.NodAddr(mdname)
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
	cmodeNode := ir.NewInt(int64(cmode))
	cgranNode := ir.NewInt(int64(cgran))
	pkPathNode := ir.NewString(base.Ctxt.Pkgpath)
	callx := typecheck.Call(pos, fn, []ir.Node{mdauspx, lenx, hashx,
		pkPathNode, pkIdNode, cmodeNode, cgranNode}, false)
	assign := callx
	if pkid == coverage.NotHardCoded {
		assign = typecheck.Stmt(ir.NewAssignStmt(pos, pkgIdVar, callx))
	}

	// Tack the call onto the start of our init function. We do this
	// early in the init since it's possible that instrumented function
	// bodies (with counter updates) might be inlined into init.
	initfn.Body.Prepend(assign)
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
