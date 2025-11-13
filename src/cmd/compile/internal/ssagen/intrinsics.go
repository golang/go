// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssagen

import (
	"fmt"
	"internal/abi"
	"internal/buildcfg"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/sys"
)

var intrinsics intrinsicBuilders

// An intrinsicBuilder converts a call node n into an ssa value that
// implements that call as an intrinsic. args is a list of arguments to the func.
type intrinsicBuilder func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value

type intrinsicKey struct {
	arch *sys.Arch
	pkg  string
	fn   string
}

// intrinsicBuildConfig specifies the config to use for intrinsic building.
type intrinsicBuildConfig struct {
	instrumenting bool

	go386     string
	goamd64   int
	goarm     buildcfg.GoarmFeatures
	goarm64   buildcfg.Goarm64Features
	gomips    string
	gomips64  string
	goppc64   int
	goriscv64 int
}

type intrinsicBuilders map[intrinsicKey]intrinsicBuilder

// add adds the intrinsic builder b for pkg.fn for the given architecture.
func (ib intrinsicBuilders) add(arch *sys.Arch, pkg, fn string, b intrinsicBuilder) {
	if _, found := ib[intrinsicKey{arch, pkg, fn}]; found {
		panic(fmt.Sprintf("intrinsic already exists for %v.%v on %v", pkg, fn, arch.Name))
	}
	ib[intrinsicKey{arch, pkg, fn}] = b
}

// addForArchs adds the intrinsic builder b for pkg.fn for the given architectures.
func (ib intrinsicBuilders) addForArchs(pkg, fn string, b intrinsicBuilder, archs ...*sys.Arch) {
	for _, arch := range archs {
		ib.add(arch, pkg, fn, b)
	}
}

// addForFamilies does the same as addForArchs but operates on architecture families.
func (ib intrinsicBuilders) addForFamilies(pkg, fn string, b intrinsicBuilder, archFamilies ...sys.ArchFamily) {
	for _, arch := range sys.Archs {
		if arch.InFamily(archFamilies...) {
			intrinsics.add(arch, pkg, fn, b)
		}
	}
}

// alias aliases pkg.fn to targetPkg.targetFn for all architectures in archs
// for which targetPkg.targetFn already exists.
func (ib intrinsicBuilders) alias(pkg, fn, targetPkg, targetFn string, archs ...*sys.Arch) {
	// TODO(jsing): Consider making this work even if the alias is added
	// before the intrinsic.
	aliased := false
	for _, arch := range archs {
		if b := intrinsics.lookup(arch, targetPkg, targetFn); b != nil {
			intrinsics.add(arch, pkg, fn, b)
			aliased = true
		}
	}
	if !aliased {
		panic(fmt.Sprintf("attempted to alias undefined intrinsic: %s.%s", pkg, fn))
	}
}

// lookup looks up the intrinsic for a pkg.fn on the specified architecture.
func (ib intrinsicBuilders) lookup(arch *sys.Arch, pkg, fn string) intrinsicBuilder {
	return intrinsics[intrinsicKey{arch, pkg, fn}]
}

func initIntrinsics(cfg *intrinsicBuildConfig) {
	if cfg == nil {
		cfg = &intrinsicBuildConfig{
			instrumenting: base.Flag.Cfg.Instrumenting,
			go386:         buildcfg.GO386,
			goamd64:       buildcfg.GOAMD64,
			goarm:         buildcfg.GOARM,
			goarm64:       buildcfg.GOARM64,
			gomips:        buildcfg.GOMIPS,
			gomips64:      buildcfg.GOMIPS64,
			goppc64:       buildcfg.GOPPC64,
			goriscv64:     buildcfg.GORISCV64,
		}
	}
	intrinsics = intrinsicBuilders{}

	var p4 []*sys.Arch
	var p8 []*sys.Arch
	var lwatomics []*sys.Arch
	for _, a := range sys.Archs {
		if a.PtrSize == 4 {
			p4 = append(p4, a)
		} else {
			p8 = append(p8, a)
		}
		if a.Family != sys.PPC64 {
			lwatomics = append(lwatomics, a)
		}
	}
	all := sys.Archs[:]

	add := func(pkg, fn string, b intrinsicBuilder, archs ...*sys.Arch) {
		intrinsics.addForArchs(pkg, fn, b, archs...)
	}
	addF := func(pkg, fn string, b intrinsicBuilder, archFamilies ...sys.ArchFamily) {
		intrinsics.addForFamilies(pkg, fn, b, archFamilies...)
	}
	alias := func(pkg, fn, pkg2, fn2 string, archs ...*sys.Arch) {
		intrinsics.alias(pkg, fn, pkg2, fn2, archs...)
	}

	/******** runtime ********/
	if !cfg.instrumenting {
		add("runtime", "slicebytetostringtmp",
			func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
				// Compiler frontend optimizations emit OBYTES2STRTMP nodes
				// for the backend instead of slicebytetostringtmp calls
				// when not instrumenting.
				return s.newValue2(ssa.OpStringMake, n.Type(), args[0], args[1])
			},
			all...)
	}
	addF("internal/runtime/math", "MulUintptr",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			if s.config.PtrSize == 4 {
				return s.newValue2(ssa.OpMul32uover, types.NewTuple(types.Types[types.TUINT], types.Types[types.TUINT]), args[0], args[1])
			}
			return s.newValue2(ssa.OpMul64uover, types.NewTuple(types.Types[types.TUINT], types.Types[types.TUINT]), args[0], args[1])
		},
		sys.AMD64, sys.I386, sys.Loong64, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.ARM64)
	add("runtime", "KeepAlive",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			data := s.newValue1(ssa.OpIData, s.f.Config.Types.BytePtr, args[0])
			s.vars[memVar] = s.newValue2(ssa.OpKeepAlive, types.TypeMem, data, s.mem())
			return nil
		},
		all...)

	addF("runtime", "publicationBarrier",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			s.vars[memVar] = s.newValue1(ssa.OpPubBarrier, types.TypeMem, s.mem())
			return nil
		},
		sys.ARM64, sys.Loong64, sys.MIPS, sys.MIPS64, sys.PPC64, sys.RISCV64)

	/******** internal/runtime/sys ********/
	add("internal/runtime/sys", "GetCallerPC",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue0(ssa.OpGetCallerPC, s.f.Config.Types.Uintptr)
		},
		all...)

	add("internal/runtime/sys", "GetCallerSP",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpGetCallerSP, s.f.Config.Types.Uintptr, s.mem())
		},
		all...)

	add("internal/runtime/sys", "GetClosurePtr",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue0(ssa.OpGetClosurePtr, s.f.Config.Types.Uintptr)
		},
		all...)

	addF("internal/runtime/sys", "Bswap32",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBswap32, types.Types[types.TUINT32], args[0])
		},
		sys.AMD64, sys.I386, sys.ARM64, sys.ARM, sys.Loong64, sys.S390X)
	addF("internal/runtime/sys", "Bswap64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBswap64, types.Types[types.TUINT64], args[0])
		},
		sys.AMD64, sys.I386, sys.ARM64, sys.ARM, sys.Loong64, sys.S390X)

	if cfg.goppc64 >= 10 {
		// Use only on Power10 as the new byte reverse instructions that Power10 provide
		// make it worthwhile as an intrinsic
		addF("internal/runtime/sys", "Bswap32",
			func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
				return s.newValue1(ssa.OpBswap32, types.Types[types.TUINT32], args[0])
			},
			sys.PPC64)
		addF("internal/runtime/sys", "Bswap64",
			func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
				return s.newValue1(ssa.OpBswap64, types.Types[types.TUINT64], args[0])
			},
			sys.PPC64)
	}

	if cfg.goriscv64 >= 22 {
		addF("internal/runtime/sys", "Bswap32",
			func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
				return s.newValue1(ssa.OpBswap32, types.Types[types.TUINT32], args[0])
			},
			sys.RISCV64)
		addF("internal/runtime/sys", "Bswap64",
			func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
				return s.newValue1(ssa.OpBswap64, types.Types[types.TUINT64], args[0])
			},
			sys.RISCV64)
	}

	/****** Prefetch ******/
	makePrefetchFunc := func(op ssa.Op) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			s.vars[memVar] = s.newValue2(op, types.TypeMem, args[0], s.mem())
			return nil
		}
	}

	// Make Prefetch intrinsics for supported platforms
	// On the unsupported platforms stub function will be eliminated
	addF("internal/runtime/sys", "Prefetch", makePrefetchFunc(ssa.OpPrefetchCache),
		sys.AMD64, sys.ARM64, sys.Loong64, sys.PPC64)
	addF("internal/runtime/sys", "PrefetchStreamed", makePrefetchFunc(ssa.OpPrefetchCacheStreamed),
		sys.AMD64, sys.ARM64, sys.Loong64, sys.PPC64)

	/******** internal/runtime/atomic ********/
	type atomicOpEmitter func(s *state, n *ir.CallExpr, args []*ssa.Value, op ssa.Op, typ types.Kind, needReturn bool)

	addF("internal/runtime/atomic", "Load",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v := s.newValue2(ssa.OpAtomicLoad32, types.NewTuple(types.Types[types.TUINT32], types.TypeMem), args[0], s.mem())
			s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[types.TUINT32], v)
		},
		sys.AMD64, sys.ARM64, sys.Loong64, sys.MIPS, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.S390X)
	addF("internal/runtime/atomic", "Load8",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v := s.newValue2(ssa.OpAtomicLoad8, types.NewTuple(types.Types[types.TUINT8], types.TypeMem), args[0], s.mem())
			s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[types.TUINT8], v)
		},
		sys.AMD64, sys.ARM64, sys.Loong64, sys.MIPS, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.S390X)
	addF("internal/runtime/atomic", "Load64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v := s.newValue2(ssa.OpAtomicLoad64, types.NewTuple(types.Types[types.TUINT64], types.TypeMem), args[0], s.mem())
			s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[types.TUINT64], v)
		},
		sys.AMD64, sys.ARM64, sys.Loong64, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.S390X)
	addF("internal/runtime/atomic", "LoadAcq",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v := s.newValue2(ssa.OpAtomicLoadAcq32, types.NewTuple(types.Types[types.TUINT32], types.TypeMem), args[0], s.mem())
			s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[types.TUINT32], v)
		},
		sys.PPC64)
	addF("internal/runtime/atomic", "LoadAcq64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v := s.newValue2(ssa.OpAtomicLoadAcq64, types.NewTuple(types.Types[types.TUINT64], types.TypeMem), args[0], s.mem())
			s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[types.TUINT64], v)
		},
		sys.PPC64)
	addF("internal/runtime/atomic", "Loadp",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v := s.newValue2(ssa.OpAtomicLoadPtr, types.NewTuple(s.f.Config.Types.BytePtr, types.TypeMem), args[0], s.mem())
			s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, s.f.Config.Types.BytePtr, v)
		},
		sys.AMD64, sys.ARM64, sys.Loong64, sys.MIPS, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.S390X)

	addF("internal/runtime/atomic", "Store",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			s.vars[memVar] = s.newValue3(ssa.OpAtomicStore32, types.TypeMem, args[0], args[1], s.mem())
			return nil
		},
		sys.AMD64, sys.ARM64, sys.MIPS, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.S390X)
	addF("internal/runtime/atomic", "Store8",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			s.vars[memVar] = s.newValue3(ssa.OpAtomicStore8, types.TypeMem, args[0], args[1], s.mem())
			return nil
		},
		sys.AMD64, sys.ARM64, sys.MIPS, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.S390X)
	addF("internal/runtime/atomic", "Store64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			s.vars[memVar] = s.newValue3(ssa.OpAtomicStore64, types.TypeMem, args[0], args[1], s.mem())
			return nil
		},
		sys.AMD64, sys.ARM64, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.S390X)
	addF("internal/runtime/atomic", "StorepNoWB",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			s.vars[memVar] = s.newValue3(ssa.OpAtomicStorePtrNoWB, types.TypeMem, args[0], args[1], s.mem())
			return nil
		},
		sys.AMD64, sys.ARM64, sys.Loong64, sys.MIPS, sys.MIPS64, sys.RISCV64, sys.S390X)
	addF("internal/runtime/atomic", "StoreRel",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			s.vars[memVar] = s.newValue3(ssa.OpAtomicStoreRel32, types.TypeMem, args[0], args[1], s.mem())
			return nil
		},
		sys.PPC64)
	addF("internal/runtime/atomic", "StoreRel64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			s.vars[memVar] = s.newValue3(ssa.OpAtomicStoreRel64, types.TypeMem, args[0], args[1], s.mem())
			return nil
		},
		sys.PPC64)

	makeAtomicStoreGuardedIntrinsicLoong64 := func(op0, op1 ssa.Op, typ types.Kind, emit atomicOpEmitter) intrinsicBuilder {
		return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			// Target Atomic feature is identified by dynamic detection
			addr := s.entryNewValue1A(ssa.OpAddr, types.Types[types.TBOOL].PtrTo(), ir.Syms.Loong64HasLAM_BH, s.sb)
			v := s.load(types.Types[types.TBOOL], addr)
			b := s.endBlock()
			b.Kind = ssa.BlockIf
			b.SetControl(v)
			bTrue := s.f.NewBlock(ssa.BlockPlain)
			bFalse := s.f.NewBlock(ssa.BlockPlain)
			bEnd := s.f.NewBlock(ssa.BlockPlain)
			b.AddEdgeTo(bTrue)
			b.AddEdgeTo(bFalse)
			b.Likely = ssa.BranchLikely

			// We have atomic instructions - use it directly.
			s.startBlock(bTrue)
			emit(s, n, args, op1, typ, false)
			s.endBlock().AddEdgeTo(bEnd)

			// Use original instruction sequence.
			s.startBlock(bFalse)
			emit(s, n, args, op0, typ, false)
			s.endBlock().AddEdgeTo(bEnd)

			// Merge results.
			s.startBlock(bEnd)

			return nil
		}
	}

	atomicStoreEmitterLoong64 := func(s *state, n *ir.CallExpr, args []*ssa.Value, op ssa.Op, typ types.Kind, needReturn bool) {
		v := s.newValue3(op, types.NewTuple(types.Types[typ], types.TypeMem), args[0], args[1], s.mem())
		s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
		if needReturn {
			s.vars[n] = s.newValue1(ssa.OpSelect0, types.Types[typ], v)
		}
	}

	addF("internal/runtime/atomic", "Store8",
		makeAtomicStoreGuardedIntrinsicLoong64(ssa.OpAtomicStore8, ssa.OpAtomicStore8Variant, types.TUINT8, atomicStoreEmitterLoong64),
		sys.Loong64)
	addF("internal/runtime/atomic", "Store",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			s.vars[memVar] = s.newValue3(ssa.OpAtomicStore32Variant, types.TypeMem, args[0], args[1], s.mem())
			return nil
		},
		sys.Loong64)
	addF("internal/runtime/atomic", "Store64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			s.vars[memVar] = s.newValue3(ssa.OpAtomicStore64Variant, types.TypeMem, args[0], args[1], s.mem())
			return nil
		},
		sys.Loong64)

	addF("internal/runtime/atomic", "Xchg8",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v := s.newValue3(ssa.OpAtomicExchange8, types.NewTuple(types.Types[types.TUINT8], types.TypeMem), args[0], args[1], s.mem())
			s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[types.TUINT8], v)
		},
		sys.AMD64, sys.PPC64)
	addF("internal/runtime/atomic", "Xchg",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v := s.newValue3(ssa.OpAtomicExchange32, types.NewTuple(types.Types[types.TUINT32], types.TypeMem), args[0], args[1], s.mem())
			s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[types.TUINT32], v)
		},
		sys.AMD64, sys.Loong64, sys.MIPS, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.S390X)
	addF("internal/runtime/atomic", "Xchg64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v := s.newValue3(ssa.OpAtomicExchange64, types.NewTuple(types.Types[types.TUINT64], types.TypeMem), args[0], args[1], s.mem())
			s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[types.TUINT64], v)
		},
		sys.AMD64, sys.Loong64, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.S390X)

	makeAtomicGuardedIntrinsicARM64common := func(op0, op1 ssa.Op, typ types.Kind, emit atomicOpEmitter, needReturn bool) intrinsicBuilder {

		return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			if cfg.goarm64.LSE {
				emit(s, n, args, op1, typ, needReturn)
			} else {
				// Target Atomic feature is identified by dynamic detection
				addr := s.entryNewValue1A(ssa.OpAddr, types.Types[types.TBOOL].PtrTo(), ir.Syms.ARM64HasATOMICS, s.sb)
				v := s.load(types.Types[types.TBOOL], addr)
				b := s.endBlock()
				b.Kind = ssa.BlockIf
				b.SetControl(v)
				bTrue := s.f.NewBlock(ssa.BlockPlain)
				bFalse := s.f.NewBlock(ssa.BlockPlain)
				bEnd := s.f.NewBlock(ssa.BlockPlain)
				b.AddEdgeTo(bTrue)
				b.AddEdgeTo(bFalse)
				b.Likely = ssa.BranchLikely

				// We have atomic instructions - use it directly.
				s.startBlock(bTrue)
				emit(s, n, args, op1, typ, needReturn)
				s.endBlock().AddEdgeTo(bEnd)

				// Use original instruction sequence.
				s.startBlock(bFalse)
				emit(s, n, args, op0, typ, needReturn)
				s.endBlock().AddEdgeTo(bEnd)

				// Merge results.
				s.startBlock(bEnd)
			}
			if needReturn {
				return s.variable(n, types.Types[typ])
			} else {
				return nil
			}
		}
	}
	makeAtomicGuardedIntrinsicARM64 := func(op0, op1 ssa.Op, typ types.Kind, emit atomicOpEmitter) intrinsicBuilder {
		return makeAtomicGuardedIntrinsicARM64common(op0, op1, typ, emit, true)
	}
	makeAtomicGuardedIntrinsicARM64old := func(op0, op1 ssa.Op, typ types.Kind, emit atomicOpEmitter) intrinsicBuilder {
		return makeAtomicGuardedIntrinsicARM64common(op0, op1, typ, emit, false)
	}

	atomicEmitterARM64 := func(s *state, n *ir.CallExpr, args []*ssa.Value, op ssa.Op, typ types.Kind, needReturn bool) {
		v := s.newValue3(op, types.NewTuple(types.Types[typ], types.TypeMem), args[0], args[1], s.mem())
		s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
		if needReturn {
			s.vars[n] = s.newValue1(ssa.OpSelect0, types.Types[typ], v)
		}
	}
	addF("internal/runtime/atomic", "Xchg8",
		makeAtomicGuardedIntrinsicARM64(ssa.OpAtomicExchange8, ssa.OpAtomicExchange8Variant, types.TUINT8, atomicEmitterARM64),
		sys.ARM64)
	addF("internal/runtime/atomic", "Xchg",
		makeAtomicGuardedIntrinsicARM64(ssa.OpAtomicExchange32, ssa.OpAtomicExchange32Variant, types.TUINT32, atomicEmitterARM64),
		sys.ARM64)
	addF("internal/runtime/atomic", "Xchg64",
		makeAtomicGuardedIntrinsicARM64(ssa.OpAtomicExchange64, ssa.OpAtomicExchange64Variant, types.TUINT64, atomicEmitterARM64),
		sys.ARM64)

	makeAtomicXchg8GuardedIntrinsicLoong64 := func(op ssa.Op) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			addr := s.entryNewValue1A(ssa.OpAddr, types.Types[types.TBOOL].PtrTo(), ir.Syms.Loong64HasLAM_BH, s.sb)
			v := s.load(types.Types[types.TBOOL], addr)
			b := s.endBlock()
			b.Kind = ssa.BlockIf
			b.SetControl(v)
			bTrue := s.f.NewBlock(ssa.BlockPlain)
			bFalse := s.f.NewBlock(ssa.BlockPlain)
			bEnd := s.f.NewBlock(ssa.BlockPlain)
			b.AddEdgeTo(bTrue)
			b.AddEdgeTo(bFalse)
			b.Likely = ssa.BranchLikely // most loong64 machines support the amswapdb.b

			// We have the intrinsic - use it directly.
			s.startBlock(bTrue)
			s.vars[n] = s.newValue3(op, types.NewTuple(types.Types[types.TUINT8], types.TypeMem), args[0], args[1], s.mem())
			s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, s.vars[n])
			s.vars[n] = s.newValue1(ssa.OpSelect0, types.Types[types.TUINT8], s.vars[n])
			s.endBlock().AddEdgeTo(bEnd)

			// Call the pure Go version.
			s.startBlock(bFalse)
			s.vars[n] = s.callResult(n, callNormal) // types.Types[TUINT8]
			s.endBlock().AddEdgeTo(bEnd)

			// Merge results.
			s.startBlock(bEnd)
			return s.variable(n, types.Types[types.TUINT8])
		}
	}
	addF("internal/runtime/atomic", "Xchg8",
		makeAtomicXchg8GuardedIntrinsicLoong64(ssa.OpAtomicExchange8Variant),
		sys.Loong64)

	addF("internal/runtime/atomic", "Xadd",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v := s.newValue3(ssa.OpAtomicAdd32, types.NewTuple(types.Types[types.TUINT32], types.TypeMem), args[0], args[1], s.mem())
			s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[types.TUINT32], v)
		},
		sys.AMD64, sys.Loong64, sys.MIPS, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.S390X)
	addF("internal/runtime/atomic", "Xadd64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v := s.newValue3(ssa.OpAtomicAdd64, types.NewTuple(types.Types[types.TUINT64], types.TypeMem), args[0], args[1], s.mem())
			s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[types.TUINT64], v)
		},
		sys.AMD64, sys.Loong64, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.S390X)

	addF("internal/runtime/atomic", "Xadd",
		makeAtomicGuardedIntrinsicARM64(ssa.OpAtomicAdd32, ssa.OpAtomicAdd32Variant, types.TUINT32, atomicEmitterARM64),
		sys.ARM64)
	addF("internal/runtime/atomic", "Xadd64",
		makeAtomicGuardedIntrinsicARM64(ssa.OpAtomicAdd64, ssa.OpAtomicAdd64Variant, types.TUINT64, atomicEmitterARM64),
		sys.ARM64)

	addF("internal/runtime/atomic", "Cas",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v := s.newValue4(ssa.OpAtomicCompareAndSwap32, types.NewTuple(types.Types[types.TBOOL], types.TypeMem), args[0], args[1], args[2], s.mem())
			s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[types.TBOOL], v)
		},
		sys.AMD64, sys.MIPS, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.S390X)
	addF("internal/runtime/atomic", "Cas64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v := s.newValue4(ssa.OpAtomicCompareAndSwap64, types.NewTuple(types.Types[types.TBOOL], types.TypeMem), args[0], args[1], args[2], s.mem())
			s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[types.TBOOL], v)
		},
		sys.AMD64, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.S390X)
	addF("internal/runtime/atomic", "CasRel",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v := s.newValue4(ssa.OpAtomicCompareAndSwap32, types.NewTuple(types.Types[types.TBOOL], types.TypeMem), args[0], args[1], args[2], s.mem())
			s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[types.TBOOL], v)
		},
		sys.PPC64)

	atomicCasEmitterARM64 := func(s *state, n *ir.CallExpr, args []*ssa.Value, op ssa.Op, typ types.Kind, needReturn bool) {
		v := s.newValue4(op, types.NewTuple(types.Types[types.TBOOL], types.TypeMem), args[0], args[1], args[2], s.mem())
		s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
		if needReturn {
			s.vars[n] = s.newValue1(ssa.OpSelect0, types.Types[typ], v)
		}
	}

	addF("internal/runtime/atomic", "Cas",
		makeAtomicGuardedIntrinsicARM64(ssa.OpAtomicCompareAndSwap32, ssa.OpAtomicCompareAndSwap32Variant, types.TBOOL, atomicCasEmitterARM64),
		sys.ARM64)
	addF("internal/runtime/atomic", "Cas64",
		makeAtomicGuardedIntrinsicARM64(ssa.OpAtomicCompareAndSwap64, ssa.OpAtomicCompareAndSwap64Variant, types.TBOOL, atomicCasEmitterARM64),
		sys.ARM64)

	atomicCasEmitterLoong64 := func(s *state, n *ir.CallExpr, args []*ssa.Value, op ssa.Op, typ types.Kind, needReturn bool) {
		v := s.newValue4(op, types.NewTuple(types.Types[types.TBOOL], types.TypeMem), args[0], args[1], args[2], s.mem())
		s.vars[memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
		if needReturn {
			s.vars[n] = s.newValue1(ssa.OpSelect0, types.Types[typ], v)
		}
	}

	makeAtomicCasGuardedIntrinsicLoong64 := func(op0, op1 ssa.Op, emit atomicOpEmitter) intrinsicBuilder {
		return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			// Target Atomic feature is identified by dynamic detection
			addr := s.entryNewValue1A(ssa.OpAddr, types.Types[types.TBOOL].PtrTo(), ir.Syms.Loong64HasLAMCAS, s.sb)
			v := s.load(types.Types[types.TBOOL], addr)
			b := s.endBlock()
			b.Kind = ssa.BlockIf
			b.SetControl(v)
			bTrue := s.f.NewBlock(ssa.BlockPlain)
			bFalse := s.f.NewBlock(ssa.BlockPlain)
			bEnd := s.f.NewBlock(ssa.BlockPlain)
			b.AddEdgeTo(bTrue)
			b.AddEdgeTo(bFalse)
			b.Likely = ssa.BranchLikely

			// We have atomic instructions - use it directly.
			s.startBlock(bTrue)
			emit(s, n, args, op1, types.TBOOL, true)
			s.endBlock().AddEdgeTo(bEnd)

			// Use original instruction sequence.
			s.startBlock(bFalse)
			emit(s, n, args, op0, types.TBOOL, true)
			s.endBlock().AddEdgeTo(bEnd)

			// Merge results.
			s.startBlock(bEnd)

			return s.variable(n, types.Types[types.TBOOL])
		}
	}

	addF("internal/runtime/atomic", "Cas",
		makeAtomicCasGuardedIntrinsicLoong64(ssa.OpAtomicCompareAndSwap32, ssa.OpAtomicCompareAndSwap32Variant, atomicCasEmitterLoong64),
		sys.Loong64)
	addF("internal/runtime/atomic", "Cas64",
		makeAtomicCasGuardedIntrinsicLoong64(ssa.OpAtomicCompareAndSwap64, ssa.OpAtomicCompareAndSwap64Variant, atomicCasEmitterLoong64),
		sys.Loong64)

	// Old-style atomic logical operation API (all supported archs except arm64).
	addF("internal/runtime/atomic", "And8",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			s.vars[memVar] = s.newValue3(ssa.OpAtomicAnd8, types.TypeMem, args[0], args[1], s.mem())
			return nil
		},
		sys.AMD64, sys.Loong64, sys.MIPS, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.S390X)
	addF("internal/runtime/atomic", "And",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			s.vars[memVar] = s.newValue3(ssa.OpAtomicAnd32, types.TypeMem, args[0], args[1], s.mem())
			return nil
		},
		sys.AMD64, sys.Loong64, sys.MIPS, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.S390X)
	addF("internal/runtime/atomic", "Or8",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			s.vars[memVar] = s.newValue3(ssa.OpAtomicOr8, types.TypeMem, args[0], args[1], s.mem())
			return nil
		},
		sys.AMD64, sys.Loong64, sys.MIPS, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.S390X)
	addF("internal/runtime/atomic", "Or",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			s.vars[memVar] = s.newValue3(ssa.OpAtomicOr32, types.TypeMem, args[0], args[1], s.mem())
			return nil
		},
		sys.AMD64, sys.Loong64, sys.MIPS, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.S390X)

	// arm64 always uses the new-style atomic logical operations, for both the
	// old and new style API.
	addF("internal/runtime/atomic", "And8",
		makeAtomicGuardedIntrinsicARM64old(ssa.OpAtomicAnd8value, ssa.OpAtomicAnd8valueVariant, types.TUINT8, atomicEmitterARM64),
		sys.ARM64)
	addF("internal/runtime/atomic", "Or8",
		makeAtomicGuardedIntrinsicARM64old(ssa.OpAtomicOr8value, ssa.OpAtomicOr8valueVariant, types.TUINT8, atomicEmitterARM64),
		sys.ARM64)
	addF("internal/runtime/atomic", "And64",
		makeAtomicGuardedIntrinsicARM64(ssa.OpAtomicAnd64value, ssa.OpAtomicAnd64valueVariant, types.TUINT64, atomicEmitterARM64),
		sys.ARM64)
	addF("internal/runtime/atomic", "And32",
		makeAtomicGuardedIntrinsicARM64(ssa.OpAtomicAnd32value, ssa.OpAtomicAnd32valueVariant, types.TUINT32, atomicEmitterARM64),
		sys.ARM64)
	addF("internal/runtime/atomic", "And",
		makeAtomicGuardedIntrinsicARM64old(ssa.OpAtomicAnd32value, ssa.OpAtomicAnd32valueVariant, types.TUINT32, atomicEmitterARM64),
		sys.ARM64)
	addF("internal/runtime/atomic", "Or64",
		makeAtomicGuardedIntrinsicARM64(ssa.OpAtomicOr64value, ssa.OpAtomicOr64valueVariant, types.TUINT64, atomicEmitterARM64),
		sys.ARM64)
	addF("internal/runtime/atomic", "Or32",
		makeAtomicGuardedIntrinsicARM64(ssa.OpAtomicOr32value, ssa.OpAtomicOr32valueVariant, types.TUINT32, atomicEmitterARM64),
		sys.ARM64)
	addF("internal/runtime/atomic", "Or",
		makeAtomicGuardedIntrinsicARM64old(ssa.OpAtomicOr32value, ssa.OpAtomicOr32valueVariant, types.TUINT32, atomicEmitterARM64),
		sys.ARM64)

	// New-style atomic logical operations, which return the old memory value.
	addF("internal/runtime/atomic", "And64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v := s.newValue3(ssa.OpAtomicAnd64value, types.NewTuple(types.Types[types.TUINT64], types.TypeMem), args[0], args[1], s.mem())
			p0, p1 := s.split(v)
			s.vars[memVar] = p1
			return p0
		},
		sys.AMD64, sys.Loong64)
	addF("internal/runtime/atomic", "And32",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v := s.newValue3(ssa.OpAtomicAnd32value, types.NewTuple(types.Types[types.TUINT32], types.TypeMem), args[0], args[1], s.mem())
			p0, p1 := s.split(v)
			s.vars[memVar] = p1
			return p0
		},
		sys.AMD64, sys.Loong64)
	addF("internal/runtime/atomic", "Or64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v := s.newValue3(ssa.OpAtomicOr64value, types.NewTuple(types.Types[types.TUINT64], types.TypeMem), args[0], args[1], s.mem())
			p0, p1 := s.split(v)
			s.vars[memVar] = p1
			return p0
		},
		sys.AMD64, sys.Loong64)
	addF("internal/runtime/atomic", "Or32",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v := s.newValue3(ssa.OpAtomicOr32value, types.NewTuple(types.Types[types.TUINT32], types.TypeMem), args[0], args[1], s.mem())
			p0, p1 := s.split(v)
			s.vars[memVar] = p1
			return p0
		},
		sys.AMD64, sys.Loong64)

	// Aliases for atomic load operations
	alias("internal/runtime/atomic", "Loadint32", "internal/runtime/atomic", "Load", all...)
	alias("internal/runtime/atomic", "Loadint64", "internal/runtime/atomic", "Load64", all...)
	alias("internal/runtime/atomic", "Loaduintptr", "internal/runtime/atomic", "Load", p4...)
	alias("internal/runtime/atomic", "Loaduintptr", "internal/runtime/atomic", "Load64", p8...)
	alias("internal/runtime/atomic", "Loaduint", "internal/runtime/atomic", "Load", p4...)
	alias("internal/runtime/atomic", "Loaduint", "internal/runtime/atomic", "Load64", p8...)
	alias("internal/runtime/atomic", "LoadAcq", "internal/runtime/atomic", "Load", lwatomics...)
	alias("internal/runtime/atomic", "LoadAcq64", "internal/runtime/atomic", "Load64", lwatomics...)
	alias("internal/runtime/atomic", "LoadAcquintptr", "internal/runtime/atomic", "LoadAcq", p4...)
	alias("sync", "runtime_LoadAcquintptr", "internal/runtime/atomic", "LoadAcq", p4...) // linknamed
	alias("internal/runtime/atomic", "LoadAcquintptr", "internal/runtime/atomic", "LoadAcq64", p8...)
	alias("sync", "runtime_LoadAcquintptr", "internal/runtime/atomic", "LoadAcq64", p8...) // linknamed

	// Aliases for atomic store operations
	alias("internal/runtime/atomic", "Storeint32", "internal/runtime/atomic", "Store", all...)
	alias("internal/runtime/atomic", "Storeint64", "internal/runtime/atomic", "Store64", all...)
	alias("internal/runtime/atomic", "Storeuintptr", "internal/runtime/atomic", "Store", p4...)
	alias("internal/runtime/atomic", "Storeuintptr", "internal/runtime/atomic", "Store64", p8...)
	alias("internal/runtime/atomic", "StoreRel", "internal/runtime/atomic", "Store", lwatomics...)
	alias("internal/runtime/atomic", "StoreRel64", "internal/runtime/atomic", "Store64", lwatomics...)
	alias("internal/runtime/atomic", "StoreReluintptr", "internal/runtime/atomic", "StoreRel", p4...)
	alias("sync", "runtime_StoreReluintptr", "internal/runtime/atomic", "StoreRel", p4...) // linknamed
	alias("internal/runtime/atomic", "StoreReluintptr", "internal/runtime/atomic", "StoreRel64", p8...)
	alias("sync", "runtime_StoreReluintptr", "internal/runtime/atomic", "StoreRel64", p8...) // linknamed

	// Aliases for atomic swap operations
	alias("internal/runtime/atomic", "Xchgint32", "internal/runtime/atomic", "Xchg", all...)
	alias("internal/runtime/atomic", "Xchgint64", "internal/runtime/atomic", "Xchg64", all...)
	alias("internal/runtime/atomic", "Xchguintptr", "internal/runtime/atomic", "Xchg", p4...)
	alias("internal/runtime/atomic", "Xchguintptr", "internal/runtime/atomic", "Xchg64", p8...)

	// Aliases for atomic add operations
	alias("internal/runtime/atomic", "Xaddint32", "internal/runtime/atomic", "Xadd", all...)
	alias("internal/runtime/atomic", "Xaddint64", "internal/runtime/atomic", "Xadd64", all...)
	alias("internal/runtime/atomic", "Xadduintptr", "internal/runtime/atomic", "Xadd", p4...)
	alias("internal/runtime/atomic", "Xadduintptr", "internal/runtime/atomic", "Xadd64", p8...)

	// Aliases for atomic CAS operations
	alias("internal/runtime/atomic", "Casint32", "internal/runtime/atomic", "Cas", all...)
	alias("internal/runtime/atomic", "Casint64", "internal/runtime/atomic", "Cas64", all...)
	alias("internal/runtime/atomic", "Casuintptr", "internal/runtime/atomic", "Cas", p4...)
	alias("internal/runtime/atomic", "Casuintptr", "internal/runtime/atomic", "Cas64", p8...)
	alias("internal/runtime/atomic", "Casp1", "internal/runtime/atomic", "Cas", p4...)
	alias("internal/runtime/atomic", "Casp1", "internal/runtime/atomic", "Cas64", p8...)
	alias("internal/runtime/atomic", "CasRel", "internal/runtime/atomic", "Cas", lwatomics...)

	// Aliases for atomic And/Or operations
	alias("internal/runtime/atomic", "Anduintptr", "internal/runtime/atomic", "And64", sys.ArchARM64, sys.ArchLoong64)
	alias("internal/runtime/atomic", "Oruintptr", "internal/runtime/atomic", "Or64", sys.ArchARM64, sys.ArchLoong64)

	/******** math ********/
	addF("math", "sqrt",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpSqrt, types.Types[types.TFLOAT64], args[0])
		},
		sys.I386, sys.AMD64, sys.ARM, sys.ARM64, sys.Loong64, sys.MIPS, sys.MIPS64, sys.PPC64, sys.RISCV64, sys.S390X, sys.Wasm)
	addF("math", "Trunc",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpTrunc, types.Types[types.TFLOAT64], args[0])
		},
		sys.ARM64, sys.PPC64, sys.S390X, sys.Wasm)
	addF("math", "Ceil",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpCeil, types.Types[types.TFLOAT64], args[0])
		},
		sys.ARM64, sys.PPC64, sys.S390X, sys.Wasm)
	addF("math", "Floor",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpFloor, types.Types[types.TFLOAT64], args[0])
		},
		sys.ARM64, sys.PPC64, sys.S390X, sys.Wasm)
	addF("math", "Round",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpRound, types.Types[types.TFLOAT64], args[0])
		},
		sys.ARM64, sys.PPC64, sys.S390X)
	addF("math", "RoundToEven",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpRoundToEven, types.Types[types.TFLOAT64], args[0])
		},
		sys.ARM64, sys.S390X, sys.Wasm)
	addF("math", "Abs",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpAbs, types.Types[types.TFLOAT64], args[0])
		},
		sys.ARM64, sys.ARM, sys.Loong64, sys.PPC64, sys.RISCV64, sys.Wasm, sys.MIPS, sys.MIPS64)
	addF("math", "Copysign",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue2(ssa.OpCopysign, types.Types[types.TFLOAT64], args[0], args[1])
		},
		sys.Loong64, sys.PPC64, sys.RISCV64, sys.Wasm)
	addF("math", "FMA",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue3(ssa.OpFMA, types.Types[types.TFLOAT64], args[0], args[1], args[2])
		},
		sys.ARM64, sys.Loong64, sys.PPC64, sys.RISCV64, sys.S390X)
	addF("math", "FMA",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			if cfg.goamd64 >= 3 {
				return s.newValue3(ssa.OpFMA, types.Types[types.TFLOAT64], args[0], args[1], args[2])
			}

			v := s.entryNewValue0A(ssa.OpHasCPUFeature, types.Types[types.TBOOL], ir.Syms.X86HasFMA)
			b := s.endBlock()
			b.Kind = ssa.BlockIf
			b.SetControl(v)
			bTrue := s.f.NewBlock(ssa.BlockPlain)
			bFalse := s.f.NewBlock(ssa.BlockPlain)
			bEnd := s.f.NewBlock(ssa.BlockPlain)
			b.AddEdgeTo(bTrue)
			b.AddEdgeTo(bFalse)
			b.Likely = ssa.BranchLikely // >= haswell cpus are common

			// We have the intrinsic - use it directly.
			s.startBlock(bTrue)
			s.vars[n] = s.newValue3(ssa.OpFMA, types.Types[types.TFLOAT64], args[0], args[1], args[2])
			s.endBlock().AddEdgeTo(bEnd)

			// Call the pure Go version.
			s.startBlock(bFalse)
			s.vars[n] = s.callResult(n, callNormal) // types.Types[TFLOAT64]
			s.endBlock().AddEdgeTo(bEnd)

			// Merge results.
			s.startBlock(bEnd)
			return s.variable(n, types.Types[types.TFLOAT64])
		},
		sys.AMD64)
	addF("math", "FMA",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			addr := s.entryNewValue1A(ssa.OpAddr, types.Types[types.TBOOL].PtrTo(), ir.Syms.ARMHasVFPv4, s.sb)
			v := s.load(types.Types[types.TBOOL], addr)
			b := s.endBlock()
			b.Kind = ssa.BlockIf
			b.SetControl(v)
			bTrue := s.f.NewBlock(ssa.BlockPlain)
			bFalse := s.f.NewBlock(ssa.BlockPlain)
			bEnd := s.f.NewBlock(ssa.BlockPlain)
			b.AddEdgeTo(bTrue)
			b.AddEdgeTo(bFalse)
			b.Likely = ssa.BranchLikely

			// We have the intrinsic - use it directly.
			s.startBlock(bTrue)
			s.vars[n] = s.newValue3(ssa.OpFMA, types.Types[types.TFLOAT64], args[0], args[1], args[2])
			s.endBlock().AddEdgeTo(bEnd)

			// Call the pure Go version.
			s.startBlock(bFalse)
			s.vars[n] = s.callResult(n, callNormal) // types.Types[TFLOAT64]
			s.endBlock().AddEdgeTo(bEnd)

			// Merge results.
			s.startBlock(bEnd)
			return s.variable(n, types.Types[types.TFLOAT64])
		},
		sys.ARM)

	makeRoundAMD64 := func(op ssa.Op) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			if cfg.goamd64 >= 2 {
				return s.newValue1(op, types.Types[types.TFLOAT64], args[0])
			}

			v := s.entryNewValue0A(ssa.OpHasCPUFeature, types.Types[types.TBOOL], ir.Syms.X86HasSSE41)
			b := s.endBlock()
			b.Kind = ssa.BlockIf
			b.SetControl(v)
			bTrue := s.f.NewBlock(ssa.BlockPlain)
			bFalse := s.f.NewBlock(ssa.BlockPlain)
			bEnd := s.f.NewBlock(ssa.BlockPlain)
			b.AddEdgeTo(bTrue)
			b.AddEdgeTo(bFalse)
			b.Likely = ssa.BranchLikely // most machines have sse4.1 nowadays

			// We have the intrinsic - use it directly.
			s.startBlock(bTrue)
			s.vars[n] = s.newValue1(op, types.Types[types.TFLOAT64], args[0])
			s.endBlock().AddEdgeTo(bEnd)

			// Call the pure Go version.
			s.startBlock(bFalse)
			s.vars[n] = s.callResult(n, callNormal) // types.Types[TFLOAT64]
			s.endBlock().AddEdgeTo(bEnd)

			// Merge results.
			s.startBlock(bEnd)
			return s.variable(n, types.Types[types.TFLOAT64])
		}
	}
	addF("math", "RoundToEven",
		makeRoundAMD64(ssa.OpRoundToEven),
		sys.AMD64)
	addF("math", "Floor",
		makeRoundAMD64(ssa.OpFloor),
		sys.AMD64)
	addF("math", "Ceil",
		makeRoundAMD64(ssa.OpCeil),
		sys.AMD64)
	addF("math", "Trunc",
		makeRoundAMD64(ssa.OpTrunc),
		sys.AMD64)

	/******** math/bits ********/
	addF("math/bits", "TrailingZeros64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpCtz64, types.Types[types.TINT], args[0])
		},
		sys.AMD64, sys.ARM64, sys.ARM, sys.Loong64, sys.S390X, sys.MIPS, sys.PPC64, sys.Wasm)
	addF("math/bits", "TrailingZeros64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			lo := s.newValue1(ssa.OpInt64Lo, types.Types[types.TUINT32], args[0])
			hi := s.newValue1(ssa.OpInt64Hi, types.Types[types.TUINT32], args[0])
			return s.newValue2(ssa.OpCtz64On32, types.Types[types.TINT], lo, hi)
		},
		sys.I386)
	addF("math/bits", "TrailingZeros32",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpCtz32, types.Types[types.TINT], args[0])
		},
		sys.AMD64, sys.I386, sys.ARM64, sys.ARM, sys.Loong64, sys.S390X, sys.MIPS, sys.PPC64, sys.Wasm)
	addF("math/bits", "TrailingZeros16",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpCtz16, types.Types[types.TINT], args[0])
		},
		sys.AMD64, sys.ARM, sys.ARM64, sys.I386, sys.MIPS, sys.Loong64, sys.PPC64, sys.S390X, sys.Wasm)
	addF("math/bits", "TrailingZeros8",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpCtz8, types.Types[types.TINT], args[0])
		},
		sys.AMD64, sys.ARM, sys.ARM64, sys.I386, sys.MIPS, sys.Loong64, sys.PPC64, sys.S390X, sys.Wasm)

	if cfg.goriscv64 >= 22 {
		addF("math/bits", "TrailingZeros64",
			func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
				return s.newValue1(ssa.OpCtz64, types.Types[types.TINT], args[0])
			},
			sys.RISCV64)
		addF("math/bits", "TrailingZeros32",
			func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
				return s.newValue1(ssa.OpCtz32, types.Types[types.TINT], args[0])
			},
			sys.RISCV64)
		addF("math/bits", "TrailingZeros16",
			func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
				return s.newValue1(ssa.OpCtz16, types.Types[types.TINT], args[0])
			},
			sys.RISCV64)
		addF("math/bits", "TrailingZeros8",
			func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
				return s.newValue1(ssa.OpCtz8, types.Types[types.TINT], args[0])
			},
			sys.RISCV64)
	}

	// ReverseBytes inlines correctly, no need to intrinsify it.
	alias("math/bits", "ReverseBytes64", "internal/runtime/sys", "Bswap64", all...)
	alias("math/bits", "ReverseBytes32", "internal/runtime/sys", "Bswap32", all...)
	// Nothing special is needed for targets where ReverseBytes16 lowers to a rotate
	addF("math/bits", "ReverseBytes16",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBswap16, types.Types[types.TUINT16], args[0])
		},
		sys.Loong64)
	if cfg.goppc64 >= 10 {
		// On Power10, 16-bit rotate is not available so use BRH instruction
		addF("math/bits", "ReverseBytes16",
			func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
				return s.newValue1(ssa.OpBswap16, types.Types[types.TUINT], args[0])
			},
			sys.PPC64)
	}
	if cfg.goriscv64 >= 22 {
		addF("math/bits", "ReverseBytes16",
			func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
				return s.newValue1(ssa.OpBswap16, types.Types[types.TUINT16], args[0])
			},
			sys.RISCV64)
	}

	addF("math/bits", "Len64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBitLen64, types.Types[types.TINT], args[0])
		},
		sys.AMD64, sys.ARM, sys.ARM64, sys.Loong64, sys.MIPS, sys.PPC64, sys.S390X, sys.Wasm)
	addF("math/bits", "Len32",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBitLen32, types.Types[types.TINT], args[0])
		},
		sys.AMD64, sys.ARM, sys.ARM64, sys.Loong64, sys.MIPS, sys.PPC64, sys.S390X, sys.Wasm)
	addF("math/bits", "Len16",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBitLen16, types.Types[types.TINT], args[0])
		},
		sys.AMD64, sys.ARM, sys.ARM64, sys.Loong64, sys.MIPS, sys.PPC64, sys.S390X, sys.Wasm)
	addF("math/bits", "Len8",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBitLen8, types.Types[types.TINT], args[0])
		},
		sys.AMD64, sys.ARM, sys.ARM64, sys.Loong64, sys.MIPS, sys.PPC64, sys.S390X, sys.Wasm)

	if cfg.goriscv64 >= 22 {
		addF("math/bits", "Len64",
			func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
				return s.newValue1(ssa.OpBitLen64, types.Types[types.TINT], args[0])
			},
			sys.RISCV64)
		addF("math/bits", "Len32",
			func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
				return s.newValue1(ssa.OpBitLen32, types.Types[types.TINT], args[0])
			},
			sys.RISCV64)
		addF("math/bits", "Len16",
			func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
				return s.newValue1(ssa.OpBitLen16, types.Types[types.TINT], args[0])
			},
			sys.RISCV64)
		addF("math/bits", "Len8",
			func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
				return s.newValue1(ssa.OpBitLen8, types.Types[types.TINT], args[0])
			},
			sys.RISCV64)
	}

	alias("math/bits", "Len", "math/bits", "Len64", p8...)
	alias("math/bits", "Len", "math/bits", "Len32", p4...)

	// LeadingZeros is handled because it trivially calls Len.
	addF("math/bits", "Reverse64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBitRev64, types.Types[types.TINT], args[0])
		},
		sys.ARM64, sys.Loong64)
	addF("math/bits", "Reverse32",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBitRev32, types.Types[types.TINT], args[0])
		},
		sys.ARM64, sys.Loong64)
	addF("math/bits", "Reverse16",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBitRev16, types.Types[types.TINT], args[0])
		},
		sys.ARM64, sys.Loong64)
	addF("math/bits", "Reverse8",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBitRev8, types.Types[types.TINT], args[0])
		},
		sys.ARM64, sys.Loong64)
	addF("math/bits", "Reverse",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBitRev64, types.Types[types.TINT], args[0])
		},
		sys.ARM64, sys.Loong64)
	addF("math/bits", "RotateLeft8",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue2(ssa.OpRotateLeft8, types.Types[types.TUINT8], args[0], args[1])
		},
		sys.AMD64, sys.RISCV64)
	addF("math/bits", "RotateLeft16",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue2(ssa.OpRotateLeft16, types.Types[types.TUINT16], args[0], args[1])
		},
		sys.AMD64, sys.RISCV64)
	addF("math/bits", "RotateLeft32",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue2(ssa.OpRotateLeft32, types.Types[types.TUINT32], args[0], args[1])
		},
		sys.AMD64, sys.ARM, sys.ARM64, sys.Loong64, sys.PPC64, sys.RISCV64, sys.S390X, sys.Wasm)
	addF("math/bits", "RotateLeft64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue2(ssa.OpRotateLeft64, types.Types[types.TUINT64], args[0], args[1])
		},
		sys.AMD64, sys.ARM64, sys.Loong64, sys.PPC64, sys.RISCV64, sys.S390X, sys.Wasm)
	alias("math/bits", "RotateLeft", "math/bits", "RotateLeft64", p8...)

	makeOnesCountAMD64 := func(op ssa.Op) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			if cfg.goamd64 >= 2 {
				return s.newValue1(op, types.Types[types.TINT], args[0])
			}

			v := s.entryNewValue0A(ssa.OpHasCPUFeature, types.Types[types.TBOOL], ir.Syms.X86HasPOPCNT)
			b := s.endBlock()
			b.Kind = ssa.BlockIf
			b.SetControl(v)
			bTrue := s.f.NewBlock(ssa.BlockPlain)
			bFalse := s.f.NewBlock(ssa.BlockPlain)
			bEnd := s.f.NewBlock(ssa.BlockPlain)
			b.AddEdgeTo(bTrue)
			b.AddEdgeTo(bFalse)
			b.Likely = ssa.BranchLikely // most machines have popcnt nowadays

			// We have the intrinsic - use it directly.
			s.startBlock(bTrue)
			s.vars[n] = s.newValue1(op, types.Types[types.TINT], args[0])
			s.endBlock().AddEdgeTo(bEnd)

			// Call the pure Go version.
			s.startBlock(bFalse)
			s.vars[n] = s.callResult(n, callNormal) // types.Types[TINT]
			s.endBlock().AddEdgeTo(bEnd)

			// Merge results.
			s.startBlock(bEnd)
			return s.variable(n, types.Types[types.TINT])
		}
	}

	makeOnesCountLoong64 := func(op ssa.Op) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			addr := s.entryNewValue1A(ssa.OpAddr, types.Types[types.TBOOL].PtrTo(), ir.Syms.Loong64HasLSX, s.sb)
			v := s.load(types.Types[types.TBOOL], addr)
			b := s.endBlock()
			b.Kind = ssa.BlockIf
			b.SetControl(v)
			bTrue := s.f.NewBlock(ssa.BlockPlain)
			bFalse := s.f.NewBlock(ssa.BlockPlain)
			bEnd := s.f.NewBlock(ssa.BlockPlain)
			b.AddEdgeTo(bTrue)
			b.AddEdgeTo(bFalse)
			b.Likely = ssa.BranchLikely // most loong64 machines support the LSX

			// We have the intrinsic - use it directly.
			s.startBlock(bTrue)
			s.vars[n] = s.newValue1(op, types.Types[types.TINT], args[0])
			s.endBlock().AddEdgeTo(bEnd)

			// Call the pure Go version.
			s.startBlock(bFalse)
			s.vars[n] = s.callResult(n, callNormal) // types.Types[TINT]
			s.endBlock().AddEdgeTo(bEnd)

			// Merge results.
			s.startBlock(bEnd)
			return s.variable(n, types.Types[types.TINT])
		}
	}

	makeOnesCountRISCV64 := func(op ssa.Op) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			if cfg.goriscv64 >= 22 {
				return s.newValue1(op, types.Types[types.TINT], args[0])
			}

			addr := s.entryNewValue1A(ssa.OpAddr, types.Types[types.TBOOL].PtrTo(), ir.Syms.RISCV64HasZbb, s.sb)
			v := s.load(types.Types[types.TBOOL], addr)
			b := s.endBlock()
			b.Kind = ssa.BlockIf
			b.SetControl(v)
			bTrue := s.f.NewBlock(ssa.BlockPlain)
			bFalse := s.f.NewBlock(ssa.BlockPlain)
			bEnd := s.f.NewBlock(ssa.BlockPlain)
			b.AddEdgeTo(bTrue)
			b.AddEdgeTo(bFalse)
			b.Likely = ssa.BranchLikely // Majority of RISC-V support Zbb.

			// We have the intrinsic - use it directly.
			s.startBlock(bTrue)
			s.vars[n] = s.newValue1(op, types.Types[types.TINT], args[0])
			s.endBlock().AddEdgeTo(bEnd)

			// Call the pure Go version.
			s.startBlock(bFalse)
			s.vars[n] = s.callResult(n, callNormal) // types.Types[TINT]
			s.endBlock().AddEdgeTo(bEnd)

			// Merge results.
			s.startBlock(bEnd)
			return s.variable(n, types.Types[types.TINT])
		}
	}

	addF("math/bits", "OnesCount64",
		makeOnesCountAMD64(ssa.OpPopCount64),
		sys.AMD64)
	addF("math/bits", "OnesCount64",
		makeOnesCountLoong64(ssa.OpPopCount64),
		sys.Loong64)
	addF("math/bits", "OnesCount64",
		makeOnesCountRISCV64(ssa.OpPopCount64),
		sys.RISCV64)
	addF("math/bits", "OnesCount64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpPopCount64, types.Types[types.TINT], args[0])
		},
		sys.PPC64, sys.ARM64, sys.S390X, sys.Wasm)
	addF("math/bits", "OnesCount32",
		makeOnesCountAMD64(ssa.OpPopCount32),
		sys.AMD64)
	addF("math/bits", "OnesCount32",
		makeOnesCountLoong64(ssa.OpPopCount32),
		sys.Loong64)
	addF("math/bits", "OnesCount32",
		makeOnesCountRISCV64(ssa.OpPopCount32),
		sys.RISCV64)
	addF("math/bits", "OnesCount32",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpPopCount32, types.Types[types.TINT], args[0])
		},
		sys.PPC64, sys.ARM64, sys.S390X, sys.Wasm)
	addF("math/bits", "OnesCount16",
		makeOnesCountAMD64(ssa.OpPopCount16),
		sys.AMD64)
	addF("math/bits", "OnesCount16",
		makeOnesCountLoong64(ssa.OpPopCount16),
		sys.Loong64)
	addF("math/bits", "OnesCount16",
		makeOnesCountRISCV64(ssa.OpPopCount16),
		sys.RISCV64)
	addF("math/bits", "OnesCount16",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpPopCount16, types.Types[types.TINT], args[0])
		},
		sys.ARM64, sys.S390X, sys.PPC64, sys.Wasm)
	addF("math/bits", "OnesCount8",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpPopCount8, types.Types[types.TINT], args[0])
		},
		sys.S390X, sys.PPC64, sys.Wasm)

	if cfg.goriscv64 >= 22 {
		addF("math/bits", "OnesCount8",
			makeOnesCountRISCV64(ssa.OpPopCount8),
			sys.RISCV64)
	}

	alias("math/bits", "OnesCount", "math/bits", "OnesCount64", p8...)

	add("math/bits", "Mul64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue2(ssa.OpMul64uhilo, types.NewTuple(types.Types[types.TUINT64], types.Types[types.TUINT64]), args[0], args[1])
		},
		all...)
	alias("math/bits", "Mul", "math/bits", "Mul64", p8...)
	alias("internal/runtime/math", "Mul64", "math/bits", "Mul64", p8...)
	addF("math/bits", "Add64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue3(ssa.OpAdd64carry, types.NewTuple(types.Types[types.TUINT64], types.Types[types.TUINT64]), args[0], args[1], args[2])
		},
		sys.AMD64, sys.ARM64, sys.PPC64, sys.S390X, sys.RISCV64, sys.Loong64, sys.MIPS64)
	alias("math/bits", "Add", "math/bits", "Add64", p8...)
	alias("internal/runtime/math", "Add64", "math/bits", "Add64", all...)
	addF("math/bits", "Sub64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue3(ssa.OpSub64borrow, types.NewTuple(types.Types[types.TUINT64], types.Types[types.TUINT64]), args[0], args[1], args[2])
		},
		sys.AMD64, sys.ARM64, sys.PPC64, sys.S390X, sys.RISCV64, sys.Loong64, sys.MIPS64)
	alias("math/bits", "Sub", "math/bits", "Sub64", p8...)
	addF("math/bits", "Div64",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			// check for divide-by-zero/overflow and panic with appropriate message
			cmpZero := s.newValue2(s.ssaOp(ir.ONE, types.Types[types.TUINT64]), types.Types[types.TBOOL], args[2], s.zeroVal(types.Types[types.TUINT64]))
			s.check(cmpZero, ir.Syms.Panicdivide)
			cmpOverflow := s.newValue2(s.ssaOp(ir.OLT, types.Types[types.TUINT64]), types.Types[types.TBOOL], args[0], args[2])
			s.check(cmpOverflow, ir.Syms.Panicoverflow)
			return s.newValue3(ssa.OpDiv128u, types.NewTuple(types.Types[types.TUINT64], types.Types[types.TUINT64]), args[0], args[1], args[2])
		},
		sys.AMD64)
	alias("math/bits", "Div", "math/bits", "Div64", sys.ArchAMD64)

	alias("internal/runtime/sys", "TrailingZeros8", "math/bits", "TrailingZeros8", all...)
	alias("internal/runtime/sys", "TrailingZeros32", "math/bits", "TrailingZeros32", all...)
	alias("internal/runtime/sys", "TrailingZeros64", "math/bits", "TrailingZeros64", all...)
	alias("internal/runtime/sys", "Len8", "math/bits", "Len8", all...)
	alias("internal/runtime/sys", "Len64", "math/bits", "Len64", all...)
	alias("internal/runtime/sys", "OnesCount64", "math/bits", "OnesCount64", all...)

	/******** sync/atomic ********/

	// Note: these are disabled by flag_race in findIntrinsic below.
	alias("sync/atomic", "LoadInt32", "internal/runtime/atomic", "Load", all...)
	alias("sync/atomic", "LoadInt64", "internal/runtime/atomic", "Load64", all...)
	alias("sync/atomic", "LoadPointer", "internal/runtime/atomic", "Loadp", all...)
	alias("sync/atomic", "LoadUint32", "internal/runtime/atomic", "Load", all...)
	alias("sync/atomic", "LoadUint64", "internal/runtime/atomic", "Load64", all...)
	alias("sync/atomic", "LoadUintptr", "internal/runtime/atomic", "Load", p4...)
	alias("sync/atomic", "LoadUintptr", "internal/runtime/atomic", "Load64", p8...)

	alias("sync/atomic", "StoreInt32", "internal/runtime/atomic", "Store", all...)
	alias("sync/atomic", "StoreInt64", "internal/runtime/atomic", "Store64", all...)
	// Note: not StorePointer, that needs a write barrier.  Same below for {CompareAnd}Swap.
	alias("sync/atomic", "StoreUint32", "internal/runtime/atomic", "Store", all...)
	alias("sync/atomic", "StoreUint64", "internal/runtime/atomic", "Store64", all...)
	alias("sync/atomic", "StoreUintptr", "internal/runtime/atomic", "Store", p4...)
	alias("sync/atomic", "StoreUintptr", "internal/runtime/atomic", "Store64", p8...)

	alias("sync/atomic", "SwapInt32", "internal/runtime/atomic", "Xchg", all...)
	alias("sync/atomic", "SwapInt64", "internal/runtime/atomic", "Xchg64", all...)
	alias("sync/atomic", "SwapUint32", "internal/runtime/atomic", "Xchg", all...)
	alias("sync/atomic", "SwapUint64", "internal/runtime/atomic", "Xchg64", all...)
	alias("sync/atomic", "SwapUintptr", "internal/runtime/atomic", "Xchg", p4...)
	alias("sync/atomic", "SwapUintptr", "internal/runtime/atomic", "Xchg64", p8...)

	alias("sync/atomic", "CompareAndSwapInt32", "internal/runtime/atomic", "Cas", all...)
	alias("sync/atomic", "CompareAndSwapInt64", "internal/runtime/atomic", "Cas64", all...)
	alias("sync/atomic", "CompareAndSwapUint32", "internal/runtime/atomic", "Cas", all...)
	alias("sync/atomic", "CompareAndSwapUint64", "internal/runtime/atomic", "Cas64", all...)
	alias("sync/atomic", "CompareAndSwapUintptr", "internal/runtime/atomic", "Cas", p4...)
	alias("sync/atomic", "CompareAndSwapUintptr", "internal/runtime/atomic", "Cas64", p8...)

	alias("sync/atomic", "AddInt32", "internal/runtime/atomic", "Xadd", all...)
	alias("sync/atomic", "AddInt64", "internal/runtime/atomic", "Xadd64", all...)
	alias("sync/atomic", "AddUint32", "internal/runtime/atomic", "Xadd", all...)
	alias("sync/atomic", "AddUint64", "internal/runtime/atomic", "Xadd64", all...)
	alias("sync/atomic", "AddUintptr", "internal/runtime/atomic", "Xadd", p4...)
	alias("sync/atomic", "AddUintptr", "internal/runtime/atomic", "Xadd64", p8...)

	alias("sync/atomic", "AndInt32", "internal/runtime/atomic", "And32", sys.ArchARM64, sys.ArchAMD64, sys.ArchLoong64)
	alias("sync/atomic", "AndUint32", "internal/runtime/atomic", "And32", sys.ArchARM64, sys.ArchAMD64, sys.ArchLoong64)
	alias("sync/atomic", "AndInt64", "internal/runtime/atomic", "And64", sys.ArchARM64, sys.ArchAMD64, sys.ArchLoong64)
	alias("sync/atomic", "AndUint64", "internal/runtime/atomic", "And64", sys.ArchARM64, sys.ArchAMD64, sys.ArchLoong64)
	alias("sync/atomic", "AndUintptr", "internal/runtime/atomic", "And64", sys.ArchARM64, sys.ArchAMD64, sys.ArchLoong64)
	alias("sync/atomic", "OrInt32", "internal/runtime/atomic", "Or32", sys.ArchARM64, sys.ArchAMD64, sys.ArchLoong64)
	alias("sync/atomic", "OrUint32", "internal/runtime/atomic", "Or32", sys.ArchARM64, sys.ArchAMD64, sys.ArchLoong64)
	alias("sync/atomic", "OrInt64", "internal/runtime/atomic", "Or64", sys.ArchARM64, sys.ArchAMD64, sys.ArchLoong64)
	alias("sync/atomic", "OrUint64", "internal/runtime/atomic", "Or64", sys.ArchARM64, sys.ArchAMD64, sys.ArchLoong64)
	alias("sync/atomic", "OrUintptr", "internal/runtime/atomic", "Or64", sys.ArchARM64, sys.ArchAMD64, sys.ArchLoong64)

	/******** math/big ********/
	alias("math/big", "mulWW", "math/bits", "Mul64", p8...)

	/******** internal/runtime/maps ********/

	// Important: The intrinsic implementations below return a packed
	// bitset, while the portable Go implementation uses an unpacked
	// representation (one bit set in each byte).
	//
	// Thus we must replace most bitset methods with implementations that
	// work with the packed representation.
	//
	// TODO(prattmic): The bitset implementations don't use SIMD, so they
	// could be handled with build tags (though that would break
	// -d=ssa/intrinsics/off=1).

	// With a packed representation we no longer need to shift the result
	// of TrailingZeros64.
	alias("internal/runtime/maps", "bitsetFirst", "internal/runtime/sys", "TrailingZeros64", sys.ArchAMD64)

	addF("internal/runtime/maps", "bitsetRemoveBelow",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			b := args[0]
			i := args[1]

			// Clear the lower i bits in b.
			//
			// out = b &^ ((1 << i) - 1)

			one := s.constInt64(types.Types[types.TUINT64], 1)

			mask := s.newValue2(ssa.OpLsh8x8, types.Types[types.TUINT64], one, i)
			mask = s.newValue2(ssa.OpSub64, types.Types[types.TUINT64], mask, one)
			mask = s.newValue1(ssa.OpCom64, types.Types[types.TUINT64], mask)

			return s.newValue2(ssa.OpAnd64, types.Types[types.TUINT64], b, mask)
		},
		sys.AMD64)

	addF("internal/runtime/maps", "bitsetLowestSet",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			b := args[0]

			// Test the lowest bit in b.
			//
			// out = (b & 1) == 1

			one := s.constInt64(types.Types[types.TUINT64], 1)
			and := s.newValue2(ssa.OpAnd64, types.Types[types.TUINT64], b, one)
			return s.newValue2(ssa.OpEq64, types.Types[types.TBOOL], and, one)
		},
		sys.AMD64)

	addF("internal/runtime/maps", "bitsetShiftOutLowest",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			b := args[0]

			// Right shift out the lowest bit in b.
			//
			// out = b >> 1

			one := s.constInt64(types.Types[types.TUINT64], 1)
			return s.newValue2(ssa.OpRsh64Ux64, types.Types[types.TUINT64], b, one)
		},
		sys.AMD64)

	addF("internal/runtime/maps", "ctrlGroupMatchH2",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			g := args[0]
			h := args[1]

			// Explicit copies to fp registers. See
			// https://go.dev/issue/70451.
			gfp := s.newValue1(ssa.OpAMD64MOVQi2f, types.TypeInt128, g)
			hfp := s.newValue1(ssa.OpAMD64MOVQi2f, types.TypeInt128, h)

			// Broadcast h2 into each byte of a word.
			var broadcast *ssa.Value
			if buildcfg.GOAMD64 >= 4 {
				// VPBROADCASTB saves 1 instruction vs PSHUFB
				// because the input can come from a GP
				// register, while PSHUFB requires moving into
				// an FP register first.
				//
				// Nominally PSHUFB would require a second
				// additional instruction to load the control
				// mask into a FP register. But broadcast uses
				// a control mask of 0, and the register ABI
				// already defines X15 as a zero register.
				broadcast = s.newValue1(ssa.OpAMD64VPBROADCASTB, types.TypeInt128, h) // use gp copy of h
			} else if buildcfg.GOAMD64 >= 2 {
				// PSHUFB performs a byte broadcast when given
				// a control input of 0.
				broadcast = s.newValue1(ssa.OpAMD64PSHUFBbroadcast, types.TypeInt128, hfp)
			} else {
				// No direct byte broadcast. First we must
				// duplicate the lower byte and then do a
				// 16-bit broadcast.

				// "Unpack" h2 with itself. This duplicates the
				// input, resulting in h2 in the lower two
				// bytes.
				unpack := s.newValue2(ssa.OpAMD64PUNPCKLBW, types.TypeInt128, hfp, hfp)

				// Copy the lower 16-bits of unpack into every
				// 16-bit slot in the lower 64-bits of the
				// output register. Note that immediate 0
				// selects the low word as the source for every
				// destination slot.
				broadcast = s.newValue1I(ssa.OpAMD64PSHUFLW, types.TypeInt128, 0, unpack)

				// No need to broadcast into the upper 64-bits,
				// as we don't use those.
			}

			// Compare each byte of the control word with h2. Each
			// matching byte has every bit set.
			eq := s.newValue2(ssa.OpAMD64PCMPEQB, types.TypeInt128, broadcast, gfp)

			// Construct a "byte mask": each output bit is equal to
			// the sign bit each input byte.
			//
			// This results in a packed output (bit N set means
			// byte N matched).
			//
			// NOTE: See comment above on bitsetFirst.
			out := s.newValue1(ssa.OpAMD64PMOVMSKB, types.Types[types.TUINT16], eq)

			// g is only 64-bits so the upper 64-bits of the
			// 128-bit register will be zero. If h2 is also zero,
			// then we'll get matches on those bytes. Truncate the
			// upper bits to ignore such matches.
			ret := s.newValue1(ssa.OpZeroExt8to64, types.Types[types.TUINT64], out)

			return ret
		},
		sys.AMD64)

	addF("internal/runtime/maps", "ctrlGroupMatchEmpty",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			// An empty slot is   1000 0000
			// A deleted slot is  1111 1110
			// A full slot is     0??? ????

			g := args[0]

			// Explicit copy to fp register. See
			// https://go.dev/issue/70451.
			gfp := s.newValue1(ssa.OpAMD64MOVQi2f, types.TypeInt128, g)

			if buildcfg.GOAMD64 >= 2 {
				// "PSIGNB negates each data element of the
				// destination operand (the first operand) if
				// the signed integer value of the
				// corresponding data element in the source
				// operand (the second operand) is less than
				// zero. If the signed integer value of a data
				// element in the source operand is positive,
				// the corresponding data element in the
				// destination operand is unchanged. If a data
				// element in the source operand is zero, the
				// corresponding data element in the
				// destination operand is set to zero" - Intel SDM
				//
				// If we pass the group control word as both
				// arguments:
				// - Full slots are unchanged.
				// - Deleted slots are negated, becoming
				//   0000 0010.
				// - Empty slots are negated, becoming
				//   1000 0000 (unchanged!).
				//
				// The result is that only empty slots have the
				// sign bit set. We then use PMOVMSKB to
				// extract the sign bits.
				sign := s.newValue2(ssa.OpAMD64PSIGNB, types.TypeInt128, gfp, gfp)

				// Construct a "byte mask": each output bit is
				// equal to the sign bit each input byte. The
				// sign bit is only set for empty or deleted
				// slots.
				//
				// This results in a packed output (bit N set
				// means byte N matched).
				//
				// NOTE: See comment above on bitsetFirst.
				ret := s.newValue1(ssa.OpAMD64PMOVMSKB, types.Types[types.TUINT16], sign)

				// g is only 64-bits so the upper 64-bits of
				// the 128-bit register will be zero. PSIGNB
				// will keep all of these bytes zero, so no
				// need to truncate.

				return ret
			}

			// No PSIGNB, simply do byte equality with ctrlEmpty.

			// Load ctrlEmpty into each byte of a control word.
			var ctrlsEmpty uint64 = abi.MapCtrlEmpty
			e := s.constInt64(types.Types[types.TUINT64], int64(ctrlsEmpty))
			// Explicit copy to fp register. See
			// https://go.dev/issue/70451.
			efp := s.newValue1(ssa.OpAMD64MOVQi2f, types.TypeInt128, e)

			// Compare each byte of the control word with ctrlEmpty. Each
			// matching byte has every bit set.
			eq := s.newValue2(ssa.OpAMD64PCMPEQB, types.TypeInt128, efp, gfp)

			// Construct a "byte mask": each output bit is equal to
			// the sign bit each input byte.
			//
			// This results in a packed output (bit N set means
			// byte N matched).
			//
			// NOTE: See comment above on bitsetFirst.
			out := s.newValue1(ssa.OpAMD64PMOVMSKB, types.Types[types.TUINT16], eq)

			// g is only 64-bits so the upper 64-bits of the
			// 128-bit register will be zero. The upper 64-bits of
			// efp are also zero, so we'll get matches on those
			// bytes. Truncate the upper bits to ignore such
			// matches.
			return s.newValue1(ssa.OpZeroExt8to64, types.Types[types.TUINT64], out)
		},
		sys.AMD64)

	addF("internal/runtime/maps", "ctrlGroupMatchEmptyOrDeleted",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			// An empty slot is   1000 0000
			// A deleted slot is  1111 1110
			// A full slot is     0??? ????
			//
			// A slot is empty or deleted iff bit 7 (sign bit) is
			// set.

			g := args[0]

			// Explicit copy to fp register. See
			// https://go.dev/issue/70451.
			gfp := s.newValue1(ssa.OpAMD64MOVQi2f, types.TypeInt128, g)

			// Construct a "byte mask": each output bit is equal to
			// the sign bit each input byte. The sign bit is only
			// set for empty or deleted slots.
			//
			// This results in a packed output (bit N set means
			// byte N matched).
			//
			// NOTE: See comment above on bitsetFirst.
			ret := s.newValue1(ssa.OpAMD64PMOVMSKB, types.Types[types.TUINT16], gfp)

			// g is only 64-bits so the upper 64-bits of the
			// 128-bit register will be zero. Zero will never match
			// ctrlEmpty or ctrlDeleted, so no need to truncate.

			return ret
		},
		sys.AMD64)

	addF("internal/runtime/maps", "ctrlGroupMatchFull",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			// An empty slot is   1000 0000
			// A deleted slot is  1111 1110
			// A full slot is     0??? ????
			//
			// A slot is full iff bit 7 (sign bit) is unset.

			g := args[0]

			// Explicit copy to fp register. See
			// https://go.dev/issue/70451.
			gfp := s.newValue1(ssa.OpAMD64MOVQi2f, types.TypeInt128, g)

			// Construct a "byte mask": each output bit is equal to
			// the sign bit each input byte. The sign bit is only
			// set for empty or deleted slots.
			//
			// This results in a packed output (bit N set means
			// byte N matched).
			//
			// NOTE: See comment above on bitsetFirst.
			mask := s.newValue1(ssa.OpAMD64PMOVMSKB, types.Types[types.TUINT16], gfp)

			// Invert the mask to set the bits for the full slots.
			out := s.newValue1(ssa.OpCom16, types.Types[types.TUINT16], mask)

			// g is only 64-bits so the upper 64-bits of the
			// 128-bit register will be zero, with bit 7 unset.
			// Truncate the upper bits to ignore these.
			return s.newValue1(ssa.OpZeroExt8to64, types.Types[types.TUINT64], out)
		},
		sys.AMD64)

	/******** crypto/internal/constanttime ********/
	// We implement a superset of the Select promise:
	// Select returns x if v != 0 and y if v == 0.
	add("crypto/internal/constanttime", "Select",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			v, x, y := args[0], args[1], args[2]

			var checkOp ssa.Op
			var zero *ssa.Value
			switch s.config.PtrSize {
			case 8:
				checkOp = ssa.OpNeq64
				zero = s.constInt64(types.Types[types.TINT], 0)
			case 4:
				checkOp = ssa.OpNeq32
				zero = s.constInt32(types.Types[types.TINT], 0)
			default:
				panic("unreachable")
			}
			check := s.newValue2(checkOp, types.Types[types.TBOOL], zero, v)

			return s.newValue3(ssa.OpCondSelect, types.Types[types.TINT], x, y, check)
		},
		sys.ArchAMD64, sys.ArchARM64, sys.ArchLoong64, sys.ArchPPC64, sys.ArchPPC64LE, sys.ArchWasm) // all with CMOV support.
	add("crypto/internal/constanttime", "boolToUint8",
		func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpCvtBoolToUint8, types.Types[types.TUINT8], args[0])
		},
		all...)

	if buildcfg.Experiment.SIMD {
		// Only enable intrinsics, if SIMD experiment.
		simdIntrinsics(addF)

		addF("simd", "ClearAVXUpperBits",
			func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
				s.vars[memVar] = s.newValue1(ssa.OpAMD64VZEROUPPER, types.TypeMem, s.mem())
				return nil
			},
			sys.AMD64)

		addF(simdPackage, "Int8x16.IsZero", opLen1(ssa.OpIsZeroVec, types.Types[types.TBOOL]), sys.AMD64)
		addF(simdPackage, "Int16x8.IsZero", opLen1(ssa.OpIsZeroVec, types.Types[types.TBOOL]), sys.AMD64)
		addF(simdPackage, "Int32x4.IsZero", opLen1(ssa.OpIsZeroVec, types.Types[types.TBOOL]), sys.AMD64)
		addF(simdPackage, "Int64x2.IsZero", opLen1(ssa.OpIsZeroVec, types.Types[types.TBOOL]), sys.AMD64)
		addF(simdPackage, "Uint8x16.IsZero", opLen1(ssa.OpIsZeroVec, types.Types[types.TBOOL]), sys.AMD64)
		addF(simdPackage, "Uint16x8.IsZero", opLen1(ssa.OpIsZeroVec, types.Types[types.TBOOL]), sys.AMD64)
		addF(simdPackage, "Uint32x4.IsZero", opLen1(ssa.OpIsZeroVec, types.Types[types.TBOOL]), sys.AMD64)
		addF(simdPackage, "Uint64x2.IsZero", opLen1(ssa.OpIsZeroVec, types.Types[types.TBOOL]), sys.AMD64)
		addF(simdPackage, "Int8x32.IsZero", opLen1(ssa.OpIsZeroVec, types.Types[types.TBOOL]), sys.AMD64)
		addF(simdPackage, "Int16x16.IsZero", opLen1(ssa.OpIsZeroVec, types.Types[types.TBOOL]), sys.AMD64)
		addF(simdPackage, "Int32x8.IsZero", opLen1(ssa.OpIsZeroVec, types.Types[types.TBOOL]), sys.AMD64)
		addF(simdPackage, "Int64x4.IsZero", opLen1(ssa.OpIsZeroVec, types.Types[types.TBOOL]), sys.AMD64)
		addF(simdPackage, "Uint8x32.IsZero", opLen1(ssa.OpIsZeroVec, types.Types[types.TBOOL]), sys.AMD64)
		addF(simdPackage, "Uint16x16.IsZero", opLen1(ssa.OpIsZeroVec, types.Types[types.TBOOL]), sys.AMD64)
		addF(simdPackage, "Uint32x8.IsZero", opLen1(ssa.OpIsZeroVec, types.Types[types.TBOOL]), sys.AMD64)
		addF(simdPackage, "Uint64x4.IsZero", opLen1(ssa.OpIsZeroVec, types.Types[types.TBOOL]), sys.AMD64)

		sfp4 := func(method string, hwop ssa.Op, vectype *types.Type) {
			addF("simd", method,
				func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
					x, a, b, c, d, y := args[0], args[1], args[2], args[3], args[4], args[5]
					if a.Op == ssa.OpConst8 && b.Op == ssa.OpConst8 && c.Op == ssa.OpConst8 && d.Op == ssa.OpConst8 {
						return select4FromPair(x, a, b, c, d, y, s, hwop, vectype)
					} else {
						return s.callResult(n, callNormal)
					}
				},
				sys.AMD64)
		}

		sfp4("Int32x4.SelectFromPair", ssa.OpconcatSelectedConstantInt32x4, types.TypeVec128)
		sfp4("Uint32x4.SelectFromPair", ssa.OpconcatSelectedConstantUint32x4, types.TypeVec128)
		sfp4("Float32x4.SelectFromPair", ssa.OpconcatSelectedConstantFloat32x4, types.TypeVec128)

		sfp4("Int32x8.SelectFromPairGrouped", ssa.OpconcatSelectedConstantGroupedInt32x8, types.TypeVec256)
		sfp4("Uint32x8.SelectFromPairGrouped", ssa.OpconcatSelectedConstantGroupedUint32x8, types.TypeVec256)
		sfp4("Float32x8.SelectFromPairGrouped", ssa.OpconcatSelectedConstantGroupedFloat32x8, types.TypeVec256)

		sfp4("Int32x16.SelectFromPairGrouped", ssa.OpconcatSelectedConstantGroupedInt32x16, types.TypeVec512)
		sfp4("Uint32x16.SelectFromPairGrouped", ssa.OpconcatSelectedConstantGroupedUint32x16, types.TypeVec512)
		sfp4("Float32x16.SelectFromPairGrouped", ssa.OpconcatSelectedConstantGroupedFloat32x16, types.TypeVec512)

		sfp2 := func(method string, hwop ssa.Op, vectype *types.Type, cscimm func(i, j uint8) int64) {
			addF("simd", method,
				func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
					x, a, b, y := args[0], args[1], args[2], args[3]
					if a.Op == ssa.OpConst8 && b.Op == ssa.OpConst8 {
						return select2FromPair(x, a, b, y, s, hwop, vectype, cscimm)
					} else {
						return s.callResult(n, callNormal)
					}
				},
				sys.AMD64)
		}

		sfp2("Uint64x2.SelectFromPair", ssa.OpconcatSelectedConstantUint64x2, types.TypeVec128, cscimm2)
		sfp2("Int64x2.SelectFromPair", ssa.OpconcatSelectedConstantInt64x2, types.TypeVec128, cscimm2)
		sfp2("Float64x2.SelectFromPair", ssa.OpconcatSelectedConstantFloat64x2, types.TypeVec128, cscimm2)

		sfp2("Uint64x4.SelectFromPairGrouped", ssa.OpconcatSelectedConstantGroupedUint64x4, types.TypeVec256, cscimm2g2)
		sfp2("Int64x4.SelectFromPairGrouped", ssa.OpconcatSelectedConstantGroupedInt64x4, types.TypeVec256, cscimm2g2)
		sfp2("Float64x4.SelectFromPairGrouped", ssa.OpconcatSelectedConstantGroupedFloat64x4, types.TypeVec256, cscimm2g2)

		sfp2("Uint64x8.SelectFromPairGrouped", ssa.OpconcatSelectedConstantGroupedUint64x8, types.TypeVec512, cscimm2g4)
		sfp2("Int64x8.SelectFromPairGrouped", ssa.OpconcatSelectedConstantGroupedInt64x8, types.TypeVec512, cscimm2g4)
		sfp2("Float64x8.SelectFromPairGrouped", ssa.OpconcatSelectedConstantGroupedFloat64x8, types.TypeVec512, cscimm2g4)

	}
}

func cscimm4(a, b, c, d uint8) int64 {
	return se(a + b<<2 + c<<4 + d<<6)
}

func cscimm2(a, b uint8) int64 {
	return se(a + b<<1)
}

func cscimm2g2(a, b uint8) int64 {
	g := cscimm2(a, b)
	return int64(int8(g + g<<2))
}

func cscimm2g4(a, b uint8) int64 {
	g := cscimm2g2(a, b)
	return int64(int8(g + g<<4))
}

const (
	_LLLL = iota
	_HLLL
	_LHLL
	_HHLL
	_LLHL
	_HLHL
	_LHHL
	_HHHL
	_LLLH
	_HLLH
	_LHLH
	_HHLH
	_LLHH
	_HLHH
	_LHHH
	_HHHH
)

const (
	_LL = iota
	_HL
	_LH
	_HH
)

func select2FromPair(x, _a, _b, y *ssa.Value, s *state, op ssa.Op, t *types.Type, csc func(a, b uint8) int64) *ssa.Value {
	a, b := uint8(_a.AuxInt8()), uint8(_b.AuxInt8())
	pattern := (a&2)>>1 + (b & 2)
	a, b = a&1, b&1

	switch pattern {
	case _LL:
		return s.newValue2I(op, t, csc(a, b), x, x)
	case _HH:
		return s.newValue2I(op, t, csc(a, b), y, y)
	case _LH:
		return s.newValue2I(op, t, csc(a, b), x, y)
	case _HL:
		return s.newValue2I(op, t, csc(a, b), y, x)
	}
	panic("The preceding switch should have been exhaustive")
}

func select4FromPair(x, _a, _b, _c, _d, y *ssa.Value, s *state, op ssa.Op, t *types.Type) *ssa.Value {
	a, b, c, d := uint8(_a.AuxInt8()), uint8(_b.AuxInt8()), uint8(_c.AuxInt8()), uint8(_d.AuxInt8())
	pattern := a>>2 + (b&4)>>1 + (c & 4) + (d&4)<<1

	a, b, c, d = a&3, b&3, c&3, d&3

	switch pattern {
	case _LLLL:
		// TODO DETECT 0,1,2,3, 0,0,0,0
		return s.newValue2I(op, t, cscimm4(a, b, c, d), x, x)
	case _HHHH:
		// TODO DETECT 0,1,2,3, 0,0,0,0
		return s.newValue2I(op, t, cscimm4(a, b, c, d), y, y)
	case _LLHH:
		return s.newValue2I(op, t, cscimm4(a, b, c, d), x, y)
	case _HHLL:
		return s.newValue2I(op, t, cscimm4(a, b, c, d), y, x)

	case _HLLL:
		z := s.newValue2I(op, t, cscimm4(a, a, b, b), y, x)
		return s.newValue2I(op, t, cscimm4(0, 2, c, d), z, x)
	case _LHLL:
		z := s.newValue2I(op, t, cscimm4(a, a, b, b), x, y)
		return s.newValue2I(op, t, cscimm4(0, 2, c, d), z, x)
	case _HLHH:
		z := s.newValue2I(op, t, cscimm4(a, a, b, b), y, x)
		return s.newValue2I(op, t, cscimm4(0, 2, c, d), z, y)
	case _LHHH:
		z := s.newValue2I(op, t, cscimm4(a, a, b, b), x, y)
		return s.newValue2I(op, t, cscimm4(0, 2, c, d), z, y)

	case _LLLH:
		z := s.newValue2I(op, t, cscimm4(c, c, d, d), x, y)
		return s.newValue2I(op, t, cscimm4(a, b, 0, 2), x, z)
	case _LLHL:
		z := s.newValue2I(op, t, cscimm4(c, c, d, d), y, x)
		return s.newValue2I(op, t, cscimm4(a, b, 0, 2), x, z)

	case _HHLH:
		z := s.newValue2I(op, t, cscimm4(c, c, d, d), x, y)
		return s.newValue2I(op, t, cscimm4(a, b, 0, 2), y, z)

	case _HHHL:
		z := s.newValue2I(op, t, cscimm4(c, c, d, d), y, x)
		return s.newValue2I(op, t, cscimm4(a, b, 0, 2), y, z)

	case _LHLH:
		z := s.newValue2I(op, t, cscimm4(a, c, b, d), x, y)
		return s.newValue2I(op, t, se(0b11_01_10_00), z, z)
	case _HLHL:
		z := s.newValue2I(op, t, cscimm4(b, d, a, c), x, y)
		return s.newValue2I(op, t, se(0b01_11_00_10), z, z)
	case _HLLH:
		z := s.newValue2I(op, t, cscimm4(b, c, a, d), x, y)
		return s.newValue2I(op, t, se(0b11_01_00_10), z, z)
	case _LHHL:
		z := s.newValue2I(op, t, cscimm4(a, d, b, c), x, y)
		return s.newValue2I(op, t, se(0b01_11_10_00), z, z)
	}
	panic("The preceding switch should have been exhaustive")
}

// se smears the not-really-a-sign bit of a uint8 to conform to the conventions
// for representing AuxInt in ssa.
func se(x uint8) int64 {
	return int64(int8(x))
}

func opLen1(op ssa.Op, t *types.Type) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return s.newValue1(op, t, args[0])
	}
}

func opLen2(op ssa.Op, t *types.Type) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return s.newValue2(op, t, args[0], args[1])
	}
}

func opLen2_21(op ssa.Op, t *types.Type) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return s.newValue2(op, t, args[1], args[0])
	}
}

func opLen3(op ssa.Op, t *types.Type) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return s.newValue3(op, t, args[0], args[1], args[2])
	}
}

func opLen3_31(op ssa.Op, t *types.Type) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return s.newValue3(op, t, args[2], args[1], args[0])
	}
}

func opLen3_21(op ssa.Op, t *types.Type) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return s.newValue3(op, t, args[1], args[0], args[2])
	}
}

func opLen3_231(op ssa.Op, t *types.Type) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return s.newValue3(op, t, args[2], args[0], args[1])
	}
}

func opLen4(op ssa.Op, t *types.Type) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return s.newValue4(op, t, args[0], args[1], args[2], args[3])
	}
}

func opLen4_231(op ssa.Op, t *types.Type) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return s.newValue4(op, t, args[2], args[0], args[1], args[3])
	}
}

func opLen4_31(op ssa.Op, t *types.Type) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return s.newValue4(op, t, args[2], args[1], args[0], args[3])
	}
}

func immJumpTable(s *state, idx *ssa.Value, intrinsicCall *ir.CallExpr, genOp func(*state, int)) *ssa.Value {
	// Make blocks we'll need.
	bEnd := s.f.NewBlock(ssa.BlockPlain)

	if !idx.Type.IsKind(types.TUINT8) {
		panic("immJumpTable expects uint8 value")
	}

	// We will exhaust 0-255, so no need to check the bounds.
	t := types.Types[types.TUINTPTR]
	idx = s.conv(nil, idx, idx.Type, t)

	b := s.curBlock
	b.Kind = ssa.BlockJumpTable
	b.Pos = intrinsicCall.Pos()
	if base.Flag.Cfg.SpectreIndex {
		// Potential Spectre vulnerability hardening?
		idx = s.newValue2(ssa.OpSpectreSliceIndex, t, idx, s.uintptrConstant(255))
	}
	b.SetControl(idx)
	targets := [256]*ssa.Block{}
	for i := range 256 {
		t := s.f.NewBlock(ssa.BlockPlain)
		targets[i] = t
		b.AddEdgeTo(t)
	}
	s.endBlock()

	for i, t := range targets {
		s.startBlock(t)
		genOp(s, i)
		if t.Kind != ssa.BlockExit {
			t.AddEdgeTo(bEnd)
		}
		s.endBlock()
	}

	s.startBlock(bEnd)
	ret := s.variable(intrinsicCall, intrinsicCall.Type())
	return ret
}

func opLen1Imm8(op ssa.Op, t *types.Type, offset int) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		if args[1].Op == ssa.OpConst8 {
			return s.newValue1I(op, t, args[1].AuxInt<<int64(offset), args[0])
		}
		return immJumpTable(s, args[1], n, func(sNew *state, idx int) {
			// Encode as int8 due to requirement of AuxInt, check its comment for details.
			s.vars[n] = sNew.newValue1I(op, t, int64(int8(idx<<offset)), args[0])
		})
	}
}

func opLen2Imm8(op ssa.Op, t *types.Type, offset int) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		if args[1].Op == ssa.OpConst8 {
			return s.newValue2I(op, t, args[1].AuxInt<<int64(offset), args[0], args[2])
		}
		return immJumpTable(s, args[1], n, func(sNew *state, idx int) {
			// Encode as int8 due to requirement of AuxInt, check its comment for details.
			s.vars[n] = sNew.newValue2I(op, t, int64(int8(idx<<offset)), args[0], args[2])
		})
	}
}

func opLen3Imm8(op ssa.Op, t *types.Type, offset int) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		if args[1].Op == ssa.OpConst8 {
			return s.newValue3I(op, t, args[1].AuxInt<<int64(offset), args[0], args[2], args[3])
		}
		return immJumpTable(s, args[1], n, func(sNew *state, idx int) {
			// Encode as int8 due to requirement of AuxInt, check its comment for details.
			s.vars[n] = sNew.newValue3I(op, t, int64(int8(idx<<offset)), args[0], args[2], args[3])
		})
	}
}

func opLen2Imm8_2I(op ssa.Op, t *types.Type, offset int) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		if args[2].Op == ssa.OpConst8 {
			return s.newValue2I(op, t, args[2].AuxInt<<int64(offset), args[0], args[1])
		}
		return immJumpTable(s, args[2], n, func(sNew *state, idx int) {
			// Encode as int8 due to requirement of AuxInt, check its comment for details.
			s.vars[n] = sNew.newValue2I(op, t, int64(int8(idx<<offset)), args[0], args[1])
		})
	}
}

// Two immediates instead of just 1.  Offset is ignored, so it is a _ parameter instead.
func opLen2Imm8_II(op ssa.Op, t *types.Type, _ int) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		if args[1].Op == ssa.OpConst8 && args[2].Op == ssa.OpConst8 && args[1].AuxInt & ^3 == 0 && args[2].AuxInt & ^3 == 0 {
			i1, i2 := args[1].AuxInt, args[2].AuxInt
			return s.newValue2I(op, t, int64(int8(i1+i2<<4)), args[0], args[3])
		}
		four := s.constInt64(types.Types[types.TUINT8], 4)
		shifted := s.newValue2(ssa.OpLsh8x8, types.Types[types.TUINT8], args[2], four)
		combined := s.newValue2(ssa.OpAdd8, types.Types[types.TUINT8], args[1], shifted)
		return immJumpTable(s, combined, n, func(sNew *state, idx int) {
			// Encode as int8 due to requirement of AuxInt, check its comment for details.
			// TODO for "zeroing" values, panic instead.
			if idx & ^(3+3<<4) == 0 {
				s.vars[n] = sNew.newValue2I(op, t, int64(int8(idx)), args[0], args[3])
			} else {
				sNew.rtcall(ir.Syms.PanicSimdImm, false, nil)
			}
		})
	}
}

// The assembler requires the imm value of a SHA1RNDS4 instruction to be one of 0,1,2,3...
func opLen2Imm8_SHA1RNDS4(op ssa.Op, t *types.Type, offset int) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		if args[1].Op == ssa.OpConst8 {
			return s.newValue2I(op, t, (args[1].AuxInt<<int64(offset))&0b11, args[0], args[2])
		}
		return immJumpTable(s, args[1], n, func(sNew *state, idx int) {
			// Encode as int8 due to requirement of AuxInt, check its comment for details.
			s.vars[n] = sNew.newValue2I(op, t, int64(int8(idx<<offset))&0b11, args[0], args[2])
		})
	}
}

func opLen3Imm8_2I(op ssa.Op, t *types.Type, offset int) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		if args[2].Op == ssa.OpConst8 {
			return s.newValue3I(op, t, args[2].AuxInt<<int64(offset), args[0], args[1], args[3])
		}
		return immJumpTable(s, args[2], n, func(sNew *state, idx int) {
			// Encode as int8 due to requirement of AuxInt, check its comment for details.
			s.vars[n] = sNew.newValue3I(op, t, int64(int8(idx<<offset)), args[0], args[1], args[3])
		})
	}
}

func opLen4Imm8(op ssa.Op, t *types.Type, offset int) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		if args[1].Op == ssa.OpConst8 {
			return s.newValue4I(op, t, args[1].AuxInt<<int64(offset), args[0], args[2], args[3], args[4])
		}
		return immJumpTable(s, args[1], n, func(sNew *state, idx int) {
			// Encode as int8 due to requirement of AuxInt, check its comment for details.
			s.vars[n] = sNew.newValue4I(op, t, int64(int8(idx<<offset)), args[0], args[2], args[3], args[4])
		})
	}
}

func simdLoad() func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return s.newValue2(ssa.OpLoad, n.Type(), args[0], s.mem())
	}
}

func simdStore() func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		s.store(args[0].Type, args[1], args[0])
		return nil
	}
}

var cvtVToMaskOpcodes = map[int]map[int]ssa.Op{
	8:  {16: ssa.OpCvt16toMask8x16, 32: ssa.OpCvt32toMask8x32, 64: ssa.OpCvt64toMask8x64},
	16: {8: ssa.OpCvt8toMask16x8, 16: ssa.OpCvt16toMask16x16, 32: ssa.OpCvt32toMask16x32},
	32: {4: ssa.OpCvt8toMask32x4, 8: ssa.OpCvt8toMask32x8, 16: ssa.OpCvt16toMask32x16},
	64: {2: ssa.OpCvt8toMask64x2, 4: ssa.OpCvt8toMask64x4, 8: ssa.OpCvt8toMask64x8},
}

var cvtMaskToVOpcodes = map[int]map[int]ssa.Op{
	8:  {16: ssa.OpCvtMask8x16to16, 32: ssa.OpCvtMask8x32to32, 64: ssa.OpCvtMask8x64to64},
	16: {8: ssa.OpCvtMask16x8to8, 16: ssa.OpCvtMask16x16to16, 32: ssa.OpCvtMask16x32to32},
	32: {4: ssa.OpCvtMask32x4to8, 8: ssa.OpCvtMask32x8to8, 16: ssa.OpCvtMask32x16to16},
	64: {2: ssa.OpCvtMask64x2to8, 4: ssa.OpCvtMask64x4to8, 8: ssa.OpCvtMask64x8to8},
}

func simdCvtVToMask(elemBits, lanes int) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		op := cvtVToMaskOpcodes[elemBits][lanes]
		if op == 0 {
			panic(fmt.Sprintf("Unknown mask shape: Mask%dx%d", elemBits, lanes))
		}
		return s.newValue1(op, types.TypeMask, args[0])
	}
}

func simdCvtMaskToV(elemBits, lanes int) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		op := cvtMaskToVOpcodes[elemBits][lanes]
		if op == 0 {
			panic(fmt.Sprintf("Unknown mask shape: Mask%dx%d", elemBits, lanes))
		}
		return s.newValue1(op, n.Type(), args[0])
	}
}

func simdMaskedLoad(op ssa.Op) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return s.newValue3(op, n.Type(), args[0], args[1], s.mem())
	}
}

func simdMaskedStore(op ssa.Op) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		s.vars[memVar] = s.newValue4A(op, types.TypeMem, args[0].Type, args[1], args[2], args[0], s.mem())
		return nil
	}
}

// findIntrinsic returns a function which builds the SSA equivalent of the
// function identified by the symbol sym.  If sym is not an intrinsic call, returns nil.
func findIntrinsic(sym *types.Sym) intrinsicBuilder {
	if sym == nil || sym.Pkg == nil {
		return nil
	}
	pkg := sym.Pkg.Path
	if sym.Pkg == ir.Pkgs.Runtime {
		pkg = "runtime"
	}
	if base.Flag.Race && pkg == "sync/atomic" {
		// The race detector needs to be able to intercept these calls.
		// We can't intrinsify them.
		return nil
	}
	// Skip intrinsifying math functions (which may contain hard-float
	// instructions) when soft-float
	if Arch.SoftFloat && pkg == "math" {
		return nil
	}

	fn := sym.Name
	if ssa.IntrinsicsDisable {
		if pkg == "internal/runtime/sys" && (fn == "GetCallerPC" || fn == "GrtCallerSP" || fn == "GetClosurePtr") ||
			pkg == "internal/simd" || pkg == "simd" { // TODO after simd has been moved to package simd, remove internal/simd
			// These runtime functions don't have definitions, must be intrinsics.
		} else {
			return nil
		}
	}
	return intrinsics.lookup(Arch.LinkArch.Arch, pkg, fn)
}

func IsIntrinsicCall(n *ir.CallExpr) bool {
	if n == nil {
		return false
	}
	name, ok := n.Fun.(*ir.Name)
	if !ok {
		if n.Fun.Op() == ir.OMETHEXPR {
			if meth := ir.MethodExprName(n.Fun); meth != nil {
				if fn := meth.Func; fn != nil {
					return IsIntrinsicSym(fn.Sym())
				}
			}
		}
		return false
	}
	return IsIntrinsicSym(name.Sym())
}

func IsIntrinsicSym(sym *types.Sym) bool {
	return findIntrinsic(sym) != nil
}

// GenIntrinsicBody generates the function body for a bodyless intrinsic.
// This is used when the intrinsic is used in a non-call context, e.g.
// as a function pointer, or (for a method) being referenced from the type
// descriptor.
//
// The compiler already recognizes a call to fn as an intrinsic and can
// directly generate code for it. So we just fill in the body with a call
// to fn.
func GenIntrinsicBody(fn *ir.Func) {
	if ir.CurFunc != nil {
		base.FatalfAt(fn.Pos(), "enqueueFunc %v inside %v", fn, ir.CurFunc)
	}

	if base.Flag.LowerR != 0 {
		fmt.Println("generate intrinsic for", ir.FuncName(fn))
	}

	pos := fn.Pos()
	ft := fn.Type()
	var ret ir.Node

	// For a method, it usually starts with an ODOTMETH (pre-typecheck) or
	// OMETHEXPR (post-typecheck) referencing the method symbol without the
	// receiver type, and Walk rewrites it to a call directly to the
	// type-qualified method symbol, moving the receiver to an argument.
	// Here fn has already the type-qualified method symbol, and it is hard
	// to get the unqualified symbol. So we just generate the post-Walk form
	// and mark it typechecked and Walked.
	call := ir.NewCallExpr(pos, ir.OCALLFUNC, fn.Nname, nil)
	call.Args = ir.RecvParamNames(ft)
	call.IsDDD = ft.IsVariadic()
	typecheck.Exprs(call.Args)
	call.SetTypecheck(1)
	call.SetWalked(true)
	ret = call
	if ft.NumResults() > 0 {
		if ft.NumResults() == 1 {
			call.SetType(ft.Result(0).Type)
		} else {
			call.SetType(ft.ResultsTuple())
		}
		n := ir.NewReturnStmt(base.Pos, nil)
		n.Results = []ir.Node{call}
		ret = n
	}
	fn.Body.Append(ret)

	if base.Flag.LowerR != 0 {
		ir.DumpList("generate intrinsic body", fn.Body)
	}

	ir.CurFunc = fn
	typecheck.Stmts(fn.Body)
	ir.CurFunc = nil // we know CurFunc is nil at entry
}
