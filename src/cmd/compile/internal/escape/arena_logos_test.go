// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"os"
	"testing"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"internal/buildcfg"
)

func TestMain(m *testing.M) {
	types.LocalPkg = types.NewPkg("p", "p")
	types.PtrSize = 8
	types.RegSize = 8
	types.MaxWidth = 1 << 50
	typecheck.InitUniverse()
	os.Exit(m.Run())
}

func enableLogosArena(t *testing.T) {
	t.Helper()
	old := buildcfg.Experiment.LogosArena
	buildcfg.Experiment.LogosArena = true
	t.Cleanup(func() { buildcfg.Experiment.LogosArena = old })
}

func logosArenaTestMake(lenNode, capNode ir.Node, esc uint16) *ir.MakeExpr {
	n := ir.NewMakeExpr(src.NoXPos, ir.OMAKESLICE, lenNode, capNode)
	n.SetType(types.NewSlice(types.Types[types.TUINT8]))
	n.SetEsc(esc)
	return n
}

func logosArenaTestLen() ir.Node {
	return ir.NewNameAt(src.NoXPos, types.LocalPkg.Lookup("n"), types.Types[types.TINT])
}

func logosArenaDynamicMake(esc uint16) *ir.MakeExpr {
	return logosArenaTestMake(logosArenaTestLen(), nil, esc)
}

func TestLogosArenaShadowNeverReducesEscape(t *testing.T) {
	enableLogosArena(t)

	tests := []struct {
		name string
		make *ir.MakeExpr
		want bool
	}{
		{name: "non-escaping dynamic make", make: logosArenaDynamicMake(ir.EscNone), want: true},
		{name: "escaping dynamic make", make: logosArenaDynamicMake(ir.EscHeap), want: false},
		{name: "make stored in sync.Pool", make: logosArenaDynamicMake(ir.EscHeap), want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := LogosArenaMake(tt.make, &ir.Func{}); got != tt.want {
				t.Errorf("LogosArenaMake() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLogosArenaForeignPrecedence(t *testing.T) {
	enableLogosArena(t)

	control := logosArenaDynamicMake(ir.EscNone)
	if !foreignBacking(&location{n: control}) {
		t.Fatal("dynamic make is not foreignBacking-shaped; test would not exercise the foreign shape")
	}
	if !LogosArenaMake(control, &ir.Func{}) {
		t.Fatal("variable-capacity make without a foreign heap decision was vetoed")
	}

	foreignForced := logosArenaDynamicMake(ir.EscHeap)
	if !logosArenaSolvedHeap(foreignForced) {
		t.Fatal("EscHeap make not reported as heap-forced")
	}
	if LogosArenaMake(foreignForced, &ir.Func{}) {
		t.Error("foreign-forced make was arena-placed by LogosArenaMake")
	}
	if LogosArenaConstMake(foreignForced, &ir.Func{}) {
		t.Error("foreign-forced make was arena-placed by LogosArenaConstMake")
	}

	fn := &ir.Func{}
	b := &batch{}
	shadowForced := &location{n: foreignForced, curfn: fn, logosNoRet: true}
	shadowForced.attrs |= attrEscapes
	b.logosArena.makes = append(b.logosArena.makes, logosArenaMakeLoc{n: foreignForced, shadow: shadowForced})

	returned := logosArenaDynamicMake(ir.EscHeap)
	shadowReturned := &location{n: returned, curfn: fn, logosNoRet: true}
	b.logosArena.makes = append(b.logosArena.makes, logosArenaMakeLoc{n: returned, shadow: shadowReturned})

	b.logosArenaMark([]*ir.Func{fn})
	t.Cleanup(func() {
		logosArenaReg.mu.Lock()
		defer logosArenaReg.mu.Unlock()
		delete(logosArenaReg.retSites, foreignForced)
		delete(logosArenaReg.retSites, returned)
		logosArenaReg.recompute()
	})

	logosArenaReg.mu.RLock()
	_, forcedRegistered := logosArenaReg.retSites[foreignForced]
	_, returnedRegistered := logosArenaReg.retSites[returned]
	logosArenaReg.mu.RUnlock()
	if forcedRegistered {
		t.Error("foreign-forced returned make was recorded as a caller-region site")
	}
	if !returnedRegistered {
		t.Error("result-only returned make was not recorded as a caller-region site")
	}
	if LogosArenaRetMake(foreignForced, fn) {
		t.Error("foreign-forced make reported by LogosArenaRetMake")
	}
}

func TestLogosArenaVetoes(t *testing.T) {
	enableLogosArena(t)
	pos := src.NoXPos

	t.Run("goto", func(t *testing.T) {
		fn := &ir.Func{Body: ir.Nodes{ir.NewBranchStmt(pos, ir.OGOTO, nil)}}
		if !LogosArenaVetoed(fn) {
			t.Error("goto function not vetoed")
		}
		if LogosArenaPlaceable(fn) {
			t.Error("goto function is placeable")
		}
		if LogosArenaMake(logosArenaDynamicMake(ir.EscNone), fn) {
			t.Error("goto function selected arena placement")
		}
	})

	t.Run("closure", func(t *testing.T) {
		fn := &ir.Func{OClosure: &ir.ClosureExpr{}}
		if logosArenaRetCapable(fn) {
			t.Error("closure is return-capable")
		}
	})

	t.Run("dupok", func(t *testing.T) {
		fn := &ir.Func{}
		fn.SetDupok(true)
		if logosArenaRetCapable(fn) {
			t.Error("dupok function is return-capable")
		}
	})

	t.Run("nosplit", func(t *testing.T) {
		fn := &ir.Func{Pragma: ir.Nosplit}
		if LogosArenaPlaceable(fn) {
			t.Error("nosplit function is placeable")
		}
		if LogosArenaMake(logosArenaDynamicMake(ir.EscNone), fn) {
			t.Error("nosplit function selected arena placement")
		}
	})

	t.Run("wrapper", func(t *testing.T) {
		fn := &ir.Func{}
		fn.SetWrapper(true)
		if LogosArenaPlaceable(fn) {
			t.Error("wrapper function is placeable")
		}
		if LogosArenaMake(logosArenaDynamicMake(ir.EscNone), fn) {
			t.Error("wrapper function selected arena placement")
		}
	})

	t.Run("defer", func(t *testing.T) {
		fn := &ir.Func{Body: ir.Nodes{ir.NewGoDeferStmt(pos, ir.ODEFER, nil)}}
		if LogosArenaPlaceable(fn) {
			t.Error("defer function is placeable")
		}
		if LogosArenaMake(logosArenaDynamicMake(ir.EscNone), fn) {
			t.Error("defer function selected arena placement")
		}
		if LogosArenaConstMake(logosArenaTestMake(ir.NewInt(pos, 4096), nil, ir.EscNone), fn) {
			t.Error("defer function selected const arena placement")
		}
	})

	t.Run("compiling runtime", func(t *testing.T) {
		old := base.Flag.CompilingRuntime
		base.Flag.CompilingRuntime = true
		t.Cleanup(func() { base.Flag.CompilingRuntime = old })
		fn := &ir.Func{}
		if LogosArenaPlaceable(fn) {
			t.Error("runtime function is placeable")
		}
		if LogosArenaMake(logosArenaDynamicMake(ir.EscNone), fn) {
			t.Error("runtime function selected arena placement")
		}
		if LogosArenaConstMake(logosArenaTestMake(ir.NewInt(pos, 4096), nil, ir.EscNone), fn) {
			t.Error("runtime function selected const arena placement")
		}
	})
}

func TestLogosArenaConstProductOverflow(t *testing.T) {
	enableLogosArena(t)
	fn := &ir.Func{}
	pos := src.NoXPos

	constMake := func(elem *types.Type, capVal int64) *ir.MakeExpr {
		n := logosArenaTestMake(ir.NewInt(pos, capVal), nil, ir.EscNone)
		n.SetType(types.NewSlice(elem))
		return n
	}
	array := func(size int64) *types.Type {
		return types.NewArray(types.Types[types.TUINT8], size)
	}

	wrapElem, wrapCap := types.MaxWidth/2+1, int64(1)<<15
	if uint64(wrapElem)*uint64(wrapCap) != 1<<15 {
		t.Fatalf("premise: %d * %d does not wrap to 2^15", wrapElem, wrapCap)
	}
	zeroElem, zeroCap := types.MaxWidth/2, int64(1)<<15
	if uint64(zeroElem)*uint64(zeroCap) != 0 {
		t.Fatalf("premise: %d * %d does not wrap to 0", zeroElem, zeroCap)
	}

	tests := []struct {
		name      string
		make      *ir.MakeExpr
		wantBytes int64
		wantOK    bool
	}{
		{"byte slice", constMake(types.Types[types.TUINT8], 4096), 4096, true},
		{"largest byte slice", constMake(types.Types[types.TUINT8], LogosArenaMaxBytes), LogosArenaMaxBytes, true},
		{"one byte too large", constMake(types.Types[types.TUINT8], LogosArenaMaxBytes+1), 0, false},
		{"largest word slice", constMake(types.Types[types.TUINT64], LogosArenaMaxBytes/8), LogosArenaMaxBytes / 8 * 8, true},
		{"one word too large", constMake(types.Types[types.TUINT64], LogosArenaMaxBytes/8+1), 0, false},
		{"element of one chunk", constMake(array(LogosArenaMaxBytes), 1), LogosArenaMaxBytes, true},
		{"element past one chunk", constMake(array(LogosArenaMaxBytes+1), 1), 0, false},
		{"zero capacity", constMake(types.Types[types.TUINT8], 0), 0, false},
		{"product wraps to small", constMake(array(wrapElem), wrapCap), 0, false},
		{"product wraps to zero", constMake(array(zeroElem), zeroCap), 0, false},
		{"dynamic capacity", logosArenaDynamicMake(ir.EscNone), 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotBytes, gotOK := LogosArenaConstBytes(tt.make)
			if gotBytes != tt.wantBytes || gotOK != tt.wantOK {
				t.Errorf("LogosArenaConstBytes() = %d, %v, want %d, %v", gotBytes, gotOK, tt.wantBytes, tt.wantOK)
			}
			if got := LogosArenaConstMake(tt.make, fn); got != (tt.wantOK && tt.wantBytes > int64(base.Debug.VariableMakeThreshold)) {
				t.Errorf("LogosArenaConstMake() = %v for %d bytes (ok %v)", got, tt.wantBytes, tt.wantOK)
			}
		})
	}

	for _, tt := range []struct {
		name         string
		elemSize, n  int64
		wantFitsOnce bool
	}{
		{"wraps to small", wrapElem, wrapCap, false},
		{"wraps to zero", zeroElem, zeroCap, false},
		{"exact fit", 8, LogosArenaMaxBytes / 8, true},
		{"one past", 8, LogosArenaMaxBytes/8 + 1, false},
	} {
		if got := logosArenaConstFits(tt.n, tt.elemSize); got != tt.wantFitsOnce {
			t.Errorf("logosArenaConstFits(%d, %d) = %v (%s), want %v", tt.n, tt.elemSize, got, tt.name, tt.wantFitsOnce)
		}
	}
}
