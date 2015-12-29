// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5

package pointer

// This package defines the treatment of intrinsics, i.e. library
// functions requiring special analytical treatment.
//
// Most of these are C or assembly functions, but even some Go
// functions require may special treatment if the analysis completely
// replaces the implementation of an API such as reflection.

// TODO(adonovan): support a means of writing analytic summaries in
// the target code, so that users can summarise the effects of their
// own C functions using a snippet of Go.

import (
	"fmt"
	"go/types"

	"golang.org/x/tools/go/ssa"
)

// Instances of 'intrinsic' generate analysis constraints for calls to
// intrinsic functions.
// Implementations may exploit information from the calling site
// via cgn.callersite; for shared contours this is nil.
type intrinsic func(a *analysis, cgn *cgnode)

// Initialized in explicit init() to defeat (spurious) initialization
// cycle error.
var intrinsicsByName = make(map[string]intrinsic)

func init() {
	// Key strings are from Function.String().
	// That little dot ۰ is an Arabic zero numeral (U+06F0),
	// categories [Nd].
	for name, fn := range map[string]intrinsic{
		// Other packages.
		"bytes.Equal":                           ext۰NoEffect,
		"bytes.IndexByte":                       ext۰NoEffect,
		"crypto/aes.decryptBlockAsm":            ext۰NoEffect,
		"crypto/aes.encryptBlockAsm":            ext۰NoEffect,
		"crypto/aes.expandKeyAsm":               ext۰NoEffect,
		"crypto/aes.hasAsm":                     ext۰NoEffect,
		"crypto/md5.block":                      ext۰NoEffect,
		"crypto/rc4.xorKeyStream":               ext۰NoEffect,
		"crypto/sha1.block":                     ext۰NoEffect,
		"crypto/sha256.block":                   ext۰NoEffect,
		"hash/crc32.castagnoliSSE42":            ext۰NoEffect,
		"hash/crc32.haveSSE42":                  ext۰NoEffect,
		"math.Abs":                              ext۰NoEffect,
		"math.Acos":                             ext۰NoEffect,
		"math.Asin":                             ext۰NoEffect,
		"math.Atan":                             ext۰NoEffect,
		"math.Atan2":                            ext۰NoEffect,
		"math.Ceil":                             ext۰NoEffect,
		"math.Cos":                              ext۰NoEffect,
		"math.Dim":                              ext۰NoEffect,
		"math.Exp":                              ext۰NoEffect,
		"math.Exp2":                             ext۰NoEffect,
		"math.Expm1":                            ext۰NoEffect,
		"math.Float32bits":                      ext۰NoEffect,
		"math.Float32frombits":                  ext۰NoEffect,
		"math.Float64bits":                      ext۰NoEffect,
		"math.Float64frombits":                  ext۰NoEffect,
		"math.Floor":                            ext۰NoEffect,
		"math.Frexp":                            ext۰NoEffect,
		"math.Hypot":                            ext۰NoEffect,
		"math.Ldexp":                            ext۰NoEffect,
		"math.Log":                              ext۰NoEffect,
		"math.Log10":                            ext۰NoEffect,
		"math.Log1p":                            ext۰NoEffect,
		"math.Log2":                             ext۰NoEffect,
		"math.Max":                              ext۰NoEffect,
		"math.Min":                              ext۰NoEffect,
		"math.Mod":                              ext۰NoEffect,
		"math.Modf":                             ext۰NoEffect,
		"math.Remainder":                        ext۰NoEffect,
		"math.Sin":                              ext۰NoEffect,
		"math.Sincos":                           ext۰NoEffect,
		"math.Sqrt":                             ext۰NoEffect,
		"math.Tan":                              ext۰NoEffect,
		"math.Trunc":                            ext۰NoEffect,
		"math/big.addMulVVW":                    ext۰NoEffect,
		"math/big.addVV":                        ext۰NoEffect,
		"math/big.addVW":                        ext۰NoEffect,
		"math/big.bitLen":                       ext۰NoEffect,
		"math/big.divWVW":                       ext۰NoEffect,
		"math/big.divWW":                        ext۰NoEffect,
		"math/big.mulAddVWW":                    ext۰NoEffect,
		"math/big.mulWW":                        ext۰NoEffect,
		"math/big.shlVU":                        ext۰NoEffect,
		"math/big.shrVU":                        ext۰NoEffect,
		"math/big.subVV":                        ext۰NoEffect,
		"math/big.subVW":                        ext۰NoEffect,
		"net.runtime_Semacquire":                ext۰NoEffect,
		"net.runtime_Semrelease":                ext۰NoEffect,
		"net.runtime_pollClose":                 ext۰NoEffect,
		"net.runtime_pollOpen":                  ext۰NoEffect,
		"net.runtime_pollReset":                 ext۰NoEffect,
		"net.runtime_pollServerInit":            ext۰NoEffect,
		"net.runtime_pollSetDeadline":           ext۰NoEffect,
		"net.runtime_pollUnblock":               ext۰NoEffect,
		"net.runtime_pollWait":                  ext۰NoEffect,
		"net.runtime_pollWaitCanceled":          ext۰NoEffect,
		"os.epipecheck":                         ext۰NoEffect,
		"runtime.BlockProfile":                  ext۰NoEffect,
		"runtime.Breakpoint":                    ext۰NoEffect,
		"runtime.CPUProfile":                    ext۰NoEffect, // good enough
		"runtime.Caller":                        ext۰NoEffect,
		"runtime.Callers":                       ext۰NoEffect, // good enough
		"runtime.FuncForPC":                     ext۰NoEffect,
		"runtime.GC":                            ext۰NoEffect,
		"runtime.GOMAXPROCS":                    ext۰NoEffect,
		"runtime.Goexit":                        ext۰NoEffect,
		"runtime.GoroutineProfile":              ext۰NoEffect,
		"runtime.Gosched":                       ext۰NoEffect,
		"runtime.MemProfile":                    ext۰NoEffect,
		"runtime.NumCPU":                        ext۰NoEffect,
		"runtime.NumGoroutine":                  ext۰NoEffect,
		"runtime.ReadMemStats":                  ext۰NoEffect,
		"runtime.SetBlockProfileRate":           ext۰NoEffect,
		"runtime.SetCPUProfileRate":             ext۰NoEffect,
		"runtime.SetFinalizer":                  ext۰runtime۰SetFinalizer,
		"runtime.Stack":                         ext۰NoEffect,
		"runtime.ThreadCreateProfile":           ext۰NoEffect,
		"runtime.cstringToGo":                   ext۰NoEffect,
		"runtime.funcentry_go":                  ext۰NoEffect,
		"runtime.funcline_go":                   ext۰NoEffect,
		"runtime.funcname_go":                   ext۰NoEffect,
		"runtime.getgoroot":                     ext۰NoEffect,
		"runtime/pprof.runtime_cyclesPerSecond": ext۰NoEffect,
		"strings.IndexByte":                     ext۰NoEffect,
		"sync.runtime_Semacquire":               ext۰NoEffect,
		"sync.runtime_Semrelease":               ext۰NoEffect,
		"sync.runtime_Syncsemacquire":           ext۰NoEffect,
		"sync.runtime_Syncsemcheck":             ext۰NoEffect,
		"sync.runtime_Syncsemrelease":           ext۰NoEffect,
		"sync.runtime_procPin":                  ext۰NoEffect,
		"sync.runtime_procUnpin":                ext۰NoEffect,
		"sync.runtime_registerPool":             ext۰NoEffect,
		"sync/atomic.AddInt32":                  ext۰NoEffect,
		"sync/atomic.AddInt64":                  ext۰NoEffect,
		"sync/atomic.AddUint32":                 ext۰NoEffect,
		"sync/atomic.AddUint64":                 ext۰NoEffect,
		"sync/atomic.AddUintptr":                ext۰NoEffect,
		"sync/atomic.CompareAndSwapInt32":       ext۰NoEffect,
		"sync/atomic.CompareAndSwapUint32":      ext۰NoEffect,
		"sync/atomic.CompareAndSwapUint64":      ext۰NoEffect,
		"sync/atomic.CompareAndSwapUintptr":     ext۰NoEffect,
		"sync/atomic.LoadInt32":                 ext۰NoEffect,
		"sync/atomic.LoadInt64":                 ext۰NoEffect,
		"sync/atomic.LoadPointer":               ext۰NoEffect, // ignore unsafe.Pointers
		"sync/atomic.LoadUint32":                ext۰NoEffect,
		"sync/atomic.LoadUint64":                ext۰NoEffect,
		"sync/atomic.LoadUintptr":               ext۰NoEffect,
		"sync/atomic.StoreInt32":                ext۰NoEffect,
		"sync/atomic.StorePointer":              ext۰NoEffect, // ignore unsafe.Pointers
		"sync/atomic.StoreUint32":               ext۰NoEffect,
		"sync/atomic.StoreUintptr":              ext۰NoEffect,
		"syscall.Close":                         ext۰NoEffect,
		"syscall.Exit":                          ext۰NoEffect,
		"syscall.Getpid":                        ext۰NoEffect,
		"syscall.Getwd":                         ext۰NoEffect,
		"syscall.Kill":                          ext۰NoEffect,
		"syscall.RawSyscall":                    ext۰NoEffect,
		"syscall.RawSyscall6":                   ext۰NoEffect,
		"syscall.Syscall":                       ext۰NoEffect,
		"syscall.Syscall6":                      ext۰NoEffect,
		"syscall.runtime_AfterFork":             ext۰NoEffect,
		"syscall.runtime_BeforeFork":            ext۰NoEffect,
		"syscall.setenv_c":                      ext۰NoEffect,
		"time.Sleep":                            ext۰NoEffect,
		"time.now":                              ext۰NoEffect,
		"time.startTimer":                       ext۰time۰startTimer,
		"time.stopTimer":                        ext۰NoEffect,
	} {
		intrinsicsByName[name] = fn
	}
}

// findIntrinsic returns the constraint generation function for an
// intrinsic function fn, or nil if the function should be handled normally.
//
func (a *analysis) findIntrinsic(fn *ssa.Function) intrinsic {
	// Consult the *Function-keyed cache.
	// A cached nil indicates a normal non-intrinsic function.
	impl, ok := a.intrinsics[fn]
	if !ok {
		impl = intrinsicsByName[fn.String()] // may be nil

		if a.isReflect(fn) {
			if !a.config.Reflection {
				impl = ext۰NoEffect // reflection disabled
			} else if impl == nil {
				// Ensure all "reflect" code is treated intrinsically.
				impl = ext۰NotYetImplemented
			}
		}

		a.intrinsics[fn] = impl
	}
	return impl
}

// isReflect reports whether fn belongs to the "reflect" package.
func (a *analysis) isReflect(fn *ssa.Function) bool {
	if a.reflectValueObj == nil {
		return false // "reflect" package not loaded
	}
	reflectPackage := a.reflectValueObj.Pkg()
	if fn.Pkg != nil && fn.Pkg.Pkg == reflectPackage {
		return true
	}
	// Synthetic wrappers have a nil Pkg, so they slip through the
	// previous check.  Check the receiver package.
	// TODO(adonovan): should synthetic wrappers have a non-nil Pkg?
	if recv := fn.Signature.Recv(); recv != nil {
		if named, ok := deref(recv.Type()).(*types.Named); ok {
			if named.Obj().Pkg() == reflectPackage {
				return true // e.g. wrapper of (reflect.Value).f
			}
		}
	}
	return false
}

// A trivial intrinsic suitable for any function that does not:
// 1) induce aliases between its arguments or any global variables;
// 2) call any functions; or
// 3) create any labels.
//
// Many intrinsics (such as CompareAndSwapInt32) have a fourth kind of
// effect: loading or storing through a pointer.  Though these could
// be significant, we deliberately ignore them because they are
// generally not worth the effort.
//
// We sometimes violate condition #3 if the function creates only
// non-function labels, as the control-flow graph is still sound.
//
func ext۰NoEffect(a *analysis, cgn *cgnode) {}

func ext۰NotYetImplemented(a *analysis, cgn *cgnode) {
	fn := cgn.fn
	a.warnf(fn.Pos(), "unsound: intrinsic treatment of %s not yet implemented", fn)
}

// ---------- func runtime.SetFinalizer(x, f interface{}) ----------

// runtime.SetFinalizer(x, f)
type runtimeSetFinalizerConstraint struct {
	targets nodeid // (indirect)
	f       nodeid // (ptr)
	x       nodeid
}

func (c *runtimeSetFinalizerConstraint) ptr() nodeid { return c.f }
func (c *runtimeSetFinalizerConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.targets), "SetFinalizer.targets")
}
func (c *runtimeSetFinalizerConstraint) renumber(mapping []nodeid) {
	c.targets = mapping[c.targets]
	c.f = mapping[c.f]
	c.x = mapping[c.x]
}

func (c *runtimeSetFinalizerConstraint) String() string {
	return fmt.Sprintf("runtime.SetFinalizer(n%d, n%d)", c.x, c.f)
}

func (c *runtimeSetFinalizerConstraint) solve(a *analysis, delta *nodeset) {
	for _, fObj := range delta.AppendTo(a.deltaSpace) {
		tDyn, f, indirect := a.taggedValue(nodeid(fObj))
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		tSig, ok := tDyn.Underlying().(*types.Signature)
		if !ok {
			continue // not a function
		}
		if tSig.Recv() != nil {
			panic(tSig)
		}
		if tSig.Params().Len() != 1 {
			continue //  not a unary function
		}

		// Extract x to tmp.
		tx := tSig.Params().At(0).Type()
		tmp := a.addNodes(tx, "SetFinalizer.tmp")
		a.typeAssert(tx, tmp, c.x, false)

		// Call f(tmp).
		a.store(f, tmp, 1, a.sizeof(tx))

		// Add dynamic call target.
		if a.onlineCopy(c.targets, f) {
			a.addWork(c.targets)
		}
	}
}

func ext۰runtime۰SetFinalizer(a *analysis, cgn *cgnode) {
	// This is the shared contour, used for dynamic calls.
	targets := a.addOneNode(tInvalid, "SetFinalizer.targets", nil)
	cgn.sites = append(cgn.sites, &callsite{targets: targets})
	params := a.funcParams(cgn.obj)
	a.addConstraint(&runtimeSetFinalizerConstraint{
		targets: targets,
		x:       params,
		f:       params + 1,
	})
}

// ---------- func time.startTimer(t *runtimeTimer) ----------

// time.StartTimer(t)
type timeStartTimerConstraint struct {
	targets nodeid // (indirect)
	t       nodeid // (ptr)
}

func (c *timeStartTimerConstraint) ptr() nodeid { return c.t }
func (c *timeStartTimerConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.targets), "StartTimer.targets")
}
func (c *timeStartTimerConstraint) renumber(mapping []nodeid) {
	c.targets = mapping[c.targets]
	c.t = mapping[c.t]
}

func (c *timeStartTimerConstraint) String() string {
	return fmt.Sprintf("time.startTimer(n%d)", c.t)
}

func (c *timeStartTimerConstraint) solve(a *analysis, delta *nodeset) {
	for _, tObj := range delta.AppendTo(a.deltaSpace) {
		t := nodeid(tObj)

		// We model startTimer as if it was defined thus:
		// 	func startTimer(t *runtimeTimer) { t.f(t.arg) }

		// We hard-code the field offsets of time.runtimeTimer:
		// type runtimeTimer struct {
		//  0     __identity__
		//  1    i      int32
		//  2    when   int64
		//  3    period int64
		//  4    f      func(int64, interface{})
		//  5    arg    interface{}
		// }
		f := t + 4
		arg := t + 5

		// store t.arg to t.f.params[0]
		// (offset 1 => skip identity)
		a.store(f, arg, 1, 1)

		// Add dynamic call target.
		if a.onlineCopy(c.targets, f) {
			a.addWork(c.targets)
		}
	}
}

func ext۰time۰startTimer(a *analysis, cgn *cgnode) {
	// This is the shared contour, used for dynamic calls.
	targets := a.addOneNode(tInvalid, "startTimer.targets", nil)
	cgn.sites = append(cgn.sites, &callsite{targets: targets})
	params := a.funcParams(cgn.obj)
	a.addConstraint(&timeStartTimerConstraint{
		targets: targets,
		t:       params,
	})
}
