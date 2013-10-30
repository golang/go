// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/ssa"
)

// Instances of 'intrinsic' generate analysis constraints for calls to
// intrinsic functions.
// Implementations may exploit information from the calling site
// via cgn.callersite; for shared contours this is nil.
type intrinsic func(a *analysis, cgn *cgnode)

// Initialized in explicit init() to defeat (spurious) initialization
// cycle error.
var intrinsicsByName map[string]intrinsic

func init() {
	// Key strings are from Function.String().
	// That little dot ۰ is an Arabic zero numeral (U+06F0),
	// categories [Nd].
	intrinsicsByName = map[string]intrinsic{
		// reflect.Value methods.
		"(reflect.Value).Addr":            ext۰reflect۰Value۰Addr,
		"(reflect.Value).Bool":            ext۰NoEffect,
		"(reflect.Value).Bytes":           ext۰reflect۰Value۰Bytes,
		"(reflect.Value).Call":            ext۰reflect۰Value۰Call,
		"(reflect.Value).CallSlice":       ext۰reflect۰Value۰CallSlice,
		"(reflect.Value).CanAddr":         ext۰NoEffect,
		"(reflect.Value).CanInterface":    ext۰NoEffect,
		"(reflect.Value).CanSet":          ext۰NoEffect,
		"(reflect.Value).Cap":             ext۰NoEffect,
		"(reflect.Value).Close":           ext۰NoEffect,
		"(reflect.Value).Complex":         ext۰NoEffect,
		"(reflect.Value).Convert":         ext۰reflect۰Value۰Convert,
		"(reflect.Value).Elem":            ext۰reflect۰Value۰Elem,
		"(reflect.Value).Field":           ext۰reflect۰Value۰Field,
		"(reflect.Value).FieldByIndex":    ext۰reflect۰Value۰FieldByIndex,
		"(reflect.Value).FieldByName":     ext۰reflect۰Value۰FieldByName,
		"(reflect.Value).FieldByNameFunc": ext۰reflect۰Value۰FieldByNameFunc,
		"(reflect.Value).Float":           ext۰NoEffect,
		"(reflect.Value).Index":           ext۰reflect۰Value۰Index,
		"(reflect.Value).Int":             ext۰NoEffect,
		"(reflect.Value).Interface":       ext۰reflect۰Value۰Interface,
		"(reflect.Value).InterfaceData":   ext۰NoEffect,
		"(reflect.Value).IsNil":           ext۰NoEffect,
		"(reflect.Value).IsValid":         ext۰NoEffect,
		"(reflect.Value).Kind":            ext۰NoEffect,
		"(reflect.Value).Len":             ext۰NoEffect,
		"(reflect.Value).MapIndex":        ext۰reflect۰Value۰MapIndex,
		"(reflect.Value).MapKeys":         ext۰reflect۰Value۰MapKeys,
		"(reflect.Value).Method":          ext۰reflect۰Value۰Method,
		"(reflect.Value).MethodByName":    ext۰reflect۰Value۰MethodByName,
		"(reflect.Value).NumField":        ext۰NoEffect,
		"(reflect.Value).NumMethod":       ext۰NoEffect,
		"(reflect.Value).OverflowComplex": ext۰NoEffect,
		"(reflect.Value).OverflowFloat":   ext۰NoEffect,
		"(reflect.Value).OverflowInt":     ext۰NoEffect,
		"(reflect.Value).OverflowUint":    ext۰NoEffect,
		"(reflect.Value).Pointer":         ext۰NoEffect,
		"(reflect.Value).Recv":            ext۰reflect۰Value۰Recv,
		"(reflect.Value).Send":            ext۰reflect۰Value۰Send,
		"(reflect.Value).Set":             ext۰reflect۰Value۰Set,
		"(reflect.Value).SetBool":         ext۰NoEffect,
		"(reflect.Value).SetBytes":        ext۰reflect۰Value۰SetBytes,
		"(reflect.Value).SetComplex":      ext۰NoEffect,
		"(reflect.Value).SetFloat":        ext۰NoEffect,
		"(reflect.Value).SetInt":          ext۰NoEffect,
		"(reflect.Value).SetLen":          ext۰NoEffect,
		"(reflect.Value).SetMapIndex":     ext۰reflect۰Value۰SetMapIndex,
		"(reflect.Value).SetPointer":      ext۰reflect۰Value۰SetPointer,
		"(reflect.Value).SetString":       ext۰NoEffect,
		"(reflect.Value).SetUint":         ext۰NoEffect,
		"(reflect.Value).Slice":           ext۰reflect۰Value۰Slice,
		"(reflect.Value).String":          ext۰NoEffect,
		"(reflect.Value).TryRecv":         ext۰reflect۰Value۰Recv,
		"(reflect.Value).TrySend":         ext۰reflect۰Value۰Send,
		"(reflect.Value).Type":            ext۰NoEffect,
		"(reflect.Value).Uint":            ext۰NoEffect,
		"(reflect.Value).UnsafeAddr":      ext۰NoEffect,

		// Standalone reflect.* functions.
		"reflect.Append":      ext۰reflect۰Append,
		"reflect.AppendSlice": ext۰reflect۰AppendSlice,
		"reflect.Copy":        ext۰reflect۰Copy,
		"reflect.ChanOf":      ext۰reflect۰ChanOf,
		"reflect.DeepEqual":   ext۰NoEffect,
		"reflect.Indirect":    ext۰reflect۰Indirect,
		"reflect.MakeChan":    ext۰reflect۰MakeChan,
		"reflect.MakeFunc":    ext۰reflect۰MakeFunc,
		"reflect.MakeMap":     ext۰reflect۰MakeMap,
		"reflect.MakeSlice":   ext۰reflect۰MakeSlice,
		"reflect.MapOf":       ext۰reflect۰MapOf,
		"reflect.New":         ext۰reflect۰New,
		"reflect.NewAt":       ext۰reflect۰NewAt,
		"reflect.PtrTo":       ext۰reflect۰PtrTo,
		"reflect.Select":      ext۰reflect۰Select,
		"reflect.SliceOf":     ext۰reflect۰SliceOf,
		"reflect.TypeOf":      ext۰reflect۰TypeOf,
		"reflect.ValueOf":     ext۰reflect۰ValueOf,
		"reflect.Zero":        ext۰reflect۰Zero,
		"reflect.init":        ext۰NoEffect,

		// *reflect.rtype methods
		"(*reflect.rtype).Align":           ext۰NoEffect,
		"(*reflect.rtype).AssignableTo":    ext۰NoEffect,
		"(*reflect.rtype).Bits":            ext۰NoEffect,
		"(*reflect.rtype).ChanDir":         ext۰NoEffect,
		"(*reflect.rtype).ConvertibleTo":   ext۰NoEffect,
		"(*reflect.rtype).Elem":            ext۰reflect۰rtype۰Elem,
		"(*reflect.rtype).Field":           ext۰reflect۰rtype۰Field,
		"(*reflect.rtype).FieldAlign":      ext۰NoEffect,
		"(*reflect.rtype).FieldByIndex":    ext۰reflect۰rtype۰FieldByIndex,
		"(*reflect.rtype).FieldByName":     ext۰reflect۰rtype۰FieldByName,
		"(*reflect.rtype).FieldByNameFunc": ext۰reflect۰rtype۰FieldByNameFunc,
		"(*reflect.rtype).Implements":      ext۰NoEffect,
		"(*reflect.rtype).In":              ext۰reflect۰rtype۰In,
		"(*reflect.rtype).IsVariadic":      ext۰NoEffect,
		"(*reflect.rtype).Key":             ext۰reflect۰rtype۰Key,
		"(*reflect.rtype).Kind":            ext۰NoEffect,
		"(*reflect.rtype).Len":             ext۰NoEffect,
		"(*reflect.rtype).Method":          ext۰reflect۰rtype۰Method,
		"(*reflect.rtype).MethodByName":    ext۰reflect۰rtype۰MethodByName,
		"(*reflect.rtype).Name":            ext۰NoEffect,
		"(*reflect.rtype).NumField":        ext۰NoEffect,
		"(*reflect.rtype).NumIn":           ext۰NoEffect,
		"(*reflect.rtype).NumMethod":       ext۰NoEffect,
		"(*reflect.rtype).NumOut":          ext۰NoEffect,
		"(*reflect.rtype).Out":             ext۰reflect۰rtype۰Out,
		"(*reflect.rtype).PkgPath":         ext۰NoEffect,
		"(*reflect.rtype).Size":            ext۰NoEffect,
		"(*reflect.rtype).String":          ext۰NoEffect,

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
		"os.epipecheck":                         ext۰NoEffect,
		"runtime.BlockProfile":                  ext۰NoEffect,
		"runtime.Breakpoint":                    ext۰NoEffect,
		"runtime.CPUProfile":                    ext۰NotYetImplemented,
		"runtime.Caller":                        ext۰NoEffect,
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
		"sync/atomic.AddInt32":                  ext۰NoEffect,
		"sync/atomic.AddUint32":                 ext۰NoEffect,
		"sync/atomic.CompareAndSwapInt32":       ext۰NoEffect,
		"sync/atomic.CompareAndSwapUint32":      ext۰NoEffect,
		"sync/atomic.CompareAndSwapUint64":      ext۰NoEffect,
		"sync/atomic.CompareAndSwapUintptr":     ext۰NoEffect,
		"sync/atomic.LoadInt32":                 ext۰NoEffect,
		"sync/atomic.LoadUint32":                ext۰NoEffect,
		"sync/atomic.LoadUint64":                ext۰NoEffect,
		"sync/atomic.StoreInt32":                ext۰NoEffect,
		"sync/atomic.StoreUint32":               ext۰NoEffect,
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
		"time.Sleep":                            ext۰NoEffect,
		"time.now":                              ext۰NoEffect,
		"time.startTimer":                       ext۰NoEffect,
		"time.stopTimer":                        ext۰NoEffect,
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

		if fn.Pkg != nil && a.reflectValueObj != nil && a.reflectValueObj.Pkg() == fn.Pkg.Object {
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
	// TODO(adonovan): enable this warning when we've implemented
	// enough that it's not unbearably annoying.
	// a.warnf(fn.Pos(), "unsound: intrinsic treatment of %s not yet implemented", fn)
}

// ---------- func runtime.SetFinalizer(x, f interface{}) ----------

// runtime.SetFinalizer(x, f)
type runtimeSetFinalizerConstraint struct {
	targets nodeid
	f       nodeid // (ptr)
	x       nodeid
}

func (c *runtimeSetFinalizerConstraint) String() string {
	return fmt.Sprintf("runtime.SetFinalizer(n%d, n%d)", c.x, c.f)
}

func (c *runtimeSetFinalizerConstraint) ptr() nodeid {
	return c.f
}

func (c *runtimeSetFinalizerConstraint) solve(a *analysis, _ *node, delta nodeset) {
	for fObj := range delta {
		tDyn, f, indirect := a.taggedValue(fObj)
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
