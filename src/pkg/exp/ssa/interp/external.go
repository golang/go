// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package interp

// Emulated functions that we cannot interpret because they are
// external or because they use "unsafe" or "reflect" operations.

import (
	"exp/ssa"
	"math"
	"os"
	"runtime"
	"syscall"
	"time"
)

type externalFn func(fn *ssa.Function, args []value) value

// Key strings are from Function.FullName().
// That little dot ۰ is an Arabic zero numeral (U+06F0), categories [Nd].
var externals = map[string]externalFn{
	"(reflect.Value).CanAddr":         ext۰reflect۰Value۰CanAddr,
	"(reflect.Value).CanInterface":    ext۰reflect۰Value۰CanInterface,
	"(reflect.Value).Elem":            ext۰reflect۰Value۰Elem,
	"(reflect.Value).Field":           ext۰reflect۰Value۰Field,
	"(reflect.Value).Index":           ext۰reflect۰Value۰Index,
	"(reflect.Value).Int":             ext۰reflect۰Value۰Int,
	"(reflect.Value).Interface":       ext۰reflect۰Value۰Interface,
	"(reflect.Value).IsNil":           ext۰reflect۰Value۰IsNil,
	"(reflect.Value).IsValid":         ext۰reflect۰Value۰IsValid,
	"(reflect.Value).Kind":            ext۰reflect۰Value۰Kind,
	"(reflect.Value).Len":             ext۰reflect۰Value۰Len,
	"(reflect.Value).NumField":        ext۰reflect۰Value۰NumField,
	"(reflect.Value).Pointer":         ext۰reflect۰Value۰Pointer,
	"(reflect.Value).String":          ext۰reflect۰Value۰String,
	"(reflect.Value).Type":            ext۰reflect۰Value۰Type,
	"(reflect.rtype).Bits":            ext۰reflect۰rtype۰Bits,
	"(reflect.rtype).Elem":            ext۰reflect۰rtype۰Elem,
	"(reflect.rtype).Kind":            ext۰reflect۰rtype۰Kind,
	"(reflect.rtype).String":          ext۰reflect۰rtype۰String,
	"math.Float32bits":                ext۰math۰Float32bits,
	"math.Float32frombits":            ext۰math۰Float32frombits,
	"math.Float64bits":                ext۰math۰Float64bits,
	"math.Float64frombits":            ext۰math۰Float64frombits,
	"reflect.TypeOf":                  ext۰reflect۰TypeOf,
	"reflect.ValueOf":                 ext۰reflect۰ValueOf,
	"reflect.init":                    ext۰reflect۰Init,
	"reflect.valueInterface":          ext۰reflect۰valueInterface,
	"runtime.Breakpoint":              ext۰runtime۰Breakpoint,
	"runtime.GC":                      ext۰runtime۰GC,
	"runtime.GOMAXPROCS":              ext۰runtime۰GOMAXPROCS,
	"runtime.Gosched":                 ext۰runtime۰Gosched,
	"runtime.ReadMemStats":            ext۰runtime۰ReadMemStats,
	"runtime.SetFinalizer":            ext۰runtime۰SetFinalizer,
	"runtime.getgoroot":               ext۰runtime۰getgoroot,
	"sync/atomic.AddInt32":            ext۰atomic۰AddInt32,
	"sync/atomic.CompareAndSwapInt32": ext۰atomic۰CompareAndSwapInt32,
	"sync/atomic.LoadInt32":           ext۰atomic۰LoadInt32,
	"sync/atomic.LoadUint32":          ext۰atomic۰LoadUint32,
	"sync/atomic.StoreInt32":          ext۰atomic۰StoreInt32,
	"sync/atomic.StoreUint32":         ext۰atomic۰StoreUint32,
	"syscall.Exit":                    ext۰syscall۰Exit,
	"syscall.Getpid":                  ext۰syscall۰Getpid,
	"syscall.Kill":                    ext۰syscall۰Kill,
	"syscall.Write":                   ext۰syscall۰Write,
	"time.Sleep":                      ext۰time۰Sleep,
	"time.now":                        ext۰time۰now,
}

func ext۰math۰Float64frombits(fn *ssa.Function, args []value) value {
	return math.Float64frombits(args[0].(uint64))
}

func ext۰math۰Float64bits(fn *ssa.Function, args []value) value {
	return math.Float64bits(args[0].(float64))
}

func ext۰math۰Float32frombits(fn *ssa.Function, args []value) value {
	return math.Float32frombits(args[0].(uint32))
}

func ext۰math۰Float32bits(fn *ssa.Function, args []value) value {
	return math.Float32bits(args[0].(float32))
}

func ext۰runtime۰Breakpoint(fn *ssa.Function, args []value) value {
	runtime.Breakpoint()
	return nil
}

func ext۰runtime۰getgoroot(fn *ssa.Function, args []value) value {
	return os.Getenv("GOROOT")
}

func ext۰runtime۰GOMAXPROCS(fn *ssa.Function, args []value) value {
	return runtime.GOMAXPROCS(args[0].(int))
}

func ext۰runtime۰GC(fn *ssa.Function, args []value) value {
	runtime.GC()
	return nil
}

func ext۰runtime۰Gosched(fn *ssa.Function, args []value) value {
	runtime.Gosched()
	return nil
}

func ext۰runtime۰ReadMemStats(fn *ssa.Function, args []value) value {
	// TODO(adonovan): populate args[0].(Struct)
	return nil
}

func ext۰atomic۰LoadUint32(fn *ssa.Function, args []value) value {
	// TODO(adonovan): fix: not atomic!
	return (*args[0].(*value)).(uint32)
}

func ext۰atomic۰StoreUint32(fn *ssa.Function, args []value) value {
	// TODO(adonovan): fix: not atomic!
	*args[0].(*value) = args[1].(uint32)
	return nil
}

func ext۰atomic۰LoadInt32(fn *ssa.Function, args []value) value {
	// TODO(adonovan): fix: not atomic!
	return (*args[0].(*value)).(int32)
}

func ext۰atomic۰StoreInt32(fn *ssa.Function, args []value) value {
	// TODO(adonovan): fix: not atomic!
	*args[0].(*value) = args[1].(int32)
	return nil
}

func ext۰atomic۰CompareAndSwapInt32(fn *ssa.Function, args []value) value {
	// TODO(adonovan): fix: not atomic!
	p := args[0].(*value)
	if (*p).(int32) == args[1].(int32) {
		*p = args[2].(int32)
		return true
	}
	return false
}

func ext۰atomic۰AddInt32(fn *ssa.Function, args []value) value {
	// TODO(adonovan): fix: not atomic!
	p := args[0].(*value)
	newv := (*p).(int32) + args[1].(int32)
	*p = newv
	return newv
}

func ext۰runtime۰SetFinalizer(fn *ssa.Function, args []value) value {
	return nil // ignore
}

func ext۰time۰now(fn *ssa.Function, args []value) value {
	nano := time.Now().UnixNano()
	return tuple{int64(nano / 1e9), int32(nano % 1e9)}
}

func ext۰time۰Sleep(fn *ssa.Function, args []value) value {
	time.Sleep(time.Duration(args[0].(int64)))
	return nil
}

func ext۰syscall۰Exit(fn *ssa.Function, args []value) value {
	panic(exitPanic(args[0].(int)))
}

func ext۰syscall۰Getpid(fn *ssa.Function, args []value) value {
	// We could emulate syscall.Syscall but it's more effort.
	return syscall.Getpid()
}

// The set of remaining native functions we need to implement (as needed):

// bytes/bytes.go:42:func Equal(a, b []byte) bool
// bytes/bytes_decl.go:8:func IndexByte(s []byte, c byte) int // asm_$GOARCH.s
// crypto/aes/cipher_asm.go:10:func hasAsm() bool
// crypto/aes/cipher_asm.go:11:func encryptBlockAsm(nr int, xk *uint32, dst, src *byte)
// crypto/aes/cipher_asm.go:12:func decryptBlockAsm(nr int, xk *uint32, dst, src *byte)
// crypto/aes/cipher_asm.go:13:func expandKeyAsm(nr int, key *byte, enc *uint32, dec *uint32)
// hash/crc32/crc32_amd64.go:12:func haveSSE42() bool
// hash/crc32/crc32_amd64.go:16:func castagnoliSSE42(crc uint32, p []byte) uint32
// math/abs.go:12:func Abs(x float64) float64
// math/asin.go:19:func Asin(x float64) float64
// math/asin.go:51:func Acos(x float64) float64
// math/atan.go:95:func Atan(x float64) float64
// math/atan2.go:29:func Atan2(y, x float64) float64
// math/big/arith_decl.go:8:func mulWW(x, y Word) (z1, z0 Word)
// math/big/arith_decl.go:9:func divWW(x1, x0, y Word) (q, r Word)
// math/big/arith_decl.go:10:func addVV(z, x, y []Word) (c Word)
// math/big/arith_decl.go:11:func subVV(z, x, y []Word) (c Word)
// math/big/arith_decl.go:12:func addVW(z, x []Word, y Word) (c Word)
// math/big/arith_decl.go:13:func subVW(z, x []Word, y Word) (c Word)
// math/big/arith_decl.go:14:func shlVU(z, x []Word, s uint) (c Word)
// math/big/arith_decl.go:15:func shrVU(z, x []Word, s uint) (c Word)
// math/big/arith_decl.go:16:func mulAddVWW(z, x []Word, y, r Word) (c Word)
// math/big/arith_decl.go:17:func addMulVVW(z, x []Word, y Word) (c Word)
// math/big/arith_decl.go:18:func divWVW(z []Word, xn Word, x []Word, y Word) (r Word)
// math/big/arith_decl.go:19:func bitLen(x Word) (n int)
// math/dim.go:13:func Dim(x, y float64) float64
// math/dim.go:26:func Max(x, y float64) float64
// math/dim.go:53:func Min(x, y float64) float64
// math/exp.go:14:func Exp(x float64) float64
// math/exp.go:135:func Exp2(x float64) float64
// math/expm1.go:124:func Expm1(x float64) float64
// math/floor.go:13:func Floor(x float64) float64
// math/floor.go:36:func Ceil(x float64) float64
// math/floor.go:48:func Trunc(x float64) float64
// math/frexp.go:16:func Frexp(f float64) (frac float64, exp int)
// math/hypot.go:17:func Hypot(p, q float64) float64
// math/ldexp.go:14:func Ldexp(frac float64, exp int) float64
// math/log.go:80:func Log(x float64) float64
// math/log10.go:9:func Log10(x float64) float64
// math/log10.go:17:func Log2(x float64) float64
// math/log1p.go:95:func Log1p(x float64) float64
// math/mod.go:21:func Mod(x, y float64) float64
// math/modf.go:13:func Modf(f float64) (int float64, frac float64)
// math/remainder.go:37:func Remainder(x, y float64) float64
// math/sin.go:117:func Cos(x float64) float64
// math/sin.go:174:func Sin(x float64) float64
// math/sincos.go:15:func Sincos(x float64) (sin, cos float64)
// math/sqrt.go:14:func Sqrt(x float64) float64
// math/tan.go:82:func Tan(x float64) float64
// os/file_posix.go:14:func sigpipe() // implemented in package runtime
// os/signal/signal_unix.go:15:func signal_enable(uint32)
// os/signal/signal_unix.go:16:func signal_recv() uint32
// runtime/debug.go:13:func LockOSThread()
// runtime/debug.go:17:func UnlockOSThread()
// runtime/debug.go:27:func NumCPU() int
// runtime/debug.go:30:func NumCgoCall() int64
// runtime/debug.go:33:func NumGoroutine() int
// runtime/debug.go:90:func MemProfile(p []MemProfileRecord, inuseZero bool) (n int, ok bool)
// runtime/debug.go:114:func ThreadCreateProfile(p []StackRecord) (n int, ok bool)
// runtime/debug.go:122:func GoroutineProfile(p []StackRecord) (n int, ok bool)
// runtime/debug.go:132:func CPUProfile() []byte
// runtime/debug.go:141:func SetCPUProfileRate(hz int)
// runtime/debug.go:149:func SetBlockProfileRate(rate int)
// runtime/debug.go:166:func BlockProfile(p []BlockProfileRecord) (n int, ok bool)
// runtime/debug.go:172:func Stack(buf []byte, all bool) int
// runtime/error.go:81:func typestring(interface{}) string
// runtime/extern.go:19:func Goexit()
// runtime/extern.go:27:func Caller(skip int) (pc uintptr, file string, line int, ok bool)
// runtime/extern.go:34:func Callers(skip int, pc []uintptr) int
// runtime/extern.go:51:func FuncForPC(pc uintptr) *Func
// runtime/extern.go:68:func funcline_go(*Func, uintptr) (string, int)
// runtime/extern.go:71:func mid() uint32
// runtime/pprof/pprof.go:667:func runtime_cyclesPerSecond() int64
// runtime/race.go:16:func RaceDisable()
// runtime/race.go:19:func RaceEnable()
// runtime/race.go:21:func RaceAcquire(addr unsafe.Pointer)
// runtime/race.go:22:func RaceRelease(addr unsafe.Pointer)
// runtime/race.go:23:func RaceReleaseMerge(addr unsafe.Pointer)
// runtime/race.go:25:func RaceRead(addr unsafe.Pointer)
// runtime/race.go:26:func RaceWrite(addr unsafe.Pointer)
// runtime/race.go:28:func RaceSemacquire(s *uint32)
// runtime/race.go:29:func RaceSemrelease(s *uint32)
// sync/atomic/doc.go:49:func CompareAndSwapInt64(addr *int64, old, new int64) (swapped bool)
// sync/atomic/doc.go:52:func CompareAndSwapUint32(addr *uint32, old, new uint32) (swapped bool)
// sync/atomic/doc.go:55:func CompareAndSwapUint64(addr *uint64, old, new uint64) (swapped bool)
// sync/atomic/doc.go:58:func CompareAndSwapUintptr(addr *uintptr, old, new uintptr) (swapped bool)
// sync/atomic/doc.go:61:func CompareAndSwapPointer(addr *unsafe.Pointer, old, new unsafe.Pointer) (swapped bool)
// sync/atomic/doc.go:67:func AddUint32(addr *uint32, delta uint32) (new uint32)
// sync/atomic/doc.go:70:func AddInt64(addr *int64, delta int64) (new int64)
// sync/atomic/doc.go:73:func AddUint64(addr *uint64, delta uint64) (new uint64)
// sync/atomic/doc.go:76:func AddUintptr(addr *uintptr, delta uintptr) (new uintptr)
// sync/atomic/doc.go:82:func LoadInt64(addr *int64) (val int64)
// sync/atomic/doc.go:88:func LoadUint64(addr *uint64) (val uint64)
// sync/atomic/doc.go:91:func LoadUintptr(addr *uintptr) (val uintptr)
// sync/atomic/doc.go:94:func LoadPointer(addr *unsafe.Pointer) (val unsafe.Pointer)
// sync/atomic/doc.go:100:func StoreInt64(addr *int64, val int64)
// sync/atomic/doc.go:106:func StoreUint64(addr *uint64, val uint64)
// sync/atomic/doc.go:109:func StoreUintptr(addr *uintptr, val uintptr)
// sync/atomic/doc.go:112:func StorePointer(addr *unsafe.Pointer, val unsafe.Pointer)
// sync/runtime.go:12:func runtime_Semacquire(s *uint32)
// sync/runtime.go:18:func runtime_Semrelease(s *uint32)
// syscall/env_unix.go:30:func setenv_c(k, v string)
// syscall/syscall_linux_amd64.go:60:func Gettimeofday(tv *Timeval) (err error)
// syscall/syscall_linux_amd64.go:61:func Time(t *Time_t) (tt Time_t, err error)
// syscall/syscall_linux_arm.go:28:func Seek(fd int, offset int64, whence int) (newoffset int64, err error)
// syscall/syscall_unix.go:23:func Syscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno)
// syscall/syscall_unix.go:24:func Syscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)
// syscall/syscall_unix.go:25:func RawSyscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno)
// syscall/syscall_unix.go:26:func RawSyscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)
// time/sleep.go:25:func startTimer(*runtimeTimer)
// time/sleep.go:26:func stopTimer(*runtimeTimer) bool
// time/time.go:758:func now() (sec int64, nsec int32)
// unsafe/unsafe.go:27:func Sizeof(v ArbitraryType) uintptr
// unsafe/unsafe.go:32:func Offsetof(v ArbitraryType) uintptr
// unsafe/unsafe.go:37:func Alignof(v ArbitraryType) uintptr
