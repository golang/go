// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"bufio"
	"internal/buildcfg"
	"internal/testenv"
	"io"
	"math/bits"
	"os/exec"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

// TestIntendedInlining tests that specific functions are inlined.
// This allows refactoring for code clarity and re-use without fear that
// changes to the compiler will cause silent performance regressions.
func TestIntendedInlining(t *testing.T) {
	if testing.Short() && testenv.Builder() == "" {
		t.Skip("skipping in short mode")
	}
	testenv.MustHaveGoRun(t)
	t.Parallel()

	// want is the list of function names (by package) that should
	// be inlinable. If they have no callers in their packages, they
	// might not actually be inlined anywhere.
	want := map[string][]string{
		"runtime": {
			"add",
			"acquirem",
			"add1",
			"addb",
			"adjustpanics",
			"adjustpointer",
			"alignDown",
			"alignUp",
			"bucketMask",
			"bucketShift",
			"chanbuf",
			"evacuated",
			"fastlog2",
			"fastrand",
			"float64bits",
			"funcspdelta",
			"getm",
			"getMCache",
			"isDirectIface",
			"itabHashFunc",
			"noescape",
			"pcvalueCacheKey",
			"readUnaligned32",
			"readUnaligned64",
			"releasem",
			"roundupsize",
			"stackmapdata",
			"stringStructOf",
			"subtract1",
			"subtractb",
			"tophash",
			"(*bmap).keys",
			"(*bmap).overflow",
			"(*waitq).enqueue",
			"funcInfo.entry",

			// GC-related ones
			"cgoInRange",
			"gclinkptr.ptr",
			"guintptr.ptr",
			"writeHeapBitsForAddr",
			"markBits.isMarked",
			"muintptr.ptr",
			"puintptr.ptr",
			"spanOf",
			"spanOfUnchecked",
			"(*gcWork).putFast",
			"(*gcWork).tryGetFast",
			"(*guintptr).set",
			"(*markBits).advance",
			"(*mspan).allocBitsForIndex",
			"(*mspan).base",
			"(*mspan).markBitsForBase",
			"(*mspan).markBitsForIndex",
			"(*muintptr).set",
			"(*puintptr).set",
		},
		"runtime/internal/sys": {},
		"runtime/internal/math": {
			"MulUintptr",
		},
		"bytes": {
			"(*Buffer).Bytes",
			"(*Buffer).Cap",
			"(*Buffer).Len",
			"(*Buffer).Grow",
			"(*Buffer).Next",
			"(*Buffer).Read",
			"(*Buffer).ReadByte",
			"(*Buffer).Reset",
			"(*Buffer).String",
			"(*Buffer).UnreadByte",
			"(*Buffer).tryGrowByReslice",
		},
		"compress/flate": {
			"byLiteral.Len",
			"byLiteral.Less",
			"byLiteral.Swap",
			"(*dictDecoder).tryWriteCopy",
		},
		"encoding/base64": {
			"assemble32",
			"assemble64",
		},
		"unicode/utf8": {
			"FullRune",
			"FullRuneInString",
			"RuneLen",
			"AppendRune",
			"ValidRune",
		},
		"reflect": {
			"Value.Bool",
			"Value.Bytes",
			"Value.CanAddr",
			"Value.CanComplex",
			"Value.CanFloat",
			"Value.CanInt",
			"Value.CanInterface",
			"Value.CanSet",
			"Value.CanUint",
			"Value.Cap",
			"Value.Complex",
			"Value.Float",
			"Value.Int",
			"Value.Interface",
			"Value.IsNil",
			"Value.IsValid",
			"Value.Kind",
			"Value.Len",
			"Value.MapRange",
			"Value.OverflowComplex",
			"Value.OverflowFloat",
			"Value.OverflowInt",
			"Value.OverflowUint",
			"Value.String",
			"Value.Type",
			"Value.Uint",
			"Value.UnsafeAddr",
			"Value.pointer",
			"add",
			"align",
			"flag.mustBe",
			"flag.mustBeAssignable",
			"flag.mustBeExported",
			"flag.kind",
			"flag.ro",
		},
		"regexp": {
			"(*bitState).push",
		},
		"math/big": {
			"bigEndianWord",
			// The following functions require the math_big_pure_go build tag.
			"addVW",
			"subVW",
		},
		"math/rand": {
			"(*rngSource).Int63",
			"(*rngSource).Uint64",
		},
		"net": {
			"(*UDPConn).ReadFromUDP",
		},
		"sync/atomic": {
			// (*Bool).CompareAndSwap handled below.
			"(*Bool).Load",
			"(*Bool).Store",
			"(*Bool).Swap",
			"(*Int32).Add",
			"(*Int32).CompareAndSwap",
			"(*Int32).Load",
			"(*Int32).Store",
			"(*Int32).Swap",
			"(*Int64).Add",
			"(*Int64).CompareAndSwap",
			"(*Int64).Load",
			"(*Int64).Store",
			"(*Int64).Swap",
			"(*Uint32).Add",
			"(*Uint32).CompareAndSwap",
			"(*Uint32).Load",
			"(*Uint32).Store",
			"(*Uint32).Swap",
			"(*Uint64).Add",
			"(*Uint64).CompareAndSwap",
			"(*Uint64).Load",
			"(*Uint64).Store",
			"(*Uint64).Swap",
			"(*Uintptr).Add",
			"(*Uintptr).CompareAndSwap",
			"(*Uintptr).Load",
			"(*Uintptr).Store",
			"(*Uintptr).Swap",
			// (*Pointer[T])'s methods' handled below.
		},
	}

	if runtime.GOARCH != "386" && runtime.GOARCH != "loong64" && runtime.GOARCH != "mips64" && runtime.GOARCH != "mips64le" && runtime.GOARCH != "riscv64" {
		// nextFreeFast calls sys.Ctz64, which on 386 is implemented in asm and is not inlinable.
		// We currently don't have midstack inlining so nextFreeFast is also not inlinable on 386.
		// On loong64, mips64x and riscv64, Ctz64 is not intrinsified and causes nextFreeFast too expensive
		// to inline (Issue 22239).
		want["runtime"] = append(want["runtime"], "nextFreeFast")
		// Same behavior for heapBits.nextFast.
		want["runtime"] = append(want["runtime"], "heapBits.nextFast")
	}
	if runtime.GOARCH != "386" {
		// As explained above, Ctz64 and Ctz32 are not Go code on 386.
		// The same applies to Bswap32.
		want["runtime/internal/sys"] = append(want["runtime/internal/sys"], "Ctz64")
		want["runtime/internal/sys"] = append(want["runtime/internal/sys"], "Ctz32")
		want["runtime/internal/sys"] = append(want["runtime/internal/sys"], "Bswap32")
	}
	if bits.UintSize == 64 {
		// mix is only defined on 64-bit architectures
		want["runtime"] = append(want["runtime"], "mix")
		// (*Bool).CompareAndSwap is just over budget on 32-bit systems (386, arm).
		want["sync/atomic"] = append(want["sync/atomic"], "(*Bool).CompareAndSwap")
	}
	if buildcfg.Experiment.Unified {
		// Non-unified IR does not report "inlining call ..." for atomic.Pointer[T]'s methods.
		// TODO(cuonglm): remove once non-unified IR frontend gone.
		want["sync/atomic"] = append(want["sync/atomic"], "(*Pointer[go.shape.int]).CompareAndSwap")
		want["sync/atomic"] = append(want["sync/atomic"], "(*Pointer[go.shape.int]).Load")
		want["sync/atomic"] = append(want["sync/atomic"], "(*Pointer[go.shape.int]).Store")
		want["sync/atomic"] = append(want["sync/atomic"], "(*Pointer[go.shape.int]).Swap")
	}

	switch runtime.GOARCH {
	case "386", "wasm", "arm":
	default:
		// TODO(mvdan): As explained in /test/inline_sync.go, some
		// architectures don't have atomic intrinsics, so these go over
		// the inlining budget. Move back to the main table once that
		// problem is solved.
		want["sync"] = []string{
			"(*Mutex).Lock",
			"(*Mutex).Unlock",
			"(*RWMutex).RLock",
			"(*RWMutex).RUnlock",
			"(*Once).Do",
		}
	}

	// Functions that must actually be inlined; they must have actual callers.
	must := map[string]bool{
		"compress/flate.byLiteral.Len":  true,
		"compress/flate.byLiteral.Less": true,
		"compress/flate.byLiteral.Swap": true,
	}

	notInlinedReason := make(map[string]string)
	pkgs := make([]string, 0, len(want))
	for pname, fnames := range want {
		pkgs = append(pkgs, pname)
		for _, fname := range fnames {
			fullName := pname + "." + fname
			if _, ok := notInlinedReason[fullName]; ok {
				t.Errorf("duplicate func: %s", fullName)
			}
			notInlinedReason[fullName] = "unknown reason"
		}
	}

	args := append([]string{"build", "-gcflags=-m -m", "-tags=math_big_pure_go"}, pkgs...)
	cmd := testenv.CleanCmdEnv(exec.Command(testenv.GoToolPath(t), args...))
	pr, pw := io.Pipe()
	cmd.Stdout = pw
	cmd.Stderr = pw
	cmdErr := make(chan error, 1)
	go func() {
		cmdErr <- cmd.Run()
		pw.Close()
	}()
	scanner := bufio.NewScanner(pr)
	curPkg := ""
	canInline := regexp.MustCompile(`: can inline ([^ ]*)`)
	haveInlined := regexp.MustCompile(`: inlining call to ([^ ]*)`)
	cannotInline := regexp.MustCompile(`: cannot inline ([^ ]*): (.*)`)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "# ") {
			curPkg = line[2:]
			continue
		}
		if m := haveInlined.FindStringSubmatch(line); m != nil {
			fname := m[1]
			delete(notInlinedReason, curPkg+"."+fname)
			continue
		}
		if m := canInline.FindStringSubmatch(line); m != nil {
			fname := m[1]
			fullname := curPkg + "." + fname
			// If function must be inlined somewhere, being inlinable is not enough
			if _, ok := must[fullname]; !ok {
				delete(notInlinedReason, fullname)
				continue
			}
		}
		if m := cannotInline.FindStringSubmatch(line); m != nil {
			fname, reason := m[1], m[2]
			fullName := curPkg + "." + fname
			if _, ok := notInlinedReason[fullName]; ok {
				// cmd/compile gave us a reason why
				notInlinedReason[fullName] = reason
			}
			continue
		}
	}
	if err := <-cmdErr; err != nil {
		t.Fatal(err)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	for fullName, reason := range notInlinedReason {
		t.Errorf("%s was not inlined: %s", fullName, reason)
	}
}
