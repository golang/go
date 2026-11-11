// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"bufio"
	"cmd/internal/sys"
	"internal/testenv"
	"io"
	"math/bits"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

// TestIntendedInlining tests that specific functions can be inlined,
// as determined by parsing the compiler's -m diagnostic output when
// building each function's package. This allows refactoring for code
// clarity and re-use without fear that changes to the compiler will
// cause silent performance regressions.
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
			"chanbuf",
			"fastlog2",
			"float64bits",
			"funcspdelta",
			"getm",
			"getMCache",
			"heapSetTypeNoHeader",
			"heapSetTypeSmallHeader",
			"itabHashFunc",
			"nextslicecap",
			"noescape",
			"pcvalueCacheKey",
			"rand32",
			"readUnaligned64",
			"releasem",
			"roundupsize",
			"stackmapdata",
			"stringStructOf",
			"subtract1",
			"subtractb",
			"(*waitq).enqueue",
			"funcInfo.entry",

			// GC-related ones
			"cgoInRange",
			"gclinkptr.ptr",
			"gcUsesSpanInlineMarkBits",
			"guintptr.ptr",
			"heapBitsSlice",
			"markBits.isMarked",
			"muintptr.ptr",
			"puintptr.ptr",
			"spanHeapBitsRange",
			"spanOf",
			"spanOfUnchecked",
			"typePointers.nextFast",
			"(*gcWork).putObjFast",
			"(*gcWork).tryGetObjFast",
			"(*guintptr).set",
			"(*markBits).advance",
			"(*mspan).allocBitsForIndex",
			"(*mspan).base",
			"(*mspan).markBitsForBase",
			"(*mspan).markBitsForIndex",
			"(*mspan).writeUserArenaHeapBits",
			"(*muintptr).set",
			"(*puintptr).set",
			"(*wbBuf).get1",
			"(*wbBuf).get2",

			// Trace-related ones.
			"traceLocker.ok",
			"traceEnabled",
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
		"internal/abi": {
			"(*Type).IsDirectIface",
			"UseInterfaceSwitchCache",
		},
		"internal/runtime/math": {
			"MulUintptr",
		},
		"internal/runtime/maps": {
			"readUnaligned32",
			"readUnaligned64",
		},
		"internal/runtime/sys": {},
		"compress/flate": {
			"(*dictDecoder).tryWriteCopy",
		},
		"encoding/base64": {
			"assemble32",
			"assemble64",
		},
		"unicode/utf8": {
			"DecodeRune",
			"DecodeRuneInString",
			"FullRune",
			"FullRuneInString",
			"RuneLen",
			"AppendRune",
			"ValidRune",
		},
		"unicode/utf16": {
			"Decode",
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
		"sort": {
			"Find",
			"Search",
		},
		"math/big": {
			"bigEndianWord",
		},
		"math/rand": {
			"(*rngSource).Int63",
			"(*rngSource).Uint64",
		},
		"net": {
			"(*UDPConn).ReadFromUDP",
		},
		"sync": {
			// Both OnceFunc and its returned closure need to be inlinable so
			// that the returned closure can be inlined into the caller of OnceFunc.
			"OnceFunc",
			"OnceFunc.func1", // The returned closure.
			// TODO(austin): It would be good to check OnceValue and OnceValues,
			// too, but currently they aren't reported because they have type
			// parameters and aren't instantiated in sync.
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
			"(*Pointer[go.shape.int]).CompareAndSwap",
			"(*Pointer[go.shape.int]).Load",
			"(*Pointer[go.shape.int]).Store",
			"(*Pointer[go.shape.int]).Swap",
		},
		"testing": {
			"(*B).Loop",
		},
		"time": {
			"Duration.String",
		},
		"uuid": {
			"UUID.String",
		},
		"path": {
			"Base",
			"scanChunk",
		},
		"path/filepath": {
			"scanChunk",
		},
		"internal/strconv": {
			"prescale",
			"uscale",
		},
	}

	if runtime.GOARCH != "386" && runtime.GOARCH != "loong64" && runtime.GOARCH != "mips64" && runtime.GOARCH != "mips64le" && runtime.GOARCH != "riscv64" {
		// nextFreeFast calls sys.TrailingZeros64, which on 386 is implemented in asm and is not inlinable.
		// We currently don't have midstack inlining so nextFreeFast is also not inlinable on 386.
		// On loong64, mips64x and riscv64, TrailingZeros64 is not intrinsified and causes nextFreeFast
		// too expensive to inline (Issue 22239).
		want["runtime"] = append(want["runtime"], "nextFreeFast")
	}
	if runtime.GOARCH != "386" {
		// As explained above, TrailingZeros64 and TrailingZeros32 are not Go code on 386.
		// The same applies to Bswap32.
		want["internal/runtime/sys"] = append(want["internal/runtime/sys"], "TrailingZeros64")
		want["internal/runtime/sys"] = append(want["internal/runtime/sys"], "TrailingZeros32")
		want["internal/runtime/sys"] = append(want["internal/runtime/sys"], "Bswap32")
	}
	if runtime.GOARCH == "amd64" || runtime.GOARCH == "arm64" || runtime.GOARCH == "loong64" || runtime.GOARCH == "mips" || runtime.GOARCH == "mips64" || runtime.GOARCH == "ppc64" || runtime.GOARCH == "riscv64" || runtime.GOARCH == "s390x" {
		// internal/runtime/atomic.Loaduintptr is only intrinsified on these platforms.
		want["runtime"] = append(want["runtime"], "traceAcquire")
	}
	if bits.UintSize == 64 {
		// mix is only defined on 64-bit architectures
		want["internal/runtime/maps"] = append(want["internal/runtime/maps"], "mix")
		// (*Bool).CompareAndSwap is just over budget on 32-bit systems (386, arm).
		want["sync/atomic"] = append(want["sync/atomic"], "(*Bool).CompareAndSwap")
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

	if runtime.GOARCH != "wasm" {
		// mutex implementation for multi-threaded GOARCHes
		want["runtime"] = append(want["runtime"],
			// in the fast paths of lock2 and unlock2
			"key8",
			"(*mLockProfile).store",
		)
		if bits.UintSize == 64 {
			// these use 64-bit arithmetic, which is hard to inline on 32-bit platforms
			want["runtime"] = append(want["runtime"],
				// in the fast paths of lock2 and unlock2
				"mutexSampleContention",

				// in a slow path of lock2, but within the critical section
				"(*mLockProfile).end",
			)
		}
	}

	// Functions that must actually be inlined; they must have actual callers.
	must := map[string]bool{}

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
	cmd := testenv.CleanCmdEnv(testenv.Command(t, testenv.GoToolPath(t), args...))
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

func collectInlCands(msgs string) map[string]struct{} {
	rv := make(map[string]struct{})
	lines := strings.Split(msgs, "\n")
	re := regexp.MustCompile(`^\S+\s+can\s+inline\s+(\S+)`)
	for _, line := range lines {
		m := re.FindStringSubmatch(line)
		if m != nil {
			rv[m[1]] = struct{}{}
		}
	}
	return rv
}

func TestIssue56044(t *testing.T) {
	if testing.Short() {
		t.Skipf("skipping test: too long for short mode")
	}
	testenv.MustHaveGoBuild(t)

	modes := []string{"-covermode=set", "-covermode=atomic"}

	for _, mode := range modes {
		// Build the Go runtime with "-m", capturing output.
		args := []string{"build", "-gcflags=runtime=-m", "runtime"}
		cmd := testenv.Command(t, testenv.GoToolPath(t), args...)
		b, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("build failed (%v): %s", err, b)
		}
		mbase := collectInlCands(string(b))

		// Redo the build with -cover, also with "-m".
		args = []string{"build", "-gcflags=runtime=-m", mode, "runtime"}
		cmd = testenv.Command(t, testenv.GoToolPath(t), args...)
		b, err = cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("build failed (%v): %s", err, b)
		}
		mcov := collectInlCands(string(b))

		// Make sure that there aren't any functions that are marked
		// as inline candidates at base but not with coverage.
		for k := range mbase {
			if _, ok := mcov[k]; !ok {
				t.Errorf("error: did not find %s in coverage -m output", k)
			}
		}
	}
}

func TestByteOrderInline(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	for _, a := range sys.Archs {
		if a.Name == runtime.GOARCH && !a.CanMergeLoads {
			t.Skipf("%s does not merge loads", runtime.GOARCH)
		}
	}

	cmd := testenv.CleanCmdEnv(testenv.Command(t, testenv.GoToolPath(t), "build",
		"-gcflags=encoding/binary=-m=2",
		"-gcflags=internal/byteorder=-m=2",
		"encoding/binary", "internal/byteorder",
	))
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build failed: %v\n%s", err, out)
	}
	for _, prefix := range []string{"littleEndian.", "bigEndian.", "LE", "BE"} {
		for _, op := range []string{"Uint16", "Uint32", "Uint64", "PutUint16", "PutUint32", "PutUint64"} {
			want := "can inline " + prefix + op + " with cost 1 "
			if !strings.Contains(string(out), want) {
				t.Errorf("missing %q in -m=2 output", want)
			}
		}
		for _, op := range []string{"AppendUint16", "AppendUint32", "AppendUint64"} {
			unwanted := "can inline " + prefix + op + " with cost 1 "
			if strings.Contains(string(out), unwanted) {
				t.Errorf("unexpected %q in -m=2 output", unwanted)
			}
		}
	}

	dir := t.TempDir()
	src := filepath.Join(dir, "p.go")
	const marker = "// big calls"
	code := `package p
import "encoding/binary"
func wrap(b []byte) uint64 { return binary.LittleEndian.Uint64(b) + binary.BigEndian.Uint64(b) }
func medium(b []byte) uint64 {
	return uint64(b[0]) + uint64(b[1]) + uint64(b[2]) + uint64(b[3]) + uint64(b[4]) + uint64(b[5]) + uint64(b[6]) + uint64(b[7])
}
func big(b []byte) ([]byte, uint64) {
	var x byte
` + strings.Repeat("\tx += b[0]\n", 1700) + "\t" + marker + `
	v := uint64(x) + medium(b) + wrap(b) + uint64(binary.LittleEndian.Uint32(b))
	binary.LittleEndian.PutUint64(b, v)
	b = binary.LittleEndian.AppendUint64(b, v)
	return b, v
}
`
	if err := os.WriteFile(src, []byte(code), 0644); err != nil {
		t.Fatal(err)
	}
	cmd = testenv.CleanCmdEnv(testenv.Command(t, testenv.GoToolPath(t), "build", "-gcflags=-m=2", "-o", filepath.Join(dir, "p.a"), src))
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("compile p.go failed: %v\n%s", err, out)
	}
	s := string(out)
	if !strings.Contains(s, "function big considered 'big'") {
		t.Fatalf("big was not considered 'big':\n%s", s)
	}

	// Collect inlined calls originating at or after the marker inside big,
	// excluding wrap's own standalone compilation earlier in the file.
	bigCallLine := 1 + strings.Count(code[:strings.Index(code, marker)], "\n")
	inlRE := regexp.MustCompile(`p\.go:(\d+):\d+: inlining call to (\S+)`)
	bigInlined := make(map[string]bool)
	for _, m := range inlRE.FindAllStringSubmatch(s, -1) {
		line, err := strconv.Atoi(m[1])
		if err == nil && line >= bigCallLine {
			bigInlined[m[2]] = true
		}
	}
	for _, fn := range []string{"medium", "binary.littleEndian.AppendUint64"} {
		if bigInlined[fn] {
			t.Errorf("%s unexpectedly inlined into big", fn)
		}
	}
	for _, fn := range []string{
		"wrap",
		"binary.littleEndian.Uint64",
		"binary.bigEndian.Uint64",
		"binary.littleEndian.Uint32",
		"binary.littleEndian.PutUint64",
	} {
		if !bigInlined[fn] {
			t.Errorf("expected %s to inline into big:\n%s", fn, s)
		}
	}
}
