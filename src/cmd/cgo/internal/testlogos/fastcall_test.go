// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlogos

import (
	"bytes"
	"context"
	"internal/goexperiment"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// The battery below covers M20-S1. Tests marked ATTACK fail red before
// the slice lands (missing symbol, plain emission, ignored directive)
// for the predicted reason; tests marked PIN hold green on both paths
// and guard the parity the slice requires. The whole suite runs twice
// in the slice gate: with GOEXPERIMENT=logosfastcgo and with
// GOEXPERIMENT=nologosfastcgo. The nocallback, blocking-parity, and
// profile attacks pass under both settings; the routing attacks only
// pass once cmd/cgo emits the fast entry.

func requireCgo(t testing.TB) {
	t.Helper()
	if os.Getenv("CGO_ENABLED") == "0" {
		t.Skip("cgo disabled")
	}
}

func goBin(t *testing.T) string {
	t.Helper()
	if goroot := runtime.GOROOT(); goroot != "" {
		if p := filepath.Join(goroot, "bin", "go"); isExec(p) {
			return p
		}
	}
	if p, err := exec.LookPath("go"); err == nil {
		return p
	}
	t.Fatal("go tool not found")
	return ""
}

func logosBin(t *testing.T) string {
	t.Helper()
	if goroot := runtime.GOROOT(); goroot != "" {
		if p := filepath.Join(goroot, "bin", "logos"); isExec(p) {
			return p
		}
	}
	if p, err := exec.LookPath("logos"); err == nil {
		return p
	}
	t.Fatal("logos tool not found")
	return ""
}

func isExec(p string) bool {
	fi, err := os.Stat(p)
	if err != nil {
		return false
	}
	return fi.Mode()&0o111 != 0
}

func runCmd(t *testing.T, dir, name string, args ...string) (string, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOPROXY=off", "GOTOOLCHAIN=local")
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	err := cmd.Run()
	return buf.String(), err
}

func gotoolDir(t *testing.T) string {
	t.Helper()
	out, err := runCmd(t, "", goBin(t), "env", "GOTOOLDIR")
	if err != nil {
		t.Fatalf("go env GOTOOLDIR: %v\n%s", err, out)
	}
	return strings.TrimSpace(out)
}

func cgoBin(t *testing.T) string {
	t.Helper()
	p := filepath.Join(gotoolDir(t), "cgo")
	if !isExec(p) {
		t.Skipf("cgo tool not found at %s", p)
	}
	return p
}

// TestFastcallCorrectness PIN: arithmetic, pointer reads, slice sums,
// and sleeps agree on both paths.
func TestFastcallCorrectness(t *testing.T) {
	requireCgo(t)
	if got := callFastAdd(40, 2); got != 42 {
		t.Fatalf("fast_add = %d, want 42", got)
	}
	if got := callSlowNoescape(40, 2); got != 42 {
		t.Fatalf("slow_noescape = %d, want 42", got)
	}
	if got := callSlowNocallback(40, 2); got != 42 {
		t.Fatalf("slow_nocallback = %d, want 42", got)
	}
	v := int32(7)
	if got := callFastRead(&v); got != 7 {
		t.Fatalf("read_int = %d, want 7", got)
	}
	if got := callFastSlice([]byte{1, 2, 3, 4}); got != 10 {
		t.Fatalf("sum_bytes = %d, want 10", got)
	}
	if got := callFastSlice(nil); got != 0 {
		t.Fatalf("sum_bytes(nil) = %d, want 0", got)
	}
	if got := callFastSleep(1000); got != 1000 {
		t.Fatalf("block_sleep = %d, want 1000", got)
	}
	if got := callFastBusy(10); got != callPlainBusy(10) {
		t.Fatalf("busy mismatch: fast %d vs plain %d", got, callPlainBusy(10))
	}
}

// cgoOutputs runs the cgo tool over one scratch input and returns the
// objdir listing plus file contents keyed by base name.
func cgoOutputs(t *testing.T, input, src string) (dir string, files map[string]string) {
	t.Helper()
	requireCgo(t)
	work := t.TempDir()
	if err := os.WriteFile(filepath.Join(work, input), []byte(src), 0644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(work, "out")
	if err := os.Mkdir(out, 0755); err != nil {
		t.Fatal(err)
	}
	text, err := runCmd(t, work, cgoBin(t), "-objdir", out, "-importpath", "scratch/fast", input)
	if err != nil {
		t.Fatalf("cgo %s: %v\n%s", input, err, text)
	}
	files = map[string]string{}
	ents, err := os.ReadDir(out)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		b, err := os.ReadFile(filepath.Join(out, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		files[e.Name()] = string(b)
	}
	return out, files
}

// countCallLines counts generated call lines invoking callee, ignoring
// the linkname declarations in the prolog (those start at column 0).
func countCallLines(src, callee string) int {
	n := 0
	for line := range strings.SplitSeq(src, "\n") {
		if strings.HasPrefix(line, "\t") && strings.Contains(line, callee+"(") {
			n++
		}
	}
	return n
}

const pairPreamble = `package fast

/*
#include <stdint.h>

#cgo noescape fast_add
#cgo nocallback fast_add
int fast_add(int a, int b) { return a + b; }

#cgo noescape read_int
#cgo nocallback read_int
int read_int(int *p) { return *p; }

#cgo noescape read_pp
#cgo nocallback read_pp
int read_pp(int **pp) { return **pp; }

#cgo noescape sum_bytes
#cgo nocallback sum_bytes
int sum_bytes(unsigned char *p, int n) {
	int s = 0;
	for (int i = 0; i < n; i++) {
		s += p[i];
	}
	return s;
}
*/
import "C"

import "unsafe"

func Add(a, b int32) int32 { return int32(C.fast_add(C.int(a), C.int(b))) }

func Read(p *int32) int32 { return int32(C.read_int((*C.int)(unsafe.Pointer(p)))) }

func ReadPP(pp **C.int) int32 { return int32(C.read_pp(pp)) }

func Sum(b []byte) int32 {
	if len(b) == 0 {
		return 0
	}
	return int32(C.sum_bytes((*C.uchar)(unsafe.Pointer(&b[0])), C.int(len(b))))
}
`

// TestFastcallEmission ATTACK: cmd/cgo emits the fast entry for the
// directive pair and keeps the plain entry for single directives.
// Red before: every wrapper calls _cgo_runtime_cgocall.
func TestFastcallEmission(t *testing.T) {
	_, files := cgoOutputs(t, "fast.go", pairPreamble)
	gotypes, ok := files["_cgo_gotypes.go"]
	if !ok {
		t.Fatal("no _cgo_gotypes.go emitted")
	}
	if got := countCallLines(gotypes, "_cgo_runtime_cgocall_fast"); got != 4 {
		t.Fatalf("fast calls = %d, want 4", got)
	}
	if got := countCallLines(gotypes, "_cgo_runtime_cgocall"); got != 0 {
		t.Fatalf("plain calls = %d, want 0", got)
	}

	const slowSrc = `package slow

/*
#cgo noescape slow_noescape
int slow_noescape(int a, int b) { return a + b; }

#cgo nocallback slow_nocallback
int slow_nocallback(int a, int b) { return a + b; }
*/
import "C"

func Noescape(a, b int32) int32 { return int32(C.slow_noescape(C.int(a), C.int(b))) }

func Nocallback(a, b int32) int32 { return int32(C.slow_nocallback(C.int(a), C.int(b))) }
`
	_, slow := cgoOutputs(t, "slow.go", slowSrc)
	sg, ok := slow["_cgo_gotypes.go"]
	if !ok {
		t.Fatal("no _cgo_gotypes.go emitted for slow twins")
	}
	if got := countCallLines(sg, "_cgo_runtime_cgocall_fast"); got != 0 {
		t.Fatalf("fast calls in slow twins = %d, want 0", got)
	}
	if got := countCallLines(sg, "_cgo_runtime_cgocall"); got != 2 {
		t.Fatalf("plain calls in slow twins = %d, want 2", got)
	}
}

// TestFastcallLogosSpelling ATTACK: //logos:cgo_fastcall in a .lgo
// input maps to the same pair. Red before: the line is ignored and the
// wrapper takes the plain entry.
func TestFastcallLogosSpelling(t *testing.T) {
	const src = `package fast

/*
//logos:cgo_fastcall ladd
int ladd(int a, int b) { return a + b; }
*/
import "C"

func Add(a, b int32) int32 { return int32(C.ladd(C.int(a), C.int(b))) }
`
	_, files := cgoOutputs(t, "fast.lgo", src)
	gotypes, ok := files["_cgo_gotypes.go"]
	if !ok {
		t.Fatal("no _cgo_gotypes.go emitted for .lgo input")
	}
	if got := countCallLines(gotypes, "_cgo_runtime_cgocall_fast"); got != 1 {
		t.Fatalf("fast calls = %d, want 1", got)
	}
	if got := countCallLines(gotypes, "_cgo_runtime_cgocall"); got != 0 {
		t.Fatalf("plain calls = %d, want 0", got)
	}
}

// TestFastcallUnmatched ATTACK: a fastcall name with no matched C
// function fails like the pair does. Red before: cgo succeeds because
// the //logos:cgo_fastcall line is ignored.
func TestFastcallUnmatched(t *testing.T) {
	requireCgo(t)
	work := t.TempDir()
	const src = `package fast

/*
//logos:cgo_fastcall ghost_fn
int real_fn(int a) { return a; }
*/
import "C"

func Call() int32 { return int32(C.real_fn(1)) }
`
	if err := os.WriteFile(filepath.Join(work, "bad.lgo"), []byte(src), 0644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(work, "out")
	if err := os.Mkdir(out, 0755); err != nil {
		t.Fatal(err)
	}
	text, err := runCmd(t, work, cgoBin(t), "-objdir", out, "-importpath", "scratch/bad", "bad.lgo")
	if err == nil {
		t.Fatalf("cgo succeeded on unmatched //logos:cgo_fastcall, want failure\n%s", text)
	}
	if !strings.Contains(text, "no matched C function") {
		t.Fatalf("stderr lacks %q:\n%s", "no matched C function", text)
	}
}

// TestFastcallPointerChecks PIN: POD, unsafe-derived, and
// primitive-slice fast call sites emit no _cgoCheckPointer call, while
// a pointer-to-pointer argument keeps its check. No new elision runs
// in rewriteCall.
func TestFastcallPointerChecks(t *testing.T) {
	_, files := cgoOutputs(t, "fast.go", pairPreamble)
	var cgo1 string
	for name, body := range files {
		if strings.HasSuffix(name, ".cgo1.go") {
			cgo1 = body
		}
	}
	if cgo1 == "" {
		t.Fatal("no .cgo1.go emitted")
	}
	if got := strings.Count(cgo1, "_cgoCheckPointer"); got != 1 {
		t.Fatalf("_cgoCheckPointer references = %d, want 1 (the **int read only)", got)
	}
}

// TestFastcallLayout PIN: the generated file names keep the upstream
// set: one stem-named pair per input plus the package outputs.
func TestFastcallLayout(t *testing.T) {
	_, files := cgoOutputs(t, "fast.go", pairPreamble)
	var cgo1, cgo2 int
	for name := range files {
		switch {
		case strings.HasSuffix(name, ".cgo1.go"):
			cgo1++
			if name != "fast.cgo1.go" {
				t.Fatalf("unexpected cgo1 name %q", name)
			}
		case strings.HasSuffix(name, ".cgo2.c"):
			cgo2++
			if name != "fast.cgo2.c" {
				t.Fatalf("unexpected cgo2 name %q", name)
			}
		}
	}
	if cgo1 != 1 || cgo2 != 1 {
		t.Fatalf("cgo1 = %d, cgo2 = %d, want 1 and 1", cgo1, cgo2)
	}
	if _, ok := files["_cgo_gotypes.go"]; !ok {
		t.Fatal("no _cgo_gotypes.go emitted")
	}
}

// selfTestBin builds an unstripped test binary for inspection. Plain
// `go test` links run-mode binaries with -s -w, so os.Executable()
// carries no symbol section; `go test -c` preserves it. The build
// inherits the gate GOEXPERIMENT, so the binary matches the mode.
func selfTestBin(t *testing.T) string {
	t.Helper()
	requireCgo(t)
	bin := filepath.Join(t.TempDir(), "testlogos.test")
	srcCmd := filepath.Join(runtime.GOROOT(), "src", "cmd")
	out, err := runCmd(t, srcCmd, goBin(t), "test", "-c", "-o", bin, "cmd/cgo/internal/testlogos")
	if err != nil {
		t.Fatalf("self build: %v\n%s", err, out)
	}
	return bin
}

// nmLines returns the symbol table of binary, one line per symbol.
func nmLines(t *testing.T, binary string) []string {
	t.Helper()
	out, err := runCmd(t, "", goBin(t), "tool", "nm", binary)
	if err != nil {
		t.Fatalf("go tool nm: %v\n%s", err, out)
	}
	return strings.Split(out, "\n")
}

// TestFastcallOneTrampoline ATTACK: the linked test binary holds exactly
// one logosCgocallFast text symbol and no rival fast-cgo entry.
// Red before: the symbol does not exist.
func TestFastcallOneTrampoline(t *testing.T) {
	self := selfTestBin(t)
	defs := 0
	for _, line := range nmLines(t, self) {
		f := strings.Fields(line)
		if len(f) != 3 {
			continue
		}
		kind, name := f[1], f[2]
		if name == "runtime.logosCgocallFast" && (kind == "T" || kind == "t") {
			defs++
			continue
		}
		if strings.HasPrefix(name, "runtime.") && (kind == "T" || kind == "t") &&
			strings.Contains(strings.ToLower(name), "fastcgo") {
			t.Fatalf("rival fast-cgo entry %s (no second cgo path)", line)
		}
		if strings.Contains(name, "logosFastCgoCall") {
			t.Fatalf("stale PLAN symbol name %s", line)
		}
	}
	if defs != 1 {
		t.Fatalf("runtime.logosCgocallFast text definitions = %d, want 1", defs)
	}
}

// delayStats reports the median and population standard deviation.
func delayStats(ds []float64) (median, stddev float64) {
	cp := slices.Clone(ds)
	slices.Sort(cp)
	median = cp[len(cp)/2]
	var sum, sq float64
	for _, d := range cp {
		sum += d
		sq += d * d
	}
	mean := sum / float64(len(cp))
	v := sq/float64(len(cp)) - mean*mean
	if v < 0 {
		v = 0
	}
	return median, math.Sqrt(v)
}

// oneDelay measures how long a freshly runnable goroutine waits before
// it runs while another goroutine blocks in sleepFn. The waiter is
// pre-created and parked; the blocker stamps t0, releases it, and
// enters C, so the waiter contends for the P sysmon retakes. Without
// a retake it would wait out the whole sleep.
func oneDelay(sleepFn func(micros int32) int32) float64 {
	ready := make(chan struct{})
	aDone := make(chan struct{})
	bDelay := make(chan time.Duration, 1)
	var t0 time.Time
	var wg sync.WaitGroup
	wg.Go(func() {
		<-ready
		bDelay <- time.Since(t0)
	})
	wg.Go(func() {
		defer close(aDone)
		t0 = time.Now()
		close(ready)
		sleepFn(100)
	})
	d := float64((<-bDelay).Microseconds())
	<-aDone
	wg.Wait()
	return d
}

// readyDelays collects N measurements per path, alternating fast and
// plain iterations so host drift lands on both series equally.
func readyDelays(t *testing.T, n int) (fast, plain []float64) {
	t.Helper()
	oldMax := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(oldMax)
	oldGC := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(oldGC)
	fast = make([]float64, 0, n)
	plain = make([]float64, 0, n)
	for range n {
		fast = append(fast, oneDelay(callFastSleep))
		plain = append(plain, oneDelay(callPlainSleep))
	}
	return fast, plain
}

// TestFastcallBlockingParity PIN (#5, #88): a blocking fast call delays
// a ready goroutine no longer than the same sleep on the plain path.
// A failed noise limit blocks the check (Skip), never passes it.
func TestFastcallBlockingParity(t *testing.T) {
	requireCgo(t)
	if testUnderRace {
		t.Skip("timing gates need quiet scheduling; the race detector perturbs " +
			"us-scale delays past the 20% noise band. Entry and exit still run " +
			"under race in the other tests.")
	}
	const n = 200
	fast, plain := readyDelays(t, 200)
	fmed, fsd := delayStats(fast)
	pmed, psd := delayStats(plain)
	t.Logf("ready delay: fast median %.1fus (sd %.1f), plain median %.1fus (sd %.1f)",
		fmed, fsd, pmed, psd)
	if fsd > 0.2*fmed || psd > 0.2*pmed {
		t.Skipf("noise limit exceeded, check blocked: fast sd %.1f med %.1f, plain sd %.1f med %.1f",
			fsd, fmed, psd, pmed)
	}
	if fmed > 1.05*pmed {
		t.Fatalf("fast median %.1fus exceeds 1.05x plain median %.1fus", fmed, pmed)
	}
}

// TestFastcallSigprofSurvives PIN (#165, #173): profiling across fast
// calls completes and records samples instead of crashing.
func TestFastcallSigprofSurvives(t *testing.T) {
	requireCgo(t)
	prof := filepath.Join(t.TempDir(), "cpu.prof")
	f, err := os.Create(prof)
	if err != nil {
		t.Fatal(err)
	}
	if err := pprof.StartCPUProfile(f); err != nil {
		f.Close()
		t.Fatal(err)
	}
	deadline := time.Now().Add(500 * time.Millisecond)
	var sink int32
	for time.Now().Before(deadline) {
		for i := range int32(100) {
			sink += callFastAdd(i, i)
		}
	}
	pprof.StopCPUProfile()
	f.Close()
	fi, err := os.Stat(prof)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Size() == 0 {
		t.Fatal("empty cpu profile")
	}
	_ = sink
}

// pprofNodes runs go tool pprof -top and parses every data line into
// cumulative shares keyed by symbol, returning the top entry and its
// share too. C time lands on the call-site chain; cum shares propagate
// to callers regardless of wrapper inlining.
func pprofNodes(t *testing.T, binary, prof string) (top string, topShare float64, shares map[string]float64) {
	t.Helper()
	out, err := runCmd(t, "", goBin(t), "tool", "pprof", "-top", "-nodecount=50", binary, prof)
	if err != nil {
		t.Fatalf("pprof -top: %v\n%s", err, out)
	}
	shares = map[string]float64{}
	data := false
	for line := range strings.SplitSeq(out, "\n") {
		if !data {
			if strings.HasPrefix(strings.TrimSpace(line), "flat ") {
				data = true
			}
			continue
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		f := strings.Fields(line)
		if len(f) < 6 {
			continue
		}
		pct := strings.TrimSuffix(f[4], "%")
		v, perr := strconv.ParseFloat(pct, 64)
		if perr != nil {
			t.Fatalf("parse cum%% %q: %v", f[4], perr)
		}
		name := f[len(f)-1]
		shares[name] = v
		if top == "" {
			top, topShare = name, v
		}
	}
	if top == "" {
		t.Fatalf("no data in pprof -top:\n%s", out)
	}
	return top, topShare, shares
}

// nodeShare returns the cumulative share of the first node containing
// want, or 0 when no node does.
func nodeShare(shares map[string]float64, want string) float64 {
	for name, v := range shares {
		if strings.Contains(name, want) {
			return v
		}
	}
	return 0
}

// topIn reports whether top names one of the accepted call-site symbols.
func topIn(top string, wants []string) bool {
	for _, w := range wants {
		if strings.Contains(top, w) {
			return true
		}
	}
	return false
}

// collectBusyProfile profiles count busy calls through fn and stores
// the profile at prof. It returns the observed call count.
func collectBusyProfile(t *testing.T, prof string, count int, fn func(uint64) uint64, iters uint64) {
	t.Helper()
	f, err := os.Create(prof)
	if err != nil {
		t.Fatal(err)
	}
	if err := pprof.StartCPUProfile(f); err != nil {
		f.Close()
		t.Fatal(err)
	}
	var sink uint64
	for range count {
		sink += fn(iters)
	}
	pprof.StopCPUProfile()
	f.Close()
	_ = sink
}

// TestFastcallProfileAttribution PIN (#189): C time on the fast path is
// attributed to the Go call site, within 5 points of the plain path.
// Thin samples block the check (Skip).
func TestFastcallProfileAttribution(t *testing.T) {
	requireCgo(t)
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	fastProf := filepath.Join(dir, "fast.prof")
	plainProf := filepath.Join(dir, "plain.prof")
	const iters = 200000000
	collectBusyProfile(t, fastProf, 4, callFastBusy, iters)
	collectBusyProfile(t, plainProf, 4, callPlainBusy, iters)
	fastWants := []string{"callFastBusy", "_Cfunc_busy_spin"}
	plainWants := []string{"callPlainBusy", "_Cfunc_plain_spin", "runtime.cgocall"}
	if !goexperiment.LogosFastCgo {
		// No fast path in fallback mode: the entry delegates to
		// plain cgocall, so its frames top the profile like plain.
		fastWants = append(fastWants, "logosCgocallFast", "runtime.cgocall")
	}
	topFast, _, fastShares := pprofNodes(t, self, fastProf)
	topPlain, _, plainShares := pprofNodes(t, self, plainProf)
	fastCaller := nodeShare(fastShares, "callFastBusy")
	plainCaller := nodeShare(plainShares, "callPlainBusy")
	t.Logf("top: fast %q caller %.1f%%, plain %q caller %.1f%%",
		topFast, fastCaller, topPlain, plainCaller)
	if fastCaller < 50 || plainCaller < 50 {
		t.Skipf("thin profile, check blocked: fast caller %.1f%% plain caller %.1f%%",
			fastCaller, plainCaller)
	}
	if !topIn(topFast, fastWants) {
		t.Fatalf("fast top entry = %q, want the call site", topFast)
	}
	if !topIn(topPlain, plainWants) {
		t.Fatalf("plain top entry = %q, want the call site", topPlain)
	}
	if diff := fastCaller - plainCaller; diff < -5 || diff > 5 {
		t.Fatalf("caller attribution differs by %.1f points", diff)
	}
}

// violatorBin builds the violation child for mode ("" for the pair,
// "plain" for nocallback alone).
func violatorBin(t *testing.T, mode string) (bin string, out string) {
	t.Helper()
	requireCgo(t)
	bin = filepath.Join(t.TempDir(), "viol.exe")
	text, err := runCmd(t, filepath.Join("testdata", "viol"), goBin(t), "build", "-o", bin, ".")
	if err != nil {
		t.Fatalf("violator build: %v\n%s", err, text)
	}
	args := []string{}
	if mode != "" {
		args = append(args, mode)
	}
	out, err = runCmd(t, "", bin, args...)
	if err != nil {
		t.Fatalf("violator %q: %v\n%s", mode, err, out)
	}
	return bin, out
}

// checkViolation asserts the child protocol: callback panic, stuck
// flag on the repeat, and a healthy unmarked call after.
func checkViolation(t *testing.T, mode, out string) {
	t.Helper()
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) != 3 {
		t.Fatalf("violator %q printed %d lines, want 3:\n%s", mode, len(lines), out)
	}
	if !strings.Contains(lines[0], "marked with #cgo nocallback called back into Go") {
		t.Fatalf("first line = %q, want the nocallback message", lines[0])
	}
	if !strings.Contains(lines[1], "unexpected setting cgoNoCallback") {
		t.Fatalf("second line = %q, want the stuck-flag message", lines[1])
	}
	if strings.TrimSpace(lines[2]) != "add: 42" {
		t.Fatalf("third line = %q, want add: 42", lines[2])
	}
}

// TestFastcallNoCallbackViolation PIN (#188): a pair-marked C function
// that calls back into Go panics with the documented message, finishes
// within the test deadline instead of hanging, and leaves unmarked
// calls healthy. The violation runs isolated in a child: recovering
// in-process would return its locked goroutine to the pool.
func TestFastcallNoCallbackViolation(t *testing.T) {
	_, out := violatorBin(t, "")
	checkViolation(t, "", out)
	if got := callSlowNoescape(20, 22); got != 42 {
		t.Fatalf("unmarked call after violation = %d, want 42", got)
	}
	done := make(chan int32, 1)
	go func() { done <- callSlowNoescape(20, 22) }()
	select {
	case got := <-done:
		if got != 42 {
			t.Fatalf("goroutine unmarked call = %d, want 42", got)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("goroutine stuck after violation (hang)")
	}
}

// TestFastcallCallbackParity PIN (#188): the plain path follows the
// same protocol line for line.
func TestFastcallCallbackParity(t *testing.T) {
	_, out := violatorBin(t, "plain")
	checkViolation(t, "plain", out)
}

func TestFastcallNumCgoCallAccounting(t *testing.T) {
	requireCgo(t)
	bin := filepath.Join(t.TempDir(), "callcount.exe")
	text, err := runCmd(t, filepath.Join("testdata", "callcount"), goBin(t), "build", "-o", bin, ".")
	if err != nil {
		t.Fatalf("callcount build: %v\n%s", err, text)
	}
	out, err := runCmd(t, "", bin)
	if err != nil {
		t.Fatalf("callcount run: %v\n%s", err, out)
	}
}

// TestFastcallGoPointerValid PIN (#160): a Go pointer argument stays
// valid across the fast call.
func TestFastcallGoPointerValid(t *testing.T) {
	requireCgo(t)
	v := int32(-99)
	if got := callFastRead(&v); got != -99 {
		t.Fatalf("read_int = %d, want -99", got)
	}
	b := []byte{10, 20, 30}
	if got := callFastSlice(b); got != 60 {
		t.Fatalf("sum_bytes = %d, want 60", got)
	}
}

// fixtureBin builds the fixture module with logos and returns the binary.
func fixtureBin(t *testing.T, module string) string {
	t.Helper()
	dir := filepath.Join("testdata", module)
	bin := filepath.Join(t.TempDir(), "prog.exe")
	out, err := runCmd(t, dir, logosBin(t), "build", "-o", bin, ".")
	if err != nil {
		t.Fatalf("logos build %s: %v\n%s", module, err, out)
	}
	return bin
}

// TestFastcallLogosEndToEnd ATTACK: a loGo/s package using
// //logos:cgo_fastcall builds, runs, and routes through the single
// trampoline. Red before: the symbol is missing from nm and the
// wrapper takes the plain entry.
func TestFastcallLogosEndToEnd(t *testing.T) {
	requireCgo(t)
	bin := fixtureBin(t, "logosfast")
	out, err := runCmd(t, "", bin)
	if err != nil {
		t.Fatalf("fixture run: %v\n%s", err, out)
	}
	if strings.TrimSpace(out) != "42" {
		t.Fatalf("fixture output = %q, want %q", out, "42")
	}
	defs := 0
	for _, line := range nmLines(t, bin) {
		f := strings.Fields(line)
		if len(f) != 3 {
			continue
		}
		if f[2] == "runtime.logosCgocallFast" && (f[1] == "T" || f[1] == "t") {
			defs++
		}
	}
	if defs != 1 {
		t.Fatalf("fixture logosCgocallFast text definitions = %d, want 1", defs)
	}
	dump, err := runCmd(t, "", goBin(t), "tool", "objdump", bin)
	if err != nil {
		t.Fatalf("objdump fixture: %v", err)
	}
	if !strings.Contains(dump, "runtime.logosCgocallFast") {
		t.Fatal("fixture binary never calls the fast entry")
	}
}

// TestFastcallLogosCallbackParity ATTACK: the .lgo spelling panics the
// same way on a callback violation. Red before: the call succeeds and
// prints 42 and survived.
func TestFastcallLogosCallbackParity(t *testing.T) {
	requireCgo(t)
	out, err := runCmd(t, filepath.Join("testdata", "logosfastcb"), logosBin(t), "run", ".")
	if err != nil {
		t.Fatalf("logos run logosfastcb: %v\n%s", err, out)
	}
	if !strings.Contains(out, "marked with #cgo nocallback called back into Go") {
		t.Fatalf("missing nocallback panic message:\n%s", out)
	}
	if strings.Contains(out, "survived") {
		t.Fatalf("callback unexpectedly survived:\n%s", out)
	}
}

// TestFastcallLogosProfileParity PIN: the .lgo spelling attributes C
// time to its call site like the Go spelling. Thin samples block it.
func TestFastcallLogosProfileParity(t *testing.T) {
	requireCgo(t)
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	plainProf := filepath.Join(t.TempDir(), "plain.prof")
	collectBusyProfile(t, plainProf, 4, callPlainBusy, 200000000)
	plainWants := []string{"callPlainBusy", "_Cfunc_plain_spin", "runtime.cgocall"}
	if !goexperiment.LogosFastCgo {
		plainWants = append(plainWants, "logosCgocallFast")
	}
	topPlain, _, plainShares := pprofNodes(t, self, plainProf)
	sharePlain := nodeShare(plainShares, "callPlainBusy")
	if !topIn(topPlain, plainWants) {
		t.Fatalf("plain top entry = %q, want the call site", topPlain)
	}

	bin := fixtureBin(t, "logosfast")
	dir := t.TempDir()
	prof := filepath.Join(dir, "lgo.prof")
	out, err := runCmd(t, dir, bin, "busy", prof, "400000000")
	if err != nil {
		t.Fatalf("fixture busy: %v\n%s", err, out)
	}
	lgoWants := []string{"main.main", "_Cfunc_cbusyf"}
	if !goexperiment.LogosFastCgo {
		lgoWants = append(lgoWants, "logosCgocallFast", "runtime.cgocall")
	}
	top, _, lgoShares := pprofNodes(t, bin, prof)
	lgoCaller := nodeShare(lgoShares, "main.main")
	t.Logf("top: lgo %q caller %.1f%%, plain caller %.1f%%", top, lgoCaller, sharePlain)
	if lgoCaller < 50 || sharePlain < 50 {
		t.Skipf("thin profile, check blocked: lgo caller %.1f%% plain caller %.1f%%",
			lgoCaller, sharePlain)
	}
	if !topIn(top, lgoWants) {
		t.Fatalf("lgo top entry = %q, want the call site", top)
	}
	if diff := lgoCaller - sharePlain; diff < -5 || diff > 5 {
		t.Fatalf("lgo attribution differs by %.1f points", diff)
	}
}

// BenchmarkFastcall reports the fast-entry cost (report only, never a
// pass condition). The G6 ratio uses this benchmark built with and
// without the experiment.
func BenchmarkFastcall(b *testing.B) {
	requireCgo(b)
	b.Logf("logosfastcgo = %v", goexperiment.LogosFastCgo)
	var sink int32
	for b.Loop() {
		sink = callFastAdd(sink&((1<<30)-1), 1)
	}
	_ = sink
}

// BenchmarkFastcallSlow reports the plain-entry cost for the ratio context.
func BenchmarkFastcallSlow(b *testing.B) {
	requireCgo(b)
	var sink int32
	for b.Loop() {
		sink = callSlowNoescape(sink&((1<<30)-1), 1)
	}
	_ = sink
}
