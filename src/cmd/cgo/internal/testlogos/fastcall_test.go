// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlogos

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"github.com/google/pprof/profile"
	"internal/goexperiment"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strconv"
	"strings"
	"testing"
	"time"
	"weak"
)

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

func TestFastcallEmission(t *testing.T) {
	t.Setenv("GOEXPERIMENT", "logosfastcgo")
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

func TestFastcallLogosSpelling(t *testing.T) {
	t.Setenv("GOEXPERIMENT", "logosfastcgo")
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
	if got := countCallLines(gotypes, "_Cgo_keepalive"); got != 2 {
		t.Fatalf("keepalive uses = %d, want 2", got)
	}
	if got := countCallLines(gotypes, "_Cgo_use"); got != 0 {
		t.Fatalf("escape uses = %d, want 0", got)
	}
	if got := countCallLines(gotypes, "_Cgo_no_callback"); got != 0 {
		t.Fatalf("callback guards = %d, want 0", got)
	}
}

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

func TestFastcallMarksCarryBoth(t *testing.T) {
	const head = `package fast

/*
#include <stdint.h>

`
	const tail = `
int mark_fn(int *p) { return *p + 1; }
*/
import "C"

import "unsafe"

func Mark(p *int32) int32 { return int32(C.mark_fn((*C.int)(unsafe.Pointer(p)))) }
`
	cases := []struct {
		name       string
		directives string
		fast       int
		plain      int
		keepalive  int
		use        int
		guards     int
	}{
		{"both", "#cgo noescape mark_fn\n#cgo nocallback mark_fn\n", 1, 0, 1, 0, 0},
		{"noescape", "#cgo noescape mark_fn\n", 0, 1, 0, 1, 0},
		{"nocallback", "#cgo nocallback mark_fn\n", 0, 1, 0, 1, 2},
		{"none", "", 0, 1, 0, 1, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fast, plain, keepalive, use, guards := c.fast, c.plain, c.keepalive, c.use, c.guards
			if !goexperiment.LogosFastCgo && c.name == "both" {
				fast, plain, keepalive, use, guards = 0, 1, 1, 0, 2
			}
			_, files := cgoOutputs(t, "mark.go", head+c.directives+tail)
			gotypes, ok := files["_cgo_gotypes.go"]
			if !ok {
				t.Fatal("no _cgo_gotypes.go emitted")
			}
			if got := countCallLines(gotypes, "_cgo_runtime_cgocall_fast"); got != fast {
				t.Errorf("fast calls = %d, want %d", got, fast)
			}
			plainGot := 0
			for line := range strings.SplitSeq(gotypes, "\n") {
				if strings.HasPrefix(line, "\t") && strings.Contains(line, "_cgo_runtime_cgocall(") && !strings.Contains(line, "_fast(") {
					plainGot++
				}
			}
			if plainGot != plain {
				t.Errorf("plain calls = %d, want %d", plainGot, plain)
			}
			if got := countCallLines(gotypes, "_Cgo_keepalive"); got != keepalive {
				t.Errorf("keepalive uses = %d, want %d", got, keepalive)
			}
			if got := countCallLines(gotypes, "_Cgo_use"); got != use {
				t.Errorf("escape uses = %d, want %d", got, use)
			}
			if got := countCallLines(gotypes, "_Cgo_no_callback"); got != guards {
				t.Errorf("callback guards = %d, want %d", got, guards)
			}
		})
	}
}

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

// fastcallToolFlags mirrors this test binary's experiment and instrumentation
// into explicit test -c builds so disassembly assertions inspect the
// configuration under test, including the fast-call experiment bit itself.
func fastcallToolFlags(t *testing.T) []string {
	t.Helper()
	return cgoTestToolFlags(t, goexperiment.LogosFastCgo)
}

func selfTestBin(t *testing.T) string {
	t.Helper()
	requireCgo(t)
	bin := filepath.Join(t.TempDir(), "testlogos.test")
	srcCmd := filepath.Join(runtime.GOROOT(), "src", "cmd")
	args := append([]string{"test", "-c", "-o", bin}, fastcallToolFlags(t)...)
	out, err := runCmd(t, srcCmd, goBin(t), append(args, "cmd/cgo/internal/testlogos")...)
	if err != nil {
		t.Fatalf("self build: %v\n%s", err, out)
	}
	return bin
}

func nmLines(t *testing.T, binary string) []string {
	t.Helper()
	out, err := runCmd(t, "", goBin(t), "tool", "nm", binary)
	if err != nil {
		t.Fatalf("go tool nm: %v\n%s", err, out)
	}
	return strings.Split(out, "\n")
}

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
	wantDefs := 1
	if !goexperiment.LogosFastCgo {
		wantDefs = 0
	}
	if defs != wantDefs {
		t.Fatalf("runtime.logosCgocallFast text definitions = %d, want %d", defs, wantDefs)
	}
}

func objdumpFunc(t *testing.T, binary, symbol string) string {
	t.Helper()
	out, err := runCmd(t, "", goBin(t), "tool", "objdump", "-s", symbol, binary)
	if err != nil {
		t.Fatalf("go tool objdump -s %s: %v\n%s", symbol, err, out)
	}
	return out
}

func TestFastcallStructure(t *testing.T) {
	self := selfTestBin(t)
	if !goexperiment.LogosFastCgo {
		for _, line := range nmLines(t, self) {
			if strings.Contains(line, "logosCgocallFast") {
				t.Fatalf("unflagged binary references the fast path: %s", line)
			}
		}
		body := objdumpFunc(t, self, "testlogos\\._Cfunc_fast_add")
		if !strings.Contains(body, "CALL runtime.cgocall(SB)") {
			t.Fatalf("unflagged fallback lacks the plain cgocall:\n%s", body)
		}
		if strings.Contains(body, "asmcgocall") || strings.Contains(body, "logosCgocallFast") {
			t.Fatalf("unflagged fallback leaves the fast path:\n%s", body)
		}
		return
	}
	body := objdumpFunc(t, self, "runtime\\.logosCgocallFast")
	if !strings.Contains(body, "logosCgocallFast") {
		t.Fatalf("no logosCgocallFast disassembly:\n%s", body)
	}
	for _, want := range []string{"asmcgocall", "reentersyscall", "exitsyscall"} {
		if !strings.Contains(body, want) {
			t.Fatalf("fast body lacks %s:\n%s", want, body)
		}
	}
	if got := strings.Count(body, "CALL runtime.cgocall(SB)"); got != 0 {
		t.Fatalf("fast body has %d plain cgocall calls, want 0 (fallback must be pruned):\n%s", got, body)
	}
}

func TestFastcallBlockingParity(t *testing.T) {
	requireCgo(t)
	oldMax := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(oldMax)
	readyR, readyW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer readyR.Close()
	defer readyW.Close()
	gateR, gateW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer gateR.Close()
	defer gateW.Close()
	done := make(chan int32, 1)
	go func() {
		done <- callFastGateWait(int32(readyW.Fd()), int32(gateR.Fd()))
	}()
	type readyRes struct {
		n    int
		rerr error
	}
	readyCh := make(chan readyRes, 1)
	go func() {
		var b [1]byte
		n, rerr := readyR.Read(b[:])
		readyCh <- readyRes{n, rerr}
	}()
	select {
	case rr := <-readyCh:
		if rr.n != 1 || rr.rerr != nil {
			t.Fatalf("ready read = %d, %v, want 1 byte", rr.n, rr.rerr)
		}
	case code := <-done:
		t.Fatalf("gate_wait = %d before ready signal", code)
	}
	prog := make(chan struct{})
	go func() {
		close(prog)
	}()
	select {
	case <-prog:
	case code := <-done:
		t.Fatalf("gate_wait = %d while waiting for progress, P was not released", code)
	}
	select {
	case code := <-done:
		if code == 2 {
			t.Fatal("fast call timed out waiting for gate release, P was not released")
		}
		t.Fatalf("fast call returned %d before gate release", code)
	default:
	}
	if _, werr := gateW.Write([]byte{0}); werr != nil {
		t.Fatal(werr)
	}
	select {
	case code := <-done:
		if code != 0 {
			t.Fatalf("gate_wait = %d, want 0", code)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("fast call did not return within 30s of gate release")
	}
}

func TestFastcallSigprofSurvives(t *testing.T) {
	requireCgo(t)
	prof := filepath.Join(t.TempDir(), "cpu.prof")
	const floor = 50
	const maxIters = 2000000000
	deadline := time.Now().Add(120 * time.Second)
	iters := uint64(100000000)
	var n, total int
	for {
		collectBusyProfile(t, prof, 10, callFastBusy, iters)
		n, total = profileCallerSamples(t, prof, "callFastBusy")
		t.Logf("fast caller %d of %d samples", n, total)
		if n == 0 && total >= floor {
			t.Fatalf("fast-call frames missing from profile: fast caller %d of %d samples (floor %d)",
				n, total, floor)
		}
		if n >= floor {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("sigprof floor unmet: fast caller %d of %d samples (floor %d)",
				n, total, floor)
		}
		iters = growProfileWork(total, floor, iters, maxIters)
	}
	if total == 0 {
		t.Fatal("empty cpu profile")
	}
	if fi, serr := os.Stat(prof); serr != nil || fi.Size() == 0 {
		t.Fatal("empty cpu profile")
	}
}

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

func topIn(top string, wants []string) bool {
	for _, w := range wants {
		if strings.Contains(top, w) {
			return true
		}
	}
	return false
}

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

func goSpinSink(iters uint64) uint64 {
	x := uint64(1)
	for i := uint64(0); i < iters; i++ {
		x += x*6364136223846793005 + 1442695040888963407
	}
	return x
}

func profileCanarySamples(t *testing.T) int {
	t.Helper()
	prof := filepath.Join(t.TempDir(), "canary.prof")
	collectBusyProfile(t, prof, 10, goSpinSink, 100000000)
	_, total := profileCallerSamples(t, prof, "goSpinSink")
	return total
}

const fastcallProfileFloorSamples = 200
const fastcallAttributionFloorSamples = 100
const fastcallProfileCap = 300 * time.Second

func growProfileWork(total, floor int, work, maxWork uint64) uint64 {
	factor := uint64(2)
	if total > 0 {
		f := (2*uint64(floor) + uint64(total) - 1) / uint64(total)
		factor = max(f, factor)
	}
	return min(work*factor, maxWork)
}

func profileCallerSamples(t *testing.T, prof string, want string) (found, total int) {
	t.Helper()
	f, err := os.Open(prof)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	p, err := profile.Parse(f)
	if err != nil {
		t.Fatalf("parse profile %s: %v", prof, err)
	}
	for _, s := range p.Sample {
		n := 1
		if len(s.Value) > 0 {
			n = int(s.Value[0])
		}
		total += n
		for _, loc := range s.Location {
			hit := false
			for _, line := range loc.Line {
				if line.Function != nil && strings.Contains(line.Function.Name, want) {
					hit = true
					break
				}
			}
			if hit {
				found += n
				break
			}
		}
	}
	return found, total
}

func profileCallerAboveLeafSamples(t *testing.T, prof, caller, leaf string) (above, total int) {
	t.Helper()
	f, err := os.Open(prof)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	p, err := profile.Parse(f)
	if err != nil {
		t.Fatalf("parse profile %s: %v", prof, err)
	}
	for _, s := range p.Sample {
		n := 1
		if len(s.Value) > 0 {
			n = int(s.Value[0])
		}
		total += n
		if sampleCallerAboveLeaf(s, caller, leaf) {
			above += n
		}
	}
	return above, total
}

func sampleCallerAboveLeaf(s *profile.Sample, caller, leaf string) bool {
	ci, li := -1, -1
	for i, loc := range s.Location {
		hasCaller, hasLeaf := locationMentions(loc, caller, leaf)
		if hasCaller && ci < 0 {
			ci = i
		}
		if hasLeaf && li < 0 {
			li = i
		}
	}
	return li >= 0 && ci > li
}

func locationMentions(loc *profile.Location, caller, leaf string) (bool, bool) {
	if loc == nil {
		return false, false
	}
	var hasCaller, hasLeaf bool
	for _, line := range loc.Line {
		if line.Function == nil {
			continue
		}
		hasCaller = hasCaller || strings.Contains(line.Function.Name, caller)
		hasLeaf = hasLeaf || strings.Contains(line.Function.Name, leaf)
	}
	return hasCaller, hasLeaf
}

func TestFastcallProfileAttribution(t *testing.T) {
	requireCgo(t)
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	fastProf := filepath.Join(dir, "fast.prof")
	plainProf := filepath.Join(dir, "plain.prof")
	const maxIters = 1000000000
	fastIters := uint64(100000000)
	plainIters := uint64(100000000)
	fastWants := []string{"callFastBusy", "_Cfunc_busy_spin"}
	plainWants := []string{"callPlainBusy", "_Cfunc_plain_spin", "runtime.cgocall"}
	if !goexperiment.LogosFastCgo {
		fastWants = append(fastWants, "logosCgocallFast", "runtime.cgocall")
	}
	var fastN, plainN, fastTotal, plainTotal int
	met := false
	deadline := time.Now().Add(fastcallProfileCap)
	for {
		collectBusyProfile(t, fastProf, 30, callFastBusy, fastIters)
		collectBusyProfile(t, plainProf, 30, callPlainBusy, plainIters)
		fastN, fastTotal = profileCallerSamples(t, fastProf, "callFastBusy")
		plainN, plainTotal = profileCallerSamples(t, plainProf, "callPlainBusy")
		t.Logf("fast caller %d of %d samples, plain caller %d of %d samples",
			fastN, fastTotal, plainN, plainTotal)
		if fastN == 0 && plainN >= fastcallProfileFloorSamples {
			t.Fatalf("fast-call frames missing from profile: fast caller %d of %d samples, plain caller %d of %d samples (floor %d)",
				fastN, fastTotal, plainN, plainTotal, fastcallProfileFloorSamples)
		}
		if fastN >= fastcallProfileFloorSamples && plainN >= fastcallProfileFloorSamples {
			met = true
			break
		}
		if time.Now().After(deadline) {
			break
		}
		fastIters = growProfileWork(fastTotal, fastcallProfileFloorSamples, fastIters, maxIters)
		plainIters = growProfileWork(plainTotal, fastcallProfileFloorSamples, plainIters, maxIters)
	}
	if !met {
		if fastTotal == 0 && plainTotal == 0 {
			if profileCanarySamples(t) > 0 {
				t.Fatalf("profiler works but yielded no fast/plain samples: fast caller %d of %d, plain caller %d of %d (floor %d)",
					fastN, fastTotal, plainN, plainTotal, fastcallProfileFloorSamples)
			}
			t.Skipf("CPU profiler yielded no samples in %v of collection; cannot attribute fast-call frames", fastcallProfileCap)
		}
		t.Fatalf("profile floor unmet: fast caller %d of %d samples, plain caller %d of %d samples (floor %d)",
			fastN, fastTotal, plainN, plainTotal, fastcallProfileFloorSamples)
	}
	topFast, _, _ := pprofNodes(t, self, fastProf)
	topPlain, _, _ := pprofNodes(t, self, plainProf)
	if !topIn(topFast, fastWants) {
		t.Fatalf("fast top entry = %q, want the call site", topFast)
	}
	if !topIn(topPlain, plainWants) {
		t.Fatalf("plain top entry = %q, want the call site", topPlain)
	}
	above, aboveTotal := profileCallerAboveLeafSamples(t, fastProf, "callFastBusy", "busy_spin")
	if above < fastcallAttributionFloorSamples {
		t.Fatalf("%d of %d fast samples hold the Go call site above the fast C frame, want at least %d",
			above, aboveTotal, fastcallAttributionFloorSamples)
	}
}

func TestFastcallMarkerCallPath(t *testing.T) {
	requireCgo(t)
	wantFast := goexperiment.LogosFastCgo
	bin := fixtureBin(t, "markers")
	for _, c := range []struct {
		pkg  string
		fn   string
		fast bool
	}{
		{"ok", "ladd_ok", wantFast},
		{"arity", "ladd_arity", false},
		{"nonpre", "ladd_nonpre", false},
	} {
		body := objdumpFunc(t, bin, fmt.Sprintf(`example\.com/markers/%s\._Cfunc_%s`, c.pkg, c.fn))
		fast := strings.Contains(body, "runtime.logosCgocallFast")
		slow := strings.Contains(body, "runtime.cgocall")
		if c.fast && !fast {
			t.Fatalf("%s marker did not call the fast trampoline:\n%s", c.pkg, body)
		}
		if !c.fast && fast {
			t.Fatalf("%s marker with a malformed preamble still called the fast trampoline:\n%s", c.pkg, body)
		}
		if !c.fast && !slow {
			t.Fatalf("%s marker called neither trampoline:\n%s", c.pkg, body)
		}
	}
}

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

func checkViolation(t *testing.T, mode, out string) {
	t.Helper()
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) != 3 || !strings.Contains(lines[0], "marked with #cgo nocallback called back into Go") {
		t.Fatalf("violator %q output = %q, want the recovered nocallback message first", mode, out)
	}
	if mode == "plain" {
		if !strings.Contains(lines[1], "unexpected setting cgoNoCallback") {
			t.Fatalf("second line = %q, want the stuck-flag message", lines[1])
		}
		if strings.TrimSpace(lines[2]) != "add: 42" {
			t.Fatalf("third line = %q, want add: 42", lines[2])
		}
		return
	}
	var ok, total int
	if n, err := fmt.Sscanf(lines[1], "loop: %d/%d", &ok, &total); n != 2 || err != nil || total <= 0 {
		t.Fatalf("violator %q loop line = %q, want loop: <ok>/<total>", mode, lines[1])
	}
	if goexperiment.LogosFastCgo {
		if ok != total {
			t.Fatalf("violator %q recovered the nocallback panic %d of %d times", mode, ok, total)
		}
		if strings.TrimSpace(lines[2]) != "add: 42" {
			t.Fatalf("third line = %q, want the clean fast call result", lines[2])
		}
		return
	}
	if !strings.Contains(lines[2], "unexpected setting cgoNoCallback") {
		t.Fatalf("third line = %q, want the upstream stuck-flag message", lines[2])
	}
}

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

func TestFastcallNumCgoCallAccounting(t *testing.T) {
	requireCgo(t)
	before := runtime.NumCgoCall()
	if got := callFastAdd(40, 2); got != 42 {
		t.Fatalf("fast_add = %d, want 42", got)
	}
	if mid := runtime.NumCgoCall(); mid != before+1 {
		t.Fatalf("NumCgoCall after fast call = %d, want %d", mid, before+1)
	}
	if got := callSlowNoescape(40, 2); got != 42 {
		t.Fatalf("slow_noescape = %d, want 42", got)
	}
	if after := runtime.NumCgoCall(); after != before+2 {
		t.Fatalf("NumCgoCall after slow call = %d, want %d", after, before+2)
	}
}

func growStack(depth int) byte {
	if depth == 0 {
		return 0
	}
	var pad [96]byte
	pad[0] = byte(depth)
	pad[95] = growStack(depth - 1)
	return pad[0] + pad[95]
}

func gateWatchHeap() (*byte, *watchHeap) {
	w := &watchHeap{}
	w.b[0] = 42
	return &w.b[0], w
}

type watchHeap struct {
	b [256 << 10]byte
}

func TestFastcallPinUnderCollection(t *testing.T) {
	requireCgo(t)
	readyR, readyW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer readyR.Close()
	defer readyW.Close()
	gateR, gateW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer gateR.Close()
	defer gateW.Close()
	done := make(chan int32, 1)
	pinnedCh := make(chan weak.Pointer[watchHeap], 1)
	go func(readyFD, gateFD int32) {
		watchPtr, watchOwner := gateWatchHeap()
		pinnedCh <- weak.Make(watchOwner)
		done <- callFastGateWatch(readyFD, gateFD, watchPtr, 42)
	}(int32(readyW.Fd()), int32(gateR.Fd()))
	pinned := <-pinnedCh
	type readyRes struct {
		n    int
		rerr error
	}
	readyCh := make(chan readyRes, 1)
	go func() {
		var b [1]byte
		n, rerr := readyR.Read(b[:])
		readyCh <- readyRes{n, rerr}
	}()
	select {
	case rr := <-readyCh:
		if rr.n != 1 || rr.rerr != nil {
			t.Fatalf("ready read = %d, %v, want 1 byte", rr.n, rr.rerr)
		}
	case code := <-done:
		t.Fatalf("gate_watch = %d before ready signal", code)
	}
	for range 5 {
		runtime.GC()
	}
	for range 8 {
		q := make([]byte, 256<<10)
		for j := range q {
			q[j] = 0xFF
		}
		runtime.KeepAlive(q)
		runtime.GC()
	}
	if pinned.Value() == nil {
		t.Fatal("heap buffer collected while fast call holds it")
	}
	select {
	case code := <-done:
		t.Fatalf("gate_watch returned %d before gate release", code)
	default:
	}
	if _, werr := gateW.Write([]byte{0}); werr != nil {
		t.Fatal(werr)
	}
	select {
	case code := <-done:
		if code != 0 {
			t.Fatalf("gate_watch = %d, want 0", code)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("fast call did not return within 30s of gate release")
	}
}

func TestFastcallPinStackUnderCollection(t *testing.T) {
	requireCgo(t)
	readyR, readyW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer readyR.Close()
	defer readyW.Close()
	gateR, gateW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer gateR.Close()
	defer gateW.Close()
	done := make(chan int32, 1)
	go func(readyFD, gateFD int32) {
		growStack(4000)
		var buf [8192]byte
		buf[0] = 42
		done <- callFastGateWatch(readyFD, gateFD, &buf[0], 42)
	}(int32(readyW.Fd()), int32(gateR.Fd()))
	type readyRes struct {
		n    int
		rerr error
	}
	readyCh := make(chan readyRes, 1)
	go func() {
		var b [1]byte
		n, rerr := readyR.Read(b[:])
		readyCh <- readyRes{n, rerr}
	}()
	select {
	case rr := <-readyCh:
		if rr.n != 1 || rr.rerr != nil {
			t.Fatalf("ready read = %d, %v, want 1 byte", rr.n, rr.rerr)
		}
	case code := <-done:
		t.Fatalf("gate_watch = %d before ready signal", code)
	}
	for range 5 {
		runtime.GC()
	}
	for range 8 {
		q := make([]byte, 256<<10)
		for j := range q {
			q[j] = 0xFF
		}
		runtime.KeepAlive(q)
		runtime.GC()
	}
	select {
	case code := <-done:
		t.Fatalf("gate_watch returned %d before gate release", code)
	default:
	}
	if _, werr := gateW.Write([]byte{0}); werr != nil {
		t.Fatal(werr)
	}
	select {
	case code := <-done:
		if code != 0 {
			t.Fatalf("gate_watch = %d, want 0", code)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("fast call did not return within 30s of gate release")
	}
}

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
	wantDefs := 1
	if !goexperiment.LogosFastCgo {
		wantDefs = 0
	}
	if defs != wantDefs {
		t.Fatalf("fixture logosCgocallFast text definitions = %d, want %d", defs, wantDefs)
	}
	if !goexperiment.LogosFastCgo {
		dump, err := runCmd(t, "", goBin(t), "tool", "objdump", bin)
		if err != nil {
			t.Fatalf("objdump fixture: %v", err)
		}
		if strings.Contains(dump, "runtime.logosCgocallFast") {
			t.Fatal("unflagged fixture binary calls the fast entry")
		}
		wrap := objdumpFunc(t, bin, "main\\._Cfunc_caddf")
		if !strings.Contains(wrap, "CALL runtime.cgocall(SB)") {
			t.Fatalf("unflagged fixture lacks the plain cgocall:\n%s", wrap)
		}
		return
	}
	dump, err := runCmd(t, "", goBin(t), "tool", "objdump", bin)
	if err != nil {
		t.Fatalf("objdump fixture: %v", err)
	}
	if !strings.Contains(dump, "runtime.logosCgocallFast") {
		t.Fatal("fixture binary never calls the fast entry")
	}
}

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

func TestFastcallLogosProfileParity(t *testing.T) {
	requireCgo(t)
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	plainProf := filepath.Join(t.TempDir(), "plain.prof")
	plainWants := []string{"callPlainBusy", "_Cfunc_plain_spin", "runtime.cgocall"}
	if !goexperiment.LogosFastCgo {
		plainWants = append(plainWants, "logosCgocallFast")
	}
	bin := fixtureBin(t, "logosfast")
	dir := t.TempDir()
	prof := filepath.Join(dir, "lgo.prof")
	lgoWants := []string{"main.main", "_Cfunc_cbusyf"}
	if !goexperiment.LogosFastCgo {
		lgoWants = append(lgoWants, "logosCgocallFast", "runtime.cgocall")
	}
	plainIters := uint64(100000000)
	const maxPlainIters = 1000000000
	fixtureIters := uint64(3000000000)
	const maxFixtureIters = 30000000000
	var plainN, lgoN, plainTotal, lgoTotal int
	met := false
	deadline := time.Now().Add(fastcallProfileCap)
	for {
		collectBusyProfile(t, plainProf, 30, callPlainBusy, plainIters)
		plainN, plainTotal = profileCallerSamples(t, plainProf, "callPlainBusy")
		out, err := runCmd(t, dir, bin, "busy", prof, strconv.FormatUint(fixtureIters, 10))
		if err != nil {
			t.Fatalf("fixture busy: %v\n%s", err, out)
		}
		lgoN, lgoTotal = profileCallerSamples(t, prof, "main.main")
		t.Logf("lgo caller %d of %d samples, plain caller %d of %d samples",
			lgoN, lgoTotal, plainN, plainTotal)
		if lgoN == 0 && plainN >= fastcallProfileFloorSamples {
			t.Fatalf("fast-call frames missing from profile: lgo caller %d of %d samples, plain caller %d of %d samples (floor %d)",
				lgoN, lgoTotal, plainN, plainTotal, fastcallProfileFloorSamples)
		}
		if lgoN >= fastcallProfileFloorSamples && plainN >= fastcallProfileFloorSamples {
			met = true
			break
		}
		if time.Now().After(deadline) {
			break
		}
		plainIters = growProfileWork(plainTotal, fastcallProfileFloorSamples, plainIters, maxPlainIters)
		fixtureIters = growProfileWork(lgoTotal, fastcallProfileFloorSamples, fixtureIters, maxFixtureIters)
	}
	if !met {
		if lgoTotal == 0 && plainTotal == 0 {
			if profileCanarySamples(t) > 0 {
				t.Fatalf("profiler works but yielded no lgo/plain samples: lgo caller %d of %d, plain caller %d of %d (floor %d)",
					lgoN, lgoTotal, plainN, plainTotal, fastcallProfileFloorSamples)
			}
			t.Skipf("CPU profiler yielded no samples in %v of collection; cannot attribute fast-call frames", fastcallProfileCap)
		}
		t.Fatalf("profile floor unmet: lgo caller %d of %d samples, plain caller %d of %d samples (floor %d)",
			lgoN, lgoTotal, plainN, plainTotal, fastcallProfileFloorSamples)
	}
	topPlain, _, _ := pprofNodes(t, self, plainProf)
	top, _, _ := pprofNodes(t, bin, prof)
	if !topIn(topPlain, plainWants) {
		t.Fatalf("plain top entry = %q, want the call site", topPlain)
	}
	if !topIn(top, lgoWants) {
		t.Fatalf("lgo top entry = %q, want the call site", top)
	}
	above, aboveTotal := profileCallerAboveLeafSamples(t, prof, "main.main", "cbusyf")
	if above < fastcallAttributionFloorSamples {
		t.Fatalf("%d of %d lgo samples hold the lgo call site above the fast C frame, want at least %d",
			above, aboveTotal, fastcallAttributionFloorSamples)
	}
}

func BenchmarkFastcall(b *testing.B) {
	requireCgo(b)
	b.Logf("logosfastcgo = %v", goexperiment.LogosFastCgo)
	var sink int32
	for b.Loop() {
		sink = callFastAdd(sink&((1<<30)-1), 1)
	}
	_ = sink
}

func BenchmarkFastcallSlow(b *testing.B) {
	requireCgo(b)
	var sink int32
	for b.Loop() {
		sink = callSlowNoescape(sink&((1<<30)-1), 1)
	}
	_ = sink
}

func TestFastcallViolationNeverHangs(t *testing.T) {
	requireCgo(t)
	dir := filepath.Join("testdata", "viol")
	for _, mode := range []string{"", "plain"} {
		bin := filepath.Join(t.TempDir(), "viol.exe")
		if text, err := runCmd(t, dir, goBin(t), "build", "-o", bin, "."); err != nil {
			t.Fatalf("violator build: %v\n%s", err, text)
		}
		ctx, cancel := context.WithTimeoutCause(t.Context(), 60*time.Second, errors.New("violator timed out"))
		out, err := exec.CommandContext(ctx, bin, mode).CombinedOutput()
		cancel()
		if err != nil {
			t.Fatalf("violator %q: %v\n%s", mode, err, out)
		}
		checkViolation(t, mode, string(out))
	}
}

func TestFastcallNoClockInFastBody(t *testing.T) {
	t.Run("disassembly", func(t *testing.T) {
		if !goexperiment.LogosFastCgo {
			t.Skip("runtime.logosCgocallFast is linked only with GOEXPERIMENT=logosfastcgo")
		}
		self := selfTestBin(t)
		body := objdumpFunc(t, self, "runtime\\.logosCgocallFast")
		if !strings.Contains(body, "TEXT runtime.logosCgocallFast(SB)") {
			t.Fatalf("no disassembly of runtime.logosCgocallFast:\n%s", body)
		}
		for _, want := range []string{"nanotime", "cputicks", "clock_gettime", "monotonic"} {
			if strings.Contains(strings.ToLower(body), want) {
				t.Fatalf("fast body references clock %q:\n%s", want, body)
			}
		}
	})
	ctx, cancel := context.WithTimeoutCause(t.Context(), 10*time.Second, errors.New("fast calls exceeded their time bound"))
	defer cancel()
	const calls = 20000
	for i := range calls {
		if ctx.Err() != nil {
			t.Fatalf("fast call %d of %d: %v", i, calls, context.Cause(ctx))
		}
		if got := callFastAdd(int32(i), 1); got != int32(i)+1 {
			t.Fatalf("fast_add(%d, 1) = %d, want %d", i, got, i+1)
		}
	}
}

func TestFastcallNumCgoCallBoundary(t *testing.T) {
	requireCgo(t)
	before := runtime.NumCgoCall()
	if got := callFastAdd(1, 1); got != 2 {
		t.Fatalf("fast_add = %d, want 2", got)
	}
	if mid := runtime.NumCgoCall(); mid != before+1 {
		t.Fatalf("after 1 call = %d, want %d", mid, before+1)
	}
	const n = 5
	for i := range n {
		if got := callFastAdd(int32(i), 1); got != int32(i)+1 {
			t.Fatalf("call %d = %d, want %d", i, got, i+1)
		}
	}
	if after := runtime.NumCgoCall(); after != before+1+n {
		t.Fatalf("after %d calls = %d, want %d", 1+n, after, before+1+n)
	}
}

func TestFastcallLockOSThreadParity(t *testing.T) {
	requireCgo(t)
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if got := callFastAdd(20, 22); got != 42 {
		t.Fatalf("fast_add on locked thread = %d, want 42", got)
	}
	done := make(chan int32, 1)
	go func() { done <- callFastAdd(1, 2) }()
	select {
	case got := <-done:
		if got != 3 {
			t.Fatalf("unlocked goroutine fast_add = %d, want 3", got)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("unlocked goroutine stuck while thread locked")
	}
}
