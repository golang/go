// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"cmd/compile/internal/logos/facts"
)

const wasmNoYieldProgram = `package main

import (
	"fmt"
	"runtime"
	"sync"
)

//go:noinline
func leaf(a, b int) int { return a*b + 3 }

//go:noinline
func straight(x int) int { return leaf(x, 1) + leaf(x, 2) }

func withDefer() (r int) {
	defer func() { r += 100 }()
	return straight(3)
}

//go:noinline
func park() {
	ready := make(chan struct{})
	go func() { close(ready) }()
	<-ready
}

func panicParks() (s string) {
	defer func() {
		recover()
		park()
		s = "woke"
	}()
	var e error
	_ = e.Error()
	return "unreachable"
}

//go:noinline
func fault(s []int, i int) int { return s[i] }

func boundsParks() (s string) {
	defer func() {
		recover()
		park()
		s = "bounds"
	}()
	return fmt.Sprint(fault(nil, 3))
}

func main() {
	fmt.Println(straight(5), leaf(2, 3), withDefer())
	var wg sync.WaitGroup
	var mu sync.Mutex
	total := 0
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				mu.Lock()
				total += straight(j) % 7
				mu.Unlock()
				if j%10 == 0 {
					runtime.Gosched()
				}
			}
		}()
	}
	wg.Wait()
	ch := make(chan int)
	done := make(chan int)
	go func() {
		s := 0
		for v := range ch {
			s += leaf(v, 2)
		}
		done <- s
	}()
	for i := 0; i < 50; i++ {
		ch <- i
	}
	close(ch)
	fmt.Println(total, <-done, panicParks(), boundsParks())
}
`

const wasmNoYieldWant = "21 9 115\n6040 2600 woke bounds\n"

const wasmSpeedProgram = `package main

import "time"

//go:noinline
func leaf(a, b int) int { return a*b + 3 }

//go:noinline
func mix(x int) int { return leaf(x, 1) ^ leaf(x, 2) }

func main() {
	start := time.Now()
	s := 0
	for i := 0; i < 20000000; i++ {
		s += mix(i)
	}
	println(s, time.Since(start).Nanoseconds())
}
`

const wasmTrapProgram = `package main

import "runtime"

//go:noinline
func park() { runtime.Gosched() }

//go:noinline
func caller() int {
	park()
	return 7
}

func main() { println(caller()) }
`

func wasmNode(tb testing.TB) string {
	tb.Helper()
	p, err := exec.LookPath("node")
	if err != nil {
		tb.Skip("node not found: the wasm run is blocked (A-NODE)")
	}
	return p
}

func wasmLogos(tb testing.TB) string {
	tb.Helper()
	p := filepath.Join(filepath.Dir(testenv.GoToolPath(tb)), "logos")
	if _, err := os.Stat(p); err != nil {
		tb.Skipf("logos binary not found beside the go tool: %v", err)
	}
	return p
}

func wasmGoEnv(cmd *exec.Cmd, goos string) []string {
	var env []string
	for _, kv := range cmd.Environ() {
		switch {
		case strings.HasPrefix(kv, "GOOS="), strings.HasPrefix(kv, "GOARCH="),
			strings.HasPrefix(kv, "GOMAXPROCS="), strings.HasPrefix(kv, "GOFLAGS="),
			strings.HasPrefix(kv, "GOWORK="), strings.HasPrefix(kv, "GO111MODULE="):
		default:
			env = append(env, kv)
		}
	}
	return append(env, "GOOS="+goos, "GOARCH=wasm", "GOMAXPROCS=1", "GOFLAGS=", "GOWORK=off", "GO111MODULE=on")
}

func buildWasm(tb testing.TB, tool, goos, src, gcflags string) string {
	tb.Helper()
	dir := tb.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module wasmfix\n\ngo 1.27\n"), 0o666); err != nil {
		tb.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(src), 0o666); err != nil {
		tb.Fatal(err)
	}
	out := filepath.Join(dir, "app.wasm")
	args := []string{"build", "-o", out}
	if gcflags != "" {
		args = append(args, "-gcflags="+gcflags)
	}
	args = append(args, ".")
	cmd := testenv.Command(tb, tool, args...)
	cmd.Dir = dir
	cmd.Env = wasmGoEnv(cmd, goos)
	if b, err := cmd.CombinedOutput(); err != nil {
		tb.Fatalf("%s %s: %v\n%s", tool, strings.Join(args, " "), err, b)
	}
	return out
}

func runWasmNode(tb testing.TB, wasm string) (string, error) {
	tb.Helper()
	node := wasmNode(tb)
	host := filepath.Join(testenv.GOROOT(tb), "lib", "wasm", "wasm_exec_node.js")
	cmd := testenv.Command(tb, node, host, wasm)
	cmd.Env = wasmGoEnv(cmd, "js")
	var out bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &out
	err := cmd.Run()
	return out.String(), err
}

func TestWasmProgramRunsUnderNode(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	wasmNode(t)
	for name, build := range map[string]func() string{
		"resume points everywhere": func() string {
			return buildWasm(t, testenv.GoToolPath(t), "js", wasmNoYieldProgram, "")
		},
		"selective resume points": func() string {
			return buildWasm(t, wasmLogos(t), "js", wasmNoYieldProgram, "")
		},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := runWasmNode(t, build())
			if err != nil || got != wasmNoYieldWant {
				t.Fatalf("run: %v\ngot %q\nwant %q", err, got, wasmNoYieldWant)
			}
		})
	}
}

func TestWasmWrongMayYieldTraps(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	wasmNode(t)
	right, err := runWasmNode(t, buildWasm(t, wasmLogos(t), "js", wasmTrapProgram, ""))
	if err != nil || right != "7\n" {
		t.Fatalf("correct facts: %v, output %q, want 7", err, right)
	}
	wrong := buildWasm(t, wasmLogos(t), "js", wasmTrapProgram, "-logos.wasmassume=main.park")
	got, err := runWasmNode(t, wrong)
	if err == nil {
		t.Fatalf("a wrong MayYield=0 callee that unwinds must trap, ran to completion with %q", got)
	}
	if !strings.Contains(got, "unreachable") {
		t.Fatalf("trap output lacks the unreachable instruction: %q", got)
	}
}

func goExports(tb testing.TB, tool string, pkgs ...string) map[string]string {
	tb.Helper()
	args := append([]string{"list", "-export", "-deps", "-f", "{{if .Export}}{{.ImportPath}} {{.Export}}{{end}}"}, pkgs...)
	cmd := testenv.Command(tb, tool, args...)
	cmd.Env = wasmGoEnv(cmd, "js")
	out, err := cmd.Output()
	if err != nil {
		tb.Fatalf("%s %s: %v", tool, strings.Join(args, " "), err)
	}
	archives := map[string]string{}
	for line := range strings.Lines(string(out)) {
		if path, file, ok := strings.Cut(strings.TrimSpace(line), " "); ok {
			archives[path] = file
		}
	}
	return archives
}

func readFacts(tb testing.TB, archive string) *facts.Facts {
	tb.Helper()
	f, err := os.Open(archive)
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()
	member, err := facts.FindMember(f)
	if err != nil {
		tb.Fatalf("%s: %v", archive, err)
	}
	if member == nil {
		return nil
	}
	got, err := facts.Decode(member)
	if err != nil {
		tb.Fatalf("%s: %v", archive, err)
	}
	return got
}

func TestWasmGoPackagesCarryMayYieldFacts(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	runtimeFacts := readFacts(t, goExports(t, testenv.GoToolPath(t), "runtime")["runtime"])
	if runtimeFacts == nil {
		t.Fatal("the runtime archive for js/wasm has no __.LOGOS member")
	}
	if !runtimeFacts.GoPackage {
		t.Error("a Go package must carry the lang go record so importers do not read it as loGo/s")
	}
	yield := map[string]bool{}
	for _, fn := range runtimeFacts.Fns {
		if fn.HasMayYield {
			yield[fn.Object] = fn.MayYield
		}
	}
	for _, name := range []string{"gopark", "Gosched", "chanrecv1", "chansend1", "selectgo", "mallocgc"} {
		if got, ok := yield[name]; !ok || !got {
			t.Errorf("runtime.%s: MayYield = %v (recorded %v), want 1", name, got, ok)
		}
	}
}

func BenchmarkWasmAsyncifySize(b *testing.B) {
	testenv.MustHaveGoBuild(b)
	full := buildWasm(b, testenv.GoToolPath(b), "js", wasmNoYieldProgram, "")
	selective := buildWasm(b, wasmLogos(b), "js", wasmNoYieldProgram, "")
	fullSize, selectiveSize := wasmSize(b, full), wasmSize(b, selective)
	for b.Loop() {
		wasmSize(b, selective)
	}
	b.ReportMetric(float64(fullSize), "full-bytes")
	b.ReportMetric(float64(selectiveSize), "selective-bytes")
	b.ReportMetric(float64(fullSize)/float64(selectiveSize), "full/selective")
}

func wasmSize(tb testing.TB, path string) int64 {
	tb.Helper()
	st, err := os.Stat(path)
	if err != nil {
		tb.Fatal(err)
	}
	return st.Size()
}

func BenchmarkWasmAsyncifySpeed(b *testing.B) {
	testenv.MustHaveGoBuild(b)
	wasmNode(b)
	full := buildWasm(b, testenv.GoToolPath(b), "js", wasmSpeedProgram, "")
	selective := buildWasm(b, wasmLogos(b), "js", wasmSpeedProgram, "")
	var fullRuns, selectiveRuns []float64
	for b.Loop() {
		fullRuns = append(fullRuns, loopNanos(b, full))
		selectiveRuns = append(selectiveRuns, loopNanos(b, selective))
	}
	b.ReportMetric(quantile(fullRuns, 0.5)/quantile(selectiveRuns, 0.5), "full/selective-p50")
	for name, runs := range map[string][]float64{"full": fullRuns, "selective": selectiveRuns} {
		b.ReportMetric(quantile(runs, 0.5)/1e6, name+"-p50-ms")
		b.ReportMetric(quantile(runs, 0.95)/1e6, name+"-p95-ms")
		b.ReportMetric(quantile(runs, 1)/1e6, name+"-max-ms")
	}
}

func loopNanos(tb testing.TB, wasm string) float64 {
	tb.Helper()
	out, err := runWasmNode(tb, wasm)
	if err != nil {
		tb.Fatalf("run: %v\n%s", err, out)
	}
	fields := strings.Fields(out)
	if len(fields) != 2 {
		tb.Fatalf("speed program printed %q, want a sum and a duration", out)
	}
	ns, err := strconv.ParseFloat(fields[1], 64)
	if err != nil {
		tb.Fatalf("speed program duration %q: %v", fields[1], err)
	}
	return ns
}

func quantile(xs []float64, q float64) float64 {
	sorted := slices.Sorted(slices.Values(xs))
	return sorted[min(len(sorted)-1, int(q*float64(len(sorted))))]
}

var wasmShareCorpus = []string{"fmt", "strconv", "strings", "sort", "bytes", "encoding/json", "regexp", "bufio", "path/filepath", "container/heap"}

func BenchmarkWasmSyncPureShare(b *testing.B) {
	testenv.MustHaveGoBuild(b)
	pure, yield := mayYieldCounts(b, wasmLogos(b), wasmShareCorpus...)
	offPure, offYield := mayYieldCounts(b, testenv.GoToolPath(b), wasmShareCorpus...)
	if pure+yield == 0 || offPure+offYield == 0 {
		b.Fatal("no MayYield facts found in the standard packages")
	}
	for b.Loop() {
		_ = pure + yield
	}
	b.ReportMetric(100*float64(pure)/float64(pure+yield), "pure-%")
	b.ReportMetric(100*float64(yield)/float64(pure+yield), "yield-%")
	b.ReportMetric(float64(pure+yield), "functions")
	b.ReportMetric(100*float64(offPure)/float64(offPure+offYield), "pure-%-resume-all")
}

func mayYieldCounts(tb testing.TB, tool string, pkgs ...string) (pure, yield int) {
	tb.Helper()
	for _, archive := range goExports(tb, tool, pkgs...) {
		if got := readFacts(tb, archive); got != nil {
			p, y := countMayYield(got)
			pure, yield = pure+p, yield+y
		}
	}
	return pure, yield
}

func countMayYield(f *facts.Facts) (pure, yield int) {
	for _, fn := range f.Fns {
		switch {
		case !fn.HasMayYield:
		case fn.MayYield:
			yield++
		default:
			pure++
		}
	}
	return pure, yield
}

const wasmYieldFreeProgram = `package main

import "fmt"

//go:noinline
func nilderef(p *int) int { return *p }

//go:noinline
func leaf2(x int) int { return x + 1 }

//go:noinline
func bigframe(v int) int {
	var buf [8192]byte
	buf[0] = byte(leaf2(v))
	buf[8191] = byte(v >> 8)
	return int(buf[0]) + int(buf[8191]) + leaf2(v)
}

func main() {
	x := 41
	fmt.Println(nilderef(&x), bigframe(x))
}
`

const wasmYieldFreeWant = "41 84\n"

func TestWasmYieldFreeSourcesRunUnderNode(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	wasmNode(t)
	for name, tool := range map[string]string{
		"plain go": testenv.GoToolPath(t),
		"logos":    wasmLogos(t),
	} {
		t.Run(name, func(t *testing.T) {
			got, err := runWasmNode(t, buildWasm(t, tool, "js", wasmYieldFreeProgram, ""))
			if err != nil || got != wasmYieldFreeWant {
				t.Fatalf("run: %v\ngot %q\nwant %q", err, got, wasmYieldFreeWant)
			}
		})
	}
}
