// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"math"
	"math/rand/v2"
	"regexp"
	"runtime"
	"runtime/metrics"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"internal/testenv"
)

// M19-S2 attacks (SPEC Sec 19.1, T-301.2, #161). Under greentea the
// background utilization goal follows the PLAN curve of live heap over
// heap goal, mark assist stays as backstop, and a build without a
// collector mode keeps the upstream pacing.

const (
	logosPlanFloor   = 0.25
	logosPlanCeiling = 0.75
	logosPlanRise    = 0.80
	logosPlanTop     = 0.95
	logosPlanSlack   = 1e-6
)

var logosPlanPoints = []struct{ u, util float64 }{
	{0.79, 0.25},
	{0.80, 0.25},
	{0.875, 0.50},
	{0.95, 0.75},
	{0.96, 0.75},
}

var logosPlanProcs = []int{1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 64}

func logosPlan(u float64) float64 {
	switch {
	case u < logosPlanRise:
		return logosPlanFloor
	case u >= logosPlanTop:
		return logosPlanCeiling
	}
	return logosPlanFloor + (logosPlanCeiling-logosPlanFloor)*(u-logosPlanRise)/(logosPlanTop-logosPlanRise)
}

func logosNearly(got, want, tol float64) bool {
	return math.Abs(got-want) <= tol
}

func newLogosController() *runtime.GCController {
	return runtime.NewGCController(100, math.MaxInt64)
}

const logosGreentea = "greentea"

func TestLogosElasticCurve(t *testing.T) {
	for _, p := range logosPlanPoints {
		if got := runtime.LogosElasticUtilization(p.u); !logosNearly(got, p.util, 1e-9) {
			t.Errorf("elastic(%v) = %v, want %v", p.u, got, p.util)
		}
	}
	prev := runtime.LogosElasticUtilization(0)
	for i := 1; i <= 12000; i++ {
		u := float64(i) * 1e-4
		got := runtime.LogosElasticUtilization(u)
		if want := logosPlan(u); !logosNearly(got, want, 1e-12) {
			t.Fatalf("elastic(%v) = %v, want %v", u, got, want)
		}
		if got < logosPlanFloor || got > logosPlanCeiling {
			t.Fatalf("elastic(%v) = %v outside [%v, %v]", u, got, logosPlanFloor, logosPlanCeiling)
		}
		if got < prev {
			t.Fatalf("elastic is not monotone at u=%v: %v after %v", u, got, prev)
		}
		if got-prev > 1e-4*(logosPlanCeiling-logosPlanFloor)/(logosPlanTop-logosPlanRise)+1e-12 {
			t.Fatalf("elastic jumps at u=%v: %v after %v", u, got, prev)
		}
		prev = got
	}
}

func TestLogosElasticGoalAtCycleStart(t *testing.T) {
	for _, procs := range logosPlanProcs {
		for _, p := range logosPlanPoints {
			c := newLogosController()
			c.LogosStartCycleAt(logosGreentea, p.u, procs)
			if !c.LogosElasticActive() {
				t.Fatalf("procs=%d u=%v: elastic pacing is not active under greentea", procs, p.u)
			}
			got := c.LogosBackgroundUtilization(procs, 0)
			if want := logosPlan(c.LogosLiveOverGoal()); !logosNearly(got, want, 1e-9) {
				t.Errorf("procs=%d u=%v: utilization %v, want curve value %v", procs, c.LogosLiveOverGoal(), got, want)
			}
			if !logosNearly(got, p.util, logosPlanSlack) {
				t.Errorf("procs=%d u=%v: utilization %v, want %v", procs, p.u, got, p.util)
			}
		}
	}
}

func TestLogosElasticGoalFollowsReviseInCycle(t *testing.T) {
	for _, procs := range logosPlanProcs {
		for _, p := range logosPlanPoints {
			c := newLogosController()
			c.LogosStartCycleAt(logosGreentea, 0.5, procs)
			if got := c.LogosBackgroundUtilization(procs, 0); !logosNearly(got, logosPlanFloor, 1e-9) {
				t.Fatalf("procs=%d: utilization at u=0.5 is %v, want %v", procs, got, logosPlanFloor)
			}
			c.LogosReviseAt(p.u)
			got := c.LogosBackgroundUtilization(procs, 0)
			if want := logosPlan(c.LogosLiveOverGoal()); !logosNearly(got, want, 1e-9) {
				t.Errorf("procs=%d u=%v: utilization %v, want curve value %v", procs, c.LogosLiveOverGoal(), got, want)
			}
			if !logosNearly(got, p.util, logosPlanSlack) {
				t.Errorf("procs=%d u=%v: utilization %v, want %v", procs, p.u, got, p.util)
			}
		}
	}
}

func TestLogosElasticNeverLowersInCycle(t *testing.T) {
	const procs = 8
	c := newLogosController()
	c.LogosStartCycleAt(logosGreentea, 0.5, procs)
	c.LogosReviseAt(0.96)
	peak := c.LogosBackgroundUtilization(procs, 0)
	if !logosNearly(peak, logosPlanCeiling, 1e-9) {
		t.Fatalf("utilization at u=0.96 is %v, want %v", peak, logosPlanCeiling)
	}
	for _, u := range []float64{0.9, 0.5, 0.0} {
		c.LogosReviseAt(u)
		if got := c.LogosBackgroundUtilization(procs, 0); got < peak-1e-9 {
			t.Errorf("utilization fell to %v at u=%v after reaching %v", got, u, peak)
		}
	}
	c.LogosStartCycleAt(logosGreentea, 0.5, procs)
	if got := c.LogosBackgroundUtilization(procs, 0); !logosNearly(got, logosPlanFloor, 1e-9) {
		t.Errorf("a new cycle starts at utilization %v, want %v", got, logosPlanFloor)
	}
}

func TestLogosElasticWorkerAccounting(t *testing.T) {
	rng := rand.New(rand.NewPCG(19, 2))
	for _, procs := range []int{2, 3, 4, 8, 12, 16} {
		c := newLogosController()
		c.LogosStartCycleAt(logosGreentea, 0.4, procs)
		var running int64
		var peakU, lastTarget = 0.0, int64(0)
		limit := int64(float64(procs)*logosPlanCeiling + 1e-6)
		for step := range 4000 {
			switch rng.IntN(3) {
			case 0:
				u := rng.Float64() * 1.2
				c.LogosReviseAt(u)
				peakU = math.Max(peakU, c.LogosLiveOverGoal())
			case 1:
				if c.LogosStartDedicated() {
					running++
				}
			case 2:
				if running > 0 {
					c.LogosStopDedicated()
					running--
				}
			}
			target := c.LogosDedicatedTarget()
			if needed := c.LogosDedicatedNeeded(); needed < 0 || needed+running != target {
				t.Fatalf("procs=%d step=%d: needed %d + running %d != target %d", procs, step, needed, running, target)
			}
			if target < lastTarget || target > limit {
				t.Fatalf("procs=%d step=%d: target %d after %d, limit %d", procs, step, target, lastTarget, limit)
			}
			lastTarget = target
			if f, bound := c.LogosFractionalGoal(), 1/float64(int64(procs)-target); f < 0 || f >= bound {
				t.Fatalf("procs=%d step=%d: fractional goal %v outside [0, %v)", procs, step, f, bound)
			}
		}
		if got, want := c.LogosBackgroundUtilization(procs, running), logosPlan(math.Max(peakU, 0.4)); !logosNearly(got, want, 1e-8) {
			t.Errorf("procs=%d: final utilization %v, want curve at peak u %v", procs, got, want)
		}
	}
}

func TestLogosElasticConcurrentRaisesAddEachWorkerOnce(t *testing.T) {
	const procs, goroutines, steps, rounds = 64, 16, 4000, 50
	const span = logosPlanTop + 0.02 - logosPlanRise
	for round := range rounds {
		c := newLogosController()
		c.LogosStartCycleAt(logosGreentea, 0.5, procs)
		ratios := make([]float64, goroutines)
		var mu sync.Mutex
		var running int64
		adjust := func(g int) {
			mu.Lock()
			defer mu.Unlock()
			if g%2 == 0 {
				if c.LogosStartDedicated() {
					running++
				}
			} else if running > 0 {
				c.LogosStopDedicated()
				running--
			}
		}
		release := make(chan struct{})
		var wg sync.WaitGroup
		for g := range goroutines {
			wg.Go(func() {
				<-release
				for k := range steps {
					u := logosPlanRise + span*float64(k*goroutines+g)/float64(goroutines*steps)
					ratios[g] = max(ratios[g], c.LogosElasticStepAt(u))
					if k%4 == 0 {
						adjust(g + k)
					}
				}
			})
		}
		close(release)
		wg.Wait()
		peak := slices.Max(ratios)
		wantTarget := int64(float64(procs)*logosPlan(peak) + 1e-9)
		needed, target := c.LogosDedicatedNeeded(), c.LogosDedicatedTarget()
		if target != wantTarget {
			t.Fatalf("round %d: target %d, want %d at peak u %v", round, target, wantTarget, peak)
		}
		if needed+running != target {
			t.Fatalf("round %d: needed %d + running %d != target %d", round, needed, running, target)
		}
		if got, want := c.LogosBackgroundUtilization(procs, running), logosPlan(peak); !logosNearly(got, want, 1e-8) {
			t.Fatalf("round %d: utilization %v, want %v at peak u %v", round, got, want, peak)
		}
	}
}

func TestLogosElasticKeepsAssistBackstop(t *testing.T) {
	assistAt := func(mode string, u float64) float64 {
		c := newLogosController()
		c.LogosStartCycleAt(mode, 0.5, 8)
		c.LogosReviseAt(u)
		return c.AssistWorkPerByte()
	}
	prev := 0.0
	for _, u := range []float64{0.5, 0.8, 0.9, 0.95, 0.96, 1.0, 1.2} {
		elastic, upstream := assistAt(logosGreentea, u), assistAt("", u)
		if elastic != upstream {
			t.Errorf("u=%v: assist work per byte %v under greentea, %v upstream", u, elastic, upstream)
		}
		if u >= logosPlanTop && !(elastic > 0) {
			t.Errorf("u=%v: assist work per byte %v, want assists to keep running", u, elastic)
		}
		if elastic < prev {
			t.Errorf("u=%v: assist work per byte %v fell below %v at a lower u", u, elastic, prev)
		}
		prev = elastic
	}
}

func logosUpstreamSplit(procs int) (int64, float64) {
	total := float64(procs) * runtime.GCBackgroundUtilization
	dedicated := int64(total + 0.5)
	if e := float64(dedicated)/total - 1; e < -0.3 || e > 0.3 {
		if float64(dedicated) > total {
			dedicated--
		}
		return dedicated, (total - float64(dedicated)) / float64(procs)
	}
	return dedicated, 0
}

func TestLogosEmptyModeKeepsUpstreamPacing(t *testing.T) {
	for _, procs := range logosPlanProcs {
		wantD, wantF := logosUpstreamSplit(procs)
		c := newLogosController()
		c.LogosStartCycleAt("", 0.5, procs)
		if c.LogosElasticActive() {
			t.Fatalf("procs=%d: elastic pacing is active with an empty mode", procs)
		}
		for _, u := range []float64{0.5, 0.8, 0.9, 0.95, 0.99, 1.1} {
			c.LogosReviseAt(u)
			if got := c.LogosDedicatedNeeded(); got != wantD {
				t.Errorf("procs=%d u=%v: %d dedicated workers, upstream has %d", procs, u, got, wantD)
			}
			if got := c.LogosFractionalGoal(); got != wantF {
				t.Errorf("procs=%d u=%v: fractional goal %v, upstream has %v", procs, u, got, wantF)
			}
		}
	}
}

func TestLogosOtherModesKeepUpstreamPacing(t *testing.T) {
	for _, mode := range []string{"zgc", "immix", "wasm", "batch", "epsilon"} {
		t.Run(mode, func(t *testing.T) {
			c := newLogosController()
			c.LogosStartCycleAt(mode, 0.9, 8)
			c.LogosReviseAt(0.99)
			wantD, wantF := logosUpstreamSplit(8)
			if c.LogosElasticActive() || c.LogosDedicatedNeeded() != wantD || c.LogosFractionalGoal() != wantF {
				t.Errorf("mode %q changes the upstream pacing", mode)
			}
		})
	}
}

var logosElasticLine = regexp.MustCompile(`pacer: elastic u=(\S+) util=(\S+) workers=(\d+)\+(\S+)`)

func logosPacerRun(t *testing.T, prog string, flags ...string) string {
	t.Helper()
	exe, err := buildTestProg(t, "testprog", flags...)
	if err != nil {
		t.Fatal(err)
	}
	out := runBuiltTestProg(t, exe, prog, "GODEBUG=gcpacertrace=1", "GOGC=100", "GOMEMLIMIT=off")
	if !strings.Contains(out, "OK\n") {
		t.Fatalf("helper program %s did not finish:\n%s", prog, out)
	}
	if !strings.Contains(out, "pacer: assist ratio=") {
		t.Fatalf("helper program %s printed no pacer trace:\n%s", prog, out)
	}
	return out
}

func logosPacerTrace(t *testing.T, flags ...string) string {
	t.Helper()
	return logosPacerRun(t, "LogosPacerElastic", flags...)
}

func logosFloat(t *testing.T, s string) float64 {
	t.Helper()
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		t.Fatalf("bad number %q in pacer trace: %v", s, err)
	}
	return v
}

func logosCheckElasticTrace(t *testing.T, out string) {
	t.Helper()
	lines := logosElasticLine.FindAllStringSubmatch(out, -1)
	if len(lines) < 3 {
		t.Fatalf("pacer trace has %d elastic lines, want at least 3:\n%s", len(lines), out)
	}
	peak := 0.0
	for _, m := range lines {
		u, util := logosFloat(t, m[1]), logosFloat(t, m[2])
		dedicated, fractional := logosFloat(t, m[3]), logosFloat(t, m[4])
		if want := logosPlan(u); !logosNearly(util, want, 1e-5) {
			t.Errorf("trace u=%v: utilization %v, want %v", u, util, want)
		}
		if got := (dedicated + fractional*(4-dedicated)) / 4; !logosNearly(got, util, 1e-5) {
			t.Errorf("trace u=%v: workers %v+%v give utilization %v, want %v", u, dedicated, fractional, got, util)
		}
		peak = math.Max(peak, util)
	}
	if peak <= logosPlanFloor+1e-3 {
		t.Errorf("no cycle started above the floor utilization; trace:\n%s", out)
	}
}

func TestLogosElasticLinkerModeInBuiltProgram(t *testing.T) {
	logosCheckElasticTrace(t, logosPacerTrace(t, "-ldflags=-X=runtime.logosGCMode=greentea"))
}

func TestLogosElasticGCFlagInBuiltProgram(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	probe := testenv.Command(t, testenv.GoToolPath(t), "build", "-gc=greentea", "-n", "-o", t.TempDir()+"/x", "runtime/testdata/testprog")
	probe.Dir = "testdata/testprog"
	if out, err := probe.CombinedOutput(); err != nil && strings.Contains(string(out), "flag provided but not defined") {
		t.Skip("go build -gc is delivered by M19-S1")
	}
	logosCheckElasticTrace(t, logosPacerTrace(t, "-gc=greentea"))
}

func TestLogosElasticAssistsRunInBuiltProgram(t *testing.T) {
	out := logosPacerRun(t, "LogosPacerAssist", "-ldflags=-X=runtime.logosGCMode=greentea")
	if !logosElasticLine.MatchString(out) {
		t.Fatalf("the program did not pace elastically:\n%s", out)
	}
}

func TestLogosPlainBuildKeepsUpstreamPacing(t *testing.T) {
	out := logosPacerTrace(t)
	if logosElasticLine.MatchString(out) {
		t.Fatalf("a build without -gc paces elastically:\n%s", out)
	}
}

type logosBenchNode struct {
	left, right *logosBenchNode
	pad         [6]uint64
}

func logosBenchTree(depth int) *logosBenchNode {
	if depth == 0 {
		return nil
	}
	return &logosBenchNode{left: logosBenchTree(depth - 1), right: logosBenchTree(depth - 1)}
}

func logosReadMetrics(b *testing.B, names ...string) []metrics.Value {
	b.Helper()
	s := make([]metrics.Sample, len(names))
	for i, name := range names {
		s[i].Name = name
	}
	metrics.Read(s)
	values := make([]metrics.Value, len(s))
	for i := range s {
		if s[i].Value.Kind() == metrics.KindBad {
			b.Fatalf("metric %s is not supported", s[i].Name)
		}
		values[i] = s[i].Value
	}
	return values
}

func logosAssistAndTotalCPU(b *testing.B) (assist, total float64) {
	b.Helper()
	v := logosReadMetrics(b, "/cpu/classes/gc/mark/assist:cpu-seconds", "/cpu/classes/total:cpu-seconds")
	return v[0].Float64(), v[1].Float64()
}

func logosGCCycles(b *testing.B) uint64 {
	b.Helper()
	return logosReadMetrics(b, "/gc/cycles/total:gc-cycles")[0].Uint64()
}

func logosAllocBurst(workers, perWorker int) {
	var wg sync.WaitGroup
	sinks := make([]*logosBenchNode, workers)
	for w := range workers {
		wg.Go(func() {
			var keep *logosBenchNode
			for range perWorker {
				keep = &logosBenchNode{left: keep}
				if keep.left != nil && keep.left.left != nil {
					keep.left.left = nil
				}
			}
			sinks[w] = keep
		})
	}
	wg.Wait()
	runtime.KeepAlive(sinks)
}

// BenchmarkLogosAssistShare reports the share of CPU time spent in mark
// assist, from /cpu/classes/gc/mark/assist:cpu-seconds over
// /cpu/classes/total:cpu-seconds, for the upstream pacing and for greentea.
// The target for greentea is a lower share; it is recorded, not asserted.
func BenchmarkLogosAssistShare(b *testing.B) {
	for _, mode := range []struct{ name, gc string }{{"upstream", ""}, {"greentea", "greentea"}} {
		b.Run(mode.name, func(b *testing.B) {
			old := runtime.LogosPacerSetMode(mode.gc)
			b.Cleanup(func() { runtime.LogosPacerSetMode(old) })
			live := logosBenchTree(19)
			runtime.GC()
			assist0, total0 := logosAssistAndTotalCPU(b)
			cycles0 := logosGCCycles(b)
			b.ResetTimer()
			for range b.N {
				logosAllocBurst(runtime.GOMAXPROCS(0), 1<<18)
			}
			b.StopTimer()
			runtime.GC()
			assist1, total1 := logosAssistAndTotalCPU(b)
			if total1 > total0 {
				b.ReportMetric((assist1-assist0)/(total1-total0), "assist-share")
			}
			b.ReportMetric(float64(logosGCCycles(b)-cycles0)/float64(b.N), "gc-cycles/op")
			var ms runtime.MemStats
			runtime.ReadMemStats(&ms)
			b.ReportMetric(float64(ms.HeapSys)/(1<<20), "heap-sys-MB")
			runtime.KeepAlive(live)
		})
	}
}
