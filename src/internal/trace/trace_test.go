// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace_test

import (
	"bufio"
	"bytes"
	"fmt"
	"internal/race"
	"internal/testenv"
	"internal/trace"
	"internal/trace/testtrace"
	"internal/trace/version"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestTraceAnnotations(t *testing.T) {
	testTraceProg(t, "annotations.go", func(t *testing.T, tb, _ []byte, _ string) {
		type evDesc struct {
			kind trace.EventKind
			task trace.TaskID
			args []string
		}
		want := []evDesc{
			{trace.EventTaskBegin, trace.TaskID(1), []string{"task0"}},
			{trace.EventRegionBegin, trace.TaskID(1), []string{"region0"}},
			{trace.EventRegionBegin, trace.TaskID(1), []string{"region1"}},
			{trace.EventLog, trace.TaskID(1), []string{"key0", "0123456789abcdef"}},
			{trace.EventRegionEnd, trace.TaskID(1), []string{"region1"}},
			{trace.EventRegionEnd, trace.TaskID(1), []string{"region0"}},
			{trace.EventTaskEnd, trace.TaskID(1), []string{"task0"}},
			//  Currently, pre-existing region is not recorded to avoid allocations.
			{trace.EventRegionBegin, trace.BackgroundTask, []string{"post-existing region"}},
		}
		r, err := trace.NewReader(bytes.NewReader(tb))
		if err != nil {
			t.Error(err)
		}
		for {
			ev, err := r.ReadEvent()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			for i, wantEv := range want {
				if wantEv.kind != ev.Kind() {
					continue
				}
				match := false
				switch ev.Kind() {
				case trace.EventTaskBegin, trace.EventTaskEnd:
					task := ev.Task()
					match = task.ID == wantEv.task && task.Type == wantEv.args[0]
				case trace.EventRegionBegin, trace.EventRegionEnd:
					reg := ev.Region()
					match = reg.Task == wantEv.task && reg.Type == wantEv.args[0]
				case trace.EventLog:
					log := ev.Log()
					match = log.Task == wantEv.task && log.Category == wantEv.args[0] && log.Message == wantEv.args[1]
				}
				if match {
					want[i] = want[len(want)-1]
					want = want[:len(want)-1]
					break
				}
			}
		}
		if len(want) != 0 {
			for _, ev := range want {
				t.Errorf("no match for %s TaskID=%d Args=%#v", ev.kind, ev.task, ev.args)
			}
		}
	})
}

func TestTraceAnnotationsStress(t *testing.T) {
	testTraceProg(t, "annotations-stress.go", nil)
}

func TestTraceCgoCallback(t *testing.T) {
	testenv.MustHaveCGO(t)

	switch runtime.GOOS {
	case "plan9", "windows":
		t.Skipf("cgo callback test requires pthreads and is not supported on %s", runtime.GOOS)
	}
	testTraceProg(t, "cgo-callback.go", nil)
}

func TestTraceCPUProfile(t *testing.T) {
	testTraceProg(t, "cpu-profile.go", func(t *testing.T, tb, stderr []byte, _ string) {
		// Parse stderr which has a CPU profile summary, if everything went well.
		// (If it didn't, we shouldn't even make it here.)
		scanner := bufio.NewScanner(bytes.NewReader(stderr))
		pprofSamples := 0
		pprofStacks := make(map[string]int)
		for scanner.Scan() {
			var stack string
			var samples int
			_, err := fmt.Sscanf(scanner.Text(), "%s\t%d", &stack, &samples)
			if err != nil {
				t.Fatalf("failed to parse CPU profile summary in stderr: %s\n\tfull:\n%s", scanner.Text(), stderr)
			}
			pprofStacks[stack] = samples
			pprofSamples += samples
		}
		if err := scanner.Err(); err != nil {
			t.Fatalf("failed to parse CPU profile summary in stderr: %v", err)
		}
		if pprofSamples == 0 {
			t.Skip("CPU profile did not include any samples while tracing was active")
		}

		// Examine the execution tracer's view of the CPU profile samples. Filter it
		// to only include samples from the single test goroutine. Use the goroutine
		// ID that was recorded in the events: that should reflect getg().m.curg,
		// same as the profiler's labels (even when the M is using its g0 stack).
		totalTraceSamples := 0
		traceSamples := 0
		traceStacks := make(map[string]int)
		r, err := trace.NewReader(bytes.NewReader(tb))
		if err != nil {
			t.Error(err)
		}
		var hogRegion *trace.Event
		var hogRegionClosed bool
		for {
			ev, err := r.ReadEvent()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			if ev.Kind() == trace.EventRegionBegin && ev.Region().Type == "cpuHogger" {
				hogRegion = &ev
			}
			if ev.Kind() == trace.EventStackSample {
				totalTraceSamples++
				if hogRegion != nil && ev.Goroutine() == hogRegion.Goroutine() {
					traceSamples++
					var fns []string
					for frame := range ev.Stack().Frames() {
						if frame.Func != "runtime.goexit" {
							fns = append(fns, fmt.Sprintf("%s:%d", frame.Func, frame.Line))
						}
					}
					stack := strings.Join(fns, "|")
					traceStacks[stack]++
				}
			}
			if ev.Kind() == trace.EventRegionEnd && ev.Region().Type == "cpuHogger" {
				hogRegionClosed = true
			}
		}
		if hogRegion == nil {
			t.Fatalf("execution trace did not identify cpuHogger goroutine")
		} else if !hogRegionClosed {
			t.Fatalf("execution trace did not close cpuHogger region")
		}

		// The execution trace may drop CPU profile samples if the profiling buffer
		// overflows. Based on the size of profBufWordCount, that takes a bit over
		// 1900 CPU samples or 19 thread-seconds at a 100 Hz sample rate. If we've
		// hit that case, then we definitely have at least one full buffer's worth
		// of CPU samples, so we'll call that success.
		overflowed := totalTraceSamples >= 1900
		if traceSamples < pprofSamples {
			t.Logf("execution trace did not include all CPU profile samples; %d in profile, %d in trace", pprofSamples, traceSamples)
			if !overflowed {
				t.Fail()
			}
		}

		for stack, traceSamples := range traceStacks {
			pprofSamples := pprofStacks[stack]
			delete(pprofStacks, stack)
			if traceSamples < pprofSamples {
				t.Logf("execution trace did not include all CPU profile samples for stack %q; %d in profile, %d in trace",
					stack, pprofSamples, traceSamples)
				if !overflowed {
					t.Fail()
				}
			}
		}
		for stack, pprofSamples := range pprofStacks {
			t.Logf("CPU profile included %d samples at stack %q not present in execution trace", pprofSamples, stack)
			if !overflowed {
				t.Fail()
			}
		}

		if t.Failed() {
			t.Logf("execution trace CPU samples:")
			for stack, samples := range traceStacks {
				t.Logf("%d: %q", samples, stack)
			}
			t.Logf("CPU profile:\n%s", stderr)
		}
	})
}

func TestTraceFutileWakeup(t *testing.T) {
	testTraceProg(t, "futile-wakeup.go", func(t *testing.T, tb, _ []byte, _ string) {
		// Check to make sure that no goroutine in the "special" trace region
		// ends up blocking, unblocking, then immediately blocking again.
		//
		// The goroutines are careful to call runtime.Gosched in between blocking,
		// so there should never be a clean block/unblock on the goroutine unless
		// the runtime was generating extraneous events.
		const (
			entered = iota
			blocked
			runnable
			running
		)
		gs := make(map[trace.GoID]int)
		seenSpecialGoroutines := false
		r, err := trace.NewReader(bytes.NewReader(tb))
		if err != nil {
			t.Error(err)
		}
		for {
			ev, err := r.ReadEvent()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			// Only track goroutines in the special region we control, so runtime
			// goroutines don't interfere (it's totally valid in traces for a
			// goroutine to block, run, and block again; that's not what we care about).
			if ev.Kind() == trace.EventRegionBegin && ev.Region().Type == "special" {
				seenSpecialGoroutines = true
				gs[ev.Goroutine()] = entered
			}
			if ev.Kind() == trace.EventRegionEnd && ev.Region().Type == "special" {
				delete(gs, ev.Goroutine())
			}
			// Track state transitions for goroutines we care about.
			//
			// The goroutines we care about will advance through the state machine
			// of entered -> blocked -> runnable -> running. If in the running state
			// we block, then we have a futile wakeup. Because of the runtime.Gosched
			// on these specially marked goroutines, we should end up back in runnable
			// first. If at any point we go to a different state, switch back to entered
			// and wait for the next time the goroutine blocks.
			if ev.Kind() != trace.EventStateTransition {
				continue
			}
			st := ev.StateTransition()
			if st.Resource.Kind != trace.ResourceGoroutine {
				continue
			}
			id := st.Resource.Goroutine()
			state, ok := gs[id]
			if !ok {
				continue
			}
			_, new := st.Goroutine()
			switch state {
			case entered:
				if new == trace.GoWaiting {
					state = blocked
				} else {
					state = entered
				}
			case blocked:
				if new == trace.GoRunnable {
					state = runnable
				} else {
					state = entered
				}
			case runnable:
				if new == trace.GoRunning {
					state = running
				} else {
					state = entered
				}
			case running:
				if new == trace.GoWaiting {
					t.Fatalf("found futile wakeup on goroutine %d", id)
				} else {
					state = entered
				}
			}
			gs[id] = state
		}
		if !seenSpecialGoroutines {
			t.Fatal("did not see a goroutine in a the region 'special'")
		}
	})
}

func TestTraceGCStress(t *testing.T) {
	testTraceProg(t, "gc-stress.go", nil)
}

func TestTraceGOMAXPROCS(t *testing.T) {
	testTraceProg(t, "gomaxprocs.go", nil)
}

func TestTraceStacks(t *testing.T) {
	testTraceProg(t, "stacks.go", func(t *testing.T, tb, _ []byte, godebug string) {
		type frame struct {
			fn   string
			line int
		}
		type evDesc struct {
			kind   trace.EventKind
			match  string
			frames []frame
		}
		// mainLine is the line number of `func main()` in testprog/stacks.go.
		const mainLine = 21
		want := []evDesc{
			{trace.EventStateTransition, "Goroutine Running->Runnable", []frame{
				{"runtime.Gosched", 0},
				{"main.main", mainLine + 87},
			}},
			{trace.EventStateTransition, "Goroutine NotExist->Runnable", []frame{
				{"main.main", mainLine + 11},
			}},
			{trace.EventStateTransition, "Goroutine Running->Waiting", []frame{
				{"runtime.block", 0},
				{"main.main.func1", 0},
			}},
			{trace.EventStateTransition, "Goroutine Running->Waiting", []frame{
				{"runtime.chansend1", 0},
				{"main.main.func2", 0},
			}},
			{trace.EventStateTransition, "Goroutine Running->Waiting", []frame{
				{"runtime.chanrecv1", 0},
				{"main.main.func3", 0},
			}},
			{trace.EventStateTransition, "Goroutine Running->Waiting", []frame{
				{"runtime.chanrecv1", 0},
				{"main.main.func4", 0},
			}},
			{trace.EventStateTransition, "Goroutine Waiting->Runnable", []frame{
				{"runtime.chansend1", 0},
				{"main.main", mainLine + 89},
			}},
			{trace.EventStateTransition, "Goroutine Running->Waiting", []frame{
				{"runtime.chansend1", 0},
				{"main.main.func5", 0},
			}},
			{trace.EventStateTransition, "Goroutine Waiting->Runnable", []frame{
				{"runtime.chanrecv1", 0},
				{"main.main", mainLine + 90},
			}},
			{trace.EventStateTransition, "Goroutine Running->Waiting", []frame{
				{"runtime.selectgo", 0},
				{"main.main.func6", 0},
			}},
			{trace.EventStateTransition, "Goroutine Waiting->Runnable", []frame{
				{"runtime.selectgo", 0},
				{"main.main", mainLine + 91},
			}},
			{trace.EventStateTransition, "Goroutine Running->Waiting", []frame{
				{"sync.(*Mutex).Lock", 0},
				{"main.main.func7", 0},
			}},
			{trace.EventStateTransition, "Goroutine Waiting->Runnable", []frame{
				{"sync.(*Mutex).Unlock", 0},
				{"main.main", 0},
			}},
			{trace.EventStateTransition, "Goroutine Running->Waiting", []frame{
				{"sync.(*WaitGroup).Wait", 0},
				{"main.main.func8", 0},
			}},
			{trace.EventStateTransition, "Goroutine Waiting->Runnable", []frame{
				{"sync.(*WaitGroup).Add", 0},
				{"sync.(*WaitGroup).Done", 0},
				{"main.main", mainLine + 96},
			}},
			{trace.EventStateTransition, "Goroutine Running->Waiting", []frame{
				{"sync.(*Cond).Wait", 0},
				{"main.main.func9", 0},
			}},
			{trace.EventStateTransition, "Goroutine Waiting->Runnable", []frame{
				{"sync.(*Cond).Signal", 0},
				{"main.main", 0},
			}},
			{trace.EventStateTransition, "Goroutine Running->Waiting", []frame{
				{"time.Sleep", 0},
				{"main.main", 0},
			}},
			{trace.EventMetric, "/sched/gomaxprocs:threads", []frame{
				{"runtime.startTheWorld", 0}, // this is when the current gomaxprocs is logged.
				{"runtime.startTheWorldGC", 0},
				{"runtime.GOMAXPROCS", 0},
				{"main.main", 0},
			}},
		}
		asyncPreemptOff := strings.Contains(godebug, "asyncpreemptoff=1")
		if asyncPreemptOff {
			// Only check for syncPreemptPoint if asynchronous preemption is disabled.
			// Otherwise, the syncPreemptPoint goroutine might be exclusively async
			// preempted, causing this test to flake.
			syncPreemptEv := evDesc{trace.EventStateTransition, "Goroutine Running->Runnable", []frame{
				{"main.syncPreemptPoint", 0},
				{"main.main.func12", 0},
			}}
			want = append(want, syncPreemptEv)
		}
		stress := strings.Contains(godebug, "traceadvanceperiod=0")
		if !stress {
			// Only check for this stack if !stress because traceAdvance alone could
			// allocate enough memory to trigger a GC if called frequently enough.
			// This might cause the runtime.GC call we're trying to match against to
			// coalesce with an active GC triggered this by traceAdvance. In that case
			// we won't have an EventRangeBegin event that matches the stace trace we're
			// looking for, since runtime.GC will not have triggered the GC.
			gcEv := evDesc{trace.EventRangeBegin, "GC concurrent mark phase", []frame{
				{"runtime.GC", 0},
				{"main.main", 0},
			}}
			want = append(want, gcEv)
		}
		if runtime.GOOS != "windows" && runtime.GOOS != "plan9" {
			want = append(want, []evDesc{
				{trace.EventStateTransition, "Goroutine Running->Waiting", []frame{
					{"internal/poll.(*FD).Accept", 0},
					{"net.(*netFD).accept", 0},
					{"net.(*TCPListener).accept", 0},
					{"net.(*TCPListener).Accept", 0},
					{"main.main.func10", 0},
				}},
				{trace.EventStateTransition, "Goroutine Running->Syscall", []frame{
					{"syscall.read", 0},
					{"syscall.Read", 0},
					{"internal/poll.ignoringEINTRIO", 0},
					{"internal/poll.(*FD).Read", 0},
					{"os.(*File).read", 0},
					{"os.(*File).Read", 0},
					{"main.main.func11", 0},
				}},
			}...)
			if runtime.GOOS == "darwin" {
				want[len(want)-1].frames = append([]frame{{"syscall.syscall", 0}}, want[len(want)-1].frames...)
			}
		}
		stackMatches := func(stk trace.Stack, frames []frame) bool {
			for i, f := range slices.Collect(stk.Frames()) {
				if f.Func != frames[i].fn {
					return false
				}
				if line := uint64(frames[i].line); line != 0 && line != f.Line {
					return false
				}
			}
			return true
		}
		r, err := trace.NewReader(bytes.NewReader(tb))
		if err != nil {
			t.Error(err)
		}
		for {
			ev, err := r.ReadEvent()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			for i, wantEv := range want {
				if wantEv.kind != ev.Kind() {
					continue
				}
				match := false
				switch ev.Kind() {
				case trace.EventStateTransition:
					st := ev.StateTransition()
					str := ""
					switch st.Resource.Kind {
					case trace.ResourceGoroutine:
						old, new := st.Goroutine()
						str = fmt.Sprintf("%s %s->%s", st.Resource.Kind, old, new)
					}
					match = str == wantEv.match
				case trace.EventRangeBegin:
					rng := ev.Range()
					match = rng.Name == wantEv.match
				case trace.EventMetric:
					metric := ev.Metric()
					match = metric.Name == wantEv.match
				}
				match = match && stackMatches(ev.Stack(), wantEv.frames)
				if match {
					want[i] = want[len(want)-1]
					want = want[:len(want)-1]
					break
				}
			}
		}
		if len(want) != 0 {
			for _, ev := range want {
				t.Errorf("no match for %s Match=%s Stack=%#v", ev.kind, ev.match, ev.frames)
			}
		}
	})
}

func TestTraceStress(t *testing.T) {
	switch runtime.GOOS {
	case "js", "wasip1":
		t.Skip("no os.Pipe on " + runtime.GOOS)
	}
	testTraceProg(t, "stress.go", checkReaderDeterminism)
}

func TestTraceStressStartStop(t *testing.T) {
	switch runtime.GOOS {
	case "js", "wasip1":
		t.Skip("no os.Pipe on " + runtime.GOOS)
	}
	testTraceProg(t, "stress-start-stop.go", nil)
}

func TestTraceManyStartStop(t *testing.T) {
	testTraceProg(t, "many-start-stop.go", nil)
}

func TestTraceWaitOnPipe(t *testing.T) {
	switch runtime.GOOS {
	case "dragonfly", "freebsd", "linux", "netbsd", "openbsd", "solaris":
		testTraceProg(t, "wait-on-pipe.go", nil)
		return
	}
	t.Skip("no applicable syscall.Pipe on " + runtime.GOOS)
}

func TestTraceIterPull(t *testing.T) {
	testTraceProg(t, "iter-pull.go", nil)
}

func checkReaderDeterminism(t *testing.T, tb, _ []byte, _ string) {
	events := func() []trace.Event {
		var evs []trace.Event

		r, err := trace.NewReader(bytes.NewReader(tb))
		if err != nil {
			t.Error(err)
		}
		for {
			ev, err := r.ReadEvent()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			evs = append(evs, ev)
		}

		return evs
	}

	evs1 := events()
	evs2 := events()

	if l1, l2 := len(evs1), len(evs2); l1 != l2 {
		t.Fatalf("re-reading trace gives different event count (%d != %d)", l1, l2)
	}
	for i, ev1 := range evs1 {
		ev2 := evs2[i]
		if s1, s2 := ev1.String(), ev2.String(); s1 != s2 {
			t.Errorf("re-reading trace gives different event %d:\n%s\n%s\n", i, s1, s2)
			break
		}
	}
}

func testTraceProg(t *testing.T, progName string, extra func(t *testing.T, trace, stderr []byte, godebug string)) {
	testenv.MustHaveGoRun(t)

	// Check if we're on a builder.
	onBuilder := testenv.Builder() != ""
	onOldBuilder := !strings.Contains(testenv.Builder(), "gotip") && !strings.Contains(testenv.Builder(), "go1")

	if progName == "cgo-callback.go" && onBuilder && !onOldBuilder &&
		runtime.GOOS == "freebsd" && runtime.GOARCH == "amd64" && race.Enabled {
		t.Skip("test fails on freebsd-amd64-race in LUCI; see go.dev/issue/71556")
	}

	testPath := filepath.Join("./testdata/testprog", progName)
	testName := progName
	runTest := func(t *testing.T, stress bool, extraGODEBUG string) {
		// Build the program.
		binFile, err := os.CreateTemp("", progName)
		if err != nil {
			t.Fatalf("failed to create temporary output file: %v", err)
		}
		bin := binFile.Name()
		binFile.Close()
		t.Cleanup(func() {
			os.Remove(bin)
		})
		buildCmd := testenv.CommandContext(t, t.Context(), testenv.GoToolPath(t), "build", "-o", bin)
		if race.Enabled {
			buildCmd.Args = append(buildCmd.Args, "-race")
		}
		buildCmd.Args = append(buildCmd.Args, testPath)
		buildOutput, err := buildCmd.CombinedOutput()
		if err != nil {
			t.Fatalf("failed to build %s: %v: output:\n%s", testPath, err, buildOutput)
		}

		// Run the program and capture the trace, which is always written to stdout.
		cmd := testenv.CommandContext(t, t.Context(), bin)

		// Add a stack ownership check. This is cheap enough for testing.
		godebug := "tracecheckstackownership=1"
		if stress {
			// Advance a generation constantly to stress the tracer.
			godebug += ",traceadvanceperiod=0"
		}
		if extraGODEBUG != "" {
			// Add extra GODEBUG flags.
			godebug += "," + extraGODEBUG
		}
		cmd.Env = append(cmd.Env, "GODEBUG="+godebug)
		if _, ok := os.LookupEnv("GOTRACEBACK"); !ok {
			// Unless overriden, set GOTRACEBACK=crash.
			cmd.Env = append(cmd.Env, "GOTRACEBACK=crash")
		}

		// Capture stdout and stderr.
		//
		// The protocol for these programs is that stdout contains the trace data
		// and stderr is an expectation in string format.
		var traceBuf, errBuf bytes.Buffer
		cmd.Stdout = &traceBuf
		cmd.Stderr = &errBuf
		// Run the program.
		if err := cmd.Run(); err != nil {
			if errBuf.Len() != 0 {
				t.Logf("stderr: %s", string(errBuf.Bytes()))
			}
			t.Fatal(err)
		}
		tb := traceBuf.Bytes()

		// Test the trace and the parser.
		v := testtrace.NewValidator()
		v.GoVersion = version.Current
		if runtime.GOOS == "windows" && stress {
			// Under stress mode we're constantly advancing trace generations.
			// Windows' clock granularity is too coarse to guarantee monotonic
			// timestamps for monotonic and wall clock time in this case, so
			// skip the checks.
			v.SkipClockSnapshotChecks()
		}
		testReader(t, bytes.NewReader(tb), v, testtrace.ExpectSuccess())

		// Run some extra validation.
		if !t.Failed() && extra != nil {
			extra(t, tb, errBuf.Bytes(), godebug)
		}

		// Dump some more information on failure.
		if t.Failed() || *dumpTraces {
			suffix := func(stress bool) string {
				if stress {
					return "stress"
				}
				return "default"
			}
			testtrace.Dump(t, fmt.Sprintf("%s.%s", testName, suffix(stress)), tb, *dumpTraces)
		}
	}
	t.Run("Default", func(t *testing.T) {
		runTest(t, false, "")
	})
	t.Run("AsyncPreemptOff", func(t *testing.T) {
		if testing.Short() && runtime.NumCPU() < 2 {
			t.Skip("skipping trace async preempt off tests in short mode")
		}
		runTest(t, false, "asyncpreemptoff=1")
	})
	t.Run("Stress", func(t *testing.T) {
		if testing.Short() {
			t.Skip("skipping trace stress tests in short mode")
		}
		runTest(t, true, "")
	})
	t.Run("AllocFree", func(t *testing.T) {
		if testing.Short() {
			t.Skip("skipping trace alloc/free tests in short mode")
		}
		runTest(t, false, "traceallocfree=1")
	})
}
