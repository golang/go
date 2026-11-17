package goparity

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestParityAt08FanoutProbe(t *testing.T) {
	const program = "at08_recursion_bound"
	const copies = 24
	if got := runtime.NumCPU(); got != 2 {
		t.Fatalf("probe requires root-imposed two-CPU affinity; runtime.NumCPU() = %d", got)
	}
	inputs := []struct {
		path string
		sum  string
	}{
		{"parity_test.go", "1be848e4e5f94f76d3484e12b253d4eaad81990a89635a59cf59f86fa1149bb4"},
		{filepath.Join("testdata", "run", program, "main.go"), "e1aa8d47d3fe4ddc0d43be98ec361c1ad2dcf9ca8711f1feaa49afc2081bc359"},
	}
	for _, input := range inputs {
		if got := parityAt08ProbeHash(t, input.path); got != input.sum {
			t.Fatalf("probe input %s hash = %s, want %s", input.path, got, input.sum)
		}
		t.Logf("probe input path=%s sha256=%s", input.path, input.sum)
	}
	t.Logf("probe CPUs=%d parentGOMAXPROCS=%d children=%d childGOMAXPROCS=4 budget=10s",
		runtime.NumCPU(), runtime.GOMAXPROCS(0), 2*copies)

	fork := forkTool(t)
	up, upOn := upstreamTool(t)
	if !upOn {
		t.Fatal("the fanout probe requires the pinned upstream oracle")
	}
	var k6 config
	for _, c := range forkConfigs {
		if c.id == "K6" {
			k6 = c
			break
		}
	}
	var u3n upConfig
	for _, u := range upConfigs {
		if u.id == "U3N" {
			u3n = u
			break
		}
	}
	if k6.id == "" || u3n.id == "" {
		t.Fatal("the frozen K6 or U3N configuration is absent")
	}

	programs := []string{program}
	base := t.TempDir()
	start := time.Now()
	forkBuild := buildFork(fork, base, programs, k6)
	t.Logf("probe build id=K6 elapsed=%s error=%v", time.Since(start), forkBuild.err)
	if forkBuild.err != nil {
		t.Fatalf("K6 build: %v\n%s", forkBuild.err, forkBuild.output)
	}
	start = time.Now()
	upBuild := buildUpstream(up, base, programs, u3n)
	t.Logf("probe build id=U3N elapsed=%s error=%v", time.Since(start), upBuild.err)
	if upBuild.err != nil {
		t.Fatalf("U3N build: %v\n%s", upBuild.err, upBuild.output)
	}
	rows := []struct {
		id    string
		args  []string
		build buildResult
		sum   string
	}{
		{id: "K6", args: k6.args, build: forkBuild},
		{id: "U3N", args: u3n.args, build: upBuild},
	}
	for i := range rows {
		row := &rows[i]
		if !slices.Equal(row.build.progs, programs) {
			t.Fatalf("%s built %v, want %v", row.id, row.build.progs, programs)
		}
		original := filepath.Join(row.build.outDir, program)
		row.sum = parityAt08ProbeHash(t, original)
		info, err := os.Stat(original)
		if err != nil {
			t.Fatal(err)
		}
		row.build.progs = nil
		for n := 0; n < copies; n++ {
			name := fmt.Sprintf("at08_%s_%02d", row.id, n)
			path := filepath.Join(row.build.outDir, name)
			if err := os.Link(original, path); err != nil {
				t.Fatal(err)
			}
			linkInfo, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			if !os.SameFile(info, linkInfo) {
				t.Fatalf("%s is not a hardlink to %s", path, original)
			}
			row.build.progs = append(row.build.progs, name)
		}
		t.Logf("probe binary id=%s sha256=%s hardlinks=%d args=%q", row.id, row.sum, len(row.build.progs), row.args)
	}

	want := observation{stdout: "ok\n"}
	type rowResult struct {
		obs     map[string]observation
		err     error
		elapsed time.Duration
	}
	unbounded := make([]rowResult, len(rows))
	t.Logf("probe phase=unbounded begin=%s", time.Now().UTC().Format(time.RFC3339Nano))
	start = time.Now()
	var rowWG sync.WaitGroup
	for i := range rows {
		rowWG.Go(func() {
			row := rows[i]
			begin := time.Now()
			obs, err := runAll(row.build, "4", row.args)
			unbounded[i] = rowResult{obs, err, time.Since(begin)}
		})
	}
	rowWG.Wait()
	t.Logf("probe phase=unbounded elapsed=%s", time.Since(start))
	sawTimeout := false
	for i, result := range unbounded {
		id := rows[i].id
		t.Logf("probe unbounded id=%s completed=%d/%d elapsed=%s first_error=%v",
			id, len(result.obs), copies, result.elapsed, result.err)
		if result.err != nil {
			if !strings.HasSuffix(result.err.Error(), ": goparity run timeout") {
				t.Errorf("unbounded %s exposed a non-timeout error: %v", id, result.err)
			} else {
				sawTimeout = true
			}
		}
		names := make([]string, 0, len(result.obs))
		for name := range result.obs {
			names = append(names, name)
		}
		slices.Sort(names)
		for _, name := range names {
			obs := result.obs[name]
			t.Logf("probe unbounded retained id=%s name=%s exit=%d stdout=%q stderr=%q",
				id, name, obs.exit, obs.stdout, obs.stderr)
			if corpusDefect(obs) || !obsEqual(obs, want) {
				t.Errorf("unbounded %s/%s: unexpected observation %s", id, name, obsDesc(obs))
			}
		}
		if got := parityAt08ProbeHash(t, filepath.Join(rows[i].build.outDir, program)); got != rows[i].sum {
			t.Fatalf("%s binary changed after unbounded phase: got %s, want %s", id, got, rows[i].sum)
		}
	}

	type task struct {
		row  int
		name string
	}
	type taskResult struct {
		task    task
		obs     observation
		err     error
		elapsed time.Duration
	}
	jobs := make(chan task)
	results := make(chan taskResult)
	var workers sync.WaitGroup
	t.Logf("probe phase=bounded workers=2 begin=%s", time.Now().UTC().Format(time.RFC3339Nano))
	start = time.Now()
	for range 2 {
		workers.Go(func() {
			for job := range jobs {
				row := rows[job.row]
				begin := time.Now()
				obs, err := runBinary(filepath.Join(row.build.outDir, job.name), "4", row.args)
				results <- taskResult{job, obs, err, time.Since(begin)}
			}
		})
	}
	go func() {
		for i := 0; i < copies; i++ {
			for row := range rows {
				jobs <- task{row, rows[row].build.progs[i]}
			}
		}
		close(jobs)
		workers.Wait()
		close(results)
	}()
	completed := 0
	for result := range results {
		completed++
		id := rows[result.task.row].id
		t.Logf("probe bounded id=%s name=%s elapsed=%s error=%v exit=%d stdout=%q stderr=%q",
			id, result.task.name, result.elapsed, result.err, result.obs.exit, result.obs.stdout, result.obs.stderr)
		if result.err != nil {
			t.Errorf("bounded %s/%s: %v", id, result.task.name, result.err)
			continue
		}
		if corpusDefect(result.obs) || !obsEqual(result.obs, want) {
			t.Errorf("bounded %s/%s: unexpected observation %s", id, result.task.name, obsDesc(result.obs))
		}
	}
	t.Logf("probe phase=bounded completed=%d/%d elapsed=%s", completed, 2*copies, time.Since(start))
	if completed != 2*copies {
		t.Errorf("bounded phase completed %d jobs, want %d", completed, 2*copies)
	}
	if !sawTimeout {
		t.Error("contention witness absent: neither unbounded row reported the unchanged runtime timeout")
	}
	for _, row := range rows {
		if got := parityAt08ProbeHash(t, filepath.Join(row.build.outDir, program)); got != row.sum {
			t.Errorf("%s binary changed: got %s, want %s", row.id, got, row.sum)
		}
	}
	for _, input := range inputs {
		if got := parityAt08ProbeHash(t, input.path); got != input.sum {
			t.Errorf("probe input %s changed: got %s, want %s", input.path, got, input.sum)
		}
	}
}
