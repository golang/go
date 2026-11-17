// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package goparity

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// Each scenario starts a new test process so GOMAXPROCS is set before package
// initialization. The tests observe only the existing execution helpers.
func TestRunBinaryExecutionQueue(t *testing.T) {
	if scenario := os.Getenv("GOPARITY_EXECUTION_QUEUE_CASE"); scenario != "" {
		if got := runtime.GOMAXPROCS(0); got != 2 {
			t.Fatalf("scenario GOMAXPROCS = %d, want 2", got)
		}
		dir := os.Getenv("GOPARITY_EXECUTION_QUEUE_DIR")
		switch scenario {
		case "shared_queue_budget":
			testExecutionQueueBudget(t, dir)
		case "start_error_release":
			testExecutionQueueStartError(t, dir)
		case "timeout_release":
			testExecutionQueueTimeout(t, dir)
		default:
			t.Fatalf("unknown execution queue scenario %q", scenario)
		}
		return
	}

	fork := forkTool(t)
	buildDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(buildDir, "go.mod"), []byte("module queuehelper\n\ngo 1.27\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(buildDir, "main.go"), []byte(executionQueueHelper), 0o644); err != nil {
		t.Fatal(err)
	}
	helper := filepath.Join(buildDir, "helper")
	if out, err := goBuild(fork.bin, buildDir, []string{"build", "-o", helper, "."}, baseEnv(forkCache(), fork.goroot), 5*time.Minute); err != nil {
		t.Fatalf("building execution queue helper: %v\n%s", err, out)
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, scenario := range []string{"shared_queue_budget", "start_error_release", "timeout_release"} {
		t.Run(scenario, func(t *testing.T) {
			dir := t.TempDir()
			programs := []string{"holder0", "holder1", "control", "queued", "probe", "timeout0", "timeout1", "timeout2"}
			for _, name := range programs {
				cloneBinary(t, helper, filepath.Join(dir, name))
			}
			t.Cleanup(func() {
				// Release surviving helpers even if the scenario process hit its
				// watchdog before its own cleanup could run.
				for _, name := range programs {
					if err := os.WriteFile(filepath.Join(dir, name+".release"), nil, 0o600); err != nil {
						t.Errorf("releasing %s: %v", name, err)
					}
				}
			})
			ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, self,
				"-test.run=^TestRunBinaryExecutionQueue$", "-test.count=1",
				"-test.parallel=1", "-test.timeout=40s", "-test.v")
			cmd.Env = appendEnv(os.Environ(), "GOMAXPROCS=2",
				"GOPARITY_EXECUTION_QUEUE_CASE="+scenario,
				"GOPARITY_EXECUTION_QUEUE_DIR="+dir)
			cmd.WaitDelay = pipeWaitDelay
			setNewProcessGroup(cmd)
			cmd.Cancel = func() error {
				killProcessGroup(cmd)
				return nil
			}
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("execution queue scenario %s: %v (watchdog: %v)\n%s", scenario, err, ctx.Err(), out)
			}
			t.Logf("%s", out)
		})
	}
}

type executionQueueResult struct {
	obs observation
	err error
}

func testExecutionQueueBudget(t *testing.T, dir string) {
	var wg sync.WaitGroup
	t.Cleanup(func() {
		for _, name := range []string{"holder0", "holder1"} {
			if err := os.WriteFile(filepath.Join(dir, name+".release"), nil, 0o600); err != nil {
				t.Errorf("releasing %s: %v", name, err)
			}
		}
		wg.Wait()
	})
	startRow := func(name string) <-chan executionQueueResult {
		done := make(chan executionQueueResult, 1)
		wg.Go(func() {
			rows, err := runAll(buildResult{outDir: dir, progs: []string{name}}, "4", nil)
			obs, ok := rows[name]
			if err == nil && (!ok || len(rows) != 1) {
				err = fmt.Errorf("row %s returned %d observations, named observation present: %t", name, len(rows), ok)
			}
			done <- executionQueueResult{obs, err}
		})
		return done
	}

	row0 := startRow("holder0")
	executionQueueReady(t, filepath.Join(dir, "holder0"), row0)
	obs, err := runBinaryTimeout(filepath.Join(dir, "control"), "4", 2*time.Second, corpusNorm)
	executionQueueOK(t, "control with one holder", executionQueueResult{obs, err})
	t.Log("capacity N-1: one holder leaves one runnable slot")

	row1 := startRow("holder1")
	executionQueueReady(t, filepath.Join(dir, "holder1"), row1)
	t.Log("capacity N: two independent runAll rows are ready")
	queued := make(chan executionQueueResult, 1)
	submitted := make(chan struct{})
	wg.Go(func() {
		close(submitted)
		obs, err := runBinaryTimeout(filepath.Join(dir, "queued"), "4", 2*time.Second, corpusNorm)
		queued <- executionQueueResult{obs, err}
	})
	<-submitted
	hold := time.NewTimer(3 * time.Second)
	defer hold.Stop()
	select {
	case got := <-queued:
		t.Fatalf("third call returned while two independent rows still held their slots: observation=%+v error=%v", got.obs, got.err)
	case got := <-row0:
		t.Fatalf("first holder returned before release: observation=%+v error=%v", got.obs, got.err)
	case got := <-row1:
		t.Fatalf("second holder returned before release: observation=%+v error=%v", got.obs, got.err)
	case <-hold.C:
	}
	if _, err := os.Stat(filepath.Join(dir, "queued.ready")); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("third child started before a slot was released: marker stat = %v", err)
	}
	t.Log("capacity N+1: third call remained queued longer than its two-second child budget")
	if err := os.WriteFile(filepath.Join(dir, "holder0.release"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	executionQueueOK(t, "first released row", executionQueueReceive(t, row0))
	executionQueueOK(t, "queued call after release", executionQueueReceive(t, queued))
	executionQueueReady(t, filepath.Join(dir, "queued"), nil)
	if err := os.WriteFile(filepath.Join(dir, "holder1.release"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	executionQueueOK(t, "second released row", executionQueueReceive(t, row1))
	obs, err = runBinaryTimeout(filepath.Join(dir, "probe"), "4", 2*time.Second, corpusNorm)
	executionQueueOK(t, "call after successful rows", executionQueueResult{obs, err})
}

func testExecutionQueueStartError(t *testing.T, dir string) {
	for i := range 3 {
		_, err := runBinaryTimeout(filepath.Join(dir, fmt.Sprintf("missing%d", i)), "4", 2*time.Second, corpusNorm)
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("missing executable %d: got %v, want file-not-found error", i, err)
		}
		t.Logf("start failure %d returned its file-not-found error", i)
	}
	obs, err := runBinaryTimeout(filepath.Join(dir, "probe"), "4", 2*time.Second, corpusNorm)
	executionQueueOK(t, "call after N+1 start failures", executionQueueResult{obs, err})
}

func testExecutionQueueTimeout(t *testing.T, dir string) {
	var wg sync.WaitGroup
	t.Cleanup(func() {
		for i := range 3 {
			if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("timeout%d.release", i)), nil, 0o600); err != nil {
				t.Errorf("releasing timeout child %d: %v", i, err)
			}
		}
		wg.Wait()
	})
	for i := range 3 {
		path := filepath.Join(dir, fmt.Sprintf("timeout%d", i))
		done := make(chan executionQueueResult, 1)
		wg.Go(func() {
			obs, err := runBinaryTimeout(path, "4", 2*time.Second, corpusNorm)
			done <- executionQueueResult{obs, err}
		})
		pid := executionQueueReady(t, path, done)
		got := executionQueueReceive(t, done)
		if got.err == nil || !strings.Contains(got.err.Error(), "goparity run timeout") {
			t.Fatalf("ready child %d: got %v, want goparity run timeout", i, got.err)
		}
		if got.obs.stdout != "" || got.obs.stderr != "" {
			t.Fatalf("timed-out child %d produced unexpected output: %+v", i, got.obs)
		}
		if processGroupKillSupported() && !procGroupGone(pid, 4*time.Second) {
			t.Fatalf("timed-out child %d process group %d survived", i, pid)
		}
		t.Logf("timeout %d returned after child readiness", i)
	}
	obs, err := runBinaryTimeout(filepath.Join(dir, "probe"), "4", 2*time.Second, corpusNorm)
	executionQueueOK(t, "call after N+1 real child timeouts", executionQueueResult{obs, err})
}

func executionQueueReady(t *testing.T, path string, done <-chan executionQueueResult) int {
	t.Helper()
	timer := time.NewTimer(5 * time.Second)
	defer timer.Stop()
	tick := time.NewTicker(10 * time.Millisecond)
	defer tick.Stop()
	for {
		data, err := os.ReadFile(path + ".ready")
		if err == nil {
			pid, err := strconv.Atoi(string(data))
			if err != nil || pid <= 0 {
				t.Fatalf("invalid readiness marker for %s: %q (%v)", path, data, err)
			}
			return pid
		}
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatal(err)
		}
		select {
		case got := <-done:
			t.Fatalf("%s returned before readiness: observation=%+v error=%v", path, got.obs, got.err)
		case <-timer.C:
			t.Fatalf("%s did not write its readiness marker", path)
		case <-tick.C:
		}
	}
}

func executionQueueReceive(t *testing.T, done <-chan executionQueueResult) executionQueueResult {
	t.Helper()
	select {
	case got := <-done:
		return got
	case <-time.After(5 * time.Second):
		t.Fatal("execution call did not return after its release or child deadline")
		return executionQueueResult{}
	}
}

func executionQueueOK(t *testing.T, label string, got executionQueueResult) {
	t.Helper()
	if got.err != nil || got.obs.exit != 0 || got.obs.stdout != "ok\n" || got.obs.stderr != "" {
		t.Fatalf("%s: observation=%+v error=%v, want exit 0, ok newline, and empty stderr", label, got.obs, got.err)
	}
}

const executionQueueHelper = `package main

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

func fail(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}

func main() {
	if got := os.Getenv("GOMAXPROCS"); got != "4" {
		fail(fmt.Errorf("child GOMAXPROCS = %q, want 4", got))
	}
	path := os.Args[0]
	ready := path + ".ready"
	if err := os.WriteFile(ready+".tmp", []byte(strconv.Itoa(os.Getpid())), 0o600); err != nil {
		fail(err)
	}
	if err := os.Rename(ready+".tmp", ready); err != nil {
		fail(err)
	}
	name := filepath.Base(path)
	if strings.HasPrefix(name, "holder") || strings.HasPrefix(name, "timeout") {
		deadline := time.Now().Add(30 * time.Second)
		for {
			_, err := os.Stat(path + ".release")
			if err == nil {
				break
			}
			if !errors.Is(err, fs.ErrNotExist) {
				fail(err)
			}
			if time.Now().After(deadline) {
				fail(fmt.Errorf("%s was never released", name))
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
	fmt.Print("ok\n")
}
`
