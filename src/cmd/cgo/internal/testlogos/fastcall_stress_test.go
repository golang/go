// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlogos

import (
	"fmt"
	"internal/goexperiment"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func stressBusyExpect(iters uint64) uint64 {
	var x uint64
	for i := uint64(0); i < iters; i++ {
		x += i*6364136223846793005 + 1442695040888963407
	}
	return x
}

func stressPerSetting(nsettings int) time.Duration {
	if testing.Short() {
		return time.Second
	}
	if s := os.Getenv("FASTCGO_SOAK"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			return time.Duration(n) * time.Second / time.Duration(nsettings)
		}
	}
	return 3 * time.Second
}

func armStressWatchdog(d time.Duration) func() {
	debug.SetTraceback("all")
	t := time.AfterFunc(d, func() {
		panic("fastcall stress watchdog: soak did not finish in time")
	})
	return func() { t.Stop() }
}

type stressState struct {
	stop      chan struct{}
	errs      chan string
	closeOnce sync.Once
}

func newStressState() *stressState {
	return &stressState{stop: make(chan struct{}), errs: make(chan string, 1)}
}

func (s *stressState) finish() {
	s.closeOnce.Do(func() { close(s.stop) })
}

func (s *stressState) fail(format string, args ...any) {
	select {
	case s.errs <- fmt.Sprintf(format, args...):
	default:
	}
	s.finish()
}

func (s *stressState) done() bool {
	select {
	case <-s.stop:
		return true
	default:
		return false
	}
}

func stressVerifyAdd(s *stressState, a, b int32) bool {
	a &= 0x7fff
	b &= 0x7fff
	if got := callFastAdd(a, b); got != a+b {
		s.fail("fast_add(%d, %d) = %d, want %d", a, b, got, a+b)
		return false
	}
	return true
}

func stressMixedWorker(s *stressState, seed uint64) {
	x := seed | 1
	var buf [64]byte
	for {
		if s.done() {
			return
		}
		switch x & 7 {
		case 0, 1:
			if !stressVerifyAdd(s, int32(x>>11), int32(x>>29)) {
				return
			}
		case 2:
			v := int32(x >> 7)
			if got := callFastRead(&v); got != v {
				s.fail("read_int = %d, want %d", got, v)
				return
			}
		case 3:
			n := int(x>>5)%len(buf) + 1
			var want int32
			for i := range n {
				buf[i] = byte(x>>uint(i%8)) + byte(i*31)
				want += int32(buf[i])
			}
			if got := callFastSlice(buf[:n]); got != want {
				s.fail("sum_bytes = %d, want %d", got, want)
				return
			}
		case 4:
			if got := callFastBusy(500); got != stressBusyExpect(500) {
				s.fail("busy_spin(500) = %d, want %d", got, stressBusyExpect(500))
				return
			}
		case 5:
			growStack(300)
			if !stressVerifyAdd(s, int32(x>>13)^-12345, int32(x>>31)^6789) {
				return
			}
		case 6:
			if got := callStressUsleep(200); got != 200 {
				s.fail("stress_usleep(200) = %d, want 200", got)
				return
			}
			if !stressVerifyAdd(s, int32(x>>17), 7) {
				return
			}
		default:
			v := int32(x >> 9)
			if got := callFastRead(&v); got != v {
				s.fail("read_int = %d, want %d", got, v)
				return
			}
			if !stressVerifyAdd(s, v, -v) {
				return
			}
		}
		x = x*6364136223846793005 + 1442695040888963407
	}
}

func stressLockedWorker(s *stressState, seed uint64) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	x := seed | 1
	for {
		if s.done() {
			return
		}
		if !stressVerifyAdd(s, int32(x>>11), int32(x>>23)) {
			return
		}
		v := int32(x >> 5)
		if got := callFastRead(&v); got != v {
			s.fail("locked read_int = %d, want %d", got, v)
			return
		}
		x = x*6364136223846793005 + 1442695040888963407
	}
}

func stressGCChurn(s *stressState) {
	for {
		if s.done() {
			return
		}
		runtime.GC()
		b := make([]byte, 1<<20)
		for i := range b {
			b[i] = byte(i)
		}
		runtime.KeepAlive(b)
	}
}

func stressPreempt(s *stressState, sink *atomic.Uint64) {
	var x uint64
	for {
		if s.done() {
			sink.Add(x)
			return
		}
		for i := range 20000 {
			x += uint64(i) * 2654435761
		}
		runtime.Gosched()
	}
}

func stressChurner(s *stressState, ids *atomic.Uint64) {
	var inner sync.WaitGroup
	defer inner.Wait()
	for {
		if s.done() {
			return
		}
		for range 4 {
			v := int32(ids.Add(1))
			inner.Go(func() {
				if got := callFastAdd(v, 1); got != v+1 {
					s.fail("churn fast_add(%d, 1) = %d, want %d", v, got, v+1)
				}
			})
		}
		time.Sleep(time.Millisecond)
	}
}

func runStressSoakSetting(procs int, d time.Duration) string {
	old := runtime.GOMAXPROCS(procs)
	defer runtime.GOMAXPROCS(old)
	s := newStressState()
	var wg sync.WaitGroup
	var sink atomic.Uint64
	var ids atomic.Uint64
	for i := range 8 {
		wg.Go(func() {
			stressMixedWorker(s, uint64(procs)*1<<32+uint64(i)*0x9e3779b9+1)
		})
	}
	for i := range 2 {
		wg.Go(func() {
			stressLockedWorker(s, uint64(procs)*1<<40+uint64(i)*0x85ebca6b+1)
		})
	}
	for range 2 {
		wg.Go(func() {
			stressGCChurn(s)
		})
	}
	for range 2 {
		wg.Go(func() {
			stressPreempt(s, &sink)
		})
	}
	wg.Go(func() {
		stressChurner(s, &ids)
	})
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		s.finish()
	case <-s.stop:
	}
	wg.Wait()
	select {
	case msg := <-s.errs:
		return msg
	default:
		return ""
	}
}

func stressUniqueProcs() []int {
	unique := make([]int, 0, 4)
	for _, p := range []int{1, 2, 4, runtime.NumCPU()} {
		dup := false
		for _, q := range unique {
			if p == q {
				dup = true
				break
			}
		}
		if p >= 1 && !dup {
			unique = append(unique, p)
		}
	}
	return unique
}

func TestFastcallStressSoak(t *testing.T) {
	requireCgo(t)
	t.Logf("logosfastcgo = %v", goexperiment.LogosFastCgo)
	settings := stressUniqueProcs()
	per := stressPerSetting(len(settings))
	total := per * time.Duration(len(settings))
	disarm := armStressWatchdog(total + 120*time.Second)
	defer disarm()
	profPath := t.TempDir() + "/stress-cpu.prof"
	f, err := os.Create(profPath)
	if err != nil {
		t.Fatal(err)
	}
	runtime.SetCPUProfileRate(2000)
	if err := pprof.StartCPUProfile(f); err != nil {
		f.Close()
		t.Fatal(err)
	}
	defer func() {
		pprof.StopCPUProfile()
		f.Close()
		if fi, serr := os.Stat(profPath); serr != nil || fi.Size() == 0 {
			t.Errorf("stress cpu profile missing or empty")
		}
	}()
	t.Logf("cpu profile at elevated rate: per-setting %s over %d settings", per, len(settings))
	for _, procs := range settings {
		t.Run(fmt.Sprintf("GOMAXPROCS=%d", procs), func(t *testing.T) {
			if msg := runStressSoakSetting(procs, per); msg != "" {
				t.Fatal(msg)
			}
		})
	}
}
func TestFastcallStressBlocking(t *testing.T) {
	requireCgo(t)
	t.Logf("logosfastcgo = %v", goexperiment.LogosFastCgo)
	disarm := armStressWatchdog(90 * time.Second)
	defer disarm()
	old := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(old)
	var counter atomic.Uint64
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Go(func() {
		for {
			select {
			case <-stop:
				return
			default:
				counter.Add(1)
				runtime.Gosched()
			}
		}
	})
	time.Sleep(20 * time.Millisecond)
	passed := false
	var before, during, after uint64
	for attempt := range 3 {
		before = counter.Load()
		if got := callStressUsleep(50000); got != 50000 {
			close(stop)
			wg.Wait()
			t.Fatalf("stress_usleep(50000) = %d, want 50000", got)
		}
		during = counter.Load()
		if got := callFastSleep(30000); got != 30000 {
			close(stop)
			wg.Wait()
			t.Fatalf("block_sleep(30000) = %d, want 30000", got)
		}
		after = counter.Load()
		if during != before && after != during {
			passed = true
			t.Logf("attempt %d progress: before=%d during=%d after=%d", attempt, before, during, after)
			break
		}
	}
	close(stop)
	wg.Wait()
	if !passed {
		t.Fatalf("counter stalled across 3 attempts with GOMAXPROCS=1: before=%d during=%d after=%d", before, during, after)
	}
}
