// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof_test

import (
	"bytes"
	"hash/crc32"
	"os/exec"
	"regexp"
	"runtime"
	. "runtime/pprof"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"
)

func TestCPUProfile(t *testing.T) {
	buf := make([]byte, 100000)
	testCPUProfile(t, []string{"crc32.ChecksumIEEE"}, func() {
		// This loop takes about a quarter second on a 2 GHz laptop.
		// We only need to get one 100 Hz clock tick, so we've got
		// a 25x safety buffer.
		for i := 0; i < 1000; i++ {
			crc32.ChecksumIEEE(buf)
		}
	})
}

func TestCPUProfileMultithreaded(t *testing.T) {
	buf := make([]byte, 100000)
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))
	testCPUProfile(t, []string{"crc32.ChecksumIEEE", "crc32.Update"}, func() {
		c := make(chan int)
		go func() {
			for i := 0; i < 2000; i++ {
				crc32.Update(0, crc32.IEEETable, buf)
			}
			c <- 1
		}()
		// This loop takes about a quarter second on a 2 GHz laptop.
		// We only need to get one 100 Hz clock tick, so we've got
		// a 25x safety buffer.
		for i := 0; i < 2000; i++ {
			crc32.ChecksumIEEE(buf)
		}
		<-c
	})
}

func testCPUProfile(t *testing.T, need []string, f func()) {
	switch runtime.GOOS {
	case "darwin":
		out, err := exec.Command("uname", "-a").CombinedOutput()
		if err != nil {
			t.Fatal(err)
		}
		vers := string(out)
		t.Logf("uname -a: %v", vers)
	case "plan9":
		// unimplemented
		return
	}

	var prof bytes.Buffer
	if err := StartCPUProfile(&prof); err != nil {
		t.Fatal(err)
	}
	f()
	StopCPUProfile()

	// Convert []byte to []uintptr.
	bytes := prof.Bytes()
	l := len(bytes) / int(unsafe.Sizeof(uintptr(0)))
	val := *(*[]uintptr)(unsafe.Pointer(&bytes))
	val = val[:l]

	if l < 13 {
		t.Logf("profile too short: %#x", val)
		if badOS[runtime.GOOS] {
			t.Skipf("ignoring failure on %s; see golang.org/issue/6047", runtime.GOOS)
			return
		}
		t.FailNow()
	}

	hd, val, tl := val[:5], val[5:l-3], val[l-3:]
	if hd[0] != 0 || hd[1] != 3 || hd[2] != 0 || hd[3] != 1e6/100 || hd[4] != 0 {
		t.Fatalf("unexpected header %#x", hd)
	}

	if tl[0] != 0 || tl[1] != 1 || tl[2] != 0 {
		t.Fatalf("malformed end-of-data marker %#x", tl)
	}

	// Check that profile is well formed and contains ChecksumIEEE.
	have := make([]uintptr, len(need))
	for len(val) > 0 {
		if len(val) < 2 || val[0] < 1 || val[1] < 1 || uintptr(len(val)) < 2+val[1] {
			t.Fatalf("malformed profile.  leftover: %#x", val)
		}
		for _, pc := range val[2 : 2+val[1]] {
			f := runtime.FuncForPC(pc)
			if f == nil {
				continue
			}
			for i, name := range need {
				if strings.Contains(f.Name(), name) {
					have[i] += val[0]
				}
			}
		}
		val = val[2+val[1]:]
	}

	var total uintptr
	for i, name := range need {
		total += have[i]
		t.Logf("%s: %d\n", name, have[i])
	}
	ok := true
	if total == 0 {
		t.Logf("no CPU profile samples collected")
		ok = false
	}
	min := total / uintptr(len(have)) / 3
	for i, name := range need {
		if have[i] < min {
			t.Logf("%s has %d samples out of %d, want at least %d, ideally %d", name, have[i], total, min, total/uintptr(len(have)))
			ok = false
		}
	}

	if !ok {
		if badOS[runtime.GOOS] {
			t.Skipf("ignoring failure on %s; see golang.org/issue/6047", runtime.GOOS)
			return
		}
		t.FailNow()
	}
}

func TestCPUProfileWithFork(t *testing.T) {
	// Fork can hang if preempted with signals frequently enough (see issue 5517).
	// Ensure that we do not do this.
	heap := 1 << 30
	if testing.Short() {
		heap = 100 << 20
	}
	// This makes fork slower.
	garbage := make([]byte, heap)
	// Need to touch the slice, otherwise it won't be paged in.
	done := make(chan bool)
	go func() {
		for i := range garbage {
			garbage[i] = 42
		}
		done <- true
	}()
	<-done

	var prof bytes.Buffer
	if err := StartCPUProfile(&prof); err != nil {
		t.Fatal(err)
	}
	defer StopCPUProfile()

	for i := 0; i < 10; i++ {
		exec.Command("go").CombinedOutput()
	}
}

// Operating systems that are expected to fail the tests. See issue 6047.
var badOS = map[string]bool{
	"darwin": true,
	//"netbsd":  true,
	//"openbsd": true,
}

func TestBlockProfile(t *testing.T) {
	runtime.SetBlockProfileRate(1)
	defer runtime.SetBlockProfileRate(0)
	produceChanContention()
	produceMutexContention()
	var w bytes.Buffer
	Lookup("block").WriteTo(&w, 1)
	prof := w.String()

	if !strings.HasPrefix(prof, "--- contention:\ncycles/second=") {
		t.Fatalf("Bad profile header:\n%v", prof)
	}

	reChan := regexp.MustCompile(`
[0-9]+ [0-9]+ @ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+
#	0x[0-9,a-f]+	runtime/pprof_test\.produceChanContention\+0x[0-9,a-f]+	.*/src/pkg/runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9,a-f]+	runtime/pprof_test\.TestBlockProfile\+0x[0-9,a-f]+	.*/src/pkg/runtime/pprof/pprof_test.go:[0-9]+
`)
	if !reChan.MatchString(prof) {
		t.Fatalf("Bad chan entry, expect:\n%v\ngot:\n%v", reChan, prof)
	}

	reMutex := regexp.MustCompile(`
[0-9]+ [0-9]+ @ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+
#	0x[0-9,a-f]+	sync\.\(\*Mutex\)\.Lock\+0x[0-9,a-f]+	.*/src/pkg/sync/mutex\.go:[0-9]+
#	0x[0-9,a-f]+	runtime/pprof_test\.produceMutexContention\+0x[0-9,a-f]+	.*/src/pkg/runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9,a-f]+	runtime/pprof_test\.TestBlockProfile\+0x[0-9,a-f]+	.*/src/pkg/runtime/pprof/pprof_test.go:[0-9]+
`)
	if !reMutex.MatchString(prof) {
		t.Fatalf("Bad mutex entry, expect:\n%v\ngot:\n%v", reMutex, prof)
	}
}

func produceChanContention() {
	c := make(chan bool)
	go func() {
		time.Sleep(10 * time.Millisecond)
		c <- true
	}()
	<-c
}

func produceMutexContention() {
	var mu sync.Mutex
	mu.Lock()
	go func() {
		time.Sleep(10 * time.Millisecond)
		mu.Unlock()
	}()
	mu.Lock()
}
