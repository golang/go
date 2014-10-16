// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof_test

import (
	"bufio"
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	. "runtime/pprof"
	"testing"
	"unsafe"
)

var memSink interface{}

func allocateTransient1M() {
	for i := 0; i < 1024; i++ {
		memSink = &struct{ x [1024]byte }{}
	}
}

func allocateTransient2M() {
	// prevent inlining
	if memSink == nil {
		panic("bad")
	}
	memSink = make([]byte, 2<<20)
}

type Obj32 struct {
	link *Obj32
	pad  [32 - unsafe.Sizeof(uintptr(0))]byte
}

var persistentMemSink *Obj32

func allocatePersistent1K() {
	for i := 0; i < 32; i++ {
		// Can't use slice because that will introduce implicit allocations.
		obj := &Obj32{link: persistentMemSink}
		persistentMemSink = obj
	}
}

var memoryProfilerRun = 0

func TestMemoryProfiler(t *testing.T) {
	// Create temp file for the profile.
	f, err := ioutil.TempFile("", "memprof")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer func() {
		f.Close()
		os.Remove(f.Name())
	}()

	// Disable sampling, otherwise it's difficult to assert anything.
	oldRate := runtime.MemProfileRate
	runtime.MemProfileRate = 1
	defer func() {
		runtime.MemProfileRate = oldRate
	}()
	// Allocate a meg to ensure that mcache.next_sample is updated to 1.
	for i := 0; i < 1024; i++ {
		memSink = make([]byte, 1024)
	}

	// Do the interesting allocations.
	allocateTransient1M()
	allocateTransient2M()
	allocatePersistent1K()
	memSink = nil

	runtime.GC() // materialize stats
	if err := WriteHeapProfile(f); err != nil {
		t.Fatalf("failed to write heap profile: %v", err)
	}
	f.Close()

	memoryProfilerRun++
	checkMemProfile(t, f.Name(), []string{"--alloc_space", "--show_bytes", "--lines"}, []string{
		fmt.Sprintf(`%v .* runtime/pprof_test\.allocateTransient1M .*mprof_test.go:25`, 1<<20*memoryProfilerRun),
		fmt.Sprintf(`%v .* runtime/pprof_test\.allocateTransient2M .*mprof_test.go:34`, 2<<20*memoryProfilerRun),
		fmt.Sprintf(`%v .* runtime/pprof_test\.allocatePersistent1K .*mprof_test.go:47`, 1<<10*memoryProfilerRun),
	}, []string{})

	checkMemProfile(t, f.Name(), []string{"--inuse_space", "--show_bytes", "--lines"}, []string{
		fmt.Sprintf(`%v .* runtime/pprof_test\.allocatePersistent1K .*mprof_test.go:47`, 1<<10*memoryProfilerRun),
	}, []string{
		"allocateTransient1M",
		"allocateTransient2M",
	})
}

func checkMemProfile(t *testing.T, file string, addArgs []string, what []string, whatnot []string) {
	args := []string{"tool", "pprof", "--text"}
	args = append(args, addArgs...)
	args = append(args, os.Args[0], file)
	out, err := exec.Command("go", args...).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to execute pprof: %v\n%v\n", err, string(out))
	}

	matched := make(map[*regexp.Regexp]bool)
	for _, s := range what {
		matched[regexp.MustCompile(s)] = false
	}
	var not []*regexp.Regexp
	for _, s := range whatnot {
		not = append(not, regexp.MustCompile(s))
	}

	s := bufio.NewScanner(bytes.NewReader(out))
	for s.Scan() {
		ln := s.Text()
		for re := range matched {
			if re.MatchString(ln) {
				if matched[re] {
					t.Errorf("entry '%s' is matched twice", re.String())
				}
				matched[re] = true
			}
		}
		for _, re := range not {
			if re.MatchString(ln) {
				t.Errorf("entry '%s' is matched, but must not", re.String())
			}
		}
	}
	for re, ok := range matched {
		if !ok {
			t.Errorf("entry '%s' is not matched", re.String())
		}
	}
	if t.Failed() {
		t.Logf("profile:\n%v", string(out))
	}
}
