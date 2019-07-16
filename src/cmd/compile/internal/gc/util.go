// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"os"
	"runtime"
	"runtime/pprof"
)

// Line returns n's position as a string. If n has been inlined,
// it uses the outermost position where n has been inlined.
func (n *Node) Line() string {
	return linestr(n.Pos)
}

var atExitFuncs []func()

func atExit(f func()) {
	atExitFuncs = append(atExitFuncs, f)
}

func Exit(code int) {
	for i := len(atExitFuncs) - 1; i >= 0; i-- {
		f := atExitFuncs[i]
		atExitFuncs = atExitFuncs[:i]
		f()
	}
	os.Exit(code)
}

var (
	blockprofile   string
	cpuprofile     string
	memprofile     string
	memprofilerate int64
	traceprofile   string
	traceHandler   func(string)
	mutexprofile   string
)

func startProfile() {
	if cpuprofile != "" {
		f, err := os.Create(cpuprofile)
		if err != nil {
			Fatalf("%v", err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			Fatalf("%v", err)
		}
		atExit(pprof.StopCPUProfile)
	}
	if memprofile != "" {
		if memprofilerate != 0 {
			runtime.MemProfileRate = int(memprofilerate)
		}
		f, err := os.Create(memprofile)
		if err != nil {
			Fatalf("%v", err)
		}
		atExit(func() {
			// Profile all outstanding allocations.
			runtime.GC()
			// compilebench parses the memory profile to extract memstats,
			// which are only written in the legacy pprof format.
			// See golang.org/issue/18641 and runtime/pprof/pprof.go:writeHeap.
			const writeLegacyFormat = 1
			if err := pprof.Lookup("heap").WriteTo(f, writeLegacyFormat); err != nil {
				Fatalf("%v", err)
			}
		})
	} else {
		// Not doing memory profiling; disable it entirely.
		runtime.MemProfileRate = 0
	}
	if blockprofile != "" {
		f, err := os.Create(blockprofile)
		if err != nil {
			Fatalf("%v", err)
		}
		runtime.SetBlockProfileRate(1)
		atExit(func() {
			pprof.Lookup("block").WriteTo(f, 0)
			f.Close()
		})
	}
	if mutexprofile != "" {
		f, err := os.Create(mutexprofile)
		if err != nil {
			Fatalf("%v", err)
		}
		startMutexProfiling()
		atExit(func() {
			pprof.Lookup("mutex").WriteTo(f, 0)
			f.Close()
		})
	}
	if traceprofile != "" && traceHandler != nil {
		traceHandler(traceprofile)
	}
}
