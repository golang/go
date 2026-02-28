// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package benchmark provides a Metrics object that enables memory and CPU
// profiling for the linker. The Metrics objects can be used to mark stages
// of the code, and name the measurements during that stage. There is also
// optional GCs that can be performed at the end of each stage, so you
// can get an accurate measurement of how each stage changes live memory.
package benchmark

import (
	"fmt"
	"io"
	"os"
	"runtime"
	"runtime/pprof"
	"time"
	"unicode"
)

type Flags int

const (
	GC         = 1 << iota
	NoGC Flags = 0
)

type Metrics struct {
	gc        Flags
	marks     []*mark
	curMark   *mark
	filebase  string
	pprofFile *os.File
}

type mark struct {
	name              string
	startM, endM, gcM runtime.MemStats
	startT, endT      time.Time
}

// New creates a new Metrics object.
//
// Typical usage should look like:
//
//	func main() {
//	  filename := "" // Set to enable per-phase pprof file output.
//	  bench := benchmark.New(benchmark.GC, filename)
//	  defer bench.Report(os.Stdout)
//	  // etc
//	  bench.Start("foo")
//	  foo()
//	  bench.Start("bar")
//	  bar()
//	}
//
// Note that a nil Metrics object won't cause any errors, so one could write
// code like:
//
//	func main() {
//	  enableBenchmarking := flag.Bool("enable", true, "enables benchmarking")
//	  flag.Parse()
//	  var bench *benchmark.Metrics
//	  if *enableBenchmarking {
//	    bench = benchmark.New(benchmark.GC)
//	  }
//	  bench.Start("foo")
//	  // etc.
//	}
func New(gc Flags, filebase string) *Metrics {
	if gc == GC {
		runtime.GC()
	}
	return &Metrics{gc: gc, filebase: filebase}
}

// Report reports the metrics.
// Closes the currently Start(ed) range, and writes the report to the given io.Writer.
func (m *Metrics) Report(w io.Writer) {
	if m == nil {
		return
	}

	m.closeMark()

	gcString := ""
	if m.gc == GC {
		gcString = "_GC"
	}

	var totTime time.Duration
	for _, curMark := range m.marks {
		dur := curMark.endT.Sub(curMark.startT)
		totTime += dur
		fmt.Fprintf(w, "%s 1 %d ns/op", makeBenchString(curMark.name+gcString), dur.Nanoseconds())
		fmt.Fprintf(w, "\t%d B/op", curMark.endM.TotalAlloc-curMark.startM.TotalAlloc)
		fmt.Fprintf(w, "\t%d allocs/op", curMark.endM.Mallocs-curMark.startM.Mallocs)
		if m.gc == GC {
			fmt.Fprintf(w, "\t%d live-B", curMark.gcM.HeapAlloc)
		} else {
			fmt.Fprintf(w, "\t%d heap-B", curMark.endM.HeapAlloc)
		}
		fmt.Fprintf(w, "\n")
	}
	fmt.Fprintf(w, "%s 1 %d ns/op\n", makeBenchString("total time"+gcString), totTime.Nanoseconds())
}

// Start marks the beginning of a new measurement phase.
// Once a metric is started, it continues until either a Report is issued, or another Start is called.
func (m *Metrics) Start(name string) {
	if m == nil {
		return
	}
	m.closeMark()
	m.curMark = &mark{name: name}
	// Unlikely we need to a GC here, as one was likely just done in closeMark.
	if m.shouldPProf() {
		f, err := os.Create(makePProfFilename(m.filebase, name, "cpuprof"))
		if err != nil {
			panic(err)
		}
		m.pprofFile = f
		if err = pprof.StartCPUProfile(m.pprofFile); err != nil {
			panic(err)
		}
	}
	runtime.ReadMemStats(&m.curMark.startM)
	m.curMark.startT = time.Now()
}

func (m *Metrics) closeMark() {
	if m == nil || m.curMark == nil {
		return
	}
	m.curMark.endT = time.Now()
	if m.shouldPProf() {
		pprof.StopCPUProfile()
		m.pprofFile.Close()
		m.pprofFile = nil
	}
	runtime.ReadMemStats(&m.curMark.endM)
	if m.gc == GC {
		runtime.GC()
		runtime.ReadMemStats(&m.curMark.gcM)
		if m.shouldPProf() {
			// Collect a profile of the live heap. Do a
			// second GC to force sweep completion so we
			// get a complete snapshot of the live heap at
			// the end of this phase.
			runtime.GC()
			f, err := os.Create(makePProfFilename(m.filebase, m.curMark.name, "memprof"))
			if err != nil {
				panic(err)
			}
			err = pprof.WriteHeapProfile(f)
			if err != nil {
				panic(err)
			}
			err = f.Close()
			if err != nil {
				panic(err)
			}
		}
	}
	m.marks = append(m.marks, m.curMark)
	m.curMark = nil
}

// shouldPProf returns true if we should be doing pprof runs.
func (m *Metrics) shouldPProf() bool {
	return m != nil && len(m.filebase) > 0
}

// makeBenchString makes a benchmark string consumable by Go's benchmarking tools.
func makeBenchString(name string) string {
	needCap := true
	ret := []rune("Benchmark")
	for _, r := range name {
		if unicode.IsSpace(r) {
			needCap = true
			continue
		}
		if needCap {
			r = unicode.ToUpper(r)
			needCap = false
		}
		ret = append(ret, r)
	}
	return string(ret)
}

func makePProfFilename(filebase, name, typ string) string {
	return fmt.Sprintf("%s_%s.%s", filebase, makeBenchString(name), typ)
}
