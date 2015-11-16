package gc

import (
	"os"
	"runtime"
	"runtime/pprof"
)

func (n *Node) Line() string {
	return Ctxt.LineHist.LineString(int(n.Lineno))
}

var atExitFuncs []func()

func AtExit(f func()) {
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
	cpuprofile     string
	memprofile     string
	memprofilerate int64
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
		AtExit(pprof.StopCPUProfile)
	}
	if memprofile != "" {
		if memprofilerate != 0 {
			runtime.MemProfileRate = int(memprofilerate)
		}
		f, err := os.Create(memprofile)
		if err != nil {
			Fatalf("%v", err)
		}
		AtExit(func() {
			runtime.GC() // profile all outstanding allocations
			if err := pprof.WriteHeapProfile(f); err != nil {
				Fatalf("%v", err)
			}
		})
	}
}
