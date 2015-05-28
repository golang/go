package gc

import (
	"os"
	"runtime"
	"runtime/pprof"
	"strconv"
	"strings"
)

func (n *Node) Line() string {
	return Ctxt.LineHist.LineString(int(n.Lineno))
}

func atoi(s string) int {
	// NOTE: Not strconv.Atoi, accepts hex and octal prefixes.
	n, _ := strconv.ParseInt(s, 0, 0)
	return int(n)
}

func isalnum(c int) bool {
	return isalpha(c) || isdigit(c)
}

func isalpha(c int) bool {
	return 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z'
}

func isdigit(c int) bool {
	return '0' <= c && c <= '9'
}

func plan9quote(s string) string {
	if s == "" {
		return "'" + strings.Replace(s, "'", "''", -1) + "'"
	}
	for i := 0; i < len(s); i++ {
		if s[i] <= ' ' || s[i] == '\'' {
			return "'" + strings.Replace(s, "'", "''", -1) + "'"
		}
	}
	return s
}

// strings.Compare, introduced in Go 1.5.
func stringsCompare(a, b string) int {
	if a == b {
		return 0
	}
	if a < b {
		return -1
	}
	return +1
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
			Fatal("%v", err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			Fatal("%v", err)
		}
		AtExit(pprof.StopCPUProfile)
	}
	if memprofile != "" {
		if memprofilerate != 0 {
			runtime.MemProfileRate = int(memprofilerate)
		}
		f, err := os.Create(memprofile)
		if err != nil {
			Fatal("%v", err)
		}
		AtExit(func() {
			runtime.GC() // profile all outstanding allocations
			if err := pprof.WriteHeapProfile(f); err != nil {
				Fatal("%v", err)
			}
		})
	}
}
