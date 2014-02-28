package main

import (
	"fmt"
	"path"
	"runtime"
	"strings"
)

var stack string

func f() {
	pc := make([]uintptr, 6)
	pc = pc[:runtime.Callers(1, pc)]
	for _, f := range pc {
		Func := runtime.FuncForPC(f)
		name := Func.Name()
		if strings.Contains(name, "$") || strings.Contains(name, ".func") {
			name = "func" // anon funcs vary across toolchains
		}
		file, line := Func.FileLine(0)
		stack += fmt.Sprintf("%s at %s:%d\n", name, path.Base(file), line)
	}
}

func g() { f() }
func h() { g() }
func i() { func() { h() }() }

// Hack: the 'func' and the call to Caller are on the same line,
// to paper over differences between toolchains.
// (The interpreter's location info isn't yet complete.)
func runtimeCaller0() (uintptr, string, int, bool) { return runtime.Caller(0) }

func main() {
	i()
	if stack != `main.f at callstack.go:12
main.g at callstack.go:26
main.h at callstack.go:27
func at callstack.go:28
main.i at callstack.go:28
main.main at callstack.go:35
` {
		panic("unexpected stack: " + stack)
	}

	pc, file, line, _ := runtimeCaller0()
	got := fmt.Sprintf("%s @ %s:%d", runtime.FuncForPC(pc).Name(), path.Base(file), line)
	if got != "main.runtimeCaller0 @ callstack.go:33" {
		panic("runtime.Caller: " + got)
	}
}
