// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

var failed bool

func emit(name string, ok bool, detail string) {
	if ok {
		fmt.Printf("%s ok %s\n", name, detail)
		return
	}
	fmt.Printf("LAW %s %s\n", name, detail)
	failed = true
}

func doRecover() any {
	return recover()
}

func setRec(p *any) {
	*p = recover()
}

func direct() (v any) {
	defer setRec(&v)
	panic("direct")
}

func closureRec() (v any) {
	defer func() {
		v = recover()
	}()
	panic("closure")
}

func indirect() (inner, outer any) {
	inner = "unset"
	defer func() {
		outer = recover()
	}()
	defer func() {
		inner = doRecover()
	}()
	panic("boom")
}

func setDirectFat(p *any) {
	*p = recover()
}

func deferNamedFat() (v any) {
	defer setDirectFat(&v)
	panic("named")
}

func repan() {
	defer func() {
		if recover() == "first" {
			panic("second")
		}
	}()
	panic("first")
}

func chain() (v any) {
	defer func() {
		v = recover()
	}()
	repan()
	return nil
}

func deepPanic() int {
	panic("deep")
}

func catchDeep() (v any) {
	defer func() {
		v = recover()
	}()
	return deepPanic()
}

func intPanic() (v any) {
	defer func() {
		v = recover()
	}()
	panic(42)
}

func goexitOrder() []string {
	ch := make(chan string, 3)
	done := make(chan bool)
	go func() {
		defer func() { ch <- "d1" }()
		defer func() { ch <- "d2" }()
		defer func() { done <- true }()
		runtime.Goexit()
	}()
	<-done
	a := <-ch
	b := <-ch
	return []string{a, b}
}

var order []string

func l1() { defer func() { order = append(order, "l1") }(); l2() }
func l2() { defer func() { order = append(order, "l2") }(); l3() }
func l3() { defer func() { order = append(order, "l3") }(); l4() }
func l4() { defer func() { order = append(order, "l4") }(); l5() }
func l5() { defer func() { order = append(order, "l5") }() }

func deferArgs() string {
	s := ""
	defer func(v string) {
		s = v
	}("eager")
	s = "late"
	return s
}

func namedMod() (v int) {
	defer func() {
		v = 42
	}()
	return 1
}

func panicNil() (v any) {
	defer func() {
		v = recover()
	}()
	panic(nil)
}

func loopDefers() (out []int) {
	for i := range 5 {
		defer func(v int) {
			out = append(out, v)
		}(i)
	}
	return
}

func triple() (s string) {
	defer func() { s += "a" }()
	defer func() { s += "b" }()
	defer func() { s += "c" }()
	return
}

func f1() []string { return f2() }
func f2() []string { return f3() }
func f3() []string { return f4() }
func f4() []string { return f5() }
func f5() []string { return f6() }
func f6() []string { return f7() }
func f7() []string { return capture() }

func capture() []string {
	var pcs [16]uintptr
	n := runtime.Callers(1, pcs[:])
	frames := runtime.CallersFrames(pcs[:n])
	var out []string
	for len(out) < 8 {
		fr, more := frames.Next()
		if strings.HasPrefix(fr.Function, "main.") {
			out = append(out, fmt.Sprintf("%s:%d", fr.Function, fr.Line))
		}
		if !more {
			break
		}
	}
	return out
}

func child() {
	panic("boom")
}

func parentHeader() string {
	exe, err := os.Executable()
	if err != nil {
		return "noexe:" + err.Error()
	}
	cmd := exec.Command(exe, "panicchild")
	cmd.Env = os.Environ()
	out, _ := cmd.CombinedOutput()
	for _, ln := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(ln, "panic: ") || strings.HasPrefix(ln, "fatal error: ") {
			return ln
		}
	}
	return "none"
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "panicchild" {
		child()
		return
	}
	v := direct()
	emit("r01", v == "direct", fmt.Sprintf("v=%v", v))
	v = closureRec()
	emit("r02", v == "closure", fmt.Sprintf("v=%v", v))
	inner, outer := indirect()
	emit("r03", inner == nil, fmt.Sprintf("inner=%v", inner))
	emit("r04", outer == "boom", fmt.Sprintf("outer=%v", outer))
	v = deferNamedFat()
	emit("r17", v == "named", fmt.Sprintf("v=%v", v))
	v = chain()
	emit("r05", v == "second", fmt.Sprintf("v=%v", v))
	g := goexitOrder()
	emit("r06", len(g) == 2 && g[0] == "d2" && g[1] == "d1", fmt.Sprintf("g=%v", g))
	order = nil
	l1()
	emit("r07", len(order) == 5 && order[0] == "l5" && order[4] == "l1", fmt.Sprintf("order=%v", order))
	v = catchDeep()
	emit("r08", v == "deep", fmt.Sprintf("v=%v", v))
	v = intPanic()
	emit("r09", v == 42, fmt.Sprintf("v=%v", v))
	s := deferArgs()
	emit("r10", s == "late", fmt.Sprintf("s=%v", s))
	m := namedMod()
	emit("r11", m == 42, fmt.Sprintf("m=%v", m))
	v = panicNil()
	emit("r12", v != nil, fmt.Sprintf("v=%v", v))
	ld := loopDefers()
	emit("r13", len(ld) == 5 && ld[0] == 4 && ld[4] == 0, fmt.Sprintf("ld=%v", ld))
	t := triple()
	emit("r14", t == "cba", fmt.Sprintf("t=%v", t))
	fr := f1()
	emit("r15", len(fr) >= 7, fmt.Sprintf("frames=%v", strings.Join(fr, ",")))
	h := parentHeader()
	emit("r16", h == "panic: boom", fmt.Sprintf("h=%v", h))
	if failed {
		os.Exit(1)
	}
	fmt.Println("ok")
}
