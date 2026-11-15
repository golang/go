// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"strings"
)

var (
	failed   bool
	panicNow bool
)

func emit(name string, ok bool, detail string) {
	if ok {
		fmt.Printf("%s ok %s\n", name, detail)
		return
	}
	fmt.Printf("LAW %s %s\n", name, detail)
	failed = true
}

func frameHere() string {
	if panicNow {
		panic("boom")
	}
	var pcs [4]uintptr
	n := runtime.Callers(2, pcs[:])
	fr, _ := runtime.CallersFrames(pcs[:n]).Next()
	return fr.Function
}

type T struct{}

//go:noinline
func f_logosct1() string { return frameHere() }

//go:noinline
func f_logosct007() string { return frameHere() }

//go:noinline
func g_logosct1_logosct2() string { return frameHere() }

//go:noinline
func _logosct9() string { return frameHere() }

//go:noinline
func (T) m_logosct3() string { return frameHere() }

//go:noinline
func h_logosct4() func() string {
	return func() string { return frameHere() }
}

//go:noinline
func k_logosct5[V any](v V) string { return frameHere() }

type probe struct {
	id   string
	want string
	fn   any
	call func() string
}

func probes() []probe {
	return []probe{
		{"f1", "main.f_logosct1", f_logosct1, f_logosct1},
		{"f007", "main.f_logosct007", f_logosct007, f_logosct007},
		{"g12", "main.g_logosct1_logosct2", g_logosct1_logosct2, g_logosct1_logosct2},
		{"u9", "main._logosct9", _logosct9, _logosct9},
		{"m3", "main.T.m_logosct3", T.m_logosct3, T{}.m_logosct3},
		{"h4", "main.h_logosct4", h_logosct4(), h_logosct4()},
		{"k5", "main.k_logosct5", k_logosct5[int], func() string { return k_logosct5(7) }},
	}
}

func tracebackFrame(id string) string {
	exe, err := os.Executable()
	if err != nil {
		return "none"
	}
	cmd := exec.Command(exe, "panicchild", id)
	cmd.Env = os.Environ()
	out, err := cmd.CombinedOutput()
	if _, exited := errors.AsType[*exec.ExitError](err); !exited {
		return "none"
	}
	for ln := range strings.SplitSeq(string(out), "\n") {
		if strings.HasPrefix(ln, "main.") && !strings.HasPrefix(ln, "main.frameHere(") {
			name, _, _ := strings.Cut(ln, "(")
			return name
		}
	}
	return "none"
}

func child(id string) {
	panicNow = true
	for _, p := range probes() {
		if p.id == id {
			p.call()
		}
	}
}

func main() {
	if len(os.Args) > 2 && os.Args[1] == "panicchild" {
		child(os.Args[2])
		return
	}
	for _, p := range probes() {
		name := runtime.FuncForPC(reflect.ValueOf(p.fn).Pointer()).Name()
		emit(p.id+"-funcforpc", strings.HasPrefix(name, p.want), name)
		frame := p.call()
		emit(p.id+"-frames", strings.HasPrefix(frame, p.want), frame)
		tb := tracebackFrame(p.id)
		emit(p.id+"-traceback", strings.HasPrefix(tb, p.want), tb)
	}
	if failed {
		os.Exit(1)
	}
	fmt.Println("ok")
}
