// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package generator

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"strings"
)

const (
	RandCtlNochecks = 0
	RandCtlChecks   = 1 << iota
	RandCtlCapture
	RandCtlPanic
)

func NewWrapRand(seed int64, ctl int) *wraprand {
	rand.Seed(seed)
	return &wraprand{seed: seed, ctl: ctl}
}

type wraprand struct {
	f32calls  int
	f64calls  int
	intncalls int
	seed      int64
	tag       string
	calls     []string
	ctl       int
}

func (w *wraprand) captureCall(tag string, val string) {
	call := tag + ": " + val + "\n"
	pc := make([]uintptr, 10)
	n := runtime.Callers(1, pc)
	if n == 0 {
		panic("why?")
	}
	pc = pc[:n] // pass only valid pcs to runtime.CallersFrames
	frames := runtime.CallersFrames(pc)
	for {
		frame, more := frames.Next()
		if strings.Contains(frame.File, "testing.") {
			break
		}
		call += fmt.Sprintf("%s %s:%d\n", frame.Function, frame.File, frame.Line)
		if !more {
			break
		}

	}
	w.calls = append(w.calls, call)
}

func (w *wraprand) Intn(n int64) int64 {
	w.intncalls++
	rv := rand.Int63n(n)
	if w.ctl&RandCtlCapture != 0 {
		w.captureCall("Intn", fmt.Sprintf("%d", rv))
	}
	return rv
}

func (w *wraprand) Float32() float32 {
	w.f32calls++
	rv := rand.Float32()
	if w.ctl&RandCtlCapture != 0 {
		w.captureCall("Float32", fmt.Sprintf("%f", rv))
	}
	return rv
}

func (w *wraprand) NormFloat64() float64 {
	w.f64calls++
	rv := rand.NormFloat64()
	if w.ctl&RandCtlCapture != 0 {
		w.captureCall("NormFloat64", fmt.Sprintf("%f", rv))
	}
	return rv
}

func (w *wraprand) emitCalls(fn string) {
	outf, err := os.OpenFile(fn, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	if err != nil {
		panic(err)
	}
	for _, c := range w.calls {
		fmt.Fprint(outf, c)
	}
	outf.Close()
}

func (w *wraprand) Equal(w2 *wraprand) bool {
	return w.f32calls == w2.f32calls &&
		w.f64calls == w2.f64calls &&
		w.intncalls == w2.intncalls
}

func (w *wraprand) Check(w2 *wraprand) {
	if w.ctl != 0 && !w.Equal(w2) {
		fmt.Fprintf(os.Stderr, "wraprand consistency check failed:\n")
		t := "w"
		if w.tag != "" {
			t = w.tag
		}
		t2 := "w2"
		if w2.tag != "" {
			t2 = w2.tag
		}
		fmt.Fprintf(os.Stderr, " %s: {f32:%d f64:%d i:%d}\n", t,
			w.f32calls, w.f64calls, w.intncalls)
		fmt.Fprintf(os.Stderr, " %s: {f32:%d f64:%d i:%d}\n", t2,
			w2.f32calls, w2.f64calls, w2.intncalls)
		if w.ctl&RandCtlCapture != 0 {
			f := fmt.Sprintf("/tmp/%s.txt", t)
			f2 := fmt.Sprintf("/tmp/%s.txt", t2)
			w.emitCalls(f)
			w2.emitCalls(f2)
			fmt.Fprintf(os.Stderr, "=-= emitted calls to %s, %s\n", f, f2)
		}
		if w.ctl&RandCtlPanic != 0 {
			panic("bad")
		}
	}
}

func (w *wraprand) Checkpoint(tag string) {
	if w.ctl&RandCtlCapture != 0 {
		w.calls = append(w.calls, "=-=\n"+tag+"\n=-=\n")
	}
}
