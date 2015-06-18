// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Serving of pprof-like profiles.

package main

import (
	"bufio"
	"fmt"
	"internal/trace"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
)

func init() {
	http.HandleFunc("/io", httpIO)
	http.HandleFunc("/block", httpBlock)
	http.HandleFunc("/syscall", httpSyscall)
	http.HandleFunc("/sched", httpSched)
}

// Record represents one entry in pprof-like profiles.
type Record struct {
	stk  []*trace.Frame
	n    uint64
	time int64
}

// httpIO serves IO pprof-like profile (time spent in IO wait).
func httpIO(w http.ResponseWriter, r *http.Request) {
	events, err := parseEvents()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	prof := make(map[uint64]Record)
	for _, ev := range events {
		if ev.Type != trace.EvGoBlockNet || ev.Link == nil || ev.StkID == 0 || len(ev.Stk) == 0 {
			continue
		}
		rec := prof[ev.StkID]
		rec.stk = ev.Stk
		rec.n++
		rec.time += ev.Link.Ts - ev.Ts
		prof[ev.StkID] = rec
	}
	serveSVGProfile(w, r, prof)
}

// httpBlock serves blocking pprof-like profile (time spent blocked on synchronization primitives).
func httpBlock(w http.ResponseWriter, r *http.Request) {
	events, err := parseEvents()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	prof := make(map[uint64]Record)
	for _, ev := range events {
		switch ev.Type {
		case trace.EvGoBlockSend, trace.EvGoBlockRecv, trace.EvGoBlockSelect,
			trace.EvGoBlockSync, trace.EvGoBlockCond:
		default:
			continue
		}
		if ev.Link == nil || ev.StkID == 0 || len(ev.Stk) == 0 {
			continue
		}
		rec := prof[ev.StkID]
		rec.stk = ev.Stk
		rec.n++
		rec.time += ev.Link.Ts - ev.Ts
		prof[ev.StkID] = rec
	}
	serveSVGProfile(w, r, prof)
}

// httpSyscall serves syscall pprof-like profile (time spent blocked in syscalls).
func httpSyscall(w http.ResponseWriter, r *http.Request) {
	events, err := parseEvents()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	prof := make(map[uint64]Record)
	for _, ev := range events {
		if ev.Type != trace.EvGoSysCall || ev.Link == nil || ev.StkID == 0 || len(ev.Stk) == 0 {
			continue
		}
		rec := prof[ev.StkID]
		rec.stk = ev.Stk
		rec.n++
		rec.time += ev.Link.Ts - ev.Ts
		prof[ev.StkID] = rec
	}
	serveSVGProfile(w, r, prof)
}

// httpSched serves scheduler latency pprof-like profile
// (time between a goroutine become runnable and actually scheduled for execution).
func httpSched(w http.ResponseWriter, r *http.Request) {
	events, err := parseEvents()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	prof := make(map[uint64]Record)
	for _, ev := range events {
		if (ev.Type != trace.EvGoUnblock && ev.Type != trace.EvGoCreate) ||
			ev.Link == nil || ev.StkID == 0 || len(ev.Stk) == 0 {
			continue
		}
		rec := prof[ev.StkID]
		rec.stk = ev.Stk
		rec.n++
		rec.time += ev.Link.Ts - ev.Ts
		prof[ev.StkID] = rec
	}
	serveSVGProfile(w, r, prof)
}

// generateSVGProfile generates pprof-like profile stored in prof and writes in to w.
func serveSVGProfile(w http.ResponseWriter, r *http.Request, prof map[uint64]Record) {
	if len(prof) == 0 {
		http.Error(w, "The profile is empty", http.StatusNotFound)
		return
	}
	blockf, err := ioutil.TempFile("", "block")
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to create temp file: %v", err), http.StatusInternalServerError)
		return
	}
	defer os.Remove(blockf.Name())
	blockb := bufio.NewWriter(blockf)
	fmt.Fprintf(blockb, "--- contention:\ncycles/second=1000000000\n")
	for _, rec := range prof {
		fmt.Fprintf(blockb, "%v %v @", rec.time, rec.n)
		for _, f := range rec.stk {
			fmt.Fprintf(blockb, " 0x%x", f.PC)
		}
		fmt.Fprintf(blockb, "\n")
	}
	err = blockb.Flush()
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to flush temp file: %v", err), http.StatusInternalServerError)
		return
	}
	err = blockf.Close()
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to close temp file: %v", err), http.StatusInternalServerError)
		return
	}

	svgFilename := blockf.Name() + ".svg"
	_, err = exec.Command("go", "tool", "pprof", "-svg", "-output", svgFilename, programBinary, blockf.Name()).CombinedOutput()
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to execute go tool pprof: %v", err), http.StatusInternalServerError)
		return
	}
	defer os.Remove(svgFilename)
	w.Header().Set("Content-Type", "image/svg+xml")
	http.ServeFile(w, r, svgFilename)
}
