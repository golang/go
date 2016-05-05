// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Serving of pprof-like profiles.

package main

import (
	"bufio"
	"cmd/internal/pprof/profile"
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
	defer func() {
		blockf.Close()
		os.Remove(blockf.Name())
	}()
	blockb := bufio.NewWriter(blockf)
	if err := buildProfile(prof).Write(blockb); err != nil {
		http.Error(w, fmt.Sprintf("failed to write profile: %v", err), http.StatusInternalServerError)
		return
	}
	if err := blockb.Flush(); err != nil {
		http.Error(w, fmt.Sprintf("failed to flush temp file: %v", err), http.StatusInternalServerError)
		return
	}
	if err := blockf.Close(); err != nil {
		http.Error(w, fmt.Sprintf("failed to close temp file: %v", err), http.StatusInternalServerError)
		return
	}
	svgFilename := blockf.Name() + ".svg"
	if output, err := exec.Command("go", "tool", "pprof", "-svg", "-output", svgFilename, blockf.Name()).CombinedOutput(); err != nil {
		http.Error(w, fmt.Sprintf("failed to execute go tool pprof: %v\n%s", err, output), http.StatusInternalServerError)
		return
	}
	defer os.Remove(svgFilename)
	w.Header().Set("Content-Type", "image/svg+xml")
	http.ServeFile(w, r, svgFilename)
}

func buildProfile(prof map[uint64]Record) *profile.Profile {
	p := &profile.Profile{
		PeriodType: &profile.ValueType{Type: "trace", Unit: "count"},
		Period:     1,
		SampleType: []*profile.ValueType{
			{Type: "contentions", Unit: "count"},
			{Type: "delay", Unit: "nanoseconds"},
		},
	}
	locs := make(map[uint64]*profile.Location)
	funcs := make(map[string]*profile.Function)
	for _, rec := range prof {
		var sloc []*profile.Location
		for _, frame := range rec.stk {
			loc := locs[frame.PC]
			if loc == nil {
				fn := funcs[frame.File+frame.Fn]
				if fn == nil {
					fn = &profile.Function{
						ID:         uint64(len(p.Function) + 1),
						Name:       frame.Fn,
						SystemName: frame.Fn,
						Filename:   frame.File,
					}
					p.Function = append(p.Function, fn)
					funcs[frame.File+frame.Fn] = fn
				}
				loc = &profile.Location{
					ID:      uint64(len(p.Location) + 1),
					Address: frame.PC,
					Line: []profile.Line{
						profile.Line{
							Function: fn,
							Line:     int64(frame.Line),
						},
					},
				}
				p.Location = append(p.Location, loc)
				locs[frame.PC] = loc
			}
			sloc = append(sloc, loc)
		}
		p.Sample = append(p.Sample, &profile.Sample{
			Value:    []int64{int64(rec.n), rec.time},
			Location: sloc,
		})
	}
	return p
}
