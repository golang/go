// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"io"
	"os"
	"strings"
	"sync"
	"time"
)

var (
	debugTrace       string
	debugTraceAppend bool // append trace events instead of overwriting
	traceMu          sync.Mutex
	debugTraces      []string
	distEvents       []*traceEvent
)

// This is a subset of the fields on internal/trace/traceviewer/format.go,
// which we can't use here because it's an internal package and cmd/dist
// needs to be buildable using another Go toolchain. That struct is in turn
// used to represent the Chrome Trace Viewer JSON format whose canonical
// documentation is this Google doc:
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
type traceEvent struct {
	Name  string  `json:"name,omitempty"`
	Phase string  `json:"ph"`
	Time  float64 `json:"ts"`
	PID   uint64  `json:"pid"`
	TID   uint64  `json:"tid"`
}

func maybeTraceFlag(name string) []string {
	if debugTrace == "" {
		return nil
	}
	f := pathf("%s/%s.trace", workdir, name)
	debugTraces = append(debugTraces, f)
	return []string{"-debug-trace=" + f}
}

func timeMicros(t time.Time) float64 {
	return float64(t.UnixNano()) / float64(time.Microsecond)
}

type span struct {
	name string
	tid  uint64
}

func startSpan(name string) span {
	return startSpanTID(name, 0)
}

func startSpanTID(name string, tid uint64) span {
	if debugTrace == "" {
		return span{}
	}
	traceMu.Lock()
	distEvents = append(distEvents, &traceEvent{Name: name, Phase: "B", Time: timeMicros(time.Now()), TID: tid})
	traceMu.Unlock()
	return span{name: name, tid: tid}
}

func (s span) done() {
	if s.name == "" {
		return
	}
	traceMu.Lock()
	distEvents = append(distEvents, &traceEvent{Name: s.name, Phase: "E", Time: timeMicros(time.Now()), TID: s.tid})
	traceMu.Unlock()
}

// writeTrace merges the -debug-trace outputs from the go command invocations and adds
// dist's own trace events in too. Because the traces have absolute timpstamps, we can
// essentially just concatenate them.
func writeTrace() {
	var f *os.File
	var err error
	if debugTraceAppend {
		f, err = os.OpenFile(debugTrace, os.O_RDWR|os.O_CREATE, 0o777)
		if err != nil {
			fatalf("open trace file for read/write: %v", err)
		}
		ok := func() bool {
			// Try to seek before close brace so we can append.
			fi, err := f.Stat()
			if err != nil {
				return false
			}
			size := fi.Size()
			var lastTwo [2]byte
			if size < 2 {
				return false
			}
			if _, err := f.ReadAt(lastTwo[:], size-2); err != nil {
				return false
			}
			if lastTwo != [2]byte{']', '\n'} {
				return false
			}
			if _, err = f.Seek(-2, io.SeekEnd); err != nil {
				return false
			}
			if _, err = f.WriteString(",\n"); err != nil {
				return false
			}
			return true
		}()
		if !ok {
			if err = f.Truncate(0); err != nil {
				fatalf("failed to truncate trace file for writing: %v", err)
			}
			f.Seek(0, io.SeekStart)
			f.WriteString("[\n")
		}
	} else {
		// Not appending to already existing file. Create a new file.
		f, err = os.Create(debugTrace)
		if err != nil {
			fatalf("opening trace file for write: %v", err)
		}
		f.WriteString("[\n")
	}

	enc := json.NewEncoder(f)
	for _, ev := range distEvents {
		if err := enc.Encode(ev); err != nil {
			fatalf("marshaling trace event: %v", err)
		}
		f.WriteString(",")
	}

	for i, name := range debugTraces {
		data, err := os.ReadFile(name)
		if err != nil {
			fatalf("reading trace file: %v", err)
		}
		// The trace is a json list: trim the start and end brackets.
		events := strings.TrimSpace(string(data))
		events = strings.TrimSuffix(strings.TrimPrefix(events, "["), "]")
		f.WriteString(events)
		if i != len(debugTraces)-1 {
			f.WriteString(",")
		}
	}

	f.WriteString("]\n")
	if err := f.Close(); err != nil {
		fatalf("closing trace file: %v", err)
	}
}
