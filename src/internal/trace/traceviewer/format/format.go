// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package traceviewer provides definitions of the JSON data structures
// used by the Chrome trace viewer.
//
// The official description of the format is in this file:
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
//
// Note: This can't be part of the parent traceviewer package as that would
// throw. go_bootstrap cannot depend on the cgo version of package net in ./make.bash.
package format

type Data struct {
	Events   []*Event         `json:"traceEvents"`
	Frames   map[string]Frame `json:"stackFrames"`
	TimeUnit string           `json:"displayTimeUnit"`
}

type Event struct {
	Name      string  `json:"name,omitempty"`
	Phase     string  `json:"ph"`
	Scope     string  `json:"s,omitempty"`
	Time      float64 `json:"ts"`
	Dur       float64 `json:"dur,omitempty"`
	PID       uint64  `json:"pid"`
	TID       uint64  `json:"tid"`
	ID        uint64  `json:"id,omitempty"`
	BindPoint string  `json:"bp,omitempty"`
	Stack     int     `json:"sf,omitempty"`
	EndStack  int     `json:"esf,omitempty"`
	Arg       any     `json:"args,omitempty"`
	Cname     string  `json:"cname,omitempty"`
	Category  string  `json:"cat,omitempty"`
}

type Frame struct {
	Name   string `json:"name"`
	Parent int    `json:"parent,omitempty"`
}

type NameArg struct {
	Name string `json:"name"`
}

type BlockedArg struct {
	Blocked string `json:"blocked"`
}

type SortIndexArg struct {
	Index int `json:"sort_index"`
}

type HeapCountersArg struct {
	Allocated uint64
	NextGC    uint64
}

const (
	ProcsSection = 0 // where Goroutines or per-P timelines are presented.
	StatsSection = 1 // where counters are presented.
	TasksSection = 2 // where Task hierarchy & timeline is presented.
)

type GoroutineCountersArg struct {
	Running   uint64
	Runnable  uint64
	GCWaiting uint64
}

type ThreadCountersArg struct {
	Running   int64
	InSyscall int64
}

type ThreadIDArg struct {
	ThreadID uint64
}
