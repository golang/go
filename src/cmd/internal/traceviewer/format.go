// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package traceviewer provides definitions of the JSON data structures
// used by the Chrome trace viewer.
//
// The official description of the format is in this file:
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
package traceviewer

type Data struct {
	Events   []*Event         `json:"traceEvents"`
	Frames   map[string]Frame `json:"stackFrames"`
	TimeUnit string           `json:"displayTimeUnit"`
}

type Event struct {
	Name      string      `json:"name,omitempty"`
	Phase     string      `json:"ph"`
	Scope     string      `json:"s,omitempty"`
	Time      float64     `json:"ts"`
	Dur       float64     `json:"dur,omitempty"`
	PID       uint64      `json:"pid"`
	TID       uint64      `json:"tid"`
	ID        uint64      `json:"id,omitempty"`
	BindPoint string      `json:"bp,omitempty"`
	Stack     int         `json:"sf,omitempty"`
	EndStack  int         `json:"esf,omitempty"`
	Arg       interface{} `json:"args,omitempty"`
	Cname     string      `json:"cname,omitempty"`
	Category  string      `json:"cat,omitempty"`
}

type Frame struct {
	Name   string `json:"name"`
	Parent int    `json:"parent,omitempty"`
}
