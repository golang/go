// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Trace is a tool for viewing trace files.

Trace files can be generated with:
  - runtime/trace.Start
  - net/http/pprof package
  - go test -trace

Example usage:
Generate a trace file with 'go test':

	go test -trace trace.out pkg

View the trace in a web browser:

	go tool trace trace.out

Generate a pprof-like profile from the trace:

	go tool trace -pprof=TYPE trace.out > TYPE.pprof

Supported profile types are:
  - net: network blocking profile
  - sync: synchronization blocking profile
  - syscall: syscall blocking profile
  - sched: scheduler latency profile

Then, you can use the pprof tool to analyze the profile:

	go tool pprof TYPE.pprof

Note that while the various profiles available when launching
'go tool trace' work on every browser, the trace viewer itself
(the 'view trace' page) comes from the Chrome/Chromium project
and is only actively tested on that browser.
*/
package main
