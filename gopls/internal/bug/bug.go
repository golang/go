// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bug provides utilities for reporting internal bugs, and being
// notified when they occur.
//
// Philosophically, because gopls runs as a sidecar process that the user does
// not directly control, sometimes it keeps going on broken invariants rather
// than panicking. In those cases, bug reports provide a mechanism to alert
// developers and capture relevant metadata.
package bug

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"sort"
	"sync"
	"time"

	"golang.org/x/telemetry/counter"
)

// PanicOnBugs controls whether to panic when bugs are reported.
//
// It may be set to true during testing.
var PanicOnBugs = false

var (
	mu        sync.Mutex
	exemplars map[string]Bug
	handlers  []func(Bug)
)

// A Bug represents an unexpected event or broken invariant. They are used for
// capturing metadata that helps us understand the event.
//
// Bugs are JSON-serializable.
type Bug struct {
	File        string    // file containing the call to bug.Report
	Line        int       // line containing the call to bug.Report
	Description string    // description of the bug
	Key         string    // key identifying the bug (file:line if available)
	Stack       string    // call stack
	AtTime      time.Time // time the bug was reported
}

// Reportf reports a formatted bug message.
func Reportf(format string, args ...interface{}) {
	report(fmt.Sprintf(format, args...))
}

// Errorf calls fmt.Errorf for the given arguments, and reports the resulting
// error message as a bug.
func Errorf(format string, args ...interface{}) error {
	err := fmt.Errorf(format, args...)
	report(err.Error())
	return err
}

// Report records a new bug encountered on the server.
// It uses reflection to report the position of the immediate caller.
func Report(description string) {
	report(description)
}

// BugReportCount is a telemetry counter that tracks # of bug reports.
var BugReportCount = counter.NewStack("gopls/bug", 16)

func report(description string) {
	_, file, line, ok := runtime.Caller(2) // all exported reporting functions call report directly

	key := "<missing callsite>"
	if ok {
		key = fmt.Sprintf("%s:%d", file, line)
	}

	if PanicOnBugs {
		panic(fmt.Sprintf("%s: %s", key, description))
	}

	bug := Bug{
		File:        file,
		Line:        line,
		Description: description,
		Key:         key,
		Stack:       string(debug.Stack()),
		AtTime:      time.Now(),
	}

	newBug := false
	mu.Lock()
	if _, ok := exemplars[key]; !ok {
		if exemplars == nil {
			exemplars = make(map[string]Bug)
		}
		exemplars[key] = bug // capture one exemplar per key
		newBug = true
	}
	hh := handlers
	handlers = nil
	mu.Unlock()

	if newBug {
		BugReportCount.Inc()
	}
	// Call the handlers outside the critical section since a
	// handler may itself fail and call bug.Report. Since handlers
	// are one-shot, the inner call should be trivial.
	for _, handle := range hh {
		handle(bug)
	}
}

// Handle adds a handler function that will be called with the next
// bug to occur on the server. The handler only ever receives one bug.
// It is called synchronously, and should return in a timely manner.
func Handle(h func(Bug)) {
	mu.Lock()
	defer mu.Unlock()
	handlers = append(handlers, h)
}

// List returns a slice of bug exemplars -- the first bugs to occur at each
// callsite.
func List() []Bug {
	mu.Lock()
	defer mu.Unlock()

	var bugs []Bug

	for _, bug := range exemplars {
		bugs = append(bugs, bug)
	}

	sort.Slice(bugs, func(i, j int) bool {
		return bugs[i].Key < bugs[j].Key
	})

	return bugs
}
