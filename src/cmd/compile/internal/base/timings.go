// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"fmt"
	"io"
	"strings"
	"time"
)

var Timer Timings

// Timings collects the execution times of labeled phases
// which are added trough a sequence of Start/Stop calls.
// Events may be associated with each phase via AddEvent.
type Timings struct {
	list   []timestamp
	events map[int][]*event // lazily allocated
}

type timestamp struct {
	time  time.Time
	label string
	start bool
}

type event struct {
	size int64  // count or amount of data processed (allocations, data size, lines, funcs, ...)
	unit string // unit of size measure (count, MB, lines, funcs, ...)
}

func (t *Timings) append(labels []string, start bool) {
	t.list = append(t.list, timestamp{time.Now(), strings.Join(labels, ":"), start})
}

// Start marks the beginning of a new phase and implicitly stops the previous phase.
// The phase name is the colon-separated concatenation of the labels.
func (t *Timings) Start(labels ...string) {
	t.append(labels, true)
}

// Stop marks the end of a phase and implicitly starts a new phase.
// The labels are added to the labels of the ended phase.
func (t *Timings) Stop(labels ...string) {
	t.append(labels, false)
}

// AddEvent associates an event, i.e., a count, or an amount of data,
// with the most recently started or stopped phase; or the very first
// phase if Start or Stop hasn't been called yet. The unit specifies
// the unit of measurement (e.g., MB, lines, no. of funcs, etc.).
func (t *Timings) AddEvent(size int64, unit string) {
	m := t.events
	if m == nil {
		m = make(map[int][]*event)
		t.events = m
	}
	i := len(t.list)
	if i > 0 {
		i--
	}
	m[i] = append(m[i], &event{size, unit})
}

// Write prints the phase times to w.
// The prefix is printed at the start of each line.
func (t *Timings) Write(w io.Writer, prefix string) {
	if len(t.list) > 0 {
		var lines lines

		// group of phases with shared non-empty label prefix
		var group struct {
			label string        // label prefix
			tot   time.Duration // accumulated phase time
			size  int           // number of phases collected in group
		}

		// accumulated time between Stop/Start timestamps
		var unaccounted time.Duration

		// process Start/Stop timestamps
		pt := &t.list[0] // previous timestamp
		tot := t.list[len(t.list)-1].time.Sub(pt.time)
		for i := 1; i < len(t.list); i++ {
			qt := &t.list[i] // current timestamp
			dt := qt.time.Sub(pt.time)

			var label string
			var events []*event
			if pt.start {
				// previous phase started
				label = pt.label
				events = t.events[i-1]
				if qt.start {
					// start implicitly ended previous phase; nothing to do
				} else {
					// stop ended previous phase; append stop labels, if any
					if qt.label != "" {
						label += ":" + qt.label
					}
					// events associated with stop replace prior events
					if e := t.events[i]; e != nil {
						events = e
					}
				}
			} else {
				// previous phase stopped
				if qt.start {
					// between a stopped and started phase; unaccounted time
					unaccounted += dt
				} else {
					// previous stop implicitly started current phase
					label = qt.label
					events = t.events[i]
				}
			}
			if label != "" {
				// add phase to existing group, or start a new group
				l := commonPrefix(group.label, label)
				if group.size == 1 && l != "" || group.size > 1 && l == group.label {
					// add to existing group
					group.label = l
					group.tot += dt
					group.size++
				} else {
					// start a new group
					if group.size > 1 {
						lines.add(prefix+group.label+"subtotal", 1, group.tot, tot, nil)
					}
					group.label = label
					group.tot = dt
					group.size = 1
				}

				// write phase
				lines.add(prefix+label, 1, dt, tot, events)
			}

			pt = qt
		}

		if group.size > 1 {
			lines.add(prefix+group.label+"subtotal", 1, group.tot, tot, nil)
		}

		if unaccounted != 0 {
			lines.add(prefix+"unaccounted", 1, unaccounted, tot, nil)
		}

		lines.add(prefix+"total", 1, tot, tot, nil)

		lines.write(w)
	}
}

func commonPrefix(a, b string) string {
	i := 0
	for i < len(a) && i < len(b) && a[i] == b[i] {
		i++
	}
	return a[:i]
}

type lines [][]string

func (lines *lines) add(label string, n int, dt, tot time.Duration, events []*event) {
	var line []string
	add := func(format string, args ...interface{}) {
		line = append(line, fmt.Sprintf(format, args...))
	}

	add("%s", label)
	add("    %d", n)
	add("    %d ns/op", dt)
	add("    %.2f %%", float64(dt)/float64(tot)*100)

	for _, e := range events {
		add("    %d", e.size)
		add(" %s", e.unit)
		add("    %d", int64(float64(e.size)/dt.Seconds()+0.5))
		add(" %s/s", e.unit)
	}

	*lines = append(*lines, line)
}

func (lines lines) write(w io.Writer) {
	// determine column widths and contents
	var widths []int
	var number []bool
	for _, line := range lines {
		for i, col := range line {
			if i < len(widths) {
				if len(col) > widths[i] {
					widths[i] = len(col)
				}
			} else {
				widths = append(widths, len(col))
				number = append(number, isnumber(col)) // first line determines column contents
			}
		}
	}

	// make column widths a multiple of align for more stable output
	const align = 1 // set to a value > 1 to enable
	if align > 1 {
		for i, w := range widths {
			w += align - 1
			widths[i] = w - w%align
		}
	}

	// print lines taking column widths and contents into account
	for _, line := range lines {
		for i, col := range line {
			format := "%-*s"
			if number[i] {
				format = "%*s" // numbers are right-aligned
			}
			fmt.Fprintf(w, format, widths[i], col)
		}
		fmt.Fprintln(w)
	}
}

func isnumber(s string) bool {
	for _, ch := range s {
		if ch <= ' ' {
			continue // ignore leading whitespace
		}
		return '0' <= ch && ch <= '9' || ch == '.' || ch == '-' || ch == '+'
	}
	return false
}
