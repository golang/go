// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stack

import (
	"bytes"
	"fmt"
	"io"
	"runtime"
	"sort"
)

// Capture get the current stack traces from the runtime.
func Capture() Dump {
	buf := make([]byte, 2<<20)
	buf = buf[:runtime.Stack(buf, true)]
	scanner := NewScanner(bytes.NewReader(buf))
	dump, _ := Parse(scanner)
	return dump
}

// Summarize a dump for easier consumption.
// This collates goroutines with equivalent stacks.
func Summarize(dump Dump) Summary {
	s := Summary{
		Total: len(dump),
	}
	for _, gr := range dump {
		s.addGoroutine(gr)
	}
	return s
}

// Process and input stream to an output stream, summarizing any stacks that
// are detected in place.
func Process(out io.Writer, in io.Reader) error {
	scanner := NewScanner(in)
	for {
		dump, err := Parse(scanner)
		summary := Summarize(dump)
		switch {
		case len(dump) > 0:
			fmt.Fprintf(out, "%+v\n\n", summary)
		case err != nil:
			return err
		case scanner.Done():
			return scanner.Err()
		default:
			// must have been a line that is not part of a dump
			fmt.Fprintln(out, scanner.Next())
		}
	}
}

// Diff calculates the delta between two dumps.
func Diff(before, after Dump) Delta {
	result := Delta{}
	processed := make(map[int]bool)
	for _, gr := range before {
		processed[gr.ID] = false
	}
	for _, gr := range after {
		if _, found := processed[gr.ID]; found {
			result.Shared = append(result.Shared, gr)
		} else {
			result.After = append(result.After, gr)
		}
		processed[gr.ID] = true
	}
	for _, gr := range before {
		if done := processed[gr.ID]; !done {
			result.Before = append(result.Before, gr)
		}
	}
	return result
}

// TODO: do we want to allow contraction of stacks before comparison?
func (s *Summary) addGoroutine(gr Goroutine) {
	index := sort.Search(len(s.Calls), func(i int) bool {
		return !s.Calls[i].Stack.less(gr.Stack)
	})
	if index >= len(s.Calls) || !s.Calls[index].Stack.equal(gr.Stack) {
		// insert new stack, first increase the length
		s.Calls = append(s.Calls, Call{})
		// move the top part upward to make space
		copy(s.Calls[index+1:], s.Calls[index:])
		// insert the new call
		s.Calls[index] = Call{
			Stack: gr.Stack,
		}
	}
	// merge the goroutine into the matched call
	s.Calls[index].merge(gr)
}

// TODO: do we want other grouping strategies?
func (c *Call) merge(gr Goroutine) {
	for i := range c.Groups {
		canditate := &c.Groups[i]
		if canditate.State == gr.State {
			canditate.Goroutines = append(canditate.Goroutines, gr)
			return
		}
	}
	c.Groups = append(c.Groups, Group{
		State:      gr.State,
		Goroutines: []Goroutine{gr},
	})
}
