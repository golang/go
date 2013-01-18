// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
	"time"
)

type InternalExample struct {
	Name   string
	F      func()
	Output string
}

func RunExamples(matchString func(pat, str string) (bool, error), examples []InternalExample) (ok bool) {
	ok = true

	var eg InternalExample

	for _, eg = range examples {
		matched, err := matchString(*match, eg.Name)
		if err != nil {
			fmt.Fprintf(os.Stderr, "testing: invalid regexp for -test.run: %s\n", err)
			os.Exit(1)
		}
		if !matched {
			continue
		}
		if !runExample(eg) {
			ok = false
		}
	}

	return
}

func runExample(eg InternalExample) (ok bool) {
	if *chatty {
		fmt.Printf("=== RUN: %s\n", eg.Name)
	}

	// Capture stdout.
	stdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	os.Stdout = w
	outC := make(chan string)
	go func() {
		buf := new(bytes.Buffer)
		_, err := io.Copy(buf, r)
		r.Close()
		if err != nil {
			fmt.Fprintf(os.Stderr, "testing: copying pipe: %v\n", err)
			os.Exit(1)
		}
		outC <- buf.String()
	}()

	start := time.Now()
	ok = true

	// Clean up in a deferred call so we can recover if the example panics.
	defer func() {
		d := time.Now().Sub(start)

		// Close pipe, restore stdout, get output.
		w.Close()
		os.Stdout = stdout
		out := <-outC

		var fail string
		err := recover()
		if g, e := strings.TrimSpace(out), strings.TrimSpace(eg.Output); g != e && err == nil {
			fail = fmt.Sprintf("got:\n%s\nwant:\n%s\n", g, e)
		}
		if fail != "" || err != nil {
			fmt.Printf("--- FAIL: %s (%v)\n%s", eg.Name, d, fail)
			ok = false
		} else if *chatty {
			fmt.Printf("--- PASS: %s (%v)\n", eg.Name, d)
		}
		if err != nil {
			panic(err)
		}
	}()

	// Run example.
	eg.F()
	return
}
