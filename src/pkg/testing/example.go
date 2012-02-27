// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"
)

var matchExamples = flag.String("test.example", "", "regular expression to select examples to run")

type InternalExample struct {
	Name   string
	F      func()
	Output string
}

func RunExamples(matchString func(pat, str string) (bool, error), examples []InternalExample) (ok bool) {
	if *match != "" {
		return // Don't run examples if testing is restricted: we're debugging.
	}
	ok = true

	var eg InternalExample

	stdout, stderr := os.Stdout, os.Stderr

	for _, eg = range examples {
		matched, err := matchString(*matchExamples, eg.Name)
		if err != nil {
			fmt.Fprintf(os.Stderr, "testing: invalid regexp for -test.example: %s\n", err)
			os.Exit(1)
		}
		if !matched {
			continue
		}
		if *chatty {
			fmt.Printf("=== RUN: %s\n", eg.Name)
		}

		// capture stdout and stderr
		r, w, err := os.Pipe()
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Stdout, os.Stderr = w, w
		outC := make(chan string)
		go func() {
			buf := new(bytes.Buffer)
			_, err := io.Copy(buf, r)
			if err != nil {
				fmt.Fprintf(stderr, "testing: copying pipe: %v\n", err)
				os.Exit(1)
			}
			outC <- buf.String()
		}()

		// run example
		t0 := time.Now()
		eg.F()
		dt := time.Now().Sub(t0)

		// close pipe, restore stdout/stderr, get output
		w.Close()
		os.Stdout, os.Stderr = stdout, stderr
		out := <-outC

		// report any errors
		tstr := fmt.Sprintf("(%.2f seconds)", dt.Seconds())
		if g, e := strings.TrimSpace(out), strings.TrimSpace(eg.Output); g != e {
			fmt.Printf("--- FAIL: %s %s\ngot:\n%s\nwant:\n%s\n",
				eg.Name, tstr, g, e)
			ok = false
		} else if *chatty {
			fmt.Printf("--- PASS: %s %s\n", eg.Name, tstr)
		}
	}

	return
}
