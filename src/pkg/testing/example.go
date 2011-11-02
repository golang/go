// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"time"
)

type InternalExample struct {
	Name   string
	F      func()
	Output string
}

func RunExamples(examples []InternalExample) (ok bool) {
	ok = true

	stdout, stderr := os.Stdout, os.Stderr
	defer func() {
		os.Stdout, os.Stderr = stdout, stderr
		if e := recover(); e != nil {
			if err, ok := e.(error); ok {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			panic(e)
		}
	}()

	for _, eg := range examples {
		if *chatty {
			fmt.Fprintln(os.Stderr, "=== RUN:", eg.Name)
		}

		// capture stdout and stderr for testing purposes
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
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			outC <- buf.String()
		}()

		// run example
		ns := -time.Nanoseconds()
		eg.F()
		ns += time.Nanoseconds()

		// close pipe, restore stdout/stderr, get output
		w.Close()
		os.Stdout, os.Stderr = stdout, stderr
		out := <-outC

		// report any errors
		if out != eg.Output {
			fmt.Fprintf(
				os.Stderr,
				"--- FAIL: %s\ngot:\n%s\nwant:\n%s\n",
				eg.Name, out, eg.Output,
			)
			ok = false
		} else if *chatty {
			tstr := fmt.Sprintf("(%.2f seconds)", float64(ns)/1e9)
			fmt.Fprintln(os.Stderr, "--- PASS:", eg.Name, tstr)
		}
	}

	return
}
