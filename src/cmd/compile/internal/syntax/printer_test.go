// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strings"
	"testing"
)

func TestPrint(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	// provide a no-op error handler so parsing doesn't stop after first error
	ast, err := ParseFile(*src_, func(error) {}, nil, 0)
	if err != nil {
		t.Error(err)
	}

	if ast != nil {
		Fprint(testOut(), ast, true)
		fmt.Println()
	}
}

var stringTests = []string{
	"package p",
	"package p; type _ int; type T1 = struct{}; type ( _ *struct{}; T2 = float32 )",

	// channels
	"package p; type _ chan chan int",
	"package p; type _ chan (<-chan int)",
	"package p; type _ chan chan<- int",

	"package p; type _ <-chan chan int",
	"package p; type _ <-chan <-chan int",
	"package p; type _ <-chan chan<- int",

	"package p; type _ chan<- chan int",
	"package p; type _ chan<- <-chan int",
	"package p; type _ chan<- chan<- int",

	// TODO(gri) expand
}

func TestPrintString(t *testing.T) {
	for _, want := range stringTests {
		ast, err := Parse(nil, strings.NewReader(want), nil, nil, 0)
		if err != nil {
			t.Error(err)
			continue
		}
		if got := String(ast); got != want {
			t.Errorf("%q: got %q", want, got)
		}
	}
}

func testOut() io.Writer {
	if testing.Verbose() {
		return os.Stdout
	}
	return ioutil.Discard
}
