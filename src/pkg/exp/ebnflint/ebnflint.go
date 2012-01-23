// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"exp/ebnf"
	"flag"
	"fmt"
	"go/scanner"
	"go/token"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
)

var fset = token.NewFileSet()
var start = flag.String("start", "Start", "name of start production")

func usage() {
	fmt.Fprintf(os.Stderr, "usage: ebnflint [flags] [filename]\n")
	flag.PrintDefaults()
	os.Exit(1)
}

// Markers around EBNF sections in .html files
var (
	open  = []byte(`<pre class="ebnf">`)
	close = []byte(`</pre>`)
)

func report(err error) {
	scanner.PrintError(os.Stderr, err)
	os.Exit(1)
}

func extractEBNF(src []byte) []byte {
	var buf bytes.Buffer

	for {
		// i = beginning of EBNF text
		i := bytes.Index(src, open)
		if i < 0 {
			break // no EBNF found - we are done
		}
		i += len(open)

		// write as many newlines as found in the excluded text
		// to maintain correct line numbers in error messages
		for _, ch := range src[0:i] {
			if ch == '\n' {
				buf.WriteByte('\n')
			}
		}

		// j = end of EBNF text (or end of source)
		j := bytes.Index(src[i:], close) // close marker
		if j < 0 {
			j = len(src) - i
		}
		j += i

		// copy EBNF text
		buf.Write(src[i:j])

		// advance
		src = src[j:]
	}

	return buf.Bytes()
}

func main() {
	flag.Parse()

	var (
		name string
		r    io.Reader
	)
	switch flag.NArg() {
	case 0:
		name, r = "<stdin>", os.Stdin
	case 1:
		name = flag.Arg(0)
	default:
		usage()
	}

	if err := verify(name, *start, r); err != nil {
		report(err)
	}
}

func verify(name, start string, r io.Reader) error {
	if r == nil {
		f, err := os.Open(name)
		if err != nil {
			return err
		}
		defer f.Close()
		r = f
	}

	src, err := ioutil.ReadAll(r)
	if err != nil {
		return err
	}

	if filepath.Ext(name) == ".html" || bytes.Index(src, open) >= 0 {
		src = extractEBNF(src)
	}

	grammar, err := ebnf.Parse(name, bytes.NewBuffer(src))
	if err != nil {
		return err
	}

	return ebnf.Verify(grammar, start)
}
