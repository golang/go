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

func report(err os.Error) {
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
		filename string
		src      []byte
		err      os.Error
	)
	switch flag.NArg() {
	case 0:
		filename = "<stdin>"
		src, err = ioutil.ReadAll(os.Stdin)
	case 1:
		filename = flag.Arg(0)
		src, err = ioutil.ReadFile(filename)
	default:
		usage()
	}
	if err != nil {
		report(err)
	}

	if filepath.Ext(filename) == ".html" || bytes.Index(src, open) >= 0 {
		src = extractEBNF(src)
	}

	grammar, err := ebnf.Parse(filename, bytes.NewBuffer(src))
	if err != nil {
		report(err)
	}

	if err = ebnf.Verify(grammar, *start); err != nil {
		report(err)
	}
}
