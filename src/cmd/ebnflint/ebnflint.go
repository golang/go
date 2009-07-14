// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes";
	"ebnf";
	"flag";
	"fmt";
	"go/scanner";
	"io";
	"os";
	"path";
	"sort";
	"strings";
)


var start = flag.String("start", "Start", "name of start production");


func usage() {
	fmt.Fprintf(os.Stderr, "usage: ebnflint [flags] [filename]\n");
	flag.PrintDefaults();
	os.Exit(1);
}


// Markers around EBNF sections in .html files
var (
	open = strings.Bytes(`<pre class="ebnf">`);
	close = strings.Bytes(`</pre>`);
)


func extractEBNF(src []byte) []byte {
	var buf bytes.Buffer;

	for i, j, n := 0, 0, len(src); ; {
		// i = beginning of EBNF section
		i = bytes.Index(src[j : n], open);
		if i < 0 {
			break;
		}
		i += j+len(open);

		// write as many newlines as found in the excluded text
		// to maintain correct line numbers in error messages 
		for _, ch := range src[j : i] {
			if ch == '\n' {
				buf.WriteByte('\n');
			}
		}

		// j = end of EBNF section
		j = bytes.Index(src[i : n], close);
		if j < 0 {
			// missing closing
			// TODO(gri) should this be an error?
			j = n-i;
		}
		j += i;

		// copy EBNF section
		buf.Write(src[i : j]);
	}

	return buf.Data();
}


func main() {
	flag.Parse();

	var filename string;
	switch flag.NArg() {
	case 0:
		filename = "/dev/stdin";
	case 1:
		filename = flag.Arg(0);
	default:
		usage();
	}

	src, err := io.ReadFile(filename);
	if err != nil {
		scanner.PrintError(os.Stderr, err);
	}

	if path.Ext(filename) == ".html" {
		src = extractEBNF(src);
	}

	grammar, err := ebnf.Parse(filename, src);
	if err != nil {
		scanner.PrintError(os.Stderr, err);
	}

	if err = ebnf.Verify(grammar, *start); err != nil {
		scanner.PrintError(os.Stderr, err);
	}
}
