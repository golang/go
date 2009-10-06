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
	"strings";
)


var start = flag.String("start", "Start", "name of start production")


func usage() {
	fmt.Fprintf(os.Stderr, "usage: ebnflint [flags] [filename]\n");
	flag.PrintDefaults();
	os.Exit(1);
}


// Markers around EBNF sections in .html files
var (
	open	= strings.Bytes(`<pre class="ebnf">`);
	close	= strings.Bytes(`</pre>`);
)


func extractEBNF(src []byte) []byte {
	var buf bytes.Buffer;

	for {
		// i = beginning of EBNF text
		i := bytes.Index(src, open);
		if i < 0 {
			break;	// no EBNF found - we are done
		}
		i += len(open);

		// write as many newlines as found in the excluded text
		// to maintain correct line numbers in error messages
		for _, ch := range src[0:i] {
			if ch == '\n' {
				buf.WriteByte('\n');
			}
		}

		// j = end of EBNF text (or end of source)
		j := bytes.Index(src[i:len(src)], close);	// close marker
		if j < 0 {
			j = len(src)-i;
		}
		j += i;

		// copy EBNF text
		buf.Write(src[i:j]);

		// advance
		src = src[j:len(src)];
	}

	return buf.Bytes();
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
