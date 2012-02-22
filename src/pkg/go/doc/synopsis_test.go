// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import "testing"

var tests = []struct {
	txt string
	fsl int
	syn string
}{
	{"", 0, ""},
	{"foo", 3, "foo"},
	{"foo.", 4, "foo."},
	{"foo.bar", 7, "foo.bar"},
	{"  foo.  ", 6, "foo."},
	{"  foo\t  bar.\n", 12, "foo bar."},
	{"  foo\t  bar.\n", 12, "foo bar."},
	{"a  b\n\nc\r\rd\t\t", 12, "a b c d"},
	{"a  b\n\nc\r\rd\t\t  . BLA", 15, "a b c d ."},
	{"Package poems by T.S.Eliot. To rhyme...", 27, "Package poems by T.S.Eliot."},
	{"Package poems by T. S. Eliot. To rhyme...", 29, "Package poems by T. S. Eliot."},
	{"foo implements the foo ABI. The foo ABI is...", 27, "foo implements the foo ABI."},
	{"Package\nfoo. ..", 12, "Package foo."},
	{"P . Q.", 3, "P ."},
	{"P. Q.   ", 8, "P. Q."},
	{"Package Καλημέρα κόσμε.", 36, "Package Καλημέρα κόσμε."},
	{"Package こんにちは 世界\n", 31, "Package こんにちは 世界"},
}

func TestSynopsis(t *testing.T) {
	for _, e := range tests {
		fsl := firstSentenceLen(e.txt)
		if fsl != e.fsl {
			t.Errorf("got fsl = %d; want %d for %q\n", fsl, e.fsl, e.txt)
		}
		syn := Synopsis(e.txt)
		if syn != e.syn {
			t.Errorf("got syn = %q; want %q for %q\n", syn, e.syn, e.txt)
		}
	}
}
