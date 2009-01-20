// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bufio";
	"net";
	"os";
	"testing";
)

func TestReadLine(t *testing.T) {
	filename := "/etc/services";	// a nice big file

	fd, err := os.Open(filename, os.O_RDONLY, 0);
	if err != nil {
		t.Fatalf("open %s: %v", filename, err);
	}
	br, err1 := bufio.NewBufRead(fd);
	if err1 != nil {
		t.Fatalf("bufio.NewBufRead: %v", err1);
	}

	file := _Open(filename);
	if file == nil {
		t.Fatalf("net._Open(%s) = nil", filename);
	}

	lineno := 1;
	byteno := 0;
	for {
		bline, berr := br.ReadLineString('\n', false);
		line, ok := file.ReadLine();
		if (berr != nil) != !ok || bline != line {
			t.Fatalf("%s:%d (#%d)\nbufio => %q, %v\nnet => %q, %v",
				filename, lineno, byteno, bline, berr, line, ok);
		}
		if !ok {
			break
		}
		lineno++;
		byteno += len(line) + 1;
	}
}
