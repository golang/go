// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import (
	"io";
	"io/ioutil";
	"testing";
)

func TestRunCat(t *testing.T) {
	cmd, err := Run("/bin/cat", []string{"cat"}, nil,
		Pipe, Pipe, DevNull);
	if err != nil {
		t.Fatalf("opencmd /bin/cat: %v", err)
	}
	io.WriteString(cmd.Stdin, "hello, world\n");
	cmd.Stdin.Close();
	buf, err := ioutil.ReadAll(cmd.Stdout);
	if err != nil {
		t.Fatalf("reading from /bin/cat: %v", err)
	}
	if string(buf) != "hello, world\n" {
		t.Fatalf("reading from /bin/cat: got %q", buf)
	}
	if err = cmd.Close(); err != nil {
		t.Fatalf("closing /bin/cat: %v", err)
	}
}

func TestRunEcho(t *testing.T) {
	cmd, err := Run("/bin/echo", []string{"echo", "hello", "world"}, nil,
		DevNull, Pipe, DevNull);
	if err != nil {
		t.Fatalf("opencmd /bin/echo: %v", err)
	}
	buf, err := ioutil.ReadAll(cmd.Stdout);
	if err != nil {
		t.Fatalf("reading from /bin/echo: %v", err)
	}
	if string(buf) != "hello world\n" {
		t.Fatalf("reading from /bin/echo: got %q", buf)
	}
	if err = cmd.Close(); err != nil {
		t.Fatalf("closing /bin/echo: %v", err)
	}
}
