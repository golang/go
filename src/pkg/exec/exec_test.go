// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import (
	"io"
	"io/ioutil"
	"testing"
)

func TestRunCat(t *testing.T) {
	cmd, err := Run("/bin/cat", []string{"cat"}, nil, "",
		Pipe, Pipe, DevNull)
	if err != nil {
		t.Fatal("run:", err)
	}
	io.WriteString(cmd.Stdin, "hello, world\n")
	cmd.Stdin.Close()
	buf, err := ioutil.ReadAll(cmd.Stdout)
	if err != nil {
		t.Fatal("read:", err)
	}
	if string(buf) != "hello, world\n" {
		t.Fatalf("read: got %q", buf)
	}
	if err = cmd.Close(); err != nil {
		t.Fatal("close:", err)
	}
}

func TestRunEcho(t *testing.T) {
	cmd, err := Run("/bin/echo", []string{"echo", "hello", "world"}, nil, "",
		DevNull, Pipe, DevNull)
	if err != nil {
		t.Fatal("run:", err)
	}
	buf, err := ioutil.ReadAll(cmd.Stdout)
	if err != nil {
		t.Fatal("read:", err)
	}
	if string(buf) != "hello world\n" {
		t.Fatalf("read: got %q", buf)
	}
	if err = cmd.Close(); err != nil {
		t.Fatal("close:", err)
	}
}

func TestStderr(t *testing.T) {
	cmd, err := Run("/bin/sh", []string{"sh", "-c", "echo hello world 1>&2"}, nil, "",
		DevNull, DevNull, Pipe)
	if err != nil {
		t.Fatal("run:", err)
	}
	buf, err := ioutil.ReadAll(cmd.Stderr)
	if err != nil {
		t.Fatal("read:", err)
	}
	if string(buf) != "hello world\n" {
		t.Fatalf("read: got %q", buf)
	}
	if err = cmd.Close(); err != nil {
		t.Fatal("close:", err)
	}
}


func TestMergeWithStdout(t *testing.T) {
	cmd, err := Run("/bin/sh", []string{"sh", "-c", "echo hello world 1>&2"}, nil, "",
		DevNull, Pipe, MergeWithStdout)
	if err != nil {
		t.Fatal("run:", err)
	}
	buf, err := ioutil.ReadAll(cmd.Stdout)
	if err != nil {
		t.Fatal("read:", err)
	}
	if string(buf) != "hello world\n" {
		t.Fatalf("read: got %q", buf)
	}
	if err = cmd.Close(); err != nil {
		t.Fatal("close:", err)
	}
}
