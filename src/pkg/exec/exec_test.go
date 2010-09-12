// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import (
	"io"
	"io/ioutil"
	"testing"
	"os"
)

func TestRunCat(t *testing.T) {
	cat, err := LookPath("cat")
	if err != nil {
		t.Fatal("cat: ", err)
	}
	cmd, err := Run(cat, []string{"cat"}, nil, "",
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
	echo, err := LookPath("echo")
	if err != nil {
		t.Fatal("echo: ", err)
	}
	cmd, err := Run(echo, []string{"echo", "hello", "world"}, nil, "",
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
	sh, err := LookPath("sh")
	if err != nil {
		t.Fatal("sh: ", err)
	}
	cmd, err := Run(sh, []string{"sh", "-c", "echo hello world 1>&2"}, nil, "",
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
	sh, err := LookPath("sh")
	if err != nil {
		t.Fatal("sh: ", err)
	}
	cmd, err := Run(sh, []string{"sh", "-c", "echo hello world 1>&2"}, nil, "",
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

func TestAddEnvVar(t *testing.T) {
	err := os.Setenv("NEWVAR", "hello world")
	if err != nil {
		t.Fatal("setenv:", err)
	}
	sh, err := LookPath("sh")
	if err != nil {
		t.Fatal("sh: ", err)
	}
	cmd, err := Run(sh, []string{"sh", "-c", "echo $NEWVAR"}, nil, "",
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
