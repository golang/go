// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import (
	"io"
	"io/ioutil"
	"testing"
	"os"
	"runtime"
)

func run(argv []string, stdin, stdout, stderr int) (p *Cmd, err os.Error) {
	if runtime.GOOS == "windows" {
		argv = append([]string{"cmd", "/c"}, argv...)
	}
	exe, err := LookPath(argv[0])
	if err != nil {
		return nil, err
	}
	p, err = Run(exe, argv, nil, "", stdin, stdout, stderr)
	return p, err
}

func TestRunCat(t *testing.T) {
	cmd, err := run([]string{"cat"}, Pipe, Pipe, DevNull)
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
	cmd, err := run([]string{"sh", "-c", "echo hello world"},
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
	cmd, err := run([]string{"sh", "-c", "echo hello world 1>&2"},
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
	cmd, err := run([]string{"sh", "-c", "echo hello world 1>&2"},
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
	cmd, err := run([]string{"sh", "-c", "echo $NEWVAR"},
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
