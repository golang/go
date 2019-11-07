// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"os"
	"os/exec"
	"runtime"
	"testing"
)

func TestExpandUser(t *testing.T) {
	env := "HOME"
	if runtime.GOOS == "windows" {
		env = "USERPROFILE"
	} else if runtime.GOOS == "plan9" {
		env = "home"
	}

	oldenv := os.Getenv(env)
	os.Setenv(env, "/home/gopher")
	defer os.Setenv(env, oldenv)

	tests := []struct {
		input string
		want  string
	}{
		{input: "~/foo", want: "/home/gopher/foo"},
		{input: "${HOME}/foo", want: "/home/gopher/foo"},
		{input: "/~/foo", want: "/~/foo"},
	}
	for _, tt := range tests {
		got := expandUser(tt.input)
		if got != tt.want {
			t.Fatalf("want %q, but %q", tt.want, got)
		}
	}
}

func TestCmdErr(t *testing.T) {
	tests := []struct {
		input error
		want  string
	}{
		{input: errors.New("cmd error"), want: "cmd error"},
		{input: &exec.ExitError{ProcessState: nil, Stderr: nil}, want: "<nil>"},
		{input: &exec.ExitError{ProcessState: nil, Stderr: []byte("test")}, want: "<nil>: test"},
	}

	for i, tt := range tests {
		got := cmdErr(tt.input)
		if got != tt.want {
			t.Fatalf("%d. got %q, want %q", i, got, tt.want)
		}
	}
}
