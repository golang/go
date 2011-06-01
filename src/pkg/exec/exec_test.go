// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import (
	"fmt"
	"io"
	"testing"
	"os"
	"strconv"
	"strings"
)

func helperCommand(s ...string) *Cmd {
	cs := []string{"-test.run=exec.TestHelperProcess", "--"}
	cs = append(cs, s...)
	cmd := Command(os.Args[0], cs...)
	cmd.Env = append([]string{"GO_WANT_HELPER_PROCESS=1"}, os.Environ()...)
	return cmd
}

func TestEcho(t *testing.T) {
	bs, err := helperCommand("echo", "foo bar", "baz").Output()
	if err != nil {
		t.Errorf("echo: %v", err)
	}
	if g, e := string(bs), "foo bar baz\n"; g != e {
		t.Errorf("echo: want %q, got %q", e, g)
	}
}

func TestCatStdin(t *testing.T) {
	// Cat, testing stdin and stdout.
	input := "Input string\nLine 2"
	p := helperCommand("cat")
	p.Stdin = strings.NewReader(input)
	bs, err := p.Output()
	if err != nil {
		t.Errorf("cat: %v", err)
	}
	s := string(bs)
	if s != input {
		t.Errorf("cat: want %q, got %q", input, s)
	}
}

func TestCatGoodAndBadFile(t *testing.T) {
	// Testing combined output and error values.
	bs, err := helperCommand("cat", "/bogus/file.foo", "exec_test.go").CombinedOutput()
	if _, ok := err.(*os.Waitmsg); !ok {
		t.Errorf("expected Waitmsg from cat combined; got %T: %v", err, err)
	}
	s := string(bs)
	sp := strings.Split(s, "\n", 2)
	if len(sp) != 2 {
		t.Fatalf("expected two lines from cat; got %q", s)
	}
	errLine, body := sp[0], sp[1]
	if !strings.HasPrefix(errLine, "Error: open /bogus/file.foo") {
		t.Errorf("expected stderr to complain about file; got %q", errLine)
	}
	if !strings.Contains(body, "func TestHelperProcess(t *testing.T)") {
		t.Errorf("expected test code; got %q (len %d)", body, len(body))
	}
}


func TestNoExistBinary(t *testing.T) {
	// Can't run a non-existent binary
	err := Command("/no-exist-binary").Run()
	if err == nil {
		t.Error("expected error from /no-exist-binary")
	}
}

func TestExitStatus(t *testing.T) {
	// Test that exit values are returned correctly
	err := helperCommand("exit", "42").Run()
	if werr, ok := err.(*os.Waitmsg); ok {
		if s, e := werr.String(), "exit status 42"; s != e {
			t.Errorf("from exit 42 got exit %q, want %q", s, e)
		}
	} else {
		t.Fatalf("expected Waitmsg from exit 42; got %T: %v", err, err)
	}
}

// TestHelperProcess isn't a real test. It's used as a helper process
// for TestParameterRun.
func TestHelperProcess(*testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	defer os.Exit(0)

	args := os.Args
	for len(args) > 0 {
		if args[0] == "--" {
			args = args[1:]
			break
		}
		args = args[1:]
	}
	if len(args) == 0 {
		fmt.Fprintf(os.Stderr, "No command\n")
		os.Exit(2)
	}

	cmd, args := args[0], args[1:]
	switch cmd {
	case "echo":
		iargs := []interface{}{}
		for _, s := range args {
			iargs = append(iargs, s)
		}
		fmt.Println(iargs...)
	case "cat":
		if len(args) == 0 {
			io.Copy(os.Stdout, os.Stdin)
			return
		}
		exit := 0
		for _, fn := range args {
			f, err := os.Open(fn)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				exit = 2
			} else {
				defer f.Close()
				io.Copy(os.Stdout, f)
			}
		}
		os.Exit(exit)
	case "exit":
		n, _ := strconv.Atoi(args[0])
		os.Exit(n)
	default:
		fmt.Fprintf(os.Stderr, "Unknown command %q\n", cmd)
		os.Exit(2)
	}
}
