// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build openbsd

// This, on the face of it, bizarre testing mechanism is necessary because
// the only reliable way to gauge whether or not a pledge(2) call has succeeded
// is that the program has been killed as a result of breaking its pledge.

package unix_test

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"golang.org/x/sys/unix"
)

type testProc struct {
	fn      func()       // should always exit instead of returning
	cleanup func() error // for instance, delete coredumps from testing pledge
	success bool         // whether zero-exit means success or failure
}

var (
	testProcs = map[string]testProc{}
	procName  = ""
)

const (
	optName = "sys-unix-internal-procname"
)

func init() {
	flag.StringVar(&procName, optName, "", "internal use only")
}

// testCmd generates a proper command that, when executed, runs the test
// corresponding to the given key.
func testCmd(procName string) (*exec.Cmd, error) {
	exe, err := filepath.Abs(os.Args[0])
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(exe, "-"+optName+"="+procName)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	return cmd, nil
}

// ExitsCorrectly is a comprehensive, one-line-of-use wrapper for testing
// a testProc with a key.
func ExitsCorrectly(procName string, t *testing.T) {
	s := testProcs[procName]
	c, err := testCmd(procName)
	defer func() {
		if s.cleanup() != nil {
			t.Fatalf("Failed to run cleanup for %s", procName)
		}
	}()
	if err != nil {
		t.Fatalf("Failed to construct command for %s", procName)
	}
	if (c.Run() == nil) != s.success {
		result := "succeed"
		if !s.success {
			result = "fail"
		}
		t.Fatalf("Process did not %s when it was supposed to", result)
	}
}

func TestMain(m *testing.M) {
	flag.Parse()
	if procName != "" {
		testProcs[procName].fn()
	}
	os.Exit(m.Run())
}

// For example, add a test for pledge.
func init() {
	testProcs["pledge"] = testProc{
		func() {
			fmt.Println(unix.Pledge("", ""))
			os.Exit(0)
		},
		func() error {
			files, err := ioutil.ReadDir(".")
			if err != nil {
				return err
			}
			for _, file := range files {
				if filepath.Ext(file.Name()) == ".core" {
					if err := os.Remove(file.Name()); err != nil {
						return err
					}
				}
			}
			return nil
		},
		false,
	}
}

func TestPledge(t *testing.T) {
	ExitsCorrectly("pledge", t)
}
