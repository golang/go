// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"text/template"
)

type crashTest struct {
	Cgo bool
}

// This test is a separate program, because it is testing
// both main (m0) and non-main threads (m).

func testCrashHandler(t *testing.T, ct *crashTest) {
	if runtime.GOOS == "freebsd" {
		// TODO(brainman): do not know why this test fails on freebsd
		t.Logf("skipping test on %q", runtime.GOOS)
		return
	}

	st := template.Must(template.New("crashSource").Parse(crashSource))

	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	src := filepath.Join(dir, "main.go")
	f, err := os.Create(src)
	if err != nil {
		t.Fatalf("failed to create %v: %v", src, err)
	}
	err = st.Execute(f, ct)
	if err != nil {
		f.Close()
		t.Fatalf("failed to execute template: %v", err)
	}
	f.Close()

	got, err := exec.Command("go", "run", src).CombinedOutput()
	if err != nil {
		t.Fatalf("program exited with error: %v\n%v", err, string(got))
	}
	want := "main: recovered done\nnew-thread: recovered done\nsecond-new-thread: recovered done\nmain-again: recovered done\n"
	if string(got) != string(want) {
		t.Fatalf("expected %q, but got %q", string(want), string(got))
	}
}

func TestCrashHandler(t *testing.T) {
	testCrashHandler(t, &crashTest{Cgo: false})
}

const crashSource = `
package main

import (
	"fmt"
	"runtime"
)

{{if .Cgo}}
import "C"
{{end}}

func test(name string) {
	defer func() {
		if x := recover(); x != nil {
			fmt.Printf(" recovered")
		}
		fmt.Printf(" done\n")
	}()
	fmt.Printf("%s:", name)
	var s *string
	_ = *s
	fmt.Print("SHOULD NOT BE HERE")
}

func testInNewThread(name string) {
	c := make(chan bool)
	go func() {
		runtime.LockOSThread()
		test(name)
		c <- true
	}()
	<-c
}

func main() {
	runtime.LockOSThread()
	test("main")
	testInNewThread("new-thread")
	testInNewThread("second-new-thread")
	test("main-again")
}
`
