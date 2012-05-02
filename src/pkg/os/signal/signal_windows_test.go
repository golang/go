// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package signal

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

func sendCtrlBreak(t *testing.T, pid int) {
	d, e := syscall.LoadDLL("kernel32.dll")
	if e != nil {
		t.Fatalf("LoadDLL: %v\n", e)
	}
	p, e := d.FindProc("GenerateConsoleCtrlEvent")
	if e != nil {
		t.Fatalf("FindProc: %v\n", e)
	}
	r, _, e := p.Call(syscall.CTRL_BREAK_EVENT, uintptr(pid))
	if r == 0 {
		t.Fatalf("GenerateConsoleCtrlEvent: %v\n", e)
	}
}

func TestCtrlBreak(t *testing.T) {
	// create source file
	const source = `
package main

import (
	"log"
	"os"
	"os/signal"
	"time"
)


func main() {
	c := make(chan os.Signal, 10)
	signal.Notify(c)
	select {
	case s := <-c:
		if s != os.Interrupt {
			log.Fatalf("Wrong signal received: got %q, want %q\n", s, os.Interrupt)
		}
	case <-time.After(3 * time.Second):
		log.Fatalf("Timeout waiting for Ctrl+Break\n")
	}
}
`
	name := filepath.Join(os.TempDir(), "ctlbreak")
	src := name + ".go"
	defer os.Remove(src)
	f, err := os.Create(src)
	if err != nil {
		t.Fatalf("Failed to create %v: %v", src, err)
	}
	defer f.Close()
	f.Write([]byte(source))

	// compile it
	exe := name + ".exe"
	defer os.Remove(exe)
	o, err := exec.Command("go", "build", "-o", exe, src).CombinedOutput()
	if err != nil {
		t.Fatalf("Failed to compile: %v\n%v", err, string(o))
	}

	// run it
	cmd := exec.Command(exe)
	var b bytes.Buffer
	cmd.Stdout = &b
	cmd.Stderr = &b
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP,
	}
	err = cmd.Start()
	if err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	go func() {
		time.Sleep(1 * time.Second)
		sendCtrlBreak(t, cmd.Process.Pid)
	}()
	err = cmd.Wait()
	if err != nil {
		t.Fatalf("Program exited with error: %v\n%v", err, string(b.Bytes()))
	}
}
