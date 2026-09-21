// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package signal

import (
	"bufio"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

func sendCtrlBreak(pid int) error {
	d, e := syscall.LoadDLL("kernel32.dll")
	if e != nil {
		return fmt.Errorf("LoadDLL: %v\n", e)
	}
	p, e := d.FindProc("GenerateConsoleCtrlEvent")
	if e != nil {
		return fmt.Errorf("FindProc: %v\n", e)
	}
	r, _, e := p.Call(syscall.CTRL_BREAK_EVENT, uintptr(pid))
	if r == 0 {
		return fmt.Errorf("GenerateConsoleCtrlEvent: %v\n", e)
	}
	return nil
}

func TestCtrlBreak(t *testing.T) {
	// create source file
	const source = `
package main

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"time"
)


func main() {
	c := make(chan os.Signal, 10)
	signal.Notify(c)
	fmt.Println("ready")
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
	tmp := t.TempDir()

	// write ctrlbreak.go
	name := filepath.Join(tmp, "ctrlbreak")
	src := name + ".go"
	f, err := os.Create(src)
	if err != nil {
		t.Fatalf("Failed to create %v: %v", src, err)
	}
	defer f.Close()
	f.Write([]byte(source))

	// compile it
	exe := name + ".exe"
	defer os.Remove(exe)
	o, err := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", exe, src).CombinedOutput()
	if err != nil {
		t.Fatalf("Failed to compile: %v\n%v", err, string(o))
	}

	// run it
	cmd := testenv.Command(t, exe)
	var buf strings.Builder
	cmd.Stderr = &buf
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("StdoutPipe failed: %v", err)
	}
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP,
	}
	err = cmd.Start()
	if err != nil {
		t.Fatalf("Start failed: %v", err)
	}

	outReader := bufio.NewReader(stdout)
	errCh := make(chan error, 1)
	go func() {
		if line, err := outReader.ReadString('\n'); err != nil {
			errCh <- fmt.Errorf("could not read stdout: %v", err)
		} else if strings.TrimSpace(line) != "ready" {
			errCh <- fmt.Errorf("unexpected message: %v", line)
		} else {
			errCh <- sendCtrlBreak(cmd.Process.Pid)
		}
	}()

	if err := <-errCh; err != nil {
		t.Fatal(err)
	}
	err = cmd.Wait()
	if err != nil {
		t.Fatalf("Program exited with error: %v\n%v", err, buf.String())
	}
}
