// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package signal

import (
	"flag"
	"os"
	"syscall"
	"testing"
	"time"
)

var runCtrlBreakTest = flag.Bool("run_ctlbrk_test", false, "force to run Ctrl+Break test")

func sendCtrlBreak(t *testing.T) {
	d, e := syscall.LoadDLL("kernel32.dll")
	if e != nil {
		t.Fatalf("LoadDLL: %v\n", e)
	}
	p, e := d.FindProc("GenerateConsoleCtrlEvent")
	if e != nil {
		t.Fatalf("FindProc: %v\n", e)
	}
	r, _, e := p.Call(0, 0)
	if r == 0 {
		t.Fatalf("GenerateConsoleCtrlEvent: %v\n", e)
	}
}

func TestCtrlBreak(t *testing.T) {
	if !*runCtrlBreakTest {
		t.Logf("test disabled; use -run_ctlbrk_test to enable")
		return
	}
	go func() {
		time.Sleep(1 * time.Second)
		sendCtrlBreak(t)
	}()
	c := make(chan os.Signal, 10)
	Notify(c)
	select {
	case s := <-c:
		if s != os.Interrupt {
			t.Fatalf("Wrong signal received: got %q, want %q\n", s, os.Interrupt)
		}
	case <-time.After(3 * time.Second):
		t.Fatalf("Timeout waiting for Ctrl+Break\n")
	}
}
