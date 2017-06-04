// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package inprogress_interrupt_test

import (
	"os"
	"os/signal"
	"sync"
	"syscall"
	"testing"
)

func TestParallel(t *testing.T) {
	t.Parallel()
}

func TestSerial(t *testing.T) {
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		<-sigCh // catch initial signal
		<-sigCh // catch propagated signal
		wg.Done()
	}()

	proc, err := os.FindProcess(syscall.Getpid())
	if err != nil {
		t.Fatalf("unable to find current process: %v", err)
	}
	err = proc.Signal(os.Interrupt)
	if err != nil {
		t.Fatalf("failed to interrupt current process: %v", err)
	}

	wg.Wait()
}
