// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Invoke signal hander in the VDSO context.
// See issue 32912 and 34391.

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"runtime/pprof"
	"syscall"
	"time"
)

func init() {
	register("SIGPROFInVDSO", signalSIGPROFInVDSO)
	register("SIGUSR1InVDSO", signalSIGUSR1InVDSO)
}

func signalSIGPROFInVDSO() {
	f, err := ioutil.TempFile("", "timeprofnow")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	if err := pprof.StartCPUProfile(f); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	t0 := time.Now()
	t1 := t0
	// We should get a profiling signal 100 times a second,
	// so running for 1 second should be sufficient.
	for t1.Sub(t0) < time.Second {
		t1 = time.Now()
	}

	pprof.StopCPUProfile()

	name := f.Name()
	if err := f.Close(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	if err := os.Remove(name); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	fmt.Println("success")
}

func signalSIGUSR1InVDSO() {
	done := make(chan struct {})

	p, _ := os.FindProcess(os.Getpid())
	go func() {
		for {
			p.Signal(syscall.SIGUSR1)
			select {
			case <- done:
				return
			default:
			}
		}
	}()

	t0 := time.Now()
	t1 := t0
	for t1.Sub(t0) < time.Second {
		t1 = time.Now()
	}
	close(done)

	fmt.Println("success")
}
