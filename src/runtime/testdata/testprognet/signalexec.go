// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd

// This is in testprognet instead of testprog because testprog
// must not import anything (like net, but also like os/signal)
// that kicks off background goroutines during init.

package main

import (
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

func init() {
	register("SignalDuringExec", SignalDuringExec)
	register("Nop", Nop)
}

func SignalDuringExec() {
	pgrp := syscall.Getpgrp()

	const tries = 10

	var wg sync.WaitGroup
	c := make(chan os.Signal, tries)
	signal.Notify(c, syscall.SIGWINCH)
	wg.Add(1)
	go func() {
		defer wg.Done()
		for range c {
		}
	}()

	for i := 0; i < tries; i++ {
		time.Sleep(time.Microsecond)
		wg.Add(2)
		go func() {
			defer wg.Done()
			cmd := exec.Command(os.Args[0], "Nop")
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			if err := cmd.Run(); err != nil {
				fmt.Printf("Start failed: %v", err)
			}
		}()
		go func() {
			defer wg.Done()
			syscall.Kill(-pgrp, syscall.SIGWINCH)
		}()
	}

	signal.Stop(c)
	close(c)
	wg.Wait()

	fmt.Println("OK")
}

func Nop() {
	// This is just for SignalDuringExec.
}
