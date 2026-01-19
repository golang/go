// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd
// +build darwin dragonfly freebsd linux netbsd openbsd

// This is in testprognet instead of testprog because testprog
// must not import anything (like net, but also like os/signal)
// that kicks off background goroutines during init.

package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"sync"
	"syscall"
	"time"
)

func init() {
	register("SignalDuringExec", SignalDuringExec)
	register("SignalDuringExecPgrp", SignalDuringExecPgrp)
	register("Nop", Nop)
}

func SignalDuringExec() {
	// Re-launch ourselves in a new process group.
	cmd := exec.Command(os.Args[0], "SignalDuringExecPgrp")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true,
	}

	// Start the new process with an extra pipe. It will
	// exit if the pipe is closed.
	rp, wp, err := os.Pipe()
	if err != nil {
		fmt.Printf("Failed to create pipe: %v", err)
		return
	}
	cmd.ExtraFiles = []*os.File{rp}

	// Run the command.
	if err := cmd.Run(); err != nil {
		fmt.Printf("Run failed: %v", err)
	}

	// We don't actually need to write to the pipe, it just
	// needs to get closed, which will happen on process
	// exit.
	runtime.KeepAlive(wp)
}

func SignalDuringExecPgrp() {
	// Grab fd 3 which is a pipe we need to read on.
	f := os.NewFile(3, "pipe")
	go func() {
		// Nothing will ever get written to the pipe, so we'll
		// just block on it. If it closes, ReadAll will return
		// one way or another, at which point we'll exit.
		io.ReadAll(f)
		os.Exit(1)
	}()

	// This is just for SignalDuringExec.
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
