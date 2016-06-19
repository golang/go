// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9,!windows

package main

/*
#include <stddef.h>
#include <signal.h>
#include <pthread.h>

// Save the signal mask at startup so that we see what it is before
// the Go runtime starts setting up signals.

static sigset_t mask;

static void init(void) __attribute__ ((constructor));

static void init() {
	sigemptyset(&mask);
	pthread_sigmask(SIG_SETMASK, NULL, &mask);
}

int SIGINTBlocked() {
	return sigismember(&mask, SIGINT);
}
*/
import "C"

import (
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"sync"
	"syscall"
)

func init() {
	register("CgoExecSignalMask", CgoExecSignalMask)
}

func CgoExecSignalMask() {
	if len(os.Args) > 2 && os.Args[2] == "testsigint" {
		if C.SIGINTBlocked() != 0 {
			os.Exit(1)
		}
		os.Exit(0)
	}

	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGTERM)
	go func() {
		for range c {
		}
	}()

	const goCount = 10
	const execCount = 10
	var wg sync.WaitGroup
	wg.Add(goCount*execCount + goCount)
	for i := 0; i < goCount; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < execCount; j++ {
				c2 := make(chan os.Signal, 1)
				signal.Notify(c2, syscall.SIGUSR1)
				syscall.Kill(os.Getpid(), syscall.SIGTERM)
				go func(j int) {
					defer wg.Done()
					cmd := exec.Command(os.Args[0], "CgoExecSignalMask", "testsigint")
					cmd.Stdin = os.Stdin
					cmd.Stdout = os.Stdout
					cmd.Stderr = os.Stderr
					if err := cmd.Run(); err != nil {
						fmt.Printf("iteration %d: %v\n", j, err)
						os.Exit(1)
					}
				}(j)
				signal.Stop(c2)
			}
		}()
	}
	wg.Wait()

	fmt.Println("OK")
}
