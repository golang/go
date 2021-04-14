// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9,!windows

package main

/*
#include <errno.h>
#include <signal.h>
#include <string.h>

static int clearRestart(int sig) {
	struct sigaction sa;

	memset(&sa, 0, sizeof sa);
	if (sigaction(sig, NULL, &sa) < 0) {
		return errno;
	}
	sa.sa_flags &=~ SA_RESTART;
	if (sigaction(sig, &sa, NULL) < 0) {
		return errno;
	}
	return 0;
}
*/
import "C"

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"
)

func init() {
	register("EINTR", EINTR)
	register("Block", Block)
}

// Test various operations when a signal handler is installed without
// the SA_RESTART flag. This tests that the os and net APIs handle EINTR.
func EINTR() {
	if errno := C.clearRestart(C.int(syscall.SIGURG)); errno != 0 {
		log.Fatal(syscall.Errno(errno))
	}
	if errno := C.clearRestart(C.int(syscall.SIGWINCH)); errno != 0 {
		log.Fatal(syscall.Errno(errno))
	}
	if errno := C.clearRestart(C.int(syscall.SIGCHLD)); errno != 0 {
		log.Fatal(syscall.Errno(errno))
	}

	var wg sync.WaitGroup
	testPipe(&wg)
	testNet(&wg)
	testExec(&wg)
	wg.Wait()
	fmt.Println("OK")
}

// spin does CPU bound spinning and allocating for a millisecond,
// to get a SIGURG.
//go:noinline
func spin() (float64, []byte) {
	stop := time.Now().Add(time.Millisecond)
	r1 := 0.0
	r2 := make([]byte, 200)
	for time.Now().Before(stop) {
		for i := 1; i < 1e6; i++ {
			r1 += r1 / float64(i)
			r2 = append(r2, bytes.Repeat([]byte{byte(i)}, 100)...)
			r2 = r2[100:]
		}
	}
	return r1, r2
}

// winch sends a few SIGWINCH signals to the process.
func winch() {
	ticker := time.NewTicker(100 * time.Microsecond)
	defer ticker.Stop()
	pid := syscall.Getpid()
	for n := 10; n > 0; n-- {
		syscall.Kill(pid, syscall.SIGWINCH)
		<-ticker.C
	}
}

// sendSomeSignals triggers a few SIGURG and SIGWINCH signals.
func sendSomeSignals() {
	done := make(chan struct{})
	go func() {
		spin()
		close(done)
	}()
	winch()
	<-done
}

// testPipe tests pipe operations.
func testPipe(wg *sync.WaitGroup) {
	r, w, err := os.Pipe()
	if err != nil {
		log.Fatal(err)
	}
	if err := syscall.SetNonblock(int(r.Fd()), false); err != nil {
		log.Fatal(err)
	}
	if err := syscall.SetNonblock(int(w.Fd()), false); err != nil {
		log.Fatal(err)
	}
	wg.Add(2)
	go func() {
		defer wg.Done()
		defer w.Close()
		// Spin before calling Write so that the first ReadFull
		// in the other goroutine will likely be interrupted
		// by a signal.
		sendSomeSignals()
		// This Write will likely be interrupted by a signal
		// as the other goroutine spins in the middle of reading.
		// We write enough data that we should always fill the
		// pipe buffer and need multiple write system calls.
		if _, err := w.Write(bytes.Repeat([]byte{0}, 2<<20)); err != nil {
			log.Fatal(err)
		}
	}()
	go func() {
		defer wg.Done()
		defer r.Close()
		b := make([]byte, 1<<20)
		// This ReadFull will likely be interrupted by a signal,
		// as the other goroutine spins before writing anything.
		if _, err := io.ReadFull(r, b); err != nil {
			log.Fatal(err)
		}
		// Spin after reading half the data so that the Write
		// in the other goroutine will likely be interrupted
		// before it completes.
		sendSomeSignals()
		if _, err := io.ReadFull(r, b); err != nil {
			log.Fatal(err)
		}
	}()
}

// testNet tests network operations.
func testNet(wg *sync.WaitGroup) {
	ln, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		if errors.Is(err, syscall.EAFNOSUPPORT) || errors.Is(err, syscall.EPROTONOSUPPORT) {
			return
		}
		log.Fatal(err)
	}
	wg.Add(2)
	go func() {
		defer wg.Done()
		defer ln.Close()
		c, err := ln.Accept()
		if err != nil {
			log.Fatal(err)
		}
		defer c.Close()
		cf, err := c.(*net.TCPConn).File()
		if err != nil {
			log.Fatal(err)
		}
		defer cf.Close()
		if err := syscall.SetNonblock(int(cf.Fd()), false); err != nil {
			log.Fatal(err)
		}
		// See comments in testPipe.
		sendSomeSignals()
		if _, err := cf.Write(bytes.Repeat([]byte{0}, 2<<20)); err != nil {
			log.Fatal(err)
		}
	}()
	go func() {
		defer wg.Done()
		sendSomeSignals()
		c, err := net.Dial("tcp", ln.Addr().String())
		if err != nil {
			log.Fatal(err)
		}
		defer c.Close()
		cf, err := c.(*net.TCPConn).File()
		if err != nil {
			log.Fatal(err)
		}
		defer cf.Close()
		if err := syscall.SetNonblock(int(cf.Fd()), false); err != nil {
			log.Fatal(err)
		}
		// See comments in testPipe.
		b := make([]byte, 1<<20)
		if _, err := io.ReadFull(cf, b); err != nil {
			log.Fatal(err)
		}
		sendSomeSignals()
		if _, err := io.ReadFull(cf, b); err != nil {
			log.Fatal(err)
		}
	}()
}

func testExec(wg *sync.WaitGroup) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		cmd := exec.Command(os.Args[0], "Block")
		stdin, err := cmd.StdinPipe()
		if err != nil {
			log.Fatal(err)
		}
		cmd.Stderr = new(bytes.Buffer)
		cmd.Stdout = cmd.Stderr
		if err := cmd.Start(); err != nil {
			log.Fatal(err)
		}

		go func() {
			sendSomeSignals()
			stdin.Close()
		}()

		if err := cmd.Wait(); err != nil {
			log.Fatalf("%v:\n%s", err, cmd.Stdout)
		}
	}()
}

// Block blocks until stdin is closed.
func Block() {
	io.Copy(ioutil.Discard, os.Stdin)
}
