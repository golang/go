// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"syscall"
	"testing"
	"time"
)

func TestAcceptIgnoreSomeErrors(t *testing.T) {
	t.Skip("skipping temporarily, see issue 8662")

	recv := func(ln Listener) (string, error) {
		c, err := ln.Accept()
		if err != nil {
			// Display windows errno in error message.
			operr, ok := err.(*OpError)
			if !ok {
				return "", err
			}
			errno, ok := operr.Err.(syscall.Errno)
			if !ok {
				return "", err
			}
			return "", fmt.Errorf("%v (windows errno=%d)", err, errno)
		}
		defer c.Close()

		b := make([]byte, 100)
		n, err := c.Read(b)
		if err != nil && err != io.EOF {
			return "", err
		}
		return string(b[:n]), nil
	}

	send := func(addr string, data string) error {
		c, err := Dial("tcp", addr)
		if err != nil {
			return err
		}
		defer c.Close()

		b := []byte(data)
		n, err := c.Write(b)
		if err != nil {
			return err
		}
		if n != len(b) {
			return fmt.Errorf(`Only %d chars of string "%s" sent`, n, data)
		}
		return nil
	}

	if envaddr := os.Getenv("GOTEST_DIAL_ADDR"); envaddr != "" {
		// In child process.
		c, err := Dial("tcp", envaddr)
		if err != nil {
			t.Fatalf("Dial failed: %v", err)
		}
		fmt.Printf("sleeping\n")
		time.Sleep(time.Minute) // process will be killed here
		c.Close()
	}

	ln, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen failed: %v", err)
	}
	defer ln.Close()

	// Start child process that connects to our listener.
	cmd := exec.Command(os.Args[0], "-test.run=TestAcceptIgnoreSomeErrors")
	cmd.Env = append(os.Environ(), "GOTEST_DIAL_ADDR="+ln.Addr().String())
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("cmd.StdoutPipe failed: %v", err)
	}
	err = cmd.Start()
	if err != nil {
		t.Fatalf("cmd.Start failed: %v\n", err)
	}
	outReader := bufio.NewReader(stdout)
	for {
		s, err := outReader.ReadString('\n')
		if err != nil {
			t.Fatalf("reading stdout failed: %v", err)
		}
		if s == "sleeping\n" {
			break
		}
	}
	defer cmd.Wait() // ignore error - we know it is getting killed

	const alittle = 100 * time.Millisecond
	time.Sleep(alittle)
	cmd.Process.Kill() // the only way to trigger the errors
	time.Sleep(alittle)

	// Send second connection data (with delay in a separate goroutine).
	result := make(chan error)
	go func() {
		time.Sleep(alittle)
		err := send(ln.Addr().String(), "abc")
		if err != nil {
			result <- err
		}
		result <- nil
	}()
	defer func() {
		err := <-result
		if err != nil {
			t.Fatalf("send failed: %v", err)
		}
	}()

	// Receive first or second connection.
	s, err := recv(ln)
	if err != nil {
		t.Fatalf("recv failed: %v", err)
	}
	switch s {
	case "":
		// First connection data is received, lets get second connection data.
	case "abc":
		// First connection is lost forever, but that is ok.
		return
	default:
		t.Fatalf(`"%s" received from recv, but "" or "abc" expected`, s)
	}

	// Get second connection data.
	s, err = recv(ln)
	if err != nil {
		t.Fatalf("recv failed: %v", err)
	}
	if s != "abc" {
		t.Fatalf(`"%s" received from recv, but "abc" expected`, s)
	}
}
