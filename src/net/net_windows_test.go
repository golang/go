// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sort"
	"strings"
	"syscall"
	"testing"
	"time"
)

func toErrno(err error) (syscall.Errno, bool) {
	operr, ok := err.(*OpError)
	if !ok {
		return 0, false
	}
	syserr, ok := operr.Err.(*os.SyscallError)
	if !ok {
		return 0, false
	}
	errno, ok := syserr.Err.(syscall.Errno)
	if !ok {
		return 0, false
	}
	return errno, true
}

// TestAcceptIgnoreSomeErrors tests that windows TCPListener.AcceptTCP
// handles broken connections. It verifies that broken connections do
// not affect future connections.
func TestAcceptIgnoreSomeErrors(t *testing.T) {
	recv := func(ln Listener, ignoreSomeReadErrors bool) (string, error) {
		c, err := ln.Accept()
		if err != nil {
			// Display windows errno in error message.
			errno, ok := toErrno(err)
			if !ok {
				return "", err
			}
			return "", fmt.Errorf("%v (windows errno=%d)", err, errno)
		}
		defer c.Close()

		b := make([]byte, 100)
		n, err := c.Read(b)
		if err == nil || err == io.EOF {
			return string(b[:n]), nil
		}
		errno, ok := toErrno(err)
		if ok && ignoreSomeReadErrors && (errno == syscall.ERROR_NETNAME_DELETED || errno == syscall.WSAECONNRESET) {
			return "", nil
		}
		return "", err
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
			t.Fatal(err)
		}
		fmt.Printf("sleeping\n")
		time.Sleep(time.Minute) // process will be killed here
		c.Close()
	}

	ln, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
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
	s, err := recv(ln, true)
	if err != nil {
		t.Fatalf("recv failed: %v", err)
	}
	switch s {
	case "":
		// First connection data is received, let's get second connection data.
	case "abc":
		// First connection is lost forever, but that is ok.
		return
	default:
		t.Fatalf(`"%s" received from recv, but "" or "abc" expected`, s)
	}

	// Get second connection data.
	s, err = recv(ln, false)
	if err != nil {
		t.Fatalf("recv failed: %v", err)
	}
	if s != "abc" {
		t.Fatalf(`"%s" received from recv, but "abc" expected`, s)
	}
}

func isWindowsXP(t *testing.T) bool {
	v, err := syscall.GetVersion()
	if err != nil {
		t.Fatalf("GetVersion failed: %v", err)
	}
	major := byte(v)
	return major < 6
}

func listInterfacesWithNetsh() ([]string, error) {
	out, err := exec.Command("netsh", "interface", "ip", "show", "config").CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("netsh failed: %v: %q", err, string(out))
	}
	lines := bytes.Split(out, []byte{'\r', '\n'})
	names := make([]string, 0)
	for _, line := range lines {
		f := bytes.Split(line, []byte{'"'})
		if len(f) == 3 {
			names = append(names, string(f[1]))
		}
	}
	return names, nil
}

func TestInterfaceList(t *testing.T) {
	if isWindowsXP(t) {
		t.Skip("Windows XP netsh command does not provide required functionality")
	}
	ift, err := Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	have := make([]string, 0)
	for _, ifi := range ift {
		have = append(have, ifi.Name)
	}
	sort.Strings(have)

	want, err := listInterfacesWithNetsh()
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(want)

	if strings.Join(want, "/") != strings.Join(have, "/") {
		t.Fatalf("unexpected interface list %q, want %q", have, want)
	}
}
