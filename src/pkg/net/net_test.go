// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"io"
	"io/ioutil"
	"os"
	"runtime"
	"testing"
	"time"
)

func TestShutdown(t *testing.T) {
	if runtime.GOOS == "plan9" {
		t.Skipf("skipping test on %q", runtime.GOOS)
	}
	ln, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		if ln, err = Listen("tcp6", "[::1]:0"); err != nil {
			t.Fatalf("ListenTCP on :0: %v", err)
		}
	}

	go func() {
		c, err := ln.Accept()
		if err != nil {
			t.Fatalf("Accept: %v", err)
		}
		var buf [10]byte
		n, err := c.Read(buf[:])
		if n != 0 || err != io.EOF {
			t.Fatalf("server Read = %d, %v; want 0, io.EOF", n, err)
		}
		c.Write([]byte("response"))
		c.Close()
	}()

	c, err := Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer c.Close()

	err = c.(*TCPConn).CloseWrite()
	if err != nil {
		t.Fatalf("CloseWrite: %v", err)
	}
	var buf [10]byte
	n, err := c.Read(buf[:])
	if err != nil {
		t.Fatalf("client Read: %d, %v", n, err)
	}
	got := string(buf[:n])
	if got != "response" {
		t.Errorf("read = %q, want \"response\"", got)
	}
}

func TestShutdownUnix(t *testing.T) {
	switch runtime.GOOS {
	case "windows", "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}
	f, err := ioutil.TempFile("", "go_net_unixtest")
	if err != nil {
		t.Fatalf("TempFile: %s", err)
	}
	f.Close()
	tmpname := f.Name()
	os.Remove(tmpname)
	ln, err := Listen("unix", tmpname)
	if err != nil {
		t.Fatalf("ListenUnix on %s: %s", tmpname, err)
	}
	defer os.Remove(tmpname)

	go func() {
		c, err := ln.Accept()
		if err != nil {
			t.Fatalf("Accept: %v", err)
		}
		var buf [10]byte
		n, err := c.Read(buf[:])
		if n != 0 || err != io.EOF {
			t.Fatalf("server Read = %d, %v; want 0, io.EOF", n, err)
		}
		c.Write([]byte("response"))
		c.Close()
	}()

	c, err := Dial("unix", tmpname)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer c.Close()

	err = c.(*UnixConn).CloseWrite()
	if err != nil {
		t.Fatalf("CloseWrite: %v", err)
	}
	var buf [10]byte
	n, err := c.Read(buf[:])
	if err != nil {
		t.Fatalf("client Read: %d, %v", n, err)
	}
	got := string(buf[:n])
	if got != "response" {
		t.Errorf("read = %q, want \"response\"", got)
	}
}

func TestTCPListenClose(t *testing.T) {
	ln, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen failed: %v", err)
	}

	done := make(chan bool, 1)
	go func() {
		time.Sleep(100 * time.Millisecond)
		ln.Close()
	}()
	go func() {
		c, err := ln.Accept()
		if err == nil {
			c.Close()
			t.Error("Accept succeeded")
		} else {
			t.Logf("Accept timeout error: %s (any error is fine)", err)
		}
		done <- true
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for TCP close")
	}
}

func TestUDPListenClose(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}
	ln, err := ListenPacket("udp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen failed: %v", err)
	}

	buf := make([]byte, 1000)
	done := make(chan bool, 1)
	go func() {
		time.Sleep(100 * time.Millisecond)
		ln.Close()
	}()
	go func() {
		_, _, err = ln.ReadFrom(buf)
		if err == nil {
			t.Error("ReadFrom succeeded")
		} else {
			t.Logf("ReadFrom timeout error: %s (any error is fine)", err)
		}
		done <- true
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for UDP close")
	}
}

func TestTCPClose(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}
	l, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()

	read := func(r io.Reader) error {
		var m [1]byte
		_, err := r.Read(m[:])
		return err
	}

	go func() {
		c, err := Dial("tcp", l.Addr().String())
		if err != nil {
			t.Fatal(err)
		}

		go read(c)

		time.Sleep(10 * time.Millisecond)
		c.Close()
	}()

	c, err := l.Accept()
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	for err == nil {
		err = read(c)
	}
	if err != nil && err != io.EOF {
		t.Fatal(err)
	}
}
