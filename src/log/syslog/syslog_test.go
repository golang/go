// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows && !plan9 && !js && !wasip1

package syslog

import (
	"bufio"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"
)

func runPktSyslog(c net.PacketConn, done chan<- string) {
	var buf [4096]byte
	var rcvd string
	ct := 0
	for {
		var n int
		var err error

		c.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
		n, _, err = c.ReadFrom(buf[:])
		rcvd += string(buf[:n])
		if err != nil {
			if oe, ok := err.(*net.OpError); ok {
				if ct < 3 && oe.Temporary() {
					ct++
					continue
				}
			}
			break
		}
	}
	c.Close()
	done <- rcvd
}

var crashy = false

func testableNetwork(network string) bool {
	switch network {
	case "unix", "unixgram":
		switch runtime.GOOS {
		case "ios", "android":
			return false
		}
	}
	return true
}

func runStreamSyslog(l net.Listener, done chan<- string, wg *sync.WaitGroup) {
	for {
		var c net.Conn
		var err error
		if c, err = l.Accept(); err != nil {
			return
		}
		wg.Add(1)
		go func(c net.Conn) {
			defer wg.Done()
			c.SetReadDeadline(time.Now().Add(5 * time.Second))
			b := bufio.NewReader(c)
			for ct := 1; !crashy || ct&7 != 0; ct++ {
				s, err := b.ReadString('\n')
				if err != nil {
					break
				}
				done <- s
			}
			c.Close()
		}(c)
	}
}

func startServer(t *testing.T, n, la string, done chan<- string) (addr string, sock io.Closer, wg *sync.WaitGroup) {
	if n == "udp" || n == "tcp" {
		la = "127.0.0.1:0"
	} else {
		// unix and unixgram: choose an address if none given.
		if la == "" {
			// The address must be short to fit in the sun_path field of the
			// sockaddr_un passed to the underlying system calls, so we use
			// os.MkdirTemp instead of t.TempDir: t.TempDir generally includes all or
			// part of the test name in the directory, which can be much more verbose
			// and risks running up against the limit.
			dir, err := os.MkdirTemp("", "")
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := os.RemoveAll(dir); err != nil {
					t.Errorf("failed to remove socket temp directory: %v", err)
				}
			})
			la = filepath.Join(dir, "sock")
		}
	}

	wg = new(sync.WaitGroup)
	if n == "udp" || n == "unixgram" {
		l, e := net.ListenPacket(n, la)
		if e != nil {
			t.Helper()
			t.Fatalf("startServer failed: %v", e)
		}
		addr = l.LocalAddr().String()
		sock = l
		wg.Add(1)
		go func() {
			defer wg.Done()
			runPktSyslog(l, done)
		}()
	} else {
		l, e := net.Listen(n, la)
		if e != nil {
			t.Helper()
			t.Fatalf("startServer failed: %v", e)
		}
		addr = l.Addr().String()
		sock = l
		wg.Add(1)
		go func() {
			defer wg.Done()
			runStreamSyslog(l, done, wg)
		}()
	}
	return
}

func TestWithSimulated(t *testing.T) {
	t.Parallel()

	msg := "Test 123"
	for _, tr := range []string{"unix", "unixgram", "udp", "tcp"} {
		if !testableNetwork(tr) {
			continue
		}

		tr := tr
		t.Run(tr, func(t *testing.T) {
			t.Parallel()

			done := make(chan string)
			addr, sock, srvWG := startServer(t, tr, "", done)
			defer srvWG.Wait()
			defer sock.Close()
			if tr == "unix" || tr == "unixgram" {
				defer os.Remove(addr)
			}
			s, err := Dial(tr, addr, LOG_INFO|LOG_USER, "syslog_test")
			if err != nil {
				t.Fatalf("Dial() failed: %v", err)
			}
			err = s.Info(msg)
			if err != nil {
				t.Fatalf("log failed: %v", err)
			}
			check(t, msg, <-done, tr)
			s.Close()
		})
	}
}

func TestFlap(t *testing.T) {
	net := "unix"
	if !testableNetwork(net) {
		t.Skipf("skipping on %s/%s; 'unix' is not supported", runtime.GOOS, runtime.GOARCH)
	}

	done := make(chan string)
	addr, sock, srvWG := startServer(t, net, "", done)
	defer srvWG.Wait()
	defer os.Remove(addr)
	defer sock.Close()

	s, err := Dial(net, addr, LOG_INFO|LOG_USER, "syslog_test")
	if err != nil {
		t.Fatalf("Dial() failed: %v", err)
	}
	msg := "Moo 2"
	err = s.Info(msg)
	if err != nil {
		t.Fatalf("log failed: %v", err)
	}
	check(t, msg, <-done, net)

	// restart the server
	if err := os.Remove(addr); err != nil {
		t.Fatal(err)
	}
	_, sock2, srvWG2 := startServer(t, net, addr, done)
	defer srvWG2.Wait()
	defer sock2.Close()

	// and try retransmitting
	msg = "Moo 3"
	err = s.Info(msg)
	if err != nil {
		t.Fatalf("log failed: %v", err)
	}
	check(t, msg, <-done, net)

	s.Close()
}

func TestNew(t *testing.T) {
	if LOG_LOCAL7 != 23<<3 {
		t.Fatalf("LOG_LOCAL7 has wrong value")
	}
	if testing.Short() {
		// Depends on syslog daemon running, and sometimes it's not.
		t.Skip("skipping syslog test during -short")
	}

	s, err := New(LOG_INFO|LOG_USER, "the_tag")
	if err != nil {
		if err.Error() == "Unix syslog delivery error" {
			t.Skip("skipping: syslogd not running")
		}
		t.Fatalf("New() failed: %s", err)
	}
	// Don't send any messages.
	s.Close()
}

func TestNewLogger(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping syslog test during -short")
	}
	f, err := NewLogger(LOG_USER|LOG_INFO, 0)
	if f == nil {
		if err.Error() == "Unix syslog delivery error" {
			t.Skip("skipping: syslogd not running")
		}
		t.Error(err)
	}
}

func TestDial(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping syslog test during -short")
	}
	f, err := Dial("", "", (LOG_LOCAL7|LOG_DEBUG)+1, "syslog_test")
	if f != nil {
		t.Fatalf("Should have trapped bad priority")
	}
	f, err = Dial("", "", -1, "syslog_test")
	if f != nil {
		t.Fatalf("Should have trapped bad priority")
	}
	l, err := Dial("", "", LOG_USER|LOG_ERR, "syslog_test")
	if err != nil {
		if err.Error() == "Unix syslog delivery error" {
			t.Skip("skipping: syslogd not running")
		}
		t.Fatalf("Dial() failed: %s", err)
	}
	l.Close()
}

func check(t *testing.T, in, out, transport string) {
	hostname, err := os.Hostname()
	if err != nil {
		t.Errorf("Error retrieving hostname: %v", err)
		return
	}

	if transport == "unixgram" || transport == "unix" {
		var month, date, ts string
		var pid int
		tmpl := fmt.Sprintf("<%d>%%s %%s %%s syslog_test[%%d]: %s\n", LOG_USER+LOG_INFO, in)
		n, err := fmt.Sscanf(out, tmpl, &month, &date, &ts, &pid)
		if n != 4 || err != nil {
			t.Errorf("Got %q, does not match template %q (%d %s)", out, tmpl, n, err)
		}
		return
	}

	// Non-UNIX domain transports.
	var parsedHostname, timestamp string
	var pid int
	tmpl := fmt.Sprintf("<%d>%%s %%s syslog_test[%%d]: %s\n", LOG_USER+LOG_INFO, in)
	n, err := fmt.Sscanf(out, tmpl, &timestamp, &parsedHostname, &pid)
	if n != 3 || err != nil {
		t.Errorf("Got %q, does not match template %q (%d %s)", out, tmpl, n, err)
	}
	if hostname != parsedHostname {
		t.Errorf("Hostname got %q want %q in %q", parsedHostname, hostname, out)
	}
}

func TestWrite(t *testing.T) {
	t.Parallel()

	tests := []struct {
		pri Priority
		pre string
		msg string
		exp string
	}{
		{LOG_USER | LOG_ERR, "syslog_test", "", "%s %s syslog_test[%d]: \n"},
		{LOG_USER | LOG_ERR, "syslog_test", "write test", "%s %s syslog_test[%d]: write test\n"},
		// Write should not add \n if there already is one
		{LOG_USER | LOG_ERR, "syslog_test", "write test 2\n", "%s %s syslog_test[%d]: write test 2\n"},
	}

	if hostname, err := os.Hostname(); err != nil {
		t.Fatalf("Error retrieving hostname")
	} else {
		for _, test := range tests {
			done := make(chan string)
			addr, sock, srvWG := startServer(t, "udp", "", done)
			defer srvWG.Wait()
			defer sock.Close()
			l, err := Dial("udp", addr, test.pri, test.pre)
			if err != nil {
				t.Fatalf("syslog.Dial() failed: %v", err)
			}
			defer l.Close()
			_, err = io.WriteString(l, test.msg)
			if err != nil {
				t.Fatalf("WriteString() failed: %v", err)
			}
			rcvd := <-done
			test.exp = fmt.Sprintf("<%d>", test.pri) + test.exp
			var parsedHostname, timestamp string
			var pid int
			if n, err := fmt.Sscanf(rcvd, test.exp, &timestamp, &parsedHostname, &pid); n != 3 || err != nil || hostname != parsedHostname {
				t.Errorf("s.Info() = '%q', didn't match '%q' (%d %s)", rcvd, test.exp, n, err)
			}
		}
	}
}

func TestConcurrentWrite(t *testing.T) {
	addr, sock, srvWG := startServer(t, "udp", "", make(chan string, 1))
	defer srvWG.Wait()
	defer sock.Close()
	w, err := Dial("udp", addr, LOG_USER|LOG_ERR, "how's it going?")
	if err != nil {
		t.Fatalf("syslog.Dial() failed: %v", err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := w.Info("test")
			if err != nil {
				t.Errorf("Info() failed: %v", err)
				return
			}
		}()
	}
	wg.Wait()
}

func TestConcurrentReconnect(t *testing.T) {
	crashy = true
	defer func() { crashy = false }()

	const N = 10
	const M = 100
	net := "unix"
	if !testableNetwork(net) {
		net = "tcp"
		if !testableNetwork(net) {
			t.Skipf("skipping on %s/%s; neither 'unix' or 'tcp' is supported", runtime.GOOS, runtime.GOARCH)
		}
	}
	done := make(chan string, N*M)
	addr, sock, srvWG := startServer(t, net, "", done)
	if net == "unix" {
		defer os.Remove(addr)
	}

	// count all the messages arriving
	count := make(chan int, 1)
	go func() {
		ct := 0
		for range done {
			ct++
			// we are looking for 500 out of 1000 events
			// here because lots of log messages are lost
			// in buffers (kernel and/or bufio)
			if ct > N*M/2 {
				break
			}
		}
		count <- ct
	}()

	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()
			w, err := Dial(net, addr, LOG_USER|LOG_ERR, "tag")
			if err != nil {
				t.Errorf("syslog.Dial() failed: %v", err)
				return
			}
			defer w.Close()
			for i := 0; i < M; i++ {
				err := w.Info("test")
				if err != nil {
					t.Errorf("Info() failed: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()
	sock.Close()
	srvWG.Wait()
	close(done)

	select {
	case <-count:
	case <-time.After(100 * time.Millisecond):
		t.Error("timeout in concurrent reconnect")
	}
}
