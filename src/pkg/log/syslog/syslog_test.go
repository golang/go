// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows,!plan9

package syslog

import (
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"testing"
	"time"
)

var serverAddr string

func runSyslog(c net.PacketConn, done chan<- string) {
	var buf [4096]byte
	var rcvd string = ""
	for {
		n, _, err := c.ReadFrom(buf[0:])
		if err != nil || n == 0 {
			break
		}
		rcvd += string(buf[0:n])
	}
	done <- rcvd
}

func startServer(done chan<- string) {
	c, e := net.ListenPacket("udp", "127.0.0.1:0")
	if e != nil {
		log.Fatalf("net.ListenPacket failed udp :0 %v", e)
	}
	serverAddr = c.LocalAddr().String()
	c.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
	go runSyslog(c, done)
}

func skipNetTest(t *testing.T) bool {
	if testing.Short() {
		// Depends on syslog daemon running, and sometimes it's not.
		t.Logf("skipping syslog test during -short")
		return true
	}
	return false
}

func TestNew(t *testing.T) {
	if LOG_LOCAL7 != 23<<3 {
		t.Fatalf("LOG_LOCAL7 has wrong value")
	}
	if skipNetTest(t) {
		return
	}

	s, err := New(LOG_INFO|LOG_USER, "")
	if err != nil {
		t.Fatalf("New() failed: %s", err)
	}
	// Don't send any messages.
	s.Close()
}

func TestNewLogger(t *testing.T) {
	if skipNetTest(t) {
		return
	}
	f, err := NewLogger(LOG_USER|LOG_INFO, 0)
	if f == nil {
		t.Error(err)
	}
}

func TestDial(t *testing.T) {
	if skipNetTest(t) {
		return
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
		t.Fatalf("Dial() failed: %s", err)
	}
	l.Close()
}

func TestUDPDial(t *testing.T) {
	done := make(chan string)
	startServer(done)
	l, err := Dial("udp", serverAddr, LOG_USER|LOG_INFO, "syslog_test")
	if err != nil {
		t.Fatalf("syslog.Dial() failed: %s", err)
	}
	msg := "udp test"
	l.Info(msg)
	expected := fmt.Sprintf("<%d>1 ", LOG_USER+LOG_INFO) + "%s %s syslog_test[%d]: udp test\n"
	rcvd := <-done
	var parsedHostname, timestamp string
	var pid int
	if hostname, err := os.Hostname(); err != nil {
		t.Fatalf("Error retrieving hostname")
	} else {
		if n, err := fmt.Sscanf(rcvd, expected, &timestamp, &parsedHostname, &pid); n != 3 ||
			err != nil || hostname != parsedHostname {
			t.Fatalf("s.Info() = '%q', didn't match '%q'", rcvd, expected)
		}
	}
}

func TestWrite(t *testing.T) {
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
			startServer(done)
			l, err := Dial("udp", serverAddr, test.pri, test.pre)
			if err != nil {
				t.Fatalf("syslog.Dial() failed: %s", err)
			}
			_, err = io.WriteString(l, test.msg)
			if err != nil {
				t.Fatalf("WriteString() failed: %s", err)
			}
			rcvd := <-done
			test.exp = fmt.Sprintf("<%d>1 ", test.pri) + test.exp
			var parsedHostname, timestamp string
			var pid int
			if n, err := fmt.Sscanf(rcvd, test.exp, &timestamp, &parsedHostname, &pid); n != 3 ||
				err != nil || hostname != parsedHostname {
				t.Fatalf("s.Info() = '%q', didn't match '%q'", rcvd, test.exp)
			}
		}
	}
}
