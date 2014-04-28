// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows,!nacl,!plan9

// Package syslog provides a simple interface to the system log
// service. It can send messages to the syslog daemon using UNIX
// domain sockets, UDP or TCP.
//
// Only one call to Dial is necessary. On write failures,
// the syslog client will attempt to reconnect to the server
// and write again.
package syslog

import (
	"errors"
	"fmt"
	"log"
	"net"
	"os"
	"strings"
	"sync"
	"time"
)

// The Priority is a combination of the syslog facility and
// severity. For example, LOG_ALERT | LOG_FTP sends an alert severity
// message from the FTP facility. The default severity is LOG_EMERG;
// the default facility is LOG_KERN.
type Priority int

const severityMask = 0x07
const facilityMask = 0xf8

const (
	// Severity.

	// From /usr/include/sys/syslog.h.
	// These are the same on Linux, BSD, and OS X.
	LOG_EMERG Priority = iota
	LOG_ALERT
	LOG_CRIT
	LOG_ERR
	LOG_WARNING
	LOG_NOTICE
	LOG_INFO
	LOG_DEBUG
)

const (
	// Facility.

	// From /usr/include/sys/syslog.h.
	// These are the same up to LOG_FTP on Linux, BSD, and OS X.
	LOG_KERN Priority = iota << 3
	LOG_USER
	LOG_MAIL
	LOG_DAEMON
	LOG_AUTH
	LOG_SYSLOG
	LOG_LPR
	LOG_NEWS
	LOG_UUCP
	LOG_CRON
	LOG_AUTHPRIV
	LOG_FTP
	_ // unused
	_ // unused
	_ // unused
	_ // unused
	LOG_LOCAL0
	LOG_LOCAL1
	LOG_LOCAL2
	LOG_LOCAL3
	LOG_LOCAL4
	LOG_LOCAL5
	LOG_LOCAL6
	LOG_LOCAL7
)

// A Writer is a connection to a syslog server.
type Writer struct {
	priority Priority
	tag      string
	hostname string
	network  string
	raddr    string

	mu   sync.Mutex // guards conn
	conn serverConn
}

// This interface and the separate syslog_unix.go file exist for
// Solaris support as implemented by gccgo.  On Solaris you can not
// simply open a TCP connection to the syslog daemon.  The gccgo
// sources have a syslog_solaris.go file that implements unixSyslog to
// return a type that satisfies this interface and simply calls the C
// library syslog function.
type serverConn interface {
	writeString(p Priority, hostname, tag, s, nl string) error
	close() error
}

type netConn struct {
	local bool
	conn  net.Conn
}

// New establishes a new connection to the system log daemon.  Each
// write to the returned writer sends a log message with the given
// priority and prefix.
func New(priority Priority, tag string) (w *Writer, err error) {
	return Dial("", "", priority, tag)
}

// Dial establishes a connection to a log daemon by connecting to
// address raddr on the specified network.  Each write to the returned
// writer sends a log message with the given facility, severity and
// tag.
// If network is empty, Dial will connect to the local syslog server.
func Dial(network, raddr string, priority Priority, tag string) (*Writer, error) {
	if priority < 0 || priority > LOG_LOCAL7|LOG_DEBUG {
		return nil, errors.New("log/syslog: invalid priority")
	}

	if tag == "" {
		tag = os.Args[0]
	}
	hostname, _ := os.Hostname()

	w := &Writer{
		priority: priority,
		tag:      tag,
		hostname: hostname,
		network:  network,
		raddr:    raddr,
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	err := w.connect()
	if err != nil {
		return nil, err
	}
	return w, err
}

// connect makes a connection to the syslog server.
// It must be called with w.mu held.
func (w *Writer) connect() (err error) {
	if w.conn != nil {
		// ignore err from close, it makes sense to continue anyway
		w.conn.close()
		w.conn = nil
	}

	if w.network == "" {
		w.conn, err = unixSyslog()
		if w.hostname == "" {
			w.hostname = "localhost"
		}
	} else {
		var c net.Conn
		c, err = net.Dial(w.network, w.raddr)
		if err == nil {
			w.conn = &netConn{conn: c}
			if w.hostname == "" {
				w.hostname = c.LocalAddr().String()
			}
		}
	}
	return
}

// Write sends a log message to the syslog daemon.
func (w *Writer) Write(b []byte) (int, error) {
	return w.writeAndRetry(w.priority, string(b))
}

// Close closes a connection to the syslog daemon.
func (w *Writer) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.conn != nil {
		err := w.conn.close()
		w.conn = nil
		return err
	}
	return nil
}

// Emerg logs a message with severity LOG_EMERG, ignoring the severity
// passed to New.
func (w *Writer) Emerg(m string) (err error) {
	_, err = w.writeAndRetry(LOG_EMERG, m)
	return err
}

// Alert logs a message with severity LOG_ALERT, ignoring the severity
// passed to New.
func (w *Writer) Alert(m string) (err error) {
	_, err = w.writeAndRetry(LOG_ALERT, m)
	return err
}

// Crit logs a message with severity LOG_CRIT, ignoring the severity
// passed to New.
func (w *Writer) Crit(m string) (err error) {
	_, err = w.writeAndRetry(LOG_CRIT, m)
	return err
}

// Err logs a message with severity LOG_ERR, ignoring the severity
// passed to New.
func (w *Writer) Err(m string) (err error) {
	_, err = w.writeAndRetry(LOG_ERR, m)
	return err
}

// Warning logs a message with severity LOG_WARNING, ignoring the
// severity passed to New.
func (w *Writer) Warning(m string) (err error) {
	_, err = w.writeAndRetry(LOG_WARNING, m)
	return err
}

// Notice logs a message with severity LOG_NOTICE, ignoring the
// severity passed to New.
func (w *Writer) Notice(m string) (err error) {
	_, err = w.writeAndRetry(LOG_NOTICE, m)
	return err
}

// Info logs a message with severity LOG_INFO, ignoring the severity
// passed to New.
func (w *Writer) Info(m string) (err error) {
	_, err = w.writeAndRetry(LOG_INFO, m)
	return err
}

// Debug logs a message with severity LOG_DEBUG, ignoring the severity
// passed to New.
func (w *Writer) Debug(m string) (err error) {
	_, err = w.writeAndRetry(LOG_DEBUG, m)
	return err
}

func (w *Writer) writeAndRetry(p Priority, s string) (int, error) {
	pr := (w.priority & facilityMask) | (p & severityMask)

	w.mu.Lock()
	defer w.mu.Unlock()

	if w.conn != nil {
		if n, err := w.write(pr, s); err == nil {
			return n, err
		}
	}
	if err := w.connect(); err != nil {
		return 0, err
	}
	return w.write(pr, s)
}

// write generates and writes a syslog formatted string. The
// format is as follows: <PRI>TIMESTAMP HOSTNAME TAG[PID]: MSG
func (w *Writer) write(p Priority, msg string) (int, error) {
	// ensure it ends in a \n
	nl := ""
	if !strings.HasSuffix(msg, "\n") {
		nl = "\n"
	}

	err := w.conn.writeString(p, w.hostname, w.tag, msg, nl)
	if err != nil {
		return 0, err
	}
	// Note: return the length of the input, not the number of
	// bytes printed by Fprintf, because this must behave like
	// an io.Writer.
	return len(msg), nil
}

func (n *netConn) writeString(p Priority, hostname, tag, msg, nl string) error {
	if n.local {
		// Compared to the network form below, the changes are:
		//	1. Use time.Stamp instead of time.RFC3339.
		//	2. Drop the hostname field from the Fprintf.
		timestamp := time.Now().Format(time.Stamp)
		_, err := fmt.Fprintf(n.conn, "<%d>%s %s[%d]: %s%s",
			p, timestamp,
			tag, os.Getpid(), msg, nl)
		return err
	}
	timestamp := time.Now().Format(time.RFC3339)
	_, err := fmt.Fprintf(n.conn, "<%d>%s %s %s[%d]: %s%s",
		p, timestamp, hostname,
		tag, os.Getpid(), msg, nl)
	return err
}

func (n *netConn) close() error {
	return n.conn.Close()
}

// NewLogger creates a log.Logger whose output is written to
// the system log service with the specified priority. The logFlag
// argument is the flag set passed through to log.New to create
// the Logger.
func NewLogger(p Priority, logFlag int) (*log.Logger, error) {
	s, err := New(p, "")
	if err != nil {
		return nil, err
	}
	return log.New(s, "", logFlag), nil
}
