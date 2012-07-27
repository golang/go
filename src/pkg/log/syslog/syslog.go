// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows,!plan9

// Package syslog provides a simple interface to the system log service. It
// can send messages to the syslog daemon using UNIX domain sockets, UDP, or
// TCP connections.
package syslog

import (
	"errors"
	"fmt"
	"log"
	"net"
	"os"
)

type Priority int

const (
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

// A Writer is a connection to a syslog server.
type Writer struct {
	priority Priority
	prefix   string
	conn     serverConn
}

type serverConn interface {
	writeBytes(p Priority, prefix string, b []byte) (int, error)
	writeString(p Priority, prefix string, s string) (int, error)
	close() error
}

type netConn struct {
	conn net.Conn
}

// New establishes a new connection to the system log daemon.
// Each write to the returned writer sends a log message with
// the given priority and prefix.
func New(priority Priority, prefix string) (w *Writer, err error) {
	return Dial("", "", priority, prefix)
}

// Dial establishes a connection to a log daemon by connecting
// to address raddr on the network net.
// Each write to the returned writer sends a log message with
// the given priority and prefix.
func Dial(network, raddr string, priority Priority, prefix string) (w *Writer, err error) {
	if prefix == "" {
		prefix = os.Args[0]
	}
	var conn serverConn
	if network == "" {
		conn, err = unixSyslog()
	} else {
		var c net.Conn
		c, err = net.Dial(network, raddr)
		conn = netConn{c}
	}
	return &Writer{priority, prefix, conn}, err
}

// Write sends a log message to the syslog daemon.
func (w *Writer) Write(b []byte) (int, error) {
	if w.priority > LOG_DEBUG || w.priority < LOG_EMERG {
		return 0, errors.New("log/syslog: invalid priority")
	}
	return w.conn.writeBytes(w.priority, w.prefix, b)
}

func (w *Writer) writeString(p Priority, s string) (int, error) {
	return w.conn.writeString(p, w.prefix, s)
}

func (w *Writer) Close() error { return w.conn.close() }

// Emerg logs a message using the LOG_EMERG priority.
func (w *Writer) Emerg(m string) (err error) {
	_, err = w.writeString(LOG_EMERG, m)
	return err
}

// Alert logs a message using the LOG_ALERT priority.
func (w *Writer) Alert(m string) (err error) {
	_, err = w.writeString(LOG_ALERT, m)
	return err
}

// Crit logs a message using the LOG_CRIT priority.
func (w *Writer) Crit(m string) (err error) {
	_, err = w.writeString(LOG_CRIT, m)
	return err
}

// Err logs a message using the LOG_ERR priority.
func (w *Writer) Err(m string) (err error) {
	_, err = w.writeString(LOG_ERR, m)
	return err
}

// Warning logs a message using the LOG_WARNING priority.
func (w *Writer) Warning(m string) (err error) {
	_, err = w.writeString(LOG_WARNING, m)
	return err
}

// Notice logs a message using the LOG_NOTICE priority.
func (w *Writer) Notice(m string) (err error) {
	_, err = w.writeString(LOG_NOTICE, m)
	return err
}

// Info logs a message using the LOG_INFO priority.
func (w *Writer) Info(m string) (err error) {
	_, err = w.writeString(LOG_INFO, m)
	return err
}

// Debug logs a message using the LOG_DEBUG priority.
func (w *Writer) Debug(m string) (err error) {
	_, err = w.writeString(LOG_DEBUG, m)
	return err
}

func (n netConn) writeBytes(p Priority, prefix string, b []byte) (int, error) {
	nl := ""
	if len(b) == 0 || b[len(b)-1] != '\n' {
		nl = "\n"
	}
	_, err := fmt.Fprintf(n.conn, "<%d>%s: %s%s", p, prefix, b, nl)
	if err != nil {
		return 0, err
	}
	return len(b), nil
}

func (n netConn) writeString(p Priority, prefix string, s string) (int, error) {
	nl := ""
	if len(s) == 0 || s[len(s)-1] != '\n' {
		nl = "\n"
	}
	_, err := fmt.Fprintf(n.conn, "<%d>%s: %s%s", p, prefix, s, nl)
	if err != nil {
		return 0, err
	}
	return len(s), nil
}

func (n netConn) close() error {
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
