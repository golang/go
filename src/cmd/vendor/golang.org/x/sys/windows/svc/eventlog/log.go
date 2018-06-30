// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

// Package eventlog implements access to Windows event log.
//
package eventlog

import (
	"errors"
	"syscall"

	"golang.org/x/sys/windows"
)

// Log provides access to the system log.
type Log struct {
	Handle windows.Handle
}

// Open retrieves a handle to the specified event log.
func Open(source string) (*Log, error) {
	return OpenRemote("", source)
}

// OpenRemote does the same as Open, but on different computer host.
func OpenRemote(host, source string) (*Log, error) {
	if source == "" {
		return nil, errors.New("Specify event log source")
	}
	var s *uint16
	if host != "" {
		s = syscall.StringToUTF16Ptr(host)
	}
	h, err := windows.RegisterEventSource(s, syscall.StringToUTF16Ptr(source))
	if err != nil {
		return nil, err
	}
	return &Log{Handle: h}, nil
}

// Close closes event log l.
func (l *Log) Close() error {
	return windows.DeregisterEventSource(l.Handle)
}

func (l *Log) report(etype uint16, eid uint32, msg string) error {
	ss := []*uint16{syscall.StringToUTF16Ptr(msg)}
	return windows.ReportEvent(l.Handle, etype, 0, eid, 0, 1, 0, &ss[0], nil)
}

// Info writes an information event msg with event id eid to the end of event log l.
// When EventCreate.exe is used, eid must be between 1 and 1000.
func (l *Log) Info(eid uint32, msg string) error {
	return l.report(windows.EVENTLOG_INFORMATION_TYPE, eid, msg)
}

// Warning writes an warning event msg with event id eid to the end of event log l.
// When EventCreate.exe is used, eid must be between 1 and 1000.
func (l *Log) Warning(eid uint32, msg string) error {
	return l.report(windows.EVENTLOG_WARNING_TYPE, eid, msg)
}

// Error writes an error event msg with event id eid to the end of event log l.
// When EventCreate.exe is used, eid must be between 1 and 1000.
func (l *Log) Error(eid uint32, msg string) error {
	return l.report(windows.EVENTLOG_ERROR_TYPE, eid, msg)
}
