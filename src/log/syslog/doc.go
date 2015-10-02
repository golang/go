// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package syslog provides a simple interface to the system log
// service. It can send messages to the syslog daemon using UNIX
// domain sockets, UDP or TCP.
//
// Only one call to Dial is necessary. On write failures,
// the syslog client will attempt to reconnect to the server
// and write again.
package syslog

// BUG(brainman): This package is not implemented on Windows yet.

// BUG(akumar): This package is not implemented on Plan 9 yet.

// BUG(minux): This package is not implemented on NaCl (Native Client) yet.
