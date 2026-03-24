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
//
// The syslog package is frozen and is not accepting new features.
// Some external packages provide more functionality. See:
//
//	https://godoc.org/?q=syslog
package syslog

// BUG(brainman): This package is not implemented on Windows. As the
// syslog package is frozen, Windows users are encouraged to
// use a package outside of the standard library. For background,
// see https://golang.org/issue/1108.

// BUG(akumar): This package is not implemented on Plan 9.

// BUG(odeke-em): As of Darwin's macOS 12.X (Monterey), Apple's syslog daemon no
// longer listens on a UNIX domain socket and instead now uses their proprietary
// Apple Unified Logging (AUL) system, hence this package will no longer work.
// Please see https://developer.apple.com/documentation/os/logging.
