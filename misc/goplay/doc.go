// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Goplay is a web interface for experimenting with Go code.
// It is similar to the Go Playground: http://golang.org/doc/play/
//
// To use goplay:
//   $ cd $GOROOT/misc/goplay
//   $ go run goplay.go
// and load http://localhost:3999/ in a web browser.
//
// You should see a Hello World program, which you can compile and run by
// pressing shift-enter. There is also a "compile-on-keypress" feature that can
// be enabled by checking a checkbox.
//
// WARNING! CUIDADO! ACHTUNG! ATTENZIONE!
// A note on security: anyone with access to the goplay web interface can run
// arbitrary code on your computer. Goplay is not a sandbox, and has no other
// security mechanisms. Do not deploy it in untrusted environments.
// By default, goplay listens only on localhost. This can be overridden with
// the -http parameter. Do so at your own risk.
package main
