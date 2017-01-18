// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package web defines helper routines for accessing HTTP/HTTPS resources.
package web

// SecurityMode specifies whether a function should make network
// calls using insecure transports (eg, plain text HTTP).
// The zero value is "secure".
type SecurityMode int

const (
	Secure SecurityMode = iota
	Insecure
)
