// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build openbsd

// Stub cgo routines for systems that do not use cgo to do network lookups.

package net

import "os"

func cgoLookupHost(name string) (addrs []string, err os.Error, completed bool) {
	return nil, nil, false
}

func cgoLookupPort(network, service string) (port int, err os.Error, completed bool) {
	return 0, nil, false
}

func cgoLookupIP(name string) (addrs []IP, err os.Error, completed bool) {
	return nil, nil, false
}

func cgoLookupCNAME(name string) (cname string, err os.Error, completed bool) {
	return "", nil, false
}
