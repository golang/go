// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Network service port manipulations

package net

// parsePort parses port as a network service port number for both
// TCP and UDP.
func parsePort(net, port string) (int, error) {
	p, i, ok := dtoi(port, 0)
	if !ok || i != len(port) {
		var err error
		p, err = LookupPort(net, port)
		if err != nil {
			return 0, err
		}
	}
	if p < 0 || p > 0xFFFF {
		return 0, &AddrError{"invalid port", port}
	}
	return p, nil
}
