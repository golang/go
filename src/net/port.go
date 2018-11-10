// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

// parsePort parses service as a decimal interger and returns the
// corresponding value as port. It is the caller's responsibility to
// parse service as a non-decimal integer when needsLookup is true.
//
// Some system resolvers will return a valid port number when given a number
// over 65536 (see https://github.com/golang/go/issues/11715). Alas, the parser
// can't bail early on numbers > 65536. Therefore reasonably large/small
// numbers are parsed in full and rejected if invalid.
func parsePort(service string) (port int, needsLookup bool) {
	if service == "" {
		// Lock in the legacy behavior that an empty string
		// means port 0. See golang.org/issue/13610.
		return 0, false
	}
	const (
		max    = uint32(1<<32 - 1)
		cutoff = uint32(1 << 30)
	)
	neg := false
	if service[0] == '+' {
		service = service[1:]
	} else if service[0] == '-' {
		neg = true
		service = service[1:]
	}
	var n uint32
	for _, d := range service {
		if '0' <= d && d <= '9' {
			d -= '0'
		} else {
			return 0, true
		}
		if n >= cutoff {
			n = max
			break
		}
		n *= 10
		nn := n + uint32(d)
		if nn < n || nn > max {
			n = max
			break
		}
		n = nn
	}
	if !neg && n >= cutoff {
		port = int(cutoff - 1)
	} else if neg && n > cutoff {
		port = int(cutoff)
	} else {
		port = int(n)
	}
	if neg {
		port = -port
	}
	return port, false
}
