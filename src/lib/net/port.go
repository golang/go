// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Read system port mappings from /etc/services

package net

import (
	"io";
	"net";
	"once";
	"os";
	"strconv";
)

var services map[string] map[string] int

func _ReadServices() {
	services = make(map[string] map[string] int);
	file := _Open("/etc/services");
	for line, ok := file.ReadLine(); ok; line, ok = file.ReadLine() {
		// "http 80/tcp www www-http # World Wide Web HTTP"
		if i := _ByteIndex(line, '#'); i >= 0 {
			line = line[0:i];
		}
		f := _GetFields(line);
		if len(f) < 2 {
			continue;
		}
		portnet := f[1];	// "tcp/80"
		port, j, ok := _Dtoi(portnet, 0);
		if !ok || port <= 0 || j >= len(portnet) || portnet[j] != '/' {
			continue
		}
		netw := portnet[j+1:len(portnet)];	// "tcp"
		m, ok1 := services[netw];
		if !ok1 {
			m = make(map[string] int);
			services[netw] = m;
		}
		for i := 0; i < len(f); i++ {
			if i != 1 {	// f[1] was port/net
				m[f[i]] = port;
			}
		}
	}
	file.Close();
}

export func LookupPort(netw, name string) (port int, ok bool) {
	once.Do(&_ReadServices);

	switch netw {
	case "tcp4", "tcp6":
		netw = "tcp";
	case "udp4", "udp6":
		netw = "udp";
	}

	m, mok := services[netw];
	if !mok {
		return
	}
	port, ok = m[name];
	return
}

