// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

// Read system DNS config from /etc/resolv.conf

package net

var defaultNS = []string{"127.0.0.1", "::1"}

type dnsConfig struct {
	servers    []string // servers to use
	search     []string // suffixes to append to local name
	ndots      int      // number of dots in name to trigger absolute lookup
	timeout    int      // seconds before giving up on packet
	attempts   int      // lost packets before giving up on server
	rotate     bool     // round robin among servers
	unknownOpt bool     // anything unknown was encountered
	lookup     []string // OpenBSD top-level database "lookup" order
	err        error    // any error that occurs during open of resolv.conf
}

// See resolv.conf(5) on a Linux machine.
// TODO(rsc): Supposed to call uname() and chop the beginning
// of the host name to get the default search domain.
func dnsReadConfig(filename string) *dnsConfig {
	conf := &dnsConfig{
		ndots:    1,
		timeout:  5,
		attempts: 2,
	}
	file, err := open(filename)
	if err != nil {
		conf.servers = defaultNS
		conf.err = err
		return conf
	}
	defer file.close()
	for line, ok := file.readLine(); ok; line, ok = file.readLine() {
		if len(line) > 0 && (line[0] == ';' || line[0] == '#') {
			// comment.
			continue
		}
		f := getFields(line)
		if len(f) < 1 {
			continue
		}
		switch f[0] {
		case "nameserver": // add one name server
			if len(f) > 1 && len(conf.servers) < 3 { // small, but the standard limit
				// One more check: make sure server name is
				// just an IP address.  Otherwise we need DNS
				// to look it up.
				if parseIPv4(f[1]) != nil {
					conf.servers = append(conf.servers, f[1])
				} else if ip, _ := parseIPv6(f[1], true); ip != nil {
					conf.servers = append(conf.servers, f[1])
				}
			}

		case "domain": // set search path to just this domain
			if len(f) > 1 {
				conf.search = []string{f[1]}
			}

		case "search": // set search path to given servers
			conf.search = make([]string, len(f)-1)
			for i := 0; i < len(conf.search); i++ {
				conf.search[i] = f[i+1]
			}

		case "options": // magic options
			for _, s := range f[1:] {
				switch {
				case hasPrefix(s, "ndots:"):
					n, _, _ := dtoi(s, 6)
					if n < 1 {
						n = 1
					}
					conf.ndots = n
				case hasPrefix(s, "timeout:"):
					n, _, _ := dtoi(s, 8)
					if n < 1 {
						n = 1
					}
					conf.timeout = n
				case hasPrefix(s, "attempts:"):
					n, _, _ := dtoi(s, 9)
					if n < 1 {
						n = 1
					}
					conf.attempts = n
				case s == "rotate":
					conf.rotate = true
				default:
					conf.unknownOpt = true
				}
			}

		case "lookup":
			// OpenBSD option:
			// http://www.openbsd.org/cgi-bin/man.cgi/OpenBSD-current/man5/resolv.conf.5
			// "the legal space-separated values are: bind, file, yp"
			conf.lookup = f[1:]

		default:
			conf.unknownOpt = true
		}
	}
	if len(conf.servers) == 0 {
		conf.servers = defaultNS
	}
	return conf
}

func hasPrefix(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
}
