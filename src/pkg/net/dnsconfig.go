// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Read system DNS config from /etc/resolv.conf

package net

import "os"

type dnsConfig struct {
	servers  []string // servers to use
	search   []string // suffixes to append to local name
	ndots    int      // number of dots in name to trigger absolute lookup
	timeout  int      // seconds before giving up on packet
	attempts int      // lost packets before giving up on server
	rotate   bool     // round robin among servers
}

var dnsconfigError os.Error

type DNSConfigError struct {
	Error os.Error
}

func (e *DNSConfigError) String() string {
	return "error reading DNS config: " + e.Error.String()
}

func (e *DNSConfigError) Timeout() bool   { return false }
func (e *DNSConfigError) Temporary() bool { return false }


// See resolv.conf(5) on a Linux machine.
// TODO(rsc): Supposed to call uname() and chop the beginning
// of the host name to get the default search domain.
// We assume it's in resolv.conf anyway.
func dnsReadConfig() (*dnsConfig, os.Error) {
	file, err := open("/etc/resolv.conf")
	if err != nil {
		return nil, &DNSConfigError{err}
	}
	conf := new(dnsConfig)
	conf.servers = make([]string, 3)[0:0] // small, but the standard limit
	conf.search = make([]string, 0)
	conf.ndots = 1
	conf.timeout = 5
	conf.attempts = 2
	conf.rotate = false
	for line, ok := file.readLine(); ok; line, ok = file.readLine() {
		f := getFields(line)
		if len(f) < 1 {
			continue
		}
		switch f[0] {
		case "nameserver": // add one name server
			a := conf.servers
			n := len(a)
			if len(f) > 1 && n < cap(a) {
				// One more check: make sure server name is
				// just an IP address.  Otherwise we need DNS
				// to look it up.
				name := f[1]
				switch len(ParseIP(name)) {
				case 16:
					name = "[" + name + "]"
					fallthrough
				case 4:
					a = a[0 : n+1]
					a[n] = name
					conf.servers = a
				}
			}

		case "domain": // set search path to just this domain
			if len(f) > 1 {
				conf.search = make([]string, 1)
				conf.search[0] = f[1]
			} else {
				conf.search = make([]string, 0)
			}

		case "search": // set search path to given servers
			conf.search = make([]string, len(f)-1)
			for i := 0; i < len(conf.search); i++ {
				conf.search[i] = f[i+1]
			}

		case "options": // magic options
			for i := 1; i < len(f); i++ {
				s := f[i]
				switch {
				case len(s) >= 6 && s[0:6] == "ndots:":
					n, _, _ := dtoi(s, 6)
					if n < 1 {
						n = 1
					}
					conf.ndots = n
				case len(s) >= 8 && s[0:8] == "timeout:":
					n, _, _ := dtoi(s, 8)
					if n < 1 {
						n = 1
					}
					conf.timeout = n
				case len(s) >= 8 && s[0:9] == "attempts:":
					n, _, _ := dtoi(s, 9)
					if n < 1 {
						n = 1
					}
					conf.attempts = n
				case s == "rotate":
					conf.rotate = true
				}
			}
		}
	}
	file.close()

	return conf, nil
}
