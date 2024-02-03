// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows

// Read system DNS config from /etc/resolv.conf

package net

import (
	"internal/bytealg"
	"net/netip"
	"time"
)

// See resolv.conf(5) on a Linux machine.
func dnsReadConfig(filename string) *dnsConfig {
	conf := &dnsConfig{
		ndots:    1,
		timeout:  5 * time.Second,
		attempts: 2,
	}
	file, err := open(filename)
	if err != nil {
		conf.servers = defaultNS
		conf.search = dnsDefaultSearch()
		conf.err = err
		return conf
	}
	defer file.close()
	if fi, err := file.file.Stat(); err == nil {
		conf.mtime = fi.ModTime()
	} else {
		conf.servers = defaultNS
		conf.search = dnsDefaultSearch()
		conf.err = err
		return conf
	}
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
				// just an IP address. Otherwise we need DNS
				// to look it up.
				if _, err := netip.ParseAddr(f[1]); err == nil {
					conf.servers = append(conf.servers, JoinHostPort(f[1], "53"))
				}
			}

		case "domain": // set search path to just this domain
			if len(f) > 1 {
				conf.search = []string{ensureRooted(f[1])}
			}

		case "search": // set search path to given servers
			conf.search = make([]string, 0, len(f)-1)
			for i := 1; i < len(f); i++ {
				name := ensureRooted(f[i])
				if name == "." {
					continue
				}
				conf.search = append(conf.search, name)
			}

		case "options": // magic options
			for _, s := range f[1:] {
				switch {
				case hasPrefix(s, "ndots:"):
					n, _, _ := dtoi(s[6:])
					if n < 0 {
						n = 0
					} else if n > 15 {
						n = 15
					}
					conf.ndots = n
				case hasPrefix(s, "timeout:"):
					n, _, _ := dtoi(s[8:])
					if n < 1 {
						n = 1
					}
					conf.timeout = time.Duration(n) * time.Second
				case hasPrefix(s, "attempts:"):
					n, _, _ := dtoi(s[9:])
					if n < 1 {
						n = 1
					}
					conf.attempts = n
				case s == "rotate":
					conf.rotate = true
				case s == "single-request" || s == "single-request-reopen":
					// Linux option:
					// http://man7.org/linux/man-pages/man5/resolv.conf.5.html
					// "By default, glibc performs IPv4 and IPv6 lookups in parallel [...]
					//  This option disables the behavior and makes glibc
					//  perform the IPv6 and IPv4 requests sequentially."
					conf.singleRequest = true
				case s == "use-vc" || s == "usevc" || s == "tcp":
					// Linux (use-vc), FreeBSD (usevc) and OpenBSD (tcp) option:
					// http://man7.org/linux/man-pages/man5/resolv.conf.5.html
					// "Sets RES_USEVC in _res.options.
					//  This option forces the use of TCP for DNS resolutions."
					// https://www.freebsd.org/cgi/man.cgi?query=resolv.conf&sektion=5&manpath=freebsd-release-ports
					// https://man.openbsd.org/resolv.conf.5
					conf.useTCP = true
				case s == "trust-ad":
					conf.trustAD = true
				case s == "edns0":
					// We use EDNS by default.
					// Ignore this option.
				case s == "no-reload":
					conf.noReload = true
				default:
					conf.unknownOpt = true
				}
			}

		case "lookup":
			// OpenBSD option:
			// https://www.openbsd.org/cgi-bin/man.cgi/OpenBSD-current/man5/resolv.conf.5
			// "the legal space-separated values are: bind, file, yp"
			conf.lookup = f[1:]

		default:
			conf.unknownOpt = true
		}
	}
	if len(conf.servers) == 0 {
		conf.servers = defaultNS
	}
	if len(conf.search) == 0 {
		conf.search = dnsDefaultSearch()
	}
	return conf
}

func dnsDefaultSearch() []string {
	hn, err := getHostname()
	if err != nil {
		// best effort
		return nil
	}
	if i := bytealg.IndexByteString(hn, '.'); i >= 0 && i < len(hn)-1 {
		return []string{ensureRooted(hn[i+1:])}
	}
	return nil
}

func hasPrefix(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
}

func ensureRooted(s string) string {
	if len(s) > 0 && s[len(s)-1] == '.' {
		return s
	}
	return s + "."
}
