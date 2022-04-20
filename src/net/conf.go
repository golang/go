// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package net

import (
	"internal/bytealg"
	"internal/godebug"
	"os"
	"runtime"
	"sync"
	"syscall"
)

// conf represents a system's network configuration.
type conf struct {
	// forceCgoLookupHost forces CGO to always be used, if available.
	forceCgoLookupHost bool

	netGo  bool // go DNS resolution forced
	netCgo bool // cgo DNS resolution forced

	// machine has an /etc/mdns.allow file
	hasMDNSAllow bool

	goos          string // the runtime.GOOS, to ease testing
	dnsDebugLevel int

	nss    *nssConf
	resolv *dnsConfig
}

var (
	confOnce sync.Once // guards init of confVal via initConfVal
	confVal  = &conf{goos: runtime.GOOS}
)

// systemConf returns the machine's network configuration.
func systemConf() *conf {
	confOnce.Do(initConfVal)
	return confVal
}

func initConfVal() {
	dnsMode, debugLevel := goDebugNetDNS()
	confVal.dnsDebugLevel = debugLevel
	confVal.netGo = netGo || dnsMode == "go"
	confVal.netCgo = netCgo || dnsMode == "cgo"

	if confVal.dnsDebugLevel > 0 {
		defer func() {
			switch {
			case confVal.netGo:
				if netGo {
					println("go package net: built with netgo build tag; using Go's DNS resolver")
				} else {
					println("go package net: GODEBUG setting forcing use of Go's resolver")
				}
			case confVal.forceCgoLookupHost:
				println("go package net: using cgo DNS resolver")
			default:
				println("go package net: dynamic selection of DNS resolver")
			}
		}()
	}

	// Darwin pops up annoying dialog boxes if programs try to do
	// their own DNS requests. So always use cgo instead, which
	// avoids that.
	if runtime.GOOS == "darwin" || runtime.GOOS == "ios" {
		confVal.forceCgoLookupHost = true
		return
	}

	// If any environment-specified resolver options are specified,
	// force cgo. Note that LOCALDOMAIN can change behavior merely
	// by being specified with the empty string.
	_, localDomainDefined := syscall.Getenv("LOCALDOMAIN")
	if os.Getenv("RES_OPTIONS") != "" ||
		os.Getenv("HOSTALIASES") != "" ||
		confVal.netCgo ||
		localDomainDefined {
		confVal.forceCgoLookupHost = true
		return
	}

	// OpenBSD apparently lets you override the location of resolv.conf
	// with ASR_CONFIG. If we notice that, defer to libc.
	if runtime.GOOS == "openbsd" && os.Getenv("ASR_CONFIG") != "" {
		confVal.forceCgoLookupHost = true
		return
	}

	if runtime.GOOS != "openbsd" {
		confVal.nss = parseNSSConfFile("/etc/nsswitch.conf")
	}

	confVal.resolv = dnsReadConfig("/etc/resolv.conf")
	if confVal.resolv.err != nil && !os.IsNotExist(confVal.resolv.err) &&
		!os.IsPermission(confVal.resolv.err) {
		// If we can't read the resolv.conf file, assume it
		// had something important in it and defer to cgo.
		// libc's resolver might then fail too, but at least
		// it wasn't our fault.
		confVal.forceCgoLookupHost = true
	}

	if _, err := os.Stat("/etc/mdns.allow"); err == nil {
		confVal.hasMDNSAllow = true
	}
}

// canUseCgo reports whether calling cgo functions is allowed
// for non-hostname lookups.
func (c *conf) canUseCgo() bool {
	return c.hostLookupOrder(nil, "") == hostLookupCgo
}

// hostLookupOrder determines which strategy to use to resolve hostname.
// The provided Resolver is optional. nil means to not consider its options.
func (c *conf) hostLookupOrder(r *Resolver, hostname string) (ret hostLookupOrder) {
	if c.dnsDebugLevel > 1 {
		defer func() {
			print("go package net: hostLookupOrder(", hostname, ") = ", ret.String(), "\n")
		}()
	}
	fallbackOrder := hostLookupCgo
	if c.netGo || r.preferGo() {
		fallbackOrder = hostLookupFilesDNS
	}
	if c.forceCgoLookupHost || c.resolv.unknownOpt || c.goos == "android" {
		return fallbackOrder
	}
	if bytealg.IndexByteString(hostname, '\\') != -1 || bytealg.IndexByteString(hostname, '%') != -1 {
		// Don't deal with special form hostnames with backslashes
		// or '%'.
		return fallbackOrder
	}

	// OpenBSD is unique and doesn't use nsswitch.conf.
	// It also doesn't support mDNS.
	if c.goos == "openbsd" {
		// OpenBSD's resolv.conf manpage says that a non-existent
		// resolv.conf means "lookup" defaults to only "files",
		// without DNS lookups.
		if os.IsNotExist(c.resolv.err) {
			return hostLookupFiles
		}
		lookup := c.resolv.lookup
		if len(lookup) == 0 {
			// https://www.openbsd.org/cgi-bin/man.cgi/OpenBSD-current/man5/resolv.conf.5
			// "If the lookup keyword is not used in the
			// system's resolv.conf file then the assumed
			// order is 'bind file'"
			return hostLookupDNSFiles
		}
		if len(lookup) < 1 || len(lookup) > 2 {
			return fallbackOrder
		}
		switch lookup[0] {
		case "bind":
			if len(lookup) == 2 {
				if lookup[1] == "file" {
					return hostLookupDNSFiles
				}
				return fallbackOrder
			}
			return hostLookupDNS
		case "file":
			if len(lookup) == 2 {
				if lookup[1] == "bind" {
					return hostLookupFilesDNS
				}
				return fallbackOrder
			}
			return hostLookupFiles
		default:
			return fallbackOrder
		}
	}

	// Canonicalize the hostname by removing any trailing dot.
	if stringsHasSuffix(hostname, ".") {
		hostname = hostname[:len(hostname)-1]
	}
	if stringsHasSuffixFold(hostname, ".local") {
		// Per RFC 6762, the ".local" TLD is special. And
		// because Go's native resolver doesn't do mDNS or
		// similar local resolution mechanisms, assume that
		// libc might (via Avahi, etc) and use cgo.
		return fallbackOrder
	}

	nss := c.nss
	srcs := nss.sources["hosts"]
	// If /etc/nsswitch.conf doesn't exist or doesn't specify any
	// sources for "hosts", assume Go's DNS will work fine.
	if os.IsNotExist(nss.err) || (nss.err == nil && len(srcs) == 0) {
		if c.goos == "solaris" {
			// illumos defaults to "nis [NOTFOUND=return] files"
			return fallbackOrder
		}
		return hostLookupFilesDNS
	}
	if nss.err != nil {
		// We failed to parse or open nsswitch.conf, so
		// conservatively assume we should use cgo if it's
		// available.
		return fallbackOrder
	}

	var mdnsSource, filesSource, dnsSource bool
	var first string
	for _, src := range srcs {
		if src.source == "myhostname" {
			if isLocalhost(hostname) || isGateway(hostname) {
				return fallbackOrder
			}
			hn, err := getHostname()
			if err != nil || stringsEqualFold(hostname, hn) {
				return fallbackOrder
			}
			continue
		}
		if src.source == "files" || src.source == "dns" {
			if !src.standardCriteria() {
				return fallbackOrder // non-standard; let libc deal with it.
			}
			if src.source == "files" {
				filesSource = true
			} else if src.source == "dns" {
				dnsSource = true
			}
			if first == "" {
				first = src.source
			}
			continue
		}
		if stringsHasPrefix(src.source, "mdns") {
			// e.g. "mdns4", "mdns4_minimal"
			// We already returned true before if it was *.local.
			// libc wouldn't have found a hit on this anyway.
			mdnsSource = true
			continue
		}
		// Some source we don't know how to deal with.
		return fallbackOrder
	}

	// We don't parse mdns.allow files. They're rare. If one
	// exists, it might list other TLDs (besides .local) or even
	// '*', so just let libc deal with it.
	if mdnsSource && c.hasMDNSAllow {
		return fallbackOrder
	}

	// Cases where Go can handle it without cgo and C thread
	// overhead.
	switch {
	case filesSource && dnsSource:
		if first == "files" {
			return hostLookupFilesDNS
		} else {
			return hostLookupDNSFiles
		}
	case filesSource:
		return hostLookupFiles
	case dnsSource:
		return hostLookupDNS
	}

	// Something weird. Let libc deal with it.
	return fallbackOrder
}

// goDebugNetDNS parses the value of the GODEBUG "netdns" value.
// The netdns value can be of the form:
//
//	1       // debug level 1
//	2       // debug level 2
//	cgo     // use cgo for DNS lookups
//	go      // use go for DNS lookups
//	cgo+1   // use cgo for DNS lookups + debug level 1
//	1+cgo   // same
//	cgo+2   // same, but debug level 2
//
// etc.
func goDebugNetDNS() (dnsMode string, debugLevel int) {
	goDebug := godebug.Get("netdns")
	parsePart := func(s string) {
		if s == "" {
			return
		}
		if '0' <= s[0] && s[0] <= '9' {
			debugLevel, _, _ = dtoi(s)
		} else {
			dnsMode = s
		}
	}
	if i := bytealg.IndexByteString(goDebug, '+'); i != -1 {
		parsePart(goDebug[:i])
		parsePart(goDebug[i+1:])
		return
	}
	parsePart(goDebug)
	return
}

// isLocalhost reports whether h should be considered a "localhost"
// name for the myhostname NSS module.
func isLocalhost(h string) bool {
	return stringsEqualFold(h, "localhost") || stringsEqualFold(h, "localhost.localdomain") || stringsHasSuffixFold(h, ".localhost") || stringsHasSuffixFold(h, ".localhost.localdomain")
}

// isGateway reports whether h should be considered a "gateway"
// name for the myhostname NSS module.
func isGateway(h string) bool {
	return stringsEqualFold(h, "gateway")
}
