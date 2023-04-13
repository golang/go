// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js && !wasip1

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
	resolver resolverType

	// machine has an /etc/mdns.allow file
	hasMDNSAllow bool

	goos          string // the runtime.GOOS, to ease testing
	dnsDebugLevel int
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
	confVal.resolver = defaultResolver

	// When build without support for cgo, force the go resolver, unless
	// "cgo" resolver is avaliable without the cgo support.
	if !cgoBuild {
		if runtime.GOOS != "windows" && runtime.GOOS != "darwin" {
			confVal.resolver = resolverGo
		}
	}

	dnsMode, debugLevel := goDebugNetDNS()
	confVal.dnsDebugLevel = debugLevel

	goDebug := false
	if confVal.dnsDebugLevel > 0 {
		defer func() {
			if confVal.dnsDebugLevel > 1 {
				println("go package net: confVal.resolver is ", confVal.resolver.String())
			}

			switch confVal.resolver {
			case resolverGo:
				if goDebug {
					println("go package net: GODEBUG setting forcing use of Go's resolver")
				} else {
					println("go package net: using the Go resolver")
				}
			case resolverCgo:
				// In this case the Go resolver might be still used when PreferGo
				// is set to true in the Resolver, so using the "preferred" word.
				if goDebug {
					println("go package net: GODEBUG setting: setting the preferred resolver to the Cgo resolver")
				} else {
					println("go package net: setting the preferred resolver to the Cgo resolver")
				}
			case resolverDynamic:
				println("go package net: dynamic selection of the DNS resolver, preferring the Go resolver")
			default:
				panic("unreachable")
			}
		}()
	}

	// Is only possible to change the resolver when
	// compiled with support for both resolvers.
	if confVal.resolver == resolverDynamic || confVal.resolver == resolverCgo {
		if dnsMode == "go" {
			goDebug = true
			confVal.resolver = resolverGo
		}
		if dnsMode == "cgo" {
			goDebug = true
			confVal.resolver = resolverCgo
		}
	}

	if confVal.resolver == resolverDynamic {
		// Keep the list of unsupported GOOS for resolverDynamic
		// in sysc with conf_test.go (testConfHostLookupOrder).

		// Neither of these platforms actually use cgo.
		// The meaning of "cgo" mode in the net package is
		// really "the native OS way", which for libc meant
		// cgo on the original platforms that motivated
		//
		// Darwin pops up annoying dialog boxes if programs try to do
		// their own DNS requests. So always use cgo instead, which
		// avoids that.
		if runtime.GOOS == "darwin" || runtime.GOOS == "ios" ||
			runtime.GOOS == "windows" || runtime.GOOS == "plan9" {
			confVal.resolver = resolverCgo
			return
		}

		if runtime.GOOS == "android" {
			confVal.resolver = resolverCgo
			return
		}
	}

	if confVal.resolver == resolverDynamic {
		// If any environment-specified resolver options are specified,
		// force cgo (when available). Note that LOCALDOMAIN can change behavior merely
		// by being specified with the empty string.
		_, localDomainDefined := syscall.Getenv("LOCALDOMAIN")
		if os.Getenv("RES_OPTIONS") != "" ||
			os.Getenv("HOSTALIASES") != "" ||
			localDomainDefined {
			confVal.resolver = resolverCgo
		}

		// OpenBSD apparently lets you override the location of resolv.conf
		// with ASR_CONFIG. If we notice that, defer to libc.
		if runtime.GOOS == "openbsd" && os.Getenv("ASR_CONFIG") != "" {
			confVal.resolver = resolverCgo
		}
	}

	if runtime.GOOS != "openbsd" {
		if _, err := os.Stat("/etc/mdns.allow"); err == nil {
			confVal.hasMDNSAllow = true
		}
	}
}

// canUseCgo reports whether calling cgo functions is allowed
// for non-hostname lookups.
func (c *conf) canUseCgo() bool {
	ret, _ := c.hostLookupOrder(nil, "")
	return ret == hostLookupCgo
}

// hostLookupOrder determines which strategy to use to resolve hostname.
// The provided Resolver is optional. nil means to not consider its options.
// It also returns dnsConfig when it was used to determine the lookup order.
func (c *conf) hostLookupOrder(r *Resolver, hostname string) (ret hostLookupOrder, dnsConfig *dnsConfig) {
	if c.dnsDebugLevel > 1 {
		defer func() {
			print("go package net: hostLookupOrder(", hostname, ") = ", ret.String(), "\n")
		}()
	}

	switch {
	case c.resolver == resolverGo || r.preferGo():
		if c.goos == "windows" {
			// TODO(bradfitz): implement files-based
			// lookup on Windows too? I guess /etc/hosts
			// kinda exists on Windows. But for now, only
			// do DNS.
			return hostLookupDNS, nil
		}

		if c.goos == "android" || c.goos == "plan9" {
			return hostLookupFilesDNS, nil
		}

		order, conf := c.unixResolverOrder(r, hostname)
		if order == hostLookupCgo {
			// Something unsupported in the configuration/hostname detected, but
			// we don't have cgo support, so try with the go resolver instead,
			// can't do any better at this point.
			return hostLookupFilesDNS, conf
		}
		return order, conf
	case c.resolver == resolverCgo:
		return hostLookupCgo, nil
	case c.resolver == resolverDynamic:
		// Try to use the go resolver, but if for some reason
		// we don't support it, use the cgo resolver.
		return c.unixResolverOrder(r, hostname)
	default:
		panic("unreachable, unknown resolver")
	}
}

func (c *conf) unixResolverOrder(r *Resolver, hostname string) (ret hostLookupOrder, dnsConfig *dnsConfig) {
	if bytealg.IndexByteString(hostname, '\\') != -1 || bytealg.IndexByteString(hostname, '%') != -1 {
		// Don't deal with special form hostnames with backslashes
		// or '%'.
		return hostLookupCgo, nil
	}

	conf := getSystemDNSConfig()
	if conf.err != nil && !os.IsNotExist(conf.err) && !os.IsPermission(conf.err) {
		// If we can't read the resolv.conf file, assume it
		// had something important in it and defer to cgo.
		// libc's resolver might then fail too, but at least
		// it wasn't our fault.
		return hostLookupCgo, conf
	}

	if conf.unknownOpt {
		return hostLookupCgo, conf
	}

	// OpenBSD is unique and doesn't use nsswitch.conf.
	// It also doesn't support mDNS.
	if c.goos == "openbsd" {
		// OpenBSD's resolv.conf manpage says that a non-existent
		// resolv.conf means "lookup" defaults to only "files",
		// without DNS lookups.
		if os.IsNotExist(conf.err) {
			return hostLookupFiles, conf
		}

		if os.IsPermission(conf.err) {
			return hostLookupCgo, conf
		}

		lookup := conf.lookup
		if len(lookup) == 0 {
			// https://www.openbsd.org/cgi-bin/man.cgi/OpenBSD-current/man5/resolv.conf.5
			// "If the lookup keyword is not used in the
			// system's resolv.conf file then the assumed
			// order is 'bind file'"
			return hostLookupDNSFiles, conf
		}
		if len(lookup) > 2 {
			return hostLookupCgo, conf
		}
		switch lookup[0] {
		case "bind":
			if len(lookup) == 2 {
				if lookup[1] == "file" {
					return hostLookupDNSFiles, conf
				}
				return hostLookupCgo, conf
			}
			return hostLookupDNS, conf
		case "file":
			if len(lookup) == 2 {
				if lookup[1] == "bind" {
					return hostLookupFilesDNS, conf
				}
				return hostLookupCgo, conf
			}
			return hostLookupFiles, conf
		default:
			return hostLookupCgo, conf
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
		return hostLookupCgo, conf
	}

	nss := getSystemNSS()
	srcs := nss.sources["hosts"]
	// If /etc/nsswitch.conf doesn't exist or doesn't specify any
	// sources for "hosts", assume Go's DNS will work fine.
	if os.IsNotExist(nss.err) || (nss.err == nil && len(srcs) == 0) {
		if c.goos == "solaris" {
			// illumos defaults to "nis [NOTFOUND=return] files"
			return hostLookupCgo, conf
		}

		return hostLookupFilesDNS, conf
	}
	if nss.err != nil {
		// We failed to parse or open nsswitch.conf, so
		// conservatively assume we should use cgo if it's
		// available.
		return hostLookupCgo, conf
	}

	var mdnsSource, filesSource, dnsSource bool
	var first string
	for _, src := range srcs {
		if src.source == "myhostname" {
			if isLocalhost(hostname) || isGateway(hostname) || isOutbound(hostname) {
				return hostLookupCgo, conf
			}
			hn, err := getHostname()
			if err != nil || stringsEqualFold(hostname, hn) {
				return hostLookupCgo, conf
			}
			continue
		}
		if src.source == "files" || src.source == "dns" {
			if !src.standardCriteria() {
				return hostLookupCgo, conf // non-standard; let libc deal with it.
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
		return hostLookupCgo, conf
	}

	// We don't parse mdns.allow files. They're rare. If one
	// exists, it might list other TLDs (besides .local) or even
	// '*', so just let libc deal with it.
	if mdnsSource && c.hasMDNSAllow {
		return hostLookupCgo, conf
	}

	// Cases where Go can handle it without cgo and C thread
	// overhead.
	switch {
	case filesSource && dnsSource:
		if first == "files" {
			return hostLookupFilesDNS, conf
		} else {
			return hostLookupDNSFiles, conf
		}
	case filesSource:
		return hostLookupFiles, conf
	case dnsSource:
		return hostLookupDNS, conf
	}

	// Something weird. Let libc deal with it.
	return hostLookupCgo, conf
}

var netdns = godebug.New("netdns")

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
	goDebug := netdns.Value()
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
	return stringsEqualFold(h, "_gateway")
}

// isOutbound reports whether h should be considered a "outbound"
// name for the myhostname NSS module.
func isOutbound(h string) bool {
	return stringsEqualFold(h, "_outbound")
}
