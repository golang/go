// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"errors"
	"internal/bytealg"
	"internal/godebug"
	"io/fs"
	"os"
	"runtime"
	"sync"
	"syscall"
)

// The net package's name resolution is rather complicated.
// There are two main approaches, go and cgo.
// The cgo resolver uses C functions like getaddrinfo.
// The go resolver reads system files directly and
// sends DNS packets directly to servers.
//
// The netgo build tag prefers the go resolver.
// The netcgo build tag prefers the cgo resolver.
//
// The netgo build tag also prohibits the use of the cgo tool.
// However, on Darwin, Plan 9, and Windows the cgo resolver is still available.
// On those systems the cgo resolver does not require the cgo tool.
// (The term "cgo resolver" was locked in by GODEBUG settings
// at a time when the cgo resolver did require the cgo tool.)
//
// Adding netdns=go to GODEBUG will prefer the go resolver.
// Adding netdns=cgo to GODEBUG will prefer the cgo resolver.
//
// The Resolver struct has a PreferGo field that user code
// may set to prefer the go resolver. It is documented as being
// equivalent to adding netdns=go to GODEBUG.
//
// When deciding which resolver to use, we first check the PreferGo field.
// If that is not set, we check the GODEBUG setting.
// If that is not set, we check the netgo or netcgo build tag.
// If none of those are set, we normally prefer the go resolver by default.
// However, if the cgo resolver is available,
// there is a complex set of conditions for which we prefer the cgo resolver.
//
// Other files define the netGoBuildTag, netCgoBuildTag, and cgoAvailable
// constants.

// conf is used to determine name resolution configuration.
type conf struct {
	netGo  bool // prefer go approach, based on build tag and GODEBUG
	netCgo bool // prefer cgo approach, based on build tag and GODEBUG

	dnsDebugLevel int // from GODEBUG

	preferCgo bool // if no explicit preference, use cgo

	goos     string   // copy of runtime.GOOS, used for testing
	mdnsTest mdnsTest // assume /etc/mdns.allow exists, for testing
}

// mdnsTest is for testing only.
type mdnsTest int

const (
	mdnsFromSystem mdnsTest = iota
	mdnsAssumeExists
	mdnsAssumeDoesNotExist
)

var (
	confOnce sync.Once // guards init of confVal via initConfVal
	confVal  = &conf{goos: runtime.GOOS}
)

// systemConf returns the machine's network configuration.
func systemConf() *conf {
	confOnce.Do(initConfVal)
	return confVal
}

// initConfVal initializes confVal based on the environment
// that will not change during program execution.
func initConfVal() {
	dnsMode, debugLevel := goDebugNetDNS()
	confVal.netGo = netGoBuildTag || dnsMode == "go"
	confVal.netCgo = netCgoBuildTag || dnsMode == "cgo"
	confVal.dnsDebugLevel = debugLevel

	if confVal.dnsDebugLevel > 0 {
		defer func() {
			if confVal.dnsDebugLevel > 1 {
				println("go package net: confVal.netCgo =", confVal.netCgo, " netGo =", confVal.netGo)
			}
			switch {
			case confVal.netGo:
				if netGoBuildTag {
					println("go package net: built with netgo build tag; using Go's DNS resolver")
				} else {
					println("go package net: GODEBUG setting forcing use of Go's resolver")
				}
			case !cgoAvailable:
				println("go package net: cgo resolver not supported; using Go's DNS resolver")
			case confVal.netCgo || confVal.preferCgo:
				println("go package net: using cgo DNS resolver")
			default:
				println("go package net: dynamic selection of DNS resolver")
			}
		}()
	}

	// The remainder of this function sets preferCgo based on
	// conditions that will not change during program execution.

	// By default, prefer the go resolver.
	confVal.preferCgo = false

	// If the cgo resolver is not available, we can't prefer it.
	if !cgoAvailable {
		return
	}

	// Some operating systems always prefer the cgo resolver.
	if goosPrefersCgo() {
		confVal.preferCgo = true
		return
	}

	// The remaining checks are specific to Unix systems.
	switch runtime.GOOS {
	case "plan9", "windows", "js", "wasip1":
		return
	}

	// If any environment-specified resolver options are specified,
	// prefer the cgo resolver.
	// Note that LOCALDOMAIN can change behavior merely by being
	// specified with the empty string.
	_, localDomainDefined := syscall.Getenv("LOCALDOMAIN")
	if localDomainDefined || os.Getenv("RES_OPTIONS") != "" || os.Getenv("HOSTALIASES") != "" {
		confVal.preferCgo = true
		return
	}

	// OpenBSD apparently lets you override the location of resolv.conf
	// with ASR_CONFIG. If we notice that, defer to libc.
	if runtime.GOOS == "openbsd" && os.Getenv("ASR_CONFIG") != "" {
		confVal.preferCgo = true
		return
	}
}

// goosPrefersCgo reports whether the GOOS value passed in prefers
// the cgo resolver.
func goosPrefersCgo() bool {
	switch runtime.GOOS {
	// Historically on Windows and Plan 9 we prefer the
	// cgo resolver (which doesn't use the cgo tool) rather than
	// the go resolver. This is because originally these
	// systems did not support the go resolver.
	// Keep it this way for better compatibility.
	// Perhaps we can revisit this some day.
	case "windows", "plan9":
		return true

	// Darwin pops up annoying dialog boxes if programs try to
	// do their own DNS requests, so prefer cgo.
	case "darwin", "ios":
		return true

	// DNS requests don't work on Android, so prefer the cgo resolver.
	// Issue #10714.
	case "android":
		return true

	default:
		return false
	}
}

// mustUseGoResolver reports whether a DNS lookup of any sort is
// required to use the go resolver. The provided Resolver is optional.
// This will report true if the cgo resolver is not available.
func (c *conf) mustUseGoResolver(r *Resolver) bool {
	if !cgoAvailable {
		return true
	}

	if runtime.GOOS == "plan9" {
		// TODO(bradfitz): for now we only permit use of the PreferGo
		// implementation when there's a non-nil Resolver with a
		// non-nil Dialer. This is a sign that they the code is trying
		// to use their DNS-speaking net.Conn (such as an in-memory
		// DNS cache) and they don't want to actually hit the network.
		// Once we add support for looking the default DNS servers
		// from plan9, though, then we can relax this.
		if r == nil || r.Dial == nil {
			return false
		}
	}

	return c.netGo || r.preferGo()
}

// addrLookupOrder determines which strategy to use to resolve addresses.
// The provided Resolver is optional. nil means to not consider its options.
// It also returns dnsConfig when it was used to determine the lookup order.
func (c *conf) addrLookupOrder(r *Resolver, addr string) (ret hostLookupOrder, dnsConf *dnsConfig) {
	if c.dnsDebugLevel > 1 {
		defer func() {
			print("go package net: addrLookupOrder(", addr, ") = ", ret.String(), "\n")
		}()
	}
	return c.lookupOrder(r, "")
}

// hostLookupOrder determines which strategy to use to resolve hostname.
// The provided Resolver is optional. nil means to not consider its options.
// It also returns dnsConfig when it was used to determine the lookup order.
func (c *conf) hostLookupOrder(r *Resolver, hostname string) (ret hostLookupOrder, dnsConf *dnsConfig) {
	if c.dnsDebugLevel > 1 {
		defer func() {
			print("go package net: hostLookupOrder(", hostname, ") = ", ret.String(), "\n")
		}()
	}
	return c.lookupOrder(r, hostname)
}

func (c *conf) lookupOrder(r *Resolver, hostname string) (ret hostLookupOrder, dnsConf *dnsConfig) {
	// fallbackOrder is the order we return if we can't figure it out.
	var fallbackOrder hostLookupOrder

	var canUseCgo bool
	if c.mustUseGoResolver(r) {
		// Go resolver was explicitly requested
		// or cgo resolver is not available.
		// Figure out the order below.
		fallbackOrder = hostLookupFilesDNS
		canUseCgo = false
	} else if c.netCgo {
		// Cgo resolver was explicitly requested.
		return hostLookupCgo, nil
	} else if c.preferCgo {
		// Given a choice, we prefer the cgo resolver.
		return hostLookupCgo, nil
	} else {
		// Neither resolver was explicitly requested
		// and we have no preference.

		if bytealg.IndexByteString(hostname, '\\') != -1 || bytealg.IndexByteString(hostname, '%') != -1 {
			// Don't deal with special form hostnames
			// with backslashes or '%'.
			return hostLookupCgo, nil
		}

		// If something is unrecognized, use cgo.
		fallbackOrder = hostLookupCgo
		canUseCgo = true
	}

	// On systems that don't use /etc/resolv.conf or /etc/nsswitch.conf, we are done.
	switch c.goos {
	case "windows", "plan9", "android", "ios":
		return fallbackOrder, nil
	}

	// Try to figure out the order to use for searches.
	// If we don't recognize something, use fallbackOrder.
	// That will use cgo unless the Go resolver was explicitly requested.
	// If we do figure out the order, return something other
	// than fallbackOrder to use the Go resolver with that order.

	dnsConf = getSystemDNSConfig()

	if canUseCgo && dnsConf.err != nil && !errors.Is(dnsConf.err, fs.ErrNotExist) && !errors.Is(dnsConf.err, fs.ErrPermission) {
		// We can't read the resolv.conf file, so use cgo if we can.
		return hostLookupCgo, dnsConf
	}

	if canUseCgo && dnsConf.unknownOpt {
		// We didn't recognize something in resolv.conf,
		// so use cgo if we can.
		return hostLookupCgo, dnsConf
	}

	// OpenBSD is unique and doesn't use nsswitch.conf.
	// It also doesn't support mDNS.
	if c.goos == "openbsd" {
		// OpenBSD's resolv.conf manpage says that a
		// non-existent resolv.conf means "lookup" defaults
		// to only "files", without DNS lookups.
		if errors.Is(dnsConf.err, fs.ErrNotExist) {
			return hostLookupFiles, dnsConf
		}

		lookup := dnsConf.lookup
		if len(lookup) == 0 {
			// https://www.openbsd.org/cgi-bin/man.cgi/OpenBSD-current/man5/resolv.conf.5
			// "If the lookup keyword is not used in the
			// system's resolv.conf file then the assumed
			// order is 'bind file'"
			return hostLookupDNSFiles, dnsConf
		}
		if len(lookup) < 1 || len(lookup) > 2 {
			// We don't recognize this format.
			return fallbackOrder, dnsConf
		}
		switch lookup[0] {
		case "bind":
			if len(lookup) == 2 {
				if lookup[1] == "file" {
					return hostLookupDNSFiles, dnsConf
				}
				// Unrecognized.
				return fallbackOrder, dnsConf
			}
			return hostLookupDNS, dnsConf
		case "file":
			if len(lookup) == 2 {
				if lookup[1] == "bind" {
					return hostLookupFilesDNS, dnsConf
				}
				// Unrecognized.
				return fallbackOrder, dnsConf
			}
			return hostLookupFiles, dnsConf
		default:
			// Unrecognized.
			return fallbackOrder, dnsConf
		}

		// We always return before this point.
		// The code below is for non-OpenBSD.
	}

	// Canonicalize the hostname by removing any trailing dot.
	if stringsHasSuffix(hostname, ".") {
		hostname = hostname[:len(hostname)-1]
	}

	nss := getSystemNSS()
	srcs := nss.sources["hosts"]
	// If /etc/nsswitch.conf doesn't exist or doesn't specify any
	// sources for "hosts", assume Go's DNS will work fine.
	if errors.Is(nss.err, fs.ErrNotExist) || (nss.err == nil && len(srcs) == 0) {
		if canUseCgo && c.goos == "solaris" {
			// illumos defaults to
			// "nis [NOTFOUND=return] files",
			// which the go resolver doesn't support.
			return hostLookupCgo, dnsConf
		}

		return hostLookupFilesDNS, dnsConf
	}
	if nss.err != nil {
		// We failed to parse or open nsswitch.conf, so
		// we have nothing to base an order on.
		return fallbackOrder, dnsConf
	}

	var hasDNSSource bool
	var hasDNSSourceChecked bool

	var filesSource, dnsSource bool
	var first string
	for i, src := range srcs {
		if src.source == "files" || src.source == "dns" {
			if canUseCgo && !src.standardCriteria() {
				// non-standard; let libc deal with it.
				return hostLookupCgo, dnsConf
			}
			if src.source == "files" {
				filesSource = true
			} else {
				hasDNSSource = true
				hasDNSSourceChecked = true
				dnsSource = true
			}
			if first == "" {
				first = src.source
			}
			continue
		}

		if canUseCgo {
			switch {
			case hostname != "" && src.source == "myhostname":
				// Let the cgo resolver handle myhostname
				// if we are looking up the local hostname.
				if isLocalhost(hostname) || isGateway(hostname) || isOutbound(hostname) {
					return hostLookupCgo, dnsConf
				}
				hn, err := getHostname()
				if err != nil || stringsEqualFold(hostname, hn) {
					return hostLookupCgo, dnsConf
				}
				continue
			case hostname != "" && stringsHasPrefix(src.source, "mdns"):
				if stringsHasSuffixFold(hostname, ".local") {
					// Per RFC 6762, the ".local" TLD is special. And
					// because Go's native resolver doesn't do mDNS or
					// similar local resolution mechanisms, assume that
					// libc might (via Avahi, etc) and use cgo.
					return hostLookupCgo, dnsConf
				}

				// We don't parse mdns.allow files. They're rare. If one
				// exists, it might list other TLDs (besides .local) or even
				// '*', so just let libc deal with it.
				var haveMDNSAllow bool
				switch c.mdnsTest {
				case mdnsFromSystem:
					_, err := os.Stat("/etc/mdns.allow")
					if err != nil && !errors.Is(err, fs.ErrNotExist) {
						// Let libc figure out what is going on.
						return hostLookupCgo, dnsConf
					}
					haveMDNSAllow = err == nil
				case mdnsAssumeExists:
					haveMDNSAllow = true
				case mdnsAssumeDoesNotExist:
					haveMDNSAllow = false
				}
				if haveMDNSAllow {
					return hostLookupCgo, dnsConf
				}
				continue
			default:
				// Some source we don't know how to deal with.
				return hostLookupCgo, dnsConf
			}
		}

		if !hasDNSSourceChecked {
			hasDNSSourceChecked = true
			for _, v := range srcs[i+1:] {
				if v.source == "dns" {
					hasDNSSource = true
					break
				}
			}
		}

		// If we saw a source we don't recognize, which can only
		// happen if we can't use the cgo resolver, treat it as DNS,
		// but only when there is no dns in all other sources.
		if !hasDNSSource {
			dnsSource = true
			if first == "" {
				first = "dns"
			}
		}
	}

	// Cases where Go can handle it without cgo and C thread overhead,
	// or where the Go resolver has been forced.
	switch {
	case filesSource && dnsSource:
		if first == "files" {
			return hostLookupFilesDNS, dnsConf
		} else {
			return hostLookupDNSFiles, dnsConf
		}
	case filesSource:
		return hostLookupFiles, dnsConf
	case dnsSource:
		return hostLookupDNS, dnsConf
	}

	// Something weird. Fallback to the default.
	return fallbackOrder, dnsConf
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

// isOutbound reports whether h should be considered an "outbound"
// name for the myhostname NSS module.
func isOutbound(h string) bool {
	return stringsEqualFold(h, "_outbound")
}
