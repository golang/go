// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/bytealg"
	"sync"
	"time"
)

const cacheMaxAge = 5 * time.Second

func parseLiteralIP(addr string) string {
	var ip IP
	var zone string
	ip = parseIPv4(addr)
	if ip == nil {
		ip, zone = parseIPv6Zone(addr)
	}
	if ip == nil {
		return ""
	}
	if zone == "" {
		return ip.String()
	}
	return ip.String() + "%" + zone
}

// hosts contains known host entries.
var hosts struct {
	sync.Mutex

	// Key for the list of literal IP addresses must be a host
	// name. It would be part of DNS labels, a FQDN or an absolute
	// FQDN.
	// For now the key is converted to lower case for convenience.
	byName map[string][]string

	// Key for the list of host names must be a literal IP address
	// including IPv6 address with zone identifier.
	// We don't support old-classful IP address notation.
	byAddr map[string][]string

	expire time.Time
	path   string
	mtime  time.Time
	size   int64
}

func readHosts() {
	now := time.Now()
	hp := testHookHostsPath

	if now.Before(hosts.expire) && hosts.path == hp && len(hosts.byName) > 0 {
		return
	}
	mtime, size, err := stat(hp)
	if err == nil && hosts.path == hp && hosts.mtime.Equal(mtime) && hosts.size == size {
		hosts.expire = now.Add(cacheMaxAge)
		return
	}

	hs := make(map[string][]string)
	is := make(map[string][]string)
	var file *file
	if file, _ = open(hp); file == nil {
		return
	}
	for line, ok := file.readLine(); ok; line, ok = file.readLine() {
		if i := bytealg.IndexByteString(line, '#'); i >= 0 {
			// Discard comments.
			line = line[0:i]
		}
		f := getFields(line)
		if len(f) < 2 {
			continue
		}
		addr := parseLiteralIP(f[0])
		if addr == "" {
			continue
		}
		for i := 1; i < len(f); i++ {
			name := absDomainName(f[i])
			h := []byte(f[i])
			lowerASCIIBytes(h)
			key := absDomainName(string(h))
			hs[key] = append(hs[key], addr)
			is[addr] = append(is[addr], name)
		}
	}
	// Update the data cache.
	hosts.expire = now.Add(cacheMaxAge)
	hosts.path = hp
	hosts.byName = hs
	hosts.byAddr = is
	hosts.mtime = mtime
	hosts.size = size
	file.close()
}

// lookupStaticHost looks up the addresses for the given host from /etc/hosts.
func lookupStaticHost(host string) []string {
	hosts.Lock()
	defer hosts.Unlock()
	readHosts()
	if len(hosts.byName) != 0 {
		if hasUpperCase(host) {
			lowerHost := []byte(host)
			lowerASCIIBytes(lowerHost)
			host = string(lowerHost)
		}
		if ips, ok := hosts.byName[absDomainName(host)]; ok {
			ipsCp := make([]string, len(ips))
			copy(ipsCp, ips)
			return ipsCp
		}
	}
	return nil
}

// lookupStaticAddr looks up the hosts for the given address from /etc/hosts.
func lookupStaticAddr(addr string) []string {
	hosts.Lock()
	defer hosts.Unlock()
	readHosts()
	addr = parseLiteralIP(addr)
	if addr == "" {
		return nil
	}
	if len(hosts.byAddr) != 0 {
		if hosts, ok := hosts.byAddr[addr]; ok {
			hostsCp := make([]string, len(hosts))
			copy(hostsCp, hosts)
			return hostsCp
		}
	}
	return nil
}
