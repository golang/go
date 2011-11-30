// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Read static host/IP entries from /etc/hosts.

package net

import (
	"sync"
	"time"
)

const cacheMaxAge = 5 * time.Minute

// hostsPath points to the file with static IP/address entries.
var hostsPath = "/etc/hosts"

// Simple cache.
var hosts struct {
	sync.Mutex
	byName map[string][]string
	byAddr map[string][]string
	expire time.Time
	path   string
}

func readHosts() {
	now := time.Now()
	hp := hostsPath
	if len(hosts.byName) == 0 || now.After(hosts.expire) || hosts.path != hp {
		hs := make(map[string][]string)
		is := make(map[string][]string)
		var file *file
		if file, _ = open(hp); file == nil {
			return
		}
		for line, ok := file.readLine(); ok; line, ok = file.readLine() {
			if i := byteIndex(line, '#'); i >= 0 {
				// Discard comments.
				line = line[0:i]
			}
			f := getFields(line)
			if len(f) < 2 || ParseIP(f[0]) == nil {
				continue
			}
			for i := 1; i < len(f); i++ {
				h := f[i]
				hs[h] = append(hs[h], f[0])
				is[f[0]] = append(is[f[0]], h)
			}
		}
		// Update the data cache.
		hosts.expire = time.Now().Add(cacheMaxAge)
		hosts.path = hp
		hosts.byName = hs
		hosts.byAddr = is
		file.close()
	}
}

// lookupStaticHost looks up the addresses for the given host from /etc/hosts.
func lookupStaticHost(host string) []string {
	hosts.Lock()
	defer hosts.Unlock()
	readHosts()
	if len(hosts.byName) != 0 {
		if ips, ok := hosts.byName[host]; ok {
			return ips
		}
	}
	return nil
}

// lookupStaticAddr looks up the hosts for the given address from /etc/hosts.
func lookupStaticAddr(addr string) []string {
	hosts.Lock()
	defer hosts.Unlock()
	readHosts()
	if len(hosts.byAddr) != 0 {
		if hosts, ok := hosts.byAddr[addr]; ok {
			return hosts
		}
	}
	return nil
}
