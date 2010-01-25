// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Read static host/IP entries from /etc/hosts.

package net

import (
	"os"
	"sync"
)

const cacheMaxAge = int64(300) // 5 minutes.

// hostsPath points to the file with static IP/address entries.
var hostsPath = "/etc/hosts"

// Simple cache.
var hosts struct {
	sync.Mutex
	data map[string][]string
	time int64
	path string
}

func readHosts() {
	now, _, _ := os.Time()
	hp := hostsPath
	if len(hosts.data) == 0 || hosts.time+cacheMaxAge <= now || hosts.path != hp {
		hs := make(map[string][]string)
		var file *file
		file, _ = open(hp)
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
				old, _ := hs[h]
				hs[h] = appendHost(old, f[0])
			}
		}
		// Update the data cache.
		hosts.time, _, _ = os.Time()
		hosts.path = hp
		hosts.data = hs
		file.close()
	}
}

func appendHost(hosts []string, address string) []string {
	n := len(hosts)
	if n+1 > cap(hosts) { // reallocate
		a := make([]string, n, 2*n+1)
		copy(a, hosts)
		hosts = a
	}
	hosts = hosts[0 : n+1]
	hosts[n] = address
	return hosts
}

// lookupStaticHosts looks up the addresses for the given host from /etc/hosts.
func lookupStaticHost(host string) []string {
	hosts.Lock()
	defer hosts.Unlock()
	readHosts()
	if len(hosts.data) != 0 {
		if ips, ok := hosts.data[host]; ok {
			return ips
		}
	}
	return nil
}
