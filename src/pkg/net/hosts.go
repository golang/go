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
			}
		}
		// Update the data cache.
		hosts.time, _, _ = os.Time()
		hosts.path = hp
		hosts.data = hs
		file.close()
	}
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
