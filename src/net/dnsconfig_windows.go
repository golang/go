// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/syscall/windows"
	"syscall"
	"time"
)

func dnsReadConfig(ignoredFilename string) (conf *dnsConfig) {
	conf = &dnsConfig{
		ndots:    1,
		timeout:  5 * time.Second,
		attempts: 2,
	}
	defer func() {
		if len(conf.servers) == 0 {
			conf.servers = defaultNS
		}
	}()
	aas, err := adapterAddresses()
	if err != nil {
		return
	}
	// TODO(bradfitz): this just collects all the DNS servers on all
	// the interfaces in some random order. It should order it by
	// default route, or only use the default route(s) instead.
	// In practice, however, it mostly works.
	for _, aa := range aas {
		for dns := aa.FirstDnsServerAddress; dns != nil; dns = dns.Next {
			// Only take interfaces whose OperStatus is IfOperStatusUp(0x01) into DNS configs.
			if aa.OperStatus != windows.IfOperStatusUp {
				continue
			}
			sa, err := dns.Address.Sockaddr.Sockaddr()
			if err != nil {
				continue
			}
			var ip IP
			switch sa := sa.(type) {
			case *syscall.SockaddrInet4:
				ip = IPv4(sa.Addr[0], sa.Addr[1], sa.Addr[2], sa.Addr[3])
			case *syscall.SockaddrInet6:
				ip = make(IP, IPv6len)
				copy(ip, sa.Addr[:])
				if ip[0] == 0xfe && ip[1] == 0xc0 {
					// Ignore these fec0/10 ones. Windows seems to
					// populate them as defaults on its misc rando
					// interfaces.
					continue
				}
			default:
				// Unexpected type.
				continue
			}
			conf.servers = append(conf.servers, JoinHostPort(ip.String(), "53"))
		}
	}
	return conf
}
