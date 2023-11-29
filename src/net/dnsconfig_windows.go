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

	for _, aa := range aas {
		// Only take interfaces whose OperStatus is IfOperStatusUp(0x01) into DNS configs.
		if aa.OperStatus != windows.IfOperStatusUp {
			continue
		}

		// Only take interfaces which have at least one gateway
		if aa.FirstGatewayAddress == nil {
			continue
		}

		for dns := aa.FirstDnsServerAddress; dns != nil; dns = dns.Next {
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
					// fec0/10 IPv6 addresses are site local anycast DNS
					// addresses Microsoft sets by default if no other
					// IPv6 DNS address is set. Site local anycast is
					// deprecated since 2004, see
					// https://datatracker.ietf.org/doc/html/rfc3879
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
