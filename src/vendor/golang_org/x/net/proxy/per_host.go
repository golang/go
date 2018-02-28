// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proxy

import (
	"net"
	"strings"
)

// A PerHost directs connections to a default Dialer unless the hostname
// requested matches one of a number of exceptions.
type PerHost struct {
	def, bypass Dialer

	bypassNetworks []*net.IPNet
	bypassIPs      []net.IP
	bypassZones    []string
	bypassHosts    []string
}

// NewPerHost returns a PerHost Dialer that directs connections to either
// defaultDialer or bypass, depending on whether the connection matches one of
// the configured rules.
func NewPerHost(defaultDialer, bypass Dialer) *PerHost {
	return &PerHost{
		def:    defaultDialer,
		bypass: bypass,
	}
}

// Dial connects to the address addr on the given network through either
// defaultDialer or bypass.
func (p *PerHost) Dial(network, addr string) (c net.Conn, err error) {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return nil, err
	}

	return p.dialerForRequest(host).Dial(network, addr)
}

func (p *PerHost) dialerForRequest(host string) Dialer {
	if ip := net.ParseIP(host); ip != nil {
		for _, net := range p.bypassNetworks {
			if net.Contains(ip) {
				return p.bypass
			}
		}
		for _, bypassIP := range p.bypassIPs {
			if bypassIP.Equal(ip) {
				return p.bypass
			}
		}
		return p.def
	}

	for _, zone := range p.bypassZones {
		if strings.HasSuffix(host, zone) {
			return p.bypass
		}
		if host == zone[1:] {
			// For a zone "example.com", we match "example.com"
			// too.
			return p.bypass
		}
	}
	for _, bypassHost := range p.bypassHosts {
		if bypassHost == host {
			return p.bypass
		}
	}
	return p.def
}

// AddFromString parses a string that contains comma-separated values
// specifying hosts that should use the bypass proxy. Each value is either an
// IP address, a CIDR range, a zone (*.example.com) or a hostname
// (localhost). A best effort is made to parse the string and errors are
// ignored.
func (p *PerHost) AddFromString(s string) {
	hosts := strings.Split(s, ",")
	for _, host := range hosts {
		host = strings.TrimSpace(host)
		if len(host) == 0 {
			continue
		}
		if strings.Contains(host, "/") {
			// We assume that it's a CIDR address like 127.0.0.0/8
			if _, net, err := net.ParseCIDR(host); err == nil {
				p.AddNetwork(net)
			}
			continue
		}
		if ip := net.ParseIP(host); ip != nil {
			p.AddIP(ip)
			continue
		}
		if strings.HasPrefix(host, "*.") {
			p.AddZone(host[1:])
			continue
		}
		p.AddHost(host)
	}
}

// AddIP specifies an IP address that will use the bypass proxy. Note that
// this will only take effect if a literal IP address is dialed. A connection
// to a named host will never match an IP.
func (p *PerHost) AddIP(ip net.IP) {
	p.bypassIPs = append(p.bypassIPs, ip)
}

// AddNetwork specifies an IP range that will use the bypass proxy. Note that
// this will only take effect if a literal IP address is dialed. A connection
// to a named host will never match.
func (p *PerHost) AddNetwork(net *net.IPNet) {
	p.bypassNetworks = append(p.bypassNetworks, net)
}

// AddZone specifies a DNS suffix that will use the bypass proxy. A zone of
// "example.com" matches "example.com" and all of its subdomains.
func (p *PerHost) AddZone(zone string) {
	if strings.HasSuffix(zone, ".") {
		zone = zone[:len(zone)-1]
	}
	if !strings.HasPrefix(zone, ".") {
		zone = "." + zone
	}
	p.bypassZones = append(p.bypassZones, zone)
}

// AddHost specifies a hostname that will use the bypass proxy.
func (p *PerHost) AddHost(host string) {
	if strings.HasSuffix(host, ".") {
		host = host[:len(host)-1]
	}
	p.bypassHosts = append(p.bypassHosts, host)
}
