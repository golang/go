// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !netgo,!cgo
// +build darwin

package net

import (
	"context"
	"errors"
	"sync"

	"golang.org/x/net/dns/dnsmessage"
)

func cgoLookupHost(ctx context.Context, name string) (addrs []string, err error, completed bool) {
	resources, err := resolverGetResources(ctx, name, int32(dnsmessage.TypeALL), int32(dnsmessage.ClassINET))
	if err != nil {
		return
	}
	addrs, err = parseHostsFromResources(resources)
	if err != nil {
		return
	}
	return addrs, nil, true
}

func cgoLookupPort(ctx context.Context, network, service string) (port int, err error, completed bool) {
	port, err = goLookupPort(network, service) // just use netgo lookup
	return port, err, err == nil
}

func cgoLookupIP(ctx context.Context, network, name string) (addrs []IPAddr, err error, completed bool) {

	var resources []dnsmessage.Resource
	switch ipVersion(network) {
	case '4':
		resources, err = resolverGetResources(ctx, name, int32(dnsmessage.TypeA), int32(dnsmessage.ClassINET))
	case '6':
		resources, err = resolverGetResources(ctx, name, int32(dnsmessage.TypeAAAA), int32(dnsmessage.ClassINET))
	default:
		resources, err = resolverGetResources(ctx, name, int32(dnsmessage.TypeALL), int32(dnsmessage.ClassINET))
	}
	if err != nil {
		return
	}

	addrs, err = parseIPsFromResources(resources)
	if err != nil {
		return
	}

	return addrs, nil, true
}

func cgoLookupCNAME(ctx context.Context, name string) (cname string, err error, completed bool) {
	resources, err := resolverGetResources(ctx, name, int32(dnsmessage.TypeCNAME), int32(dnsmessage.ClassINET))
	if err != nil {
		return
	}
	cname, err = parseCNAMEFromResources(resources)
	if err != nil {
		return "", err, false
	}
	return cname, nil, true
}

func cgoLookupPTR(ctx context.Context, addr string) (ptrs []string, err error, completed bool) {
	resources, err := resolverGetResources(ctx, addr, int32(dnsmessage.TypePTR), int32(dnsmessage.ClassINET))
	if err != nil {
		return
	}
	ptrs, err = parsePTRsFromResources(resources)
	if err != nil {
		return
	}
	return ptrs, nil, true
}

var (
	resInitOnce sync.Once
	retcode     int32
)

// resolverGetResources will make a call to the 'res_search' routine in libSystem
// and parse the output as a slice of resource resources which can then be parsed
func resolverGetResources(ctx context.Context, hostname string, rtype, class int32) ([]dnsmessage.Resource, error) {

	resInitOnce.Do(func() {
		retcode = res_init()
	})
	if retcode < 0 {
		return nil, errors.New("could not initialize name resolver data")
	}

	var byteHostname = []byte(hostname)
	var responseBuffer [512]byte

	retcode = res_search(&byteHostname[0], class, rtype, &responseBuffer[0], int32(len(responseBuffer)))
	if retcode < 0 {
		return nil, errors.New("could not complete domain resolution")
	}

	var msg dnsmessage.Message
	err := msg.Unpack(responseBuffer[:])
	if err != nil {
		return nil, err
	}

	var dnsParser dnsmessage.Parser
	if _, err := dnsParser.Start(responseBuffer[:]); err != nil {
		return nil, err
	}

	var resources []dnsmessage.Resource
	for {
		r, err := dnsParser.Answer()
		if err == dnsmessage.ErrSectionDone {
			break
		}
		if err != nil {
			return nil, err
		}
		resources = append(resources, r)
	}
	return resources, nil
}

func parseHostsFromResources(resources []dnsmessage.Resource) ([]string, error) {
	var answers []string

	for i := range resources {
		switch resources[i].Header.Type {
		case dnsmessage.TypeA:
			b := resources[i].Body.(*dnsmessage.AResource)
			answers = append(answers, string(b.A[:]))
		case dnsmessage.TypeAAAA:
			b := resources[i].Body.(*dnsmessage.AAAAResource)
			answers = append(answers, string(b.AAAA[:]))
		default:
			return nil, errors.New("could not parse an A or AAAA response from message buffer")
		}
	}
	return answers, nil
}

func parseIPsFromResources(resources []dnsmessage.Resource) ([]IPAddr, error) {
	var answers []IPAddr

	for i := range resources {
		switch resources[i].Header.Type {
		case dnsmessage.TypeA:
			b := resources[i].Body.(*dnsmessage.AResource)
			ip := parseIPv4(string(b.A[:]))
			answers = append(answers, IPAddr{IP: ip})
		case dnsmessage.TypeAAAA:
			b := resources[i].Body.(*dnsmessage.AAAAResource)
			ip, zone := parseIPv6Zone(string(b.AAAA[:]))
			answers = append(answers, IPAddr{IP: ip, Zone: zone})
		default:
			return nil, errors.New("could not parse an A or AAAA response from message buffer")
		}
	}
	return answers, nil
}

func parseCNAMEFromResources(resources []dnsmessage.Resource) (string, error) {
	if len(resources) == 0 {
		return "", errors.New("no CNAME record received")
	}
	c, ok := resources[0].Body.(*dnsmessage.CNAMEResource)
	if !ok {
		return "", errors.New("could not parse CNAME record")
	}
	return c.CNAME.String(), nil
}

func parsePTRsFromResources(resources []dnsmessage.Resource) ([]string, error) {
	var answers []string
	for i := range resources {
		switch resources[i].Header.Type {
		case dnsmessage.TypePTR:
			p := resources[0].Body.(*dnsmessage.PTRResource)
			answers = append(answers, p.PTR.String())
		default:
			return nil, errors.New("could not parse a PTR response from message buffer")

		}
	}
	return answers, nil
}

// res_init and res_search are defined in runtime/lookup_darwin.go

func res_init() int32

func res_search(dname *byte, class int32, rtype int32, answer *byte, anslen int32) int32
