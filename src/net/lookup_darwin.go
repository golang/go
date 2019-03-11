// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"errors"
	"strings"

	"internal/x/net/dns/dnsmessage"
)

func (r *Resolver) lookupHost(ctx context.Context, host string) (addrs []string, err error) {
	order := systemConf().hostLookupOrder(r, host)
	if !r.preferGo() && order == hostLookupCgo {
		if addrs, err, ok := cgoLookupHost(ctx, host); ok {
			return addrs, err
		}
		// cgo not available (or netgo); fall back to linked bindings
		order = hostLookupFilesDNS
	}

	// darwin has unique resolution files, use libSystem binding if cgo is disabled.
	addrs, err := resolverSearch(ctx, host, int32(dnsmessage.TypeALL), int32(dnsmessage.ClassINET))
	if err == nil {
		return addrs, nil
	}
	// something went wrong, fallback to Go's DNS resolver

	return r.goLookupHostOrder(ctx, host, order)
}

func (r *Resolver) lookupIP(ctx context.Context, network, host string) (addrs []IPAddr, err error) {
	if r.preferGo() {
		return r.goLookupIP(ctx, host)
	}
	order := systemConf().hostLookupOrder(r, host)
	if order == hostLookupCgo {
		if addrs, err, ok := cgoLookupIP(ctx, network, host); ok {
			return addrs, err
		}

		// darwin has unique resolution files, use libSystem binding if cgo is disabled.
		addrs, err := resolverSearch(ctx, host, int32(dnsmessage.TypeALL), int32(dnsmessage.ClassINET))
		if err == nil {
			return addrs, nil
		}
		// something went wrong, fallback to Go's DNS resolver

		order = hostLookupFilesDNS
	}
	ips, _, err := r.goLookupIPCNAMEOrder(ctx, host, order)
	return ips, err
}

func (r *Resolver) lookupCNAME(ctx context.Context, name string) (string, error) {
	if !r.preferGo() && systemConf().canUseCgo() {
		if cname, err, ok := cgoLookupCNAME(ctx, name); ok {
			return cname, err
		}
	}

	// darwin has unique resolution files, use libSystem binding if cgo is not an option.
	addrs, err := resolverSearch(ctx, name, int32(dnsmessage.TypeCNAME), int32(dnsmessage.ClassINET))
	if err == nil {
		return addrs, nil
	}

	// something went wrong, fallback to Go's DNS resolver
	return r.goLookupCNAME(ctx, name)
}

// resolverSearch will make a call to the 'res_search' routine in libSystem
// and parse the output as a slice of IPAddr's
func resolverSearch(ctx context.Context, hostname string, rtype, class int32) ([]string, error) {

	var byteHostname = []byte(hostname)
	var responseBuffer = [512]byte{}

	retcode := res_search(&byteHostname[0], class, rtype, &responseBuffer[0], 512)
	if retcode < 0 {
		return nil, errors.New("could not complete domain resolution")
	}

	msg := &dnsmessage.Message{}
	err := msg.Unpack(responseBuffer[:])
	if err != nil {
		return nil, fmt.Errorf("could not parse dns response: %s", err.Error())
	}

	// parse received answers
	var dnsParser dnsmessage.Parser

	if _, err := dnsParser.Start(responseBuffer); err != nil {
		return nil, err
	}

	var answers []string
	for {
		h, err := dnsParser.AnswerHeader()
		if err == dnsmessage.ErrSectionDone {
			break
		}
		if err != nil {
			return nil, err
		}

		if !strings.EqualFold(h.Name.String(), hostname) {
			if err := dnsParser.SkipAnswer(); err != nil {
				return nil, err
			}
			continue
		}

		switch h.Type {
		case dnsmessage.TypeA:
			r, err := dnsParser.AResource()
			if err != nil {
				return nil, err
			}
			answers = append(answers, fmt.Stringf("%s", r.A))
		case dnsmessage.TypeAAAA:
			r, err := dnsParser.AAAAResource()
			if err != nil {
				return nil, err
			}
			answers = append(answers, fmt.Stringf("%s", r.AAAA))

		case dnsmessage.TypeCNAME:
			r, err := dnsParser.CNAMEResource()
			if err != nil {
				return nil, err
			}
			answers = append(answers, fmt.Stringf("%s", r.Name))
		}
	}
	return answers, nil
}

// res_search is defined in runtimne/lookup_darwin.go

func res_search(name *byte, class int32, rtype int32, answer *byte, anslen int32) int32
