// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// DNS client.
// Has to be linked into package net for Dial.

// TODO(rsc):
//	Check periodically whether /etc/resolv.conf has changed.
//	Could potentially handle many outstanding lookups faster.
//	Could have a small cache.
//	Random UDP source port (net.Dial should do that for us).
//	Random request IDs.
//	More substantial error reporting.
//	Remove use of fmt?

package net

import (
	"fmt";
	"io";
	"net";
	"once";
	"os";
	"strings";
)

var (
	DNS_InternalError = os.NewError("internal dns error");
	DNS_MissingConfig = os.NewError("no dns configuration");
	DNS_No_Answer = os.NewError("dns got no answer");
	DNS_BadRequest = os.NewError("malformed dns request");
	DNS_BadReply = os.NewError("malformed dns reply");
	DNS_ServerFailure = os.NewError("dns server failure");
	DNS_NoServers = os.NewError("no dns servers");
	DNS_NameTooLong = os.NewError("dns name too long");
	DNS_RedirectLoop = os.NewError("dns redirect loop");
	DNS_NameNotFound = os.NewError("dns name not found");
);

// Send a request on the connection and hope for a reply.
// Up to cfg.attempts attempts.
func _Exchange(cfg *DNS_Config, c Conn, name string) (m *DNS_Msg, err *os.Error) {
	if len(name) >= 256 {
		return nil, DNS_NameTooLong
	}
	out := new(DNS_Msg);
	out.id = 0x1234;
	out.question = []DNS_Question(
		DNS_Question( name, DNS_TypeA, DNS_ClassINET )
	);
	out.recursion_desired = true;
	msg, ok := out.Pack();
	if !ok {
		return nil, DNS_InternalError
	}

	for attempt := 0; attempt < cfg.attempts; attempt++ {
		n, err := c.Write(msg);
		if err != nil {
			return nil, err
		}

		// TODO(rsc): set up timeout or call ReadTimeout.
		// right now net does not support that.

		buf := make([]byte, 2000);	// More than enough.
		n, err = c.Read(buf);
		if err != nil {
			// TODO(rsc): only continue if timed out
			continue
		}
		buf = buf[0:n];
		in := new(DNS_Msg);
		if !in.Unpack(buf) || in.id != out.id {
			continue
		}
		return in, nil
	}
	return nil, DNS_No_Answer
}


// Find answer for name in dns message.
// On return, if err == nil, addrs != nil.
// TODO(rsc): Maybe return [][]byte (==[]IPAddr) instead?
func _Answer(name string, dns *DNS_Msg) (addrs []string, err *os.Error) {
	addrs = make([]string, 0, len(dns.answer));

	if dns.rcode == DNS_RcodeNameError && dns.authoritative {
		return nil, DNS_NameNotFound	// authoritative "no such host"
	}
	if dns.rcode != DNS_RcodeSuccess {
		// None of the error codes make sense
		// for the query we sent.  If we didn't get
		// a name error and we didn't get success,
		// the server is behaving incorrectly.
		return nil, DNS_ServerFailure
	}

	// Look for the name.
	// Presotto says it's okay to assume that servers listed in
	// /etc/resolv.conf are recursive resolvers.
	// We asked for recursion, so it should have included
	// all the answers we need in this one packet.
Cname:
	for cnameloop := 0; cnameloop < 10; cnameloop++ {
		addrs = addrs[0:0];
		for i := 0; i < len(dns.answer); i++ {
			rr := dns.answer[i];
			h := rr.Header();
			if h.class == DNS_ClassINET && h.name == name {
				switch h.rrtype {
				case DNS_TypeA:
					n := len(addrs);
					a := rr.(*DNS_RR_A).a;
					addrs = addrs[0:n+1];
					addrs[n] = fmt.Sprintf("%d.%d.%d.%d", (a>>24), (a>>16)&0xFF, (a>>8)&0xFF, a&0xFF);
				case DNS_TypeCNAME:
					// redirect to cname
					name = rr.(*DNS_RR_CNAME).cname;
					continue Cname
				}
			}
		}
		if len(addrs) == 0 {
			return nil, DNS_NameNotFound
		}
		return addrs, nil
	}

	// Too many redirects
	return nil, DNS_RedirectLoop
}

// Do a lookup for a single name, which must be rooted
// (otherwise _Answer will not find the answers).
func _TryOneName(cfg *DNS_Config, name string) (addrs []string, err *os.Error) {
	err = DNS_NoServers;
	for i := 0; i < len(cfg.servers); i++ {
		// Calling Dial here is scary -- we have to be sure
		// not to dial a name that will require a DNS lookup,
		// or Dial will call back here to translate it.
		// The DNS config parser has already checked that
		// all the cfg.servers[i] are IP addresses, which
		// Dial will use without a DNS lookup.
		c, cerr := Dial("udp", "", cfg.servers[i] + ":53");
		if cerr != nil {
			err = cerr;
			continue;
		}
		msg, merr := _Exchange(cfg, c, name);
		c.Close();
		if merr != nil {
			err = merr;
			continue;
		}
		addrs, aerr := _Answer(name, msg);
		if aerr != nil && aerr != DNS_NameNotFound {
			err = aerr;
			continue;
		}
		return addrs, aerr;
	}
	return;
}

var cfg *DNS_Config

func _LoadConfig() {
	cfg = DNS_ReadConfig();
}

func LookupHost(name string) (name1 string, addrs []string, err *os.Error) {
	// TODO(rsc): Pick out obvious non-DNS names to avoid
	// sending stupid requests to the server?

	once.Do(_LoadConfig);
	if cfg == nil {
		err = DNS_MissingConfig;
		return;
	}

	// If name is rooted (trailing dot) or has enough dots,
	// try it by itself first.
	rooted := len(name) > 0 && name[len(name)-1] == '.';
	if rooted || strings.Count(name, ".") >= cfg.ndots {
		rname := name;
		if !rooted {
			rname += ".";
		}
		// Can try as ordinary name.
		addrs, aerr := _TryOneName(cfg, rname);
		if aerr == nil {
			return rname, addrs, nil;
		}
		err = aerr;
	}
	if rooted {
		return
	}

	// Otherwise, try suffixes.
	for i := 0; i < len(cfg.search); i++ {
		newname := name+"."+cfg.search[i];
		if newname[len(newname)-1] != '.' {
			newname += "."
		}
		addrs, aerr := _TryOneName(cfg, newname);
		if aerr == nil {
			return newname, addrs, nil;
		}
		err = aerr;
	}
	return
}
