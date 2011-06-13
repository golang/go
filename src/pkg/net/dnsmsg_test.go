// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"encoding/hex"
	"testing"
)

func TestDNSParseSRVReply(t *testing.T) {
	data, err := hex.DecodeString(dnsSRVReply)
	if err != nil {
		t.Fatal(err)
	}
	msg := new(dnsMsg)
	ok := msg.Unpack(data)
	if !ok {
		t.Fatalf("unpacking packet failed")
	}
	if g, e := len(msg.answer), 5; g != e {
		t.Errorf("len(msg.answer) = %d; want %d", g, e)
	}
	for idx, rr := range msg.answer {
		if g, e := rr.Header().Rrtype, uint16(dnsTypeSRV); g != e {
			t.Errorf("rr[%d].Header().Rrtype = %d; want %d", idx, g, e)
		}
		if _, ok := rr.(*dnsRR_SRV); !ok {
			t.Errorf("answer[%d] = %T; want *dnsRR_SRV", idx, rr)
		}
	}
	_, addrs, err := answer("_xmpp-server._tcp.google.com.", "foo:53", msg, uint16(dnsTypeSRV))
	if err != nil {
		t.Fatalf("answer: %v", err)
	}
	if g, e := len(addrs), 5; g != e {
		t.Errorf("len(addrs) = %d; want %d", g, e)
		t.Logf("addrs = %#v", addrs)
	}
}

func TestDNSParseCorruptSRVReply(t *testing.T) {
	data, err := hex.DecodeString(dnsSRVCorruptReply)
	if err != nil {
		t.Fatal(err)
	}
	msg := new(dnsMsg)
	ok := msg.Unpack(data)
	if !ok {
		t.Fatalf("unpacking packet failed")
	}
	if g, e := len(msg.answer), 5; g != e {
		t.Errorf("len(msg.answer) = %d; want %d", g, e)
	}
	for idx, rr := range msg.answer {
		if g, e := rr.Header().Rrtype, uint16(dnsTypeSRV); g != e {
			t.Errorf("rr[%d].Header().Rrtype = %d; want %d", idx, g, e)
		}
		if idx == 4 {
			if _, ok := rr.(*dnsRR_Header); !ok {
				t.Errorf("answer[%d] = %T; want *dnsRR_Header", idx, rr)
			}
		} else {
			if _, ok := rr.(*dnsRR_SRV); !ok {
				t.Errorf("answer[%d] = %T; want *dnsRR_SRV", idx, rr)
			}
		}
	}
	_, addrs, err := answer("_xmpp-server._tcp.google.com.", "foo:53", msg, uint16(dnsTypeSRV))
	if err != nil {
		t.Fatalf("answer: %v", err)
	}
	if g, e := len(addrs), 4; g != e {
		t.Errorf("len(addrs) = %d; want %d", g, e)
		t.Logf("addrs = %#v", addrs)
	}
}

// Valid DNS SRV reply
const dnsSRVReply = "0901818000010005000000000c5f786d70702d736572766572045f74637006676f6f67" +
	"6c6503636f6d0000210001c00c002100010000012c00210014000014950c786d70702d" +
	"73657276657234016c06676f6f676c6503636f6d00c00c002100010000012c00210014" +
	"000014950c786d70702d73657276657232016c06676f6f676c6503636f6d00c00c0021" +
	"00010000012c00210014000014950c786d70702d73657276657233016c06676f6f676c" +
	"6503636f6d00c00c002100010000012c00200005000014950b786d70702d7365727665" +
	"72016c06676f6f676c6503636f6d00c00c002100010000012c00210014000014950c78" +
	"6d70702d73657276657231016c06676f6f676c6503636f6d00"

// Corrupt DNS SRV reply, with its final RR having a bogus length
// (perhaps it was truncated, or it's malicious) The mutation is the
// capital "FF" below, instead of the proper "21".
const dnsSRVCorruptReply = "0901818000010005000000000c5f786d70702d736572766572045f74637006676f6f67" +
	"6c6503636f6d0000210001c00c002100010000012c00210014000014950c786d70702d" +
	"73657276657234016c06676f6f676c6503636f6d00c00c002100010000012c00210014" +
	"000014950c786d70702d73657276657232016c06676f6f676c6503636f6d00c00c0021" +
	"00010000012c00210014000014950c786d70702d73657276657233016c06676f6f676c" +
	"6503636f6d00c00c002100010000012c00200005000014950b786d70702d7365727665" +
	"72016c06676f6f676c6503636f6d00c00c002100010000012c00FF0014000014950c78" +
	"6d70702d73657276657231016c06676f6f676c6503636f6d00"
