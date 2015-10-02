// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"encoding/hex"
	"reflect"
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
		t.Fatal("unpacking packet failed")
	}
	msg.String() // exercise this code path
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
	for _, name := range [...]string{
		"_xmpp-server._tcp.google.com.",
		"_XMPP-Server._TCP.Google.COM.",
		"_XMPP-SERVER._TCP.GOOGLE.COM.",
	} {
		_, addrs, err := answer(name, "foo:53", msg, uint16(dnsTypeSRV))
		if err != nil {
			t.Error(err)
		}
		if g, e := len(addrs), 5; g != e {
			t.Errorf("len(addrs) = %d; want %d", g, e)
			t.Logf("addrs = %#v", addrs)
		}
	}
	// repack and unpack.
	data2, ok := msg.Pack()
	msg2 := new(dnsMsg)
	msg2.Unpack(data2)
	switch {
	case !ok:
		t.Error("failed to repack message")
	case !reflect.DeepEqual(msg, msg2):
		t.Error("repacked message differs from original")
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
		t.Fatal("unpacking packet failed")
	}
	msg.String() // exercise this code path
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

func TestDNSParseTXTReply(t *testing.T) {
	expectedTxt1 := "v=spf1 redirect=_spf.google.com"
	expectedTxt2 := "v=spf1 ip4:69.63.179.25 ip4:69.63.178.128/25 ip4:69.63.184.0/25 " +
		"ip4:66.220.144.128/25 ip4:66.220.155.0/24 " +
		"ip4:69.171.232.0/25 ip4:66.220.157.0/25 " +
		"ip4:69.171.244.0/24 mx -all"

	replies := []string{dnsTXTReply1, dnsTXTReply2}
	expectedTxts := []string{expectedTxt1, expectedTxt2}

	for i := range replies {
		data, err := hex.DecodeString(replies[i])
		if err != nil {
			t.Fatal(err)
		}

		msg := new(dnsMsg)
		ok := msg.Unpack(data)
		if !ok {
			t.Errorf("test %d: unpacking packet failed", i)
			continue
		}

		if len(msg.answer) != 1 {
			t.Errorf("test %d: len(rr.answer) = %d; want 1", i, len(msg.answer))
			continue
		}

		rr := msg.answer[0]
		rrTXT, ok := rr.(*dnsRR_TXT)
		if !ok {
			t.Errorf("test %d: answer[0] = %T; want *dnsRR_TXT", i, rr)
			continue
		}

		if rrTXT.Txt != expectedTxts[i] {
			t.Errorf("test %d: Txt = %s; want %s", i, rrTXT.Txt, expectedTxts[i])
		}
	}
}

func TestDNSParseTXTCorruptDataLengthReply(t *testing.T) {
	replies := []string{dnsTXTCorruptDataLengthReply1, dnsTXTCorruptDataLengthReply2}

	for i := range replies {
		data, err := hex.DecodeString(replies[i])
		if err != nil {
			t.Fatal(err)
		}

		msg := new(dnsMsg)
		ok := msg.Unpack(data)
		if ok {
			t.Errorf("test %d: expected to fail on unpacking corrupt packet", i)
		}
	}
}

func TestDNSParseTXTCorruptTXTLengthReply(t *testing.T) {
	replies := []string{dnsTXTCorruptTXTLengthReply1, dnsTXTCorruptTXTLengthReply2}

	for i := range replies {
		data, err := hex.DecodeString(replies[i])
		if err != nil {
			t.Fatal(err)
		}

		msg := new(dnsMsg)
		ok := msg.Unpack(data)
		// Unpacking should succeed, but we should just get the header.
		if !ok {
			t.Errorf("test %d: unpacking packet failed", i)
			continue
		}

		if len(msg.answer) != 1 {
			t.Errorf("test %d: len(rr.answer) = %d; want 1", i, len(msg.answer))
			continue
		}

		rr := msg.answer[0]
		if _, justHeader := rr.(*dnsRR_Header); !justHeader {
			t.Errorf("test %d: rr = %T; expected *dnsRR_Header", i, rr)
		}
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

// TXT reply with one <character-string>
const dnsTXTReply1 = "b3458180000100010004000505676d61696c03636f6d0000100001c00c001000010000012c00" +
	"201f763d737066312072656469726563743d5f7370662e676f6f676c652e636f6dc00" +
	"c0002000100025d4c000d036e733406676f6f676c65c012c00c0002000100025d4c00" +
	"06036e7331c057c00c0002000100025d4c0006036e7333c057c00c0002000100025d4" +
	"c0006036e7332c057c06c00010001000248b50004d8ef200ac09000010001000248b5" +
	"0004d8ef220ac07e00010001000248b50004d8ef240ac05300010001000248b50004d" +
	"8ef260a0000291000000000000000"

// TXT reply with more than one <character-string>.
// See https://tools.ietf.org/html/rfc1035#section-3.3.14
const dnsTXTReply2 = "a0a381800001000100020002045f7370660866616365626f6f6b03636f6d0000100001c00c0010000" +
	"100000e1000af7f763d73706631206970343a36392e36332e3137392e3235206970343a36392e" +
	"36332e3137382e3132382f3235206970343a36392e36332e3138342e302f3235206970343a363" +
	"62e3232302e3134342e3132382f3235206970343a36362e3232302e3135352e302f3234206970" +
	"343a36392e3137312e3233322e302f323520692e70343a36362e3232302e3135372e302f32352" +
	"06970343a36392e3137312e3234342e302f3234206d78202d616c6cc0110002000100025d1500" +
	"070161026e73c011c0110002000100025d1500040162c0ecc0ea0001000100025d15000445abe" +
	"f0cc0fd0001000100025d15000445abff0c"

// DataLength field should be sum of all TXT fields. In this case it's less.
const dnsTXTCorruptDataLengthReply1 = "a0a381800001000100020002045f7370660866616365626f6f6b03636f6d0000100001c00c0010000" +
	"100000e1000967f763d73706631206970343a36392e36332e3137392e3235206970343a36392e" +
	"36332e3137382e3132382f3235206970343a36392e36332e3138342e302f3235206970343a363" +
	"62e3232302e3134342e3132382f3235206970343a36362e3232302e3135352e302f3234206970" +
	"343a36392e3137312e3233322e302f323520692e70343a36362e3232302e3135372e302f32352" +
	"06970343a36392e3137312e3234342e302f3234206d78202d616c6cc0110002000100025d1500" +
	"070161026e73c011c0110002000100025d1500040162c0ecc0ea0001000100025d15000445abe" +
	"f0cc0fd0001000100025d15000445abff0c"

// Same as above but DataLength is more than sum of TXT fields.
const dnsTXTCorruptDataLengthReply2 = "a0a381800001000100020002045f7370660866616365626f6f6b03636f6d0000100001c00c0010000" +
	"100000e1001227f763d73706631206970343a36392e36332e3137392e3235206970343a36392e" +
	"36332e3137382e3132382f3235206970343a36392e36332e3138342e302f3235206970343a363" +
	"62e3232302e3134342e3132382f3235206970343a36362e3232302e3135352e302f3234206970" +
	"343a36392e3137312e3233322e302f323520692e70343a36362e3232302e3135372e302f32352" +
	"06970343a36392e3137312e3234342e302f3234206d78202d616c6cc0110002000100025d1500" +
	"070161026e73c011c0110002000100025d1500040162c0ecc0ea0001000100025d15000445abe" +
	"f0cc0fd0001000100025d15000445abff0c"

// TXT Length field is less than actual length.
const dnsTXTCorruptTXTLengthReply1 = "a0a381800001000100020002045f7370660866616365626f6f6b03636f6d0000100001c00c0010000" +
	"100000e1000af7f763d73706631206970343a36392e36332e3137392e3235206970343a36392e" +
	"36332e3137382e3132382f3235206970343a36392e36332e3138342e302f3235206970343a363" +
	"62e3232302e3134342e3132382f3235206970343a36362e3232302e3135352e302f3234206970" +
	"343a36392e3137312e3233322e302f323520691470343a36362e3232302e3135372e302f32352" +
	"06970343a36392e3137312e3234342e302f3234206d78202d616c6cc0110002000100025d1500" +
	"070161026e73c011c0110002000100025d1500040162c0ecc0ea0001000100025d15000445abe" +
	"f0cc0fd0001000100025d15000445abff0c"

// TXT Length field is more than actual length.
const dnsTXTCorruptTXTLengthReply2 = "a0a381800001000100020002045f7370660866616365626f6f6b03636f6d0000100001c00c0010000" +
	"100000e1000af7f763d73706631206970343a36392e36332e3137392e3235206970343a36392e" +
	"36332e3137382e3132382f3235206970343a36392e36332e3138342e302f3235206970343a363" +
	"62e3232302e3134342e3132382f3235206970343a36362e3232302e3135352e302f3234206970" +
	"343a36392e3137312e3233322e302f323520693370343a36362e3232302e3135372e302f32352" +
	"06970343a36392e3137312e3234342e302f3234206d78202d616c6cc0110002000100025d1500" +
	"070161026e73c011c0110002000100025d1500040162c0ecc0ea0001000100025d15000445abe" +
	"f0cc0fd0001000100025d15000445abff0c"
