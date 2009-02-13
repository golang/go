// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// DNS packet assembly.
//
// This is intended to support name resolution during net.Dial.
// It doesn't have to be blazing fast.
//
// Rather than write the usual handful of routines to pack and
// unpack every message that can appear on the wire, we use
// reflection to write a generic pack/unpack for structs and then
// use it.  Thus, if in the future we need to define new message
// structs, no new pack/unpack/printing code needs to be written.
//
// The first half of this file defines the DNS message formats.
// The second half implements the conversion to and from wire format.
// A few of the structure elements have string tags to aid the
// generic pack/unpack routines.
//
// TODO(rsc)  There are enough names defined in this file that they're all
// prefixed with DNS_.  Perhaps put this in its own package later.

package net

import (
	"fmt";
	"os";
	"reflect";
)

// _Packet formats

// Wire constants.
const (
	// valid DNS_RR_Header.rrtype and DNS_Question.qtype
	DNS_TypeA = 1;
	DNS_TypeNS = 2;
	DNS_TypeMD = 3;
	DNS_TypeMF = 4;
	DNS_TypeCNAME = 5;
	DNS_TypeSOA = 6;
	DNS_TypeMB = 7;
	DNS_TypeMG = 8;
	DNS_TypeMR = 9;
	DNS_TypeNULL = 10;
	DNS_TypeWKS = 11;
	DNS_TypePTR = 12;
	DNS_TypeHINFO = 13;
	DNS_TypeMINFO = 14;
	DNS_TypeMX = 15;
	DNS_TypeTXT = 16;

	// valid DNS_Question.qtype only
	DNS_TypeAXFR = 252;
	DNS_TypeMAILB = 253;
	DNS_TypeMAILA = 254;
	DNS_TypeALL = 255;

	// valid DNS_Question.qclass
	DNS_ClassINET = 1;
	DNS_ClassCSNET = 2;
	DNS_ClassCHAOS = 3;
	DNS_ClassHESIOD = 4;
	DNS_ClassANY = 255;

	// DNS_Msg.rcode
	DNS_RcodeSuccess = 0;
	DNS_RcodeFormatError = 1;
	DNS_RcodeServerFailure = 2;
	DNS_RcodeNameError = 3;
	DNS_RcodeNotImplemented = 4;
	DNS_RcodeRefused = 5;
)

// The wire format for the DNS packet header.
type _DNS_Header struct {
	id uint16;
	bits uint16;
	qdcount, ancount, nscount, arcount uint16;
}

const (
	// _DNS_Header.bits
	_QR = 1<<15;	// query/response (response=1)
	_AA = 1<<10;	// authoritative
	_TC = 1<<9;	// truncated
	_RD = 1<<8;	// recursion desired
	_RA = 1<<7;	// recursion available
)

// DNS queries.
type DNS_Question struct {
	name string "domain-name";	// "domain-name" specifies encoding; see packers below
	qtype uint16;
	qclass uint16;
}

// DNS responses (resource records).
// There are many types of messages,
// but they all share the same header.
type DNS_RR_Header struct {
	name string "domain-name";
	rrtype uint16;
	class uint16;
	ttl uint32;
	rdlength uint16;	// length of data after header
}

func (h *DNS_RR_Header) Header() *DNS_RR_Header {
	return h
}

type DNS_RR interface {
	Header() *DNS_RR_Header
}


// Specific DNS RR formats for each query type.

type DNS_RR_CNAME struct {
	DNS_RR_Header;
	cname string "domain-name";
}

type DNS_RR_HINFO struct {
	DNS_RR_Header;
	cpu string;
	os string;
}

type DNS_RR_MB struct {
	DNS_RR_Header;
	mb string "domain-name";
}

type DNS_RR_MG struct {
	DNS_RR_Header;
	mg string "domain-name";
}

type DNS_RR_MINFO struct {
	DNS_RR_Header;
	rmail string "domain-name";
	email string "domain-name";
}

type DNS_RR_MR struct {
	DNS_RR_Header;
	mr string "domain-name";
}

type DNS_RR_MX struct {
	DNS_RR_Header;
	pref uint16;
	mx string "domain-name";
}

type DNS_RR_NS struct {
	DNS_RR_Header;
	ns string "domain-name";
}

type DNS_RR_PTR struct {
	DNS_RR_Header;
	ptr string "domain-name";
}

type DNS_RR_SOA struct {
	DNS_RR_Header;
	ns string "domain-name";
	mbox string "domain-name";
	serial uint32;
	refresh uint32;
	retry uint32;
	expire uint32;
	minttl uint32;
}

type DNS_RR_TXT struct {
	DNS_RR_Header;
	txt string;	// not domain name
}

type DNS_RR_A struct {
	DNS_RR_Header;
	a uint32 "ipv4";
}


// _Packing and unpacking.
//
// All the packers and unpackers take a (msg []byte, off int)
// and return (off1 int, ok bool).  If they return ok==false, they
// also return off1==len(msg), so that the next unpacker will
// also fail.  This lets us avoid checks of ok until the end of a
// packing sequence.

// Map of constructors for each RR wire type.
var rr_mk = map[int] func()DNS_RR (
	DNS_TypeCNAME: func() DNS_RR { return new(DNS_RR_CNAME) },
	DNS_TypeHINFO: func() DNS_RR { return new(DNS_RR_HINFO) },
	DNS_TypeMB: func() DNS_RR { return new(DNS_RR_MB) },
	DNS_TypeMG: func() DNS_RR { return new(DNS_RR_MG) },
	DNS_TypeMINFO: func() DNS_RR { return new(DNS_RR_MINFO) },
	DNS_TypeMR: func() DNS_RR { return new(DNS_RR_MR) },
	DNS_TypeMX: func() DNS_RR { return new(DNS_RR_MX) },
	DNS_TypeNS: func() DNS_RR { return new(DNS_RR_NS) },
	DNS_TypePTR: func() DNS_RR { return new(DNS_RR_PTR) },
	DNS_TypeSOA: func() DNS_RR { return new(DNS_RR_SOA) },
	DNS_TypeTXT: func() DNS_RR { return new(DNS_RR_TXT) },
	DNS_TypeA: func() DNS_RR { return new(DNS_RR_A) },
)

// _Pack a domain name s into msg[off:].
// Domain names are a sequence of counted strings
// split at the dots.  They end with a zero-length string.
func _PackDomainName(s string, msg []byte, off int) (off1 int, ok bool) {
	// Add trailing dot to canonicalize name.
	if n := len(s); n == 0 || s[n-1] != '.' {
		s += ".";
	}

	// Each dot ends a segment of the name.
	// We trade each dot byte for a length byte.
	// There is also a trailing zero.
	// Check that we have all the space we need.
	tot := len(s) + 1;
	if off+tot > len(msg) {
		return len(msg), false
	}

	// Emit sequence of counted strings, chopping at dots.
	begin := 0;
	for i := 0; i < len(s); i++ {
		if s[i] == '.' {
			if i - begin >= 1<<6 { // top two bits of length must be clear
				return len(msg), false
			}
			msg[off] = byte(i - begin);
			off++;
			for j := begin; j < i; j++ {
				msg[off] = s[j];
				off++;
			}
			begin = i+1;
		}
	}
	msg[off] = 0;
	off++;
	return off, true
}

// _Unpack a domain name.
// In addition to the simple sequences of counted strings above,
// domain names are allowed to refer to strings elsewhere in the
// packet, to avoid repeating common suffixes when returning
// many entries in a single domain.  The pointers are marked
// by a length byte with the top two bits set.  Ignoring those
// two bits, that byte and the next give a 14 bit offset from msg[0]
// where we should pick up the trail.
// Note that if we jump elsewhere in the packet,
// we return off1 == the offset after the first pointer we found,
// which is where the next record will start.
// In theory, the pointers are only allowed to jump backward.
// We let them jump anywhere and stop jumping after a while.
func _UnpackDomainName(msg []byte, off int) (s string, off1 int, ok bool) {
	s = "";
	ptr := 0;	// number of pointers followed
Loop:
	for {
		if off >= len(msg) {
			return "", len(msg), false
		}
		c := int(msg[off]);
		off++;
		switch c&0xC0 {
		case 0x00:
			if c == 0x00 {
				// end of name
				break Loop
			}
			// literal string
			if off+c > len(msg) {
				return "", len(msg), false
			}
			s += string(msg[off:off+c]) + ".";
			off += c;
		case 0xC0:
			// pointer to somewhere else in msg.
			// remember location after first ptr,
			// since that's how many bytes we consumed.
			// also, don't follow too many pointers --
			// maybe there's a loop.
			if off >= len(msg) {
				return "", len(msg), false
			}
			c1 := msg[off];
			off++;
			if ptr == 0 {
				off1 = off
			}
			if ptr++; ptr > 10 {
				return "", len(msg), false
			}
			off = (c^0xC0)<<8 | int(c1);
		default:
			// 0x80 and 0x40 are reserved
			return "", len(msg), false
		}
	}
	if ptr == 0 {
		off1 = off
	}
	return s, off1, true
}

// _Pack a reflect.StructValue into msg.  Struct members can only be uint16, uint32, string,
// and other (often anonymous) structs.
func _PackStructValue(val reflect.StructValue, msg []byte, off int) (off1 int, ok bool) {
	for i := 0; i < val.Len(); i++ {
		fld := val.Field(i);
		name, typ, tag, xxx := val.Type().(reflect.StructType).Field(i);
		switch fld.Kind() {
		default:
			fmt.Fprintf(os.Stderr, "net: dns: unknown packing type %v", fld.Type());
			return len(msg), false;
		case reflect.StructKind:
			off, ok = _PackStructValue(fld.(reflect.StructValue), msg, off);
		case reflect.Uint16Kind:
			i := fld.(reflect.Uint16Value).Get();
			if off+2 > len(msg) {
				return len(msg), false
			}
			msg[off] = byte(i>>8);
			msg[off+1] = byte(i);
			off += 2;
		case reflect.Uint32Kind:
			i := fld.(reflect.Uint32Value).Get();
			if off+4 > len(msg) {
				return len(msg), false
			}
			msg[off] = byte(i>>24);
			msg[off+1] = byte(i>>16);
			msg[off+2] = byte(i>>8);
			msg[off+4] = byte(i);
			off += 4;
		case reflect.StringKind:
			// There are multiple string encodings.
			// The tag distinguishes ordinary strings from domain names.
			s := fld.(reflect.StringValue).Get();
			switch tag {
			default:
				fmt.Fprintf(os.Stderr, "net: dns: unknown string tag %v", tag);
				return len(msg), false;
			case "domain-name":
				off, ok = _PackDomainName(s, msg, off);
				if !ok {
					return len(msg), false
				}
			case "":
				// Counted string: 1 byte length.
				if len(s) > 255 || off + 1 + len(s) > len(msg) {
					return len(msg), false
				}
				msg[off] = byte(len(s));
				off++;
				for i := 0; i < len(s); i++ {
					msg[off+i] = s[i];
				}
				off += len(s);
			}
		}
	}
	return off, true
}

func _PackStruct(any interface{}, msg []byte, off int) (off1 int, ok bool) {
	val := reflect.NewValue(any).(reflect.PtrValue).Sub().(reflect.StructValue);
	off, ok = _PackStructValue(val, msg, off);
	return off, ok
}

// _Unpack a reflect.StructValue from msg.
// Same restrictions as _PackStructValue.
func _UnpackStructValue(val reflect.StructValue, msg []byte, off int) (off1 int, ok bool) {
	for i := 0; i < val.Len(); i++ {
		name, typ, tag, xxx := val.Type().(reflect.StructType).Field(i);
		fld := val.Field(i);
		switch fld.Kind() {
		default:
			fmt.Fprintf(os.Stderr, "net: dns: unknown packing type %v", fld.Type());
			return len(msg), false;
		case reflect.StructKind:
			off, ok = _UnpackStructValue(fld.(reflect.StructValue), msg, off);
		case reflect.Uint16Kind:
			if off+2 > len(msg) {
				return len(msg), false
			}
			i := uint16(msg[off])<<8 | uint16(msg[off+1]);
			fld.(reflect.Uint16Value).Set(i);
			off += 2;
		case reflect.Uint32Kind:
			if off+4 > len(msg) {
				return len(msg), false
			}
			i := uint32(msg[off])<<24 | uint32(msg[off+1])<<16 | uint32(msg[off+2])<<8 | uint32(msg[off+3]);
			fld.(reflect.Uint32Value).Set(i);
			off += 4;
		case reflect.StringKind:
			var s string;
			switch tag {
			default:
				fmt.Fprintf(os.Stderr, "net: dns: unknown string tag %v", tag);
				return len(msg), false;
			case "domain-name":
				s, off, ok = _UnpackDomainName(msg, off);
				if !ok {
					return len(msg), false
				}
			case "":
				if off >= len(msg) || off+1+int(msg[off]) > len(msg) {
					return len(msg), false
				}
				n := int(msg[off]);
				off++;
				b := make([]byte, n);
				for i := 0; i < n; i++ {
					b[i] = msg[off+i];
				}
				off += n;
				s = string(b);
			}
			fld.(reflect.StringValue).Set(s);
		}
	}
	return off, true
}

func _UnpackStruct(any interface{}, msg []byte, off int) (off1 int, ok bool) {
	val := reflect.NewValue(any).(reflect.PtrValue).Sub().(reflect.StructValue);
	off, ok = _UnpackStructValue(val, msg, off);
	return off, ok
}

// Generic struct printer.
// Doesn't care about the string tag "domain-name",
// but does look for an "ipv4" tag on uint32 variables,
// printing them as IP addresses.
func _PrintStructValue(val reflect.StructValue) string {
	s := "{";
	for i := 0; i < val.Len(); i++ {
		if i > 0 {
			s += ", ";
		}
		name, typ, tag, xxx := val.Type().(reflect.StructType).Field(i);
		fld := val.Field(i);
		if name != "" && name != "?" {	// BUG? Shouldn't the reflect library hide "?" ?
			s += name + "=";
		}
		kind := fld.Kind();
		switch {
		case kind == reflect.StructKind:
			s += _PrintStructValue(fld.(reflect.StructValue));
		case kind == reflect.Uint32Kind && tag == "ipv4":
			i := fld.(reflect.Uint32Value).Get();
			s += fmt.Sprintf("%d.%d.%d.%d", (i>>24)&0xFF, (i>>16)&0xFF, (i>>8)&0xFF, i&0xFF);
		default:
			s += fmt.Sprint(fld.Interface())
		}
	}
	s += "}";
	return s;
}

func _PrintStruct(any interface{}) string {
	val := reflect.NewValue(any).(reflect.PtrValue).Sub().(reflect.StructValue);
	s := _PrintStructValue(val);
	return s
}

// Resource record packer.
func _PackRR(rr DNS_RR, msg []byte, off int) (off2 int, ok bool) {
	var off1 int;
	// pack twice, once to find end of header
	// and again to find end of packet.
	// a bit inefficient but this doesn't need to be fast.
	// off1 is end of header
	// off2 is end of rr
	off1, ok = _PackStruct(rr.Header(), msg, off);
	off2, ok = _PackStruct(rr, msg, off);
	if !ok {
		return len(msg), false
	}
	// pack a third time; redo header with correct data length
	rr.Header().rdlength = uint16(off2 - off1);
	_PackStruct(rr.Header(), msg, off);
	return off2, true
}

// Resource record unpacker.
func _UnpackRR(msg []byte, off int) (rr DNS_RR, off1 int, ok bool) {
	// unpack just the header, to find the rr type and length
	var h DNS_RR_Header;
	off0 := off;
	if off, ok = _UnpackStruct(&h, msg, off); !ok {
		return nil, len(msg), false
	}
	end := off+int(h.rdlength);

	// make an rr of that type and re-unpack.
	// again inefficient but doesn't need to be fast.
	mk, known := rr_mk[int(h.rrtype)];
	if !known {
		return &h, end, true
	}
	rr = mk();
	off, ok = _UnpackStruct(rr, msg, off0);
	if off != end {
		return &h, end, true
	}
	return rr, off, ok
}

// Usable representation of a DNS packet.

// A manually-unpacked version of (id, bits).
// This is in its own struct for easy printing.
type _DNS_Msg_Top struct {
	id uint16;
	response bool;
	opcode int;
	authoritative bool;
	truncated bool;
	recursion_desired bool;
	recursion_available bool;
	rcode int;
}

type DNS_Msg struct {
	_DNS_Msg_Top;
	question []DNS_Question;
	answer []DNS_RR;
	ns []DNS_RR;
	extra []DNS_RR;
}


func (dns *DNS_Msg) Pack() (msg []byte, ok bool) {
	var dh _DNS_Header;

	// Convert convenient DNS_Msg into wire-like _DNS_Header.
	dh.id = dns.id;
	dh.bits = uint16(dns.opcode)<<11 | uint16(dns.rcode);
	if dns.recursion_available {
		dh.bits |= _RA;
	}
	if dns.recursion_desired {
		dh.bits |= _RD;
	}
	if dns.truncated {
		dh.bits |= _TC;
	}
	if dns.authoritative {
		dh.bits |= _AA;
	}
	if dns.response {
		dh.bits |= _QR;
	}

	// Prepare variable sized arrays.
	question := dns.question;
	answer := dns.answer;
	ns := dns.ns;
	extra := dns.extra;

	dh.qdcount = uint16(len(question));
	dh.ancount = uint16(len(answer));
	dh.nscount = uint16(len(ns));
	dh.arcount = uint16(len(extra));

	// Could work harder to calculate message size,
	// but this is far more than we need and not
	// big enough to hurt the allocator.
	msg = make([]byte, 2000);

	// _Pack it in: header and then the pieces.
	off := 0;
	off, ok = _PackStruct(&dh, msg, off);
	for i := 0; i < len(question); i++ {
		off, ok = _PackStruct(&question[i], msg, off);
	}
	for i := 0; i < len(answer); i++ {
		off, ok = _PackStruct(answer[i], msg, off);
	}
	for i := 0; i < len(ns); i++ {
		off, ok = _PackStruct(ns[i], msg, off);
	}
	for i := 0; i < len(extra); i++ {
		off, ok = _PackStruct(extra[i], msg, off);
	}
	if !ok {
		return nil, false
	}
	return msg[0:off], true
}

func (dns *DNS_Msg) Unpack(msg []byte) bool {
	// Header.
	var dh _DNS_Header;
	off := 0;
	var ok bool;
	if off, ok = _UnpackStruct(&dh, msg, off); !ok {
		return false
	}
	dns.id = dh.id;
	dns.response = (dh.bits & _QR) != 0;
	dns.opcode = int(dh.bits >> 11) & 0xF;
	dns.authoritative = (dh.bits & _AA) != 0;
	dns.truncated = (dh.bits & _TC) != 0;
	dns.recursion_desired = (dh.bits & _RD) != 0;
	dns.recursion_available = (dh.bits & _RA) != 0;
	dns.rcode = int(dh.bits & 0xF);

	// Arrays.
	dns.question = make([]DNS_Question, dh.qdcount);
	dns.answer = make([]DNS_RR, dh.ancount);
	dns.ns = make([]DNS_RR, dh.nscount);
	dns.extra = make([]DNS_RR, dh.arcount);

	for i := 0; i < len(dns.question); i++ {
		off, ok = _UnpackStruct(&dns.question[i], msg, off);
	}
	for i := 0; i < len(dns.answer); i++ {
		dns.answer[i], off, ok = _UnpackRR(msg, off);
	}
	for i := 0; i < len(dns.ns); i++ {
		dns.ns[i], off, ok = _UnpackRR(msg, off);
	}
	for i := 0; i < len(dns.extra); i++ {
		dns.extra[i], off, ok = _UnpackRR(msg, off);
	}
	if !ok {
		return false
	}
//	if off != len(msg) {
//		println("extra bytes in dns packet", off, "<", len(msg));
//	}
	return true
}

func (dns *DNS_Msg) String() string {
	s := "DNS: "+_PrintStruct(&dns._DNS_Msg_Top)+"\n";
	if len(dns.question) > 0 {
		s += "-- Questions\n";
		for i := 0; i < len(dns.question); i++ {
			s += _PrintStruct(&dns.question[i])+"\n";
		}
	}
	if len(dns.answer) > 0 {
		s += "-- Answers\n";
		for i := 0; i < len(dns.answer); i++ {
			s += _PrintStruct(dns.answer[i])+"\n";
		}
	}
	if len(dns.ns) > 0 {
		s += "-- Name servers\n";
		for i := 0; i < len(dns.ns); i++ {
			s += _PrintStruct(dns.ns[i])+"\n";
		}
	}
	if len(dns.extra) > 0 {
		s += "-- Extra\n";
		for i := 0; i < len(dns.extra); i++ {
			s += _PrintStruct(dns.extra[i])+"\n";
		}
	}
	return s;
}
