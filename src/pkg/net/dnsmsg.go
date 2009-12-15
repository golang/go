// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// DNS packet assembly.  See RFC 1035.
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
// prefixed with _DNS_.  Perhaps put this in its own package later.

package net

import (
	"fmt"
	"os"
	"reflect"
)

// Packet formats

// Wire constants.
const (
	// valid _DNS_RR_Header.Rrtype and _DNS_Question.qtype
	_DNS_TypeA     = 1
	_DNS_TypeNS    = 2
	_DNS_TypeMD    = 3
	_DNS_TypeMF    = 4
	_DNS_TypeCNAME = 5
	_DNS_TypeSOA   = 6
	_DNS_TypeMB    = 7
	_DNS_TypeMG    = 8
	_DNS_TypeMR    = 9
	_DNS_TypeNULL  = 10
	_DNS_TypeWKS   = 11
	_DNS_TypePTR   = 12
	_DNS_TypeHINFO = 13
	_DNS_TypeMINFO = 14
	_DNS_TypeMX    = 15
	_DNS_TypeTXT   = 16

	// valid _DNS_Question.qtype only
	_DNS_TypeAXFR  = 252
	_DNS_TypeMAILB = 253
	_DNS_TypeMAILA = 254
	_DNS_TypeALL   = 255

	// valid _DNS_Question.qclass
	_DNS_ClassINET   = 1
	_DNS_ClassCSNET  = 2
	_DNS_ClassCHAOS  = 3
	_DNS_ClassHESIOD = 4
	_DNS_ClassANY    = 255

	// _DNS_Msg.rcode
	_DNS_RcodeSuccess        = 0
	_DNS_RcodeFormatError    = 1
	_DNS_RcodeServerFailure  = 2
	_DNS_RcodeNameError      = 3
	_DNS_RcodeNotImplemented = 4
	_DNS_RcodeRefused        = 5
)

// The wire format for the DNS packet header.
type __DNS_Header struct {
	Id                                 uint16
	Bits                               uint16
	Qdcount, Ancount, Nscount, Arcount uint16
}

const (
	// __DNS_Header.Bits
	_QR = 1 << 15 // query/response (response=1)
	_AA = 1 << 10 // authoritative
	_TC = 1 << 9  // truncated
	_RD = 1 << 8  // recursion desired
	_RA = 1 << 7  // recursion available
)

// DNS queries.
type _DNS_Question struct {
	Name   string "domain-name" // "domain-name" specifies encoding; see packers below
	Qtype  uint16
	Qclass uint16
}

// DNS responses (resource records).
// There are many types of messages,
// but they all share the same header.
type _DNS_RR_Header struct {
	Name     string "domain-name"
	Rrtype   uint16
	Class    uint16
	Ttl      uint32
	Rdlength uint16 // length of data after header
}

func (h *_DNS_RR_Header) Header() *_DNS_RR_Header {
	return h
}

type _DNS_RR interface {
	Header() *_DNS_RR_Header
}


// Specific DNS RR formats for each query type.

type _DNS_RR_CNAME struct {
	Hdr   _DNS_RR_Header
	Cname string "domain-name"
}

func (rr *_DNS_RR_CNAME) Header() *_DNS_RR_Header {
	return &rr.Hdr
}

type _DNS_RR_HINFO struct {
	Hdr _DNS_RR_Header
	Cpu string
	Os  string
}

func (rr *_DNS_RR_HINFO) Header() *_DNS_RR_Header {
	return &rr.Hdr
}

type _DNS_RR_MB struct {
	Hdr _DNS_RR_Header
	Mb  string "domain-name"
}

func (rr *_DNS_RR_MB) Header() *_DNS_RR_Header {
	return &rr.Hdr
}

type _DNS_RR_MG struct {
	Hdr _DNS_RR_Header
	Mg  string "domain-name"
}

func (rr *_DNS_RR_MG) Header() *_DNS_RR_Header {
	return &rr.Hdr
}

type _DNS_RR_MINFO struct {
	Hdr   _DNS_RR_Header
	Rmail string "domain-name"
	Email string "domain-name"
}

func (rr *_DNS_RR_MINFO) Header() *_DNS_RR_Header {
	return &rr.Hdr
}

type _DNS_RR_MR struct {
	Hdr _DNS_RR_Header
	Mr  string "domain-name"
}

func (rr *_DNS_RR_MR) Header() *_DNS_RR_Header {
	return &rr.Hdr
}

type _DNS_RR_MX struct {
	Hdr  _DNS_RR_Header
	Pref uint16
	Mx   string "domain-name"
}

func (rr *_DNS_RR_MX) Header() *_DNS_RR_Header {
	return &rr.Hdr
}

type _DNS_RR_NS struct {
	Hdr _DNS_RR_Header
	Ns  string "domain-name"
}

func (rr *_DNS_RR_NS) Header() *_DNS_RR_Header {
	return &rr.Hdr
}

type _DNS_RR_PTR struct {
	Hdr _DNS_RR_Header
	Ptr string "domain-name"
}

func (rr *_DNS_RR_PTR) Header() *_DNS_RR_Header {
	return &rr.Hdr
}

type _DNS_RR_SOA struct {
	Hdr     _DNS_RR_Header
	Ns      string "domain-name"
	Mbox    string "domain-name"
	Serial  uint32
	Refresh uint32
	Retry   uint32
	Expire  uint32
	Minttl  uint32
}

func (rr *_DNS_RR_SOA) Header() *_DNS_RR_Header {
	return &rr.Hdr
}

type _DNS_RR_TXT struct {
	Hdr _DNS_RR_Header
	Txt string // not domain name
}

func (rr *_DNS_RR_TXT) Header() *_DNS_RR_Header {
	return &rr.Hdr
}

type _DNS_RR_A struct {
	Hdr _DNS_RR_Header
	A   uint32 "ipv4"
}

func (rr *_DNS_RR_A) Header() *_DNS_RR_Header { return &rr.Hdr }


// Packing and unpacking.
//
// All the packers and unpackers take a (msg []byte, off int)
// and return (off1 int, ok bool).  If they return ok==false, they
// also return off1==len(msg), so that the next unpacker will
// also fail.  This lets us avoid checks of ok until the end of a
// packing sequence.

// Map of constructors for each RR wire type.
var rr_mk = map[int]func() _DNS_RR{
	_DNS_TypeCNAME: func() _DNS_RR { return new(_DNS_RR_CNAME) },
	_DNS_TypeHINFO: func() _DNS_RR { return new(_DNS_RR_HINFO) },
	_DNS_TypeMB: func() _DNS_RR { return new(_DNS_RR_MB) },
	_DNS_TypeMG: func() _DNS_RR { return new(_DNS_RR_MG) },
	_DNS_TypeMINFO: func() _DNS_RR { return new(_DNS_RR_MINFO) },
	_DNS_TypeMR: func() _DNS_RR { return new(_DNS_RR_MR) },
	_DNS_TypeMX: func() _DNS_RR { return new(_DNS_RR_MX) },
	_DNS_TypeNS: func() _DNS_RR { return new(_DNS_RR_NS) },
	_DNS_TypePTR: func() _DNS_RR { return new(_DNS_RR_PTR) },
	_DNS_TypeSOA: func() _DNS_RR { return new(_DNS_RR_SOA) },
	_DNS_TypeTXT: func() _DNS_RR { return new(_DNS_RR_TXT) },
	_DNS_TypeA: func() _DNS_RR { return new(_DNS_RR_A) },
}

// Pack a domain name s into msg[off:].
// Domain names are a sequence of counted strings
// split at the dots.  They end with a zero-length string.
func packDomainName(s string, msg []byte, off int) (off1 int, ok bool) {
	// Add trailing dot to canonicalize name.
	if n := len(s); n == 0 || s[n-1] != '.' {
		s += "."
	}

	// Each dot ends a segment of the name.
	// We trade each dot byte for a length byte.
	// There is also a trailing zero.
	// Check that we have all the space we need.
	tot := len(s) + 1
	if off+tot > len(msg) {
		return len(msg), false
	}

	// Emit sequence of counted strings, chopping at dots.
	begin := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '.' {
			if i-begin >= 1<<6 { // top two bits of length must be clear
				return len(msg), false
			}
			msg[off] = byte(i - begin)
			off++
			for j := begin; j < i; j++ {
				msg[off] = s[j]
				off++
			}
			begin = i + 1
		}
	}
	msg[off] = 0
	off++
	return off, true
}

// Unpack a domain name.
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
func unpackDomainName(msg []byte, off int) (s string, off1 int, ok bool) {
	s = ""
	ptr := 0 // number of pointers followed
Loop:
	for {
		if off >= len(msg) {
			return "", len(msg), false
		}
		c := int(msg[off])
		off++
		switch c & 0xC0 {
		case 0x00:
			if c == 0x00 {
				// end of name
				break Loop
			}
			// literal string
			if off+c > len(msg) {
				return "", len(msg), false
			}
			s += string(msg[off:off+c]) + "."
			off += c
		case 0xC0:
			// pointer to somewhere else in msg.
			// remember location after first ptr,
			// since that's how many bytes we consumed.
			// also, don't follow too many pointers --
			// maybe there's a loop.
			if off >= len(msg) {
				return "", len(msg), false
			}
			c1 := msg[off]
			off++
			if ptr == 0 {
				off1 = off
			}
			if ptr++; ptr > 10 {
				return "", len(msg), false
			}
			off = (c^0xC0)<<8 | int(c1)
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

// TODO(rsc): Move into generic library?
// Pack a reflect.StructValue into msg.  Struct members can only be uint16, uint32, string,
// and other (often anonymous) structs.
func packStructValue(val *reflect.StructValue, msg []byte, off int) (off1 int, ok bool) {
	for i := 0; i < val.NumField(); i++ {
		f := val.Type().(*reflect.StructType).Field(i)
		switch fv := val.Field(i).(type) {
		default:
			fmt.Fprintf(os.Stderr, "net: dns: unknown packing type %v", f.Type)
			return len(msg), false
		case *reflect.StructValue:
			off, ok = packStructValue(fv, msg, off)
		case *reflect.Uint16Value:
			i := fv.Get()
			if off+2 > len(msg) {
				return len(msg), false
			}
			msg[off] = byte(i >> 8)
			msg[off+1] = byte(i)
			off += 2
		case *reflect.Uint32Value:
			i := fv.Get()
			if off+4 > len(msg) {
				return len(msg), false
			}
			msg[off] = byte(i >> 24)
			msg[off+1] = byte(i >> 16)
			msg[off+2] = byte(i >> 8)
			msg[off+4] = byte(i)
			off += 4
		case *reflect.StringValue:
			// There are multiple string encodings.
			// The tag distinguishes ordinary strings from domain names.
			s := fv.Get()
			switch f.Tag {
			default:
				fmt.Fprintf(os.Stderr, "net: dns: unknown string tag %v", f.Tag)
				return len(msg), false
			case "domain-name":
				off, ok = packDomainName(s, msg, off)
				if !ok {
					return len(msg), false
				}
			case "":
				// Counted string: 1 byte length.
				if len(s) > 255 || off+1+len(s) > len(msg) {
					return len(msg), false
				}
				msg[off] = byte(len(s))
				off++
				for i := 0; i < len(s); i++ {
					msg[off+i] = s[i]
				}
				off += len(s)
			}
		}
	}
	return off, true
}

func structValue(any interface{}) *reflect.StructValue {
	return reflect.NewValue(any).(*reflect.PtrValue).Elem().(*reflect.StructValue)
}

func packStruct(any interface{}, msg []byte, off int) (off1 int, ok bool) {
	off, ok = packStructValue(structValue(any), msg, off)
	return off, ok
}

// TODO(rsc): Move into generic library?
// Unpack a reflect.StructValue from msg.
// Same restrictions as packStructValue.
func unpackStructValue(val *reflect.StructValue, msg []byte, off int) (off1 int, ok bool) {
	for i := 0; i < val.NumField(); i++ {
		f := val.Type().(*reflect.StructType).Field(i)
		switch fv := val.Field(i).(type) {
		default:
			fmt.Fprintf(os.Stderr, "net: dns: unknown packing type %v", f.Type)
			return len(msg), false
		case *reflect.StructValue:
			off, ok = unpackStructValue(fv, msg, off)
		case *reflect.Uint16Value:
			if off+2 > len(msg) {
				return len(msg), false
			}
			i := uint16(msg[off])<<8 | uint16(msg[off+1])
			fv.Set(i)
			off += 2
		case *reflect.Uint32Value:
			if off+4 > len(msg) {
				return len(msg), false
			}
			i := uint32(msg[off])<<24 | uint32(msg[off+1])<<16 | uint32(msg[off+2])<<8 | uint32(msg[off+3])
			fv.Set(i)
			off += 4
		case *reflect.StringValue:
			var s string
			switch f.Tag {
			default:
				fmt.Fprintf(os.Stderr, "net: dns: unknown string tag %v", f.Tag)
				return len(msg), false
			case "domain-name":
				s, off, ok = unpackDomainName(msg, off)
				if !ok {
					return len(msg), false
				}
			case "":
				if off >= len(msg) || off+1+int(msg[off]) > len(msg) {
					return len(msg), false
				}
				n := int(msg[off])
				off++
				b := make([]byte, n)
				for i := 0; i < n; i++ {
					b[i] = msg[off+i]
				}
				off += n
				s = string(b)
			}
			fv.Set(s)
		}
	}
	return off, true
}

func unpackStruct(any interface{}, msg []byte, off int) (off1 int, ok bool) {
	off, ok = unpackStructValue(structValue(any), msg, off)
	return off, ok
}

// Generic struct printer.
// Doesn't care about the string tag "domain-name",
// but does look for an "ipv4" tag on uint32 variables,
// printing them as IP addresses.
func printStructValue(val *reflect.StructValue) string {
	s := "{"
	for i := 0; i < val.NumField(); i++ {
		if i > 0 {
			s += ", "
		}
		f := val.Type().(*reflect.StructType).Field(i)
		if !f.Anonymous {
			s += f.Name + "="
		}
		fval := val.Field(i)
		if fv, ok := fval.(*reflect.StructValue); ok {
			s += printStructValue(fv)
		} else if fv, ok := fval.(*reflect.Uint32Value); ok && f.Tag == "ipv4" {
			i := fv.Get()
			s += IPv4(byte(i>>24), byte(i>>16), byte(i>>8), byte(i)).String()
		} else {
			s += fmt.Sprint(fval.Interface())
		}
	}
	s += "}"
	return s
}

func printStruct(any interface{}) string { return printStructValue(structValue(any)) }

// Resource record packer.
func packRR(rr _DNS_RR, msg []byte, off int) (off2 int, ok bool) {
	var off1 int
	// pack twice, once to find end of header
	// and again to find end of packet.
	// a bit inefficient but this doesn't need to be fast.
	// off1 is end of header
	// off2 is end of rr
	off1, ok = packStruct(rr.Header(), msg, off)
	off2, ok = packStruct(rr, msg, off)
	if !ok {
		return len(msg), false
	}
	// pack a third time; redo header with correct data length
	rr.Header().Rdlength = uint16(off2 - off1)
	packStruct(rr.Header(), msg, off)
	return off2, true
}

// Resource record unpacker.
func unpackRR(msg []byte, off int) (rr _DNS_RR, off1 int, ok bool) {
	// unpack just the header, to find the rr type and length
	var h _DNS_RR_Header
	off0 := off
	if off, ok = unpackStruct(&h, msg, off); !ok {
		return nil, len(msg), false
	}
	end := off + int(h.Rdlength)

	// make an rr of that type and re-unpack.
	// again inefficient but doesn't need to be fast.
	mk, known := rr_mk[int(h.Rrtype)]
	if !known {
		return &h, end, true
	}
	rr = mk()
	off, ok = unpackStruct(rr, msg, off0)
	if off != end {
		return &h, end, true
	}
	return rr, off, ok
}

// Usable representation of a DNS packet.

// A manually-unpacked version of (id, bits).
// This is in its own struct for easy printing.
type __DNS_Msg_Top struct {
	id                  uint16
	response            bool
	opcode              int
	authoritative       bool
	truncated           bool
	recursion_desired   bool
	recursion_available bool
	rcode               int
}

type _DNS_Msg struct {
	__DNS_Msg_Top
	question []_DNS_Question
	answer   []_DNS_RR
	ns       []_DNS_RR
	extra    []_DNS_RR
}


func (dns *_DNS_Msg) Pack() (msg []byte, ok bool) {
	var dh __DNS_Header

	// Convert convenient _DNS_Msg into wire-like __DNS_Header.
	dh.Id = dns.id
	dh.Bits = uint16(dns.opcode)<<11 | uint16(dns.rcode)
	if dns.recursion_available {
		dh.Bits |= _RA
	}
	if dns.recursion_desired {
		dh.Bits |= _RD
	}
	if dns.truncated {
		dh.Bits |= _TC
	}
	if dns.authoritative {
		dh.Bits |= _AA
	}
	if dns.response {
		dh.Bits |= _QR
	}

	// Prepare variable sized arrays.
	question := dns.question
	answer := dns.answer
	ns := dns.ns
	extra := dns.extra

	dh.Qdcount = uint16(len(question))
	dh.Ancount = uint16(len(answer))
	dh.Nscount = uint16(len(ns))
	dh.Arcount = uint16(len(extra))

	// Could work harder to calculate message size,
	// but this is far more than we need and not
	// big enough to hurt the allocator.
	msg = make([]byte, 2000)

	// Pack it in: header and then the pieces.
	off := 0
	off, ok = packStruct(&dh, msg, off)
	for i := 0; i < len(question); i++ {
		off, ok = packStruct(&question[i], msg, off)
	}
	for i := 0; i < len(answer); i++ {
		off, ok = packStruct(answer[i], msg, off)
	}
	for i := 0; i < len(ns); i++ {
		off, ok = packStruct(ns[i], msg, off)
	}
	for i := 0; i < len(extra); i++ {
		off, ok = packStruct(extra[i], msg, off)
	}
	if !ok {
		return nil, false
	}
	return msg[0:off], true
}

func (dns *_DNS_Msg) Unpack(msg []byte) bool {
	// Header.
	var dh __DNS_Header
	off := 0
	var ok bool
	if off, ok = unpackStruct(&dh, msg, off); !ok {
		return false
	}
	dns.id = dh.Id
	dns.response = (dh.Bits & _QR) != 0
	dns.opcode = int(dh.Bits>>11) & 0xF
	dns.authoritative = (dh.Bits & _AA) != 0
	dns.truncated = (dh.Bits & _TC) != 0
	dns.recursion_desired = (dh.Bits & _RD) != 0
	dns.recursion_available = (dh.Bits & _RA) != 0
	dns.rcode = int(dh.Bits & 0xF)

	// Arrays.
	dns.question = make([]_DNS_Question, dh.Qdcount)
	dns.answer = make([]_DNS_RR, dh.Ancount)
	dns.ns = make([]_DNS_RR, dh.Nscount)
	dns.extra = make([]_DNS_RR, dh.Arcount)

	for i := 0; i < len(dns.question); i++ {
		off, ok = unpackStruct(&dns.question[i], msg, off)
	}
	for i := 0; i < len(dns.answer); i++ {
		dns.answer[i], off, ok = unpackRR(msg, off)
	}
	for i := 0; i < len(dns.ns); i++ {
		dns.ns[i], off, ok = unpackRR(msg, off)
	}
	for i := 0; i < len(dns.extra); i++ {
		dns.extra[i], off, ok = unpackRR(msg, off)
	}
	if !ok {
		return false
	}
	//	if off != len(msg) {
	//		println("extra bytes in dns packet", off, "<", len(msg));
	//	}
	return true
}

func (dns *_DNS_Msg) String() string {
	s := "DNS: " + printStruct(&dns.__DNS_Msg_Top) + "\n"
	if len(dns.question) > 0 {
		s += "-- Questions\n"
		for i := 0; i < len(dns.question); i++ {
			s += printStruct(&dns.question[i]) + "\n"
		}
	}
	if len(dns.answer) > 0 {
		s += "-- Answers\n"
		for i := 0; i < len(dns.answer); i++ {
			s += printStruct(dns.answer[i]) + "\n"
		}
	}
	if len(dns.ns) > 0 {
		s += "-- Name servers\n"
		for i := 0; i < len(dns.ns); i++ {
			s += printStruct(dns.ns[i]) + "\n"
		}
	}
	if len(dns.extra) > 0 {
		s += "-- Extra\n"
		for i := 0; i < len(dns.extra); i++ {
			s += printStruct(dns.extra[i]) + "\n"
		}
	}
	return s
}
