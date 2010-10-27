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
// TODO(rsc):  There are enough names defined in this file that they're all
// prefixed with dns.  Perhaps put this in its own package later.

package net

import (
	"fmt"
	"os"
	"reflect"
)

// Packet formats

// Wire constants.
const (
	// valid dnsRR_Header.Rrtype and dnsQuestion.qtype
	dnsTypeA     = 1
	dnsTypeNS    = 2
	dnsTypeMD    = 3
	dnsTypeMF    = 4
	dnsTypeCNAME = 5
	dnsTypeSOA   = 6
	dnsTypeMB    = 7
	dnsTypeMG    = 8
	dnsTypeMR    = 9
	dnsTypeNULL  = 10
	dnsTypeWKS   = 11
	dnsTypePTR   = 12
	dnsTypeHINFO = 13
	dnsTypeMINFO = 14
	dnsTypeMX    = 15
	dnsTypeTXT   = 16
	dnsTypeSRV   = 33

	// valid dnsQuestion.qtype only
	dnsTypeAXFR  = 252
	dnsTypeMAILB = 253
	dnsTypeMAILA = 254
	dnsTypeALL   = 255

	// valid dnsQuestion.qclass
	dnsClassINET   = 1
	dnsClassCSNET  = 2
	dnsClassCHAOS  = 3
	dnsClassHESIOD = 4
	dnsClassANY    = 255

	// dnsMsg.rcode
	dnsRcodeSuccess        = 0
	dnsRcodeFormatError    = 1
	dnsRcodeServerFailure  = 2
	dnsRcodeNameError      = 3
	dnsRcodeNotImplemented = 4
	dnsRcodeRefused        = 5
)

// The wire format for the DNS packet header.
type dnsHeader struct {
	Id                                 uint16
	Bits                               uint16
	Qdcount, Ancount, Nscount, Arcount uint16
}

const (
	// dnsHeader.Bits
	_QR = 1 << 15 // query/response (response=1)
	_AA = 1 << 10 // authoritative
	_TC = 1 << 9  // truncated
	_RD = 1 << 8  // recursion desired
	_RA = 1 << 7  // recursion available
)

// DNS queries.
type dnsQuestion struct {
	Name   string "domain-name" // "domain-name" specifies encoding; see packers below
	Qtype  uint16
	Qclass uint16
}

// DNS responses (resource records).
// There are many types of messages,
// but they all share the same header.
type dnsRR_Header struct {
	Name     string "domain-name"
	Rrtype   uint16
	Class    uint16
	Ttl      uint32
	Rdlength uint16 // length of data after header
}

func (h *dnsRR_Header) Header() *dnsRR_Header {
	return h
}

type dnsRR interface {
	Header() *dnsRR_Header
}


// Specific DNS RR formats for each query type.

type dnsRR_CNAME struct {
	Hdr   dnsRR_Header
	Cname string "domain-name"
}

func (rr *dnsRR_CNAME) Header() *dnsRR_Header {
	return &rr.Hdr
}

type dnsRR_HINFO struct {
	Hdr dnsRR_Header
	Cpu string
	Os  string
}

func (rr *dnsRR_HINFO) Header() *dnsRR_Header {
	return &rr.Hdr
}

type dnsRR_MB struct {
	Hdr dnsRR_Header
	Mb  string "domain-name"
}

func (rr *dnsRR_MB) Header() *dnsRR_Header {
	return &rr.Hdr
}

type dnsRR_MG struct {
	Hdr dnsRR_Header
	Mg  string "domain-name"
}

func (rr *dnsRR_MG) Header() *dnsRR_Header {
	return &rr.Hdr
}

type dnsRR_MINFO struct {
	Hdr   dnsRR_Header
	Rmail string "domain-name"
	Email string "domain-name"
}

func (rr *dnsRR_MINFO) Header() *dnsRR_Header {
	return &rr.Hdr
}

type dnsRR_MR struct {
	Hdr dnsRR_Header
	Mr  string "domain-name"
}

func (rr *dnsRR_MR) Header() *dnsRR_Header {
	return &rr.Hdr
}

type dnsRR_MX struct {
	Hdr  dnsRR_Header
	Pref uint16
	Mx   string "domain-name"
}

func (rr *dnsRR_MX) Header() *dnsRR_Header {
	return &rr.Hdr
}

type dnsRR_NS struct {
	Hdr dnsRR_Header
	Ns  string "domain-name"
}

func (rr *dnsRR_NS) Header() *dnsRR_Header {
	return &rr.Hdr
}

type dnsRR_PTR struct {
	Hdr dnsRR_Header
	Ptr string "domain-name"
}

func (rr *dnsRR_PTR) Header() *dnsRR_Header {
	return &rr.Hdr
}

type dnsRR_SOA struct {
	Hdr     dnsRR_Header
	Ns      string "domain-name"
	Mbox    string "domain-name"
	Serial  uint32
	Refresh uint32
	Retry   uint32
	Expire  uint32
	Minttl  uint32
}

func (rr *dnsRR_SOA) Header() *dnsRR_Header {
	return &rr.Hdr
}

type dnsRR_TXT struct {
	Hdr dnsRR_Header
	Txt string // not domain name
}

func (rr *dnsRR_TXT) Header() *dnsRR_Header {
	return &rr.Hdr
}

type dnsRR_SRV struct {
	Hdr      dnsRR_Header
	Priority uint16
	Weight   uint16
	Port     uint16
	Target   string "domain-name"
}

func (rr *dnsRR_SRV) Header() *dnsRR_Header {
	return &rr.Hdr
}

type dnsRR_A struct {
	Hdr dnsRR_Header
	A   uint32 "ipv4"
}

func (rr *dnsRR_A) Header() *dnsRR_Header { return &rr.Hdr }


// Packing and unpacking.
//
// All the packers and unpackers take a (msg []byte, off int)
// and return (off1 int, ok bool).  If they return ok==false, they
// also return off1==len(msg), so that the next unpacker will
// also fail.  This lets us avoid checks of ok until the end of a
// packing sequence.

// Map of constructors for each RR wire type.
var rr_mk = map[int]func() dnsRR{
	dnsTypeCNAME: func() dnsRR { return new(dnsRR_CNAME) },
	dnsTypeHINFO: func() dnsRR { return new(dnsRR_HINFO) },
	dnsTypeMB:    func() dnsRR { return new(dnsRR_MB) },
	dnsTypeMG:    func() dnsRR { return new(dnsRR_MG) },
	dnsTypeMINFO: func() dnsRR { return new(dnsRR_MINFO) },
	dnsTypeMR:    func() dnsRR { return new(dnsRR_MR) },
	dnsTypeMX:    func() dnsRR { return new(dnsRR_MX) },
	dnsTypeNS:    func() dnsRR { return new(dnsRR_NS) },
	dnsTypePTR:   func() dnsRR { return new(dnsRR_PTR) },
	dnsTypeSOA:   func() dnsRR { return new(dnsRR_SOA) },
	dnsTypeTXT:   func() dnsRR { return new(dnsRR_TXT) },
	dnsTypeSRV:   func() dnsRR { return new(dnsRR_SRV) },
	dnsTypeA:     func() dnsRR { return new(dnsRR_A) },
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
		BadType:
			fmt.Fprintf(os.Stderr, "net: dns: unknown packing type %v", f.Type)
			return len(msg), false
		case *reflect.StructValue:
			off, ok = packStructValue(fv, msg, off)
		case *reflect.UintValue:
			i := fv.Get()
			switch fv.Type().Kind() {
			default:
				goto BadType
			case reflect.Uint16:
				if off+2 > len(msg) {
					return len(msg), false
				}
				msg[off] = byte(i >> 8)
				msg[off+1] = byte(i)
				off += 2
			case reflect.Uint32:
				if off+4 > len(msg) {
					return len(msg), false
				}
				msg[off] = byte(i >> 24)
				msg[off+1] = byte(i >> 16)
				msg[off+2] = byte(i >> 8)
				msg[off+3] = byte(i)
				off += 4
			}
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
				off += copy(msg[off:], s)
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
		BadType:
			fmt.Fprintf(os.Stderr, "net: dns: unknown packing type %v", f.Type)
			return len(msg), false
		case *reflect.StructValue:
			off, ok = unpackStructValue(fv, msg, off)
		case *reflect.UintValue:
			switch fv.Type().Kind() {
			default:
				goto BadType
			case reflect.Uint16:
				if off+2 > len(msg) {
					return len(msg), false
				}
				i := uint16(msg[off])<<8 | uint16(msg[off+1])
				fv.Set(uint64(i))
				off += 2
			case reflect.Uint32:
				if off+4 > len(msg) {
					return len(msg), false
				}
				i := uint32(msg[off])<<24 | uint32(msg[off+1])<<16 | uint32(msg[off+2])<<8 | uint32(msg[off+3])
				fv.Set(uint64(i))
				off += 4
			}
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
		} else if fv, ok := fval.(*reflect.UintValue); ok && f.Tag == "ipv4" {
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
func packRR(rr dnsRR, msg []byte, off int) (off2 int, ok bool) {
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
func unpackRR(msg []byte, off int) (rr dnsRR, off1 int, ok bool) {
	// unpack just the header, to find the rr type and length
	var h dnsRR_Header
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
type dnsMsgHdr struct {
	id                  uint16
	response            bool
	opcode              int
	authoritative       bool
	truncated           bool
	recursion_desired   bool
	recursion_available bool
	rcode               int
}

type dnsMsg struct {
	dnsMsgHdr
	question []dnsQuestion
	answer   []dnsRR
	ns       []dnsRR
	extra    []dnsRR
}


func (dns *dnsMsg) Pack() (msg []byte, ok bool) {
	var dh dnsHeader

	// Convert convenient dnsMsg into wire-like dnsHeader.
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
		off, ok = packRR(answer[i], msg, off)
	}
	for i := 0; i < len(ns); i++ {
		off, ok = packRR(ns[i], msg, off)
	}
	for i := 0; i < len(extra); i++ {
		off, ok = packRR(extra[i], msg, off)
	}
	if !ok {
		return nil, false
	}
	return msg[0:off], true
}

func (dns *dnsMsg) Unpack(msg []byte) bool {
	// Header.
	var dh dnsHeader
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
	dns.question = make([]dnsQuestion, dh.Qdcount)
	dns.answer = make([]dnsRR, dh.Ancount)
	dns.ns = make([]dnsRR, dh.Nscount)
	dns.extra = make([]dnsRR, dh.Arcount)

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

func (dns *dnsMsg) String() string {
	s := "DNS: " + printStruct(&dns.dnsMsgHdr) + "\n"
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
