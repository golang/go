// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package dnsmessage provides a mostly RFC 1035 compliant implementation of
// DNS message packing and unpacking.
//
// The package also supports messages with Extension Mechanisms for DNS
// (EDNS(0)) as defined in RFC 6891.
//
// This implementation is designed to minimize heap allocations and avoid
// unnecessary packing and unpacking as much as possible.
package dnsmessage

import (
	"errors"
)

// Message formats
//
// To add a new Resource Record type:
// 1. Create Resource Record types
//   1.1. Add a Type constant named "Type<name>"
//   1.2. Add the corresponding entry to the typeNames map
//   1.3. Add a [ResourceBody] implementation named "<name>Resource"
// 2. Implement packing
//   2.1. Implement Builder.<name>Resource()
// 3. Implement unpacking
//   3.1. Add the unpacking code to unpackResourceBody()
//   3.2. Implement Parser.<name>Resource()

// A Type is the type of a DNS Resource Record, as defined in the [IANA registry].
//
// [IANA registry]: https://www.iana.org/assignments/dns-parameters/dns-parameters.xhtml#dns-parameters-4
type Type uint16

const (
	// ResourceHeader.Type and Question.Type
	TypeA     Type = 1
	TypeNS    Type = 2
	TypeCNAME Type = 5
	TypeSOA   Type = 6
	TypePTR   Type = 12
	TypeMX    Type = 15
	TypeTXT   Type = 16
	TypeAAAA  Type = 28
	TypeSRV   Type = 33
	TypeOPT   Type = 41
	TypeSVCB  Type = 64
	TypeHTTPS Type = 65

	// Question.Type
	TypeWKS   Type = 11
	TypeHINFO Type = 13
	TypeMINFO Type = 14
	TypeAXFR  Type = 252
	TypeALL   Type = 255
)

var typeNames = map[Type]string{
	TypeA:     "TypeA",
	TypeNS:    "TypeNS",
	TypeCNAME: "TypeCNAME",
	TypeSOA:   "TypeSOA",
	TypePTR:   "TypePTR",
	TypeMX:    "TypeMX",
	TypeTXT:   "TypeTXT",
	TypeAAAA:  "TypeAAAA",
	TypeSRV:   "TypeSRV",
	TypeOPT:   "TypeOPT",
	TypeSVCB:  "TypeSVCB",
	TypeHTTPS: "TypeHTTPS",
	TypeWKS:   "TypeWKS",
	TypeHINFO: "TypeHINFO",
	TypeMINFO: "TypeMINFO",
	TypeAXFR:  "TypeAXFR",
	TypeALL:   "TypeALL",
}

// String implements fmt.Stringer.String.
func (t Type) String() string {
	if n, ok := typeNames[t]; ok {
		return n
	}
	return printUint16(uint16(t))
}

// GoString implements fmt.GoStringer.GoString.
func (t Type) GoString() string {
	if n, ok := typeNames[t]; ok {
		return "dnsmessage." + n
	}
	return printUint16(uint16(t))
}

// A Class is a type of network.
type Class uint16

const (
	// ResourceHeader.Class and Question.Class
	ClassINET   Class = 1
	ClassCSNET  Class = 2
	ClassCHAOS  Class = 3
	ClassHESIOD Class = 4

	// Question.Class
	ClassANY Class = 255
)

var classNames = map[Class]string{
	ClassINET:   "ClassINET",
	ClassCSNET:  "ClassCSNET",
	ClassCHAOS:  "ClassCHAOS",
	ClassHESIOD: "ClassHESIOD",
	ClassANY:    "ClassANY",
}

// String implements fmt.Stringer.String.
func (c Class) String() string {
	if n, ok := classNames[c]; ok {
		return n
	}
	return printUint16(uint16(c))
}

// GoString implements fmt.GoStringer.GoString.
func (c Class) GoString() string {
	if n, ok := classNames[c]; ok {
		return "dnsmessage." + n
	}
	return printUint16(uint16(c))
}

// An OpCode is a DNS operation code.
type OpCode uint16

// GoString implements fmt.GoStringer.GoString.
func (o OpCode) GoString() string {
	return printUint16(uint16(o))
}

// An RCode is a DNS response status code.
type RCode uint16

// Header.RCode values.
const (
	RCodeSuccess        RCode = 0 // NoError
	RCodeFormatError    RCode = 1 // FormErr
	RCodeServerFailure  RCode = 2 // ServFail
	RCodeNameError      RCode = 3 // NXDomain
	RCodeNotImplemented RCode = 4 // NotImp
	RCodeRefused        RCode = 5 // Refused
)

var rCodeNames = map[RCode]string{
	RCodeSuccess:        "RCodeSuccess",
	RCodeFormatError:    "RCodeFormatError",
	RCodeServerFailure:  "RCodeServerFailure",
	RCodeNameError:      "RCodeNameError",
	RCodeNotImplemented: "RCodeNotImplemented",
	RCodeRefused:        "RCodeRefused",
}

// String implements fmt.Stringer.String.
func (r RCode) String() string {
	if n, ok := rCodeNames[r]; ok {
		return n
	}
	return printUint16(uint16(r))
}

// GoString implements fmt.GoStringer.GoString.
func (r RCode) GoString() string {
	if n, ok := rCodeNames[r]; ok {
		return "dnsmessage." + n
	}
	return printUint16(uint16(r))
}

func printPaddedUint8(i uint8) string {
	b := byte(i)
	return string([]byte{
		b/100 + '0',
		b/10%10 + '0',
		b%10 + '0',
	})
}

func printUint8Bytes(buf []byte, i uint8) []byte {
	b := byte(i)
	if i >= 100 {
		buf = append(buf, b/100+'0')
	}
	if i >= 10 {
		buf = append(buf, b/10%10+'0')
	}
	return append(buf, b%10+'0')
}

func printByteSlice(b []byte) string {
	if len(b) == 0 {
		return ""
	}
	buf := make([]byte, 0, 5*len(b))
	buf = printUint8Bytes(buf, uint8(b[0]))
	for _, n := range b[1:] {
		buf = append(buf, ',', ' ')
		buf = printUint8Bytes(buf, uint8(n))
	}
	return string(buf)
}

const hexDigits = "0123456789abcdef"

func printString(str []byte) string {
	buf := make([]byte, 0, len(str))
	for i := 0; i < len(str); i++ {
		c := str[i]
		if c == '.' || c == '-' || c == ' ' ||
			'A' <= c && c <= 'Z' ||
			'a' <= c && c <= 'z' ||
			'0' <= c && c <= '9' {
			buf = append(buf, c)
			continue
		}

		upper := c >> 4
		lower := (c << 4) >> 4
		buf = append(
			buf,
			'\\',
			'x',
			hexDigits[upper],
			hexDigits[lower],
		)
	}
	return string(buf)
}

func printUint16(i uint16) string {
	return printUint32(uint32(i))
}

func printUint32(i uint32) string {
	// Max value is 4294967295.
	buf := make([]byte, 10)
	for b, d := buf, uint32(1000000000); d > 0; d /= 10 {
		b[0] = byte(i/d%10 + '0')
		if b[0] == '0' && len(b) == len(buf) && len(buf) > 1 {
			buf = buf[1:]
		}
		b = b[1:]
		i %= d
	}
	return string(buf)
}

func printBool(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

var (
	// ErrNotStarted indicates that the prerequisite information isn't
	// available yet because the previous records haven't been appropriately
	// parsed, skipped or finished.
	ErrNotStarted = errors.New("parsing/packing of this type isn't available yet")

	// ErrSectionDone indicated that all records in the section have been
	// parsed or finished.
	ErrSectionDone = errors.New("parsing/packing of this section has completed")

	errBaseLen            = errors.New("insufficient data for base length type")
	errCalcLen            = errors.New("insufficient data for calculated length type")
	errReserved           = errors.New("segment prefix is reserved")
	errTooManyPtr         = errors.New("too many pointers (>10)")
	errInvalidPtr         = errors.New("invalid pointer")
	errInvalidName        = errors.New("invalid dns name")
	errNilResouceBody     = errors.New("nil resource body")
	errResourceLen        = errors.New("insufficient data for resource body length")
	errSegTooLong         = errors.New("segment length too long")
	errNameTooLong        = errors.New("name too long")
	errZeroSegLen         = errors.New("zero length segment")
	errResTooLong         = errors.New("resource length too long")
	errTooManyQuestions   = errors.New("too many Questions to pack (>65535)")
	errTooManyAnswers     = errors.New("too many Answers to pack (>65535)")
	errTooManyAuthorities = errors.New("too many Authorities to pack (>65535)")
	errTooManyAdditionals = errors.New("too many Additionals to pack (>65535)")
	errNonCanonicalName   = errors.New("name is not in canonical format (it must end with a .)")
	errStringTooLong      = errors.New("character string exceeds maximum length (255)")
	errParamOutOfOrder    = errors.New("parameter out of order")
	errTooLongSVCBValue   = errors.New("value too long (>65535 bytes)")
)

// Internal constants.
const (
	// packStartingCap is the default initial buffer size allocated during
	// packing.
	//
	// The starting capacity doesn't matter too much, but most DNS responses
	// Will be <= 512 bytes as it is the limit for DNS over UDP.
	packStartingCap = 512

	// uint16Len is the length (in bytes) of a uint16.
	uint16Len = 2

	// uint32Len is the length (in bytes) of a uint32.
	uint32Len = 4

	// headerLen is the length (in bytes) of a DNS header.
	//
	// A header is comprised of 6 uint16s and no padding.
	headerLen = 6 * uint16Len
)

type nestedError struct {
	// s is the current level's error message.
	s string

	// err is the nested error.
	err error
}

// nestedError implements error.Error.
func (e *nestedError) Error() string {
	return e.s + ": " + e.err.Error()
}

// Header is a representation of a DNS message header.
type Header struct {
	ID                 uint16
	Response           bool
	OpCode             OpCode
	Authoritative      bool
	Truncated          bool
	RecursionDesired   bool
	RecursionAvailable bool
	AuthenticData      bool
	CheckingDisabled   bool
	RCode              RCode
}

func (m *Header) pack() (id uint16, bits uint16) {
	id = m.ID
	bits = uint16(m.OpCode)<<11 | uint16(m.RCode)
	if m.RecursionAvailable {
		bits |= headerBitRA
	}
	if m.RecursionDesired {
		bits |= headerBitRD
	}
	if m.Truncated {
		bits |= headerBitTC
	}
	if m.Authoritative {
		bits |= headerBitAA
	}
	if m.Response {
		bits |= headerBitQR
	}
	if m.AuthenticData {
		bits |= headerBitAD
	}
	if m.CheckingDisabled {
		bits |= headerBitCD
	}
	return
}

// GoString implements fmt.GoStringer.GoString.
func (m *Header) GoString() string {
	return "dnsmessage.Header{" +
		"ID: " + printUint16(m.ID) + ", " +
		"Response: " + printBool(m.Response) + ", " +
		"OpCode: " + m.OpCode.GoString() + ", " +
		"Authoritative: " + printBool(m.Authoritative) + ", " +
		"Truncated: " + printBool(m.Truncated) + ", " +
		"RecursionDesired: " + printBool(m.RecursionDesired) + ", " +
		"RecursionAvailable: " + printBool(m.RecursionAvailable) + ", " +
		"AuthenticData: " + printBool(m.AuthenticData) + ", " +
		"CheckingDisabled: " + printBool(m.CheckingDisabled) + ", " +
		"RCode: " + m.RCode.GoString() + "}"
}

// Message is a representation of a DNS message.
type Message struct {
	Header
	Questions   []Question
	Answers     []Resource
	Authorities []Resource
	Additionals []Resource
}

type section uint8

const (
	sectionNotStarted section = iota
	sectionHeader
	sectionQuestions
	sectionAnswers
	sectionAuthorities
	sectionAdditionals
	sectionDone

	headerBitQR = 1 << 15 // query/response (response=1)
	headerBitAA = 1 << 10 // authoritative
	headerBitTC = 1 << 9  // truncated
	headerBitRD = 1 << 8  // recursion desired
	headerBitRA = 1 << 7  // recursion available
	headerBitAD = 1 << 5  // authentic data
	headerBitCD = 1 << 4  // checking disabled
)

var sectionNames = map[section]string{
	sectionHeader:      "header",
	sectionQuestions:   "Question",
	sectionAnswers:     "Answer",
	sectionAuthorities: "Authority",
	sectionAdditionals: "Additional",
}

// header is the wire format for a DNS message header.
type header struct {
	id          uint16
	bits        uint16
	questions   uint16
	answers     uint16
	authorities uint16
	additionals uint16
}

func (h *header) count(sec section) uint16 {
	switch sec {
	case sectionQuestions:
		return h.questions
	case sectionAnswers:
		return h.answers
	case sectionAuthorities:
		return h.authorities
	case sectionAdditionals:
		return h.additionals
	}
	return 0
}

// pack appends the wire format of the header to msg.
func (h *header) pack(msg []byte) []byte {
	msg = packUint16(msg, h.id)
	msg = packUint16(msg, h.bits)
	msg = packUint16(msg, h.questions)
	msg = packUint16(msg, h.answers)
	msg = packUint16(msg, h.authorities)
	return packUint16(msg, h.additionals)
}

func (h *header) unpack(msg []byte, off int) (int, error) {
	newOff := off
	var err error
	if h.id, newOff, err = unpackUint16(msg, newOff); err != nil {
		return off, &nestedError{"id", err}
	}
	if h.bits, newOff, err = unpackUint16(msg, newOff); err != nil {
		return off, &nestedError{"bits", err}
	}
	if h.questions, newOff, err = unpackUint16(msg, newOff); err != nil {
		return off, &nestedError{"questions", err}
	}
	if h.answers, newOff, err = unpackUint16(msg, newOff); err != nil {
		return off, &nestedError{"answers", err}
	}
	if h.authorities, newOff, err = unpackUint16(msg, newOff); err != nil {
		return off, &nestedError{"authorities", err}
	}
	if h.additionals, newOff, err = unpackUint16(msg, newOff); err != nil {
		return off, &nestedError{"additionals", err}
	}
	return newOff, nil
}

func (h *header) header() Header {
	return Header{
		ID:                 h.id,
		Response:           (h.bits & headerBitQR) != 0,
		OpCode:             OpCode(h.bits>>11) & 0xF,
		Authoritative:      (h.bits & headerBitAA) != 0,
		Truncated:          (h.bits & headerBitTC) != 0,
		RecursionDesired:   (h.bits & headerBitRD) != 0,
		RecursionAvailable: (h.bits & headerBitRA) != 0,
		AuthenticData:      (h.bits & headerBitAD) != 0,
		CheckingDisabled:   (h.bits & headerBitCD) != 0,
		RCode:              RCode(h.bits & 0xF),
	}
}

// A Resource is a DNS resource record.
type Resource struct {
	Header ResourceHeader
	Body   ResourceBody
}

func (r *Resource) GoString() string {
	return "dnsmessage.Resource{" +
		"Header: " + r.Header.GoString() +
		", Body: &" + r.Body.GoString() +
		"}"
}

// A ResourceBody is a DNS resource record minus the header.
type ResourceBody interface {
	// pack packs a Resource except for its header.
	pack(msg []byte, compression map[string]uint16, compressionOff int) ([]byte, error)

	// realType returns the actual type of the Resource. This is used to
	// fill in the header Type field.
	realType() Type

	// GoString implements fmt.GoStringer.GoString.
	GoString() string
}

// pack appends the wire format of the Resource to msg.
func (r *Resource) pack(msg []byte, compression map[string]uint16, compressionOff int) ([]byte, error) {
	if r.Body == nil {
		return msg, errNilResouceBody
	}
	oldMsg := msg
	r.Header.Type = r.Body.realType()
	msg, lenOff, err := r.Header.pack(msg, compression, compressionOff)
	if err != nil {
		return msg, &nestedError{"ResourceHeader", err}
	}
	preLen := len(msg)
	msg, err = r.Body.pack(msg, compression, compressionOff)
	if err != nil {
		return msg, &nestedError{"content", err}
	}
	if err := r.Header.fixLen(msg, lenOff, preLen); err != nil {
		return oldMsg, err
	}
	return msg, nil
}

// A Parser allows incrementally parsing a DNS message.
//
// When parsing is started, the Header is parsed. Next, each Question can be
// either parsed or skipped. Alternatively, all Questions can be skipped at
// once. When all Questions have been parsed, attempting to parse Questions
// will return the [ErrSectionDone] error.
// After all Questions have been either parsed or skipped, all
// Answers, Authorities and Additionals can be either parsed or skipped in the
// same way, and each type of Resource must be fully parsed or skipped before
// proceeding to the next type of Resource.
//
// Parser is safe to copy to preserve the parsing state.
//
// Note that there is no requirement to fully skip or parse the message.
type Parser struct {
	msg    []byte
	header header

	section         section
	off             int
	index           int
	resHeaderValid  bool
	resHeaderOffset int
	resHeaderType   Type
	resHeaderLength uint16
}

// Start parses the header and enables the parsing of Questions.
func (p *Parser) Start(msg []byte) (Header, error) {
	if p.msg != nil {
		*p = Parser{}
	}
	p.msg = msg
	var err error
	if p.off, err = p.header.unpack(msg, 0); err != nil {
		return Header{}, &nestedError{"unpacking header", err}
	}
	p.section = sectionQuestions
	return p.header.header(), nil
}

func (p *Parser) checkAdvance(sec section) error {
	if p.section < sec {
		return ErrNotStarted
	}
	if p.section > sec {
		return ErrSectionDone
	}
	p.resHeaderValid = false
	if p.index == int(p.header.count(sec)) {
		p.index = 0
		p.section++
		return ErrSectionDone
	}
	return nil
}

func (p *Parser) resource(sec section) (Resource, error) {
	var r Resource
	var err error
	r.Header, err = p.resourceHeader(sec)
	if err != nil {
		return r, err
	}
	p.resHeaderValid = false
	r.Body, p.off, err = unpackResourceBody(p.msg, p.off, r.Header)
	if err != nil {
		return Resource{}, &nestedError{"unpacking " + sectionNames[sec], err}
	}
	p.index++
	return r, nil
}

func (p *Parser) resourceHeader(sec section) (ResourceHeader, error) {
	if p.resHeaderValid {
		p.off = p.resHeaderOffset
	}

	if err := p.checkAdvance(sec); err != nil {
		return ResourceHeader{}, err
	}
	var hdr ResourceHeader
	off, err := hdr.unpack(p.msg, p.off)
	if err != nil {
		return ResourceHeader{}, err
	}
	p.resHeaderValid = true
	p.resHeaderOffset = p.off
	p.resHeaderType = hdr.Type
	p.resHeaderLength = hdr.Length
	p.off = off
	return hdr, nil
}

func (p *Parser) skipResource(sec section) error {
	if p.resHeaderValid && p.section == sec {
		newOff := p.off + int(p.resHeaderLength)
		if newOff > len(p.msg) {
			return errResourceLen
		}
		p.off = newOff
		p.resHeaderValid = false
		p.index++
		return nil
	}
	if err := p.checkAdvance(sec); err != nil {
		return err
	}
	var err error
	p.off, err = skipResource(p.msg, p.off)
	if err != nil {
		return &nestedError{"skipping: " + sectionNames[sec], err}
	}
	p.index++
	return nil
}

// Question parses a single Question.
func (p *Parser) Question() (Question, error) {
	if err := p.checkAdvance(sectionQuestions); err != nil {
		return Question{}, err
	}
	var name Name
	off, err := name.unpack(p.msg, p.off)
	if err != nil {
		return Question{}, &nestedError{"unpacking Question.Name", err}
	}
	typ, off, err := unpackType(p.msg, off)
	if err != nil {
		return Question{}, &nestedError{"unpacking Question.Type", err}
	}
	class, off, err := unpackClass(p.msg, off)
	if err != nil {
		return Question{}, &nestedError{"unpacking Question.Class", err}
	}
	p.off = off
	p.index++
	return Question{name, typ, class}, nil
}

// AllQuestions parses all Questions.
func (p *Parser) AllQuestions() ([]Question, error) {
	// Multiple questions are valid according to the spec,
	// but servers don't actually support them. There will
	// be at most one question here.
	//
	// Do not pre-allocate based on info in p.header, since
	// the data is untrusted.
	qs := []Question{}
	for {
		q, err := p.Question()
		if err == ErrSectionDone {
			return qs, nil
		}
		if err != nil {
			return nil, err
		}
		qs = append(qs, q)
	}
}

// SkipQuestion skips a single Question.
func (p *Parser) SkipQuestion() error {
	if err := p.checkAdvance(sectionQuestions); err != nil {
		return err
	}
	off, err := skipName(p.msg, p.off)
	if err != nil {
		return &nestedError{"skipping Question Name", err}
	}
	if off, err = skipType(p.msg, off); err != nil {
		return &nestedError{"skipping Question Type", err}
	}
	if off, err = skipClass(p.msg, off); err != nil {
		return &nestedError{"skipping Question Class", err}
	}
	p.off = off
	p.index++
	return nil
}

// SkipAllQuestions skips all Questions.
func (p *Parser) SkipAllQuestions() error {
	for {
		if err := p.SkipQuestion(); err == ErrSectionDone {
			return nil
		} else if err != nil {
			return err
		}
	}
}

// AnswerHeader parses a single Answer ResourceHeader.
func (p *Parser) AnswerHeader() (ResourceHeader, error) {
	return p.resourceHeader(sectionAnswers)
}

// Answer parses a single Answer Resource.
func (p *Parser) Answer() (Resource, error) {
	return p.resource(sectionAnswers)
}

// AllAnswers parses all Answer Resources.
func (p *Parser) AllAnswers() ([]Resource, error) {
	// The most common query is for A/AAAA, which usually returns
	// a handful of IPs.
	//
	// Pre-allocate up to a certain limit, since p.header is
	// untrusted data.
	n := int(p.header.answers)
	if n > 20 {
		n = 20
	}
	as := make([]Resource, 0, n)
	for {
		a, err := p.Answer()
		if err == ErrSectionDone {
			return as, nil
		}
		if err != nil {
			return nil, err
		}
		as = append(as, a)
	}
}

// SkipAnswer skips a single Answer Resource.
//
// It does not perform a complete validation of the resource header, which means
// it may return a nil error when the [AnswerHeader] would actually return an error.
func (p *Parser) SkipAnswer() error {
	return p.skipResource(sectionAnswers)
}

// SkipAllAnswers skips all Answer Resources.
func (p *Parser) SkipAllAnswers() error {
	for {
		if err := p.SkipAnswer(); err == ErrSectionDone {
			return nil
		} else if err != nil {
			return err
		}
	}
}

// AuthorityHeader parses a single Authority ResourceHeader.
func (p *Parser) AuthorityHeader() (ResourceHeader, error) {
	return p.resourceHeader(sectionAuthorities)
}

// Authority parses a single Authority Resource.
func (p *Parser) Authority() (Resource, error) {
	return p.resource(sectionAuthorities)
}

// AllAuthorities parses all Authority Resources.
func (p *Parser) AllAuthorities() ([]Resource, error) {
	// Authorities contains SOA in case of NXDOMAIN and friends,
	// otherwise it is empty.
	//
	// Pre-allocate up to a certain limit, since p.header is
	// untrusted data.
	n := int(p.header.authorities)
	if n > 10 {
		n = 10
	}
	as := make([]Resource, 0, n)
	for {
		a, err := p.Authority()
		if err == ErrSectionDone {
			return as, nil
		}
		if err != nil {
			return nil, err
		}
		as = append(as, a)
	}
}

// SkipAuthority skips a single Authority Resource.
//
// It does not perform a complete validation of the resource header, which means
// it may return a nil error when the [AuthorityHeader] would actually return an error.
func (p *Parser) SkipAuthority() error {
	return p.skipResource(sectionAuthorities)
}

// SkipAllAuthorities skips all Authority Resources.
func (p *Parser) SkipAllAuthorities() error {
	for {
		if err := p.SkipAuthority(); err == ErrSectionDone {
			return nil
		} else if err != nil {
			return err
		}
	}
}

// AdditionalHeader parses a single Additional ResourceHeader.
func (p *Parser) AdditionalHeader() (ResourceHeader, error) {
	return p.resourceHeader(sectionAdditionals)
}

// Additional parses a single Additional Resource.
func (p *Parser) Additional() (Resource, error) {
	return p.resource(sectionAdditionals)
}

// AllAdditionals parses all Additional Resources.
func (p *Parser) AllAdditionals() ([]Resource, error) {
	// Additionals usually contain OPT, and sometimes A/AAAA
	// glue records.
	//
	// Pre-allocate up to a certain limit, since p.header is
	// untrusted data.
	n := int(p.header.additionals)
	if n > 10 {
		n = 10
	}
	as := make([]Resource, 0, n)
	for {
		a, err := p.Additional()
		if err == ErrSectionDone {
			return as, nil
		}
		if err != nil {
			return nil, err
		}
		as = append(as, a)
	}
}

// SkipAdditional skips a single Additional Resource.
//
// It does not perform a complete validation of the resource header, which means
// it may return a nil error when the [AdditionalHeader] would actually return an error.
func (p *Parser) SkipAdditional() error {
	return p.skipResource(sectionAdditionals)
}

// SkipAllAdditionals skips all Additional Resources.
func (p *Parser) SkipAllAdditionals() error {
	for {
		if err := p.SkipAdditional(); err == ErrSectionDone {
			return nil
		} else if err != nil {
			return err
		}
	}
}

// CNAMEResource parses a single CNAMEResource.
//
// One of the XXXHeader methods must have been called before calling this
// method.
func (p *Parser) CNAMEResource() (CNAMEResource, error) {
	if !p.resHeaderValid || p.resHeaderType != TypeCNAME {
		return CNAMEResource{}, ErrNotStarted
	}
	r, err := unpackCNAMEResource(p.msg, p.off)
	if err != nil {
		return CNAMEResource{}, err
	}
	p.off += int(p.resHeaderLength)
	p.resHeaderValid = false
	p.index++
	return r, nil
}

// MXResource parses a single MXResource.
//
// One of the XXXHeader methods must have been called before calling this
// method.
func (p *Parser) MXResource() (MXResource, error) {
	if !p.resHeaderValid || p.resHeaderType != TypeMX {
		return MXResource{}, ErrNotStarted
	}
	r, err := unpackMXResource(p.msg, p.off)
	if err != nil {
		return MXResource{}, err
	}
	p.off += int(p.resHeaderLength)
	p.resHeaderValid = false
	p.index++
	return r, nil
}

// NSResource parses a single NSResource.
//
// One of the XXXHeader methods must have been called before calling this
// method.
func (p *Parser) NSResource() (NSResource, error) {
	if !p.resHeaderValid || p.resHeaderType != TypeNS {
		return NSResource{}, ErrNotStarted
	}
	r, err := unpackNSResource(p.msg, p.off)
	if err != nil {
		return NSResource{}, err
	}
	p.off += int(p.resHeaderLength)
	p.resHeaderValid = false
	p.index++
	return r, nil
}

// PTRResource parses a single PTRResource.
//
// One of the XXXHeader methods must have been called before calling this
// method.
func (p *Parser) PTRResource() (PTRResource, error) {
	if !p.resHeaderValid || p.resHeaderType != TypePTR {
		return PTRResource{}, ErrNotStarted
	}
	r, err := unpackPTRResource(p.msg, p.off)
	if err != nil {
		return PTRResource{}, err
	}
	p.off += int(p.resHeaderLength)
	p.resHeaderValid = false
	p.index++
	return r, nil
}

// SOAResource parses a single SOAResource.
//
// One of the XXXHeader methods must have been called before calling this
// method.
func (p *Parser) SOAResource() (SOAResource, error) {
	if !p.resHeaderValid || p.resHeaderType != TypeSOA {
		return SOAResource{}, ErrNotStarted
	}
	r, err := unpackSOAResource(p.msg, p.off)
	if err != nil {
		return SOAResource{}, err
	}
	p.off += int(p.resHeaderLength)
	p.resHeaderValid = false
	p.index++
	return r, nil
}

// TXTResource parses a single TXTResource.
//
// One of the XXXHeader methods must have been called before calling this
// method.
func (p *Parser) TXTResource() (TXTResource, error) {
	if !p.resHeaderValid || p.resHeaderType != TypeTXT {
		return TXTResource{}, ErrNotStarted
	}
	r, err := unpackTXTResource(p.msg, p.off, p.resHeaderLength)
	if err != nil {
		return TXTResource{}, err
	}
	p.off += int(p.resHeaderLength)
	p.resHeaderValid = false
	p.index++
	return r, nil
}

// SRVResource parses a single SRVResource.
//
// One of the XXXHeader methods must have been called before calling this
// method.
func (p *Parser) SRVResource() (SRVResource, error) {
	if !p.resHeaderValid || p.resHeaderType != TypeSRV {
		return SRVResource{}, ErrNotStarted
	}
	r, err := unpackSRVResource(p.msg, p.off)
	if err != nil {
		return SRVResource{}, err
	}
	p.off += int(p.resHeaderLength)
	p.resHeaderValid = false
	p.index++
	return r, nil
}

// AResource parses a single AResource.
//
// One of the XXXHeader methods must have been called before calling this
// method.
func (p *Parser) AResource() (AResource, error) {
	if !p.resHeaderValid || p.resHeaderType != TypeA {
		return AResource{}, ErrNotStarted
	}
	r, err := unpackAResource(p.msg, p.off)
	if err != nil {
		return AResource{}, err
	}
	p.off += int(p.resHeaderLength)
	p.resHeaderValid = false
	p.index++
	return r, nil
}

// AAAAResource parses a single AAAAResource.
//
// One of the XXXHeader methods must have been called before calling this
// method.
func (p *Parser) AAAAResource() (AAAAResource, error) {
	if !p.resHeaderValid || p.resHeaderType != TypeAAAA {
		return AAAAResource{}, ErrNotStarted
	}
	r, err := unpackAAAAResource(p.msg, p.off)
	if err != nil {
		return AAAAResource{}, err
	}
	p.off += int(p.resHeaderLength)
	p.resHeaderValid = false
	p.index++
	return r, nil
}

// OPTResource parses a single OPTResource.
//
// One of the XXXHeader methods must have been called before calling this
// method.
func (p *Parser) OPTResource() (OPTResource, error) {
	if !p.resHeaderValid || p.resHeaderType != TypeOPT {
		return OPTResource{}, ErrNotStarted
	}
	r, err := unpackOPTResource(p.msg, p.off, p.resHeaderLength)
	if err != nil {
		return OPTResource{}, err
	}
	p.off += int(p.resHeaderLength)
	p.resHeaderValid = false
	p.index++
	return r, nil
}

// UnknownResource parses a single UnknownResource.
//
// One of the XXXHeader methods must have been called before calling this
// method.
func (p *Parser) UnknownResource() (UnknownResource, error) {
	if !p.resHeaderValid {
		return UnknownResource{}, ErrNotStarted
	}
	r, err := unpackUnknownResource(p.resHeaderType, p.msg, p.off, p.resHeaderLength)
	if err != nil {
		return UnknownResource{}, err
	}
	p.off += int(p.resHeaderLength)
	p.resHeaderValid = false
	p.index++
	return r, nil
}

// Unpack parses a full Message.
func (m *Message) Unpack(msg []byte) error {
	var p Parser
	var err error
	if m.Header, err = p.Start(msg); err != nil {
		return err
	}
	if m.Questions, err = p.AllQuestions(); err != nil {
		return err
	}
	if m.Answers, err = p.AllAnswers(); err != nil {
		return err
	}
	if m.Authorities, err = p.AllAuthorities(); err != nil {
		return err
	}
	if m.Additionals, err = p.AllAdditionals(); err != nil {
		return err
	}
	return nil
}

// Pack packs a full Message.
func (m *Message) Pack() ([]byte, error) {
	return m.AppendPack(make([]byte, 0, packStartingCap))
}

// AppendPack is like Pack but appends the full Message to b and returns the
// extended buffer.
func (m *Message) AppendPack(b []byte) ([]byte, error) {
	// Validate the lengths. It is very unlikely that anyone will try to
	// pack more than 65535 of any particular type, but it is possible and
	// we should fail gracefully.
	if len(m.Questions) > int(^uint16(0)) {
		return nil, errTooManyQuestions
	}
	if len(m.Answers) > int(^uint16(0)) {
		return nil, errTooManyAnswers
	}
	if len(m.Authorities) > int(^uint16(0)) {
		return nil, errTooManyAuthorities
	}
	if len(m.Additionals) > int(^uint16(0)) {
		return nil, errTooManyAdditionals
	}

	var h header
	h.id, h.bits = m.Header.pack()

	h.questions = uint16(len(m.Questions))
	h.answers = uint16(len(m.Answers))
	h.authorities = uint16(len(m.Authorities))
	h.additionals = uint16(len(m.Additionals))

	compressionOff := len(b)
	msg := h.pack(b)

	// RFC 1035 allows (but does not require) compression for packing. RFC
	// 1035 requires unpacking implementations to support compression, so
	// unconditionally enabling it is fine.
	//
	// DNS lookups are typically done over UDP, and RFC 1035 states that UDP
	// DNS messages can be a maximum of 512 bytes long. Without compression,
	// many DNS response messages are over this limit, so enabling
	// compression will help ensure compliance.
	compression := map[string]uint16{}

	for i := range m.Questions {
		var err error
		if msg, err = m.Questions[i].pack(msg, compression, compressionOff); err != nil {
			return nil, &nestedError{"packing Question", err}
		}
	}
	for i := range m.Answers {
		var err error
		if msg, err = m.Answers[i].pack(msg, compression, compressionOff); err != nil {
			return nil, &nestedError{"packing Answer", err}
		}
	}
	for i := range m.Authorities {
		var err error
		if msg, err = m.Authorities[i].pack(msg, compression, compressionOff); err != nil {
			return nil, &nestedError{"packing Authority", err}
		}
	}
	for i := range m.Additionals {
		var err error
		if msg, err = m.Additionals[i].pack(msg, compression, compressionOff); err != nil {
			return nil, &nestedError{"packing Additional", err}
		}
	}

	return msg, nil
}

// GoString implements fmt.GoStringer.GoString.
func (m *Message) GoString() string {
	s := "dnsmessage.Message{Header: " + m.Header.GoString() + ", " +
		"Questions: []dnsmessage.Question{"
	if len(m.Questions) > 0 {
		s += m.Questions[0].GoString()
		for _, q := range m.Questions[1:] {
			s += ", " + q.GoString()
		}
	}
	s += "}, Answers: []dnsmessage.Resource{"
	if len(m.Answers) > 0 {
		s += m.Answers[0].GoString()
		for _, a := range m.Answers[1:] {
			s += ", " + a.GoString()
		}
	}
	s += "}, Authorities: []dnsmessage.Resource{"
	if len(m.Authorities) > 0 {
		s += m.Authorities[0].GoString()
		for _, a := range m.Authorities[1:] {
			s += ", " + a.GoString()
		}
	}
	s += "}, Additionals: []dnsmessage.Resource{"
	if len(m.Additionals) > 0 {
		s += m.Additionals[0].GoString()
		for _, a := range m.Additionals[1:] {
			s += ", " + a.GoString()
		}
	}
	return s + "}}"
}

// A Builder allows incrementally packing a DNS message.
//
// Example usage:
//
//	buf := make([]byte, 2, 514)
//	b := NewBuilder(buf, Header{...})
//	b.EnableCompression()
//	// Optionally start a section and add things to that section.
//	// Repeat adding sections as necessary.
//	buf, err := b.Finish()
//	// If err is nil, buf[2:] will contain the built bytes.
type Builder struct {
	// msg is the storage for the message being built.
	msg []byte

	// section keeps track of the current section being built.
	section section

	// header keeps track of what should go in the header when Finish is
	// called.
	header header

	// start is the starting index of the bytes allocated in msg for header.
	start int

	// compression is a mapping from name suffixes to their starting index
	// in msg.
	compression map[string]uint16
}

// NewBuilder creates a new builder with compression disabled.
//
// Note: Most users will want to immediately enable compression with the
// EnableCompression method. See that method's comment for why you may or may
// not want to enable compression.
//
// The DNS message is appended to the provided initial buffer buf (which may be
// nil) as it is built. The final message is returned by the (*Builder).Finish
// method, which includes buf[:len(buf)] and may return the same underlying
// array if there was sufficient capacity in the slice.
func NewBuilder(buf []byte, h Header) Builder {
	if buf == nil {
		buf = make([]byte, 0, packStartingCap)
	}
	b := Builder{msg: buf, start: len(buf)}
	b.header.id, b.header.bits = h.pack()
	var hb [headerLen]byte
	b.msg = append(b.msg, hb[:]...)
	b.section = sectionHeader
	return b
}

// EnableCompression enables compression in the Builder.
//
// Leaving compression disabled avoids compression related allocations, but can
// result in larger message sizes. Be careful with this mode as it can cause
// messages to exceed the UDP size limit.
//
// According to RFC 1035, section 4.1.4, the use of compression is optional, but
// all implementations must accept both compressed and uncompressed DNS
// messages.
//
// Compression should be enabled before any sections are added for best results.
func (b *Builder) EnableCompression() {
	b.compression = map[string]uint16{}
}

func (b *Builder) startCheck(s section) error {
	if b.section <= sectionNotStarted {
		return ErrNotStarted
	}
	if b.section > s {
		return ErrSectionDone
	}
	return nil
}

// StartQuestions prepares the builder for packing Questions.
func (b *Builder) StartQuestions() error {
	if err := b.startCheck(sectionQuestions); err != nil {
		return err
	}
	b.section = sectionQuestions
	return nil
}

// StartAnswers prepares the builder for packing Answers.
func (b *Builder) StartAnswers() error {
	if err := b.startCheck(sectionAnswers); err != nil {
		return err
	}
	b.section = sectionAnswers
	return nil
}

// StartAuthorities prepares the builder for packing Authorities.
func (b *Builder) StartAuthorities() error {
	if err := b.startCheck(sectionAuthorities); err != nil {
		return err
	}
	b.section = sectionAuthorities
	return nil
}

// StartAdditionals prepares the builder for packing Additionals.
func (b *Builder) StartAdditionals() error {
	if err := b.startCheck(sectionAdditionals); err != nil {
		return err
	}
	b.section = sectionAdditionals
	return nil
}

func (b *Builder) incrementSectionCount() error {
	var count *uint16
	var err error
	switch b.section {
	case sectionQuestions:
		count = &b.header.questions
		err = errTooManyQuestions
	case sectionAnswers:
		count = &b.header.answers
		err = errTooManyAnswers
	case sectionAuthorities:
		count = &b.header.authorities
		err = errTooManyAuthorities
	case sectionAdditionals:
		count = &b.header.additionals
		err = errTooManyAdditionals
	}
	if *count == ^uint16(0) {
		return err
	}
	*count++
	return nil
}

// Question adds a single Question.
func (b *Builder) Question(q Question) error {
	if b.section < sectionQuestions {
		return ErrNotStarted
	}
	if b.section > sectionQuestions {
		return ErrSectionDone
	}
	msg, err := q.pack(b.msg, b.compression, b.start)
	if err != nil {
		return err
	}
	if err := b.incrementSectionCount(); err != nil {
		return err
	}
	b.msg = msg
	return nil
}

func (b *Builder) checkResourceSection() error {
	if b.section < sectionAnswers {
		return ErrNotStarted
	}
	if b.section > sectionAdditionals {
		return ErrSectionDone
	}
	return nil
}

// CNAMEResource adds a single CNAMEResource.
func (b *Builder) CNAMEResource(h ResourceHeader, r CNAMEResource) error {
	if err := b.checkResourceSection(); err != nil {
		return err
	}
	h.Type = r.realType()
	msg, lenOff, err := h.pack(b.msg, b.compression, b.start)
	if err != nil {
		return &nestedError{"ResourceHeader", err}
	}
	preLen := len(msg)
	if msg, err = r.pack(msg, b.compression, b.start); err != nil {
		return &nestedError{"CNAMEResource body", err}
	}
	if err := h.fixLen(msg, lenOff, preLen); err != nil {
		return err
	}
	if err := b.incrementSectionCount(); err != nil {
		return err
	}
	b.msg = msg
	return nil
}

// MXResource adds a single MXResource.
func (b *Builder) MXResource(h ResourceHeader, r MXResource) error {
	if err := b.checkResourceSection(); err != nil {
		return err
	}
	h.Type = r.realType()
	msg, lenOff, err := h.pack(b.msg, b.compression, b.start)
	if err != nil {
		return &nestedError{"ResourceHeader", err}
	}
	preLen := len(msg)
	if msg, err = r.pack(msg, b.compression, b.start); err != nil {
		return &nestedError{"MXResource body", err}
	}
	if err := h.fixLen(msg, lenOff, preLen); err != nil {
		return err
	}
	if err := b.incrementSectionCount(); err != nil {
		return err
	}
	b.msg = msg
	return nil
}

// NSResource adds a single NSResource.
func (b *Builder) NSResource(h ResourceHeader, r NSResource) error {
	if err := b.checkResourceSection(); err != nil {
		return err
	}
	h.Type = r.realType()
	msg, lenOff, err := h.pack(b.msg, b.compression, b.start)
	if err != nil {
		return &nestedError{"ResourceHeader", err}
	}
	preLen := len(msg)
	if msg, err = r.pack(msg, b.compression, b.start); err != nil {
		return &nestedError{"NSResource body", err}
	}
	if err := h.fixLen(msg, lenOff, preLen); err != nil {
		return err
	}
	if err := b.incrementSectionCount(); err != nil {
		return err
	}
	b.msg = msg
	return nil
}

// PTRResource adds a single PTRResource.
func (b *Builder) PTRResource(h ResourceHeader, r PTRResource) error {
	if err := b.checkResourceSection(); err != nil {
		return err
	}
	h.Type = r.realType()
	msg, lenOff, err := h.pack(b.msg, b.compression, b.start)
	if err != nil {
		return &nestedError{"ResourceHeader", err}
	}
	preLen := len(msg)
	if msg, err = r.pack(msg, b.compression, b.start); err != nil {
		return &nestedError{"PTRResource body", err}
	}
	if err := h.fixLen(msg, lenOff, preLen); err != nil {
		return err
	}
	if err := b.incrementSectionCount(); err != nil {
		return err
	}
	b.msg = msg
	return nil
}

// SOAResource adds a single SOAResource.
func (b *Builder) SOAResource(h ResourceHeader, r SOAResource) error {
	if err := b.checkResourceSection(); err != nil {
		return err
	}
	h.Type = r.realType()
	msg, lenOff, err := h.pack(b.msg, b.compression, b.start)
	if err != nil {
		return &nestedError{"ResourceHeader", err}
	}
	preLen := len(msg)
	if msg, err = r.pack(msg, b.compression, b.start); err != nil {
		return &nestedError{"SOAResource body", err}
	}
	if err := h.fixLen(msg, lenOff, preLen); err != nil {
		return err
	}
	if err := b.incrementSectionCount(); err != nil {
		return err
	}
	b.msg = msg
	return nil
}

// TXTResource adds a single TXTResource.
func (b *Builder) TXTResource(h ResourceHeader, r TXTResource) error {
	if err := b.checkResourceSection(); err != nil {
		return err
	}
	h.Type = r.realType()
	msg, lenOff, err := h.pack(b.msg, b.compression, b.start)
	if err != nil {
		return &nestedError{"ResourceHeader", err}
	}
	preLen := len(msg)
	if msg, err = r.pack(msg, b.compression, b.start); err != nil {
		return &nestedError{"TXTResource body", err}
	}
	if err := h.fixLen(msg, lenOff, preLen); err != nil {
		return err
	}
	if err := b.incrementSectionCount(); err != nil {
		return err
	}
	b.msg = msg
	return nil
}

// SRVResource adds a single SRVResource.
func (b *Builder) SRVResource(h ResourceHeader, r SRVResource) error {
	if err := b.checkResourceSection(); err != nil {
		return err
	}
	h.Type = r.realType()
	msg, lenOff, err := h.pack(b.msg, b.compression, b.start)
	if err != nil {
		return &nestedError{"ResourceHeader", err}
	}
	preLen := len(msg)
	if msg, err = r.pack(msg, b.compression, b.start); err != nil {
		return &nestedError{"SRVResource body", err}
	}
	if err := h.fixLen(msg, lenOff, preLen); err != nil {
		return err
	}
	if err := b.incrementSectionCount(); err != nil {
		return err
	}
	b.msg = msg
	return nil
}

// AResource adds a single AResource.
func (b *Builder) AResource(h ResourceHeader, r AResource) error {
	if err := b.checkResourceSection(); err != nil {
		return err
	}
	h.Type = r.realType()
	msg, lenOff, err := h.pack(b.msg, b.compression, b.start)
	if err != nil {
		return &nestedError{"ResourceHeader", err}
	}
	preLen := len(msg)
	if msg, err = r.pack(msg, b.compression, b.start); err != nil {
		return &nestedError{"AResource body", err}
	}
	if err := h.fixLen(msg, lenOff, preLen); err != nil {
		return err
	}
	if err := b.incrementSectionCount(); err != nil {
		return err
	}
	b.msg = msg
	return nil
}

// AAAAResource adds a single AAAAResource.
func (b *Builder) AAAAResource(h ResourceHeader, r AAAAResource) error {
	if err := b.checkResourceSection(); err != nil {
		return err
	}
	h.Type = r.realType()
	msg, lenOff, err := h.pack(b.msg, b.compression, b.start)
	if err != nil {
		return &nestedError{"ResourceHeader", err}
	}
	preLen := len(msg)
	if msg, err = r.pack(msg, b.compression, b.start); err != nil {
		return &nestedError{"AAAAResource body", err}
	}
	if err := h.fixLen(msg, lenOff, preLen); err != nil {
		return err
	}
	if err := b.incrementSectionCount(); err != nil {
		return err
	}
	b.msg = msg
	return nil
}

// OPTResource adds a single OPTResource.
func (b *Builder) OPTResource(h ResourceHeader, r OPTResource) error {
	if err := b.checkResourceSection(); err != nil {
		return err
	}
	h.Type = r.realType()
	msg, lenOff, err := h.pack(b.msg, b.compression, b.start)
	if err != nil {
		return &nestedError{"ResourceHeader", err}
	}
	preLen := len(msg)
	if msg, err = r.pack(msg, b.compression, b.start); err != nil {
		return &nestedError{"OPTResource body", err}
	}
	if err := h.fixLen(msg, lenOff, preLen); err != nil {
		return err
	}
	if err := b.incrementSectionCount(); err != nil {
		return err
	}
	b.msg = msg
	return nil
}

// UnknownResource adds a single UnknownResource.
func (b *Builder) UnknownResource(h ResourceHeader, r UnknownResource) error {
	if err := b.checkResourceSection(); err != nil {
		return err
	}
	h.Type = r.realType()
	msg, lenOff, err := h.pack(b.msg, b.compression, b.start)
	if err != nil {
		return &nestedError{"ResourceHeader", err}
	}
	preLen := len(msg)
	if msg, err = r.pack(msg, b.compression, b.start); err != nil {
		return &nestedError{"UnknownResource body", err}
	}
	if err := h.fixLen(msg, lenOff, preLen); err != nil {
		return err
	}
	if err := b.incrementSectionCount(); err != nil {
		return err
	}
	b.msg = msg
	return nil
}

// Finish ends message building and generates a binary message.
func (b *Builder) Finish() ([]byte, error) {
	if b.section < sectionHeader {
		return nil, ErrNotStarted
	}
	b.section = sectionDone
	// Space for the header was allocated in NewBuilder.
	b.header.pack(b.msg[b.start:b.start])
	return b.msg, nil
}

// A ResourceHeader is the header of a DNS resource record. There are
// many types of DNS resource records, but they all share the same header.
type ResourceHeader struct {
	// Name is the domain name for which this resource record pertains.
	Name Name

	// Type is the type of DNS resource record.
	//
	// This field will be set automatically during packing.
	Type Type

	// Class is the class of network to which this DNS resource record
	// pertains.
	Class Class

	// TTL is the length of time (measured in seconds) which this resource
	// record is valid for (time to live). All Resources in a set should
	// have the same TTL (RFC 2181 Section 5.2).
	TTL uint32

	// Length is the length of data in the resource record after the header.
	//
	// This field will be set automatically during packing.
	Length uint16
}

// GoString implements fmt.GoStringer.GoString.
func (h *ResourceHeader) GoString() string {
	return "dnsmessage.ResourceHeader{" +
		"Name: " + h.Name.GoString() + ", " +
		"Type: " + h.Type.GoString() + ", " +
		"Class: " + h.Class.GoString() + ", " +
		"TTL: " + printUint32(h.TTL) + ", " +
		"Length: " + printUint16(h.Length) + "}"
}

// pack appends the wire format of the ResourceHeader to oldMsg.
//
// lenOff is the offset in msg where the Length field was packed.
func (h *ResourceHeader) pack(oldMsg []byte, compression map[string]uint16, compressionOff int) (msg []byte, lenOff int, err error) {
	msg = oldMsg
	if msg, err = h.Name.pack(msg, compression, compressionOff); err != nil {
		return oldMsg, 0, &nestedError{"Name", err}
	}
	msg = packType(msg, h.Type)
	msg = packClass(msg, h.Class)
	msg = packUint32(msg, h.TTL)
	lenOff = len(msg)
	msg = packUint16(msg, h.Length)
	return msg, lenOff, nil
}

func (h *ResourceHeader) unpack(msg []byte, off int) (int, error) {
	newOff := off
	var err error
	if newOff, err = h.Name.unpack(msg, newOff); err != nil {
		return off, &nestedError{"Name", err}
	}
	if h.Type, newOff, err = unpackType(msg, newOff); err != nil {
		return off, &nestedError{"Type", err}
	}
	if h.Class, newOff, err = unpackClass(msg, newOff); err != nil {
		return off, &nestedError{"Class", err}
	}
	if h.TTL, newOff, err = unpackUint32(msg, newOff); err != nil {
		return off, &nestedError{"TTL", err}
	}
	if h.Length, newOff, err = unpackUint16(msg, newOff); err != nil {
		return off, &nestedError{"Length", err}
	}
	return newOff, nil
}

// fixLen updates a packed ResourceHeader to include the length of the
// ResourceBody.
//
// lenOff is the offset of the ResourceHeader.Length field in msg.
//
// preLen is the length that msg was before the ResourceBody was packed.
func (h *ResourceHeader) fixLen(msg []byte, lenOff int, preLen int) error {
	conLen := len(msg) - preLen
	if conLen > int(^uint16(0)) {
		return errResTooLong
	}

	// Fill in the length now that we know how long the content is.
	packUint16(msg[lenOff:lenOff], uint16(conLen))
	h.Length = uint16(conLen)

	return nil
}

// EDNS(0) wire constants.
const (
	edns0Version = 0

	edns0DNSSECOK     = 0x00008000
	ednsVersionMask   = 0x00ff0000
	edns0DNSSECOKMask = 0x00ff8000
)

// SetEDNS0 configures h for EDNS(0).
//
// The provided extRCode must be an extended RCode.
func (h *ResourceHeader) SetEDNS0(udpPayloadLen int, extRCode RCode, dnssecOK bool) error {
	h.Name = Name{Data: [255]byte{'.'}, Length: 1} // RFC 6891 section 6.1.2
	h.Type = TypeOPT
	h.Class = Class(udpPayloadLen)
	h.TTL = uint32(extRCode) >> 4 << 24
	if dnssecOK {
		h.TTL |= edns0DNSSECOK
	}
	return nil
}

// DNSSECAllowed reports whether the DNSSEC OK bit is set.
func (h *ResourceHeader) DNSSECAllowed() bool {
	return h.TTL&edns0DNSSECOKMask == edns0DNSSECOK // RFC 6891 section 6.1.3
}

// ExtendedRCode returns an extended RCode.
//
// The provided rcode must be the RCode in DNS message header.
func (h *ResourceHeader) ExtendedRCode(rcode RCode) RCode {
	if h.TTL&ednsVersionMask == edns0Version { // RFC 6891 section 6.1.3
		return RCode(h.TTL>>24<<4) | rcode
	}
	return rcode
}

func skipResource(msg []byte, off int) (int, error) {
	newOff, err := skipName(msg, off)
	if err != nil {
		return off, &nestedError{"Name", err}
	}
	if newOff, err = skipType(msg, newOff); err != nil {
		return off, &nestedError{"Type", err}
	}
	if newOff, err = skipClass(msg, newOff); err != nil {
		return off, &nestedError{"Class", err}
	}
	if newOff, err = skipUint32(msg, newOff); err != nil {
		return off, &nestedError{"TTL", err}
	}
	length, newOff, err := unpackUint16(msg, newOff)
	if err != nil {
		return off, &nestedError{"Length", err}
	}
	if newOff += int(length); newOff > len(msg) {
		return off, errResourceLen
	}
	return newOff, nil
}

// packUint16 appends the wire format of field to msg.
func packUint16(msg []byte, field uint16) []byte {
	return append(msg, byte(field>>8), byte(field))
}

func unpackUint16(msg []byte, off int) (uint16, int, error) {
	if off+uint16Len > len(msg) {
		return 0, off, errBaseLen
	}
	return uint16(msg[off])<<8 | uint16(msg[off+1]), off + uint16Len, nil
}

func skipUint16(msg []byte, off int) (int, error) {
	if off+uint16Len > len(msg) {
		return off, errBaseLen
	}
	return off + uint16Len, nil
}

// packType appends the wire format of field to msg.
func packType(msg []byte, field Type) []byte {
	return packUint16(msg, uint16(field))
}

func unpackType(msg []byte, off int) (Type, int, error) {
	t, o, err := unpackUint16(msg, off)
	return Type(t), o, err
}

func skipType(msg []byte, off int) (int, error) {
	return skipUint16(msg, off)
}

// packClass appends the wire format of field to msg.
func packClass(msg []byte, field Class) []byte {
	return packUint16(msg, uint16(field))
}

func unpackClass(msg []byte, off int) (Class, int, error) {
	c, o, err := unpackUint16(msg, off)
	return Class(c), o, err
}

func skipClass(msg []byte, off int) (int, error) {
	return skipUint16(msg, off)
}

// packUint32 appends the wire format of field to msg.
func packUint32(msg []byte, field uint32) []byte {
	return append(
		msg,
		byte(field>>24),
		byte(field>>16),
		byte(field>>8),
		byte(field),
	)
}

func unpackUint32(msg []byte, off int) (uint32, int, error) {
	if off+uint32Len > len(msg) {
		return 0, off, errBaseLen
	}
	v := uint32(msg[off])<<24 | uint32(msg[off+1])<<16 | uint32(msg[off+2])<<8 | uint32(msg[off+3])
	return v, off + uint32Len, nil
}

func skipUint32(msg []byte, off int) (int, error) {
	if off+uint32Len > len(msg) {
		return off, errBaseLen
	}
	return off + uint32Len, nil
}

// packText appends the wire format of field to msg.
func packText(msg []byte, field string) ([]byte, error) {
	l := len(field)
	if l > 255 {
		return nil, errStringTooLong
	}
	msg = append(msg, byte(l))
	msg = append(msg, field...)

	return msg, nil
}

func unpackText(msg []byte, off int) (string, int, error) {
	if off >= len(msg) {
		return "", off, errBaseLen
	}
	beginOff := off + 1
	endOff := beginOff + int(msg[off])
	if endOff > len(msg) {
		return "", off, errCalcLen
	}
	return string(msg[beginOff:endOff]), endOff, nil
}

// packBytes appends the wire format of field to msg.
func packBytes(msg []byte, field []byte) []byte {
	return append(msg, field...)
}

func unpackBytes(msg []byte, off int, field []byte) (int, error) {
	newOff := off + len(field)
	if newOff > len(msg) {
		return off, errBaseLen
	}
	copy(field, msg[off:newOff])
	return newOff, nil
}

const nonEncodedNameMax = 254

// A Name is a non-encoded and non-escaped domain name. It is used instead of strings to avoid
// allocations.
type Name struct {
	Data   [255]byte
	Length uint8
}

// NewName creates a new Name from a string.
func NewName(name string) (Name, error) {
	n := Name{Length: uint8(len(name))}
	if len(name) > len(n.Data) {
		return Name{}, errCalcLen
	}
	copy(n.Data[:], name)
	return n, nil
}

// MustNewName creates a new Name from a string and panics on error.
func MustNewName(name string) Name {
	n, err := NewName(name)
	if err != nil {
		panic("creating name: " + err.Error())
	}
	return n
}

// String implements fmt.Stringer.String.
//
// Note: characters inside the labels are not escaped in any way.
func (n Name) String() string {
	return string(n.Data[:n.Length])
}

// GoString implements fmt.GoStringer.GoString.
func (n *Name) GoString() string {
	return `dnsmessage.MustNewName("` + printString(n.Data[:n.Length]) + `")`
}

// pack appends the wire format of the Name to msg.
//
// Domain names are a sequence of counted strings split at the dots. They end
// with a zero-length string. Compression can be used to reuse domain suffixes.
//
// The compression map will be updated with new domain suffixes. If compression
// is nil, compression will not be used.
func (n *Name) pack(msg []byte, compression map[string]uint16, compressionOff int) ([]byte, error) {
	oldMsg := msg

	if n.Length > nonEncodedNameMax {
		return nil, errNameTooLong
	}

	// Add a trailing dot to canonicalize name.
	if n.Length == 0 || n.Data[n.Length-1] != '.' {
		return oldMsg, errNonCanonicalName
	}

	// Allow root domain.
	if n.Data[0] == '.' && n.Length == 1 {
		return append(msg, 0), nil
	}

	var nameAsStr string

	// Emit sequence of counted strings, chopping at dots.
	for i, begin := 0, 0; i < int(n.Length); i++ {
		// Check for the end of the segment.
		if n.Data[i] == '.' {
			// The two most significant bits have special meaning.
			// It isn't allowed for segments to be long enough to
			// need them.
			if i-begin >= 1<<6 {
				return oldMsg, errSegTooLong
			}

			// Segments must have a non-zero length.
			if i-begin == 0 {
				return oldMsg, errZeroSegLen
			}

			msg = append(msg, byte(i-begin))

			for j := begin; j < i; j++ {
				msg = append(msg, n.Data[j])
			}

			begin = i + 1
			continue
		}

		// We can only compress domain suffixes starting with a new
		// segment. A pointer is two bytes with the two most significant
		// bits set to 1 to indicate that it is a pointer.
		if (i == 0 || n.Data[i-1] == '.') && compression != nil {
			if ptr, ok := compression[string(n.Data[i:n.Length])]; ok {
				// Hit. Emit a pointer instead of the rest of
				// the domain.
				return append(msg, byte(ptr>>8|0xC0), byte(ptr)), nil
			}

			// Miss. Add the suffix to the compression table if the
			// offset can be stored in the available 14 bits.
			newPtr := len(msg) - compressionOff
			if newPtr <= int(^uint16(0)>>2) {
				if nameAsStr == "" {
					// allocate n.Data on the heap once, to avoid allocating it
					// multiple times (for next labels).
					nameAsStr = string(n.Data[:n.Length])
				}
				compression[nameAsStr[i:]] = uint16(newPtr)
			}
		}
	}
	return append(msg, 0), nil
}

// unpack unpacks a domain name.
func (n *Name) unpack(msg []byte, off int) (int, error) {
	// currOff is the current working offset.
	currOff := off

	// newOff is the offset where the next record will start. Pointers lead
	// to data that belongs to other names and thus doesn't count towards to
	// the usage of this name.
	newOff := off

	// ptr is the number of pointers followed.
	var ptr int

	// Name is a slice representation of the name data.
	name := n.Data[:0]

Loop:
	for {
		if currOff >= len(msg) {
			return off, errBaseLen
		}
		c := int(msg[currOff])
		currOff++
		switch c & 0xC0 {
		case 0x00: // String segment
			if c == 0x00 {
				// A zero length signals the end of the name.
				break Loop
			}
			endOff := currOff + c
			if endOff > len(msg) {
				return off, errCalcLen
			}

			// Reject names containing dots.
			// See issue golang/go#56246
			for _, v := range msg[currOff:endOff] {
				if v == '.' {
					return off, errInvalidName
				}
			}

			name = append(name, msg[currOff:endOff]...)
			name = append(name, '.')
			currOff = endOff
		case 0xC0: // Pointer
			if currOff >= len(msg) {
				return off, errInvalidPtr
			}
			c1 := msg[currOff]
			currOff++
			if ptr == 0 {
				newOff = currOff
			}
			// Don't follow too many pointers, maybe there's a loop.
			if ptr++; ptr > 10 {
				return off, errTooManyPtr
			}
			currOff = (c^0xC0)<<8 | int(c1)
		default:
			// Prefixes 0x80 and 0x40 are reserved.
			return off, errReserved
		}
	}
	if len(name) == 0 {
		name = append(name, '.')
	}
	if len(name) > nonEncodedNameMax {
		return off, errNameTooLong
	}
	n.Length = uint8(len(name))
	if ptr == 0 {
		newOff = currOff
	}
	return newOff, nil
}

func skipName(msg []byte, off int) (int, error) {
	// newOff is the offset where the next record will start. Pointers lead
	// to data that belongs to other names and thus doesn't count towards to
	// the usage of this name.
	newOff := off

Loop:
	for {
		if newOff >= len(msg) {
			return off, errBaseLen
		}
		c := int(msg[newOff])
		newOff++
		switch c & 0xC0 {
		case 0x00:
			if c == 0x00 {
				// A zero length signals the end of the name.
				break Loop
			}
			// literal string
			newOff += c
			if newOff > len(msg) {
				return off, errCalcLen
			}
		case 0xC0:
			// Pointer to somewhere else in msg.

			// Pointers are two bytes.
			newOff++

			// Don't follow the pointer as the data here has ended.
			break Loop
		default:
			// Prefixes 0x80 and 0x40 are reserved.
			return off, errReserved
		}
	}

	return newOff, nil
}

// A Question is a DNS query.
type Question struct {
	Name  Name
	Type  Type
	Class Class
}

// pack appends the wire format of the Question to msg.
func (q *Question) pack(msg []byte, compression map[string]uint16, compressionOff int) ([]byte, error) {
	msg, err := q.Name.pack(msg, compression, compressionOff)
	if err != nil {
		return msg, &nestedError{"Name", err}
	}
	msg = packType(msg, q.Type)
	return packClass(msg, q.Class), nil
}

// GoString implements fmt.GoStringer.GoString.
func (q *Question) GoString() string {
	return "dnsmessage.Question{" +
		"Name: " + q.Name.GoString() + ", " +
		"Type: " + q.Type.GoString() + ", " +
		"Class: " + q.Class.GoString() + "}"
}

func unpackResourceBody(msg []byte, off int, hdr ResourceHeader) (ResourceBody, int, error) {
	var (
		r    ResourceBody
		err  error
		name string
	)
	switch hdr.Type {
	case TypeA:
		var rb AResource
		rb, err = unpackAResource(msg, off)
		r = &rb
		name = "A"
	case TypeNS:
		var rb NSResource
		rb, err = unpackNSResource(msg, off)
		r = &rb
		name = "NS"
	case TypeCNAME:
		var rb CNAMEResource
		rb, err = unpackCNAMEResource(msg, off)
		r = &rb
		name = "CNAME"
	case TypeSOA:
		var rb SOAResource
		rb, err = unpackSOAResource(msg, off)
		r = &rb
		name = "SOA"
	case TypePTR:
		var rb PTRResource
		rb, err = unpackPTRResource(msg, off)
		r = &rb
		name = "PTR"
	case TypeMX:
		var rb MXResource
		rb, err = unpackMXResource(msg, off)
		r = &rb
		name = "MX"
	case TypeTXT:
		var rb TXTResource
		rb, err = unpackTXTResource(msg, off, hdr.Length)
		r = &rb
		name = "TXT"
	case TypeAAAA:
		var rb AAAAResource
		rb, err = unpackAAAAResource(msg, off)
		r = &rb
		name = "AAAA"
	case TypeSRV:
		var rb SRVResource
		rb, err = unpackSRVResource(msg, off)
		r = &rb
		name = "SRV"
	case TypeSVCB:
		var rb SVCBResource
		rb, err = unpackSVCBResource(msg, off, hdr.Length)
		r = &rb
		name = "SVCB"
	case TypeHTTPS:
		var rb HTTPSResource
		rb.SVCBResource, err = unpackSVCBResource(msg, off, hdr.Length)
		r = &rb
		name = "HTTPS"
	case TypeOPT:
		var rb OPTResource
		rb, err = unpackOPTResource(msg, off, hdr.Length)
		r = &rb
		name = "OPT"
	default:
		var rb UnknownResource
		rb, err = unpackUnknownResource(hdr.Type, msg, off, hdr.Length)
		r = &rb
		name = "Unknown"
	}
	if err != nil {
		return nil, off, &nestedError{name + " record", err}
	}
	return r, off + int(hdr.Length), nil
}

// A CNAMEResource is a CNAME Resource record.
type CNAMEResource struct {
	CNAME Name
}

func (r *CNAMEResource) realType() Type {
	return TypeCNAME
}

// pack appends the wire format of the CNAMEResource to msg.
func (r *CNAMEResource) pack(msg []byte, compression map[string]uint16, compressionOff int) ([]byte, error) {
	return r.CNAME.pack(msg, compression, compressionOff)
}

// GoString implements fmt.GoStringer.GoString.
func (r *CNAMEResource) GoString() string {
	return "dnsmessage.CNAMEResource{CNAME: " + r.CNAME.GoString() + "}"
}

func unpackCNAMEResource(msg []byte, off int) (CNAMEResource, error) {
	var cname Name
	if _, err := cname.unpack(msg, off); err != nil {
		return CNAMEResource{}, err
	}
	return CNAMEResource{cname}, nil
}

// An MXResource is an MX Resource record.
type MXResource struct {
	Pref uint16
	MX   Name
}

func (r *MXResource) realType() Type {
	return TypeMX
}

// pack appends the wire format of the MXResource to msg.
func (r *MXResource) pack(msg []byte, compression map[string]uint16, compressionOff int) ([]byte, error) {
	oldMsg := msg
	msg = packUint16(msg, r.Pref)
	msg, err := r.MX.pack(msg, compression, compressionOff)
	if err != nil {
		return oldMsg, &nestedError{"MXResource.MX", err}
	}
	return msg, nil
}

// GoString implements fmt.GoStringer.GoString.
func (r *MXResource) GoString() string {
	return "dnsmessage.MXResource{" +
		"Pref: " + printUint16(r.Pref) + ", " +
		"MX: " + r.MX.GoString() + "}"
}

func unpackMXResource(msg []byte, off int) (MXResource, error) {
	pref, off, err := unpackUint16(msg, off)
	if err != nil {
		return MXResource{}, &nestedError{"Pref", err}
	}
	var mx Name
	if _, err := mx.unpack(msg, off); err != nil {
		return MXResource{}, &nestedError{"MX", err}
	}
	return MXResource{pref, mx}, nil
}

// An NSResource is an NS Resource record.
type NSResource struct {
	NS Name
}

func (r *NSResource) realType() Type {
	return TypeNS
}

// pack appends the wire format of the NSResource to msg.
func (r *NSResource) pack(msg []byte, compression map[string]uint16, compressionOff int) ([]byte, error) {
	return r.NS.pack(msg, compression, compressionOff)
}

// GoString implements fmt.GoStringer.GoString.
func (r *NSResource) GoString() string {
	return "dnsmessage.NSResource{NS: " + r.NS.GoString() + "}"
}

func unpackNSResource(msg []byte, off int) (NSResource, error) {
	var ns Name
	if _, err := ns.unpack(msg, off); err != nil {
		return NSResource{}, err
	}
	return NSResource{ns}, nil
}

// A PTRResource is a PTR Resource record.
type PTRResource struct {
	PTR Name
}

func (r *PTRResource) realType() Type {
	return TypePTR
}

// pack appends the wire format of the PTRResource to msg.
func (r *PTRResource) pack(msg []byte, compression map[string]uint16, compressionOff int) ([]byte, error) {
	return r.PTR.pack(msg, compression, compressionOff)
}

// GoString implements fmt.GoStringer.GoString.
func (r *PTRResource) GoString() string {
	return "dnsmessage.PTRResource{PTR: " + r.PTR.GoString() + "}"
}

func unpackPTRResource(msg []byte, off int) (PTRResource, error) {
	var ptr Name
	if _, err := ptr.unpack(msg, off); err != nil {
		return PTRResource{}, err
	}
	return PTRResource{ptr}, nil
}

// An SOAResource is an SOA Resource record.
type SOAResource struct {
	NS      Name
	MBox    Name
	Serial  uint32
	Refresh uint32
	Retry   uint32
	Expire  uint32

	// MinTTL the is the default TTL of Resources records which did not
	// contain a TTL value and the TTL of negative responses. (RFC 2308
	// Section 4)
	MinTTL uint32
}

func (r *SOAResource) realType() Type {
	return TypeSOA
}

// pack appends the wire format of the SOAResource to msg.
func (r *SOAResource) pack(msg []byte, compression map[string]uint16, compressionOff int) ([]byte, error) {
	oldMsg := msg
	msg, err := r.NS.pack(msg, compression, compressionOff)
	if err != nil {
		return oldMsg, &nestedError{"SOAResource.NS", err}
	}
	msg, err = r.MBox.pack(msg, compression, compressionOff)
	if err != nil {
		return oldMsg, &nestedError{"SOAResource.MBox", err}
	}
	msg = packUint32(msg, r.Serial)
	msg = packUint32(msg, r.Refresh)
	msg = packUint32(msg, r.Retry)
	msg = packUint32(msg, r.Expire)
	return packUint32(msg, r.MinTTL), nil
}

// GoString implements fmt.GoStringer.GoString.
func (r *SOAResource) GoString() string {
	return "dnsmessage.SOAResource{" +
		"NS: " + r.NS.GoString() + ", " +
		"MBox: " + r.MBox.GoString() + ", " +
		"Serial: " + printUint32(r.Serial) + ", " +
		"Refresh: " + printUint32(r.Refresh) + ", " +
		"Retry: " + printUint32(r.Retry) + ", " +
		"Expire: " + printUint32(r.Expire) + ", " +
		"MinTTL: " + printUint32(r.MinTTL) + "}"
}

func unpackSOAResource(msg []byte, off int) (SOAResource, error) {
	var ns Name
	off, err := ns.unpack(msg, off)
	if err != nil {
		return SOAResource{}, &nestedError{"NS", err}
	}
	var mbox Name
	if off, err = mbox.unpack(msg, off); err != nil {
		return SOAResource{}, &nestedError{"MBox", err}
	}
	serial, off, err := unpackUint32(msg, off)
	if err != nil {
		return SOAResource{}, &nestedError{"Serial", err}
	}
	refresh, off, err := unpackUint32(msg, off)
	if err != nil {
		return SOAResource{}, &nestedError{"Refresh", err}
	}
	retry, off, err := unpackUint32(msg, off)
	if err != nil {
		return SOAResource{}, &nestedError{"Retry", err}
	}
	expire, off, err := unpackUint32(msg, off)
	if err != nil {
		return SOAResource{}, &nestedError{"Expire", err}
	}
	minTTL, _, err := unpackUint32(msg, off)
	if err != nil {
		return SOAResource{}, &nestedError{"MinTTL", err}
	}
	return SOAResource{ns, mbox, serial, refresh, retry, expire, minTTL}, nil
}

// A TXTResource is a TXT Resource record.
type TXTResource struct {
	TXT []string
}

func (r *TXTResource) realType() Type {
	return TypeTXT
}

// pack appends the wire format of the TXTResource to msg.
func (r *TXTResource) pack(msg []byte, compression map[string]uint16, compressionOff int) ([]byte, error) {
	oldMsg := msg
	for _, s := range r.TXT {
		var err error
		msg, err = packText(msg, s)
		if err != nil {
			return oldMsg, err
		}
	}
	return msg, nil
}

// GoString implements fmt.GoStringer.GoString.
func (r *TXTResource) GoString() string {
	s := "dnsmessage.TXTResource{TXT: []string{"
	if len(r.TXT) == 0 {
		return s + "}}"
	}
	s += `"` + printString([]byte(r.TXT[0]))
	for _, t := range r.TXT[1:] {
		s += `", "` + printString([]byte(t))
	}
	return s + `"}}`
}

func unpackTXTResource(msg []byte, off int, length uint16) (TXTResource, error) {
	txts := make([]string, 0, 1)
	for n := uint16(0); n < length; {
		var t string
		var err error
		if t, off, err = unpackText(msg, off); err != nil {
			return TXTResource{}, &nestedError{"text", err}
		}
		// Check if we got too many bytes.
		if length-n < uint16(len(t))+1 {
			return TXTResource{}, errCalcLen
		}
		n += uint16(len(t)) + 1
		txts = append(txts, t)
	}
	return TXTResource{txts}, nil
}

// An SRVResource is an SRV Resource record.
type SRVResource struct {
	Priority uint16
	Weight   uint16
	Port     uint16
	Target   Name // Not compressed as per RFC 2782.
}

func (r *SRVResource) realType() Type {
	return TypeSRV
}

// pack appends the wire format of the SRVResource to msg.
func (r *SRVResource) pack(msg []byte, compression map[string]uint16, compressionOff int) ([]byte, error) {
	oldMsg := msg
	msg = packUint16(msg, r.Priority)
	msg = packUint16(msg, r.Weight)
	msg = packUint16(msg, r.Port)
	msg, err := r.Target.pack(msg, nil, compressionOff)
	if err != nil {
		return oldMsg, &nestedError{"SRVResource.Target", err}
	}
	return msg, nil
}

// GoString implements fmt.GoStringer.GoString.
func (r *SRVResource) GoString() string {
	return "dnsmessage.SRVResource{" +
		"Priority: " + printUint16(r.Priority) + ", " +
		"Weight: " + printUint16(r.Weight) + ", " +
		"Port: " + printUint16(r.Port) + ", " +
		"Target: " + r.Target.GoString() + "}"
}

func unpackSRVResource(msg []byte, off int) (SRVResource, error) {
	priority, off, err := unpackUint16(msg, off)
	if err != nil {
		return SRVResource{}, &nestedError{"Priority", err}
	}
	weight, off, err := unpackUint16(msg, off)
	if err != nil {
		return SRVResource{}, &nestedError{"Weight", err}
	}
	port, off, err := unpackUint16(msg, off)
	if err != nil {
		return SRVResource{}, &nestedError{"Port", err}
	}
	var target Name
	if _, err := target.unpack(msg, off); err != nil {
		return SRVResource{}, &nestedError{"Target", err}
	}
	return SRVResource{priority, weight, port, target}, nil
}

// An AResource is an A Resource record.
type AResource struct {
	A [4]byte
}

func (r *AResource) realType() Type {
	return TypeA
}

// pack appends the wire format of the AResource to msg.
func (r *AResource) pack(msg []byte, compression map[string]uint16, compressionOff int) ([]byte, error) {
	return packBytes(msg, r.A[:]), nil
}

// GoString implements fmt.GoStringer.GoString.
func (r *AResource) GoString() string {
	return "dnsmessage.AResource{" +
		"A: [4]byte{" + printByteSlice(r.A[:]) + "}}"
}

func unpackAResource(msg []byte, off int) (AResource, error) {
	var a [4]byte
	if _, err := unpackBytes(msg, off, a[:]); err != nil {
		return AResource{}, err
	}
	return AResource{a}, nil
}

// An AAAAResource is an AAAA Resource record.
type AAAAResource struct {
	AAAA [16]byte
}

func (r *AAAAResource) realType() Type {
	return TypeAAAA
}

// GoString implements fmt.GoStringer.GoString.
func (r *AAAAResource) GoString() string {
	return "dnsmessage.AAAAResource{" +
		"AAAA: [16]byte{" + printByteSlice(r.AAAA[:]) + "}}"
}

// pack appends the wire format of the AAAAResource to msg.
func (r *AAAAResource) pack(msg []byte, compression map[string]uint16, compressionOff int) ([]byte, error) {
	return packBytes(msg, r.AAAA[:]), nil
}

func unpackAAAAResource(msg []byte, off int) (AAAAResource, error) {
	var aaaa [16]byte
	if _, err := unpackBytes(msg, off, aaaa[:]); err != nil {
		return AAAAResource{}, err
	}
	return AAAAResource{aaaa}, nil
}

// An OPTResource is an OPT pseudo Resource record.
//
// The pseudo resource record is part of the extension mechanisms for DNS
// as defined in RFC 6891.
type OPTResource struct {
	Options []Option
}

// An Option represents a DNS message option within OPTResource.
//
// The message option is part of the extension mechanisms for DNS as
// defined in RFC 6891.
type Option struct {
	Code uint16 // option code
	Data []byte
}

// GoString implements fmt.GoStringer.GoString.
func (o *Option) GoString() string {
	return "dnsmessage.Option{" +
		"Code: " + printUint16(o.Code) + ", " +
		"Data: []byte{" + printByteSlice(o.Data) + "}}"
}

func (r *OPTResource) realType() Type {
	return TypeOPT
}

func (r *OPTResource) pack(msg []byte, compression map[string]uint16, compressionOff int) ([]byte, error) {
	for _, opt := range r.Options {
		msg = packUint16(msg, opt.Code)
		l := uint16(len(opt.Data))
		msg = packUint16(msg, l)
		msg = packBytes(msg, opt.Data)
	}
	return msg, nil
}

// GoString implements fmt.GoStringer.GoString.
func (r *OPTResource) GoString() string {
	s := "dnsmessage.OPTResource{Options: []dnsmessage.Option{"
	if len(r.Options) == 0 {
		return s + "}}"
	}
	s += r.Options[0].GoString()
	for _, o := range r.Options[1:] {
		s += ", " + o.GoString()
	}
	return s + "}}"
}

func unpackOPTResource(msg []byte, off int, length uint16) (OPTResource, error) {
	var opts []Option
	for oldOff := off; off < oldOff+int(length); {
		var err error
		var o Option
		o.Code, off, err = unpackUint16(msg, off)
		if err != nil {
			return OPTResource{}, &nestedError{"Code", err}
		}
		var l uint16
		l, off, err = unpackUint16(msg, off)
		if err != nil {
			return OPTResource{}, &nestedError{"Data", err}
		}
		o.Data = make([]byte, l)
		if copy(o.Data, msg[off:]) != int(l) {
			return OPTResource{}, &nestedError{"Data", errCalcLen}
		}
		off += int(l)
		opts = append(opts, o)
	}
	return OPTResource{opts}, nil
}

// An UnknownResource is a catch-all container for unknown record types.
type UnknownResource struct {
	Type Type
	Data []byte
}

func (r *UnknownResource) realType() Type {
	return r.Type
}

// pack appends the wire format of the UnknownResource to msg.
func (r *UnknownResource) pack(msg []byte, compression map[string]uint16, compressionOff int) ([]byte, error) {
	return packBytes(msg, r.Data[:]), nil
}

// GoString implements fmt.GoStringer.GoString.
func (r *UnknownResource) GoString() string {
	return "dnsmessage.UnknownResource{" +
		"Type: " + r.Type.GoString() + ", " +
		"Data: []byte{" + printByteSlice(r.Data) + "}}"
}

func unpackUnknownResource(recordType Type, msg []byte, off int, length uint16) (UnknownResource, error) {
	parsed := UnknownResource{
		Type: recordType,
		Data: make([]byte, length),
	}
	if _, err := unpackBytes(msg, off, parsed.Data); err != nil {
		return UnknownResource{}, err
	}
	return parsed, nil
}
