// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package mail implements parsing of mail messages.

For the most part, this package follows the syntax as specified by RFC 5322 and
extended by RFC 6532.
Notable divergences:
	* Obsolete address formats are not parsed, including addresses with
	  embedded route information.
	* Group addresses are not parsed.
	* The full range of spacing (the CFWS syntax element) is not supported,
	  such as breaking addresses across lines.
	* No unicode normalization is performed.
*/
package mail

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"net/textproto"
	"strings"
	"time"
	"unicode/utf8"
)

var debug = debugT(false)

type debugT bool

func (d debugT) Printf(format string, args ...interface{}) {
	if d {
		log.Printf(format, args...)
	}
}

// A Message represents a parsed mail message.
type Message struct {
	Header Header
	Body   io.Reader
}

// ReadMessage reads a message from r.
// The headers are parsed, and the body of the message will be available
// for reading from r.
func ReadMessage(r io.Reader) (msg *Message, err error) {
	tp := textproto.NewReader(bufio.NewReader(r))

	hdr, err := tp.ReadMIMEHeader()
	if err != nil {
		return nil, err
	}

	return &Message{
		Header: Header(hdr),
		Body:   tp.R,
	}, nil
}

// Layouts suitable for passing to time.Parse.
// These are tried in order.
var dateLayouts []string

func init() {
	// Generate layouts based on RFC 5322, section 3.3.

	dows := [...]string{"", "Mon, "}   // day-of-week
	days := [...]string{"2", "02"}     // day = 1*2DIGIT
	years := [...]string{"2006", "06"} // year = 4*DIGIT / 2*DIGIT
	seconds := [...]string{":05", ""}  // second
	// "-0700 (MST)" is not in RFC 5322, but is common.
	zones := [...]string{"-0700", "MST", "-0700 (MST)"} // zone = (("+" / "-") 4DIGIT) / "GMT" / ...

	for _, dow := range dows {
		for _, day := range days {
			for _, year := range years {
				for _, second := range seconds {
					for _, zone := range zones {
						s := dow + day + " Jan " + year + " 15:04" + second + " " + zone
						dateLayouts = append(dateLayouts, s)
					}
				}
			}
		}
	}
}

func parseDate(date string) (time.Time, error) {
	for _, layout := range dateLayouts {
		t, err := time.Parse(layout, date)
		if err == nil {
			return t, nil
		}
	}
	return time.Time{}, errors.New("mail: header could not be parsed")
}

// A Header represents the key-value pairs in a mail message header.
type Header map[string][]string

// Get gets the first value associated with the given key.
// If there are no values associated with the key, Get returns "".
func (h Header) Get(key string) string {
	return textproto.MIMEHeader(h).Get(key)
}

var ErrHeaderNotPresent = errors.New("mail: header not in message")

// Date parses the Date header field.
func (h Header) Date() (time.Time, error) {
	hdr := h.Get("Date")
	if hdr == "" {
		return time.Time{}, ErrHeaderNotPresent
	}
	return parseDate(hdr)
}

// AddressList parses the named header field as a list of addresses.
func (h Header) AddressList(key string) ([]*Address, error) {
	hdr := h.Get(key)
	if hdr == "" {
		return nil, ErrHeaderNotPresent
	}
	return ParseAddressList(hdr)
}

// Address represents a single mail address.
// An address such as "Barry Gibbs <bg@example.com>" is represented
// as Address{Name: "Barry Gibbs", Address: "bg@example.com"}.
type Address struct {
	Name    string // Proper name; may be empty.
	Address string // user@domain
}

// Parses a single RFC 5322 address, e.g. "Barry Gibbs <bg@example.com>"
func ParseAddress(address string) (*Address, error) {
	return (&addrParser{s: address}).parseSingleAddress()
}

// ParseAddressList parses the given string as a list of addresses.
func ParseAddressList(list string) ([]*Address, error) {
	return (&addrParser{s: list}).parseAddressList()
}

// An AddressParser is an RFC 5322 address parser.
type AddressParser struct {
	// WordDecoder optionally specifies a decoder for RFC 2047 encoded-words.
	WordDecoder *mime.WordDecoder
}

// Parse parses a single RFC 5322 address of the
// form "Gogh Fir <gf@example.com>" or "foo@example.com".
func (p *AddressParser) Parse(address string) (*Address, error) {
	return (&addrParser{s: address, dec: p.WordDecoder}).parseSingleAddress()
}

// ParseList parses the given string as a list of comma-separated addresses
// of the form "Gogh Fir <gf@example.com>" or "foo@example.com".
func (p *AddressParser) ParseList(list string) ([]*Address, error) {
	return (&addrParser{s: list, dec: p.WordDecoder}).parseAddressList()
}

// String formats the address as a valid RFC 5322 address.
// If the address's name contains non-ASCII characters
// the name will be rendered according to RFC 2047.
func (a *Address) String() string {
	// Format address local@domain
	at := strings.LastIndex(a.Address, "@")
	var local, domain string
	if at < 0 {
		// This is a malformed address ("@" is required in addr-spec);
		// treat the whole address as local-part.
		local = a.Address
	} else {
		local, domain = a.Address[:at], a.Address[at+1:]
	}

	// Add quotes if needed
	quoteLocal := false
	for i, r := range local {
		if isAtext(r, false) {
			continue
		}
		if r == '.' {
			// Dots are okay if they are surrounded by atext.
			// We only need to check that the previous byte is
			// not a dot, and this isn't the end of the string.
			if i > 0 && local[i-1] != '.' && i < len(local)-1 {
				continue
			}
		}
		quoteLocal = true
		break
	}
	if quoteLocal {
		local = quoteString(local)

	}

	s := "<" + local + "@" + domain + ">"

	if a.Name == "" {
		return s
	}

	// If every character is printable ASCII, quoting is simple.
	allPrintable := true
	for _, r := range a.Name {
		// isWSP here should actually be isFWS,
		// but we don't support folding yet.
		if !isVchar(r) && !isWSP(r) || isMultibyte(r) {
			allPrintable = false
			break
		}
	}
	if allPrintable {
		return quoteString(a.Name) + " " + s
	}

	// Text in an encoded-word in a display-name must not contain certain
	// characters like quotes or parentheses (see RFC 2047 section 5.3).
	// When this is the case encode the name using base64 encoding.
	if strings.ContainsAny(a.Name, "\"#$%&'(),.:;<>@[]^`{|}~") {
		return mime.BEncoding.Encode("utf-8", a.Name) + " " + s
	}
	return mime.QEncoding.Encode("utf-8", a.Name) + " " + s
}

type addrParser struct {
	s   string
	dec *mime.WordDecoder // may be nil
}

func (p *addrParser) parseAddressList() ([]*Address, error) {
	var list []*Address
	for {
		p.skipSpace()
		addr, err := p.parseAddress()
		if err != nil {
			return nil, err
		}
		list = append(list, addr)

		p.skipSpace()
		if p.empty() {
			break
		}
		if !p.consume(',') {
			return nil, errors.New("mail: expected comma")
		}
	}
	return list, nil
}

func (p *addrParser) parseSingleAddress() (*Address, error) {
	addr, err := p.parseAddress()
	if err != nil {
		return nil, err
	}
	p.skipSpace()
	if !p.empty() {
		return nil, fmt.Errorf("mail: expected single address, got %q", p.s)
	}
	return addr, nil
}

// parseAddress parses a single RFC 5322 address at the start of p.
func (p *addrParser) parseAddress() (addr *Address, err error) {
	debug.Printf("parseAddress: %q", p.s)
	p.skipSpace()
	if p.empty() {
		return nil, errors.New("mail: no address")
	}

	// address = name-addr / addr-spec
	// TODO(dsymonds): Support parsing group address.

	// addr-spec has a more restricted grammar than name-addr,
	// so try parsing it first, and fallback to name-addr.
	// TODO(dsymonds): Is this really correct?
	spec, err := p.consumeAddrSpec()
	if err == nil {
		return &Address{
			Address: spec,
		}, err
	}
	debug.Printf("parseAddress: not an addr-spec: %v", err)
	debug.Printf("parseAddress: state is now %q", p.s)

	// display-name
	var displayName string
	if p.peek() != '<' {
		displayName, err = p.consumePhrase()
		if err != nil {
			return nil, err
		}
	}
	debug.Printf("parseAddress: displayName=%q", displayName)

	// angle-addr = "<" addr-spec ">"
	p.skipSpace()
	if !p.consume('<') {
		return nil, errors.New("mail: no angle-addr")
	}
	spec, err = p.consumeAddrSpec()
	if err != nil {
		return nil, err
	}
	if !p.consume('>') {
		return nil, errors.New("mail: unclosed angle-addr")
	}
	debug.Printf("parseAddress: spec=%q", spec)

	return &Address{
		Name:    displayName,
		Address: spec,
	}, nil
}

// consumeAddrSpec parses a single RFC 5322 addr-spec at the start of p.
func (p *addrParser) consumeAddrSpec() (spec string, err error) {
	debug.Printf("consumeAddrSpec: %q", p.s)

	orig := *p
	defer func() {
		if err != nil {
			*p = orig
		}
	}()

	// local-part = dot-atom / quoted-string
	var localPart string
	p.skipSpace()
	if p.empty() {
		return "", errors.New("mail: no addr-spec")
	}
	if p.peek() == '"' {
		// quoted-string
		debug.Printf("consumeAddrSpec: parsing quoted-string")
		localPart, err = p.consumeQuotedString()
	} else {
		// dot-atom
		debug.Printf("consumeAddrSpec: parsing dot-atom")
		localPart, err = p.consumeAtom(true, false)
	}
	if err != nil {
		debug.Printf("consumeAddrSpec: failed: %v", err)
		return "", err
	}

	if !p.consume('@') {
		return "", errors.New("mail: missing @ in addr-spec")
	}

	// domain = dot-atom / domain-literal
	var domain string
	p.skipSpace()
	if p.empty() {
		return "", errors.New("mail: no domain in addr-spec")
	}
	// TODO(dsymonds): Handle domain-literal
	domain, err = p.consumeAtom(true, false)
	if err != nil {
		return "", err
	}

	return localPart + "@" + domain, nil
}

// consumePhrase parses the RFC 5322 phrase at the start of p.
func (p *addrParser) consumePhrase() (phrase string, err error) {
	debug.Printf("consumePhrase: [%s]", p.s)
	// phrase = 1*word
	var words []string
	for {
		// word = atom / quoted-string
		var word string
		p.skipSpace()
		if p.empty() {
			return "", errors.New("mail: missing phrase")
		}
		if p.peek() == '"' {
			// quoted-string
			word, err = p.consumeQuotedString()
		} else {
			// atom
			// We actually parse dot-atom here to be more permissive
			// than what RFC 5322 specifies.
			word, err = p.consumeAtom(true, true)
			if err == nil {
				word, err = p.decodeRFC2047Word(word)
			}
		}

		if err != nil {
			break
		}
		debug.Printf("consumePhrase: consumed %q", word)
		words = append(words, word)
	}
	// Ignore any error if we got at least one word.
	if err != nil && len(words) == 0 {
		debug.Printf("consumePhrase: hit err: %v", err)
		return "", fmt.Errorf("mail: missing word in phrase: %v", err)
	}
	phrase = strings.Join(words, " ")
	return phrase, nil
}

// consumeQuotedString parses the quoted string at the start of p.
func (p *addrParser) consumeQuotedString() (qs string, err error) {
	// Assume first byte is '"'.
	i := 1
	qsb := make([]rune, 0, 10)

	escaped := false

Loop:
	for {
		r, size := utf8.DecodeRuneInString(p.s[i:])

		switch {
		case size == 0:
			return "", errors.New("mail: unclosed quoted-string")

		case size == 1 && r == utf8.RuneError:
			return "", fmt.Errorf("mail: invalid utf-8 in quoted-string: %q", p.s)

		case escaped:
			//  quoted-pair = ("\" (VCHAR / WSP))

			if !isVchar(r) && !isWSP(r) {
				return "", fmt.Errorf("mail: bad character in quoted-string: %q", r)
			}

			qsb = append(qsb, r)
			escaped = false

		case isQtext(r) || isWSP(r):
			// qtext (printable US-ASCII excluding " and \), or
			// FWS (almost; we're ignoring CRLF)
			qsb = append(qsb, r)

		case r == '"':
			break Loop

		case r == '\\':
			escaped = true

		default:
			return "", fmt.Errorf("mail: bad character in quoted-string: %q", r)

		}

		i += size
	}
	p.s = p.s[i+1:]
	if len(qsb) == 0 {
		return "", errors.New("mail: empty quoted-string")
	}
	return string(qsb), nil
}

// consumeAtom parses an RFC 5322 atom at the start of p.
// If dot is true, consumeAtom parses an RFC 5322 dot-atom instead.
// If permissive is true, consumeAtom will not fail on
// leading/trailing/double dots in the atom (see golang.org/issue/4938).
func (p *addrParser) consumeAtom(dot bool, permissive bool) (atom string, err error) {
	i := 0

Loop:
	for {
		r, size := utf8.DecodeRuneInString(p.s[i:])

		switch {
		case size == 1 && r == utf8.RuneError:
			return "", fmt.Errorf("mail: invalid utf-8 in address: %q", p.s)

		case size == 0 || !isAtext(r, dot):
			break Loop

		default:
			i += size

		}
	}

	if i == 0 {
		return "", errors.New("mail: invalid string")
	}
	atom, p.s = p.s[:i], p.s[i:]
	if !permissive {
		if strings.HasPrefix(atom, ".") {
			return "", errors.New("mail: leading dot in atom")
		}
		if strings.Contains(atom, "..") {
			return "", errors.New("mail: double dot in atom")
		}
		if strings.HasSuffix(atom, ".") {
			return "", errors.New("mail: trailing dot in atom")
		}
	}
	return atom, nil
}

func (p *addrParser) consume(c byte) bool {
	if p.empty() || p.peek() != c {
		return false
	}
	p.s = p.s[1:]
	return true
}

// skipSpace skips the leading space and tab characters.
func (p *addrParser) skipSpace() {
	p.s = strings.TrimLeft(p.s, " \t")
}

func (p *addrParser) peek() byte {
	return p.s[0]
}

func (p *addrParser) empty() bool {
	return p.len() == 0
}

func (p *addrParser) len() int {
	return len(p.s)
}

func (p *addrParser) decodeRFC2047Word(s string) (string, error) {
	if p.dec != nil {
		return p.dec.DecodeHeader(s)
	}

	dec, err := rfc2047Decoder.Decode(s)
	if err == nil {
		return dec, nil
	}

	if _, ok := err.(charsetError); ok {
		return s, err
	}

	// Ignore invalid RFC 2047 encoded-word errors.
	return s, nil
}

var rfc2047Decoder = mime.WordDecoder{
	CharsetReader: func(charset string, input io.Reader) (io.Reader, error) {
		return nil, charsetError(charset)
	},
}

type charsetError string

func (e charsetError) Error() string {
	return fmt.Sprintf("charset not supported: %q", string(e))
}

// isAtext reports whether r is an RFC 5322 atext character.
// If dot is true, period is included.
func isAtext(r rune, dot bool) bool {
	switch r {
	case '.':
		return dot

	case '(', ')', '<', '>', '[', ']', ':', ';', '@', '\\', ',', '"': // RFC 5322 3.2.3. specials
		return false
	}
	return isVchar(r)
}

// isQtext reports whether r is an RFC 5322 qtext character.
func isQtext(r rune) bool {
	// Printable US-ASCII, excluding backslash or quote.
	if r == '\\' || r == '"' {
		return false
	}
	return isVchar(r)
}

// quoteString renders a string as an RFC 5322 quoted-string.
func quoteString(s string) string {
	var buf bytes.Buffer
	buf.WriteByte('"')
	for _, r := range s {
		if isQtext(r) || isWSP(r) {
			buf.WriteRune(r)
		} else if isVchar(r) {
			buf.WriteByte('\\')
			buf.WriteRune(r)
		}
	}
	buf.WriteByte('"')
	return buf.String()
}

// isVchar reports whether r is an RFC 5322 VCHAR character.
func isVchar(r rune) bool {
	// Visible (printing) characters.
	return '!' <= r && r <= '~' || isMultibyte(r)
}

// isMultibyte reports whether r is a multi-byte UTF-8 character
// as supported by RFC 6532
func isMultibyte(r rune) bool {
	return r >= utf8.RuneSelf
}

// isWSP reports whether r is a WSP (white space).
// WSP is a space or horizontal tab (RFC 5234 Appendix B).
func isWSP(r rune) bool {
	return r == ' ' || r == '\t'
}
