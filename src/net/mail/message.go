// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package mail implements parsing of mail messages.

For the most part, this package follows the syntax as specified by RFC 5322 and
extended by RFC 6532.
Notable divergences:
  - Obsolete address formats are not parsed, including addresses with
    embedded route information.
  - The full range of spacing (the CFWS syntax element) is not supported,
    such as breaking addresses across lines.
  - No unicode normalization is performed.
  - A leading From line is permitted, as in mbox format (RFC 4155).
*/
package mail

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"net"
	"net/textproto"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

var debug = debugT(false)

type debugT bool

func (d debugT) Printf(format string, args ...any) {
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
// for reading from msg.Body.
func ReadMessage(r io.Reader) (msg *Message, err error) {
	tp := textproto.NewReader(bufio.NewReader(r))

	hdr, err := readHeader(tp)
	if err != nil && (err != io.EOF || len(hdr) == 0) {
		return nil, err
	}

	return &Message{
		Header: Header(hdr),
		Body:   tp.R,
	}, nil
}

// readHeader reads the message headers from r.
// This is like textproto.ReadMIMEHeader, but doesn't validate.
// The fix for issue #53188 tightened up net/textproto to enforce
// restrictions of RFC 7230.
// This package implements RFC 5322, which does not have those restrictions.
// This function copies the relevant code from net/textproto,
// simplified for RFC 5322.
func readHeader(r *textproto.Reader) (map[string][]string, error) {
	m := make(map[string][]string)

	// The first line cannot start with a leading space.
	if buf, err := r.R.Peek(1); err == nil && (buf[0] == ' ' || buf[0] == '\t') {
		line, err := r.ReadLine()
		if err != nil {
			return m, err
		}
		return m, errors.New("malformed initial line: " + line)
	}

	for {
		kv, err := r.ReadContinuedLine()
		if kv == "" {
			return m, err
		}

		// Key ends at first colon.
		k, v, ok := strings.Cut(kv, ":")
		if !ok {
			return m, errors.New("malformed header line: " + kv)
		}
		key := textproto.CanonicalMIMEHeaderKey(k)

		// Permit empty key, because that is what we did in the past.
		if key == "" {
			continue
		}

		// Skip initial spaces in value.
		value := strings.TrimLeft(v, " \t")

		m[key] = append(m[key], value)

		if err != nil {
			return m, err
		}
	}
}

// Layouts suitable for passing to time.Parse.
// These are tried in order.
var (
	dateLayoutsBuildOnce sync.Once
	dateLayouts          []string
)

func buildDateLayouts() {
	// Generate layouts based on RFC 5322, section 3.3.

	dows := [...]string{"", "Mon, "}   // day-of-week
	days := [...]string{"2", "02"}     // day = 1*2DIGIT
	years := [...]string{"2006", "06"} // year = 4*DIGIT / 2*DIGIT
	seconds := [...]string{":05", ""}  // second
	// "-0700 (MST)" is not in RFC 5322, but is common.
	zones := [...]string{"-0700", "MST", "UT"} // zone = (("+" / "-") 4DIGIT) / "UT" / "GMT" / ...

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

// ParseDate parses an RFC 5322 date string.
func ParseDate(date string) (time.Time, error) {
	dateLayoutsBuildOnce.Do(buildDateLayouts)
	// CR and LF must match and are tolerated anywhere in the date field.
	date = strings.ReplaceAll(date, "\r\n", "")
	if strings.Contains(date, "\r") {
		return time.Time{}, errors.New("mail: header has a CR without LF")
	}
	// Re-using some addrParser methods which support obsolete text, i.e. non-printable ASCII
	p := addrParser{date, nil}
	p.skipSpace()

	// RFC 5322: zone = (FWS ( "+" / "-" ) 4DIGIT) / obs-zone
	// zone length is always 5 chars unless obsolete (obs-zone)
	if ind := strings.IndexAny(p.s, "+-"); ind != -1 && len(p.s) >= ind+5 {
		date = p.s[:ind+5]
		p.s = p.s[ind+5:]
	} else {
		ind := strings.Index(p.s, "T")
		if ind == 0 {
			// In this case we have the following date formats:
			// * Thu, 20 Nov 1997 09:55:06 MDT
			// * Thu, 20 Nov 1997 09:55:06 MDT (MDT)
			// * Thu, 20 Nov 1997 09:55:06 MDT (This comment)
			ind = strings.Index(p.s[1:], "T")
			if ind != -1 {
				ind++
			}
		}

		if ind != -1 && len(p.s) >= ind+5 {
			// The last letter T of the obsolete time zone is checked when no standard time zone is found.
			// If T is misplaced, the date to parse is garbage.
			date = p.s[:ind+1]
			p.s = p.s[ind+1:]
		}
	}
	if !p.skipCFWS() {
		return time.Time{}, errors.New("mail: misformatted parenthetical comment")
	}
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
// It is case insensitive; CanonicalMIMEHeaderKey is used
// to canonicalize the provided key.
// If there are no values associated with the key, Get returns "".
// To access multiple values of a key, or to use non-canonical keys,
// access the map directly.
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
	return ParseDate(hdr)
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

// ParseAddress parses a single RFC 5322 address, e.g. "Barry Gibbs <bg@example.com>"
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

		// allow skipping empty entries (RFC5322 obs-addr-list)
		if p.consume(',') {
			continue
		}

		addrs, err := p.parseAddress(true)
		if err != nil {
			return nil, err
		}
		list = append(list, addrs...)

		if !p.skipCFWS() {
			return nil, errors.New("mail: misformatted parenthetical comment")
		}
		if p.empty() {
			break
		}
		if p.peek() != ',' {
			return nil, errors.New("mail: expected comma")
		}

		// Skip empty entries for obs-addr-list.
		for p.consume(',') {
			p.skipSpace()
		}
		if p.empty() {
			break
		}
	}
	return list, nil
}

func (p *addrParser) parseSingleAddress() (*Address, error) {
	addrs, err := p.parseAddress(true)
	if err != nil {
		return nil, err
	}
	if !p.skipCFWS() {
		return nil, errors.New("mail: misformatted parenthetical comment")
	}
	if !p.empty() {
		return nil, fmt.Errorf("mail: expected single address, got %q", p.s)
	}
	if len(addrs) == 0 {
		return nil, errors.New("mail: empty group")
	}
	if len(addrs) > 1 {
		return nil, errors.New("mail: group with multiple addresses")
	}
	return addrs[0], nil
}

// parseAddress parses a single RFC 5322 address at the start of p.
func (p *addrParser) parseAddress(handleGroup bool) ([]*Address, error) {
	debug.Printf("parseAddress: %q", p.s)
	p.skipSpace()
	if p.empty() {
		return nil, errors.New("mail: no address")
	}

	// address = mailbox / group
	// mailbox = name-addr / addr-spec
	// group = display-name ":" [group-list] ";" [CFWS]

	// addr-spec has a more restricted grammar than name-addr,
	// so try parsing it first, and fallback to name-addr.
	// TODO(dsymonds): Is this really correct?
	spec, err := p.consumeAddrSpec()
	if err == nil {
		var displayName string
		p.skipSpace()
		if !p.empty() && p.peek() == '(' {
			displayName, err = p.consumeDisplayNameComment()
			if err != nil {
				return nil, err
			}
		}

		return []*Address{{
			Name:    displayName,
			Address: spec,
		}}, err
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

	p.skipSpace()
	if handleGroup {
		if p.consume(':') {
			return p.consumeGroupList()
		}
	}
	// angle-addr = "<" addr-spec ">"
	if !p.consume('<') {
		atext := true
		for _, r := range displayName {
			if !isAtext(r, true) {
				atext = false
				break
			}
		}
		if atext {
			// The input is like "foo.bar"; it's possible the input
			// meant to be "foo.bar@domain", or "foo.bar <...>".
			return nil, errors.New("mail: missing '@' or angle-addr")
		}
		// The input is like "Full Name", which couldn't possibly be a
		// valid email address if followed by "@domain"; the input
		// likely meant to be "Full Name <...>".
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

	return []*Address{{
		Name:    displayName,
		Address: spec,
	}}, nil
}

func (p *addrParser) consumeGroupList() ([]*Address, error) {
	var group []*Address
	// handle empty group.
	p.skipSpace()
	if p.consume(';') {
		if !p.skipCFWS() {
			return nil, errors.New("mail: misformatted parenthetical comment")
		}
		return group, nil
	}

	for {
		p.skipSpace()
		// embedded groups not allowed.
		addrs, err := p.parseAddress(false)
		if err != nil {
			return nil, err
		}
		group = append(group, addrs...)

		if !p.skipCFWS() {
			return nil, errors.New("mail: misformatted parenthetical comment")
		}
		if p.consume(';') {
			if !p.skipCFWS() {
				return nil, errors.New("mail: misformatted parenthetical comment")
			}
			break
		}
		if !p.consume(',') {
			return nil, errors.New("mail: expected comma")
		}
	}
	return group, nil
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
		if localPart == "" {
			err = errors.New("mail: empty quoted string in addr-spec")
		}
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

	if p.peek() == '[' {
		// domain-literal
		domain, err = p.consumeDomainLiteral()
		if err != nil {
			return "", err
		}
	} else {
		// dot-atom
		domain, err = p.consumeAtom(true, false)
		if err != nil {
			return "", err
		}
	}

	return localPart + "@" + domain, nil
}

// consumePhrase parses the RFC 5322 phrase at the start of p.
func (p *addrParser) consumePhrase() (phrase string, err error) {
	debug.Printf("consumePhrase: [%s]", p.s)
	// phrase = 1*word
	var words []string
	var isPrevEncoded bool
	for {
		// obs-phrase allows CFWS after one word
		if len(words) > 0 {
			if !p.skipCFWS() {
				return "", errors.New("mail: misformatted parenthetical comment")
			}
		}
		// word = atom / quoted-string
		var word string
		p.skipSpace()
		if p.empty() {
			break
		}
		isEncoded := false
		if p.peek() == '"' {
			// quoted-string
			word, err = p.consumeQuotedString()
		} else {
			// atom
			// We actually parse dot-atom here to be more permissive
			// than what RFC 5322 specifies.
			word, err = p.consumeAtom(true, true)
			if err == nil {
				word, isEncoded, err = p.decodeRFC2047Word(word)
			}
		}

		if err != nil {
			break
		}
		debug.Printf("consumePhrase: consumed %q", word)
		if isPrevEncoded && isEncoded {
			words[len(words)-1] += word
		} else {
			words = append(words, word)
		}
		isPrevEncoded = isEncoded
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
	return string(qsb), nil
}

// consumeAtom parses an RFC 5322 atom at the start of p.
// If dot is true, consumeAtom parses an RFC 5322 dot-atom instead.
// If permissive is true, consumeAtom will not fail on:
// - leading/trailing/double dots in the atom (see golang.org/issue/4938)
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

// consumeDomainLiteral parses an RFC 5322 domain-literal at the start of p.
func (p *addrParser) consumeDomainLiteral() (string, error) {
	// Skip the leading [
	if !p.consume('[') {
		return "", errors.New(`mail: missing "[" in domain-literal`)
	}

	// Parse the dtext
	var dtext string
	for {
		if p.empty() {
			return "", errors.New("mail: unclosed domain-literal")
		}
		if p.peek() == ']' {
			break
		}

		r, size := utf8.DecodeRuneInString(p.s)
		if size == 1 && r == utf8.RuneError {
			return "", fmt.Errorf("mail: invalid utf-8 in domain-literal: %q", p.s)
		}
		if !isDtext(r) {
			return "", fmt.Errorf("mail: bad character in domain-literal: %q", r)
		}

		dtext += p.s[:size]
		p.s = p.s[size:]
	}

	// Skip the trailing ]
	if !p.consume(']') {
		return "", errors.New("mail: unclosed domain-literal")
	}

	// Check if the domain literal is an IP address
	if addr, ok := strings.CutPrefix(dtext, "IPv6:"); ok {
		if len(net.ParseIP(addr)) != net.IPv6len {
			return "", fmt.Errorf("mail: invalid IPv6 address in domain-literal: %q", dtext)
		}

	} else if net.ParseIP(dtext).To4() == nil {
		return "", fmt.Errorf("mail: invalid IP address in domain-literal: %q", dtext)
	}

	return "[" + dtext + "]", nil
}

func (p *addrParser) consumeDisplayNameComment() (string, error) {
	if !p.consume('(') {
		return "", errors.New("mail: comment does not start with (")
	}
	comment, ok := p.consumeComment()
	if !ok {
		return "", errors.New("mail: misformatted parenthetical comment")
	}

	// TODO(stapelberg): parse quoted-string within comment
	words := strings.FieldsFunc(comment, func(r rune) bool { return r == ' ' || r == '\t' })
	for idx, word := range words {
		decoded, isEncoded, err := p.decodeRFC2047Word(word)
		if err != nil {
			return "", err
		}
		if isEncoded {
			words[idx] = decoded
		}
	}

	return strings.Join(words, " "), nil
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

// skipCFWS skips CFWS as defined in RFC5322.
func (p *addrParser) skipCFWS() bool {
	p.skipSpace()

	for {
		if !p.consume('(') {
			break
		}

		if _, ok := p.consumeComment(); !ok {
			return false
		}

		p.skipSpace()
	}

	return true
}

func (p *addrParser) consumeComment() (string, bool) {
	// '(' already consumed.
	depth := 1

	var comment string
	for {
		if p.empty() || depth == 0 {
			break
		}

		if p.peek() == '\\' && p.len() > 1 {
			p.s = p.s[1:]
		} else if p.peek() == '(' {
			depth++
		} else if p.peek() == ')' {
			depth--
		}
		if depth > 0 {
			comment += p.s[:1]
		}
		p.s = p.s[1:]
	}

	return comment, depth == 0
}

func (p *addrParser) decodeRFC2047Word(s string) (word string, isEncoded bool, err error) {
	dec := p.dec
	if dec == nil {
		dec = &rfc2047Decoder
	}

	// Substitute our own CharsetReader function so that we can tell
	// whether an error from the Decode method was due to the
	// CharsetReader (meaning the charset is invalid).
	// We used to look for the charsetError type in the error result,
	// but that behaves badly with CharsetReaders other than the
	// one in rfc2047Decoder.
	adec := *dec
	charsetReaderError := false
	adec.CharsetReader = func(charset string, input io.Reader) (io.Reader, error) {
		if dec.CharsetReader == nil {
			charsetReaderError = true
			return nil, charsetError(charset)
		}
		r, err := dec.CharsetReader(charset, input)
		if err != nil {
			charsetReaderError = true
		}
		return r, err
	}
	word, err = adec.Decode(s)
	if err == nil {
		return word, true, nil
	}

	// If the error came from the character set reader
	// (meaning the character set itself is invalid
	// but the decoding worked fine until then),
	// return the original text and the error,
	// with isEncoded=true.
	if charsetReaderError {
		return s, true, err
	}

	// Ignore invalid RFC 2047 encoded-word errors.
	return s, false, nil
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

	// RFC 5322 3.2.3. specials
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
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		if isQtext(r) || isWSP(r) {
			b.WriteRune(r)
		} else if isVchar(r) {
			b.WriteByte('\\')
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}

// isVchar reports whether r is an RFC 5322 VCHAR character.
func isVchar(r rune) bool {
	// Visible (printing) characters.
	return '!' <= r && r <= '~' || isMultibyte(r)
}

// isMultibyte reports whether r is a multi-byte UTF-8 character
// as supported by RFC 6532.
func isMultibyte(r rune) bool {
	return r >= utf8.RuneSelf
}

// isWSP reports whether r is a WSP (white space).
// WSP is a space or horizontal tab (RFC 5234 Appendix B).
func isWSP(r rune) bool {
	return r == ' ' || r == '\t'
}

// isDtext reports whether r is an RFC 5322 dtext character.
func isDtext(r rune) bool {
	// Printable US-ASCII, excluding "[", "]", or "\".
	if r == '[' || r == ']' || r == '\\' {
		return false
	}
	return isVchar(r)
}
