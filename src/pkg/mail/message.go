// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package mail implements parsing of mail messages.

For the most part, this package follows the syntax as specified by RFC 5322.
Notable divergences:
	* Obsolete address formats are not parsed, including addresses with
	  embedded route information.
	* Group addresses are not parsed.
	* The full range of spacing (the CFWS syntax element) is not supported,
	  such as breaking addresses across lines.
*/
package mail

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/textproto"
	"os"
	"strconv"
	"strings"
	"time"
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
// The headers are parsed, and the body of the message will be reading from r.
func ReadMessage(r io.Reader) (msg *Message, err os.Error) {
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

	dows := [...]string{"", "Mon, "}     // day-of-week
	days := [...]string{"2", "02"}       // day = 1*2DIGIT
	years := [...]string{"2006", "06"}   // year = 4*DIGIT / 2*DIGIT
	seconds := [...]string{":05", ""}    // second
	zones := [...]string{"-0700", "MST"} // zone = (("+" / "-") 4DIGIT) / "GMT" / ...

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

func parseDate(date string) (*time.Time, os.Error) {
	for _, layout := range dateLayouts {
		t, err := time.Parse(layout, date)
		if err == nil {
			return t, nil
		}
	}
	return nil, os.NewError("mail: header could not be parsed")
}

// A Header represents the key-value pairs in a mail message header.
type Header map[string][]string

// Get gets the first value associated with the given key.
// If there are no values associated with the key, Get returns "".
func (h Header) Get(key string) string {
	return textproto.MIMEHeader(h).Get(key)
}

var ErrHeaderNotPresent = os.NewError("mail: header not in message")

// Date parses the Date header field.
func (h Header) Date() (*time.Time, os.Error) {
	hdr := h.Get("Date")
	if hdr == "" {
		return nil, ErrHeaderNotPresent
	}
	return parseDate(hdr)
}

// AddressList parses the named header field as a list of addresses.
func (h Header) AddressList(key string) ([]*Address, os.Error) {
	hdr := h.Get(key)
	if hdr == "" {
		return nil, ErrHeaderNotPresent
	}
	return newAddrParser(hdr).parseAddressList()
}

// Address represents a single mail address.
// An address such as "Barry Gibbs <bg@example.com>" is represented
// as Address{Name: "Barry Gibbs", Address: "bg@example.com"}.
type Address struct {
	Name    string // Proper name; may be empty.
	Address string // user@domain
}

// String formats the address as a valid RFC 5322 address.
// If the address's name contains non-ASCII characters
// the name will be rendered according to RFC 2047.
func (a *Address) String() string {
	s := "<" + a.Address + ">"
	if a.Name == "" {
		return s
	}
	// If every character is printable ASCII, quoting is simple.
	allPrintable := true
	for i := 0; i < len(a.Name); i++ {
		if !isVchar(a.Name[i]) {
			allPrintable = false
			break
		}
	}
	if allPrintable {
		b := bytes.NewBufferString(`"`)
		for i := 0; i < len(a.Name); i++ {
			if !isQtext(a.Name[i]) {
				b.WriteByte('\\')
			}
			b.WriteByte(a.Name[i])
		}
		b.WriteString(`" `)
		b.WriteString(s)
		return b.String()
	}

	// UTF-8 "Q" encoding
	b := bytes.NewBufferString("=?utf-8?q?")
	for i := 0; i < len(a.Name); i++ {
		switch c := a.Name[i]; {
		case c == ' ':
			b.WriteByte('_')
		case isVchar(c) && c != '=' && c != '?' && c != '_':
			b.WriteByte(c)
		default:
			fmt.Fprintf(b, "=%02X", c)
		}
	}
	b.WriteString("?= ")
	b.WriteString(s)
	return b.String()
}

type addrParser []byte

func newAddrParser(s string) *addrParser {
	p := addrParser([]byte(s))
	return &p
}

func (p *addrParser) parseAddressList() ([]*Address, os.Error) {
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
			return nil, os.NewError("mail: expected comma")
		}
	}
	return list, nil
}

// parseAddress parses a single RFC 5322 address at the start of p.
func (p *addrParser) parseAddress() (addr *Address, err os.Error) {
	debug.Printf("parseAddress: %q", *p)
	p.skipSpace()
	if p.empty() {
		return nil, os.NewError("mail: no address")
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
	debug.Printf("parseAddress: state is now %q", *p)

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
		return nil, os.NewError("mail: no angle-addr")
	}
	spec, err = p.consumeAddrSpec()
	if err != nil {
		return nil, err
	}
	if !p.consume('>') {
		return nil, os.NewError("mail: unclosed angle-addr")
	}
	debug.Printf("parseAddress: spec=%q", spec)

	return &Address{
		Name:    displayName,
		Address: spec,
	}, nil
}

// consumeAddrSpec parses a single RFC 5322 addr-spec at the start of p.
func (p *addrParser) consumeAddrSpec() (spec string, err os.Error) {
	debug.Printf("consumeAddrSpec: %q", *p)

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
		return "", os.NewError("mail: no addr-spec")
	}
	if p.peek() == '"' {
		// quoted-string
		debug.Printf("consumeAddrSpec: parsing quoted-string")
		localPart, err = p.consumeQuotedString()
	} else {
		// dot-atom
		debug.Printf("consumeAddrSpec: parsing dot-atom")
		localPart, err = p.consumeAtom(true)
	}
	if err != nil {
		debug.Printf("consumeAddrSpec: failed: %v", err)
		return "", err
	}

	if !p.consume('@') {
		return "", os.NewError("mail: missing @ in addr-spec")
	}

	// domain = dot-atom / domain-literal
	var domain string
	p.skipSpace()
	if p.empty() {
		return "", os.NewError("mail: no domain in addr-spec")
	}
	// TODO(dsymonds): Handle domain-literal
	domain, err = p.consumeAtom(true)
	if err != nil {
		return "", err
	}

	return localPart + "@" + domain, nil
}

// consumePhrase parses the RFC 5322 phrase at the start of p.
func (p *addrParser) consumePhrase() (phrase string, err os.Error) {
	debug.Printf("consumePhrase: [%s]", *p)
	// phrase = 1*word
	var words []string
	for {
		// word = atom / quoted-string
		var word string
		p.skipSpace()
		if p.empty() {
			return "", os.NewError("mail: missing phrase")
		}
		if p.peek() == '"' {
			// quoted-string
			word, err = p.consumeQuotedString()
		} else {
			// atom
			word, err = p.consumeAtom(false)
		}

		// RFC 2047 encoded-word starts with =?, ends with ?=, and has two other ?s.
		if err == nil && strings.HasPrefix(word, "=?") && strings.HasSuffix(word, "?=") && strings.Count(word, "?") == 4 {
			word, err = decodeRFC2047Word(word)
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
		return "", os.NewError("mail: missing word in phrase")
	}
	phrase = strings.Join(words, " ")
	return phrase, nil
}

// consumeQuotedString parses the quoted string at the start of p.
func (p *addrParser) consumeQuotedString() (qs string, err os.Error) {
	// Assume first byte is '"'.
	i := 1
	qsb := make([]byte, 0, 10)
Loop:
	for {
		if i >= p.len() {
			return "", os.NewError("mail: unclosed quoted-string")
		}
		switch c := (*p)[i]; {
		case c == '"':
			break Loop
		case c == '\\':
			if i+1 == p.len() {
				return "", os.NewError("mail: unclosed quoted-string")
			}
			qsb = append(qsb, (*p)[i+1])
			i += 2
		case isQtext(c), c == ' ' || c == '\t':
			// qtext (printable US-ASCII excluding " and \), or
			// FWS (almost; we're ignoring CRLF)
			qsb = append(qsb, c)
			i++
		default:
			return "", fmt.Errorf("mail: bad character in quoted-string: %q", c)
		}
	}
	*p = (*p)[i+1:]
	return string(qsb), nil
}

// consumeAtom parses an RFC 5322 atom at the start of p.
// If dot is true, consumeAtom parses an RFC 5322 dot-atom instead.
func (p *addrParser) consumeAtom(dot bool) (atom string, err os.Error) {
	if !isAtext(p.peek(), false) {
		return "", os.NewError("mail: invalid string")
	}
	i := 1
	for ; i < p.len() && isAtext((*p)[i], dot); i++ {
	}
	// TODO(dsymonds): Remove the []byte() conversion here when 6g doesn't need it.
	atom, *p = string([]byte((*p)[:i])), (*p)[i:]
	return atom, nil
}

func (p *addrParser) consume(c byte) bool {
	if p.empty() || p.peek() != c {
		return false
	}
	*p = (*p)[1:]
	return true
}

// skipSpace skips the leading space and tab characters.
func (p *addrParser) skipSpace() {
	*p = bytes.TrimLeft(*p, " \t")
}

func (p *addrParser) peek() byte {
	return (*p)[0]
}

func (p *addrParser) empty() bool {
	return p.len() == 0
}

func (p *addrParser) len() int {
	return len(*p)
}

func decodeRFC2047Word(s string) (string, os.Error) {
	fields := strings.Split(s, "?")
	if len(fields) != 5 || fields[0] != "=" || fields[4] != "=" {
		return "", os.NewError("mail: address not RFC 2047 encoded")
	}
	charset, enc := strings.ToLower(fields[1]), strings.ToLower(fields[2])
	if charset != "iso-8859-1" && charset != "utf-8" {
		return "", fmt.Errorf("mail: charset not supported: %q", charset)
	}

	in := bytes.NewBufferString(fields[3])
	var r io.Reader
	switch enc {
	case "b":
		r = base64.NewDecoder(base64.StdEncoding, in)
	case "q":
		r = qDecoder{r: in}
	default:
		return "", fmt.Errorf("mail: RFC 2047 encoding not supported: %q", enc)
	}

	dec, err := ioutil.ReadAll(r)
	if err != nil {
		return "", err
	}

	switch charset {
	case "iso-8859-1":
		b := new(bytes.Buffer)
		for _, c := range dec {
			b.WriteRune(rune(c))
		}
		return b.String(), nil
	case "utf-8":
		return string(dec), nil
	}
	panic("unreachable")
}

type qDecoder struct {
	r       io.Reader
	scratch [2]byte
}

func (qd qDecoder) Read(p []byte) (n int, err os.Error) {
	// This method writes at most one byte into p.
	if len(p) == 0 {
		return 0, nil
	}
	if _, err := qd.r.Read(qd.scratch[:1]); err != nil {
		return 0, err
	}
	switch c := qd.scratch[0]; {
	case c == '=':
		if _, err := io.ReadFull(qd.r, qd.scratch[:2]); err != nil {
			return 0, err
		}
		x, err := strconv.Btoi64(string(qd.scratch[:2]), 16)
		if err != nil {
			return 0, fmt.Errorf("mail: invalid RFC 2047 encoding: %q", qd.scratch[:2])
		}
		p[0] = byte(x)
	case c == '_':
		p[0] = ' '
	default:
		p[0] = c
	}
	return 1, nil
}

var atextChars = []byte("ABCDEFGHIJKLMNOPQRSTUVWXYZ" +
	"abcdefghijklmnopqrstuvwxyz" +
	"0123456789" +
	"!#$%&'*+-/=?^_`{|}~")

// isAtext returns true if c is an RFC 5322 atext character.
// If dot is true, period is included.
func isAtext(c byte, dot bool) bool {
	if dot && c == '.' {
		return true
	}
	return bytes.IndexByte(atextChars, c) >= 0
}

// isQtext returns true if c is an RFC 5322 qtest character.
func isQtext(c byte) bool {
	// Printable US-ASCII, excluding backslash or quote.
	if c == '\\' || c == '"' {
		return false
	}
	return '!' <= c && c <= '~'
}

// isVchar returns true if c is an RFC 5322 VCHAR character.
func isVchar(c byte) bool {
	// Visible (printing) characters.
	return '!' <= c && c <= '~'
}
