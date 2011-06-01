// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mail implements parsing of mail messages according to RFC 5322.
package mail

import (
	"bufio"
	"io"
	"net/textproto"
	"os"
	"time"
)

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
	return nil, os.ErrorString("mail: header could not be parsed")
}

// TODO(dsymonds): Parsers for more specific headers such as To, From, etc.

// A Header represents the key-value pairs in a mail message header.
type Header map[string][]string

// Get gets the first value associated with the given key.
// If there are no values associated with the key, Get returns "".
func (h Header) Get(key string) string {
	return textproto.MIMEHeader(h).Get(key)
}

var ErrHeaderNotPresent = os.ErrorString("mail: header not in message")

// Date parses the Date header field.
func (h Header) Date() (*time.Time, os.Error) {
	hdr := h.Get("Date")
	if hdr == "" {
		return nil, ErrHeaderNotPresent
	}
	return parseDate(hdr)
}
