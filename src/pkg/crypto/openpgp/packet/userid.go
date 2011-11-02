// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"io"
	"io/ioutil"
	"strings"
)

// UserId contains text that is intended to represent the name and email
// address of the key holder. See RFC 4880, section 5.11. By convention, this
// takes the form "Full Name (Comment) <email@example.com>"
type UserId struct {
	Id string // By convention, this takes the form "Full Name (Comment) <email@example.com>" which is split out in the fields below.

	Name, Comment, Email string
}

func hasInvalidCharacters(s string) bool {
	for _, c := range s {
		switch c {
		case '(', ')', '<', '>', 0:
			return true
		}
	}
	return false
}

// NewUserId returns a UserId or nil if any of the arguments contain invalid
// characters. The invalid characters are '\x00', '(', ')', '<' and '>'
func NewUserId(name, comment, email string) *UserId {
	// RFC 4880 doesn't deal with the structure of userid strings; the
	// name, comment and email form is just a convention. However, there's
	// no convention about escaping the metacharacters and GPG just refuses
	// to create user ids where, say, the name contains a '('. We mirror
	// this behaviour.

	if hasInvalidCharacters(name) || hasInvalidCharacters(comment) || hasInvalidCharacters(email) {
		return nil
	}

	uid := new(UserId)
	uid.Name, uid.Comment, uid.Email = name, comment, email
	uid.Id = name
	if len(comment) > 0 {
		if len(uid.Id) > 0 {
			uid.Id += " "
		}
		uid.Id += "("
		uid.Id += comment
		uid.Id += ")"
	}
	if len(email) > 0 {
		if len(uid.Id) > 0 {
			uid.Id += " "
		}
		uid.Id += "<"
		uid.Id += email
		uid.Id += ">"
	}
	return uid
}

func (uid *UserId) parse(r io.Reader) (err error) {
	// RFC 4880, section 5.11
	b, err := ioutil.ReadAll(r)
	if err != nil {
		return
	}
	uid.Id = string(b)
	uid.Name, uid.Comment, uid.Email = parseUserId(uid.Id)
	return
}

// Serialize marshals uid to w in the form of an OpenPGP packet, including
// header.
func (uid *UserId) Serialize(w io.Writer) error {
	err := serializeHeader(w, packetTypeUserId, len(uid.Id))
	if err != nil {
		return err
	}
	_, err = w.Write([]byte(uid.Id))
	return err
}

// parseUserId extracts the name, comment and email from a user id string that
// is formatted as "Full Name (Comment) <email@example.com>".
func parseUserId(id string) (name, comment, email string) {
	var n, c, e struct {
		start, end int
	}
	var state int

	for offset, rune := range id {
		switch state {
		case 0:
			// Entering name
			n.start = offset
			state = 1
			fallthrough
		case 1:
			// In name
			if rune == '(' {
				state = 2
				n.end = offset
			} else if rune == '<' {
				state = 5
				n.end = offset
			}
		case 2:
			// Entering comment
			c.start = offset
			state = 3
			fallthrough
		case 3:
			// In comment
			if rune == ')' {
				state = 4
				c.end = offset
			}
		case 4:
			// Between comment and email
			if rune == '<' {
				state = 5
			}
		case 5:
			// Entering email
			e.start = offset
			state = 6
			fallthrough
		case 6:
			// In email
			if rune == '>' {
				state = 7
				e.end = offset
			}
		default:
			// After email
		}
	}
	switch state {
	case 1:
		// ended in the name
		n.end = len(id)
	case 3:
		// ended in comment
		c.end = len(id)
	case 6:
		// ended in email
		e.end = len(id)
	}

	name = strings.TrimSpace(id[n.start:n.end])
	comment = strings.TrimSpace(id[c.start:c.end])
	email = strings.TrimSpace(id[e.start:e.end])
	return
}
