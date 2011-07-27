// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package dict implements the Dictionary Server Protocol
// as defined in RFC 2229.
package dict

import (
	"net/textproto"
	"os"
	"strconv"
	"strings"
)

// A Client represents a client connection to a dictionary server.
type Client struct {
	text *textproto.Conn
}

// Dial returns a new client connected to a dictionary server at
// addr on the given network.
func Dial(network, addr string) (*Client, os.Error) {
	text, err := textproto.Dial(network, addr)
	if err != nil {
		return nil, err
	}
	_, _, err = text.ReadCodeLine(220)
	if err != nil {
		text.Close()
		return nil, err
	}
	return &Client{text: text}, nil
}

// Close closes the connection to the dictionary server.
func (c *Client) Close() os.Error {
	return c.text.Close()
}

// A Dict represents a dictionary available on the server.
type Dict struct {
	Name string // short name of dictionary
	Desc string // long description
}

// Dicts returns a list of the dictionaries available on the server.
func (c *Client) Dicts() ([]Dict, os.Error) {
	id, err := c.text.Cmd("SHOW DB")
	if err != nil {
		return nil, err
	}

	c.text.StartResponse(id)
	defer c.text.EndResponse(id)

	_, _, err = c.text.ReadCodeLine(110)
	if err != nil {
		return nil, err
	}
	lines, err := c.text.ReadDotLines()
	if err != nil {
		return nil, err
	}
	_, _, err = c.text.ReadCodeLine(250)

	dicts := make([]Dict, len(lines))
	for i := range dicts {
		d := &dicts[i]
		a, _ := fields(lines[i])
		if len(a) < 2 {
			return nil, textproto.ProtocolError("invalid dictionary: " + lines[i])
		}
		d.Name = a[0]
		d.Desc = a[1]
	}
	return dicts, err
}

// A Defn represents a definition.
type Defn struct {
	Dict Dict   // Dict where definition was found
	Word string // Word being defined
	Text []byte // Definition text, typically multiple lines
}

// Define requests the definition of the given word.
// The argument dict names the dictionary to use,
// the Name field of a Dict returned by Dicts.
//
// The special dictionary name "*" means to look in all the
// server's dictionaries.
// The special dictionary name "!" means to look in all the
// server's dictionaries in turn, stopping after finding the word
// in one of them.
func (c *Client) Define(dict, word string) ([]*Defn, os.Error) {
	id, err := c.text.Cmd("DEFINE %s %q", dict, word)
	if err != nil {
		return nil, err
	}

	c.text.StartResponse(id)
	defer c.text.EndResponse(id)

	_, line, err := c.text.ReadCodeLine(150)
	if err != nil {
		return nil, err
	}
	a, _ := fields(line)
	if len(a) < 1 {
		return nil, textproto.ProtocolError("malformed response: " + line)
	}
	n, err := strconv.Atoi(a[0])
	if err != nil {
		return nil, textproto.ProtocolError("invalid definition count: " + a[0])
	}
	def := make([]*Defn, n)
	for i := 0; i < n; i++ {
		_, line, err = c.text.ReadCodeLine(151)
		if err != nil {
			return nil, err
		}
		a, _ := fields(line)
		if len(a) < 3 {
			// skip it, to keep protocol in sync
			i--
			n--
			def = def[0:n]
			continue
		}
		d := &Defn{Word: a[0], Dict: Dict{a[1], a[2]}}
		d.Text, err = c.text.ReadDotBytes()
		if err != nil {
			return nil, err
		}
		def[i] = d
	}
	_, _, err = c.text.ReadCodeLine(250)
	return def, err
}

// Fields returns the fields in s.
// Fields are space separated unquoted words
// or quoted with single or double quote.
func fields(s string) ([]string, os.Error) {
	var v []string
	i := 0
	for {
		for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
			i++
		}
		if i >= len(s) {
			break
		}
		if s[i] == '"' || s[i] == '\'' {
			q := s[i]
			// quoted string
			var j int
			for j = i + 1; ; j++ {
				if j >= len(s) {
					return nil, textproto.ProtocolError("malformed quoted string")
				}
				if s[j] == '\\' {
					j++
					continue
				}
				if s[j] == q {
					j++
					break
				}
			}
			v = append(v, unquote(s[i+1:j-1]))
			i = j
		} else {
			// atom
			var j int
			for j = i; j < len(s); j++ {
				if s[j] == ' ' || s[j] == '\t' || s[j] == '\\' || s[j] == '"' || s[j] == '\'' {
					break
				}
			}
			v = append(v, s[i:j])
			i = j
		}
		if i < len(s) {
			c := s[i]
			if c != ' ' && c != '\t' {
				return nil, textproto.ProtocolError("quotes not on word boundaries")
			}
		}
	}
	return v, nil
}

func unquote(s string) string {
	if strings.Index(s, "\\") < 0 {
		return s
	}
	b := []byte(s)
	w := 0
	for r := 0; r < len(b); r++ {
		c := b[r]
		if c == '\\' {
			r++
			c = b[r]
		}
		b[w] = c
		w++
	}
	return string(b[0:w])
}
