// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package parse provides functions to parse LSP logs.
// Fully processed logs are returned by ToRLog().
package parse

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
)

// MsgType is the type of message.
type MsgType int

const (
	// ClRequest from client to server has method and id
	ClRequest MsgType = iota
	// ClResponse from server to client
	ClResponse
	// SvRequest from server to client, has method and id
	SvRequest
	// SvResponse from client to server
	SvResponse
	// ToServer notification has method, but no id
	ToServer
	// ToClient notification
	ToClient
	// ReportErr is an error message
	ReportErr // errors have method and id
)

// Logmsg is the type of a parsed log entry.
type Logmsg struct {
	Type    MsgType
	Method  string
	ID      string      // for requests/responses. Client and server request ids overlap
	Elapsed string      // for responses
	Hdr     string      // header. do we need to keep all these strings?
	Rest    string      // the unparsed result, with newlines or not
	Body    interface{} // the parsed result
}

// ReadLogs from a file. Most users should use ToRlog().
func ReadLogs(fname string) ([]*Logmsg, error) {
	byid := make(map[string]int)
	msgs := []*Logmsg{}
	fd, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer fd.Close()
	logrdr := bufio.NewScanner(fd)
	logrdr.Buffer(nil, 1<<25) //  a large buffer, for safety
	logrdr.Split(scanLogs)
	for i := 0; logrdr.Scan(); i++ {
		flds := strings.SplitN(logrdr.Text(), "\n", 2)
		if len(flds) == 1 {
			flds = append(flds, "") // for Errors
		}
		msg, err := parselog(flds[0], flds[1])
		if err != nil {
			return nil, fmt.Errorf("failed to parse %q: %v", logrdr.Text(), err)
		}
		switch msg.Type {
		case ClRequest, SvRequest:
			v, err := msg.unmarshal(Requests(msg.Method))
			if err != nil {
				return nil, fmt.Errorf("%v for %s, %T", err, msg.Method, Requests(msg.Method))
			}
			msg.Body = v
		case ClResponse, SvResponse:
			v, err := msg.doresponse()
			if err != nil {
				return nil, fmt.Errorf("%v %s", err, msg.Method)
			}
			msg.Body = v
		case ToServer, ToClient:
			v, err := msg.unmarshal(Notifs(msg.Method))
			if err != nil && Notifs(msg.Method) != nil {
				return nil, fmt.Errorf("%s/%T: %v", msg.Method, Notifs(msg.Method), err)
			}
			msg.Body = v
		case ReportErr:
			msg.Body = msg.Rest // save cause
		}
		byid[msg.ID]++
		msgs = append(msgs, msg)
	}
	if err = logrdr.Err(); err != nil {
		return msgs, err
	}
	return msgs, nil
}

// parse a single log message, given first line, and the rest
func parselog(first, rest string) (*Logmsg, error) {
	if strings.HasPrefix(rest, "Params: ") {
		rest = rest[8:]
	} else if strings.HasPrefix(rest, "Result: ") {
		rest = rest[8:]
	}
	msg := &Logmsg{Hdr: first, Rest: rest}
	fixid := func(s string) string {
		// emacs does (n)., gopls does (n)'.
		s = strings.Trim(s, "()'.{)")
		return s
	}
	flds := strings.Fields(first)
	chk := func(s string, n int) bool { return strings.Contains(first, s) && len(flds) == n }
	// gopls and emacs differ in how they report elapsed time
	switch {
	case chk("Sending request", 9):
		msg.Type = ClRequest
		msg.Method = flds[6][1:]
		msg.ID = fixid(flds[8][:len(flds[8])-2])
	case chk("Received response", 11):
		msg.Type = ClResponse
		msg.Method = flds[6][1:]
		msg.ID = fixid(flds[8])
		msg.Elapsed = flds[10]
	case chk("Received request", 9):
		msg.Type = SvRequest
		msg.Method = flds[6][1:]
		msg.ID = fixid(flds[8])
	case chk("Sending response", 11), // gopls
		chk("Sending response", 13): // emacs
		msg.Type = SvResponse
		msg.Method = flds[6][1:]
		msg.ID = fixid(flds[8][:len(flds[8])-1])
		msg.Elapsed = flds[10]
	case chk("Sending notification", 7):
		msg.Type = ToServer
		msg.Method = strings.Trim(flds[6], ".'")
		if len(flds) == 9 {
			log.Printf("len=%d method=%s %q", len(flds), msg.Method, first)
		}
	case chk("Received notification", 7):
		msg.Type = ToClient
		msg.Method = flds[6][1 : len(flds[6])-2]
	case strings.HasPrefix(first, "[Error - "):
		msg.Type = ReportErr
		both := flds[5]
		idx := strings.Index(both, "#") // relies on ID.Number
		msg.Method = both[:idx]
		msg.ID = fixid(both[idx+1:])
		msg.Rest = strings.Join(flds[6:], " ")
		msg.Rest = `"` + msg.Rest + `"`
	default:
		return nil, fmt.Errorf("surprise, first=%q with %d flds", first, len(flds))
	}
	return msg, nil
}

// unmarshal into a proposed type
func (l *Logmsg) unmarshal(p interface{}) (interface{}, error) {
	r := []byte(l.Rest)
	if err := json.Unmarshal(r, p); err != nil {
		// need general alternatives, but for now
		// if p is *[]foo and rest is {}, return an empty p (or *p?)
		// or, cheat:
		if l.Rest == "{}" {
			return nil, nil
		}
		return nil, err
	}
	return p, nil
}

func (l *Logmsg) doresponse() (interface{}, error) {
	for _, x := range Responses(l.Method) {
		v, err := l.unmarshal(x)
		if err == nil {
			return v, nil
		}
		if x == nil {
			return new(interface{}), nil
		}
	}
	// failure!
	rr := Responses(l.Method)
	for _, x := range rr {
		log.Printf("tried %T", x)
	}
	log.Fatalf("(%d) doresponse failed for %s %q", len(rr), l.Method, l.Rest)
	return nil, nil
}

// be a little forgiving in separating log records
var recSep = regexp.MustCompile("\n\n\n|\r\n\r\n\r\n")

// return offset of start of next record, contents of record, error
func scanLogs(b []byte, atEOF bool) (int, []byte, error) { //bufio.SplitFunc
	got := recSep.FindIndex(b)
	if got == nil {
		if atEOF && len(b) > 0 {
			return 0, nil, errors.New("malformed log: all logs should end with a separator")
		}
		return 0, nil, nil
	}
	return got[1], b[:got[0]], nil
}

// String returns a user-useful versin of a Direction
func (d MsgType) String() string {
	switch d {
	case ClRequest:
		return "clrequest"
	case ClResponse:
		return "clresponse"
	case SvRequest:
		return "svrequest"
	case SvResponse:
		return "svresponse"
	case ToServer:
		return "toserver"
	case ToClient:
		return "toclient"
	case ReportErr:
		return "reporterr"
	}
	return fmt.Sprintf("dirname: %d unknown", d)
}
