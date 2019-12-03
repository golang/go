// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package parse provides functions to parse LSP logs.
// Fully processed logs are returned by ToRLog().
package parse

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"regexp"
	"strings"
)

// Direction is the type of message,
type Direction int

const (
	// Clrequest from client to server has method and id
	Clrequest Direction = iota
	// Clresponse from server to client
	Clresponse
	// Svrequest from server to client, has method and id
	Svrequest
	// Svresponse from client to server
	Svresponse
	// Toserver notification has method, but no id
	Toserver
	// Toclient notification
	Toclient
	// Reporterr is an error message
	Reporterr // errors have method and id
)

// Logmsg is the type of a parsed log entry
type Logmsg struct {
	Dir     Direction
	Method  string
	ID      string      // for requests/responses. Client and server request ids overlap
	Elapsed string      // for responses
	Hdr     string      // header. do we need to keep all these strings?
	Rest    string      // the unparsed result, with newlines or not
	Body    interface{} // the parsed result
}

// ReadLogs from a file. Most users should use TRlog().
func ReadLogs(fname string) ([]*Logmsg, error) {
	byid := make(map[string]int)
	msgs := []*Logmsg{}
	fd, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	logrdr := bufio.NewScanner(fd)
	logrdr.Buffer(nil, 1<<25) //  a large buffer, for safety
	logrdr.Split(logRec)
	for i := 0; logrdr.Scan(); i++ {
		flds := strings.SplitN(logrdr.Text(), "\n", 2)
		if len(flds) == 1 {
			flds = append(flds, "") // for Errors
		}
		msg := parselog(flds[0], flds[1])
		if msg == nil {
			log.Fatalf("failed to parse %q", logrdr.Text())
			continue
		}
		switch msg.Dir {
		case Clrequest, Svrequest:
			v, err := msg.unmarshal(Requests(msg.Method))
			if err != nil {
				log.Fatalf("%v for %s, %T", err, msg.Method, Requests(msg.Method))
			}
			msg.Body = v
		case Clresponse, Svresponse:
			v, err := msg.doresponse()
			if err != nil {
				log.Fatalf("%v %s", err, msg.Method)
			}
			msg.Body = v
		case Toserver, Toclient:
			v, err := msg.unmarshal(Notifs(msg.Method))
			if err != nil && Notifs(msg.Method) != nil {
				log.Fatalf("%s/%T: %v", msg.Method, Notifs(msg.Method), err)
			}
			msg.Body = v
		case Reporterr:
			msg.Body = msg.ID // cause?
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
func parselog(first, rest string) *Logmsg {
	if strings.HasPrefix(rest, "Params: ") {
		rest = rest[8:]
	} else if strings.HasPrefix(rest, "Result: ") {
		rest = rest[8:]
	}
	ans := &Logmsg{Hdr: first, Rest: rest}
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
		ans.Dir = Clrequest
		ans.Method = flds[6][1:]
		ans.ID = fixid(flds[8][:len(flds[8])-2])
	case chk("Received response", 11):
		ans.Dir = Clresponse
		ans.Method = flds[6][1:]
		ans.ID = fixid(flds[8])
		ans.Elapsed = flds[10]
	case chk("Received request", 9):
		ans.Dir = Svrequest
		ans.Method = flds[6][1:]
		ans.ID = fixid(flds[8])
	case chk("Sending response", 11), // gopls
		chk("Sending response", 13): // emacs
		ans.Dir = Svresponse
		ans.Method = flds[6][1:]
		ans.ID = fixid(flds[8][:len(flds[8])-1])
		ans.Elapsed = flds[10]
	case chk("Sending notification", 7):
		ans.Dir = Toserver
		ans.Method = strings.Trim(flds[6], ".'")
		if len(flds) == 9 {
			log.Printf("len=%d method=%s %q", len(flds), ans.Method, first)
		}
	case chk("Received notification", 7):
		ans.Dir = Toclient
		ans.Method = flds[6][1 : len(flds[6])-2]
	case strings.HasPrefix(first, "[Error - "):
		ans.Dir = Reporterr
		both := flds[5]
		idx := strings.Index(both, "#") // relies on ID.Number
		ans.Method = both[:idx]
		ans.ID = fixid(both[idx+1:])
		ans.Rest = strings.Join(flds[6:], " ")
	default:
		log.Fatalf("surprise, first=%q with %d flds", first, len(flds))
		return nil
	}
	return ans
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
func logRec(b []byte, atEOF bool) (int, []byte, error) { //bufio.SplitFunc
	got := recSep.FindIndex(b)
	if got == nil {
		if !atEOF {
			return 0, nil, nil // need more
		}
		return 0, nil, io.EOF
	}
	return got[1], b[:got[0]], nil
}

// String returns a user-useful versin of a Direction
func (d Direction) String() string {
	switch d {
	case Clrequest:
		return "clrequest"
	case Clresponse:
		return "clresponse"
	case Svrequest:
		return "svrequest"
	case Svresponse:
		return "svresponse"
	case Toserver:
		return "toserver"
	case Toclient:
		return "toclient"
	case Reporterr:
		return "reporterr"
	}
	return fmt.Sprintf("dirname: %d unknown", d)
}
