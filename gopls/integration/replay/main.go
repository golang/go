// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Replay logs. See README.md
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/tools/gopls/integration/parse"
	"golang.org/x/tools/internal/jsonrpc2"
	p "golang.org/x/tools/internal/lsp/protocol"
)

var (
	ctx     = context.Background()
	command = flag.String("cmd", "", "location of server to send to, looks for gopls")
	logf    = flag.String("log", "", "log file to replay")
	cmp     = flag.Bool("cmp", false, "only compare log and /tmp/seen")
	logrdr  *bufio.Scanner
	msgs    []*parse.Logmsg
	// requests and responses/errors, by id
	clreq  = make(map[string]*logmsg)
	clresp = make(map[string]*logmsg)
	svreq  = make(map[string]*logmsg)
	svresp = make(map[string]*logmsg)
)

func main() {
	log.SetFlags(log.Lshortfile)
	flag.Parse()
	if *logf == "" {
		log.Fatal("need -log")
	}

	orig, err := parse.ToRlog(*logf)
	if err != nil {
		log.Fatalf("logfile %q %v", *logf, err)
	}
	msgs = orig.Logs
	log.Printf("old %d, hist:%s", len(msgs), orig.Histogram)

	if !*cmp {
		log.Print("calling mimic")
		mimic()
	}
	seen, err := parse.ToRlog("/tmp/seen")
	if err != nil {
		log.Fatal(err)
	}
	vvv := seen.Logs
	log.Printf("new %d, hist:%s", len(vvv), seen.Histogram)

	ok := make(map[string]int)
	f := func(x []*parse.Logmsg, label string, diags map[string][]p.Diagnostic) {
		cnts := make(map[parse.Direction]int)
		for _, l := range x {
			if l.Method == "window/logMessage" {
				// don't care
				//continue
			}
			if l.Method == "textDocument/publishDiagnostics" {
				v, ok := l.Body.(*p.PublishDiagnosticsParams)
				if !ok {
					log.Fatalf("got %T expected PublishDiagnosticsParams", l.Body)
				}
				diags[v.URI] = v.Diagnostics
			}
			cnts[l.Dir]++
			// notifications only
			if l.Dir != parse.Toserver && l.Dir != parse.Toclient {
				continue
			}
			s := fmt.Sprintf("%s %s %s", strings.Replace(l.Hdr, "\r", "", -1), label, l.Dir)
			if i := strings.Index(s, "notification"); i != -1 {
				s = s[i+12:]
			}
			if len(s) > 120 {
				s = s[:120]
			}
			ok[s]++
		}
		msg := ""
		for i := parse.Clrequest; i <= parse.Reporterr; i++ {
			msg += fmt.Sprintf("%s:%d ", i, cnts[i])
		}
		log.Printf("%s: %s", label, msg)
	}
	mdiags := make(map[string][]p.Diagnostic)
	f(msgs, "old", mdiags)
	vdiags := make(map[string][]p.Diagnostic)
	f(vvv, "new", vdiags)
	buf := []string{}
	for k := range ok {
		buf = append(buf, fmt.Sprintf("%s %d", k, ok[k]))
	}
	if len(buf) > 0 {
		log.Printf("counts of notifications")
		sort.Strings(buf)
		for _, k := range buf {
			log.Print(k)
		}
	}
	buf = buf[0:0]
	for k, v := range mdiags {
		va := vdiags[k]
		if len(v) != len(va) {
			buf = append(buf, fmt.Sprintf("new has %d, old has %d for %s",
				len(va), len(v), k))
		}
	}
	for ka := range vdiags {
		if _, ok := mdiags[ka]; !ok {
			buf = append(buf, fmt.Sprintf("new diagnostics, but no old ones, for %s",
				ka))
		}
	}
	if len(buf) > 0 {
		log.Print("diagnostics differ:")
		for _, s := range buf {
			log.Print(s)
		}
	}
}

type direction int // what sort of message it is
const (
	// rpc from client to server have method and id
	clrequest direction = iota
	clresponse
	// rpc from server have method and id
	svrequest
	svresponse
	// notifications have method, but no id
	toserver
	toclient
	reporterr // errors have method and id
)

// clrequest has method and id. toserver has method but no id, and svresponse has result (and id)
type logmsg struct {
	dir     direction
	method  string
	id      string      // for requests/responses. Client and server request ids overlap
	elapsed string      // for responses
	hdr     string      // do we need to keep all these strings?
	rest    string      // the unparsed result, with newlines or not
	body    interface{} // the parsed(?) result
}

// combined has all the fields of both Request and Response.
// Unmarshal this and then work out which it is.
type combined struct {
	VersionTag jsonrpc2.VersionTag `json:"jsonrpc"`
	ID         *jsonrpc2.ID        `json:"id,omitempty"`
	// RPC name
	Method string           `json:"method"`
	Params *json.RawMessage `json:"params,omitempty"`
	Result *json.RawMessage `json:"result,omitempty"`
	Error  *jsonrpc2.Error  `json:"error,omitempty"`
}

func (c *combined) dir() direction {
	// Method, Params, ID => request
	// Method, Params, no-ID => notification
	// Error => error response
	// Result, ID => response
	if c.Error != nil {
		return reporterr
	}
	if c.Params != nil && c.ID != nil {
		// $/cancel could be either, cope someday
		if parse.FromServer(c.Method) {
			return svrequest
		}
		return clrequest
	}
	if c.Params != nil {
		// we're receiving it, so it must be toclient
		return toclient
	}
	if c.Result == nil {
		if c.ID != nil {
			return clresponse
		}
		log.Printf("%+v", *c)
		panic("couldn't determine direction")
	}
	// we've received it, so it must be clresponse
	return clresponse
}

func send(l *parse.Logmsg, stream jsonrpc2.Stream, id *jsonrpc2.ID) {
	x, err := json.Marshal(l.Body)
	if err != nil {
		log.Fatal(err)
	}
	y := json.RawMessage(x)
	if id == nil {
		// need to use the number version of ID
		n, err := strconv.Atoi(l.ID)
		if err != nil {
			n = 0
		}
		id = &jsonrpc2.ID{Number: int64(n)}
	}
	var r interface{}
	switch l.Dir {
	case parse.Clrequest:
		r = jsonrpc2.WireRequest{
			ID:     id,
			Method: l.Method,
			Params: &y,
		}
	case parse.Svresponse:
		r = jsonrpc2.WireResponse{
			ID:     id,
			Result: &y,
		}
	case parse.Toserver:
		r = jsonrpc2.WireRequest{
			Method: l.Method,
			Params: &y,
		}
	default:
		log.Fatalf("sending %s", l.Dir)
	}
	data, err := json.Marshal(r)
	if err != nil {
		log.Fatal(err)
	}
	stream.Write(ctx, data)
}

func strID(x *jsonrpc2.ID) string {
	if x.Name != "" {
		log.Printf("strID returns %s", x.Name)
		return x.Name
	}
	return strconv.Itoa(int(x.Number))
}

func respond(c *combined, stream jsonrpc2.Stream) {
	// c is a server request
	// pick out the id, and look for the response in msgs
	id := strID(c.ID)
	for _, l := range msgs {
		if l.ID == id && l.Dir == parse.Svresponse {
			// check that the methods match?
			// need to send back the same ID we got.
			send(l, stream, c.ID)
			return
		}
	}
	log.Fatalf("no response found %q %+v %+v", c.Method, c.ID, c)
}

func findgopls() string {
	totry := [][]string{{"GOBIN", "/gopls"}, {"GOPATH", "/bin/gopls"}, {"HOME", "/go/bin/gopls"}}
	// looks in the places go install would install:
	// GOBIN, else GOPATH/bin, else HOME/go/bin
	ok := func(s string) bool {
		fd, err := os.Open(s)
		if err != nil {
			return false
		}
		fi, err := fd.Stat()
		if err != nil {
			return false
		}
		return fi.Mode()&0111 != 0
	}
	for _, t := range totry {
		g := os.Getenv(t[0])
		if g != "" && ok(g+t[1]) {
			return g + t[1]
		}
	}
	log.Fatal("could not find gopls")
	return ""
}

func mimic() {
	log.Printf("mimic %d", len(msgs))
	if *command == "" {
		*command = findgopls()
	}
	cmd := exec.Command(*command, "-logfile", "/tmp/seen", "-rpc.trace")
	toServer, err := cmd.StdinPipe()
	if err != nil {
		log.Fatal(err)
	}
	fromServer, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatal(err)
	}
	err = cmd.Start()
	if err != nil {
		log.Fatal(err)
	}
	stream := jsonrpc2.NewHeaderStream(fromServer, toServer)
	rchan := make(chan *combined, 10) // do we need buffering?
	rdr := func() {
		for {
			buf, _, err := stream.Read(ctx)
			if err != nil {
				rchan <- nil // close it instead?
				return
			}
			msg := &combined{}
			if err := json.Unmarshal(buf, msg); err != nil {
				log.Fatal(err)
			}
			rchan <- msg
		}
	}
	go rdr()
	// send as many as possible: all clrequests and toservers up to a clresponse
	// and loop
	seenids := make(map[string]bool) // id's that have been responded toig:
big:
	for _, l := range msgs {
		switch l.Dir {
		case parse.Toserver: // just send these as we get to them
			send(l, stream, nil)
		case parse.Clrequest:
			send(l, stream, nil) // for now, wait for a response, to make sure code is ok
			fallthrough
		case parse.Clresponse, parse.Reporterr: // don't go past these until they're received
			if seenids[l.ID] {
				break // onward, as it has been received already
			}
		done:
			for {
				x := <-rchan
				if x == nil {
					break big
				}
				// if it's svrequest, do something
				// if it's clresponse or reporterr, add to seenids, and if it
				// is l.id, break out of the loop, and continue the outer loop
				switch x.dir() {
				case svrequest:
					respond(x, stream)
					continue done // still waiting
				case clresponse, reporterr:
					id := strID(x.ID)
					seenids[id] = true
					if id == l.ID {
						break done
					}
				case toclient:
					continue
				default:
					log.Fatalf("%s", x.dir())
				}
			}
		case parse.Svrequest: // not ours to send
			continue
		case parse.Svresponse: // sent by us, if the request arrives
			continue
		case parse.Toclient: // we don't send these
			continue
		}
	}
}

func readLogs(fname string) []*logmsg {
	byid := make(map[string]int)
	msgs := []*logmsg{}
	fd, err := os.Open(fname)
	if err != nil {
		log.Fatal(err)
	}
	logrdr = bufio.NewScanner(fd)
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
		switch msg.dir {
		case clrequest, svrequest:
			v, err := msg.unmarshal(parse.Requests(msg.method))
			if err != nil {
				log.Fatal(err)
			}
			msg.body = v
		case clresponse, svresponse:
			v, err := msg.doresponse()
			if err != nil {
				log.Fatalf("%v %s", err, msg.method)
			}
			msg.body = v
		case toserver, toclient:
			v, err := msg.unmarshal(parse.Notifs(msg.method))
			if err != nil {
				log.Fatal(err)
			}
			msg.body = v
		case reporterr:
			msg.body = msg.id // cause?
		}
		byid[msg.id]++
		msgs = append(msgs, msg)
	}
	if err = logrdr.Err(); err != nil {
		log.Fatal(err)
		return msgs
	}
	// there's 2 uses of id 1, and notifications have no id
	for k, v := range byid {
		if false && v != 2 {
			log.Printf("ids %s:%d", k, v)
		}
	}
	if false {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		log.Printf("%d msgs, alloc=%d HeapAlloc=%d", len(msgs), m.Alloc, m.HeapAlloc)
	}
	return msgs
}

func (d direction) String() string {
	switch d {
	case clrequest:
		return "clrequest"
	case clresponse:
		return "clresponse"
	case svrequest:
		return "svrequest"
	case svresponse:
		return "svresponse"
	case toserver:
		return "toserver"
	case toclient:
		return "toclient"
	case reporterr:
		return "reporterr"
	}
	return fmt.Sprintf("dirname: %d unknown", d)
}

func (l *logmsg) Short() string {
	return fmt.Sprintf("%s %s %s %s", l.dir, l.method, l.id, l.elapsed)
}

func (l *logmsg) unmarshal(p interface{}) (interface{}, error) {
	r := []byte(l.rest)
	if err := json.Unmarshal(r, p); err != nil {
		// need general alternatives, but for now
		// if p is *[]foo and rest is {}, return an empty p (or *p?)
		// or, cheat:
		if l.rest == "{}" {
			return nil, nil
		}
		return nil, err
	}
	return p, nil
}

func (l *logmsg) doresponse() (interface{}, error) {
	for _, x := range parse.Responses(l.method) {
		v, err := l.unmarshal(x)
		if err == nil {
			return v, nil
		}
		if x == nil {
			return new(interface{}), nil
		}
	}
	log.Fatalf("doresponse failed for %s", l.method)
	return nil, nil
}

// parse a single log message, given first line, and the rest
func parselog(first, rest string) *logmsg {
	if strings.HasPrefix(rest, "Params: ") {
		rest = rest[8:]
	} else if strings.HasPrefix(rest, "Result: ") {
		rest = rest[8:]
	}
	ans := &logmsg{hdr: first, rest: rest}
	fixid := func(s string) string {
		if s != "" && s[0] == '(' {
			s = s[1 : len(s)-1]
		}
		return s
	}
	flds := strings.Fields(first)
	chk := func(s string, n int) bool { return strings.Contains(first, s) && len(flds) == n }
	// gopls and emacs differ in how they report elapsed time
	switch {
	case chk("Sending request", 9):
		ans.dir = clrequest
		ans.method = flds[6][1:]
		ans.id = fixid(flds[8][:len(flds[8])-2])
		clreq[ans.id] = ans
	case chk("Received response", 11):
		ans.dir = clresponse
		ans.method = flds[6][1:]
		ans.id = fixid(flds[8][:len(flds[8])-1])
		ans.elapsed = flds[10]
		clresp[ans.id] = ans
	case chk("Received request", 9):
		ans.dir = svrequest
		ans.method = flds[6][1:]
		ans.id = fixid(flds[8][:len(flds[8])-2])
		svreq[ans.id] = ans
	case chk("Sending response", 11), // gopls
		chk("Sending response", 13): // emacs
		ans.dir = svresponse
		ans.method = flds[6][1:]
		ans.id = fixid(flds[8][:len(flds[8])-1])
		ans.elapsed = flds[10]
		svresp[ans.id] = ans
	case chk("Sending notification", 7):
		ans.dir = toserver
		ans.method = strings.Trim(flds[6], ".'")
		if len(flds) == 9 {
			log.Printf("len=%d method=%s %q", len(flds), ans.method, first)
		}
	case chk("Received notification", 7):
		ans.dir = toclient
		ans.method = flds[6][1 : len(flds[6])-2]
	case strings.HasPrefix(first, "[Error - "):
		ans.dir = reporterr
		both := flds[5]
		idx := strings.Index(both, "#") // relies on ID.Number
		ans.method = both[:idx]
		ans.id = fixid(both[idx+1:])
		ans.rest = strings.Join(flds[6:], " ")
		clreq[ans.id] = ans
	default:
		log.Fatalf("surprise, first=%q with %d flds", first, len(flds))
		return nil
	}
	return ans
}

var recSep = regexp.MustCompile("\n\n\n|\r\n\r\n\r\n")

// return start of next record, contents of record, error
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
