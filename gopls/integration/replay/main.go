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
	"log"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/tools/gopls/integration/parse"
	"golang.org/x/tools/internal/jsonrpc2"
	p "golang.org/x/tools/internal/lsp/protocol"
)

var (
	command = flag.String("cmd", "", "location of server to send to, looks for gopls")
	cmp     = flag.Bool("cmp", false, "only compare log and /tmp/seen")
	logrdr  *bufio.Scanner
	msgs    []*parse.Logmsg
	// requests and responses/errors, by id
	clreq  = make(map[string]*parse.Logmsg)
	clresp = make(map[string]*parse.Logmsg)
	svreq  = make(map[string]*parse.Logmsg)
	svresp = make(map[string]*parse.Logmsg)
)

func main() {
	log.SetFlags(log.Lshortfile)
	flag.Usage = func() {
		fmt.Fprintln(flag.CommandLine.Output(), "replay [options] <logfile>")
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() != 1 {
		flag.Usage()
		os.Exit(2)
	}
	logf := flag.Arg(0)

	orig, err := parse.ToRlog(logf)
	if err != nil {
		log.Fatalf("error parsing logfile %q: %v", logf, err)
	}
	ctx := context.Background()
	msgs = orig.Logs
	log.Printf("old %d, hist:%s", len(msgs), orig.Histogram)

	if !*cmp {
		log.Print("calling mimic")
		mimic(ctx)
	}
	seen, err := parse.ToRlog("/tmp/seen")
	if err != nil {
		log.Fatal(err)
	}
	newMsgs := seen.Logs
	log.Printf("new %d, hist:%s", len(newMsgs), seen.Histogram)

	ok := make(map[string]int)
	f := func(x []*parse.Logmsg, label string, diags map[string][]p.Diagnostic) {
		counts := make(map[parse.MsgType]int)
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
			counts[l.Type]++
			// notifications only
			if l.Type != parse.ToServer && l.Type != parse.ToClient {
				continue
			}
			s := fmt.Sprintf("%s %s %s", strings.Replace(l.Hdr, "\r", "", -1), label, l.Type)
			if i := strings.Index(s, "notification"); i != -1 {
				s = s[i+12:]
			}
			if len(s) > 120 {
				s = s[:120]
			}
			ok[s]++
		}
		msg := ""
		for i := parse.ClRequest; i <= parse.ReportErr; i++ {
			msg += fmt.Sprintf("%s:%d ", i, counts[i])
		}
		log.Printf("%s: %s", label, msg)
	}
	mdiags := make(map[string][]p.Diagnostic)
	f(msgs, "old", mdiags)
	vdiags := make(map[string][]p.Diagnostic)
	f(newMsgs, "new", vdiags)
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

func msgType(c *p.Combined) parse.MsgType {
	// Method, Params, ID => request
	// Method, Params, no-ID => notification
	// Error => error response
	// Result, ID => response
	if c.Error != nil {
		return parse.ReportErr
	}
	if c.Params != nil && c.ID != nil {
		// $/cancel could be either, cope someday
		if parse.FromServer(c.Method) {
			return parse.SvRequest
		}
		return parse.ClRequest
	}
	if c.Params != nil {
		// we're receiving it, so it must be ToClient
		return parse.ToClient
	}
	if c.Result == nil {
		if c.ID != nil {
			return parse.ClResponse
		}
		log.Printf("%+v", *c)
		panic("couldn't determine direction")
	}
	// we've received it, so it must be ClResponse
	return parse.ClResponse
}

func send(ctx context.Context, l *parse.Logmsg, stream jsonrpc2.Stream, id *jsonrpc2.ID) {
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
	switch l.Type {
	case parse.ClRequest:
		r = jsonrpc2.WireRequest{
			ID:     id,
			Method: l.Method,
			Params: &y,
		}
	case parse.SvResponse:
		r = jsonrpc2.WireResponse{
			ID:     id,
			Result: &y,
		}
	case parse.ToServer:
		r = jsonrpc2.WireRequest{
			Method: l.Method,
			Params: &y,
		}
	default:
		log.Fatalf("sending %s", l.Type)
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

func respond(ctx context.Context, c *p.Combined, stream jsonrpc2.Stream) {
	// c is a server request
	// pick out the id, and look for the response in msgs
	id := strID(c.ID)
	for _, l := range msgs {
		if l.ID == id && l.Type == parse.SvResponse {
			// check that the methods match?
			// need to send back the same ID we got.
			send(ctx, l, stream, c.ID)
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
			gopls := g + t[1]
			log.Printf("using gopls at %s", gopls)
			return gopls
		}
	}
	log.Fatal("could not find gopls")
	return ""
}

func mimic(ctx context.Context) {
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
	rchan := make(chan *p.Combined, 10) // do we need buffering?
	rdr := func() {
		for {
			buf, _, err := stream.Read(ctx)
			if err != nil {
				rchan <- nil // close it instead?
				return
			}
			msg := &p.Combined{}
			if err := json.Unmarshal(buf, msg); err != nil {
				log.Fatal(err)
			}
			rchan <- msg
		}
	}
	go rdr()
	// send as many as possible: all clrequests and toservers up to a clresponse
	// and loop
	seenids := make(map[string]bool) // id's that have been responded to:
big:
	for _, l := range msgs {
		switch l.Type {
		case parse.ToServer: // just send these as we get to them
			send(ctx, l, stream, nil)
		case parse.ClRequest:
			send(ctx, l, stream, nil) // for now, wait for a response, to make sure code is ok
			fallthrough
		case parse.ClResponse, parse.ReportErr: // don't go past these until they're received
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
				switch mt := msgType(x); mt {
				case parse.SvRequest:
					respond(ctx, x, stream)
					continue done // still waiting
				case parse.ClResponse, parse.ReportErr:
					id := strID(x.ID)
					seenids[id] = true
					if id == l.ID {
						break done
					}
				case parse.ToClient:
					continue
				default:
					log.Fatalf("%s", mt)
				}
			}
		case parse.SvRequest: // not ours to send
			continue
		case parse.SvResponse: // sent by us, if the request arrives
			continue
		case parse.ToClient: // we don't send these
			continue
		}
	}
}
