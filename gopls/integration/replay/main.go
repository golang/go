// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Replay logs. See README.md
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/tools/gopls/integration/parse"
	"golang.org/x/tools/internal/fakenet"
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
	f := func(x []*parse.Logmsg, label string, diags map[p.DocumentURI][]p.Diagnostic) {
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
	mdiags := make(map[p.DocumentURI][]p.Diagnostic)
	f(msgs, "old", mdiags)
	vdiags := make(map[p.DocumentURI][]p.Diagnostic)
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

func send(ctx context.Context, l *parse.Logmsg, stream jsonrpc2.Stream, id *jsonrpc2.ID) {
	if id == nil {
		// need to use the number version of ID
		n, err := strconv.Atoi(l.ID)
		if err != nil {
			n = 0
		}
		nid := jsonrpc2.NewIntID(int64(n))
		id = &nid
	}
	var msg jsonrpc2.Message
	var err error
	switch l.Type {
	case parse.ClRequest:
		msg, err = jsonrpc2.NewCall(*id, l.Method, l.Body)
	case parse.SvResponse:
		msg, err = jsonrpc2.NewResponse(*id, l.Body, nil)
	case parse.ToServer:
		msg, err = jsonrpc2.NewNotification(l.Method, l.Body)
	default:
		log.Fatalf("sending %s", l.Type)
	}
	if err != nil {
		log.Fatal(err)
	}
	stream.Write(ctx, msg)
}

func respond(ctx context.Context, c *jsonrpc2.Call, stream jsonrpc2.Stream) {
	// c is a server request
	// pick out the id, and look for the response in msgs
	id := c.ID()
	idstr := fmt.Sprint(id)
	for _, l := range msgs {
		if l.ID == idstr && l.Type == parse.SvResponse {
			// check that the methods match?
			// need to send back the same ID we got.
			send(ctx, l, stream, &id)
			return
		}
	}
	log.Fatalf("no response found %q %+v %+v", c.Method(), c.ID(), c)
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
	conn := fakenet.NewConn("stdio", fromServer, toServer)
	stream := jsonrpc2.NewHeaderStream(conn)
	rchan := make(chan jsonrpc2.Message, 10) // do we need buffering?
	rdr := func() {
		for {
			msg, _, err := stream.Read(ctx)
			if err != nil {
				rchan <- nil // close it instead?
				return
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
				msg := <-rchan
				if msg == nil {
					break big
				}
				// if it's svrequest, do something
				// if it's clresponse or reporterr, add to seenids, and if it
				// is l.id, break out of the loop, and continue the outer loop

				switch msg := msg.(type) {
				case *jsonrpc2.Call:
					if parse.FromServer(msg.Method()) {
						respond(ctx, msg, stream)
						continue done // still waiting
					}
				case *jsonrpc2.Response:
					id := fmt.Sprint(msg.ID())
					seenids[id] = true
					if id == l.ID {
						break done
					}
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
