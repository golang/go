// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !appengine

// Package socket implements an WebSocket-based playground backend.
// Clients connect to a websocket handler and send run/kill commands, and
// the server sends the output and exit status of the running processes.
// Multiple clients running multiple processes may be served concurrently.
// The wire format is JSON and is described by the Message type.
//
// This will not run on App Engine as WebSockets are not supported there.
package socket

import (
	"bytes"
	"encoding/json"
	"errors"
	"go/parser"
	"go/token"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"code.google.com/p/go.net/websocket"
)

// Handler implements a WebSocket handler for a client connection.
var Handler = websocket.Handler(socketHandler)

// Environ provides an environment when a binary, such as the go tool, is
// invoked.
var Environ func() []string = os.Environ

const (
	// The maximum number of messages to send per session (avoid flooding).
	msgLimit = 1000

	// Batch messages sent in this interval and send as a single message.
	msgDelay = 10 * time.Millisecond
)

// Message is the wire format for the websocket connection to the browser.
// It is used for both sending output messages and receiving commands, as
// distinguished by the Kind field.
type Message struct {
	Id      string // client-provided unique id for the process
	Kind    string // in: "run", "kill" out: "stdout", "stderr", "end"
	Body    string
	Options *Options `json:",omitempty"`
}

// Options specify additional message options.
type Options struct {
	Race bool // use -race flag when building code (for "run" only)
}

// socketHandler handles the websocket connection for a given present session.
// It handles transcoding Messages to and from JSON format, and starting
// and killing processes.
func socketHandler(c *websocket.Conn) {
	in, out := make(chan *Message), make(chan *Message)
	errc := make(chan error, 1)

	// Decode messages from client and send to the in channel.
	go func() {
		dec := json.NewDecoder(c)
		for {
			var m Message
			if err := dec.Decode(&m); err != nil {
				errc <- err
				return
			}
			in <- &m
		}
	}()

	// Receive messages from the out channel and encode to the client.
	go func() {
		enc := json.NewEncoder(c)
		for m := range out {
			if err := enc.Encode(m); err != nil {
				errc <- err
				return
			}
		}
	}()

	// Start and kill processes and handle errors.
	proc := make(map[string]*process)
	for {
		select {
		case m := <-in:
			switch m.Kind {
			case "run":
				proc[m.Id].Kill()
				lOut := limiter(in, out)
				proc[m.Id] = startProcess(m.Id, m.Body, lOut, m.Options)
			case "kill":
				proc[m.Id].Kill()
			}
		case err := <-errc:
			if err != io.EOF {
				// A encode or decode has failed; bail.
				log.Println(err)
			}
			// Shut down any running processes.
			for _, p := range proc {
				p.Kill()
			}
			return
		}
	}
}

// process represents a running process.
type process struct {
	id   string
	out  chan<- *Message
	done chan struct{} // closed when wait completes
	run  *exec.Cmd
	bin  string
}

// startProcess builds and runs the given program, sending its output
// and end event as Messages on the provided channel.
func startProcess(id, body string, out chan<- *Message, opt *Options) *process {
	p := &process{
		id:   id,
		out:  out,
		done: make(chan struct{}),
	}
	var err error
	if path, args := shebang(body); path != "" {
		err = p.startProcess(path, args, body)
	} else {
		err = p.start(body, opt)
	}
	if err != nil {
		p.end(err)
		return nil
	}
	go p.wait()
	return p
}

// Kill stops the process if it is running and waits for it to exit.
func (p *process) Kill() {
	if p == nil {
		return
	}
	p.run.Process.Kill()
	<-p.done // block until process exits
}

// shebang looks for a shebang ('#!') at the beginning of the passed string.
// If found, it returns the path and args after the shebang.
func shebang(body string) (path string, args []string) {
	body = strings.TrimSpace(body)
	if !strings.HasPrefix(body, "#!") {
		return "", nil
	}
	if i := strings.Index(body, "\n"); i >= 0 {
		body = body[:i]
	}
	fs := strings.Fields(body[2:])
	return fs[0], fs[1:]
}

// startProcess starts a given program given its path and passing the given body
// to the command standard input.
func (p *process) startProcess(path string, args []string, body string) error {
	cmd := &exec.Cmd{
		Path:   path,
		Args:   args,
		Stdin:  strings.NewReader(body),
		Stdout: &messageWriter{id: p.id, kind: "stdout", out: p.out},
		Stderr: &messageWriter{id: p.id, kind: "stderr", out: p.out},
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	p.run = cmd
	return nil
}

// start builds and starts the given program, sending its output to p.out,
// and stores the running *exec.Cmd in the run field.
func (p *process) start(body string, opt *Options) error {
	// We "go build" and then exec the binary so that the
	// resultant *exec.Cmd is a handle to the user's program
	// (rather than the go tool process).
	// This makes Kill work.

	bin := filepath.Join(tmpdir, "compile"+strconv.Itoa(<-uniq))
	src := bin + ".go"
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}

	// write body to x.go
	defer os.Remove(src)
	err := ioutil.WriteFile(src, []byte(body), 0666)
	if err != nil {
		return err
	}

	// build x.go, creating x
	p.bin = bin // to be removed by p.end
	dir, file := filepath.Split(src)
	args := []string{"go", "build", "-tags", "OMIT"}
	if opt != nil && opt.Race {
		p.out <- &Message{
			Id: p.id, Kind: "stderr",
			Body: "Running with race detector.\n",
		}
		args = append(args, "-race")
	}
	args = append(args, "-o", bin, file)
	cmd := p.cmd(dir, args...)
	cmd.Stdout = cmd.Stderr // send compiler output to stderr
	if err := cmd.Run(); err != nil {
		return err
	}

	// run x
	cmd = p.cmd("", bin)
	if opt != nil && opt.Race {
		cmd.Env = append(cmd.Env, "GOMAXPROCS=2")
	}
	if err := cmd.Start(); err != nil {
		// If we failed to exec, that might be because they built
		// a non-main package instead of an executable.
		// Check and report that.
		if name, err := packageName(body); err == nil && name != "main" {
			return errors.New(`executable programs must use "package main"`)
		}
		return err
	}
	p.run = cmd
	return nil
}

// wait waits for the running process to complete
// and sends its error state to the client.
func (p *process) wait() {
	p.end(p.run.Wait())
	close(p.done) // unblock waiting Kill calls
}

// end sends an "end" message to the client, containing the process id and the
// given error value. It also removes the binary.
func (p *process) end(err error) {
	if p.bin != "" {
		defer os.Remove(p.bin)
	}
	m := &Message{Id: p.id, Kind: "end"}
	if err != nil {
		m.Body = err.Error()
	}
	// Wait for any outstanding reads to finish (potential race here).
	time.AfterFunc(msgDelay, func() { p.out <- m })
}

// cmd builds an *exec.Cmd that writes its standard output and error to the
// process' output channel.
func (p *process) cmd(dir string, args ...string) *exec.Cmd {
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir = dir
	cmd.Env = Environ()
	cmd.Stdout = &messageWriter{id: p.id, kind: "stdout", out: p.out}
	cmd.Stderr = &messageWriter{id: p.id, kind: "stderr", out: p.out}
	return cmd
}

func packageName(body string) (string, error) {
	f, err := parser.ParseFile(token.NewFileSet(), "prog.go",
		strings.NewReader(body), parser.PackageClauseOnly)
	if err != nil {
		return "", err
	}
	return f.Name.String(), nil
}

// messageWriter is an io.Writer that converts all writes to Message sends on
// the out channel with the specified id and kind.
type messageWriter struct {
	id, kind string
	out      chan<- *Message

	mu   sync.Mutex
	buf  []byte
	send *time.Timer
}

func (w *messageWriter) Write(b []byte) (n int, err error) {
	// Buffer writes that occur in a short period to send as one Message.
	w.mu.Lock()
	w.buf = append(w.buf, b...)
	if w.send == nil {
		w.send = time.AfterFunc(msgDelay, w.sendNow)
	}
	w.mu.Unlock()
	return len(b), nil
}

func (w *messageWriter) sendNow() {
	w.mu.Lock()
	body := safeString(w.buf)
	w.buf, w.send = nil, nil
	w.mu.Unlock()
	w.out <- &Message{Id: w.id, Kind: w.kind, Body: body}
}

// safeString returns b as a valid UTF-8 string.
func safeString(b []byte) string {
	if utf8.Valid(b) {
		return string(b)
	}
	var buf bytes.Buffer
	for len(b) > 0 {
		r, size := utf8.DecodeRune(b)
		b = b[size:]
		buf.WriteRune(r)
	}
	return buf.String()
}

// limiter returns a channel that wraps dest. Messages sent to the channel are
// sent to dest. After msgLimit Messages have been passed on, a "kill" Message
// is sent to the kill channel, and only "end" messages are passed.
func limiter(kill chan<- *Message, dest chan<- *Message) chan<- *Message {
	ch := make(chan *Message)
	go func() {
		n := 0
		for m := range ch {
			switch {
			case n < msgLimit || m.Kind == "end":
				dest <- m
				if m.Kind == "end" {
					return
				}
			case n == msgLimit:
				// process produced too much output. Kill it.
				kill <- &Message{Id: m.Id, Kind: "kill"}
			}
			n++
		}
	}()
	return ch
}

var tmpdir string

func init() {
	// find real path to temporary directory
	var err error
	tmpdir, err = filepath.EvalSymlinks(os.TempDir())
	if err != nil {
		log.Fatal(err)
	}
}

var uniq = make(chan int) // a source of numbers for naming temporary files

func init() {
	go func() {
		for i := 0; ; i++ {
			uniq <- i
		}
	}()
}
