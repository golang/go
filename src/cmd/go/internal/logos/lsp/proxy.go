// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
	"time"
)

const stopGrace = 3 * time.Second

const drainLimit = 30 * time.Second

const outboxCap = 64

const outboxStall = 2 * time.Second

const maxPending = 1024

const maxDiagURIs = 4096

const maxDocBytes = 8 << 20

const maxLowerErrs = 64

const maxHintTasks = 4

type Options struct {
	Gopls []string

	Env []string

	Stderr io.Writer

	Compiler Compiler

	Hints     HintsFunc
	HintsJSON HintsJSONFunc

	CodeActions CodeActionsFunc
}

type virtDoc struct {
	lw      *lowered
	version int
}

type serverDiags struct {
	version int
	hasVer  bool
	raw     json.RawMessage
}

type pending struct {
	method     string
	path       string
	lw         *lowered
	tb         *tables
	params     json.RawMessage
	fixVersion int
	fixHas     bool
}

type Proxy struct {
	opts   Options
	client *writer
	server *writer
	tables atomic.Pointer[tables]

	work sync.Mutex

	pub sync.Mutex

	tasks   sync.WaitGroup
	hintSem chan struct{}
	st      store

	mu            sync.Mutex
	files         map[string]*lowered
	virt          map[string]*virtDoc
	rootsMu       sync.RWMutex
	roots         []string
	pending       map[string]pending
	srvDiags      map[string]serverDiags
	sent          map[string]string
	lowerErr      map[string]error
	toClient      chan *message
	toServer      chan *message
	done          chan struct{}
	cancel        context.CancelFunc
	in            io.Reader
	sendErr       error
	clientPending atomic.Int64
	clientServed  atomic.Int64
	serverServed  atomic.Int64
	failed        atomic.Bool
	stopMu        sync.Mutex
	stopped       bool
	shutdown      bool
	ended         bool
	endedErr      error
	settled       chan struct{}

	lastLogErr error
}

func newProxy(out io.Writer, opts Options) *Proxy {
	if opts.Stderr == nil {
		opts.Stderr = os.Stderr
	}
	p := &Proxy{
		opts:     opts,
		client:   &writer{w: out},
		files:    make(map[string]*lowered),
		virt:     make(map[string]*virtDoc),
		pending:  make(map[string]pending),
		srvDiags: make(map[string]serverDiags),
		sent:     make(map[string]string),
		lowerErr: make(map[string]error),
		toClient: make(chan *message, outboxCap),
		toServer: make(chan *message, outboxCap),
		done:     make(chan struct{}),
		hintSem:  make(chan struct{}, maxHintTasks),
		settled:  make(chan struct{}, 1),
	}
	p.st.init()
	p.tables.Store(&tables{})
	return p
}

func (p *Proxy) logf(format string, args ...any) {
	if _, err := fmt.Fprintf(p.opts.Stderr, "logos lsp: "+format+"\n", args...); err != nil {
		p.mu.Lock()
		p.lastLogErr = errors.Join(p.lastLogErr, fmt.Errorf("writing lsp log: %w", err))
		p.mu.Unlock()
	}
}

type server struct {
	cmd  *exec.Cmd
	done chan struct{}
	err  error
}

func (s *server) reap(ctx context.Context) error {
	var errs []error
	killed := false
	select {
	case <-s.done:
	case <-time.After(stopGrace):
		if err := s.cmd.Process.Kill(); err == nil {
			killed = true
		} else if !errors.Is(err, os.ErrProcessDone) {
			errs = append(errs, fmt.Errorf("killing gopls: %w", err))
		}
		<-s.done
	}
	if err := s.cmd.Wait(); err != nil {
		_, exited := errors.AsType[*exec.ExitError](err)
		signaled := (killed || ctx.Err() != nil) && (exited || errors.Is(err, ctx.Err()))
		if !signaled {
			errs = append(errs, fmt.Errorf("waiting for gopls: %w", err))
		}
	}
	return errors.Join(errs...)
}

func Serve(ctx context.Context, in io.Reader, out io.Writer, opts Options) error {
	if len(opts.Gopls) == 0 || opts.Compiler == nil {
		return errors.New("logos lsp: no gopls command or compiler")
	}
	workCtx, cancelWork := context.WithCancel(ctx)
	defer cancelWork()
	p := newProxy(out, opts)
	srv, err := p.start(ctx, workCtx)
	if err != nil {
		return err
	}
	p.mu.Lock()
	p.cancel, p.in = cancelWork, in
	p.mu.Unlock()
	go p.pump(p.client, p.toClient)
	go p.pump(p.server, p.toServer)
	clientDone := make(chan struct{})
	var clientErr error
	go func() {
		clientErr = p.clientLoop(workCtx, newReader(in))
		close(clientDone)
	}()
	select {
	case <-clientDone:
		err = clientErr
		if err == nil {
			p.drain(srv.done)
		}
	case <-srv.done:
		err = cmp.Or(srv.err, errors.New("gopls exited unexpectedly"))
		p.mu.Lock()
		bye, byeErr := p.ended, p.endedErr
		p.mu.Unlock()
		if bye {
			err = byeErr
		}
	}
	if err == nil {
		err = p.flushClient(workCtx)
	}
	p.stop(ctx, srv, cancelWork)
	p.closeInput(in, clientDone)
	p.mu.Lock()
	logErr := p.lastLogErr
	sendErr := p.sendErr
	p.mu.Unlock()
	return errors.Join(err, logErr, sendErr)
}

func (p *Proxy) closeInput(in io.Reader, loopDone <-chan struct{}) {
	if c, ok := in.(io.Closer); ok {
		if err := c.Close(); err != nil {
			p.logf("closing input: %v", err)
		}
	}
	select {
	case <-loopDone:
	case <-time.After(stopGrace):
	}
}

func (p *Proxy) start(ctx, workCtx context.Context) (*server, error) {
	cmd := exec.CommandContext(ctx, p.opts.Gopls[0], p.opts.Gopls[1:]...)
	cmd.Env = p.opts.Env
	cmd.Stderr = p.opts.Stderr
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting gopls: %w", err)
	}
	p.server = &writer{w: stdin}
	srv := &server{cmd: cmd, done: make(chan struct{})}
	go func() {
		srv.err = p.serverLoop(workCtx, newReader(stdout))
		close(srv.done)
	}()
	return srv, nil
}

func (p *Proxy) drain(gopls <-chan struct{}) {
	limit := time.After(drainLimit)
	for {
		p.mu.Lock()
		n := len(p.pending) + len(p.toClient) + len(p.toServer)
		p.mu.Unlock()
		if n == 0 {
			return
		}
		select {
		case <-p.settled:
		case <-gopls:
			return
		case <-limit:
			return
		}
	}
}

func (p *Proxy) stop(ctx context.Context, srv *server, cancelWork context.CancelFunc) {
	p.stopMu.Lock()
	p.stopped = true
	close(p.done)
	cancelWork()
	p.tasks.Wait()
	p.stopMu.Unlock()
	if c, ok := p.server.w.(io.Closer); ok {
		if err := c.Close(); err != nil {
			p.logf("closing gopls input: %v", err)
		}
	}
	if err := srv.reap(ctx); err != nil {
		p.logf("gopls: %v", err)
	}
	p.work.Lock()
	defer p.work.Unlock()
	if c, ok := p.opts.Compiler.(io.Closer); ok {
		if err := c.Close(); err != nil {
			p.logf("closing compiler: %v", err)
		}
	}
}

func (p *Proxy) clientLoop(ctx context.Context, rd *reader) error {
	for {
		m, err := rd.read()
		if _, ok := errors.AsType[*parseError](err); ok {
			p.fail(json.RawMessage("null"), codeParseError, "parse error")
			continue
		}
		if errors.Is(err, io.EOF) {
			p.mu.Lock()
			p.ended = true
			p.mu.Unlock()
			return nil
		}
		if err != nil {
			return err
		}
		if m.Method == "exit" {
			return p.exit(m)
		}
		p.fromClient(ctx, m)
	}
}

func (p *Proxy) exit(m *message) error {
	p.mu.Lock()
	endedErr := p.endedErr
	if !p.shutdown {
		endedErr = errors.New("logos lsp: exit without shutdown")
	}
	p.ended, p.endedErr = true, endedErr
	p.mu.Unlock()
	p.send(p.server, m)
	return endedErr
}

func (p *Proxy) serverLoop(ctx context.Context, rd *reader) error {
	for {
		m, err := rd.read()
		if _, ok := errors.AsType[*parseError](err); ok {
			p.logf("gopls sent a bad frame: %v", err)
			continue
		}
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		p.fromServer(ctx, m)
	}
}

func (p *Proxy) flushClient(ctx context.Context) error {
	limit := time.NewTimer(stopGrace)
	defer limit.Stop()
	for {
		n := p.clientPending.Load()
		if err := p.sendError(); err != nil {
			return err
		}
		if n == 0 {
			return nil
		}
		select {
		case <-p.settled:
		case <-ctx.Done():
			return ctx.Err()
		case <-limit.C:
			return errors.New("logos lsp: timed out delivering client replies")
		}
	}
}

func (p *Proxy) send(w *writer, m *message) {
	ch := p.toClient
	if w == p.server {
		ch = p.toServer
	}
	if w == p.client {
		p.clientPending.Add(1)
	}
	queued := false
	select {
	case ch <- m:
		queued = true
	case <-p.done:
	default:
		queued = p.enqueueBounded(w, ch, m)
	}
	if w == p.client && !queued {
		p.clientDelivered()
	}
}

func (p *Proxy) enqueueBounded(w *writer, ch chan *message, m *message) bool {
	if p.failed.Load() {
		return false
	}
	served := &p.serverServed
	if w == p.client {
		served = &p.clientServed
	}
	delivered := served.Load()
	timer := time.NewTimer(outboxStall)
	defer timer.Stop()
	for {
		select {
		case ch <- m:
			return true
		case <-timer.C:
		case <-p.done:
			return false
		}
		now := served.Load()
		if now == delivered {
			p.failSession(errors.New("logos lsp: outbound queue full"))
			return false
		}
		delivered = now
		timer.Reset(outboxStall)
	}
}

func (p *Proxy) clientDelivered() {
	p.clientPending.Add(-1)
	select {
	case p.settled <- struct{}{}:
	default:
	}
}

func (p *Proxy) pump(w *writer, ch chan *message) {
	served := &p.serverServed
	if w == p.client {
		served = &p.clientServed
	}
	for {
		select {
		case m := <-ch:
			if err := w.write(m); err != nil {
				p.failSession(err)
				return
			}
			served.Add(1)
			if w == p.client {
				p.clientDelivered()
			}
		case <-p.done:
			return
		}
	}
}

func (p *Proxy) fail(id json.RawMessage, code int, text string) {
	raw, err := json.Marshal(rpcError{Code: code, Message: text})
	if err != nil {
		p.logf("encoding an error: %v", err)
		return
	}
	p.send(p.client, &message{ID: id, Error: raw})
}

func (p *Proxy) failSession(err error) {
	p.failed.Store(true)
	if !p.stopMu.TryLock() {
		return
	}
	stopped := p.stopped
	p.stopMu.Unlock()
	if stopped {
		return
	}
	p.mu.Lock()
	if p.sendErr == nil {
		p.sendErr = err
	}
	cancel, in := p.cancel, p.in
	p.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if c, ok := in.(io.Closer); ok && c != nil {
		_ = c.Close()
	}
}

func (p *Proxy) sendError() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.sendErr
}

func (p *Proxy) addPending(id json.RawMessage, pend pending) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pending[string(id)] = pend
}

func (p *Proxy) finish(id json.RawMessage) {
	p.mu.Lock()
	delete(p.pending, string(id))
	p.mu.Unlock()
	select {
	case p.settled <- struct{}{}:
	default:
	}
}
