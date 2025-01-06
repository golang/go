// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bufio"
	"cmd/go/internal/base"
	"cmd/go/internal/cacheprog"
	"cmd/internal/quoted"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"internal/goexperiment"
	"io"
	"log"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
	"time"
)

// ProgCache implements Cache via JSON messages over stdin/stdout to a child
// helper process which can then implement whatever caching policy/mechanism it
// wants.
//
// See https://github.com/golang/go/issues/59719
type ProgCache struct {
	cmd    *exec.Cmd
	stdout io.ReadCloser  // from the child process
	stdin  io.WriteCloser // to the child process
	bw     *bufio.Writer  // to stdin
	jenc   *json.Encoder  // to bw

	// can are the commands that the child process declared that it supports.
	// This is effectively the versioning mechanism.
	can map[cacheprog.Cmd]bool

	// fuzzDirCache is another Cache implementation to use for the FuzzDir
	// method. In practice this is the default GOCACHE disk-based
	// implementation.
	//
	// TODO(bradfitz): maybe this isn't ideal. But we'd need to extend the Cache
	// interface and the fuzzing callers to be less disk-y to do more here.
	fuzzDirCache Cache

	closing      atomic.Bool
	ctx          context.Context    // valid until Close via ctxClose
	ctxCancel    context.CancelFunc // called on Close
	readLoopDone chan struct{}      // closed when readLoop returns

	mu         sync.Mutex // guards following fields
	nextID     int64
	inFlight   map[int64]chan<- *cacheprog.Response
	outputFile map[OutputID]string // object => abs path on disk

	// writeMu serializes writing to the child process.
	// It must never be held at the same time as mu.
	writeMu sync.Mutex
}

// startCacheProg starts the prog binary (with optional space-separated flags)
// and returns a Cache implementation that talks to it.
//
// It blocks a few seconds to wait for the child process to successfully start
// and advertise its capabilities.
func startCacheProg(progAndArgs string, fuzzDirCache Cache) Cache {
	if fuzzDirCache == nil {
		panic("missing fuzzDirCache")
	}
	args, err := quoted.Split(progAndArgs)
	if err != nil {
		base.Fatalf("GOCACHEPROG args: %v", err)
	}
	var prog string
	if len(args) > 0 {
		prog = args[0]
		args = args[1:]
	}

	ctx, ctxCancel := context.WithCancel(context.Background())

	cmd := exec.CommandContext(ctx, prog, args...)
	out, err := cmd.StdoutPipe()
	if err != nil {
		base.Fatalf("StdoutPipe to GOCACHEPROG: %v", err)
	}
	in, err := cmd.StdinPipe()
	if err != nil {
		base.Fatalf("StdinPipe to GOCACHEPROG: %v", err)
	}
	cmd.Stderr = os.Stderr
	// On close, we cancel the context. Rather than killing the helper,
	// close its stdin.
	cmd.Cancel = in.Close

	if err := cmd.Start(); err != nil {
		base.Fatalf("error starting GOCACHEPROG program %q: %v", prog, err)
	}

	pc := &ProgCache{
		ctx:          ctx,
		ctxCancel:    ctxCancel,
		fuzzDirCache: fuzzDirCache,
		cmd:          cmd,
		stdout:       out,
		stdin:        in,
		bw:           bufio.NewWriter(in),
		inFlight:     make(map[int64]chan<- *cacheprog.Response),
		outputFile:   make(map[OutputID]string),
		readLoopDone: make(chan struct{}),
	}

	// Register our interest in the initial protocol message from the child to
	// us, saying what it can do.
	capResc := make(chan *cacheprog.Response, 1)
	pc.inFlight[0] = capResc

	pc.jenc = json.NewEncoder(pc.bw)
	go pc.readLoop(pc.readLoopDone)

	// Give the child process a few seconds to report its capabilities. This
	// should be instant and not require any slow work by the program.
	timer := time.NewTicker(5 * time.Second)
	defer timer.Stop()
	for {
		select {
		case <-timer.C:
			log.Printf("# still waiting for GOCACHEPROG %v ...", prog)
		case capRes := <-capResc:
			can := map[cacheprog.Cmd]bool{}
			for _, cmd := range capRes.KnownCommands {
				can[cmd] = true
			}
			if len(can) == 0 {
				base.Fatalf("GOCACHEPROG %v declared no supported commands", prog)
			}
			pc.can = can
			return pc
		}
	}
}

func (c *ProgCache) readLoop(readLoopDone chan<- struct{}) {
	defer close(readLoopDone)
	jd := json.NewDecoder(c.stdout)
	for {
		res := new(cacheprog.Response)
		if err := jd.Decode(res); err != nil {
			if c.closing.Load() {
				c.mu.Lock()
				for _, ch := range c.inFlight {
					close(ch)
				}
				c.inFlight = nil
				c.mu.Unlock()
				return // quietly
			}
			if err == io.EOF {
				c.mu.Lock()
				inFlight := len(c.inFlight)
				c.mu.Unlock()
				base.Fatalf("GOCACHEPROG exited pre-Close with %v pending requests", inFlight)
			}
			base.Fatalf("error reading JSON from GOCACHEPROG: %v", err)
		}
		c.mu.Lock()
		ch, ok := c.inFlight[res.ID]
		delete(c.inFlight, res.ID)
		c.mu.Unlock()
		if ok {
			ch <- res
		} else {
			base.Fatalf("GOCACHEPROG sent response for unknown request ID %v", res.ID)
		}
	}
}

var errCacheprogClosed = errors.New("GOCACHEPROG program closed unexpectedly")

func (c *ProgCache) send(ctx context.Context, req *cacheprog.Request) (*cacheprog.Response, error) {
	resc := make(chan *cacheprog.Response, 1)
	if err := c.writeToChild(req, resc); err != nil {
		return nil, err
	}
	select {
	case res := <-resc:
		if res == nil {
			return nil, errCacheprogClosed
		}
		if res.Err != "" {
			return nil, errors.New(res.Err)
		}
		return res, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (c *ProgCache) writeToChild(req *cacheprog.Request, resc chan<- *cacheprog.Response) (err error) {
	c.mu.Lock()
	if c.inFlight == nil {
		return errCacheprogClosed
	}
	c.nextID++
	req.ID = c.nextID
	c.inFlight[req.ID] = resc
	c.mu.Unlock()

	defer func() {
		if err != nil {
			c.mu.Lock()
			if c.inFlight != nil {
				delete(c.inFlight, req.ID)
			}
			c.mu.Unlock()
		}
	}()

	c.writeMu.Lock()
	defer c.writeMu.Unlock()

	if err := c.jenc.Encode(req); err != nil {
		return err
	}
	if err := c.bw.WriteByte('\n'); err != nil {
		return err
	}
	if req.Body != nil && req.BodySize > 0 {
		if err := c.bw.WriteByte('"'); err != nil {
			return err
		}
		e := base64.NewEncoder(base64.StdEncoding, c.bw)
		wrote, err := io.Copy(e, req.Body)
		if err != nil {
			return err
		}
		if err := e.Close(); err != nil {
			return nil
		}
		if wrote != req.BodySize {
			return fmt.Errorf("short write writing body to GOCACHEPROG for action %x, output %x: wrote %v; expected %v",
				req.ActionID, req.OutputID, wrote, req.BodySize)
		}
		if _, err := c.bw.WriteString("\"\n"); err != nil {
			return err
		}
	}
	if err := c.bw.Flush(); err != nil {
		return err
	}
	return nil
}

func (c *ProgCache) Get(a ActionID) (Entry, error) {
	if !c.can[cacheprog.CmdGet] {
		// They can't do a "get". Maybe they're a write-only cache.
		//
		// TODO(bradfitz,bcmills): figure out the proper error type here. Maybe
		// errors.ErrUnsupported? Is entryNotFoundError even appropriate? There
		// might be places where we rely on the fact that a recent Put can be
		// read through a corresponding Get. Audit callers and check, and document
		// error types on the Cache interface.
		return Entry{}, &entryNotFoundError{}
	}
	res, err := c.send(c.ctx, &cacheprog.Request{
		Command:  cacheprog.CmdGet,
		ActionID: a[:],
	})
	if err != nil {
		return Entry{}, err // TODO(bradfitz): or entryNotFoundError? Audit callers.
	}
	if res.Miss {
		return Entry{}, &entryNotFoundError{}
	}
	e := Entry{
		Size: res.Size,
	}
	if res.Time != nil {
		e.Time = *res.Time
	} else {
		e.Time = time.Now()
	}
	if res.DiskPath == "" {
		return Entry{}, &entryNotFoundError{errors.New("GOCACHEPROG didn't populate DiskPath on get hit")}
	}
	if copy(e.OutputID[:], res.OutputID) != len(res.OutputID) {
		return Entry{}, &entryNotFoundError{errors.New("incomplete ProgResponse OutputID")}
	}
	c.noteOutputFile(e.OutputID, res.DiskPath)
	return e, nil
}

func (c *ProgCache) noteOutputFile(o OutputID, diskPath string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.outputFile[o] = diskPath
}

func (c *ProgCache) OutputFile(o OutputID) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.outputFile[o]
}

func (c *ProgCache) Put(a ActionID, file io.ReadSeeker) (_ OutputID, size int64, _ error) {
	// Compute output ID.
	h := sha256.New()
	if _, err := file.Seek(0, 0); err != nil {
		return OutputID{}, 0, err
	}
	size, err := io.Copy(h, file)
	if err != nil {
		return OutputID{}, 0, err
	}
	var out OutputID
	h.Sum(out[:0])

	if _, err := file.Seek(0, 0); err != nil {
		return OutputID{}, 0, err
	}

	if !c.can[cacheprog.CmdPut] {
		// Child is a read-only cache. Do nothing.
		return out, size, nil
	}

	// For compatibility with Go 1.23/1.24 GOEXPERIMENT=gocacheprog users, also
	// populate the deprecated ObjectID field. This will be removed in Go 1.25.
	var deprecatedValue []byte
	if goexperiment.CacheProg {
		deprecatedValue = out[:]
	}

	res, err := c.send(c.ctx, &cacheprog.Request{
		Command:  cacheprog.CmdPut,
		ActionID: a[:],
		OutputID: out[:],
		ObjectID: deprecatedValue, // TODO(bradfitz): remove in Go 1.25
		Body:     file,
		BodySize: size,
	})
	if err != nil {
		return OutputID{}, 0, err
	}
	if res.DiskPath == "" {
		return OutputID{}, 0, errors.New("GOCACHEPROG didn't return DiskPath in put response")
	}
	c.noteOutputFile(out, res.DiskPath)
	return out, size, err
}

func (c *ProgCache) Close() error {
	c.closing.Store(true)
	var err error

	// First write a "close" message to the child so it can exit nicely
	// and clean up if it wants. Only after that exchange do we cancel
	// the context that kills the process.
	if c.can[cacheprog.CmdClose] {
		_, err = c.send(c.ctx, &cacheprog.Request{Command: cacheprog.CmdClose})
		if errors.Is(err, errCacheprogClosed) {
			// Allow the child to quit without responding to close.
			err = nil
		}
	}
	// Cancel the context, which will close the helper's stdin.
	c.ctxCancel()
	// Wait until the helper closes its stdout.
	<-c.readLoopDone
	return err
}

func (c *ProgCache) FuzzDir() string {
	// TODO(bradfitz): figure out what to do here. For now just use the
	// disk-based default.
	return c.fuzzDirCache.FuzzDir()
}
