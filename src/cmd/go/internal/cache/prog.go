// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bufio"
	"cmd/go/internal/base"
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
	can map[ProgCmd]bool

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
	inFlight   map[int64]chan<- *ProgResponse
	outputFile map[OutputID]string // object => abs path on disk

	// writeMu serializes writing to the child process.
	// It must never be held at the same time as mu.
	writeMu sync.Mutex
}

// The following types define the protocol for a GOCACHEPROG program.
//
// By default, the go command manages a build cache stored in the file system
// itself. GOCACHEPROG can be set to the name of a command (with optional
// space-separated flags) that implements the go command build cache externally.
// This permits defining a different cache policy.
//
// The go command will start the GOCACHEPROG as a subprocess and communicate
// with it via JSON messages over stdin/stdout. The subprocess's stderr will be
// connected to the go command's stderr.
//
// The subprocess should immediately send a [ProgResponse] with its capabilities.
// After that, the go command will send a stream of [ProgRequest] messages and the
// subprocess should reply to each [ProgRequest] with a [ProgResponse] message.

// ProgCmd is a command that can be issued to a child process.
//
// If the interface needs to grow, the go command can add new commands or new
// versioned commands like "get2" in the future. The initial [ProgResponse] from
// the child process indicates which commands it supports.
type ProgCmd string

const (
	// cmdPut tells the cache program to store an object in the cache.
	//
	// [ProgRequest.ActionID] is the cache key of this object. The cache should
	// store [ProgRequest.OutputID] and [ProgRequest.Body] under this key for a
	// later "get" request. It must also store the Body in a file in the local
	// file system and return the path to that file in [ProgResponse.DiskPath],
	// which must exist at least until a "close" request.
	cmdPut = ProgCmd("put")

	// cmdGet tells the cache program to retrieve an object from the cache.
	//
	// [ProgRequest.ActionID] specifies the key of the object to get. If the
	// cache does not contain this object, it should set [ProgResponse.Miss] to
	// true. Otherwise, it should populate the fields of [ProgResponse],
	// including setting [ProgResponse.OutputID] to the OutputID of the original
	// "put" request and [ProgResponse.DiskPath] to the path of a local file
	// containing the Body of the original "put" request. That file must
	// continue to exist at least until a "close" request.
	cmdGet = ProgCmd("get")

	// cmdClose requests that the cache program exit gracefully.
	//
	// The cache program should reply to this request and then exit
	// (thus closing its stdout).
	cmdClose = ProgCmd("close")
)

// ProgRequest is the JSON-encoded message that's sent from the go command to
// the GOCACHEPROG child process over stdin. Each JSON object is on its own
// line. A ProgRequest of Type "put" with BodySize > 0 will be followed by a
// line containing a base64-encoded JSON string literal of the body.
type ProgRequest struct {
	// ID is a unique number per process across all requests.
	// It must be echoed in the ProgResponse from the child.
	ID int64

	// Command is the type of request.
	// The go command will only send commands that were declared
	// as supported by the child.
	Command ProgCmd

	// ActionID is the cache key for "put" and "get" requests.
	ActionID []byte `json:",omitempty"` // or nil if not used

	// OutputID is stored with the body for "put" requests.
	//
	// Prior to Go 1.24, when GOCACHEPROG was still an experiment, this was
	// accidentally named ObjectID. It was renamed to OutputID in Go 1.24.
	OutputID []byte `json:",omitempty"` // or nil if not used

	// Body is the body for "put" requests. It's sent after the JSON object
	// as a base64-encoded JSON string when BodySize is non-zero.
	// It's sent as a separate JSON value instead of being a struct field
	// send in this JSON object so large values can be streamed in both directions.
	// The base64 string body of a ProgRequest will always be written
	// immediately after the JSON object and a newline.
	Body io.Reader `json:"-"`

	// BodySize is the number of bytes of Body. If zero, the body isn't written.
	BodySize int64 `json:",omitempty"`

	// ObjectID is the accidental spelling of OutputID that was used prior to Go
	// 1.24.
	//
	// Deprecated: use OutputID. This field is only populated temporarily for
	// backwards compatibility with Go 1.23 and earlier when
	// GOEXPERIMENT=gocacheprog is set. It will be removed in Go 1.25.
	ObjectID []byte `json:",omitempty"`
}

// ProgResponse is the JSON response from the child process to the go command.
//
// With the exception of the first protocol message that the child writes to its
// stdout with ID==0 and KnownCommands populated, these are only sent in
// response to a ProgRequest from the go command.
//
// ProgResponses can be sent in any order. The ID must match the request they're
// replying to.
type ProgResponse struct {
	ID  int64  // that corresponds to ProgRequest; they can be answered out of order
	Err string `json:",omitempty"` // if non-empty, the error

	// KnownCommands is included in the first message that cache helper program
	// writes to stdout on startup (with ID==0). It includes the
	// ProgRequest.Command types that are supported by the program.
	//
	// This lets the go command extend the protocol gracefully over time (adding
	// "get2", etc), or fail gracefully when needed. It also lets the go command
	// verify the program wants to be a cache helper.
	KnownCommands []ProgCmd `json:",omitempty"`

	// For "get" requests.

	Miss     bool       `json:",omitempty"` // cache miss
	OutputID []byte     `json:",omitempty"` // the ObjectID stored with the body
	Size     int64      `json:",omitempty"` // body size in bytes
	Time     *time.Time `json:",omitempty"` // when the object was put in the cache (optional; used for cache expiration)

	// For "get" and "put" requests.

	// DiskPath is the absolute path on disk of the body corresponding to a
	// "get" (on cache hit) or "put" request's ActionID.
	DiskPath string `json:",omitempty"`
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
		inFlight:     make(map[int64]chan<- *ProgResponse),
		outputFile:   make(map[OutputID]string),
		readLoopDone: make(chan struct{}),
	}

	// Register our interest in the initial protocol message from the child to
	// us, saying what it can do.
	capResc := make(chan *ProgResponse, 1)
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
			can := map[ProgCmd]bool{}
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
		res := new(ProgResponse)
		if err := jd.Decode(res); err != nil {
			if c.closing.Load() {
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

func (c *ProgCache) send(ctx context.Context, req *ProgRequest) (*ProgResponse, error) {
	resc := make(chan *ProgResponse, 1)
	if err := c.writeToChild(req, resc); err != nil {
		return nil, err
	}
	select {
	case res := <-resc:
		if res.Err != "" {
			return nil, errors.New(res.Err)
		}
		return res, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (c *ProgCache) writeToChild(req *ProgRequest, resc chan<- *ProgResponse) (err error) {
	c.mu.Lock()
	c.nextID++
	req.ID = c.nextID
	c.inFlight[req.ID] = resc
	c.mu.Unlock()

	defer func() {
		if err != nil {
			c.mu.Lock()
			delete(c.inFlight, req.ID)
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
	if !c.can[cmdGet] {
		// They can't do a "get". Maybe they're a write-only cache.
		//
		// TODO(bradfitz,bcmills): figure out the proper error type here. Maybe
		// errors.ErrUnsupported? Is entryNotFoundError even appropriate? There
		// might be places where we rely on the fact that a recent Put can be
		// read through a corresponding Get. Audit callers and check, and document
		// error types on the Cache interface.
		return Entry{}, &entryNotFoundError{}
	}
	res, err := c.send(c.ctx, &ProgRequest{
		Command:  cmdGet,
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

	if !c.can[cmdPut] {
		// Child is a read-only cache. Do nothing.
		return out, size, nil
	}

	// For compatibility with Go 1.23/1.24 GOEXPERIMENT=gocacheprog users, also
	// populate the deprecated ObjectID field. This will be removed in Go 1.25.
	var deprecatedValue []byte
	if goexperiment.CacheProg {
		deprecatedValue = out[:]
	}

	res, err := c.send(c.ctx, &ProgRequest{
		Command:  cmdPut,
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
	if c.can[cmdClose] {
		_, err = c.send(c.ctx, &ProgRequest{Command: cmdClose})
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
