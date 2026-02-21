// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cacheprog defines the protocol for a GOCACHEPROG program.
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
// The subprocess should immediately send a [Response] with its capabilities.
// After that, the go command will send a stream of [Request] messages and the
// subprocess should reply to each [Request] with a [Response] message.
package cacheprog

import (
	"io"
	"time"
)

// Cmd is a command that can be issued to a child process.
//
// If the interface needs to grow, the go command can add new commands or new
// versioned commands like "get2" in the future. The initial [Response] from
// the child process indicates which commands it supports.
type Cmd string

const (
	// CmdPut tells the cache program to store an object in the cache.
	//
	// [Request.ActionID] is the cache key of this object. The cache should
	// store [Request.OutputID] and [Request.Body] under this key for a
	// later "get" request. It must also store the Body in a file in the local
	// file system and return the path to that file in [Response.DiskPath],
	// which must exist at least until a "close" request.
	CmdPut = Cmd("put")

	// CmdGet tells the cache program to retrieve an object from the cache.
	//
	// [Request.ActionID] specifies the key of the object to get. If the
	// cache does not contain this object, it should set [Response.Miss] to
	// true. Otherwise, it should populate the fields of [Response],
	// including setting [Response.OutputID] to the OutputID of the original
	// "put" request and [Response.DiskPath] to the path of a local file
	// containing the Body of the original "put" request. That file must
	// continue to exist at least until a "close" request.
	CmdGet = Cmd("get")

	// CmdClose requests that the cache program exit gracefully.
	//
	// The cache program should reply to this request and then exit
	// (thus closing its stdout).
	CmdClose = Cmd("close")
)

// Request is the JSON-encoded message that's sent from the go command to
// the GOCACHEPROG child process over stdin. Each JSON object is on its own
// line. A ProgRequest of Type "put" with BodySize > 0 will be followed by a
// line containing a base64-encoded JSON string literal of the body.
type Request struct {
	// ID is a unique number per process across all requests.
	// It must be echoed in the Response from the child.
	ID int64

	// Command is the type of request.
	// The go command will only send commands that were declared
	// as supported by the child.
	Command Cmd

	// ActionID is the cache key for "put" and "get" requests.
	ActionID []byte `json:",omitempty"` // or nil if not used

	// OutputID is stored with the body for "put" requests.
	OutputID []byte `json:",omitempty"` // or nil if not used

	// Body is the body for "put" requests. It's sent after the JSON object
	// as a base64-encoded JSON string when BodySize is non-zero.
	// It's sent as a separate JSON value instead of being a struct field
	// send in this JSON object so large values can be streamed in both directions.
	// The base64 string body of a Request will always be written
	// immediately after the JSON object and a newline.
	Body io.Reader `json:"-"`

	// BodySize is the number of bytes of Body. If zero, the body isn't written.
	BodySize int64 `json:",omitempty"`
}

// Response is the JSON response from the child process to the go command.
//
// With the exception of the first protocol message that the child writes to its
// stdout with ID==0 and KnownCommands populated, these are only sent in
// response to a Request from the go command.
//
// Responses can be sent in any order. The ID must match the request they're
// replying to.
type Response struct {
	ID  int64  // that corresponds to Request; they can be answered out of order
	Err string `json:",omitempty"` // if non-empty, the error

	// KnownCommands is included in the first message that cache helper program
	// writes to stdout on startup (with ID==0). It includes the
	// Request.Command types that are supported by the program.
	//
	// This lets the go command extend the protocol gracefully over time (adding
	// "get2", etc), or fail gracefully when needed. It also lets the go command
	// verify the program wants to be a cache helper.
	KnownCommands []Cmd `json:",omitempty"`

	// For "get" requests.

	Miss     bool       `json:",omitempty"` // cache miss
	OutputID []byte     `json:",omitempty"` // the OutputID stored with the body
	Size     int64      `json:",omitempty"` // body size in bytes
	Time     *time.Time `json:",omitempty"` // when the object was put in the cache (optional; used for cache expiration)

	// For "get" and "put" requests.

	// DiskPath is the absolute path on disk of the body corresponding to a
	// "get" (on cache hit) or "put" request's ActionID.
	// The filename in DiskPath should not contain a file extension to ensure
	// compatibility with tools that filter files based on extensions.
	DiskPath string `json:",omitempty"`
}
