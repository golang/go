// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2

import (
	"encoding/json"
	"fmt"
)

// this file contains the go forms of the wire specification
// see http://www.jsonrpc.org/specification for details

var (
	// ErrUnknown should be used for all non coded errors.
	ErrUnknown = NewError(-32001, "JSON RPC unknown error")
	// ErrParse is used when invalid JSON was received by the server.
	ErrParse = NewError(-32700, "JSON RPC parse error")
	//ErrInvalidRequest is used when the JSON sent is not a valid Request object.
	ErrInvalidRequest = NewError(-32600, "JSON RPC invalid request")
	// ErrMethodNotFound should be returned by the handler when the method does
	// not exist / is not available.
	ErrMethodNotFound = NewError(-32601, "JSON RPC method not found")
	// ErrInvalidParams should be returned by the handler when method
	// parameter(s) were invalid.
	ErrInvalidParams = NewError(-32602, "JSON RPC invalid params")
	// ErrInternal is not currently returned but defined for completeness.
	ErrInternal = NewError(-32603, "JSON RPC internal error")

	//ErrServerOverloaded is returned when a message was refused due to a
	//server being temporarily unable to accept any new messages.
	ErrServerOverloaded = NewError(-32000, "JSON RPC overloaded")
)

// wireRequest is sent to a server to represent a Call or Notify operation.
type wireRequest struct {
	// VersionTag is always encoded as the string "2.0"
	VersionTag wireVersionTag `json:"jsonrpc"`
	// Method is a string containing the method name to invoke.
	Method string `json:"method"`
	// Params is either a struct or an array with the parameters of the method.
	Params *json.RawMessage `json:"params,omitempty"`
	// The id of this request, used to tie the Response back to the request.
	// Will be either a string or a number. If not set, the Request is a notify,
	// and no response is possible.
	ID *ID `json:"id,omitempty"`
}

// WireResponse is a reply to a Request.
// It will always have the ID field set to tie it back to a request, and will
// have either the Result or Error fields set depending on whether it is a
// success or failure response.
type wireResponse struct {
	// VersionTag is always encoded as the string "2.0"
	VersionTag wireVersionTag `json:"jsonrpc"`
	// Result is the response value, and is required on success.
	Result *json.RawMessage `json:"result,omitempty"`
	// Error is a structured error response if the call fails.
	Error *wireError `json:"error,omitempty"`
	// ID must be set and is the identifier of the Request this is a response to.
	ID *ID `json:"id,omitempty"`
}

// wireCombined has all the fields of both Request and Response.
// We can decode this and then work out which it is.
type wireCombined struct {
	VersionTag wireVersionTag   `json:"jsonrpc"`
	ID         *ID              `json:"id,omitempty"`
	Method     string           `json:"method"`
	Params     *json.RawMessage `json:"params,omitempty"`
	Result     *json.RawMessage `json:"result,omitempty"`
	Error      *wireError       `json:"error,omitempty"`
}

// wireError represents a structured error in a Response.
type wireError struct {
	// Code is an error code indicating the type of failure.
	Code int64 `json:"code"`
	// Message is a short description of the error.
	Message string `json:"message"`
	// Data is optional structured data containing additional information about the error.
	Data *json.RawMessage `json:"data,omitempty"`
}

// wireVersionTag is a special 0 sized struct that encodes as the jsonrpc version
// tag.
// It will fail during decode if it is not the correct version tag in the
// stream.
type wireVersionTag struct{}

// ID is a Request identifier.
type ID struct {
	name   string
	number int64
}

func NewError(code int64, message string) error {
	return &wireError{
		Code:    code,
		Message: message,
	}
}

func (err *wireError) Error() string {
	return err.Message
}

func (wireVersionTag) MarshalJSON() ([]byte, error) {
	return json.Marshal("2.0")
}

func (wireVersionTag) UnmarshalJSON(data []byte) error {
	version := ""
	if err := json.Unmarshal(data, &version); err != nil {
		return err
	}
	if version != "2.0" {
		return fmt.Errorf("invalid RPC version %v", version)
	}
	return nil
}

// NewIntID returns a new numerical request ID.
func NewIntID(v int64) ID { return ID{number: v} }

// NewStringID returns a new string request ID.
func NewStringID(v string) ID { return ID{name: v} }

// Format writes the ID to the formatter.
// If the rune is q the representation is non ambiguous,
// string forms are quoted, number forms are preceded by a #
func (id ID) Format(f fmt.State, r rune) {
	numF, strF := `%d`, `%s`
	if r == 'q' {
		numF, strF = `#%d`, `%q`
	}
	switch {
	case id.name != "":
		fmt.Fprintf(f, strF, id.name)
	default:
		fmt.Fprintf(f, numF, id.number)
	}
}

func (id *ID) MarshalJSON() ([]byte, error) {
	if id.name != "" {
		return json.Marshal(id.name)
	}
	return json.Marshal(id.number)
}

func (id *ID) UnmarshalJSON(data []byte) error {
	*id = ID{}
	if err := json.Unmarshal(data, &id.number); err == nil {
		return nil
	}
	return json.Unmarshal(data, &id.name)
}
