// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2

import (
	"encoding/json"
	"log"
	"time"
)

// Logger is an option you can pass to NewConn which is invoked for
// all messages flowing through a Conn.
// direction indicates if the message being recieved or sent
// id is the message id, if not set it was a notification
// elapsed is the time between a call being seen and the response, and is
// negative for anything that is not a response.
// method is the method name specified in the message
// payload is the parameters for a call or notification, and the result for a
// response
type Logger = func(direction Direction, id *ID, elapsed time.Duration, method string, payload *json.RawMessage, err *Error)

// Direction is used to indicate to a logger whether the logged message was being
// sent or received.
type Direction bool

const (
	// Send indicates the message is outgoing.
	Send = Direction(true)
	// Receive indicates the message is incoming.
	Receive = Direction(false)
)

func (d Direction) String() string {
	switch d {
	case Send:
		return "send"
	case Receive:
		return "receive"
	default:
		panic("unreachable")
	}
}

// Log is an implementation of Logger that outputs using log.Print
// It is not used by default, but is provided for easy logging in users code.
func Log(direction Direction, id *ID, elapsed time.Duration, method string, payload *json.RawMessage, err *Error) {
	switch {
	case err != nil:
		log.Printf("%v failure [%v] %s %v", direction, id, method, err)
	case id == nil:
		log.Printf("%v notification %s %s", direction, method, *payload)
	case elapsed >= 0:
		log.Printf("%v response in %v [%v] %s %s", direction, elapsed, id, method, *payload)
	default:
		log.Printf("%v call [%v] %s %s", direction, id, method, *payload)
	}
}
