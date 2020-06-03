// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"io"

	"golang.org/x/tools/internal/event/label"
)

var (
	KeyCreateSession   = NewSessionKey("create_session", "A new session was added")
	KeyUpdateSession   = NewSessionKey("update_session", "Updated information about a session")
	KeyShutdownSession = NewSessionKey("shutdown_session", "A session was shut down")
)

// SessionKey represents an event label key that has a *Session value.
type SessionKey struct {
	name        string
	description string
}

// NewSessionKey creates a new Key for *Session values.
func NewSessionKey(name, description string) *SessionKey {
	return &SessionKey{name: name, description: description}
}

func (k *SessionKey) Name() string        { return k.name }
func (k *SessionKey) Description() string { return k.description }

func (k *SessionKey) Format(w io.Writer, buf []byte, l label.Label) {
	io.WriteString(w, k.From(l).ID())
}

// Of creates a new Label with this key and the supplied session.
func (k *SessionKey) Of(v *Session) label.Label { return label.OfValue(k, v) }

// Get can be used to get the session for the key from a label.Map.
func (k *SessionKey) Get(lm label.Map) *Session {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return nil
}

// From can be used to get the session value from a Label.
func (k *SessionKey) From(t label.Label) *Session {
	err, _ := t.UnpackValue().(*Session)
	return err
}
