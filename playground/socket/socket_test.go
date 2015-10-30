// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package socket

import (
	"testing"
	"time"
)

func TestBuffer(t *testing.T) {
	ch := make(chan *Message)
	go func() {
		ch <- &Message{Kind: "err", Body: "a"}
		ch <- &Message{Kind: "err", Body: "b"}
		ch <- &Message{Kind: "out", Body: "1"}
		ch <- &Message{Kind: "out", Body: "2"}
		time.Sleep(msgDelay * 2)
		ch <- &Message{Kind: "out", Body: "3"}
		ch <- &Message{Kind: "out", Body: "4"}
		close(ch)
	}()

	var ms []*Message
	for m := range buffer(ch) {
		ms = append(ms, m)
	}
	if len(ms) != 3 {
		t.Fatalf("got %v messages, want 2", len(ms))
	}
	if g, w := ms[0].Body, "ab"; g != w {
		t.Errorf("message 0 body = %q, want %q", g, w)
	}
	if g, w := ms[1].Body, "12"; g != w {
		t.Errorf("message 1 body = %q, want %q", g, w)
	}
	if g, w := ms[2].Body, "34"; g != w {
		t.Errorf("message 2 body = %q, want %q", g, w)
	}
}

type killRecorder chan struct{}

func (k killRecorder) Kill() { close(k) }

func TestLimiter(t *testing.T) {
	ch := make(chan *Message)
	go func() {
		var m Message
		for i := 0; i < msgLimit+10; i++ {
			ch <- &m
		}
		ch <- &Message{Kind: "end"}
	}()

	kr := make(killRecorder)
	n := 0
	for m := range limiter(ch, kr) {
		n++
		if n > msgLimit && m.Kind != "end" {
			t.Errorf("received non-end message after limit")
		}
	}
	if n != msgLimit+1 {
		t.Errorf("received %v messages, want %v", n, msgLimit+1)
	}
	select {
	case <-kr:
	case <-time.After(100 * time.Millisecond):
		t.Errorf("process wasn't killed after reaching limit")
	}
}
