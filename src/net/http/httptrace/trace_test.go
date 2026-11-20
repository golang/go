// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httptrace

import (
	"context"
	"errors"
	"net/textproto"
	"strings"
	"testing"
)

func TestWithClientTrace(t *testing.T) {
	var buf strings.Builder
	connectStart := func(b byte) func(network, addr string) {
		return func(network, addr string) {
			buf.WriteByte(b)
		}
	}

	ctx := context.Background()
	oldtrace := &ClientTrace{
		ConnectStart: connectStart('O'),
	}
	ctx = WithClientTrace(ctx, oldtrace)
	newtrace := &ClientTrace{
		ConnectStart: connectStart('N'),
	}
	ctx = WithClientTrace(ctx, newtrace)
	trace := ContextClientTrace(ctx)

	buf.Reset()
	trace.ConnectStart("net", "addr")
	if got, want := buf.String(), "NO"; got != want {
		t.Errorf("got %q; want %q", got, want)
	}
}

func TestComposeGot1xxErrors(t *testing.T) {
	newErr, oldErr := errors.New("new hook"), errors.New("old hook")
	for _, errs := range [][2]error{{nil, nil}, {newErr, nil}, {nil, oldErr}, {newErr, oldErr}} {
		var calls strings.Builder
		ctx := WithClientTrace(context.Background(), &ClientTrace{
			Got1xxResponse: func(int, textproto.MIMEHeader) error {
				calls.WriteByte('O')
				return errs[1]
			},
		})
		ctx = WithClientTrace(ctx, &ClientTrace{
			Got1xxResponse: func(int, textproto.MIMEHeader) error {
				calls.WriteByte('N')
				return errs[0]
			},
		})
		err := ContextClientTrace(ctx).Got1xxResponse(103, nil)
		if (err == nil) != (errs[0] == nil && errs[1] == nil) {
			t.Errorf("hooks %v returned %v", errs, err)
		}
		for _, want := range errs {
			if want != nil && !errors.Is(err, want) {
				t.Errorf("hooks %v returned %v, want error containing %v", errs, err, want)
			}
		}
		if got := calls.String(); got != "NO" {
			t.Errorf("hook order = %q, want NO", got)
		}
	}
}

func TestCompose(t *testing.T) {
	var buf strings.Builder
	var testNum int

	connectStart := func(b byte) func(network, addr string) {
		return func(network, addr string) {
			if addr != "addr" {
				t.Errorf(`%d. args for %q case = %q, %q; want addr of "addr"`, testNum, b, network, addr)
			}
			buf.WriteByte(b)
		}
	}

	tests := [...]struct {
		trace, old *ClientTrace
		want       string
	}{
		0: {
			want: "T",
			trace: &ClientTrace{
				ConnectStart: connectStart('T'),
			},
		},
		1: {
			want: "TO",
			trace: &ClientTrace{
				ConnectStart: connectStart('T'),
			},
			old: &ClientTrace{ConnectStart: connectStart('O')},
		},
		2: {
			want:  "O",
			trace: &ClientTrace{},
			old:   &ClientTrace{ConnectStart: connectStart('O')},
		},
	}
	for i, tt := range tests {
		testNum = i
		buf.Reset()

		tr := *tt.trace
		tr.compose(tt.old)
		if tr.ConnectStart != nil {
			tr.ConnectStart("net", "addr")
		}
		if got := buf.String(); got != tt.want {
			t.Errorf("%d. got = %q; want %q", i, got, tt.want)
		}
	}

}
