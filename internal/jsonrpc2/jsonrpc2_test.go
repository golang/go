// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2_test

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net"
	"path"
	"reflect"
	"testing"

	"golang.org/x/tools/internal/event/export/eventtest"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/stack/stacktest"
)

var logRPC = flag.Bool("logrpc", false, "Enable jsonrpc2 communication logging")

type callTest struct {
	method string
	params interface{}
	expect interface{}
}

var callTests = []callTest{
	{"no_args", nil, true},
	{"one_string", "fish", "got:fish"},
	{"one_number", 10, "got:10"},
	{"join", []string{"a", "b", "c"}, "a/b/c"},
	//TODO: expand the test cases
}

func (test *callTest) newResults() interface{} {
	switch e := test.expect.(type) {
	case []interface{}:
		var r []interface{}
		for _, v := range e {
			r = append(r, reflect.New(reflect.TypeOf(v)).Interface())
		}
		return r
	case nil:
		return nil
	default:
		return reflect.New(reflect.TypeOf(test.expect)).Interface()
	}
}

func (test *callTest) verifyResults(t *testing.T, results interface{}) {
	if results == nil {
		return
	}
	val := reflect.Indirect(reflect.ValueOf(results)).Interface()
	if !reflect.DeepEqual(val, test.expect) {
		t.Errorf("%v:Results are incorrect, got %+v expect %+v", test.method, val, test.expect)
	}
}

func TestCall(t *testing.T) {
	stacktest.NoLeak(t)
	ctx := eventtest.NewContext(context.Background(), t)
	for _, headers := range []bool{false, true} {
		name := "Plain"
		if headers {
			name = "Headers"
		}
		t.Run(name, func(t *testing.T) {
			ctx := eventtest.NewContext(ctx, t)
			a, b, done := prepare(ctx, t, headers)
			defer done()
			for _, test := range callTests {
				t.Run(test.method, func(t *testing.T) {
					ctx := eventtest.NewContext(ctx, t)
					results := test.newResults()
					if _, err := a.Call(ctx, test.method, test.params, results); err != nil {
						t.Fatalf("%v:Call failed: %v", test.method, err)
					}
					test.verifyResults(t, results)
					if _, err := b.Call(ctx, test.method, test.params, results); err != nil {
						t.Fatalf("%v:Call failed: %v", test.method, err)
					}
					test.verifyResults(t, results)
				})
			}
		})
	}
}

func prepare(ctx context.Context, t *testing.T, withHeaders bool) (jsonrpc2.Conn, jsonrpc2.Conn, func()) {
	// make a wait group that can be used to wait for the system to shut down
	aPipe, bPipe := net.Pipe()
	a := run(ctx, withHeaders, aPipe)
	b := run(ctx, withHeaders, bPipe)
	return a, b, func() {
		a.Close()
		b.Close()
		<-a.Done()
		<-b.Done()
	}
}

func run(ctx context.Context, withHeaders bool, nc net.Conn) jsonrpc2.Conn {
	var stream jsonrpc2.Stream
	if withHeaders {
		stream = jsonrpc2.NewHeaderStream(nc)
	} else {
		stream = jsonrpc2.NewRawStream(nc)
	}
	conn := jsonrpc2.NewConn(stream)
	conn.Go(ctx, testHandler(*logRPC))
	return conn
}

func testHandler(log bool) jsonrpc2.Handler {
	return func(ctx context.Context, reply jsonrpc2.Replier, req jsonrpc2.Request) error {
		switch req.Method() {
		case "no_args":
			if len(req.Params()) > 0 {
				return reply(ctx, nil, fmt.Errorf("%w: expected no params", jsonrpc2.ErrInvalidParams))
			}
			return reply(ctx, true, nil)
		case "one_string":
			var v string
			if err := json.Unmarshal(req.Params(), &v); err != nil {
				return reply(ctx, nil, fmt.Errorf("%w: %s", jsonrpc2.ErrParse, err))
			}
			return reply(ctx, "got:"+v, nil)
		case "one_number":
			var v int
			if err := json.Unmarshal(req.Params(), &v); err != nil {
				return reply(ctx, nil, fmt.Errorf("%w: %s", jsonrpc2.ErrParse, err))
			}
			return reply(ctx, fmt.Sprintf("got:%d", v), nil)
		case "join":
			var v []string
			if err := json.Unmarshal(req.Params(), &v); err != nil {
				return reply(ctx, nil, fmt.Errorf("%w: %s", jsonrpc2.ErrParse, err))
			}
			return reply(ctx, path.Join(v...), nil)
		default:
			return jsonrpc2.MethodNotFound(ctx, reply, req)
		}
	}
}
