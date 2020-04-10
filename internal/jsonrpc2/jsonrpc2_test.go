// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2_test

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"path"
	"reflect"
	"testing"

	"golang.org/x/tools/internal/jsonrpc2"
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

func TestPlainCall(t *testing.T) {
	ctx := context.Background()
	a, b := prepare(ctx, t, false)
	for _, test := range callTests {
		results := test.newResults()
		if _, err := a.Call(ctx, test.method, test.params, results); err != nil {
			t.Fatalf("%v:Call failed: %v", test.method, err)
		}
		test.verifyResults(t, results)
		if _, err := b.Call(ctx, test.method, test.params, results); err != nil {
			t.Fatalf("%v:Call failed: %v", test.method, err)
		}
		test.verifyResults(t, results)
	}
}

func TestHeaderCall(t *testing.T) {
	ctx := context.Background()
	a, b := prepare(ctx, t, true)
	for _, test := range callTests {
		results := test.newResults()
		if _, err := a.Call(ctx, test.method, test.params, results); err != nil {
			t.Fatalf("%v:Call failed: %v", test.method, err)
		}
		test.verifyResults(t, results)
		if _, err := b.Call(ctx, test.method, test.params, results); err != nil {
			t.Fatalf("%v:Call failed: %v", test.method, err)
		}
		test.verifyResults(t, results)
	}
}

func prepare(ctx context.Context, t *testing.T, withHeaders bool) (*jsonrpc2.Conn, *jsonrpc2.Conn) {
	aR, bW := io.Pipe()
	bR, aW := io.Pipe()
	a := run(ctx, t, withHeaders, aR, aW)
	b := run(ctx, t, withHeaders, bR, bW)
	return a, b
}

func run(ctx context.Context, t *testing.T, withHeaders bool, r io.ReadCloser, w io.WriteCloser) *jsonrpc2.Conn {
	var stream jsonrpc2.Stream
	if withHeaders {
		stream = jsonrpc2.NewHeaderStream(r, w)
	} else {
		stream = jsonrpc2.NewStream(r, w)
	}
	conn := jsonrpc2.NewConn(stream)
	go func() {
		defer func() {
			r.Close()
			w.Close()
		}()
		if err := conn.Run(ctx, testHandler(*logRPC)); err != nil {
			t.Errorf("Stream failed: %v", err)
		}
	}()
	return conn
}

func testHandler(log bool) jsonrpc2.Handler {
	return func(ctx context.Context, reply jsonrpc2.Replier, req *jsonrpc2.Request) error {
		switch req.Method {
		case "no_args":
			if req.Params != nil {
				return reply(ctx, nil, fmt.Errorf("%w: expected no params", jsonrpc2.ErrInvalidParams))
			}
			return reply(ctx, true, nil)
		case "one_string":
			var v string
			if err := json.Unmarshal(*req.Params, &v); err != nil {
				return reply(ctx, nil, fmt.Errorf("%w: %s", jsonrpc2.ErrParse, err))
			}
			return reply(ctx, "got:"+v, nil)
		case "one_number":
			var v int
			if err := json.Unmarshal(*req.Params, &v); err != nil {
				return reply(ctx, nil, fmt.Errorf("%w: %s", jsonrpc2.ErrParse, err))
			}
			return reply(ctx, fmt.Sprintf("got:%d", v), nil)
		case "join":
			var v []string
			if err := json.Unmarshal(*req.Params, &v); err != nil {
				return reply(ctx, nil, fmt.Errorf("%w: %s", jsonrpc2.ErrParse, err))
			}
			return reply(ctx, path.Join(v...), nil)
		default:
			return jsonrpc2.MethodNotFound(ctx, reply, req)
		}
	}
}
