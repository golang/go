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
	"log"
	"path"
	"reflect"
	"testing"
	"time"

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
		if err := a.Call(ctx, test.method, test.params, results); err != nil {
			t.Fatalf("%v:Call failed: %v", test.method, err)
		}
		test.verifyResults(t, results)
		if err := b.Call(ctx, test.method, test.params, results); err != nil {
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
		if err := a.Call(ctx, test.method, test.params, results); err != nil {
			t.Fatalf("%v:Call failed: %v", test.method, err)
		}
		test.verifyResults(t, results)
		if err := b.Call(ctx, test.method, test.params, results); err != nil {
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
	conn.AddHandler(handle{})
	go func() {
		defer func() {
			r.Close()
			w.Close()
		}()
		if err := conn.Run(ctx); err != nil {
			t.Fatalf("Stream failed: %v", err)
		}
	}()
	return conn
}

type handle struct{ jsonrpc2.EmptyHandler }

func (handle) Deliver(ctx context.Context, r *jsonrpc2.Request, delivered bool) bool {
	switch r.Method {
	case "no_args":
		if r.Params != nil {
			r.Reply(ctx, nil, jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidParams, "Expected no params"))
			return true
		}
		r.Reply(ctx, true, nil)
	case "one_string":
		var v string
		if err := json.Unmarshal(*r.Params, &v); err != nil {
			r.Reply(ctx, nil, jsonrpc2.NewErrorf(jsonrpc2.CodeParseError, "%v", err.Error()))
			return true
		}
		r.Reply(ctx, "got:"+v, nil)
	case "one_number":
		var v int
		if err := json.Unmarshal(*r.Params, &v); err != nil {
			r.Reply(ctx, nil, jsonrpc2.NewErrorf(jsonrpc2.CodeParseError, "%v", err.Error()))
			return true
		}
		r.Reply(ctx, fmt.Sprintf("got:%d", v), nil)
	case "join":
		var v []string
		if err := json.Unmarshal(*r.Params, &v); err != nil {
			r.Reply(ctx, nil, jsonrpc2.NewErrorf(jsonrpc2.CodeParseError, "%v", err.Error()))
			return true
		}
		r.Reply(ctx, path.Join(v...), nil)
	default:
		r.Reply(ctx, nil, jsonrpc2.NewErrorf(jsonrpc2.CodeMethodNotFound, "method %q not found", r.Method))
	}
	return true
}

func (handle) Log(direction jsonrpc2.Direction, id *jsonrpc2.ID, elapsed time.Duration, method string, payload *json.RawMessage, err *jsonrpc2.Error) {
	if !*logRPC {
		return
	}
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
