// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2_test

import (
	"context"
	"encoding/json"
	"fmt"
	"path"
	"reflect"
	"testing"
	"time"

	"golang.org/x/tools/internal/event/export/eventtest"
	jsonrpc2 "golang.org/x/tools/internal/jsonrpc2_v2"
	"golang.org/x/tools/internal/stack/stacktest"
)

var callTests = []invoker{
	call{"no_args", nil, true},
	call{"one_string", "fish", "got:fish"},
	call{"one_number", 10, "got:10"},
	call{"join", []string{"a", "b", "c"}, "a/b/c"},
	sequence{"notify", []invoker{
		notify{"set", 3},
		notify{"add", 5},
		call{"get", nil, 8},
	}},
	sequence{"preempt", []invoker{
		async{"a", "wait", "a"},
		notify{"unblock", "a"},
		collect{"a", true, false},
	}},
	sequence{"basic cancel", []invoker{
		async{"b", "wait", "b"},
		cancel{"b"},
		collect{"b", nil, true},
	}},
	sequence{"queue", []invoker{
		async{"a", "wait", "a"},
		notify{"set", 1},
		notify{"add", 2},
		notify{"add", 3},
		notify{"add", 4},
		call{"peek", nil, 0}, // accumulator will not have any adds yet
		notify{"unblock", "a"},
		collect{"a", true, false},
		call{"get", nil, 10}, // accumulator now has all the adds
	}},
	sequence{"fork", []invoker{
		async{"a", "fork", "a"},
		notify{"set", 1},
		notify{"add", 2},
		notify{"add", 3},
		notify{"add", 4},
		call{"get", nil, 10}, // fork will not have blocked the adds
		notify{"unblock", "a"},
		collect{"a", true, false},
	}},
	sequence{"concurrent", []invoker{
		async{"a", "fork", "a"},
		notify{"unblock", "a"},
		async{"b", "fork", "b"},
		notify{"unblock", "b"},
		collect{"a", true, false},
		collect{"b", true, false},
	}},
}

type binder struct {
	framer  jsonrpc2.Framer
	runTest func(*handler)
}

type handler struct {
	conn        *jsonrpc2.Connection
	accumulator int
	waitersBox  chan map[string]chan struct{}
	calls       map[string]*jsonrpc2.AsyncCall
}

type invoker interface {
	Name() string
	Invoke(t *testing.T, ctx context.Context, h *handler)
}

type notify struct {
	method string
	params interface{}
}

type call struct {
	method string
	params interface{}
	expect interface{}
}

type async struct {
	name   string
	method string
	params interface{}
}

type collect struct {
	name   string
	expect interface{}
	fails  bool
}

type cancel struct {
	name string
}

type sequence struct {
	name  string
	tests []invoker
}

type echo call

type cancelParams struct{ ID int64 }

func TestConnectionRaw(t *testing.T) {
	testConnection(t, jsonrpc2.RawFramer())
}

func TestConnectionHeader(t *testing.T) {
	testConnection(t, jsonrpc2.HeaderFramer())
}

func testConnection(t *testing.T, framer jsonrpc2.Framer) {
	stacktest.NoLeak(t)
	ctx := eventtest.NewContext(context.Background(), t)
	listener, err := jsonrpc2.NetPipeListener(ctx)
	if err != nil {
		t.Fatal(err)
	}
	server, err := jsonrpc2.Serve(ctx, listener, binder{framer, nil})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		listener.Close()
		server.Wait()
	}()

	for _, test := range callTests {
		t.Run(test.Name(), func(t *testing.T) {
			client, err := jsonrpc2.Dial(ctx,
				listener.Dialer(), binder{framer, func(h *handler) {
					defer h.conn.Close()
					ctx := eventtest.NewContext(ctx, t)
					test.Invoke(t, ctx, h)
					if call, ok := test.(*call); ok {
						// also run all simple call tests in echo mode
						(*echo)(call).Invoke(t, ctx, h)
					}
				}})
			if err != nil {
				t.Fatal(err)
			}
			client.Wait()
		})
	}
}

func (test notify) Name() string { return test.method }
func (test notify) Invoke(t *testing.T, ctx context.Context, h *handler) {
	if err := h.conn.Notify(ctx, test.method, test.params); err != nil {
		t.Fatalf("%v:Notify failed: %v", test.method, err)
	}
}

func (test call) Name() string { return test.method }
func (test call) Invoke(t *testing.T, ctx context.Context, h *handler) {
	results := newResults(test.expect)
	if err := h.conn.Call(ctx, test.method, test.params).Await(ctx, results); err != nil {
		t.Fatalf("%v:Call failed: %v", test.method, err)
	}
	verifyResults(t, test.method, results, test.expect)
}

func (test echo) Invoke(t *testing.T, ctx context.Context, h *handler) {
	results := newResults(test.expect)
	if err := h.conn.Call(ctx, "echo", []interface{}{test.method, test.params}).Await(ctx, results); err != nil {
		t.Fatalf("%v:Echo failed: %v", test.method, err)
	}
	verifyResults(t, test.method, results, test.expect)
}

func (test async) Name() string { return test.name }
func (test async) Invoke(t *testing.T, ctx context.Context, h *handler) {
	h.calls[test.name] = h.conn.Call(ctx, test.method, test.params)
}

func (test collect) Name() string { return test.name }
func (test collect) Invoke(t *testing.T, ctx context.Context, h *handler) {
	o := h.calls[test.name]
	results := newResults(test.expect)
	err := o.Await(ctx, results)
	switch {
	case test.fails && err == nil:
		t.Fatalf("%v:Collect was supposed to fail", test.name)
	case !test.fails && err != nil:
		t.Fatalf("%v:Collect failed: %v", test.name, err)
	}
	verifyResults(t, test.name, results, test.expect)
}

func (test cancel) Name() string { return test.name }
func (test cancel) Invoke(t *testing.T, ctx context.Context, h *handler) {
	o := h.calls[test.name]
	if err := h.conn.Notify(ctx, "cancel", &cancelParams{o.ID().Raw().(int64)}); err != nil {
		t.Fatalf("%v:Collect failed: %v", test.name, err)
	}
}

func (test sequence) Name() string { return test.name }
func (test sequence) Invoke(t *testing.T, ctx context.Context, h *handler) {
	for _, child := range test.tests {
		child.Invoke(t, ctx, h)
	}
}

// newResults makes a new empty copy of the expected type to put the results into
func newResults(expect interface{}) interface{} {
	switch e := expect.(type) {
	case []interface{}:
		var r []interface{}
		for _, v := range e {
			r = append(r, reflect.New(reflect.TypeOf(v)).Interface())
		}
		return r
	case nil:
		return nil
	default:
		return reflect.New(reflect.TypeOf(expect)).Interface()
	}
}

// verifyResults compares the results to the expected values
func verifyResults(t *testing.T, method string, results interface{}, expect interface{}) {
	if expect == nil {
		if results != nil {
			t.Errorf("%v:Got results %+v where none expeted", method, expect)
		}
		return
	}
	val := reflect.Indirect(reflect.ValueOf(results)).Interface()
	if !reflect.DeepEqual(val, expect) {
		t.Errorf("%v:Results are incorrect, got %+v expect %+v", method, val, expect)
	}
}

func (b binder) Bind(ctx context.Context, conn *jsonrpc2.Connection) (jsonrpc2.ConnectionOptions, error) {
	h := &handler{
		conn:       conn,
		waitersBox: make(chan map[string]chan struct{}, 1),
		calls:      make(map[string]*jsonrpc2.AsyncCall),
	}
	h.waitersBox <- make(map[string]chan struct{})
	if b.runTest != nil {
		go b.runTest(h)
	}
	return jsonrpc2.ConnectionOptions{
		Framer:    b.framer,
		Preempter: h,
		Handler:   h,
	}, nil
}

func (h *handler) waiter(name string) chan struct{} {
	waiters := <-h.waitersBox
	defer func() { h.waitersBox <- waiters }()
	waiter, found := waiters[name]
	if !found {
		waiter = make(chan struct{})
		waiters[name] = waiter
	}
	return waiter
}

func (h *handler) Preempt(ctx context.Context, req *jsonrpc2.Request) (interface{}, error) {
	switch req.Method {
	case "unblock":
		var name string
		if err := json.Unmarshal(req.Params, &name); err != nil {
			return nil, fmt.Errorf("%w: %s", jsonrpc2.ErrParse, err)
		}
		close(h.waiter(name))
		return nil, nil
	case "peek":
		if len(req.Params) > 0 {
			return nil, fmt.Errorf("%w: expected no params", jsonrpc2.ErrInvalidParams)
		}
		return h.accumulator, nil
	case "cancel":
		var params cancelParams
		if err := json.Unmarshal(req.Params, &params); err != nil {
			return nil, fmt.Errorf("%w: %s", jsonrpc2.ErrParse, err)
		}
		h.conn.Cancel(jsonrpc2.Int64ID(params.ID))
		return nil, nil
	default:
		return nil, jsonrpc2.ErrNotHandled
	}
}

func (h *handler) Handle(ctx context.Context, req *jsonrpc2.Request) (interface{}, error) {
	switch req.Method {
	case "no_args":
		if len(req.Params) > 0 {
			return nil, fmt.Errorf("%w: expected no params", jsonrpc2.ErrInvalidParams)
		}
		return true, nil
	case "one_string":
		var v string
		if err := json.Unmarshal(req.Params, &v); err != nil {
			return nil, fmt.Errorf("%w: %s", jsonrpc2.ErrParse, err)
		}
		return "got:" + v, nil
	case "one_number":
		var v int
		if err := json.Unmarshal(req.Params, &v); err != nil {
			return nil, fmt.Errorf("%w: %s", jsonrpc2.ErrParse, err)
		}
		return fmt.Sprintf("got:%d", v), nil
	case "set":
		var v int
		if err := json.Unmarshal(req.Params, &v); err != nil {
			return nil, fmt.Errorf("%w: %s", jsonrpc2.ErrParse, err)
		}
		h.accumulator = v
		return nil, nil
	case "add":
		var v int
		if err := json.Unmarshal(req.Params, &v); err != nil {
			return nil, fmt.Errorf("%w: %s", jsonrpc2.ErrParse, err)
		}
		h.accumulator += v
		return nil, nil
	case "get":
		if len(req.Params) > 0 {
			return nil, fmt.Errorf("%w: expected no params", jsonrpc2.ErrInvalidParams)
		}
		return h.accumulator, nil
	case "join":
		var v []string
		if err := json.Unmarshal(req.Params, &v); err != nil {
			return nil, fmt.Errorf("%w: %s", jsonrpc2.ErrParse, err)
		}
		return path.Join(v...), nil
	case "echo":
		var v []interface{}
		if err := json.Unmarshal(req.Params, &v); err != nil {
			return nil, fmt.Errorf("%w: %s", jsonrpc2.ErrParse, err)
		}
		var result interface{}
		err := h.conn.Call(ctx, v[0].(string), v[1]).Await(ctx, &result)
		return result, err
	case "wait":
		var name string
		if err := json.Unmarshal(req.Params, &name); err != nil {
			return nil, fmt.Errorf("%w: %s", jsonrpc2.ErrParse, err)
		}
		select {
		case <-h.waiter(name):
			return true, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(time.Second):
			return nil, fmt.Errorf("wait for %q timed out", name)
		}
	case "fork":
		var name string
		if err := json.Unmarshal(req.Params, &name); err != nil {
			return nil, fmt.Errorf("%w: %s", jsonrpc2.ErrParse, err)
		}
		waitFor := h.waiter(name)
		go func() {
			select {
			case <-waitFor:
				h.conn.Respond(req.ID, true, nil)
			case <-ctx.Done():
				h.conn.Respond(req.ID, nil, ctx.Err())
			case <-time.After(time.Second):
				h.conn.Respond(req.ID, nil, fmt.Errorf("wait for %q timed out", name))
			}
		}()
		return nil, jsonrpc2.ErrAsyncResponse
	default:
		return nil, jsonrpc2.ErrNotHandled
	}
}
