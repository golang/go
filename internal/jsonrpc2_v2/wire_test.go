// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2_test

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	jsonrpc2 "golang.org/x/tools/internal/jsonrpc2_v2"
)

func TestWireMessage(t *testing.T) {
	for _, test := range []struct {
		name    string
		msg     jsonrpc2.Message
		encoded []byte
	}{{
		name:    "notification",
		msg:     newNotification("alive", nil),
		encoded: []byte(`{"jsonrpc":"2.0","method":"alive"}`),
	}, {
		name:    "call",
		msg:     newCall("msg1", "ping", nil),
		encoded: []byte(`{"jsonrpc":"2.0","id":"msg1","method":"ping"}`),
	}, {
		name:    "response",
		msg:     newResponse("msg2", "pong", nil),
		encoded: []byte(`{"jsonrpc":"2.0","id":"msg2","result":"pong"}`),
	}, {
		name:    "numerical id",
		msg:     newCall(1, "poke", nil),
		encoded: []byte(`{"jsonrpc":"2.0","id":1,"method":"poke"}`),
	}, {
		// originally reported in #39719, this checks that result is not present if
		// it is an error response
		name: "computing fix edits",
		msg:  newResponse(3, nil, jsonrpc2.NewError(0, "computing fix edits")),
		encoded: []byte(`{
		"jsonrpc":"2.0",
		"id":3,
		"error":{
			"code":0,
			"message":"computing fix edits"
		}
	}`),
	}} {
		b, err := jsonrpc2.EncodeMessage(test.msg)
		if err != nil {
			t.Fatal(err)
		}
		checkJSON(t, b, test.encoded)
		msg, err := jsonrpc2.DecodeMessage(test.encoded)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(msg, test.msg) {
			t.Errorf("decoded message does not match\nGot:\n%+#v\nWant:\n%+#v", msg, test.msg)
		}
	}
}

func newNotification(method string, params interface{}) jsonrpc2.Message {
	msg, err := jsonrpc2.NewNotification(method, params)
	if err != nil {
		panic(err)
	}
	return msg
}

func newID(id interface{}) jsonrpc2.ID {
	switch v := id.(type) {
	case nil:
		return jsonrpc2.ID{}
	case string:
		return jsonrpc2.StringID(v)
	case int:
		return jsonrpc2.Int64ID(int64(v))
	case int64:
		return jsonrpc2.Int64ID(v)
	default:
		panic("invalid ID type")
	}
}

func newCall(id interface{}, method string, params interface{}) jsonrpc2.Message {
	msg, err := jsonrpc2.NewCall(newID(id), method, params)
	if err != nil {
		panic(err)
	}
	return msg
}

func newResponse(id interface{}, result interface{}, rerr error) jsonrpc2.Message {
	msg, err := jsonrpc2.NewResponse(newID(id), result, rerr)
	if err != nil {
		panic(err)
	}
	return msg
}

func checkJSON(t *testing.T, got, want []byte) {
	// compare the compact form, to allow for formatting differences
	g := &bytes.Buffer{}
	if err := json.Compact(g, []byte(got)); err != nil {
		t.Fatal(err)
	}
	w := &bytes.Buffer{}
	if err := json.Compact(w, []byte(want)); err != nil {
		t.Fatal(err)
	}
	if g.String() != w.String() {
		t.Errorf("encoded message does not match\nGot:\n%s\nWant:\n%s", g, w)
	}
}
