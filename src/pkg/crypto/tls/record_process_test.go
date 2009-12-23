// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"encoding/hex"
	"testing"
	"testing/script"
)

func setup() (appDataChan chan []byte, requestChan chan interface{}, controlChan chan interface{}, recordChan chan *record, handshakeChan chan interface{}) {
	rp := new(recordProcessor)
	appDataChan = make(chan []byte)
	requestChan = make(chan interface{})
	controlChan = make(chan interface{})
	recordChan = make(chan *record)
	handshakeChan = make(chan interface{})

	go rp.loop(appDataChan, requestChan, controlChan, recordChan, handshakeChan)
	return
}

func fromHex(s string) []byte {
	b, _ := hex.DecodeString(s)
	return b
}

func TestNullConnectionState(t *testing.T) {
	_, requestChan, controlChan, recordChan, _ := setup()
	defer close(requestChan)
	defer close(controlChan)
	defer close(recordChan)

	// Test a simple request for the connection state.
	replyChan := make(chan ConnectionState)
	sendReq := script.NewEvent("send request", nil, script.Send{requestChan, getConnectionState{replyChan}})
	getReply := script.NewEvent("get reply", []*script.Event{sendReq}, script.Recv{replyChan, ConnectionState{false, "", 0, ""}})

	err := script.Perform(0, []*script.Event{sendReq, getReply})
	if err != nil {
		t.Errorf("Got error: %s", err)
	}
}

func TestWaitConnectionState(t *testing.T) {
	_, requestChan, controlChan, recordChan, _ := setup()
	defer close(requestChan)
	defer close(controlChan)
	defer close(recordChan)

	// Test that waitConnectionState doesn't get a reply until the connection state changes.
	replyChan := make(chan ConnectionState)
	sendReq := script.NewEvent("send request", nil, script.Send{requestChan, waitConnectionState{replyChan}})
	replyChan2 := make(chan ConnectionState)
	sendReq2 := script.NewEvent("send request 2", []*script.Event{sendReq}, script.Send{requestChan, getConnectionState{replyChan2}})
	getReply2 := script.NewEvent("get reply 2", []*script.Event{sendReq2}, script.Recv{replyChan2, ConnectionState{false, "", 0, ""}})
	sendState := script.NewEvent("send state", []*script.Event{getReply2}, script.Send{controlChan, ConnectionState{true, "test", 1, ""}})
	getReply := script.NewEvent("get reply", []*script.Event{sendState}, script.Recv{replyChan, ConnectionState{true, "test", 1, ""}})

	err := script.Perform(0, []*script.Event{sendReq, sendReq2, getReply2, sendState, getReply})
	if err != nil {
		t.Errorf("Got error: %s", err)
	}
}

func TestHandshakeAssembly(t *testing.T) {
	_, requestChan, controlChan, recordChan, handshakeChan := setup()
	defer close(requestChan)
	defer close(controlChan)
	defer close(recordChan)

	// Test the reassembly of a fragmented handshake message.
	send1 := script.NewEvent("send 1", nil, script.Send{recordChan, &record{recordTypeHandshake, 0, 0, fromHex("10000003")}})
	send2 := script.NewEvent("send 2", []*script.Event{send1}, script.Send{recordChan, &record{recordTypeHandshake, 0, 0, fromHex("0001")}})
	send3 := script.NewEvent("send 3", []*script.Event{send2}, script.Send{recordChan, &record{recordTypeHandshake, 0, 0, fromHex("42")}})
	recvMsg := script.NewEvent("recv", []*script.Event{send3}, script.Recv{handshakeChan, &clientKeyExchangeMsg{fromHex("10000003000142"), fromHex("42")}})

	err := script.Perform(0, []*script.Event{send1, send2, send3, recvMsg})
	if err != nil {
		t.Errorf("Got error: %s", err)
	}
}

func TestEarlyApplicationData(t *testing.T) {
	_, requestChan, controlChan, recordChan, handshakeChan := setup()
	defer close(requestChan)
	defer close(controlChan)
	defer close(recordChan)

	// Test that applicaton data received before the handshake has completed results in an error.
	send := script.NewEvent("send", nil, script.Send{recordChan, &record{recordTypeApplicationData, 0, 0, fromHex("")}})
	recv := script.NewEvent("recv", []*script.Event{send}, script.Closed{handshakeChan})

	err := script.Perform(0, []*script.Event{send, recv})
	if err != nil {
		t.Errorf("Got error: %s", err)
	}
}

func TestApplicationData(t *testing.T) {
	appDataChan, requestChan, controlChan, recordChan, handshakeChan := setup()
	defer close(requestChan)
	defer close(controlChan)
	defer close(recordChan)

	// Test that the application data is forwarded after a successful Finished message.
	send1 := script.NewEvent("send 1", nil, script.Send{recordChan, &record{recordTypeHandshake, 0, 0, fromHex("1400000c000000000000000000000000")}})
	recv1 := script.NewEvent("recv finished", []*script.Event{send1}, script.Recv{handshakeChan, &finishedMsg{fromHex("1400000c000000000000000000000000"), fromHex("000000000000000000000000")}})
	send2 := script.NewEvent("send connState", []*script.Event{recv1}, script.Send{controlChan, ConnectionState{true, "", 0, ""}})
	send3 := script.NewEvent("send 2", []*script.Event{send2}, script.Send{recordChan, &record{recordTypeApplicationData, 0, 0, fromHex("0102")}})
	recv2 := script.NewEvent("recv data", []*script.Event{send3}, script.Recv{appDataChan, []byte{0x01, 0x02}})

	err := script.Perform(0, []*script.Event{send1, recv1, send2, send3, recv2})
	if err != nil {
		t.Errorf("Got error: %s", err)
	}
}

func TestInvalidChangeCipherSpec(t *testing.T) {
	appDataChan, requestChan, controlChan, recordChan, handshakeChan := setup()
	defer close(requestChan)
	defer close(controlChan)
	defer close(recordChan)

	send1 := script.NewEvent("send 1", nil, script.Send{recordChan, &record{recordTypeChangeCipherSpec, 0, 0, []byte{1}}})
	recv1 := script.NewEvent("recv 1", []*script.Event{send1}, script.Recv{handshakeChan, changeCipherSpec{}})
	send2 := script.NewEvent("send 2", []*script.Event{recv1}, script.Send{controlChan, ConnectionState{false, "", 42, ""}})
	close := script.NewEvent("close 1", []*script.Event{send2}, script.Closed{appDataChan})
	close2 := script.NewEvent("close 2", []*script.Event{send2}, script.Closed{handshakeChan})

	err := script.Perform(0, []*script.Event{send1, recv1, send2, close, close2})
	if err != nil {
		t.Errorf("Got error: %s", err)
	}
}
