// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proxy

import (
	"errors"
	"net"
	"reflect"
	"testing"
)

type recordingProxy struct {
	addrs []string
}

func (r *recordingProxy) Dial(network, addr string) (net.Conn, error) {
	r.addrs = append(r.addrs, addr)
	return nil, errors.New("recordingProxy")
}

func TestPerHost(t *testing.T) {
	var def, bypass recordingProxy
	perHost := NewPerHost(&def, &bypass)
	perHost.AddFromString("localhost,*.zone,127.0.0.1,10.0.0.1/8,1000::/16")

	expectedDef := []string{
		"example.com:123",
		"1.2.3.4:123",
		"[1001::]:123",
	}
	expectedBypass := []string{
		"localhost:123",
		"zone:123",
		"foo.zone:123",
		"127.0.0.1:123",
		"10.1.2.3:123",
		"[1000::]:123",
	}

	for _, addr := range expectedDef {
		perHost.Dial("tcp", addr)
	}
	for _, addr := range expectedBypass {
		perHost.Dial("tcp", addr)
	}

	if !reflect.DeepEqual(expectedDef, def.addrs) {
		t.Errorf("Hosts which went to the default proxy didn't match. Got %v, want %v", def.addrs, expectedDef)
	}
	if !reflect.DeepEqual(expectedBypass, bypass.addrs) {
		t.Errorf("Hosts which went to the bypass proxy didn't match. Got %v, want %v", bypass.addrs, expectedBypass)
	}
}
