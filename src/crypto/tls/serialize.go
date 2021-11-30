// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net"
	"reflect"
	"unsafe"
)

func (c *Conn) Serialize() (jsonB []byte, err error) {

	// Store the readbuf
	rsConn := reflect.ValueOf(c.conn).Elem()
	reflectedReadBuf := rsConn.Field(0).Elem()
	reflectedReadBuf = reflect.NewAt(reflectedReadBuf.Type(), unsafe.Pointer(reflectedReadBuf.UnsafeAddr())).Elem()
	sf := reflectedReadBuf.Field(0)
	sf = reflect.NewAt(sf.Type(), unsafe.Pointer(sf.UnsafeAddr())).Elem()

	extractedReadBuf := make([]byte, 0)
	rt := reflect.ValueOf(&extractedReadBuf).Elem()
	rtw := reflect.NewAt(rt.Type(), unsafe.Pointer(rt.UnsafeAddr())).Elem()
	rtw.Set(sf)

	sf1 := reflectedReadBuf.Field(1)
	sf1 = reflect.NewAt(sf1.Type(), unsafe.Pointer(sf1.UnsafeAddr())).Elem()
	offset := 0
	rtOffset := reflect.ValueOf(&offset).Elem()
	rtOffsetW := reflect.NewAt(rtOffset.Type(), unsafe.Pointer(rtOffset.UnsafeAddr())).Elem()

	rtOffsetW.Set(sf1)

	c.ReadBuf = extractedReadBuf
	c.ReadBuffOffset = offset

	b, marshalErr := json.Marshal(c)

	return b, marshalErr
}

func RestoreFromJSON(jsonB []byte, conn net.Conn, config *Config) (sconn *Conn, err error) {

	sconn = &Conn{
		conn:     conn,
		config:   config,
		IsClient: false,
	}
	sconn.handshakeFn = sconn.serverHandshake

	sconn.conn =     conn
	sconn.config =   config
	sconn.IsClient = false

	// Restore the handshake function manually
	sconn.handshakeFn = sconn.serverHandshake

	unmarshalErr := json.Unmarshal(jsonB, sconn)
	if nil != unmarshalErr {
		err = unmarshalErr
		return nil, err
	}

	savedInSeq := sconn.In.Seq
	savedOutSeq := sconn.Out.Seq

	// Todo - Restore the EKM based on TLS version
	switch sconn.Vers {
	case tls.VersionTLS13:
	default:
		err = fmt.Errorf("unsupported TLS version")
		return nil, err
	}

	// Restore the half-conns
	sconn.In.setTrafficSecret(cipherSuiteTLS13ByID(sconn.CipherSuite), sconn.In.TrafficSecret)
	sconn.Out.setTrafficSecret(cipherSuiteTLS13ByID(sconn.CipherSuite), sconn.Out.TrafficSecret)

	sconn.In.Seq = savedInSeq
	sconn.Out.Seq = savedOutSeq









	// Restore the readbuf
	rsConn := reflect.ValueOf(sconn.conn).Elem()
	reflectedReadBuf := rsConn.Field(0).Elem()
	reflectedReadBuf = reflect.NewAt(reflectedReadBuf.Type(), unsafe.Pointer(reflectedReadBuf.UnsafeAddr())).Elem()
	sf := reflectedReadBuf.Field(0)
	sf = reflect.NewAt(sf.Type(), unsafe.Pointer(sf.UnsafeAddr())).Elem()

	rt := reflect.ValueOf(&sconn.ReadBuf).Elem()
	rtw := reflect.NewAt(rt.Type(), unsafe.Pointer(rt.UnsafeAddr())).Elem()
	sf.Set(rtw)

	sf1 := reflectedReadBuf.Field(1)
	sf1 = reflect.NewAt(sf1.Type(), unsafe.Pointer(sf1.UnsafeAddr())).Elem()
	rtOffset := reflect.ValueOf(&sconn.ReadBuffOffset).Elem()
	rtOffsetW := reflect.NewAt(rtOffset.Type(), unsafe.Pointer(rtOffset.UnsafeAddr())).Elem()

	sf1.Set(rtOffsetW)

	hs := serverHandshakeStateTLS13{
		suite: cipherSuiteTLS13ByID(sconn.CipherSuite),
	}
	hs.transcript = hs.suite.hash.New()

	// Manually restore the transcript fields
	rs := reflect.ValueOf(hs.transcript).Elem()
	rf1 := rs.Field(0)
	rf1 = reflect.NewAt(rf1.Type(), unsafe.Pointer(rf1.UnsafeAddr())).Elem()
	rt1 := reflect.ValueOf(&sconn.HS.H).Elem()
	rtw1 := reflect.NewAt(rt1.Type(), unsafe.Pointer(rt1.UnsafeAddr())).Elem()
	rf1.Set(rtw1)

	// hs.transcript.x field
	rf2 := rs.Field(1)
	rf2 = reflect.NewAt(rf2.Type(), unsafe.Pointer(rf2.UnsafeAddr())).Elem()
	rt2 := reflect.ValueOf(&sconn.HS.X).Elem()
	rtw2 := reflect.NewAt(rt2.Type(), unsafe.Pointer(rt2.UnsafeAddr())).Elem()
	rf2.Set(rtw2)

	// hs.transcript.nx field
	rf3 := rs.Field(2)
	rf3 = reflect.NewAt(rf3.Type(), unsafe.Pointer(rf3.UnsafeAddr())).Elem()
	rt3 := reflect.ValueOf(&sconn.HS.Nx).Elem()
	rtw3 := reflect.NewAt(rt3.Type(), unsafe.Pointer(rt3.UnsafeAddr())).Elem()
	rf3.Set(rtw3)

	// hs.transcript.len field
	rf4 := rs.Field(3)
	rf4 = reflect.NewAt(rf4.Type(), unsafe.Pointer(rf4.UnsafeAddr())).Elem()
	rt4 := reflect.ValueOf(&sconn.HS.Len).Elem()
	rtw4 := reflect.NewAt(rt4.Type(), unsafe.Pointer(rt4.UnsafeAddr())).Elem()
	rf4.Set(rtw4)

	// hs.transcript.is224 field
	rf5 := rs.Field(4)
	rf5 = reflect.NewAt(rf5.Type(), unsafe.Pointer(rf5.UnsafeAddr())).Elem()
	rt5 := reflect.ValueOf(&sconn.HS.Is224).Elem()
	rtw5 := reflect.NewAt(rt5.Type(), unsafe.Pointer(rt5.UnsafeAddr())).Elem()
	rf5.Set(rtw5)

	sconn.ekm = hs.suite.ExportKeyingMaterial(sconn.HS.MasterSecret, hs.transcript)

	return sconn, err
}