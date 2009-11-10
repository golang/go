// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"rand";
	"reflect";
	"testing";
	"testing/quick";
)

var tests = []interface{}{
	&clientHelloMsg{},
	&clientKeyExchangeMsg{},
	&finishedMsg{},
}

type testMessage interface {
	marshal() []byte;
	unmarshal([]byte) bool;
}

func TestMarshalUnmarshal(t *testing.T) {
	rand := rand.New(rand.NewSource(0));
	for i, iface := range tests {
		ty := reflect.NewValue(iface).Type();

		for j := 0; j < 100; j++ {
			v, ok := quick.Value(ty, rand);
			if !ok {
				t.Errorf("#%d: failed to create value", i);
				break;
			}

			m1 := v.Interface().(testMessage);
			marshaled := m1.marshal();
			m2 := iface.(testMessage);
			if !m2.unmarshal(marshaled) {
				t.Errorf("#%d failed to unmarshal %#v", i, m1);
				break;
			}
			m2.marshal();	// to fill any marshal cache in the message

			if !reflect.DeepEqual(m1, m2) {
				t.Errorf("#%d got:%#v want:%#v", i, m1, m2);
				break;
			}

			// Now check that all prefixes are invalid.
			for j := 0; j < len(marshaled); j++ {
				if m2.unmarshal(marshaled[0:j]) {
					t.Errorf("#%d unmarshaled a prefix of length %d of %#v", i, j, m1);
					break;
				}
			}
		}
	}
}

func randomBytes(n int, rand *rand.Rand) []byte {
	r := make([]byte, n);
	for i := 0; i < n; i++ {
		r[i] = byte(rand.Int31())
	}
	return r;
}

func (*clientHelloMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &clientHelloMsg{};
	m.major = uint8(rand.Intn(256));
	m.minor = uint8(rand.Intn(256));
	m.random = randomBytes(32, rand);
	m.sessionId = randomBytes(rand.Intn(32), rand);
	m.cipherSuites = make([]uint16, rand.Intn(63)+1);
	for i := 0; i < len(m.cipherSuites); i++ {
		m.cipherSuites[i] = uint16(rand.Int31())
	}
	m.compressionMethods = randomBytes(rand.Intn(63)+1, rand);

	return reflect.NewValue(m);
}

func (*clientKeyExchangeMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &clientKeyExchangeMsg{};
	m.ciphertext = randomBytes(rand.Intn(1000), rand);
	return reflect.NewValue(m);
}

func (*finishedMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &finishedMsg{};
	m.verifyData = randomBytes(12, rand);
	return reflect.NewValue(m);
}
