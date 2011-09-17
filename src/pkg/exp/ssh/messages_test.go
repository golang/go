// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"big"
	"rand"
	"reflect"
	"testing"
	"testing/quick"
)

var intLengthTests = []struct {
	val, length int
}{
	{0, 4 + 0},
	{1, 4 + 1},
	{127, 4 + 1},
	{128, 4 + 2},
	{-1, 4 + 1},
}

func TestIntLength(t *testing.T) {
	for _, test := range intLengthTests {
		v := new(big.Int).SetInt64(int64(test.val))
		length := intLength(v)
		if length != test.length {
			t.Errorf("For %d, got length %d but expected %d", test.val, length, test.length)
		}
	}
}

var messageTypes = []interface{}{
	&kexInitMsg{},
	&kexDHInitMsg{},
	&serviceRequestMsg{},
	&serviceAcceptMsg{},
	&userAuthRequestMsg{},
	&channelOpenMsg{},
	&channelOpenConfirmMsg{},
	&channelRequestMsg{},
	&channelRequestSuccessMsg{},
}

func TestMarshalUnmarshal(t *testing.T) {
	rand := rand.New(rand.NewSource(0))
	for i, iface := range messageTypes {
		ty := reflect.ValueOf(iface).Type()

		n := 100
		if testing.Short() {
			n = 5
		}
		for j := 0; j < n; j++ {
			v, ok := quick.Value(ty, rand)
			if !ok {
				t.Errorf("#%d: failed to create value", i)
				break
			}

			m1 := v.Elem().Interface()
			m2 := iface

			marshaled := marshal(msgIgnore, m1)
			if err := unmarshal(m2, marshaled, msgIgnore); err != nil {
				t.Errorf("#%d failed to unmarshal %#v: %s", i, m1, err)
				break
			}

			if !reflect.DeepEqual(v.Interface(), m2) {
				t.Errorf("#%d\ngot: %#v\nwant:%#v\n%x", i, m2, m1, marshaled)
				break
			}
		}
	}
}

func randomBytes(out []byte, rand *rand.Rand) {
	for i := 0; i < len(out); i++ {
		out[i] = byte(rand.Int31())
	}
}

func randomNameList(rand *rand.Rand) []string {
	ret := make([]string, rand.Int31()&15)
	for i := range ret {
		s := make([]byte, 1+(rand.Int31()&15))
		for j := range s {
			s[j] = 'a' + uint8(rand.Int31()&15)
		}
		ret[i] = string(s)
	}
	return ret
}

func randomInt(rand *rand.Rand) *big.Int {
	return new(big.Int).SetInt64(int64(int32(rand.Uint32())))
}

func (*kexInitMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	ki := &kexInitMsg{}
	randomBytes(ki.Cookie[:], rand)
	ki.KexAlgos = randomNameList(rand)
	ki.ServerHostKeyAlgos = randomNameList(rand)
	ki.CiphersClientServer = randomNameList(rand)
	ki.CiphersServerClient = randomNameList(rand)
	ki.MACsClientServer = randomNameList(rand)
	ki.MACsServerClient = randomNameList(rand)
	ki.CompressionClientServer = randomNameList(rand)
	ki.CompressionServerClient = randomNameList(rand)
	ki.LanguagesClientServer = randomNameList(rand)
	ki.LanguagesServerClient = randomNameList(rand)
	if rand.Int31()&1 == 1 {
		ki.FirstKexFollows = true
	}
	return reflect.ValueOf(ki)
}

func (*kexDHInitMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	dhi := &kexDHInitMsg{}
	dhi.X = randomInt(rand)
	return reflect.ValueOf(dhi)
}
