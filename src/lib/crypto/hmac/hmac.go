// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc): comments

package hmac

import (
	"hash";
	"os";
)

// k0 = key
// ipad = 0x36 byte repeated to key len
// opad = 0x5c byte repeated to key len
// hmac = 
//	H((k0 ^ opad) || H((k0 ^ ipad) || text))

const (
	padSize = 64;
)

type hmac struct {
	size int;
	key []byte;
	tmp []byte;
	inner hash.Hash;
}

func (h *hmac) tmpPad(xor byte) {
	for i, k := range h.key {
		h.tmp[i] = xor ^ k;
	}
	for i := len(h.key); i < padSize; i++ {
		h.tmp[i] = xor;
	}
}

func (h *hmac) init() {
	h.tmpPad(0x36);
	h.inner.Write(h.tmp[0:padSize]);
}

func (h *hmac) Sum() []byte {
	h.tmpPad(0x5c);
	sum := h.inner.Sum();
	for i, b := range sum {
		h.tmp[padSize + i] = b;
	}
	h.inner.Reset();
	h.inner.Write(h.tmp);
	return h.inner.Sum();
}

func (h *hmac) Write(p []byte) (n int, err os.Error) {
	return h.inner.Write(p);
}

func (h *hmac) Size() int {
	return h.size;
}

func (h *hmac) Reset() {
	h.inner.Reset();
	h.init();
}

func HMAC(h hash.Hash, key []byte) hash.Hash {
	hm := new(hmac);
	hm.inner = h;
	hm.size = h.Size();
	hm.key = make([]byte, len(key));
	for i, k := range key {
		hm.key[i] = k;
	}
	hm.tmp = make([]byte, padSize + hm.size);
	hm.init();
	return hm;
}
