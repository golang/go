// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements encoding/decoding of Ints.

package big

import (
	"bytes"
	"fmt"
)

// Gob codec version. Permits backward-compatible changes to the encoding.
const intGobVersion byte = 1

// GobEncode implements the gob.GobEncoder interface.
func (x *Int) GobEncode() ([]byte, error) {
	if x == nil {
		return nil, nil
	}
	buf := make([]byte, 1+len(x.abs)*_S) // extra byte for version and sign bit
	i := x.abs.bytes(buf) - 1            // i >= 0
	b := intGobVersion << 1              // make space for sign bit
	if x.neg {
		b |= 1
	}
	buf[i] = b
	return buf[i:], nil
}

// GobDecode implements the gob.GobDecoder interface.
func (z *Int) GobDecode(buf []byte) error {
	if len(buf) == 0 {
		// Other side sent a nil or default value.
		*z = Int{}
		return nil
	}
	b := buf[0]
	if b>>1 != intGobVersion {
		return fmt.Errorf("Int.GobDecode: encoding version %d not supported", b>>1)
	}
	z.neg = b&1 != 0
	z.abs = z.abs.setBytes(buf[1:])
	return nil
}

// MarshalText implements the encoding.TextMarshaler interface.
func (x *Int) MarshalText() (text []byte, err error) {
	if x == nil {
		return []byte("<nil>"), nil
	}
	return x.abs.itoa(x.neg, 10), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface.
func (z *Int) UnmarshalText(text []byte) error {
	if _, ok := z.setFromScanner(bytes.NewReader(text), 0); !ok {
		return fmt.Errorf("math/big: cannot unmarshal %q into a *big.Int", text)
	}
	return nil
}

// The JSON marshalers are only here for API backward compatibility
// (programs that explicitly look for these two methods). JSON works
// fine with the TextMarshaler only.

// MarshalJSON implements the json.Marshaler interface.
func (x *Int) MarshalJSON() ([]byte, error) {
	if x == nil {
		return []byte("null"), nil
	}
	return x.abs.itoa(x.neg, 10), nil
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (z *Int) UnmarshalJSON(text []byte) error {
	// Ignore null, like in the main JSON package.
	if string(text) == "null" {
		return nil
	}
	return z.UnmarshalText(text)
}
