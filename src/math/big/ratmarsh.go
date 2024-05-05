// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements encoding/decoding of Rats.

package big

import (
	"errors"
	"fmt"
	"internal/binarylite"
	"math"
)

// Gob codec version. Permits backward-compatible changes to the encoding.
const ratGobVersion byte = 1

// GobEncode implements the [encoding/gob.GobEncoder] interface.
func (x *Rat) GobEncode() ([]byte, error) {
	if x == nil {
		return nil, nil
	}
	buf := make([]byte, 1+4+(len(x.a.abs)+len(x.b.abs))*_S) // extra bytes for version and sign bit (1), and numerator length (4)
	i := x.b.abs.bytes(buf)
	j := x.a.abs.bytes(buf[:i])
	n := i - j
	if int(uint32(n)) != n {
		// this should never happen
		return nil, errors.New("Rat.GobEncode: numerator too large")
	}
	binarylite.BigEndian.PutUint32(buf[j-4:j], uint32(n))
	j -= 1 + 4
	b := ratGobVersion << 1 // make space for sign bit
	if x.a.neg {
		b |= 1
	}
	buf[j] = b
	return buf[j:], nil
}

// GobDecode implements the [encoding/gob.GobDecoder] interface.
func (z *Rat) GobDecode(buf []byte) error {
	if len(buf) == 0 {
		// Other side sent a nil or default value.
		*z = Rat{}
		return nil
	}
	if len(buf) < 5 {
		return errors.New("Rat.GobDecode: buffer too small")
	}
	b := buf[0]
	if b>>1 != ratGobVersion {
		return fmt.Errorf("Rat.GobDecode: encoding version %d not supported", b>>1)
	}
	const j = 1 + 4
	ln := binarylite.BigEndian.Uint32(buf[j-4 : j])
	if uint64(ln) > math.MaxInt-j {
		return errors.New("Rat.GobDecode: invalid length")
	}
	i := j + int(ln)
	if len(buf) < i {
		return errors.New("Rat.GobDecode: buffer too small")
	}
	z.a.neg = b&1 != 0
	z.a.abs = z.a.abs.setBytes(buf[j:i])
	z.b.abs = z.b.abs.setBytes(buf[i:])
	return nil
}

// MarshalText implements the [encoding.TextMarshaler] interface.
func (x *Rat) MarshalText() (text []byte, err error) {
	if x.IsInt() {
		return x.a.MarshalText()
	}
	return x.marshal(), nil
}

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (z *Rat) UnmarshalText(text []byte) error {
	// TODO(gri): get rid of the []byte/string conversion
	if _, ok := z.SetString(string(text)); !ok {
		return fmt.Errorf("math/big: cannot unmarshal %q into a *big.Rat", text)
	}
	return nil
}
