// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements encoding/decoding of Floats.

package big

import (
	"encoding/binary"
	"fmt"
)

// Gob codec version. Permits backward-compatible changes to the encoding.
const floatGobVersion byte = 1

// GobEncode implements the gob.GobEncoder interface.
// The Float value and all its attributes (precision,
// rounding mode, accuracy) are marshaled.
func (x *Float) GobEncode() ([]byte, error) {
	if x == nil {
		return nil, nil
	}

	// determine max. space (bytes) required for encoding
	sz := 1 + 1 + 4 // version + mode|acc|form|neg (3+2+2+1bit) + prec
	n := 0          // number of mantissa words
	if x.form == finite {
		// add space for mantissa and exponent
		n = int((x.prec + (_W - 1)) / _W) // required mantissa length in words for given precision
		// actual mantissa slice could be shorter (trailing 0's) or longer (unused bits):
		// - if shorter, only encode the words present
		// - if longer, cut off unused words when encoding in bytes
		//   (in practice, this should never happen since rounding
		//   takes care of it, but be safe and do it always)
		if len(x.mant) < n {
			n = len(x.mant)
		}
		// len(x.mant) >= n
		sz += 4 + n*_S // exp + mant
	}
	buf := make([]byte, sz)

	buf[0] = floatGobVersion
	b := byte(x.mode&7)<<5 | byte((x.acc+1)&3)<<3 | byte(x.form&3)<<1
	if x.neg {
		b |= 1
	}
	buf[1] = b
	binary.BigEndian.PutUint32(buf[2:], x.prec)

	if x.form == finite {
		binary.BigEndian.PutUint32(buf[6:], uint32(x.exp))
		x.mant[len(x.mant)-n:].bytes(buf[10:]) // cut off unused trailing words
	}

	return buf, nil
}

// GobDecode implements the gob.GobDecoder interface.
// The result is rounded per the precision and rounding mode of
// z unless z's precision is 0, in which case z is set exactly
// to the decoded value.
func (z *Float) GobDecode(buf []byte) error {
	if len(buf) == 0 {
		// Other side sent a nil or default value.
		*z = Float{}
		return nil
	}

	if buf[0] != floatGobVersion {
		return fmt.Errorf("Float.GobDecode: encoding version %d not supported", buf[0])
	}

	oldPrec := z.prec
	oldMode := z.mode

	b := buf[1]
	z.mode = RoundingMode((b >> 5) & 7)
	z.acc = Accuracy((b>>3)&3) - 1
	z.form = form((b >> 1) & 3)
	z.neg = b&1 != 0
	z.prec = binary.BigEndian.Uint32(buf[2:])

	if z.form == finite {
		z.exp = int32(binary.BigEndian.Uint32(buf[6:]))
		z.mant = z.mant.setBytes(buf[10:])
	}

	if oldPrec != 0 {
		z.mode = oldMode
		z.SetPrec(uint(oldPrec))
	}

	return nil
}

// MarshalText implements the encoding.TextMarshaler interface.
// Only the Float value is marshaled (in full precision), other
// attributes such as precision or accuracy are ignored.
func (x *Float) MarshalText() (text []byte, err error) {
	if x == nil {
		return []byte("<nil>"), nil
	}
	var buf []byte
	return x.Append(buf, 'g', -1), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface.
// The result is rounded per the precision and rounding mode of z.
// If z's precision is 0, it is changed to 64 before rounding takes
// effect.
func (z *Float) UnmarshalText(text []byte) error {
	// TODO(gri): get rid of the []byte/string conversion
	_, _, err := z.Parse(string(text), 0)
	if err != nil {
		err = fmt.Errorf("math/big: cannot unmarshal %q into a *big.Float (%v)", text, err)
	}
	return err
}
