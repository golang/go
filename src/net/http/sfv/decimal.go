// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"errors"
	"io"
	"math"
	"strconv"
)

const maxDecDigit = 3

// ErrInvalidDecimal is returned when a decimal is invalid.
var ErrInvalidDecimal = errors.New("the integer portion is larger than 12 digits: invalid decimal")

// marshalDecimal serializes as defined in
// https://httpwg.org/http-extensions/draft-ietf-httpbis-header-structure.html#ser-decimal.
//
// TODO(dunglas): add support for decimal float type when one will be available
// (https://github.com/golang/go/issues/19787)
func marshalDecimal(b io.StringWriter, d float64) error {
	const TH = 0.001

	rounded := math.RoundToEven(d/TH) * TH
	int, frac := math.Modf(math.RoundToEven(d/TH) * TH)

	if int < -999999999999 || int > 999999999999 {
		return ErrInvalidDecimal
	}

	if _, err := b.WriteString(strconv.FormatFloat(rounded, 'f', -1, 64)); err != nil {
		return err
	}

	if frac == 0 {
		_, err := b.WriteString(".0")

		return err
	}

	return nil
}

func parseDecimal(s *scanner, decSepOff int, str string, neg bool) (float64, error) {
	if decSepOff == s.off-1 {
		return 0, &UnmarshalError{s.off, ErrInvalidDecimalFormat}
	}

	if len(s.data[decSepOff+1:s.off]) > maxDecDigit {
		return 0, &UnmarshalError{s.off, ErrNumberOutOfRange}
	}

	i, err := strconv.ParseFloat(str, 64)
	if err != nil {
		// Should never happen
		return 0, &UnmarshalError{s.off, err}
	}

	if neg {
		i = -i
	}

	return i, nil
}
