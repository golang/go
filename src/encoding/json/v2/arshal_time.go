// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"bytes"
	"cmp"
	"errors"
	"fmt"
	"math"
	"math/bits"
	"reflect"
	"strconv"
	"strings"
	"time"

	"encoding/json/internal"
	"encoding/json/internal/jsonflags"
	"encoding/json/internal/jsonopts"
	"encoding/json/internal/jsonwire"
	"encoding/json/jsontext"
)

var (
	timeDurationType = reflect.TypeFor[time.Duration]()
	timeTimeType     = reflect.TypeFor[time.Time]()
)

func makeTimeArshaler(fncs *arshaler, t reflect.Type) *arshaler {
	// Ideally, time types would implement MarshalerTo and UnmarshalerFrom,
	// but that would incur a dependency on package json from package time.
	// Given how widely used time is, it is more acceptable that we incur a
	// dependency on time from json.
	//
	// Injecting the arshaling functionality like this will not be identical
	// to actually declaring methods on the time types since embedding of the
	// time types will not be able to forward this functionality.
	switch t {
	case timeDurationType:
		fncs.nonDefault = true
		marshalNano := fncs.marshal
		fncs.marshal = func(enc *jsontext.Encoder, va addressableValue, mo *jsonopts.Struct) error {
			xe := export.Encoder(enc)
			var m durationArshaler
			if mo.Format != "" && mo.FormatDepth == xe.Tokens.Depth() {
				if !m.initFormat(mo.Format) {
					return newInvalidFormatError(enc, t, mo)
				}
			} else if mo.Flags.Get(jsonflags.FormatTimeWithLegacySemantics) {
				return marshalNano(enc, va, mo)
			} else {
				// TODO(https://go.dev/issue/71631): Decide on default duration representation.
				return newMarshalErrorBefore(enc, t, errors.New("no default representation; specify an explicit format"))
			}

			// TODO(https://go.dev/issue/62121): Use reflect.Value.AssertTo.
			m.td = *va.Addr().Interface().(*time.Duration)
			k := stringOrNumberKind(!m.isNumeric() || xe.Tokens.Last.NeedObjectName() || mo.Flags.Get(jsonflags.StringifyNumbers))
			if err := xe.AppendRaw(k, true, m.appendMarshal); err != nil {
				if !isSyntacticError(err) && !export.IsIOError(err) {
					err = newMarshalErrorBefore(enc, t, err)
				}
				return err
			}
			return nil
		}
		unmarshalNano := fncs.unmarshal
		fncs.unmarshal = func(dec *jsontext.Decoder, va addressableValue, uo *jsonopts.Struct) error {
			xd := export.Decoder(dec)
			var u durationArshaler
			if uo.Format != "" && uo.FormatDepth == xd.Tokens.Depth() {
				if !u.initFormat(uo.Format) {
					return newInvalidFormatError(dec, t, uo)
				}
			} else if uo.Flags.Get(jsonflags.FormatTimeWithLegacySemantics) {
				return unmarshalNano(dec, va, uo)
			} else {
				// TODO(https://go.dev/issue/71631): Decide on default duration representation.
				return newUnmarshalErrorBeforeWithSkipping(dec, uo, t, errors.New("no default representation; specify an explicit format"))
			}

			stringify := !u.isNumeric() || xd.Tokens.Last.NeedObjectName() || uo.Flags.Get(jsonflags.StringifyNumbers)
			var flags jsonwire.ValueFlags
			td := va.Addr().Interface().(*time.Duration)
			val, err := xd.ReadValue(&flags)
			if err != nil {
				return err
			}
			switch k := val.Kind(); k {
			case 'n':
				if !uo.Flags.Get(jsonflags.MergeWithLegacySemantics) {
					*td = time.Duration(0)
				}
				return nil
			case '"':
				if !stringify {
					break
				}
				val = jsonwire.UnquoteMayCopy(val, flags.IsVerbatim())
				if err := u.unmarshal(val); err != nil {
					return newUnmarshalErrorAfter(dec, t, err)
				}
				*td = u.td
				return nil
			case '0':
				if stringify {
					break
				}
				if err := u.unmarshal(val); err != nil {
					return newUnmarshalErrorAfter(dec, t, err)
				}
				*td = u.td
				return nil
			}
			return newUnmarshalErrorAfter(dec, t, nil)
		}
	case timeTimeType:
		fncs.nonDefault = true
		fncs.marshal = func(enc *jsontext.Encoder, va addressableValue, mo *jsonopts.Struct) (err error) {
			xe := export.Encoder(enc)
			var m timeArshaler
			if mo.Format != "" && mo.FormatDepth == xe.Tokens.Depth() {
				if !m.initFormat(mo.Format) {
					return newInvalidFormatError(enc, t, mo)
				}
			}

			// TODO(https://go.dev/issue/62121): Use reflect.Value.AssertTo.
			m.tt = *va.Addr().Interface().(*time.Time)
			k := stringOrNumberKind(!m.isNumeric() || xe.Tokens.Last.NeedObjectName() || mo.Flags.Get(jsonflags.StringifyNumbers))
			if err := xe.AppendRaw(k, !m.hasCustomFormat(), m.appendMarshal); err != nil {
				if mo.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
					return internal.NewMarshalerError(va.Addr().Interface(), err, "MarshalJSON") // unlike unmarshal, always wrapped
				}
				if !isSyntacticError(err) && !export.IsIOError(err) {
					err = newMarshalErrorBefore(enc, t, err)
				}
				return err
			}
			return nil
		}
		fncs.unmarshal = func(dec *jsontext.Decoder, va addressableValue, uo *jsonopts.Struct) (err error) {
			xd := export.Decoder(dec)
			var u timeArshaler
			if uo.Format != "" && uo.FormatDepth == xd.Tokens.Depth() {
				if !u.initFormat(uo.Format) {
					return newInvalidFormatError(dec, t, uo)
				}
			} else if uo.Flags.Get(jsonflags.FormatTimeWithLegacySemantics) {
				u.looseRFC3339 = true
			}

			stringify := !u.isNumeric() || xd.Tokens.Last.NeedObjectName() || uo.Flags.Get(jsonflags.StringifyNumbers)
			var flags jsonwire.ValueFlags
			tt := va.Addr().Interface().(*time.Time)
			val, err := xd.ReadValue(&flags)
			if err != nil {
				return err
			}
			switch k := val.Kind(); k {
			case 'n':
				if !uo.Flags.Get(jsonflags.MergeWithLegacySemantics) {
					*tt = time.Time{}
				}
				return nil
			case '"':
				if !stringify {
					break
				}
				val = jsonwire.UnquoteMayCopy(val, flags.IsVerbatim())
				if err := u.unmarshal(val); err != nil {
					if uo.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
						return err // unlike marshal, never wrapped
					}
					return newUnmarshalErrorAfter(dec, t, err)
				}
				*tt = u.tt
				return nil
			case '0':
				if stringify {
					break
				}
				if err := u.unmarshal(val); err != nil {
					if uo.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
						return err // unlike marshal, never wrapped
					}
					return newUnmarshalErrorAfter(dec, t, err)
				}
				*tt = u.tt
				return nil
			}
			return newUnmarshalErrorAfter(dec, t, nil)
		}
	}
	return fncs
}

type durationArshaler struct {
	td time.Duration

	// base records the representation where:
	//   - 0 uses time.Duration.String
	//   - 1e0, 1e3, 1e6, or 1e9 use a decimal encoding of the duration as
	//     nanoseconds, microseconds, milliseconds, or seconds.
	base uint64
}

func (a *durationArshaler) initFormat(format string) (ok bool) {
	switch format {
	case "units":
		a.base = 0
	case "sec":
		a.base = 1e9
	case "milli":
		a.base = 1e6
	case "micro":
		a.base = 1e3
	case "nano":
		a.base = 1e0
	default:
		return false
	}
	return true
}

func (a *durationArshaler) isNumeric() bool {
	return a.base != 0 && a.base != 60
}

func (a *durationArshaler) appendMarshal(b []byte) ([]byte, error) {
	switch a.base {
	case 0:
		return append(b, a.td.String()...), nil
	default:
		return appendDurationBase10(b, a.td, a.base), nil
	}
}

func (a *durationArshaler) unmarshal(b []byte) (err error) {
	switch a.base {
	case 0:
		a.td, err = time.ParseDuration(string(b))
	default:
		a.td, err = parseDurationBase10(b, a.base)
	}
	return err
}

type timeArshaler struct {
	tt time.Time

	// base records the representation where:
	//   - 0 uses RFC 3339 encoding of the timestamp
	//   - 1e0, 1e3, 1e6, or 1e9 use a decimal encoding of the timestamp as
	//     seconds, milliseconds, microseconds, or nanoseconds since Unix epoch.
	//   - math.MaxUint uses time.Time.Format to encode the timestamp
	base   uint64
	format string // time format passed to time.Parse

	looseRFC3339 bool
}

func (a *timeArshaler) initFormat(format string) bool {
	// We assume that an exported constant in the time package will
	// always start with an uppercase ASCII letter.
	if len(format) == 0 {
		return false
	}
	a.base = math.MaxUint // implies custom format
	if c := format[0]; !('a' <= c && c <= 'z') && !('A' <= c && c <= 'Z') {
		a.format = format
		return true
	}
	switch format {
	case "ANSIC":
		a.format = time.ANSIC
	case "UnixDate":
		a.format = time.UnixDate
	case "RubyDate":
		a.format = time.RubyDate
	case "RFC822":
		a.format = time.RFC822
	case "RFC822Z":
		a.format = time.RFC822Z
	case "RFC850":
		a.format = time.RFC850
	case "RFC1123":
		a.format = time.RFC1123
	case "RFC1123Z":
		a.format = time.RFC1123Z
	case "RFC3339":
		a.base = 0
		a.format = time.RFC3339
	case "RFC3339Nano":
		a.base = 0
		a.format = time.RFC3339Nano
	case "Kitchen":
		a.format = time.Kitchen
	case "Stamp":
		a.format = time.Stamp
	case "StampMilli":
		a.format = time.StampMilli
	case "StampMicro":
		a.format = time.StampMicro
	case "StampNano":
		a.format = time.StampNano
	case "DateTime":
		a.format = time.DateTime
	case "DateOnly":
		a.format = time.DateOnly
	case "TimeOnly":
		a.format = time.TimeOnly
	case "unix":
		a.base = 1e0
	case "unixmilli":
		a.base = 1e3
	case "unixmicro":
		a.base = 1e6
	case "unixnano":
		a.base = 1e9
	default:
		// Reject any Go identifier in case new constants are supported.
		if strings.TrimFunc(format, isLetterOrDigit) == "" {
			return false
		}
		a.format = format
	}
	return true
}

func (a *timeArshaler) isNumeric() bool {
	return int(a.base) > 0
}

func (a *timeArshaler) hasCustomFormat() bool {
	return a.base == math.MaxUint
}

func (a *timeArshaler) appendMarshal(b []byte) ([]byte, error) {
	switch a.base {
	case 0:
		format := cmp.Or(a.format, time.RFC3339Nano)
		n0 := len(b)
		b = a.tt.AppendFormat(b, format)
		// Not all Go timestamps can be represented as valid RFC 3339.
		// Explicitly check for these edge cases.
		// See https://go.dev/issue/4556 and https://go.dev/issue/54580.
		switch b := b[n0:]; {
		case b[len("9999")] != '-': // year must be exactly 4 digits wide
			return b, errors.New("year outside of range [0,9999]")
		case b[len(b)-1] != 'Z':
			c := b[len(b)-len("Z07:00")]
			if ('0' <= c && c <= '9') || parseDec2(b[len(b)-len("07:00"):]) >= 24 {
				return b, errors.New("timezone hour outside of range [0,23]")
			}
		}
		return b, nil
	case math.MaxUint:
		return a.tt.AppendFormat(b, a.format), nil
	default:
		return appendTimeUnix(b, a.tt, a.base), nil
	}
}

func (a *timeArshaler) unmarshal(b []byte) (err error) {
	switch a.base {
	case 0:
		// Use time.Time.UnmarshalText to avoid possible string allocation.
		if err := a.tt.UnmarshalText(b); err != nil {
			return err
		}
		// TODO(https://go.dev/issue/57912):
		// RFC 3339 specifies the grammar for a valid timestamp.
		// However, the parsing functionality in "time" is too loose and
		// incorrectly accepts invalid timestamps as valid.
		// Remove these manual checks when "time" checks it for us.
		newParseError := func(layout, value, layoutElem, valueElem, message string) error {
			return &time.ParseError{Layout: layout, Value: value, LayoutElem: layoutElem, ValueElem: valueElem, Message: message}
		}
		switch {
		case a.looseRFC3339:
			return nil
		case b[len("2006-01-02T")+1] == ':': // hour must be two digits
			return newParseError(time.RFC3339, string(b), "15", string(b[len("2006-01-02T"):][:1]), "")
		case b[len("2006-01-02T15:04:05")] == ',': // sub-second separator must be a period
			return newParseError(time.RFC3339, string(b), ".", ",", "")
		case b[len(b)-1] != 'Z':
			switch {
			case parseDec2(b[len(b)-len("07:00"):]) >= 24: // timezone hour must be in range
				return newParseError(time.RFC3339, string(b), "Z07:00", string(b[len(b)-len("Z07:00"):]), ": timezone hour out of range")
			case parseDec2(b[len(b)-len("00"):]) >= 60: // timezone minute must be in range
				return newParseError(time.RFC3339, string(b), "Z07:00", string(b[len(b)-len("Z07:00"):]), ": timezone minute out of range")
			}
		}
		return nil
	case math.MaxUint:
		a.tt, err = time.Parse(a.format, string(b))
		return err
	default:
		a.tt, err = parseTimeUnix(b, a.base)
		return err
	}
}

// appendDurationBase10 appends d formatted as a decimal fractional number,
// where pow10 is a power-of-10 used to scale down the number.
func appendDurationBase10(b []byte, d time.Duration, pow10 uint64) []byte {
	b, n := mayAppendDurationSign(b, d)            // append sign
	whole, frac := bits.Div64(0, n, uint64(pow10)) // compute whole and frac fields
	b = strconv.AppendUint(b, whole, 10)           // append whole field
	return appendFracBase10(b, frac, pow10)        // append frac field
}

// parseDurationBase10 parses d from a decimal fractional number,
// where pow10 is a power-of-10 used to scale up the number.
func parseDurationBase10(b []byte, pow10 uint64) (time.Duration, error) {
	suffix, neg := consumeSign(b)                            // consume sign
	wholeBytes, fracBytes := bytesCutByte(suffix, '.', true) // consume whole and frac fields
	whole, okWhole := jsonwire.ParseUint(wholeBytes)         // parse whole field; may overflow
	frac, okFrac := parseFracBase10(fracBytes, pow10)        // parse frac field
	hi, lo := bits.Mul64(whole, uint64(pow10))               // overflow if hi > 0
	sum, co := bits.Add64(lo, uint64(frac), 0)               // overflow if co > 0
	switch d := mayApplyDurationSign(sum, neg); {            // overflow if neg != (d < 0)
	case (!okWhole && whole != math.MaxUint64) || !okFrac:
		return 0, fmt.Errorf("invalid duration %q: %w", b, strconv.ErrSyntax)
	case !okWhole || hi > 0 || co > 0 || neg != (d < 0):
		return 0, fmt.Errorf("invalid duration %q: %w", b, strconv.ErrRange)
	default:
		return d, nil
	}
}

// mayAppendDurationSign appends a negative sign if n is negative.
func mayAppendDurationSign(b []byte, d time.Duration) ([]byte, uint64) {
	if d < 0 {
		b = append(b, '-')
		d *= -1
	}
	return b, uint64(d)
}

// mayApplyDurationSign inverts n if neg is specified.
func mayApplyDurationSign(n uint64, neg bool) time.Duration {
	if neg {
		return -1 * time.Duration(n)
	} else {
		return +1 * time.Duration(n)
	}
}

// appendTimeUnix appends t formatted as a decimal fractional number,
// where pow10 is a power-of-10 used to scale up the number.
func appendTimeUnix(b []byte, t time.Time, pow10 uint64) []byte {
	sec, nsec := t.Unix(), int64(t.Nanosecond())
	if sec < 0 {
		b = append(b, '-')
		sec, nsec = negateSecNano(sec, nsec)
	}
	switch {
	case pow10 == 1e0: // fast case where units is in seconds
		b = strconv.AppendUint(b, uint64(sec), 10)
		return appendFracBase10(b, uint64(nsec), 1e9)
	case uint64(sec) < 1e9: // intermediate case where units is not seconds, but no overflow
		b = strconv.AppendUint(b, uint64(sec)*uint64(pow10)+uint64(uint64(nsec)/(1e9/pow10)), 10)
		return appendFracBase10(b, (uint64(nsec)*pow10)%1e9, 1e9)
	default: // slow case where units is not seconds and overflow would occur
		b = strconv.AppendUint(b, uint64(sec), 10)
		b = appendPaddedBase10(b, uint64(nsec)/(1e9/pow10), pow10)
		return appendFracBase10(b, (uint64(nsec)*pow10)%1e9, 1e9)
	}
}

// parseTimeUnix parses t formatted as a decimal fractional number,
// where pow10 is a power-of-10 used to scale down the number.
func parseTimeUnix(b []byte, pow10 uint64) (time.Time, error) {
	suffix, neg := consumeSign(b)                            // consume sign
	wholeBytes, fracBytes := bytesCutByte(suffix, '.', true) // consume whole and frac fields
	whole, okWhole := jsonwire.ParseUint(wholeBytes)         // parse whole field; may overflow
	frac, okFrac := parseFracBase10(fracBytes, 1e9/pow10)    // parse frac field
	var sec, nsec int64
	switch {
	case pow10 == 1e0: // fast case where units is in seconds
		sec = int64(whole) // check overflow later after negation
		nsec = int64(frac) // cannot overflow
	case okWhole: // intermediate case where units is not seconds, but no overflow
		sec = int64(whole / pow10)                     // check overflow later after negation
		nsec = int64((whole%pow10)*(1e9/pow10) + frac) // cannot overflow
	case !okWhole && whole == math.MaxUint64: // slow case where units is not seconds and overflow occurred
		width := int(math.Log10(float64(pow10)))                                // compute len(strconv.Itoa(pow10-1))
		whole, okWhole = jsonwire.ParseUint(wholeBytes[:len(wholeBytes)-width]) // parse the upper whole field
		mid, _ := parsePaddedBase10(wholeBytes[len(wholeBytes)-width:], pow10)  // parse the lower whole field
		sec = int64(whole)                                                      // check overflow later after negation
		nsec = int64(mid*(1e9/pow10) + frac)                                    // cannot overflow
	}
	if neg {
		sec, nsec = negateSecNano(sec, nsec)
	}
	switch t := time.Unix(sec, nsec).UTC(); {
	case (!okWhole && whole != math.MaxUint64) || !okFrac:
		return time.Time{}, fmt.Errorf("invalid time %q: %w", b, strconv.ErrSyntax)
	case !okWhole || neg != (t.Unix() < 0):
		return time.Time{}, fmt.Errorf("invalid time %q: %w", b, strconv.ErrRange)
	default:
		return t, nil
	}
}

// negateSecNano negates a Unix timestamp, where nsec must be within [0, 1e9).
func negateSecNano(sec, nsec int64) (int64, int64) {
	sec = ^sec               // twos-complement negation (i.e., -1*sec + 1)
	nsec = -nsec + 1e9       // negate nsec and add 1e9 (which is the extra +1 from sec negation)
	sec += int64(nsec / 1e9) // handle possible overflow of nsec if it started as zero
	nsec %= 1e9              // ensure nsec stays within [0, 1e9)
	return sec, nsec
}

// appendFracBase10 appends the fraction of n/max10,
// where max10 is a power-of-10 that is larger than n.
func appendFracBase10(b []byte, n, max10 uint64) []byte {
	if n == 0 {
		return b
	}
	return bytes.TrimRight(appendPaddedBase10(append(b, '.'), n, max10), "0")
}

// parseFracBase10 parses the fraction of n/max10,
// where max10 is a power-of-10 that is larger than n.
func parseFracBase10(b []byte, max10 uint64) (n uint64, ok bool) {
	switch {
	case len(b) == 0:
		return 0, true
	case len(b) < len(".0") || b[0] != '.':
		return 0, false
	}
	return parsePaddedBase10(b[len("."):], max10)
}

// appendPaddedBase10 appends a zero-padded encoding of n,
// where max10 is a power-of-10 that is larger than n.
func appendPaddedBase10(b []byte, n, max10 uint64) []byte {
	if n < max10/10 {
		// Formatting of n is shorter than log10(max10),
		// so add max10/10 to ensure the length is equal to log10(max10).
		i := len(b)
		b = strconv.AppendUint(b, n+max10/10, 10)
		b[i]-- // subtract the addition of max10/10
		return b
	}
	return strconv.AppendUint(b, n, 10)
}

// parsePaddedBase10 parses b as the zero-padded encoding of n,
// where max10 is a power-of-10 that is larger than n.
// Truncated suffix is treated as implicit zeros.
// Extended suffix is ignored, but verified to contain only digits.
func parsePaddedBase10(b []byte, max10 uint64) (n uint64, ok bool) {
	pow10 := uint64(1)
	for pow10 < max10 {
		n *= 10
		if len(b) > 0 {
			if b[0] < '0' || '9' < b[0] {
				return n, false
			}
			n += uint64(b[0] - '0')
			b = b[1:]
		}
		pow10 *= 10
	}
	if len(b) > 0 && len(bytes.TrimRight(b, "0123456789")) > 0 {
		return n, false // trailing characters are not digits
	}
	return n, true
}

// consumeSign consumes an optional leading negative sign.
func consumeSign(b []byte) ([]byte, bool) {
	if len(b) > 0 && b[0] == '-' {
		return b[len("-"):], true
	}
	return b, false
}

// bytesCutByte is similar to bytes.Cut(b, []byte{c}),
// except c may optionally be included as part of the suffix.
func bytesCutByte(b []byte, c byte, include bool) ([]byte, []byte) {
	if i := bytes.IndexByte(b, c); i >= 0 {
		if include {
			return b[:i], b[i:]
		}
		return b[:i], b[i+1:]
	}
	return b, nil
}

// parseDec2 parses b as an unsigned, base-10, 2-digit number.
// The result is undefined if digits are not base-10.
func parseDec2(b []byte) byte {
	if len(b) < 2 {
		return 0
	}
	return 10*(b[0]-'0') + (b[1] - '0')
}
