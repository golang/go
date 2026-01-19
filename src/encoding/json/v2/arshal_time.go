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
					return newInvalidFormatError(enc, t)
				}
			} else if mo.Flags.Get(jsonflags.FormatDurationAsNano) {
				return marshalNano(enc, va, mo)
			} else {
				// TODO(https://go.dev/issue/71631): Decide on default duration representation.
				return newMarshalErrorBefore(enc, t, errors.New("no default representation (see https://go.dev/issue/71631); specify an explicit format"))
			}

			m.td, _ = reflect.TypeAssert[time.Duration](va.Value)
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
					return newInvalidFormatError(dec, t)
				}
			} else if uo.Flags.Get(jsonflags.FormatDurationAsNano) {
				return unmarshalNano(dec, va, uo)
			} else {
				// TODO(https://go.dev/issue/71631): Decide on default duration representation.
				return newUnmarshalErrorBeforeWithSkipping(dec, t, errors.New("no default representation (see https://go.dev/issue/71631); specify an explicit format"))
			}

			stringify := !u.isNumeric() || xd.Tokens.Last.NeedObjectName() || uo.Flags.Get(jsonflags.StringifyNumbers)
			var flags jsonwire.ValueFlags
			td, _ := reflect.TypeAssert[*time.Duration](va.Addr())
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
					return newInvalidFormatError(enc, t)
				}
			}

			m.tt, _ = reflect.TypeAssert[time.Time](va.Value)
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
					return newInvalidFormatError(dec, t)
				}
			} else if uo.Flags.Get(jsonflags.ParseTimeWithLooseRFC3339) {
				u.looseRFC3339 = true
			}

			stringify := !u.isNumeric() || xd.Tokens.Last.NeedObjectName() || uo.Flags.Get(jsonflags.StringifyNumbers)
			var flags jsonwire.ValueFlags
			tt, _ := reflect.TypeAssert[*time.Time](va.Addr())
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
	//   - 8601 uses ISO 8601
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
	case "iso8601":
		a.base = 8601
	default:
		return false
	}
	return true
}

func (a *durationArshaler) isNumeric() bool {
	return a.base != 0 && a.base != 8601
}

func (a *durationArshaler) appendMarshal(b []byte) ([]byte, error) {
	switch a.base {
	case 0:
		return append(b, a.td.String()...), nil
	case 8601:
		return appendDurationISO8601(b, a.td), nil
	default:
		return appendDurationBase10(b, a.td, a.base), nil
	}
}

func (a *durationArshaler) unmarshal(b []byte) (err error) {
	switch a.base {
	case 0:
		a.td, err = time.ParseDuration(string(b))
	case 8601:
		a.td, err = parseDurationISO8601(b)
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
	suffix, neg := consumeSign(b, false)                     // consume sign
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

// appendDurationISO8601 appends an ISO 8601 duration with a restricted grammar,
// where leading and trailing zeroes and zero-value designators are omitted.
// It only uses hour, minute, and second designators since ISO 8601 defines
// those as being "accurate", while year, month, week, and day are "nominal".
func appendDurationISO8601(b []byte, d time.Duration) []byte {
	if d == 0 {
		return append(b, "PT0S"...)
	}
	b, n := mayAppendDurationSign(b, d)
	b = append(b, "PT"...)
	n, nsec := bits.Div64(0, n, 1e9)  // compute nsec field
	n, sec := bits.Div64(0, n, 60)    // compute sec field
	hour, min := bits.Div64(0, n, 60) // compute hour and min fields
	if hour > 0 {
		b = append(strconv.AppendUint(b, hour, 10), 'H')
	}
	if min > 0 {
		b = append(strconv.AppendUint(b, min, 10), 'M')
	}
	if sec > 0 || nsec > 0 {
		b = append(appendFracBase10(strconv.AppendUint(b, sec, 10), nsec, 1e9), 'S')
	}
	return b
}

// daysPerYear is the exact average number of days in a year according to
// the Gregorian calendar, which has an extra day each year that is
// a multiple of 4, unless it is evenly divisible by 100 but not by 400.
// This does not take into account leap seconds, which are not deterministic.
const daysPerYear = 365.2425

var errInaccurateDateUnits = errors.New("inaccurate year, month, week, or day units")

// parseDurationISO8601 parses a duration according to ISO 8601-1:2019,
// section 5.5.2.2 and 5.5.2.3 with the following restrictions or extensions:
//
//   - A leading minus sign is permitted for negative duration according
//     to ISO 8601-2:2019, section 4.4.1.9. We do not permit negative values
//     for each "time scale component", which is permitted by section 4.4.1.1,
//     but rarely supported by parsers.
//
//   - A leading plus sign is permitted (and ignored).
//     This is not required by ISO 8601, but not forbidden either.
//     There is some precedent for this as it is supported by the principle of
//     duration arithmetic as specified in ISO 8601-2-2019, section 14.1.
//     Of note, the JavaScript grammar for ISO 8601 permits a leading plus sign.
//
//   - A fractional value is only permitted for accurate units
//     (i.e., hour, minute, and seconds) in the last time component,
//     which is permissible by ISO 8601-1:2019, section 5.5.2.3.
//
//   - Both periods ('.') and commas (',') are supported as the separator
//     between the integer part and fraction part of a number,
//     as specified in ISO 8601-1:2019, section 3.2.6.
//     While ISO 8601 recommends comma as the default separator,
//     most formatters uses a period.
//
//   - Leading zeros are ignored. This is not required by ISO 8601,
//     but also not forbidden by the standard. Many parsers support this.
//
//   - Lowercase designators are supported. This is not required by ISO 8601,
//     but also not forbidden by the standard. Many parsers support this.
//
// If the nominal units of year, month, week, or day are present,
// this produces a best-effort value and also reports [errInaccurateDateUnits].
//
// The accepted grammar is identical to JavaScript's Duration:
//
//	https://tc39.es/proposal-temporal/#prod-Duration
//
// We follow JavaScript's grammar as JSON itself is derived from JavaScript.
// The Temporal.Duration.toJSON method is guaranteed to produce an output
// that can be parsed by this function so long as arithmetic in JavaScript
// do not use a largestUnit value higher than "hours" (which is the default).
// Even if it does, this will do a best-effort parsing with inaccurate units,
// but report [errInaccurateDateUnits].
func parseDurationISO8601(b []byte) (time.Duration, error) {
	var invalid, overflow, inaccurate, sawFrac bool
	var sumNanos, n, co uint64

	// cutBytes is like [bytes.Cut], but uses either c0 or c1 as the separator.
	cutBytes := func(b []byte, c0, c1 byte) (prefix, suffix []byte, ok bool) {
		for i, c := range b {
			if c == c0 || c == c1 {
				return b[:i], b[i+1:], true
			}
		}
		return b, nil, false
	}

	// mayParseUnit attempts to parse another date or time number
	// identified by the desHi and desLo unit characters.
	// If the part is absent for current unit, it returns b as is.
	mayParseUnit := func(b []byte, desHi, desLo byte, unit time.Duration) []byte {
		number, suffix, ok := cutBytes(b, desHi, desLo)
		if !ok || sawFrac {
			return b // designator is not present or already saw fraction, which can only be in the last component
		}

		// Parse the number.
		// A fraction allowed for the accurate units in the last part.
		whole, frac, ok := cutBytes(number, '.', ',')
		if ok {
			sawFrac = true
			invalid = invalid || len(frac) == len("") || unit > time.Hour
			if unit == time.Second {
				n, ok = parsePaddedBase10(frac, uint64(time.Second))
				invalid = invalid || !ok
			} else {
				f, err := strconv.ParseFloat("0."+string(frac), 64)
				invalid = invalid || err != nil || len(bytes.Trim(frac[len("."):], "0123456789")) > 0
				n = uint64(math.Round(f * float64(unit))) // never overflows since f is within [0..1]
			}
			sumNanos, co = bits.Add64(sumNanos, n, 0) // overflow if co > 0
			overflow = overflow || co > 0
		}
		for len(whole) > 1 && whole[0] == '0' {
			whole = whole[len("0"):] // trim leading zeros
		}
		n, ok := jsonwire.ParseUint(whole)         // overflow if !ok && MaxUint64
		hi, lo := bits.Mul64(n, uint64(unit))      // overflow if hi > 0
		sumNanos, co = bits.Add64(sumNanos, lo, 0) // overflow if co > 0
		invalid = invalid || (!ok && n != math.MaxUint64)
		overflow = overflow || (!ok && n == math.MaxUint64) || hi > 0 || co > 0
		inaccurate = inaccurate || unit > time.Hour
		return suffix
	}

	suffix, neg := consumeSign(b, true)
	prefix, suffix, okP := cutBytes(suffix, 'P', 'p')
	durDate, durTime, okT := cutBytes(suffix, 'T', 't')
	invalid = invalid || len(prefix) > 0 || !okP || (okT && len(durTime) == 0) || len(durDate)+len(durTime) == 0
	if len(durDate) > 0 { // nominal portion of the duration
		durDate = mayParseUnit(durDate, 'Y', 'y', time.Duration(daysPerYear*24*60*60*1e9))
		durDate = mayParseUnit(durDate, 'M', 'm', time.Duration(daysPerYear/12*24*60*60*1e9))
		durDate = mayParseUnit(durDate, 'W', 'w', time.Duration(7*24*60*60*1e9))
		durDate = mayParseUnit(durDate, 'D', 'd', time.Duration(24*60*60*1e9))
		invalid = invalid || len(durDate) > 0 // unknown elements
	}
	if len(durTime) > 0 { // accurate portion of the duration
		durTime = mayParseUnit(durTime, 'H', 'h', time.Duration(60*60*1e9))
		durTime = mayParseUnit(durTime, 'M', 'm', time.Duration(60*1e9))
		durTime = mayParseUnit(durTime, 'S', 's', time.Duration(1e9))
		invalid = invalid || len(durTime) > 0 // unknown elements
	}
	d := mayApplyDurationSign(sumNanos, neg)
	overflow = overflow || (neg != (d < 0) && d != 0) // overflows signed duration

	switch {
	case invalid:
		return 0, fmt.Errorf("invalid ISO 8601 duration %q: %w", b, strconv.ErrSyntax)
	case overflow:
		return 0, fmt.Errorf("invalid ISO 8601 duration %q: %w", b, strconv.ErrRange)
	case inaccurate:
		return d, fmt.Errorf("invalid ISO 8601 duration %q: %w", b, errInaccurateDateUnits)
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
	suffix, neg := consumeSign(b, false)                     // consume sign
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

// consumeSign consumes an optional leading negative or positive sign.
func consumeSign(b []byte, allowPlus bool) ([]byte, bool) {
	if len(b) > 0 {
		if b[0] == '-' {
			return b[len("-"):], true
		} else if b[0] == '+' && allowPlus {
			return b[len("+"):], false
		}
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
