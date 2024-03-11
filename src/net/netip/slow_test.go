// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netip_test

import (
	"fmt"
	. "net/netip"
	"strconv"
	"strings"
)

// zeros is a slice of eight stringified zeros. It's used in
// parseIPSlow to construct slices of specific amounts of zero fields,
// from 1 to 8.
var zeros = []string{"0", "0", "0", "0", "0", "0", "0", "0"}

// parseIPSlow is like ParseIP, but aims for readability above
// speed. It's the reference implementation for correctness checking
// and against which we measure optimized parsers.
//
// parseIPSlow understands the following forms of IP addresses:
//   - Regular IPv4: 1.2.3.4
//   - IPv4 with many leading zeros: 0000001.0000002.0000003.0000004
//   - Regular IPv6: 1111:2222:3333:4444:5555:6666:7777:8888
//   - IPv6 with many leading zeros: 00000001:0000002:0000003:0000004:0000005:0000006:0000007:0000008
//   - IPv6 with zero blocks elided: 1111:2222::7777:8888
//   - IPv6 with trailing 32 bits expressed as IPv4: 1111:2222:3333:4444:5555:6666:77.77.88.88
//
// It does not process the following IP address forms, which have been
// varyingly accepted by some programs due to an under-specification
// of the shapes of IPv4 addresses:
//
//   - IPv4 as a single 32-bit uint: 4660 (same as "1.2.3.4")
//   - IPv4 with octal numbers: 0300.0250.0.01 (same as "192.168.0.1")
//   - IPv4 with hex numbers: 0xc0.0xa8.0x0.0x1 (same as "192.168.0.1")
//   - IPv4 in "class-B style": 1.2.52 (same as "1.2.3.4")
//   - IPv4 in "class-A style": 1.564 (same as "1.2.3.4")
func parseIPSlow(s string) (Addr, error) {
	// Identify and strip out the zone, if any. There should be 0 or 1
	// '%' in the string.
	var zone string
	fs := strings.Split(s, "%")
	switch len(fs) {
	case 1:
		// No zone, that's fine.
	case 2:
		s, zone = fs[0], fs[1]
		if zone == "" {
			return Addr{}, fmt.Errorf("netaddr.ParseIP(%q): no zone after zone specifier", s)
		}
	default:
		return Addr{}, fmt.Errorf("netaddr.ParseIP(%q): too many zone specifiers", s) // TODO: less specific?
	}

	// IPv4 by itself is easy to do in a helper.
	if strings.Count(s, ":") == 0 {
		if zone != "" {
			return Addr{}, fmt.Errorf("netaddr.ParseIP(%q): IPv4 addresses cannot have a zone", s)
		}
		return parseIPv4Slow(s)
	}

	normal, err := normalizeIPv6Slow(s)
	if err != nil {
		return Addr{}, err
	}

	// At this point, we've normalized the address back into 8 hex
	// fields of 16 bits each. Parse that.
	fs = strings.Split(normal, ":")
	if len(fs) != 8 {
		return Addr{}, fmt.Errorf("netaddr.ParseIP(%q): wrong size address", s)
	}
	var ret [16]byte
	for i, f := range fs {
		a, b, err := parseWord(f)
		if err != nil {
			return Addr{}, err
		}
		ret[i*2] = a
		ret[i*2+1] = b
	}

	return AddrFrom16(ret).WithZone(zone), nil
}

// normalizeIPv6Slow expands s, which is assumed to be an IPv6
// address, to its canonical text form.
//
// The canonical form of an IPv6 address is 8 colon-separated fields,
// where each field should be a hex value from 0 to ffff. This
// function does not verify the contents of each field.
//
// This function performs two transformations:
//   - The last 32 bits of an IPv6 address may be represented in
//     IPv4-style dotted quad form, as in 1:2:3:4:5:6:7.8.9.10. That
//     address is transformed to its hex equivalent,
//     e.g. 1:2:3:4:5:6:708:90a.
//   - An address may contain one "::", which expands into as many
//     16-bit blocks of zeros as needed to make the address its correct
//     full size. For example, fe80::1:2 expands to fe80:0:0:0:0:0:1:2.
//
// Both short forms may be present in a single address,
// e.g. fe80::1.2.3.4.
func normalizeIPv6Slow(orig string) (string, error) {
	s := orig

	// Find and convert an IPv4 address in the final field, if any.
	i := strings.LastIndex(s, ":")
	if i == -1 {
		return "", fmt.Errorf("netaddr.ParseIP(%q): invalid IP address", orig)
	}
	if strings.Contains(s[i+1:], ".") {
		ip, err := parseIPv4Slow(s[i+1:])
		if err != nil {
			return "", err
		}
		a4 := ip.As4()
		s = fmt.Sprintf("%s:%02x%02x:%02x%02x", s[:i], a4[0], a4[1], a4[2], a4[3])
	}

	// Find and expand a ::, if any.
	fs := strings.Split(s, "::")
	switch len(fs) {
	case 1:
		// No ::, nothing to do.
	case 2:
		lhs, rhs := fs[0], fs[1]
		// Found a ::, figure out how many zero blocks need to be
		// inserted.
		nblocks := strings.Count(lhs, ":") + strings.Count(rhs, ":")
		if lhs != "" {
			nblocks++
		}
		if rhs != "" {
			nblocks++
		}
		if nblocks > 7 {
			return "", fmt.Errorf("netaddr.ParseIP(%q): address too long", orig)
		}
		fs = nil
		// Either side of the :: can be empty. We don't want empty
		// fields to feature in the final normalized address.
		if lhs != "" {
			fs = append(fs, lhs)
		}
		fs = append(fs, zeros[:8-nblocks]...)
		if rhs != "" {
			fs = append(fs, rhs)
		}
		s = strings.Join(fs, ":")
	default:
		// Too many ::
		return "", fmt.Errorf("netaddr.ParseIP(%q): invalid IP address", orig)
	}

	return s, nil
}

// parseIPv4Slow parses and returns an IPv4 address in dotted quad
// form, e.g. "192.168.0.1". It is slow but easy to read, and the
// reference implementation against which we compare faster
// implementations for correctness.
func parseIPv4Slow(s string) (Addr, error) {
	fs := strings.Split(s, ".")
	if len(fs) != 4 {
		return Addr{}, fmt.Errorf("netaddr.ParseIP(%q): invalid IP address", s)
	}
	var ret [4]byte
	for i := range ret {
		val, err := strconv.ParseUint(fs[i], 10, 8)
		if err != nil {
			return Addr{}, err
		}
		ret[i] = uint8(val)
	}
	return AddrFrom4([4]byte{ret[0], ret[1], ret[2], ret[3]}), nil
}

// parseWord converts a 16-bit hex string into its corresponding
// two-byte value.
func parseWord(s string) (byte, byte, error) {
	if len(s) > 4 {
		return 0, 0, fmt.Errorf("parseWord(%q): invalid word", s)
	}
	ret, err := strconv.ParseUint(s, 16, 16)
	if err != nil {
		return 0, 0, err
	}
	return uint8(ret >> 8), uint8(ret), nil
}
