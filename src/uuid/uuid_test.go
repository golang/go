// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uuid_test

import (
	"encoding/binary"
	"testing"
	"testing/synctest"
	"time"
	"uuid"
)

func TestNew(t *testing.T) {
	for _, test := range []struct {
		name    string
		newf    func() uuid.UUID
		version byte
		variant byte
	}{{
		name:    "New",
		newf:    uuid.New,
		version: 4,
		variant: 0b10,
	}, {
		name:    "NewV4",
		newf:    uuid.NewV4,
		version: 4,
		variant: 0b10,
	}, {
		name:    "NewV7",
		newf:    uuid.NewV7,
		version: 7,
		variant: 0b10,
	}} {
		u := test.newf()
		if got, want := version(u), test.version; got != want {
			t.Errorf("%v: got version %v, want %v", test.name, got, want)
		}
		if got, want := variant(u), test.variant; got != want {
			t.Errorf("%v: got variant %v, want %v", test.name, got, want)
		}
	}
}

func version(u uuid.UUID) byte {
	return u[6] >> 4
}

func variant(u uuid.UUID) byte {
	return u[8] >> 6
}

func TestNewV7Millis(t *testing.T) {
	// Verify the unix_ts_ms field of a UUIDv7 is set correctly.
	check := func(t *testing.T) {
		t.Helper()
		u := uuid.NewV7()
		got := binary.BigEndian.Uint64(u[:8]) >> 16
		want := uint64(time.Now().UnixMilli())
		if got != want {
			t.Errorf("at %v, NewV7() = %v; millis = %x, want %x", time.Now(), u, got, want)
		}
	}
	synctest.Test(t, func(t *testing.T) {
		check(t)
		time.Sleep(1 * time.Hour)
		check(t)
		time.Sleep(time.Second - 1*time.Nanosecond) // maximum fractional seconds
		check(t)
		time.Sleep(2 * time.Nanosecond) // minimum fractional seconds
		check(t)
	})
	// Testing in a new bubble causes time to go backwards.
	// UUIDs should use the new time.
	synctest.Test(t, func(t *testing.T) {
		check(t)
	})
}

func TestNewV7Collision(t *testing.T) {
	// Verify UUIDv7s generated at the same instant do not collide and
	// are monotonically increasing.
	synctest.Test(t, func(t *testing.T) {
		last := uuid.NewV7()
		for range 3 {
			// Enough iterations to overflow the fractional millisecond component
			// several times.
			for range (1 << 12) * 3 {
				u := uuid.NewV7()
				if u.Compare(last) != 1 {
					t.Fatalf("NewV7 returned UUIDs out of order:\nprevious: %v\n current: %v", last, u)
				}
				last = u
			}
			// Time advances, but not as quickly as we are generating UUIDs.
			time.Sleep(1 * time.Millisecond)
		}
	})
}

func TestEncode(t *testing.T) {
	u := uuid.UUID{0xf8, 0x1d, 0x4f, 0xae, 0x7d, 0xec, 0x11, 0xd0, 0xa7, 0x65, 0x00, 0xa0, 0xc9, 0x1e, 0x6b, 0xf6}
	want := "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"
	if got := u.String(); got != want {
		t.Errorf("u.String() = %q, want %q", got, want)
	}
	if got, err := u.MarshalText(); string(got) != want || err != nil {
		t.Errorf("u.MarshalText() = %q, %v; want %q, nil", got, err, want)
	}
	prefix := []byte("urn:uuid:")
	if got, err := u.AppendText(prefix); string(got) != string(prefix)+want || err != nil {
		t.Errorf("u.MarshalAppend(%q) = %q, %v; want %q, nil", prefix, got, err, string(prefix)+want)
	}
}

func TestUnmarshalText(t *testing.T) {
	var got uuid.UUID
	err := got.UnmarshalText([]byte("f81d4fae-7dec-11d0-a765-00a0c91e6bf6"))
	if err != nil {
		t.Errorf("UnmarshalText: %v", err)
	}
	want := uuid.UUID{0xf8, 0x1d, 0x4f, 0xae, 0x7d, 0xec, 0x11, 0xd0, 0xa7, 0x65, 0x00, 0xa0, 0xc9, 0x1e, 0x6b, 0xf6}
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestParseSuccess(t *testing.T) {
	u1 := uuid.UUID{
		0xf8, 0x1d, 0x4f, 0xae,
		0x7d, 0xec,
		0x11, 0xd0,
		0xa7, 0x65,
		0x00, 0xa0, 0xc9, 0x1e, 0x6b, 0xf6,
	}
	for _, test := range []struct {
		s string
		u uuid.UUID
	}{{
		s: "00000000-0000-0000-0000-000000000000",
		u: uuid.Nil(),
	}, {
		s: "ffffffff-ffff-ffff-ffff-ffffffffffff",
		u: uuid.Max(),
	}, {
		s: "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
		u: u1,
	}, {
		s: "F81D4FAE-7DEC-11D0-A765-00A0C91E6BF6",
		u: u1,
	}, {
		s: "f81d4fae7dec11d0a76500a0c91e6bf6",
		u: u1,
	}, {
		s: "{f81d4fae-7dec-11d0-a765-00a0c91e6bf6}",
		u: u1,
	}, {
		s: "urn:uuid:f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
		u: u1,
	}} {
		u, err := uuid.Parse(test.s)
		if err != nil {
			t.Errorf("Parse(%q) = _, %v; want success", test.s, err)
		} else if u != test.u {
			t.Errorf("Parse(%q) = %v, nil; want %v", test.s, u, test.u)
		}
	}
}

func TestParseErrors(t *testing.T) {
	for _, test := range []string{
		"",
		"0000000000000-0000-0000-000000000000",
		"00000000-000000000-0000-000000000000",
		"00000000-0000-000000000-000000000000",
		"00000000-0000-0000-00000000000000000",
		"00000000-0000-0000-0000-00000000000",
		"x0000000-0000-0000-0000-000000000000",
		"00000000-x000-0000-0000-000000000000",
		"00000000-0000-x000-0000-000000000000",
		"00000000-0000-0000-x000-000000000000",
		"00000000-0000-0000-0000-x00000000000",
		"{x0000000-0000-0000-0000-000000000000}",
		"urn:uuid:x000000-0000-0000-0000-000000000000",
		"x0000000000000000000000000000000",
		// Some parsers permit hyphens in non-standard locations,
		// but we currently do not.
		"0000-0000-0000-0000-0000-0000-0000-0000",
		// Combinations of variant encodings that we could choose to parse,
		// but currently do not.
		"{00000000000000000000000000000000}",
		"{urn:uuid:00000000-0000-0000-0000-000000000000}",
		"urn:uuid:00000000000000000000000000000000",
	} {
		got, err := uuid.Parse(test)
		if err == nil {
			t.Errorf("Parse(%q) = %v, nil; want error", test, got)
		}
	}
}

func TestCompare(t *testing.T) {
	uuids := []uuid.UUID{
		uuid.Nil(),
		uuid.MustParse("f81d4fae-7dec-11d0-a765-00a0c91e6bf6"),
		uuid.Max(),
	}
	for i, u := range uuids {
		if got, want := u.Compare(u), 0; got != want {
			t.Errorf("%v.Compare(itself) = %v, want %v", u, got, want)
		}
		if i == 0 {
			continue
		}
		prev := uuids[i-1]
		if got, want := u.Compare(prev), 1; got != want {
			t.Errorf("%v.Compare(%v) = %v, want %v", u, prev, got, want)
		}
		if got, want := prev.Compare(u), -1; got != want {
			t.Errorf("%v.Compare(%v) = %v, want %v", prev, u, got, want)
		}
	}
}

func BenchmarkNewV4(b *testing.B) {
	for b.Loop() {
		uuid.NewV4()
	}
}

func BenchmarkNewV7(b *testing.B) {
	for b.Loop() {
		uuid.NewV7()
	}
}

func BenchmarkString(b *testing.B) {
	u := uuid.MustParse("f81d4fae-7dec-11d0-a765-00a0c91e6bf6")
	for b.Loop() {
		_ = u.String()
	}
}

func BenchmarkParseSuccess(b *testing.B) {
	for b.Loop() {
		uuid.Parse("f81d4fae-7dec-11d0-a765-00a0c91e6bf6")
	}
}

func BenchmarkParseError(b *testing.B) {
	for b.Loop() {
		uuid.Parse("00000000-0000-0000-0000-00000000000X")
	}
}
