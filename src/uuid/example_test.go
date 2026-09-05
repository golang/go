// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uuid_test

import (
	"fmt"
	"log"
	"slices"
	"uuid"
)

func ExampleParse() {
	for _, s := range []string{
		"f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
		"{f81d4fae-7dec-11d0-a765-00a0c91e6bf6}",
		"urn:uuid:f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
		"f81d4fae7dec11d0a76500a0c91e6bf6",
	} {
		u, err := uuid.Parse(s)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(u)
	}
	// Output:
	// f81d4fae-7dec-11d0-a765-00a0c91e6bf6
	// f81d4fae-7dec-11d0-a765-00a0c91e6bf6
	// f81d4fae-7dec-11d0-a765-00a0c91e6bf6
	// f81d4fae-7dec-11d0-a765-00a0c91e6bf6
}

func ExampleUUID_Compare() {
	ids := []uuid.UUID{
		uuid.MustParse("f81d4fae-7dec-11d0-a765-00a0c91e6bf6"),
		uuid.MustParse("00000000-0000-0000-0000-000000000000"),
		uuid.MustParse("ffffffff-ffff-ffff-ffff-ffffffffffff"),
	}
	slices.SortFunc(ids, uuid.UUID.Compare)
	for _, id := range ids {
		fmt.Println(id)
	}
	// Output:
	// 00000000-0000-0000-0000-000000000000
	// f81d4fae-7dec-11d0-a765-00a0c91e6bf6
	// ffffffff-ffff-ffff-ffff-ffffffffffff
}
