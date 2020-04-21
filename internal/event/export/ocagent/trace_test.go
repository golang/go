// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ocagent_test

import (
	"context"
	"errors"
	"testing"

	"golang.org/x/tools/internal/event"
)

func TestTrace(t *testing.T) {
	exporter := registerExporter()
	const prefix = testNodeStr + `
	"spans":[{
		"trace_id":"AAAAAAAAAAAAAAAAAAAAAA==",
		"span_id":"AAAAAAAAAAA=",
		"parent_span_id":"AAAAAAAAAAA=",
		"name":{"value":"event span"},
		"start_time":"1970-01-01T00:00:30Z",
		"end_time":"1970-01-01T00:00:50Z",
		"time_events":{
`
	const suffix = `
		},
		"same_process_as_parent_span":true
	}]
}`

	tests := []struct {
		name string
		run  func(ctx context.Context)
		want string
	}{
		{
			name: "no labels",
			run: func(ctx context.Context) {
				event.Label(ctx)
			},
			want: prefix + `
					"timeEvent":[{"time":"1970-01-01T00:00:40Z"}]
		` + suffix,
		},
		{
			name: "description no error",
			run: func(ctx context.Context) {
				event.Log(ctx, "cache miss", keyDB.Of("godb"))
			},
			want: prefix + `"timeEvent":[{"time":"1970-01-01T00:00:40Z","annotation":{
"description": { "value": "cache miss" },
"attributes": {
	"attributeMap": {
		"db": { "stringValue": { "value": "godb" } }
	}
}
}}]` + suffix,
		},

		{
			name: "description and error",
			run: func(ctx context.Context) {
				event.Error(ctx, "cache miss",
					errors.New("no network connectivity"),
					keyDB.Of("godb"),
				)
			},
			want: prefix + `"timeEvent":[{"time":"1970-01-01T00:00:40Z","annotation":{
"description": { "value": "cache miss" },
"attributes": {
	"attributeMap": {
		"db": { "stringValue": { "value": "godb" } },
		"error": { "stringValue": { "value": "no network connectivity" } }
	}
}
}}]` + suffix,
		},
		{
			name: "no description, but error",
			run: func(ctx context.Context) {
				event.Error(ctx, "",
					errors.New("no network connectivity"),
					keyDB.Of("godb"),
				)
			},
			want: prefix + `"timeEvent":[{"time":"1970-01-01T00:00:40Z","annotation":{
"description": { "value": "no network connectivity" },
"attributes": {
	"attributeMap": {
		"db": { "stringValue": { "value": "godb" } }
	}
}
}}]` + suffix,
		},
		{
			name: "enumerate all attribute types",
			run: func(ctx context.Context) {
				event.Log(ctx, "cache miss",
					key1DB.Of("godb"),

					key2aAge.Of(0.456), // Constant converted into "float64"
					key2bTTL.Of(float32(5000)),
					key2cExpiryMS.Of(float64(1e3)),

					key3aRetry.Of(false),
					key3bStale.Of(true),

					key4aMax.Of(0x7fff), // Constant converted into "int"
					key4bOpcode.Of(int8(0x7e)),
					key4cBase.Of(int16(1<<9)),
					key4eChecksum.Of(int32(0x11f7e294)),
					key4fMode.Of(int64(0644)),

					key5aMin.Of(uint(1)),
					key5bMix.Of(uint8(44)),
					key5cPort.Of(uint16(55678)),
					key5dMinHops.Of(uint32(1<<9)),
					key5eMaxHops.Of(uint64(0xffffff)),
				)
			},
			want: prefix + `"timeEvent":[{"time":"1970-01-01T00:00:40Z","annotation":{
"description": { "value": "cache miss" },
"attributes": {
	"attributeMap": {
		"1_db": { "stringValue": { "value": "godb" } },
		"2a_age": { "doubleValue": 0.456 },
		"2b_ttl": { "doubleValue": 5000 },
		"2c_expiry_ms": { "doubleValue": 1000 },
		"3a_retry": {},
		"3b_stale": { "boolValue": true },
		"4a_max": { "intValue": 32767 },
		"4b_opcode": { "intValue": 126 },
		"4c_base": { "intValue": 512 },
		"4e_checksum": { "intValue": 301458068 },
		"4f_mode": { "intValue": 420 },
		"5a_min": { "intValue": 1 },
		"5b_mix": { "intValue": 44 },
		"5c_port": { "intValue": 55678 },
		"5d_min_hops": { "intValue": 512 },
		"5e_max_hops": { "intValue": 16777215 }
	}
}
}}]` + suffix,
		},
	}
	ctx := context.TODO()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, done := event.Start(ctx, "event span")
			tt.run(ctx)
			done()
			got := exporter.Output("/v1/trace")
			checkJSON(t, got, []byte(tt.want))
		})
	}
}
