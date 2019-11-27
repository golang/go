// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ocagent_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"golang.org/x/tools/internal/telemetry"
	"golang.org/x/tools/internal/telemetry/export/ocagent"
	"golang.org/x/tools/internal/telemetry/tag"
)

var (
	cfg = ocagent.Config{
		Host:    "tester",
		Process: 1,
		Service: "ocagent-tests",
	}
	start time.Time
	at    time.Time
	end   time.Time
)

func init() {
	cfg.Start, _ = time.Parse(time.RFC3339Nano, "1970-01-01T00:00:00Z")
}

const testNodeStr = `{
	"node":{
		"identifier":{
			"host_name":"tester",
			"pid":1,
			"start_timestamp":"1970-01-01T00:00:00Z"
		},
		"library_info":{
			"language":4,
			"exporter_version":"0.0.1",
			"core_library_version":"x/tools"
		},
		"service_info":{
			"name":"ocagent-tests"
		}
	},`

func TestEvents(t *testing.T) {
	start, _ := time.Parse(time.RFC3339Nano, "1970-01-01T00:00:30Z")
	at, _ := time.Parse(time.RFC3339Nano, "1970-01-01T00:00:40Z")
	end, _ := time.Parse(time.RFC3339Nano, "1970-01-01T00:00:50Z")
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
		name  string
		event func(ctx context.Context) telemetry.Event
		want  string
	}{
		{
			name: "no tags",
			event: func(ctx context.Context) telemetry.Event {
				return telemetry.Event{
					At: at,
				}
			},
			want: prefix + `
						"timeEvent":[{"time":"1970-01-01T00:00:40Z"}]
			` + suffix,
		},
		{
			name: "description no error",
			event: func(ctx context.Context) telemetry.Event {
				return telemetry.Event{
					At:      at,
					Message: "cache miss",
					Tags: telemetry.TagList{
						tag.Of("db", "godb"),
					},
				}
			},
			want: prefix + `"timeEvent":[{"time":"1970-01-01T00:00:40Z","annotation":{
  "description": {
    "value": "cache miss"
  },
  "attributes": {
    "attributeMap": {
      "db": {
        "stringValue": {
          "value": "godb"
        }
      }
    }
  }
}}]` + suffix,
		},

		{
			name: "description and error",
			event: func(ctx context.Context) telemetry.Event {
				return telemetry.Event{
					At:      at,
					Message: "cache miss",
					Error:   errors.New("no network connectivity"),
					Tags: telemetry.TagList{
						tag.Of("db", "godb"),
					},
				}
			},
			want: prefix + `"timeEvent":[{"time":"1970-01-01T00:00:40Z","annotation":{
  "description": {
    "value": "cache miss"
  },
  "attributes": {
    "attributeMap": {
      "Error": {
        "stringValue": {
          "value": "no network connectivity"
        }
      },
      "db": {
        "stringValue": {
          "value": "godb"
        }
      }
    }
  }
	}}]` + suffix,
		},
		{
			name: "no description, but error",
			event: func(ctx context.Context) telemetry.Event {
				return telemetry.Event{
					At:    at,
					Error: errors.New("no network connectivity"),
					Tags: telemetry.TagList{
						tag.Of("db", "godb"),
					},
				}
			},
			want: prefix + `"timeEvent":[{"time":"1970-01-01T00:00:40Z","annotation":{
  "description": {
    "value": "no network connectivity"
  },
  "attributes": {
    "attributeMap": {
      "db": {
        "stringValue": {
          "value": "godb"
        }
      }
    }
  }
	}}]` + suffix,
		},
		{
			name: "enumerate all attribute types",
			event: func(ctx context.Context) telemetry.Event {
				return telemetry.Event{
					At:      at,
					Message: "cache miss",
					Tags: telemetry.TagList{
						tag.Of("db", "godb"),

						tag.Of("age", 0.456), // Constant converted into "float64"
						tag.Of("ttl", float32(5000)),
						tag.Of("expiry_ms", float64(1e3)),

						tag.Of("retry", false),
						tag.Of("stale", true),

						tag.Of("max", 0x7fff), // Constant converted into "int"
						tag.Of("opcode", int8(0x7e)),
						tag.Of("base", int16(1<<9)),
						tag.Of("checksum", int32(0x11f7e294)),
						tag.Of("mode", int64(0644)),

						tag.Of("min", uint(1)),
						tag.Of("mix", uint8(44)),
						tag.Of("port", uint16(55678)),
						tag.Of("min_hops", uint32(1<<9)),
						tag.Of("max_hops", uint64(0xffffff)),
					},
				}
			},
			want: prefix + `"timeEvent":[{"time":"1970-01-01T00:00:40Z","annotation":{
  "description": {
    "value": "cache miss"
  },
  "attributes": {
    "attributeMap": {
      "age": {
        "doubleValue": 0.456
      },
      "base": {
        "intValue": 512
      },
      "checksum": {
        "intValue": 301458068
      },
      "db": {
        "stringValue": {
          "value": "godb"
        }
      },
      "expiry_ms": {
        "doubleValue": 1000
      },
      "max": {
        "intValue": 32767
      },
      "max_hops": {
        "intValue": 16777215
      },
      "min": {
        "intValue": 1
      },
      "min_hops": {
        "intValue": 512
      },
      "mix": {
        "intValue": 44
      },
      "mode": {
        "intValue": 420
      },
      "opcode": {
        "intValue": 126
      },
      "port": {
        "intValue": 55678
      },
      "retry": {},
      "stale": {
        "boolValue": true
      },
      "ttl": {
        "doubleValue": 5000
      }
    }
  }
}}]` + suffix,
		},
	}
	ctx := context.TODO()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			span := &telemetry.Span{
				Name:   "event span",
				Start:  start,
				Finish: end,
				Events: []telemetry.Event{tt.event(ctx)},
			}
			got, err := ocagent.EncodeSpan(cfg, span)
			if err != nil {
				t.Fatal(err)
			}
			checkJSON(t, got, []byte(tt.want))
		})
	}

}

func checkJSON(t *testing.T, got, want []byte) {
	// compare the compact form, to allow for formatting differences
	g := &bytes.Buffer{}
	if err := json.Compact(g, got); err != nil {
		t.Fatal(err)
	}
	w := &bytes.Buffer{}
	if err := json.Compact(w, want); err != nil {
		t.Fatal(err)
	}
	if g.String() != w.String() {
		t.Fatalf("Got:\n%s\nWant:\n%s", g, w)
	}
}
