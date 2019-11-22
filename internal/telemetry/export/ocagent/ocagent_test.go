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

	"golang.org/x/tools/internal/telemetry"
	"golang.org/x/tools/internal/telemetry/export/ocagent"
	"golang.org/x/tools/internal/telemetry/tag"
)

func TestConvert_annotation(t *testing.T) {
	tests := []struct {
		name  string
		event func(ctx context.Context) telemetry.Event
		want  string
	}{
		{
			name:  "no tags",
			event: func(ctx context.Context) telemetry.Event { return telemetry.Event{} },
			want:  "null",
		},
		{
			name: "description no error",
			event: func(ctx context.Context) telemetry.Event {
				return telemetry.Event{
					Message: "cache miss",
					Tags: telemetry.TagList{
						tag.Of("db", "godb"),
					},
				}
			},
			want: `{
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
}`,
		},

		{
			name: "description and error",
			event: func(ctx context.Context) telemetry.Event {
				return telemetry.Event{
					Message: "cache miss",
					Error:   errors.New("no network connectivity"),
					Tags: telemetry.TagList{
						tag.Of("db", "godb"),
					},
				}
			},
			want: `{
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
}`,
		},
		{
			name: "no description, but error",
			event: func(ctx context.Context) telemetry.Event {
				return telemetry.Event{
					Error: errors.New("no network connectivity"),
					Tags: telemetry.TagList{
						tag.Of("db", "godb"),
					},
				}
			},
			want: `{
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
}`,
		},
		{
			name: "enumerate all attribute types",
			event: func(ctx context.Context) telemetry.Event {
				return telemetry.Event{
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
			want: `{
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
}`,
		},
	}
	ctx := context.TODO()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ocagent.EncodeAnnotation(tt.event(ctx))
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
		t.Fatalf("Got:\n%s\nWant:\n%s", got, want)
	}
}
