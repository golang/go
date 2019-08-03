// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ocagent

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"golang.org/x/tools/internal/lsp/telemetry/log"
	"golang.org/x/tools/internal/lsp/telemetry/ocagent/wire"
	"golang.org/x/tools/internal/lsp/telemetry/tag"
)

func TestConvert_annotation(t *testing.T) {
	tests := []struct {
		name    string
		tagList tag.List
		want    *wire.Annotation
	}{
		{
			name:    "no tags",
			tagList: nil,
			want:    nil,
		},
		{
			name: "description no error",
			tagList: tag.List{
				tag.Of(log.MessageTag, "cache miss"),
				tag.Of("db", "godb"),
			},
			want: &wire.Annotation{
				Description: &wire.TruncatableString{Value: "cache miss"},
				Attributes: &wire.Attributes{
					AttributeMap: map[string]wire.Attribute{
						"db": wire.StringAttribute{StringValue: &wire.TruncatableString{Value: "godb"}},
					},
				},
			},
		},

		{
			name: "description and error",
			tagList: tag.List{
				tag.Of(log.MessageTag, "cache miss"),
				tag.Of("db", "godb"),
				tag.Of(log.ErrorTag, errors.New("no network connectivity")),
			},
			want: &wire.Annotation{
				Description: &wire.TruncatableString{Value: "cache miss"},
				Attributes: &wire.Attributes{
					AttributeMap: map[string]wire.Attribute{
						"Error": wire.StringAttribute{StringValue: &wire.TruncatableString{Value: "no network connectivity"}},
						"db":    wire.StringAttribute{StringValue: &wire.TruncatableString{Value: "godb"}},
					},
				},
			},
		},
		{
			name: "no description, but error",
			tagList: tag.List{
				tag.Of("db", "godb"),
				tag.Of(log.ErrorTag, errors.New("no network connectivity")),
			},
			want: &wire.Annotation{
				Description: &wire.TruncatableString{Value: "no network connectivity"},
				Attributes: &wire.Attributes{
					AttributeMap: map[string]wire.Attribute{
						"db": wire.StringAttribute{StringValue: &wire.TruncatableString{Value: "godb"}},
					},
				},
			},
		},
		{
			name: "enumerate all attribute types",
			tagList: tag.List{
				tag.Of(log.MessageTag, "cache miss"),
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
			want: &wire.Annotation{
				Description: &wire.TruncatableString{Value: "cache miss"},
				Attributes: &wire.Attributes{
					AttributeMap: map[string]wire.Attribute{
						"db": wire.StringAttribute{StringValue: &wire.TruncatableString{Value: "godb"}},

						"age":       wire.DoubleAttribute{DoubleValue: 0.456},
						"ttl":       wire.DoubleAttribute{DoubleValue: 5000.0},
						"expiry_ms": wire.DoubleAttribute{DoubleValue: 1e3},

						"retry": wire.BoolAttribute{BoolValue: false},
						"stale": wire.BoolAttribute{BoolValue: true},

						"max":      wire.IntAttribute{IntValue: 0x7fff},
						"opcode":   wire.IntAttribute{IntValue: 0x7e},
						"base":     wire.IntAttribute{IntValue: 1 << 9},
						"checksum": wire.IntAttribute{IntValue: 0x11f7e294},
						"mode":     wire.IntAttribute{IntValue: 0644},

						"min":      wire.IntAttribute{IntValue: 1},
						"mix":      wire.IntAttribute{IntValue: 44},
						"port":     wire.IntAttribute{IntValue: 55678},
						"min_hops": wire.IntAttribute{IntValue: 1 << 9},
						"max_hops": wire.IntAttribute{IntValue: 0xffffff},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := convertAnnotation(tt.tagList)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("Got:\n%s\nWant:\n%s", marshaled(got), marshaled(tt.want))
			}
		})
	}
}

func marshaled(v interface{}) string {
	blob, _ := json.MarshalIndent(v, "", "  ")
	return string(blob)
}
