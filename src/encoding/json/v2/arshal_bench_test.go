package json

import (
	"reflect"
	"testing"

	"encoding/json/jsontext"
)

type benchmarkMarshalerTo struct{}

func (benchmarkMarshalerTo) MarshalJSONTo(enc *jsontext.Encoder) error {
	return enc.WriteValue(jsontext.Value("null"))
}

func BenchmarkMakeMethodMarshaler(b *testing.B) {
	t := reflect.TypeFor[benchmarkMarshalerTo]()
	base := makeDefaultMarshaler(t)

	b.ReportAllocs()
	for b.Loop() {
		_ = makeMethodMarshaler(base, t)
	}
}

type benchmarkUnmarshalerFrom struct{}

func (*benchmarkUnmarshalerFrom) UnmarshalJSONFrom(*jsontext.Decoder) error {
	return nil
}

func BenchmarkMakeMethodUnmarshaler(b *testing.B) {
	t := reflect.TypeFor[benchmarkUnmarshalerFrom]()
	base := makeDefaultUnmarshaler(t)

	b.ReportAllocs()
	for b.Loop() {
		_ = makeMethodUnmarshaler(base, t)
	}
}

type benchmarkMarshalTo int

func BenchmarkMarshalToFunc(b *testing.B) {
	fn := func(enc *jsontext.Encoder, _ benchmarkMarshalTo) error {
		return enc.WriteValue(jsontext.Value("null"))
	}

	b.ReportAllocs()
	for b.Loop() {
		_ = MarshalToFunc(fn)
	}
}

type benchmarkUnmarshalFrom int

func BenchmarkUnmarshalFromFunc(b *testing.B) {
	fn := func(dec *jsontext.Decoder, _ *benchmarkUnmarshalFrom) error {
		_, err := dec.ReadValue()
		return err
	}

	b.ReportAllocs()
	for b.Loop() {
		_ = UnmarshalFromFunc(fn)
	}
}
