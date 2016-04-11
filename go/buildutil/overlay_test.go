package buildutil_test

import (
	"go/build"
	"io/ioutil"
	"reflect"
	"strings"
	"testing"

	"golang.org/x/tools/go/buildutil"
)

func TestParseOverlayArchive(t *testing.T) {
	var tt = []struct {
		in     string
		out    map[string][]byte
		hasErr bool
	}{
		{
			"a.go\n5\n12345",
			map[string][]byte{"a.go": []byte("12345")},
			false,
		},
		{
			"a.go\n5\n1234",
			nil,
			true,
		},
		{
			"a.go\n5\n12345b.go\n4\n1234",
			map[string][]byte{"a.go": []byte("12345"), "b.go": []byte("1234")},
			false,
		},
	}

	for _, test := range tt {
		got, err := buildutil.ParseOverlayArchive(strings.NewReader(test.in))
		if err == nil && test.hasErr {
			t.Errorf("expected error for %q", test.in)
		}
		if err != nil && !test.hasErr {
			t.Errorf("unexpected error %v for %q", err, test.in)
		}
		if !reflect.DeepEqual(got, test.out) {
			t.Errorf("got %#v, want %#v", got, test.out)
		}
	}
}

func TestOverlay(t *testing.T) {
	ctx := &build.Default
	ov := map[string][]byte{
		"/somewhere/a.go": []byte("file contents"),
	}
	names := []string{"/somewhere/a.go", "/somewhere//a.go"}
	ctx = buildutil.OverlayContext(ctx, ov)
	for _, name := range names {
		f, err := buildutil.OpenFile(ctx, name)
		if err != nil {
			t.Errorf("unexpected error %v", err)
		}
		b, err := ioutil.ReadAll(f)
		if err != nil {
			t.Errorf("unexpected error %v", err)
		}
		if got, expected := string(b), string(ov["/somewhere/a.go"]); got != expected {
			t.Errorf("read %q, expected %q", got, expected)
		}
	}
}
