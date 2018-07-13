package driver

import "testing"

func TestGetNodeShortName(t *testing.T) {
	type testCase struct {
		name string
		want string
	}
	testcases := []testCase{
		{
			"root",
			"root",
		},
		{
			"syscall.Syscall",
			"syscall.Syscall",
		},
		{
			"net/http.(*conn).serve",
			"net/http.(*conn).serve",
		},
		{
			"github.com/blah/foo.Foo",
			"foo.Foo",
		},
		{
			"github.com/blah/foo_bar.(*FooBar).Foo",
			"foo_bar.(*FooBar).Foo",
		},
		{
			"encoding/json.(*structEncoder).(encoding/json.encode)-fm",
			"encoding/json.(*structEncoder).(encoding/json.encode)-fm",
		},
		{
			"github.com/blah/blah/vendor/gopkg.in/redis.v3.(*baseClient).(github.com/blah/blah/vendor/gopkg.in/redis.v3.process)-fm",
			"redis.v3.(*baseClient).(github.com/blah/blah/vendor/gopkg.in/redis.v3.process)-fm",
		},
	}
	for _, tc := range testcases {
		name := getNodeShortName(tc.name)
		if got, want := name, tc.want; got != want {
			t.Errorf("for %s, got %q, want %q", tc.name, got, want)
		}
	}
}
