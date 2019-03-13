// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt_test

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

func TestErrorf(t *testing.T) {
	chained := &wrapped{"chained", nil}
	chain := func(s ...string) (a []string) {
		for _, s := range s {
			a = append(a, cleanPath(s))
		}
		return a
	}
	noArgsWrap := "no args: %w" // avoid vet check
	testCases := []struct {
		got  error
		want []string
	}{{
		fmt.Errorf("no args"),
		chain("no args/path.TestErrorf/path.go:xxx"),
	}, {
		fmt.Errorf(noArgsWrap),
		chain("no args: %!w(MISSING)/path.TestErrorf/path.go:xxx"),
	}, {
		fmt.Errorf("nounwrap: %s", "simple"),
		chain(`nounwrap: simple/path.TestErrorf/path.go:xxx`),
	}, {
		fmt.Errorf("nounwrap: %v", "simple"),
		chain(`nounwrap: simple/path.TestErrorf/path.go:xxx`),
	}, {
		fmt.Errorf("%s failed: %v", "foo", chained),
		chain("foo failed/path.TestErrorf/path.go:xxx",
			"chained/somefile.go:xxx"),
	}, {
		fmt.Errorf("no wrap: %s", chained),
		chain("no wrap/path.TestErrorf/path.go:xxx",
			"chained/somefile.go:xxx"),
	}, {
		fmt.Errorf("%s failed: %w", "foo", chained),
		chain("wraps:foo failed/path.TestErrorf/path.go:xxx",
			"chained/somefile.go:xxx"),
	}, {
		fmt.Errorf("nowrapv: %v", chained),
		chain("nowrapv/path.TestErrorf/path.go:xxx",
			"chained/somefile.go:xxx"),
	}, {
		fmt.Errorf("wrapw: %w", chained),
		chain("wraps:wrapw/path.TestErrorf/path.go:xxx",
			"chained/somefile.go:xxx"),
	}, {
		fmt.Errorf("not wrapped: %+v", chained),
		chain("not wrapped: chained: somefile.go:123/path.TestErrorf/path.go:xxx"),
	}}
	for i, tc := range testCases {
		t.Run(strconv.Itoa(i)+"/"+path.Join(tc.want...), func(t *testing.T) {
			got := errToParts(tc.got)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("Format:\n got: %+q\nwant: %+q", got, tc.want)
			}

			gotStr := tc.got.Error()
			wantStr := fmt.Sprint(tc.got)
			if gotStr != wantStr {
				t.Errorf("Error:\n got: %+q\nwant: %+q", gotStr, wantStr)
			}
		})
	}
}

func TestErrorFormatter(t *testing.T) {
	testCases := []struct {
		err    error
		fmt    string
		want   string
		regexp bool
	}{{
		err: errors.New("foo"),
		fmt: "%+v",
		want: "foo:" +
			"\n    fmt_test.TestErrorFormatter" +
			"\n        .+/fmt/errors_test.go:\\d\\d",
		regexp: true,
	}, {
		err:  &wrapped{"simple", nil},
		fmt:  "%s",
		want: "simple",
	}, {
		err:  &wrapped{"can't adumbrate elephant", outOfPeanuts{}},
		fmt:  "%s",
		want: "can't adumbrate elephant: out of peanuts",
	}, {
		err:  &wrapped{"a", &wrapped{"b", &wrapped{"c", nil}}},
		fmt:  "%s",
		want: "a: b: c",
	}, {
		err: &wrapped{"simple", nil},
		fmt: "%+v",
		want: "simple:" +
			"\n    somefile.go:123",
	}, {
		err: &wrapped{"can't adumbrate elephant", outOfPeanuts{}},
		fmt: "%+v",
		want: "can't adumbrate elephant:" +
			"\n    somefile.go:123" +
			"\n  - out of peanuts:" +
			"\n    the elephant is on strike" +
			"\n    and the 12 monkeys" +
			"\n    are laughing",
	}, {
		err:  &wrapped{"simple", nil},
		fmt:  "%#v",
		want: "&fmt_test.wrapped{msg:\"simple\", err:error(nil)}",
	}, {
		err:  &notAFormatterError{},
		fmt:  "%+v",
		want: "not a formatter",
	}, {
		err: &wrapped{"wrap", &notAFormatterError{}},
		fmt: "%+v",
		want: "wrap:" +
			"\n    somefile.go:123" +
			"\n  - not a formatter",
	}, {
		err: &withFrameAndMore{frame: errors.Caller(0)},
		fmt: "%+v",
		want: "something:" +
			"\n    fmt_test.TestErrorFormatter" +
			"\n        .+/fmt/errors_test.go:\\d\\d\\d" +
			"\n    something more",
		regexp: true,
	}, {
		err:  fmtTwice("Hello World!"),
		fmt:  "%#v",
		want: "2 times Hello World!",
	}, {
		err:  &wrapped{"fallback", os.ErrNotExist},
		fmt:  "%s",
		want: "fallback: file does not exist",
	}, {
		err: &wrapped{"fallback", os.ErrNotExist},
		fmt: "%+v",
		// Note: no colon after the last error, as there are no details.
		want: "fallback:" +
			"\n    somefile.go:123" +
			"\n  - file does not exist:" +
			"\n    os.init.ializers" +
			"\n        .+/os/error.go:\\d\\d",
		regexp: true,
	}, {
		err: &wrapped{"outer",
			errors.Opaque(&wrapped{"mid",
				&wrapped{"inner", nil}})},
		fmt:  "%s",
		want: "outer: mid: inner",
	}, {
		err: &wrapped{"outer",
			errors.Opaque(&wrapped{"mid",
				&wrapped{"inner", nil}})},
		fmt: "%+v",
		want: "outer:" +
			"\n    somefile.go:123" +
			"\n  - mid:" +
			"\n    somefile.go:123" +
			"\n  - inner:" +
			"\n    somefile.go:123",
	}, {
		err:  &wrapped{"new style", formatError("old style")},
		fmt:  "%v",
		want: "new style: old style",
	}, {
		err:  &wrapped{"new style", formatError("old style")},
		fmt:  "%q",
		want: `"new style: old style"`,
	}, {
		err: &wrapped{"new style", formatError("old style")},
		fmt: "%+v",
		// Note the extra indentation.
		// Colon for old style error is rendered by the fmt.Formatter
		// implementation of the old-style error.
		want: "new style:" +
			"\n    somefile.go:123" +
			"\n  - old style:" +
			"\n    otherfile.go:456",
	}, {
		err:  &wrapped{"simple", nil},
		fmt:  "%-12s",
		want: "simple      ",
	}, {
		// Don't use formatting flags for detailed view.
		err: &wrapped{"simple", nil},
		fmt: "%+12v",
		want: "simple:" +
			"\n    somefile.go:123",
	}, {
		err:  &wrapped{"can't adumbrate elephant", outOfPeanuts{}},
		fmt:  "%+50s",
		want: "          can't adumbrate elephant: out of peanuts",
	}, {
		err:  &wrapped{"café", nil},
		fmt:  "%q",
		want: `"café"`,
	}, {
		err:  &wrapped{"café", nil},
		fmt:  "%+q",
		want: `"caf\u00e9"`,
	}, {
		err:  &wrapped{"simple", nil},
		fmt:  "% x",
		want: "73 69 6d 70 6c 65",
	}, {
		err: &wrapped{"msg with\nnewline",
			&wrapped{"and another\none", nil}},
		fmt: "%s",
		want: "msg with" +
			"\nnewline: and another" +
			"\none",
	}, {
		err: &wrapped{"msg with\nnewline",
			&wrapped{"and another\none", nil}},
		fmt: "%+v",
		want: "msg with" +
			"\n    newline:" +
			"\n    somefile.go:123" +
			"\n  - and another" +
			"\n    one:" +
			"\n    somefile.go:123",
	}, {
		err: wrapped{"", wrapped{"inner message", nil}},
		fmt: "%+v",
		want: "somefile.go:123" +
			"\n  - inner message:" +
			"\n    somefile.go:123",
	}, {
		err:  detail{"empty detail", "", nil},
		fmt:  "%s",
		want: "empty detail",
	}, {
		err:  detail{"empty detail", "", nil},
		fmt:  "%+v",
		want: "empty detail",
	}, {
		err:  detail{"newline at start", "\nextra", nil},
		fmt:  "%s",
		want: "newline at start",
	}, {
		err: detail{"newline at start", "\n extra", nil},
		fmt: "%+v",
		want: "newline at start:" +
			"\n     extra",
	}, {
		err: detail{"newline at start", "\nextra",
			detail{"newline at start", "\nmore", nil}},
		fmt: "%+v",
		want: "newline at start:" +
			"\n    extra" +
			"\n  - newline at start:" +
			"\n    more",
	}, {
		err: detail{"two newlines at start", "\n\nextra",
			detail{"two newlines at start", "\n\nmore", nil}},
		fmt: "%+v",
		want: "two newlines at start:" +
			"\n    " + // note the explicit space
			"\n    extra" +
			"\n  - two newlines at start:" +
			"\n    " +
			"\n    more",
	}, {
		err:  &detail{"single newline", "\n", nil},
		fmt:  "%+v",
		want: "single newline",
	}, {
		err: &detail{"single newline", "\n",
			&detail{"single newline", "\n", nil}},
		fmt: "%+v",
		want: "single newline:" +
			"\n  - single newline",
	}, {
		err: &detail{"newline at end", "detail\n", nil},
		fmt: "%+v",
		want: "newline at end:" +
			"\n    detail",
	}, {
		err: &detail{"newline at end", "detail\n",
			&detail{"newline at end", "detail\n", nil}},
		fmt: "%+v",
		want: "newline at end:" +
			"\n    detail" +
			"\n  - newline at end:" +
			"\n    detail",
	}, {
		err: &detail{"two newlines at end", "detail\n\n",
			&detail{"two newlines at end", "detail\n\n", nil}},
		fmt: "%+v",
		want: "two newlines at end:" +
			"\n    detail" +
			"\n    " +
			"\n  - two newlines at end:" +
			"\n    detail" +
			"\n    ", // note the additional space
	}, {
		err:  nil,
		fmt:  "%+v",
		want: "<nil>",
	}, {
		err:  (*wrapped)(nil),
		fmt:  "%+v",
		want: "<nil>",
	}, {
		err:  &wrapped{"simple", nil},
		fmt:  "%T",
		want: "*fmt_test.wrapped",
	}, {
		err:  &wrapped{"simple", nil},
		fmt:  "%🤪",
		want: "&{%!🤪(string=simple) <nil>}",
	}, {
		err:  formatError("use fmt.Formatter"),
		fmt:  "%#v",
		want: "use fmt.Formatter",
	}, {
		err: wrapped{"using errors.Formatter",
			formatError("use fmt.Formatter")},
		fmt:  "%#v",
		want: "fmt_test.wrapped{msg:\"using errors.Formatter\", err:\"use fmt.Formatter\"}",
	}, {
		err:  fmtTwice("%s %s", "ok", panicValue{}),
		fmt:  "%s",
		want: "ok %!s(PANIC=String method: panic)/ok %!s(PANIC=String method: panic)",
	}, {
		err:  fmtTwice("%o %s", panicValue{}, "ok"),
		fmt:  "%s",
		want: "{} ok/{} ok",
	}, {
		err:  intError(4),
		fmt:  "%v",
		want: "error 4",
	}, {
		err:  intError(4),
		fmt:  "%d",
		want: "4",
	}, {
		err:  intError(4),
		fmt:  "%🤪",
		want: "%!🤪(fmt_test.intError=4)",
	}}
	for i, tc := range testCases {
		t.Run(fmt.Sprintf("%d/%s", i, tc.fmt), func(t *testing.T) {
			got := fmt.Sprintf(tc.fmt, tc.err)
			var ok bool
			if tc.regexp {
				var err error
				ok, err = regexp.MatchString(tc.want+"$", got)
				if err != nil {
					t.Fatal(err)
				}
			} else {
				ok = got == tc.want
			}
			if !ok {
				t.Errorf("\n got: %q\nwant: %q", got, tc.want)
			}
		})
	}
}

func TestSameType(t *testing.T) {
	err0 := errors.New("inner")
	want := fmt.Sprintf("%T", err0)

	err := fmt.Errorf("foo: %v", err0)
	if got := fmt.Sprintf("%T", err); got != want {
		t.Errorf("got %v; want %v", got, want)
	}

	err = fmt.Errorf("foo %s", "bar")
	if got := fmt.Sprintf("%T", err); got != want {
		t.Errorf("got %v; want %v", got, want)
	}
}

var _ errors.Formatter = wrapped{}

type wrapped struct {
	msg string
	err error
}

func (e wrapped) Error() string { return fmt.Sprint(e) }

func (e wrapped) FormatError(p errors.Printer) (next error) {
	p.Print(e.msg)
	p.Detail()
	p.Print("somefile.go:123")
	return e.err
}

var _ errors.Formatter = outOfPeanuts{}

type outOfPeanuts struct{}

func (e outOfPeanuts) Error() string { return fmt.Sprint(e) }

func (e outOfPeanuts) Format(fmt.State, rune) {
	panic("should never be called by one of the tests")
}

func (outOfPeanuts) FormatError(p errors.Printer) (next error) {
	p.Printf("out of %s", "peanuts")
	p.Detail()
	p.Print("the elephant is on strike\n")
	p.Printf("and the %d monkeys\nare laughing", 12)
	return nil
}

type withFrameAndMore struct {
	frame errors.Frame
}

func (e *withFrameAndMore) Error() string { return fmt.Sprint(e) }

func (e *withFrameAndMore) FormatError(p errors.Printer) (next error) {
	p.Print("something")
	if p.Detail() {
		e.frame.Format(p)
		p.Print("something more")
	}
	return nil
}

type notAFormatterError struct{}

func (e notAFormatterError) Error() string { return "not a formatter" }

type detail struct {
	msg    string
	detail string
	next   error
}

func (e detail) Error() string { return fmt.Sprint(e) }

func (e detail) FormatError(p errors.Printer) (next error) {
	p.Print(e.msg)
	p.Detail()
	p.Print(e.detail)
	return e.next
}

type intError int

func (e intError) Error() string { return fmt.Sprint(e) }

func (e wrapped) Format(w fmt.State, r rune) {
	// Test that the normal fallback handling after handleMethod for
	// non-string verbs is used. This path should not be reached.
	fmt.Fprintf(w, "Unreachable: %d", e)
}

func (e intError) FormatError(p errors.Printer) (next error) {
	p.Printf("error %d", e)
	return nil
}

// formatError is an error implementing Format instead of errors.Formatter.
// The implementation mimics the implementation of github.com/pkg/errors.
type formatError string

func (e formatError) Error() string { return string(e) }

func (e formatError) Format(s fmt.State, verb rune) {
	// Body based on pkg/errors/errors.go
	switch verb {
	case 'v':
		if s.Flag('+') {
			io.WriteString(s, string(e))
			fmt.Fprintf(s, ":\n%s", "otherfile.go:456")
			return
		}
		fallthrough
	case 's':
		io.WriteString(s, string(e))
	case 'q':
		fmt.Fprintf(s, "%q", string(e))
	}
}

func (e formatError) GoString() string {
	panic("should never be called")
}

type fmtTwiceErr struct {
	format string
	args   []interface{}
}

func fmtTwice(format string, a ...interface{}) error {
	return fmtTwiceErr{format, a}
}

func (e fmtTwiceErr) Error() string { return fmt.Sprint(e) }

func (e fmtTwiceErr) FormatError(p errors.Printer) (next error) {
	p.Printf(e.format, e.args...)
	p.Print("/")
	p.Printf(e.format, e.args...)
	return nil
}

func (e fmtTwiceErr) GoString() string {
	return "2 times " + fmt.Sprintf(e.format, e.args...)
}

type panicValue struct{}

func (panicValue) String() string { panic("panic") }

var rePath = regexp.MustCompile(`( [^ ]*)fmt.*test\.`)
var reLine = regexp.MustCompile(":[0-9]*\n?$")

func cleanPath(s string) string {
	s = rePath.ReplaceAllString(s, "/path.")
	s = reLine.ReplaceAllString(s, ":xxx")
	s = strings.Replace(s, "\n   ", "", -1)
	s = strings.Replace(s, " /", "/", -1)
	return s
}

func errToParts(err error) (a []string) {
	for err != nil {
		var p testPrinter
		if errors.Unwrap(err) != nil {
			p.str += "wraps:"
		}
		f, ok := err.(errors.Formatter)
		if !ok {
			a = append(a, err.Error())
			break
		}
		err = f.FormatError(&p)
		a = append(a, cleanPath(p.str))
	}
	return a

}

type testPrinter struct {
	str string
}

func (p *testPrinter) Print(a ...interface{}) {
	p.str += fmt.Sprint(a...)
}

func (p *testPrinter) Printf(format string, a ...interface{}) {
	p.str += fmt.Sprintf(format, a...)
}

func (p *testPrinter) Detail() bool {
	p.str += " /"
	return true
}
