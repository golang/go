package godoc

import (
	"errors"
	"expvar"
	"net/http"
	"net/http/httptest"
	"testing"
	"text/template"
)

var (
	// NOTE: with no plain-text in the template, template.Execute will not
	// return an error when http.ResponseWriter.Write does return an error.
	tmpl = template.Must(template.New("test").Parse("{{.Foo}}"))
)

type withFoo struct {
	Foo int
}

type withoutFoo struct {
}

type errResponseWriter struct {
}

func (*errResponseWriter) Header() http.Header {
	return http.Header{}
}

func (*errResponseWriter) WriteHeader(int) {
}

func (*errResponseWriter) Write(p []byte) (int, error) {
	return 0, errors.New("error")
}

func TestApplyTemplateToResponseWriter(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		rw      http.ResponseWriter
		data    interface{}
		expVars int
	}{
		{
			desc:    "no error",
			rw:      &httptest.ResponseRecorder{},
			data:    &withFoo{},
			expVars: 0,
		},
		{
			desc:    "template error",
			rw:      &httptest.ResponseRecorder{},
			data:    &withoutFoo{},
			expVars: 0,
		},
		{
			desc:    "ResponseWriter error",
			rw:      &errResponseWriter{},
			data:    &withFoo{},
			expVars: 1,
		},
	} {
		httpErrors.Init()
		applyTemplateToResponseWriter(tc.rw, tmpl, tc.data)
		gotVars := 0
		httpErrors.Do(func(expvar.KeyValue) {
			gotVars++
		})
		if gotVars != tc.expVars {
			t.Errorf("applyTemplateToResponseWriter(%q): got %d vars, want %d", tc.desc, gotVars, tc.expVars)
		}
	}
}
