package httptest_test

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func TestHttpFileServerSkipPermanentRedirect(t *testing.T) {
	URL, _ := url.Parse("/index.html")
	fs := http.Dir("./")
	respRecorder := httptest.NewRecorder()
	req := &http.Request{
		Method:     "GET",
		Proto:      "HTTP/1.1",
		ProtoMajor: 1,
		ProtoMinor: 1,
		URL:        URL,
	}

	fileServer := http.FileServer(fs)

	// skip 301 redirect
	fileServer.(*http.FileHandler).SkipPermanentRedirect()

	fileServer.ServeHTTP(respRecorder, req)
	if respRecorder.Code != 200 {
		t.Fatal("Expect code 200, result code: ", respRecorder.Code, "body response:", respRecorder.Body.String())
	}

	bodyString := respRecorder.Body.String()
	if !strings.Contains(bodyString, "<body>") {
		t.Fatalf("Unexpected body response: %s", respRecorder.Body.String())
	}

}

func TestHttpFileServerPermanentRedirect(t *testing.T) {
	URL, _ := url.Parse("/index.html")
	fs := http.Dir("./")
	respRecorder := httptest.NewRecorder()
	req := &http.Request{
		Method:     "GET",
		Proto:      "HTTP/1.1",
		ProtoMajor: 1,
		ProtoMinor: 1,
		URL:        URL,
	}

	fileServer := http.FileServer(fs)

	fileServer.ServeHTTP(respRecorder, req)
	if respRecorder.Code != 301 {
		t.Fatal("Expect code 301, result code: ", respRecorder.Code, "body:", respRecorder.Body.String())
	}
}
