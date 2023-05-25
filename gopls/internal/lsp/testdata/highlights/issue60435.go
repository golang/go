package highlights

import (
	"net/http"          //@mark(httpImp, `"net/http"`)
	"net/http/httptest" //@mark(httptestImp, `"net/http/httptest"`)
)

// This is a regression test for issue 60435:
// Highlighting "net/http" shouldn't have any effect
// on an import path that contains it as a substring,
// such as httptest.

var _ = httptest.NewRequest
var _ = http.NewRequest //@mark(here, "http"), highlight(here, here, httpImp)
