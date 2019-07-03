package httpresponse

import (
	"log"
	"net/http"
)

func goodHTTPGet() {
	res, err := http.Get("http://foo.com")
	if err != nil {
		log.Fatal(err)
	}
	defer res.Body.Close()
}

func badHTTPGet() {
	res, err := http.Get("http://foo.com")
	defer res.Body.Close() // ERROR "using res before checking for errors"
	if err != nil {
		log.Fatal(err)
	}
}
