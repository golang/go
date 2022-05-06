package a

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
	defer res.Body.Close() // want "using res before checking for errors"
	if err != nil {
		log.Fatal(err)
	}
}

func badHTTPHead() {
	res, err := http.Head("http://foo.com")
	defer res.Body.Close() // want "using res before checking for errors"
	if err != nil {
		log.Fatal(err)
	}
}

func goodClientGet() {
	client := http.DefaultClient
	res, err := client.Get("http://foo.com")
	if err != nil {
		log.Fatal(err)
	}
	defer res.Body.Close()
}

func badClientPtrGet() {
	client := http.DefaultClient
	resp, err := client.Get("http://foo.com")
	defer resp.Body.Close() // want "using resp before checking for errors"
	if err != nil {
		log.Fatal(err)
	}
}

func badClientGet() {
	client := http.Client{}
	resp, err := client.Get("http://foo.com")
	defer resp.Body.Close() // want "using resp before checking for errors"
	if err != nil {
		log.Fatal(err)
	}
}

func badClientPtrDo() {
	client := http.DefaultClient
	req, err := http.NewRequest("GET", "http://foo.com", nil)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := client.Do(req)
	defer resp.Body.Close() // want "using resp before checking for errors"
	if err != nil {
		log.Fatal(err)
	}
}

func badClientDo() {
	var client http.Client
	req, err := http.NewRequest("GET", "http://foo.com", nil)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := client.Do(req)
	defer resp.Body.Close() // want "using resp before checking for errors"
	if err != nil {
		log.Fatal(err)
	}
}

func goodUnwrapResp() {
	unwrapResp := func(resp *http.Response, err error) *http.Response {
		if err != nil {
			panic(err)
		}
		return resp
	}
	resp := unwrapResp(http.Get("https://golang.org"))
	// It is ok to call defer here immediately as err has
	// been checked in unwrapResp (see #52661).
	defer resp.Body.Close()
}

func badUnwrapResp() {
	unwrapResp := func(resp *http.Response, err error) string {
		if err != nil {
			panic(err)
		}
		return "https://golang.org/" + resp.Status
	}
	resp, err := http.Get(unwrapResp(http.Get("https://golang.org")))
	defer resp.Body.Close() // want "using resp before checking for errors"
	if err != nil {
		log.Fatal(err)
	}
}
