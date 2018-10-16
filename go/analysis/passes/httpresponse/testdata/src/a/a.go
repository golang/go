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
