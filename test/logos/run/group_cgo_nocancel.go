//go:build cgo

// run

package main

/*
#include <unistd.h>
static int logosCgoFds[2];
static int logosCgoMakePipe(void) { return pipe(logosCgoFds); }
static long logosCgoBlockRead(void) {
	char b = 0;
	ssize_t n;
	do {
		n = read(logosCgoFds[0], &b, 1);
	} while (n < 0);
	if (n == 1 && b == 'x') {
		return 1;
	}
	return -1;
}
static int logosCgoWriteByte(void) {
	if (write(logosCgoFds[1], "x", 1) == 1) {
		return 0;
	}
	return -1;
}
*/
import "C"

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"runtime"
	"time"

	"logos/rt"
)

var errBoom = errors.New("boom")

var bad bool

func check(tag string, cond bool, got, want string) {
	if !cond {
		fmt.Println("FAIL " + tag + ": got " + got + " want " + want)
		bad = true
		return
	}
	fmt.Println("ok", tag)
}

func main() {
	if C.logosCgoMakePipe() != 0 {
		fmt.Println("FAIL group-cgo-setup: got pipe make failed want pipe")
		os.Exit(1)
	}
	tmp, err := os.CreateTemp("", "groupcgoreg")
	if err != nil {
		fmt.Println("FAIL group-cgo-setup: got", err, "want temp file")
		os.Exit(1)
	}
	if _, err := tmp.WriteString("regular-ok"); err != nil {
		fmt.Println("FAIL group-cgo-setup: got", err, "want write")
		os.Exit(1)
	}
	name := tmp.Name()
	if err := tmp.Close(); err != nil {
		fmt.Println("FAIL group-cgo-setup: got", err, "want close")
		os.Exit(1)
	}
	defer os.Remove(name)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		fmt.Println("FAIL group-cgo-setup: got", err, "want listener")
		os.Exit(1)
	}
	defer ln.Close()
	cli, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		fmt.Println("FAIL group-cgo-setup: got", err, "want dial")
		os.Exit(1)
	}
	defer cli.Close()
	srv, err := ln.Accept()
	if err != nil {
		fmt.Println("FAIL group-cgo-setup: got", err, "want accept")
		os.Exit(1)
	}
	defer srv.Close()
	watch := time.AfterFunc(30*time.Second, func() {
		fmt.Println("FAIL group-cgo-cancel-deadline: group with a blocked read did not finish within 30s")
		os.Exit(1)
	})
	defer watch.Stop()
	failed := make(chan struct{})
	var tcpErr, cgoErr, regErr error
	var cgoRes C.long
	g, _ := rt.NewGroup(context.Background(), 0)
	g.Go(func(_ context.Context) error {
		_, err := cli.Read(make([]byte, 1))
		tcpErr = err
		return nil
	})
	g.Go(func(_ context.Context) error {
		cgoRes = C.logosCgoBlockRead()
		if cgoRes != 1 {
			cgoErr = errors.New("cgo read disturbed")
		}
		return nil
	})
	g.Go(func(_ context.Context) error {
		b, err := os.ReadFile(name)
		if err != nil {
			regErr = err
			return nil
		}
		if string(b) != "regular-ok" {
			regErr = errors.New("regular content changed")
		}
		return nil
	})
	g.Go(func(_ context.Context) error {
		close(failed)
		return errBoom
	})
	g.Go(func(_ context.Context) error {
		for {
			select {
			case <-failed:
				if C.logosCgoWriteByte() != 0 {
					return errors.New("cgo write failed")
				}
				return nil
			default:
				runtime.Gosched()
			}
		}
	})
	werr := g.Wait()
	if regErr != nil {
		check("group-cgo-regular-io-unaffected", false, regErr.Error(), "regular-ok")
	} else {
		check("group-cgo-regular-io-unaffected", true, "regular-ok", "regular-ok")
	}
	check("group-cgo-tcp-wakes-with-timeout", os.IsTimeout(tcpErr), fmt.Sprint(tcpErr), "timeout error")
	check("group-cgo-blocked-read-undisturbed", cgoErr == nil && cgoRes == 1, fmt.Sprint(cgoRes), "1")
	check("group-cgo-first-error-wins", werr == errBoom, fmt.Sprint(werr), errBoom.Error())
	if bad {
		os.Exit(1)
	}
}
